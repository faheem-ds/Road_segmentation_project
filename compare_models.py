import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

from config import APPLY_POSTPROCESSING, BATCH_SIZE, CHECKPOINT_DIR, DATASET_DIR, NUM_WORKERS, OUTPUT_DIR, TEST_SAMPLES, THRESHOLD
from dataset import RoadSegmentationDataset, print_selected_filenames
from metrics import average_metrics, full_segmentation_metrics
from train_custom import build_custom_model
from train_segformer import build_segformer_model
from train_unet import build_unet_model
from utils import denormalize_image, extract_logits, get_device, load_history_csv, print_metrics, resize_logits_to_mask
from postprocessing import postprocess_binary_masks


MODEL_SPECS = {
    "U-Net": (build_unet_model, CHECKPOINT_DIR / "best_unet.pth"),
    "SegFormer": (build_segformer_model, CHECKPOINT_DIR / "best_segformer.pth"),
    "Custom Hybrid": (build_custom_model, CHECKPOINT_DIR / "custom_hybrid_best.pth"),
}


def load_checkpoint(model, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _prediction_panel(logits: torch.Tensor) -> Image.Image:
    pred_mask = (torch.sigmoid(logits) >= THRESHOLD).float()
    while pred_mask.ndim < 4:
        pred_mask = pred_mask.unsqueeze(0)
    if APPLY_POSTPROCESSING:
        pred_mask = postprocess_binary_masks(pred_mask)
    pred = pred_mask.squeeze().cpu().numpy().astype("uint8") * 255
    return ImageOps.colorize(Image.fromarray(pred).convert("L"), black="black", white="white")


def save_combined_visual(image_tensor, mask_tensor, logits_by_model: dict[str, torch.Tensor], filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem

    image = Image.fromarray(denormalize_image(image_tensor))
    gt = Image.fromarray((mask_tensor.squeeze().cpu().numpy() > 0.5).astype("uint8") * 255)

    panels = [
        ("Image", image),
        ("Ground Truth", ImageOps.colorize(gt.convert("L"), black="black", white="white")),
    ]
    panels.extend((name, _prediction_panel(logits)) for name, logits in logits_by_model.items())

    border = 8
    label_height = 28
    width, height = image.size
    canvas = Image.new("RGB", (width * len(panels) + border * (len(panels) - 1), height + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for index, (label, panel) in enumerate(panels):
        x_offset = index * (width + border)
        draw.text((x_offset + 8, 6), label, fill=(0, 0, 0))
        canvas.paste(panel.convert("RGB"), (x_offset, label_height))

    canvas.save(output_dir / f"{stem}_model_comparison.png")


def plot_history_comparison() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install it with: pip install matplotlib")
        return

    history_paths = {
        "U-Net": OUTPUT_DIR / "unet" / "history.csv",
        "SegFormer": OUTPUT_DIR / "segformer" / "history.csv",
        "Custom Hybrid": OUTPUT_DIR / "custom_hybrid" / "history.csv",
    }
    histories = {name: load_history_csv(path) for name, path in history_paths.items() if path.exists()}
    if len(histories) < 2:
        print("Not enough training histories found for comparison curves.")
        return

    output_dir = OUTPUT_DIR / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        plt.plot(epochs, [row["valid_loss"] for row in history], marker="o", label=f"{name} Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "valid_loss_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        plt.plot(epochs, [row["valid_iou"] for row in history], marker="o", label=f"{name} IoU")
        plt.plot(epochs, [row["valid_dice"] for row in history], marker="s", label=f"{name} Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation IoU and Dice Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "iou_dice_comparison.png", dpi=150)
    plt.close()


def save_results_csv(results: dict[str, dict[str, float]]) -> None:
    path = OUTPUT_DIR / "comparison_results.csv"
    fieldnames = [
        "model",
        "iou",
        "dice",
        "precision",
        "recall",
        "pixel_accuracy",
        "edge_iou",
        "connected_component_difference",
        "largest_component_ratio",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results.items():
            writer.writerow({"model": model_name, **metrics})


def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    header = (
        "Model | IoU | Dice | Precision | Recall | Pixel Accuracy | "
        "Edge IoU | Component Diff | Largest Component Ratio"
    )
    print("\n" + header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name} | "
            f"{metrics['iou']:.4f} | "
            f"{metrics['dice']:.4f} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['pixel_accuracy']:.4f} | "
            f"{metrics['edge_iou']:.4f} | "
            f"{metrics['connected_component_difference']:.4f} | "
            f"{metrics['largest_component_ratio']:.4f}"
        )


def main() -> None:
    device = get_device()
    comparison_dir = OUTPUT_DIR / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    dataset = RoadSegmentationDataset(DATASET_DIR, "test", TEST_SAMPLES)
    print_selected_filenames(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    models = {
        name: load_checkpoint(builder().to(device), checkpoint_path, device)
        for name, (builder, checkpoint_path) in MODEL_SPECS.items()
    }
    metrics_by_model = {name: [] for name in models}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits_by_model = {}

            for model_name, model in models.items():
                logits = resize_logits_to_mask(extract_logits(model(images)), masks)
                metrics_by_model[model_name].append(
                    full_segmentation_metrics(logits, masks, THRESHOLD, apply_postprocessing=APPLY_POSTPROCESSING)
                )
                logits_by_model[model_name] = logits

            for sample_index, filename in enumerate(batch["filename"]):
                save_combined_visual(
                    images[sample_index].cpu(),
                    masks[sample_index].cpu(),
                    {name: logits[sample_index].cpu() for name, logits in logits_by_model.items()},
                    filename,
                    comparison_dir,
                )

    results = {name: average_metrics(items) for name, items in metrics_by_model.items()}
    for model_name, model_metrics in results.items():
        print_metrics(f"{model_name} test metrics", model_metrics)

    print_comparison_table(results)
    save_results_csv(results)
    plot_history_comparison()
    print(f"\nSaved comparison CSV to: {OUTPUT_DIR / 'comparison_results.csv'}")
    print(f"Saved combined visualizations to: {comparison_dir}")


if __name__ == "__main__":
    main()
