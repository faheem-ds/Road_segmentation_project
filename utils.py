from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from config import APPLY_POSTPROCESSING, CHECKPOINT_DIR, COMPUTE_CONNECTIVITY_DURING_VALIDATION, IMAGENET_MEAN, IMAGENET_STD, OUTPUT_DIR
from metrics import average_metrics, full_segmentation_metrics
from postprocessing import postprocess_binary_masks


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def save_prediction_visuals(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    logits: torch.Tensor,
    filename: str,
    output_dir: Path,
    threshold: float = 0.5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem

    image_np = denormalize_image(image_tensor)
    gt_np = (mask_tensor.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    pred_mask = (torch.sigmoid(logits) >= threshold).float()
    while pred_mask.ndim < 4:
        pred_mask = pred_mask.unsqueeze(0)
    if APPLY_POSTPROCESSING:
        pred_mask = postprocess_binary_masks(pred_mask)
    pred_np = pred_mask.squeeze().detach().cpu().numpy().astype(np.uint8) * 255

    Image.fromarray(pred_np).save(output_dir / f"{stem}_pred.png")

    image_pil = Image.fromarray(image_np)
    gt_pil = Image.fromarray(gt_np).convert("RGB")
    pred_pil = Image.fromarray(pred_np).convert("RGB")

    border = 8
    label_height = 28
    width, height = image_pil.size
    canvas = Image.new("RGB", (width * 3 + border * 2, height + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    panels = [
        ("Image", image_pil, 0),
        ("Ground Truth", ImageOps.colorize(gt_pil.convert("L"), black="black", white="white"), width + border),
        ("Prediction", ImageOps.colorize(pred_pil.convert("L"), black="black", white="white"), (width + border) * 2),
    ]
    for label, panel, x_offset in panels:
        draw.text((x_offset + 8, 6), label, fill=(0, 0, 0))
        canvas.paste(panel, (x_offset, label_height))

    canvas.save(output_dir / f"{stem}_comparison.png")


def extract_logits(model_output):
    if isinstance(model_output, dict) and "seg_logits" in model_output:
        return model_output["seg_logits"]
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]
    return model_output


def resize_logits_to_mask(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def run_evaluation(model, dataloader, device: torch.device, output_dir: Path | None = None) -> dict[str, float]:
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            model_output = model(images)
            logits = resize_logits_to_mask(extract_logits(model_output), masks)

            all_metrics.append(full_segmentation_metrics(logits, masks, apply_postprocessing=APPLY_POSTPROCESSING))

            if output_dir is not None:
                for i, filename in enumerate(batch["filename"]):
                    save_prediction_visuals(
                        images[i],
                        masks[i],
                        logits[i],
                        filename,
                        output_dir,
                    )

    return average_metrics(all_metrics)


def run_validation(model, dataloader, criterion, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            model_output = model(images)
            logits = resize_logits_to_mask(extract_logits(model_output), masks)
            if isinstance(model_output, dict) and "seg_logits" in model_output:
                model_output = {
                    key: resize_logits_to_mask(value, masks)
                    for key, value in model_output.items()
                }
                loss = criterion(model_output, masks)
            else:
                loss = criterion(logits, masks)

            total_loss += loss.item()
            all_metrics.append(
                full_segmentation_metrics(
                    logits,
                    masks,
                    include_connectivity=COMPUTE_CONNECTIVITY_DURING_VALIDATION,
                )
            )

    return total_loss / max(len(dataloader), 1), average_metrics(all_metrics)


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  IoU:            {metrics['iou']:.4f}")
    print(f"  Dice Score:     {metrics['dice']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    if "edge_iou" in metrics:
        print(f"  Edge IoU:       {metrics['edge_iou']:.4f}")
    if "connected_component_difference" in metrics:
        print(f"  Component Diff: {metrics['connected_component_difference']:.4f}")
    if "largest_component_ratio" in metrics:
        print(f"  Largest Ratio:  {metrics['largest_component_ratio']:.4f}")


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_history(history: list[dict[str, float]], model_name: str, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install it with: pip install matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train Loss")
    plt.plot(epochs, [row["valid_loss"] for row in history], marker="o", label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_loss_curve.png", dpi=150)
    plt.close()

    metric_names = ["valid_iou", "valid_dice", "valid_precision", "valid_recall", "valid_pixel_accuracy"]
    plt.figure(figsize=(9, 5))
    for metric in metric_names:
        plt.plot(epochs, [row[metric] for row in history], marker="o", label=metric.replace("valid_", "").replace("_", " ").title())
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{model_name} Validation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_metrics_curve.png", dpi=150)
    plt.close()


def load_history_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        return [{key: float(value) for key, value in row.items()} for row in reader]
