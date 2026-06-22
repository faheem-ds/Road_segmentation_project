import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BATCH_SIZE, CHECKPOINT_DIR, DATASET_DIR, EARLY_STOPPING_PATIENCE, LR_SEGFORMER, NUM_EPOCHS, NUM_WORKERS, OUTPUT_DIR, SEGFORMER_MODEL_NAME, TEST_SAMPLES, TRAIN_SAMPLES, VALID_SAMPLES
from dataset import RoadSegmentationDataset, print_selected_filenames
from losses import BCEDiceLoss
from utils import ensure_dirs, extract_logits, get_device, plot_training_history, print_metrics, resize_logits_to_mask, run_evaluation, run_validation, save_history_csv


def build_segformer_model():
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ImportError(
            "Hugging Face Transformers is not installed. Install it with: pip install transformers"
        ) from exc

    print(f"Loading SegFormer: {SEGFORMER_MODEL_NAME}")
    return SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_MODEL_NAME,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"
    printed_shapes = False

    for batch in tqdm(dataloader, desc="Train SegFormer", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = extract_logits(model(images))
            logits = resize_logits_to_mask(logits, masks)
            if not printed_shapes:
                print(f"First batch image shape: {tuple(images.shape)}")
                print(f"First batch mask shape: {tuple(masks.shape)}")
                print(f"First batch logits shape: {tuple(logits.shape)}")
                printed_shapes = True
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def main(run_test: bool = True) -> None:
    ensure_dirs()
    device = get_device()
    print(f"Device: {device}")

    train_dataset = RoadSegmentationDataset(DATASET_DIR, "train", TRAIN_SAMPLES)
    valid_dataset = RoadSegmentationDataset(DATASET_DIR, "valid", VALID_SAMPLES)
    print_selected_filenames(train_dataset)
    print_selected_filenames(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    model = build_segformer_model().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_SEGFORMER)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_iou = -1.0
    epochs_without_improvement = 0
    best_path = CHECKPOINT_DIR / "best_segformer.pth"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        valid_loss, valid_metrics = run_validation(model, valid_loader, criterion, device)
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print_metrics("Validation metrics", valid_metrics)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_iou": valid_metrics["iou"],
            "valid_dice": valid_metrics["dice"],
            "valid_precision": valid_metrics["precision"],
            "valid_recall": valid_metrics["recall"],
            "valid_pixel_accuracy": valid_metrics["pixel_accuracy"],
            "valid_edge_iou": valid_metrics["edge_iou"],
            "valid_connected_component_difference": valid_metrics.get("connected_component_difference", 0.0),
            "valid_largest_component_ratio": valid_metrics.get("largest_component_ratio", 0.0),
        })

        if valid_metrics["iou"] > best_iou:
            best_iou = valid_metrics["iou"]
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "best_iou": best_iou}, best_path)
            print(f"Saved best SegFormer checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"No IoU improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epoch(s).")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs. Best validation IoU: {best_iou:.4f}")
                break

    save_history_csv(history, OUTPUT_DIR / "segformer" / "history.csv")
    plot_training_history(history, "SegFormer", OUTPUT_DIR / "segformer")

    if not run_test:
        print("Skipping final SegFormer test evaluation and prediction image saving.")
        return

    test_dataset = RoadSegmentationDataset(DATASET_DIR, "test", TEST_SAMPLES)
    print_selected_filenames(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_evaluation(model, test_loader, device, OUTPUT_DIR / "segformer")
    print_metrics("Test metrics", test_metrics)


if __name__ == "__main__":
    main()
