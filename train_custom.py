import argparse

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CUSTOM_BACKBONE,
    DATASET_DIR,
    D_MODEL,
    EARLY_STOPPING_PATIENCE,
    FREEZE_BACKBONE_EPOCHS,
    LAMBDA_EDGE,
    LR_BACKBONE,
    LR_CUSTOM,
    NHEAD,
    NUM_DECODER_LAYERS,
    NUM_EPOCHS,
    NUM_QUERIES,
    NUM_WORKERS,
    OUTPUT_DIR,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
    VALID_SAMPLES,
)
from dataset import RoadSegmentationDataset, print_selected_filenames
from losses import EdgeAwareLoss
from models.custom_hybrid import EdgeAwareHybridRoadFormer
from utils import ensure_dirs, get_device, plot_training_history, print_metrics, run_evaluation, run_validation, save_history_csv


def build_custom_model() -> EdgeAwareHybridRoadFormer:
    return EdgeAwareHybridRoadFormer(
        backbone_name=CUSTOM_BACKBONE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_queries=NUM_QUERIES,
    )


def build_optimizer(model: EdgeAwareHybridRoadFormer, phase: str) -> torch.optim.Optimizer:
    if phase == "head_only":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=LR_CUSTOM, weight_decay=0.01)

    if phase == "full":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        return torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.neck.parameters(), "lr": LR_CUSTOM},
            {"params": model.decoder.parameters(), "lr": LR_CUSTOM},
            {"params": model.seg_head.parameters(), "lr": LR_CUSTOM},
            {"params": model.edge_head.parameters(), "lr": LR_CUSTOM},
            {"params": model.refine_input.parameters(), "lr": LR_CUSTOM},
            {"params": model.continuity_refine.parameters(), "lr": LR_CUSTOM},
            {"params": model.seg_refine.parameters(), "lr": LR_CUSTOM},
            {"params": model.edge_refine.parameters(), "lr": LR_CUSTOM},
            {"params": model.centerline_refine.parameters(), "lr": LR_CUSTOM},
            {"params": model.connectivity_smooth.parameters(), "lr": LR_CUSTOM},
            {"params": [model.road_queries], "lr": LR_CUSTOM},
        ], weight_decay=0.01)

    raise ValueError(f"Unknown optimizer phase: {phase}")


def build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(NUM_EPOCHS - FREEZE_BACKBONE_EPOCHS, 1),
        eta_min=1e-6,
    )


def save_resume_checkpoint(
    path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    best_iou: float,
    epochs_without_improvement: int,
    history: list[dict[str, float]],
    optimizer_phase: str,
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
        "best_iou": best_iou,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "optimizer_phase": optimizer_phase,
    }, path)


def load_resume_checkpoint(path, model, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _resize_outputs_to_mask(outputs: dict[str, torch.Tensor], masks: torch.Tensor) -> dict[str, torch.Tensor]:
    resized = {}
    for key, logits in outputs.items():
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        resized[key] = logits
    return resized


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"
    printed_shapes = False

    for batch in tqdm(dataloader, desc="Train Custom", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = _resize_outputs_to_mask(model(images), masks)
            if not printed_shapes:
                print(f"First batch image shape: {tuple(images.shape)}")
                print(f"First batch mask shape: {tuple(masks.shape)}")
                print(f"First batch seg logits shape: {tuple(outputs['seg_logits'].shape)}")
                print(f"First batch edge logits shape: {tuple(outputs['edge_logits'].shape)}")
                if "centerline_logits" in outputs:
                    print(f"First batch centerline logits shape: {tuple(outputs['centerline_logits'].shape)}")
                printed_shapes = True
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def main(run_test: bool = True, resume: bool = False) -> None:
    ensure_dirs()
    device = get_device()
    print(f"Device: {device}")

    train_dataset = RoadSegmentationDataset(DATASET_DIR, "train", TRAIN_SAMPLES)
    valid_dataset = RoadSegmentationDataset(DATASET_DIR, "valid", VALID_SAMPLES)
    print_selected_filenames(train_dataset)
    print_selected_filenames(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    model = build_custom_model().to(device)
    criterion = EdgeAwareLoss(lambda_edge=LAMBDA_EDGE)
    optimizer = build_optimizer(model, "head_only")
    scheduler = None
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_iou = -1.0
    epochs_without_improvement = 0
    best_path = CHECKPOINT_DIR / "custom_hybrid_best.pth"
    last_path = CHECKPOINT_DIR / "custom_hybrid_last.pth"
    start_epoch = 1
    optimizer_phase = "head_only"
    history = []

    if resume:
        checkpoint = load_resume_checkpoint(last_path, model, device)
        start_epoch = checkpoint["epoch"] + 1
        optimizer_phase = checkpoint.get("optimizer_phase", "full" if start_epoch > FREEZE_BACKBONE_EPOCHS else "head_only")
        optimizer = build_optimizer(model, optimizer_phase)
        scheduler = build_scheduler(optimizer) if optimizer_phase == "full" else None
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_iou = checkpoint.get("best_iou", -1.0)
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        history = checkpoint.get("history", [])
        print(f"Resumed custom training from epoch {checkpoint['epoch']}. Next epoch: {start_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            print("Unfreezing custom backbone with differential learning rates.")
            optimizer_phase = "full"
            optimizer = build_optimizer(model, "full")
            scheduler = build_scheduler(optimizer)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        valid_loss, valid_metrics = run_validation(model, valid_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
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
            print(f"Saved best custom hybrid checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"No IoU improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epoch(s).")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs. Best validation IoU: {best_iou:.4f}")
                break

        save_resume_checkpoint(
            last_path,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_iou,
            epochs_without_improvement,
            history,
            optimizer_phase,
        )
        print(f"Saved resume checkpoint: {last_path}")

    save_history_csv(history, OUTPUT_DIR / "custom_hybrid" / "history.csv")
    plot_training_history(history, "CustomHybrid", OUTPUT_DIR / "custom_hybrid")

    if not run_test:
        print("Skipping final custom hybrid test evaluation and prediction image saving.")
        return

    test_dataset = RoadSegmentationDataset(DATASET_DIR, "test", TEST_SAMPLES)
    print_selected_filenames(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_evaluation(model, test_loader, device, OUTPUT_DIR / "custom_hybrid")
    print_metrics("Test metrics", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom hybrid road segmentation model.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/custom_hybrid_last.pth")
    parser.add_argument("--no-test", action="store_true", help="Skip final test evaluation and prediction image saving")
    args = parser.parse_args()
    main(run_test=not args.no_test, resume=args.resume)
