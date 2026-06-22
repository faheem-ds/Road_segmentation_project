import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BATCH_SIZE, CHECKPOINT_DIR, DATASET_DIR, EARLY_STOPPING_PATIENCE, LR_UNET, NUM_EPOCHS, NUM_WORKERS, OUTPUT_DIR, TEST_SAMPLES, TRAIN_SAMPLES, VALID_SAMPLES
from dataset import RoadSegmentationDataset, print_selected_filenames
from losses import BCEDiceLoss
from utils import ensure_dirs, get_device, plot_training_history, print_metrics, run_evaluation, run_validation, save_history_csv


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        x = self.bottleneck(self.pool(d4))

        x = self.up4(x)
        x = self.conv4(torch.cat([x, d4], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, d1], dim=1))
        return self.final(x)


def build_unet_model() -> nn.Module:
    try:
        import segmentation_models_pytorch as smp

        print("Using segmentation_models_pytorch U-Net with pretrained ResNet34 encoder.")
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    except ImportError:
        print("segmentation_models_pytorch not installed. Falling back to a simple U-Net without pretrained encoder.")
        print("Install optional pretrained U-Net support with: pip install segmentation-models-pytorch")
        return SimpleUNet(in_channels=3, out_channels=1)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"
    printed_shapes = False

    for batch in tqdm(dataloader, desc="Train U-Net", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
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

    model = build_unet_model().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_UNET)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_iou = -1.0
    epochs_without_improvement = 0
    best_path = CHECKPOINT_DIR / "best_unet.pth"
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
            print(f"Saved best U-Net checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"No IoU improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epoch(s).")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs. Best validation IoU: {best_iou:.4f}")
                break

    save_history_csv(history, OUTPUT_DIR / "unet" / "history.csv")
    plot_training_history(history, "UNet", OUTPUT_DIR / "unet")

    if not run_test:
        print("Skipping final U-Net test evaluation and prediction image saving.")
        return

    test_dataset = RoadSegmentationDataset(DATASET_DIR, "test", TEST_SAMPLES)
    print_selected_filenames(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_evaluation(model, test_loader, device, OUTPUT_DIR / "unet")
    print_metrics("Test metrics", test_metrics)


if __name__ == "__main__":
    main()
