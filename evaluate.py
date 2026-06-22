import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECKPOINT_DIR, DATASET_DIR, NUM_WORKERS, OUTPUT_DIR, TEST_SAMPLES, VALID_SAMPLES
from dataset import RoadSegmentationDataset, print_selected_filenames
from train_custom import build_custom_model
from train_segformer import build_segformer_model
from train_unet import build_unet_model
from utils import ensure_dirs, get_device, print_metrics, run_evaluation


def build_model(model_name: str):
    if model_name == "unet":
        return build_unet_model()
    if model_name == "segformer":
        return build_segformer_model()
    if model_name == "custom_hybrid":
        return build_custom_model()
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained road segmentation model.")
    parser.add_argument("--model", choices=["unet", "segformer", "custom_hybrid"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device()
    default_checkpoint = CHECKPOINT_DIR / ("custom_hybrid_best.pth" if args.model == "custom_hybrid" else f"best_{args.model}.pth")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    limit = VALID_SAMPLES if args.split == "valid" else TEST_SAMPLES
    dataset = RoadSegmentationDataset(DATASET_DIR, args.split, limit)
    print_selected_filenames(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    model = build_model(args.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_dir = OUTPUT_DIR / args.model / args.split
    metrics = run_evaluation(model, dataloader, device, output_dir)
    print_metrics(f"{args.split.capitalize()} metrics", metrics)
    print(f"Saved predictions and visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
