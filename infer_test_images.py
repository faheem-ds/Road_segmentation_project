import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from compare_models import load_checkpoint
from config import (
    CHECKPOINT_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OUTPUT_DIR,
    THRESHOLD,
)
from dataset import IMAGE_EXTENSIONS
from postprocessing import postprocess_binary_masks
from train_custom import build_custom_model
from train_segformer import build_segformer_model
from train_unet import build_unet_model
from utils import extract_logits, get_device


MODEL_SPECS = {
    "unet": ("U-Net", build_unet_model, CHECKPOINT_DIR / "best_unet.pth"),
    "segformer": ("SegFormer", build_segformer_model, CHECKPOINT_DIR / "best_segformer.pth"),
    "custom_hybrid": ("Custom Hybrid", build_custom_model, CHECKPOINT_DIR / "custom_hybrid_best.pth"),
}


def load_image_tensor(image_path: Path, image_size: int) -> tuple[torch.Tensor, Image.Image, Image.Image]:
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    image_np = (image_np - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return tensor, original, resized


def predict_mask(model, image_tensor: torch.Tensor, device: torch.device, threshold: float, apply_postprocessing: bool) -> Image.Image:
    model.eval()
    with torch.no_grad():
        logits = extract_logits(model(image_tensor.to(device)))
        if logits.shape[-2:] != image_tensor.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=image_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pred_mask = (torch.sigmoid(logits) >= threshold).float()
        if apply_postprocessing:
            pred_mask = postprocess_binary_masks(pred_mask)

    pred_np = pred_mask.squeeze().detach().cpu().numpy().astype(np.uint8) * 255
    return Image.fromarray(pred_np, mode="L")


def make_overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    image = image.convert("RGBA")
    mask = mask.resize(image.size, Image.NEAREST)
    overlay = Image.new("RGBA", image.size, color + (0,))
    mask_np = np.asarray(mask) > 127
    alpha_np = np.zeros(mask_np.shape, dtype=np.uint8)
    alpha_np[mask_np] = int(255 * alpha)
    overlay.putalpha(Image.fromarray(alpha_np, mode="L"))
    return Image.alpha_composite(image, overlay).convert("RGB")


def labeled_panel(label: str, image: Image.Image, width: int, height: int) -> Image.Image:
    label_height = 30
    panel = Image.new("RGB", (width, height + label_height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), label, fill=(0, 0, 0))
    panel.paste(image.convert("RGB").resize((width, height), Image.NEAREST), (0, label_height))
    return panel


def save_comparison(
    image_path: Path,
    original_resized: Image.Image,
    masks_by_model: dict[str, Image.Image],
    output_dir: Path,
    panel_size: int,
) -> None:
    stem = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "U-Net": (255, 60, 60),
        "SegFormer": (60, 160, 255),
        "Custom Hybrid": (60, 210, 120),
    }

    panels = [labeled_panel("Image", original_resized, panel_size, panel_size)]
    for model_name, mask in masks_by_model.items():
        mask.save(output_dir / f"{stem}_{model_name.lower().replace(' ', '_').replace('-', '')}_mask.png")
        overlay = make_overlay(original_resized, mask, colors[model_name])
        overlay.save(output_dir / f"{stem}_{model_name.lower().replace(' ', '_').replace('-', '')}_overlay.png")

        mask_panel = ImageOps.colorize(mask.convert("L"), black="black", white="white")
        panels.append(labeled_panel(f"{model_name} Mask", mask_panel, panel_size, panel_size))
        panels.append(labeled_panel(f"{model_name} Overlay", overlay, panel_size, panel_size))

    border = 8
    canvas_width = panel_size * len(panels) + border * (len(panels) - 1)
    canvas_height = panel_size + 30
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * (panel_size + border), 0))

    canvas.save(output_dir / f"{stem}_all_models_comparison.png")


def load_models(args, device: torch.device):
    checkpoint_overrides = {
        "unet": args.unet_checkpoint,
        "segformer": args.segformer_checkpoint,
        "custom_hybrid": args.custom_checkpoint,
    }

    models = {}
    for key in args.models:
        display_name, builder, default_checkpoint = MODEL_SPECS[key]
        checkpoint_path = Path(checkpoint_overrides[key]) if checkpoint_overrides[key] else default_checkpoint
        print(f"Loading {display_name} from: {checkpoint_path}")
        models[display_name] = load_checkpoint(builder().to(device), checkpoint_path, device)
    return models


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created input folder: {input_dir}")
        print("Put satellite images in this folder and run the command again.")
        return []

    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run U-Net, SegFormer, and Custom Hybrid on images in a folder.")
    parser.add_argument("--input-dir", type=Path, default=Path("test_image"), help="Folder containing images to test.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "test_image_predictions")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--no-postprocess", action="store_true", help="Disable connected-component postprocessing.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="Models to run.",
    )
    parser.add_argument("--unet-checkpoint", type=str, default=None)
    parser.add_argument("--segformer-checkpoint", type=str, default=None)
    parser.add_argument("--custom-checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    image_paths = collect_images(args.input_dir)
    if not image_paths:
        return

    models = load_models(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        print(f"Predicting: {image_path.name}")
        image_tensor, _, resized = load_image_tensor(image_path, args.image_size)
        masks_by_model = {
            model_name: predict_mask(
                model,
                image_tensor,
                device,
                threshold=args.threshold,
                apply_postprocessing=not args.no_postprocess,
            )
            for model_name, model in models.items()
        }
        save_comparison(image_path, resized, masks_by_model, args.output_dir, args.image_size)

    print(f"Saved masks, overlays, and combined panels to: {args.output_dir}")


if __name__ == "__main__":
    main()
