from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import AUGMENT_TRAIN, BATCH_SIZE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_train_augmentations(image_size: int):
    try:
        import albumentations as A
    except ImportError:
        print("albumentations is not installed. Training will continue without augmentations.")
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.15,
            rotate_limit=25,
            border_mode=0,
            value=0,
            mask_value=0,
            p=0.6,
        ),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
    ])


def _sample_key(path: Path) -> str:
    """Return a stable key so 100034_sat.jpg matches 100034_mask.png."""
    stem = path.stem
    for suffix in ("_sat", "_image", "_img", "_mask", "_label", "_gt"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _find_split_dirs(split_dir: Path) -> tuple[Path, Path]:
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    if image_dir.exists() and mask_dir.exists():
        return image_dir, mask_dir
    return split_dir, split_dir


def collect_pairs(dataset_dir: str | Path, split: str, limit: int | None = None) -> list[tuple[Path, Path]]:
    split_dir = Path(dataset_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    image_dir, mask_dir = _find_split_dirs(split_dir)
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and "mask" not in p.stem.lower()
    )
    mask_files = sorted(
        p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS and any(x in p.stem.lower() for x in ("mask", "label", "gt"))
    )

    masks_by_key = {_sample_key(path): path for path in mask_files}
    pairs: list[tuple[Path, Path]] = []
    missing_masks: list[str] = []

    for image_path in image_files:
        key = _sample_key(image_path)
        mask_path = masks_by_key.get(key)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue
        pairs.append((image_path, mask_path))

    if not pairs and image_files and not mask_files:
        raise RuntimeError(
            f"No mask files found in {split_dir}. This pipeline needs masks for {split} to compute loss/metrics. "
            "Expected files such as 100034_mask.png, or a masks/ subfolder."
        )

    if not pairs:
        raise RuntimeError(
            f"No image-mask pairs found in {split_dir}. Expected matching names such as "
            "100034_sat.jpg and 100034_mask.png."
        )

    if missing_masks:
        preview = ", ".join(missing_masks[:5])
        print(f"[Warning] {split}: skipped {len(missing_masks)} image(s) without masks. Examples: {preview}")

    return pairs[:limit] if limit is not None else pairs


class RoadSegmentationDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str, limit: int | None = None, image_size: int = IMAGE_SIZE):
        self.split = split
        self.image_size = image_size
        self.pairs = collect_pairs(dataset_dir, split, limit)
        self.augmentations = build_train_augmentations(image_size) if split == "train" and AUGMENT_TRAIN else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[index]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.float32)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image_np, mask=mask_np)
            image_np = augmented["image"]
            mask_np = augmented["mask"]

        image_np = (image_np - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        mask_np = (mask_np > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "filename": image_path.name,
        }


def make_dataloader(dataset_dir: str | Path, split: str, limit: int, shuffle: bool) -> DataLoader:
    dataset = RoadSegmentationDataset(dataset_dir=dataset_dir, split=split, limit=limit)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def print_selected_filenames(dataset: RoadSegmentationDataset) -> None:
    print(f"\nSelected {dataset.split} files ({len(dataset)} samples):")
    for image_path, mask_path in dataset.pairs:
        print(f"  image={image_path.name} | mask={mask_path.name}")
