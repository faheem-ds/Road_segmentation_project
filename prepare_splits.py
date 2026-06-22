from pathlib import Path
import csv
import os
import shutil

from dataset import collect_pairs


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DATASET_DIR = PROJECT_DIR / "dataset"
PREPARED_DATASET_DIR = PROJECT_DIR / "dataset_prepared"

TRAIN_RATIO = 0.75
VALID_RATIO = 0.15
TEST_RATIO = 0.10


def link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_split(split_name: str, pairs: list[tuple[Path, Path]]) -> None:
    image_dir = PREPARED_DATASET_DIR / split_name / "images"
    mask_dir = PREPARED_DATASET_DIR / split_name / "masks"

    for image_path, mask_path in pairs:
        link_or_copy_file(image_path, image_dir / image_path.name)
        link_or_copy_file(mask_path, mask_dir / mask_path.name)


def write_metadata(splits: dict[str, list[tuple[Path, Path]]]) -> None:
    metadata_path = PREPARED_DATASET_DIR / "metadata.csv"
    with metadata_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "split", "sat_image_path", "mask_path"])

        for split_name, pairs in splits.items():
            for image_path, mask_path in pairs:
                image_id = image_path.stem.replace("_sat", "")
                writer.writerow([
                    image_id,
                    split_name,
                    f"{split_name}/images/{image_path.name}",
                    f"{split_name}/masks/{mask_path.name}",
                ])


def main() -> None:
    labeled_pairs = collect_pairs(SOURCE_DATASET_DIR, "train", limit=None)
    total = len(labeled_pairs)

    train_count = int(total * TRAIN_RATIO)
    valid_count = int(total * VALID_RATIO)
    test_count = total - train_count - valid_count

    splits = {
        "train": labeled_pairs[:train_count],
        "valid": labeled_pairs[train_count: train_count + valid_count],
        "test": labeled_pairs[train_count + valid_count:],
    }

    print(f"Found {total} labeled image-mask pairs in: {SOURCE_DATASET_DIR / 'train'}")
    print(f"Creating prepared dataset at: {PREPARED_DATASET_DIR}")
    print(f"Train: {len(splits['train'])} ({TRAIN_RATIO:.0%})")
    print(f"Valid: {len(splits['valid'])} ({VALID_RATIO:.0%})")
    print(f"Test:  {len(splits['test'])} ({TEST_RATIO:.0%})")

    for split_name, pairs in splits.items():
        write_split(split_name, pairs)

    write_metadata(splits)
    print("Done. Update config.py to use DATASET_DIR = PROJECT_DIR / 'dataset_prepared'.")


if __name__ == "__main__":
    main()
