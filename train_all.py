from datetime import datetime

from train_custom import main as train_custom
from train_segformer import main as train_segformer
from train_unet import main as train_unet


def run_stage(name: str, trainer) -> None:
    print("\n" + "=" * 80)
    print(f"Starting {name} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    trainer(run_test=False)
    print("\n" + "=" * 80)
    print(f"Finished {name} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def main() -> None:
    run_stage("U-Net", train_unet)
    run_stage("SegFormer", train_segformer)
    run_stage("Custom Hybrid", train_custom)
    print("\nAll models finished training. Run compare_models.py when you are ready to evaluate a limited test subset.")


if __name__ == "__main__":
    main()
