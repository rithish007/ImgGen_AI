"""Runs YOLO inference with each trained regime's weights over the DUO test images."""

from pathlib import Path
from ultralytics import YOLO

def run_yolo_testing():
    root_dir = Path(r"C:\dev\ImgGen")
    test_images_path = root_dir / "dataset" / "test_duo"
    output_base_dir = root_dir / "outputs"

    regimes = [
        "original_aug",
        "original_dr_aug",
        "original_dr_no_aug",
        "original_no_aug"
    ]

    if not test_images_path.exists():
        raise FileNotFoundError(f"Test image directory does not exist: {test_images_path}")

    print(f"Starting evaluations on images from: {test_images_path}\n")

    for regime in regimes:
        weights_path = root_dir / "runs" / "train" / regime / "weights" / "best.pt"

        if not weights_path.exists():
            print(f"⚠️  [Warning] Skipping '{regime}' — model weights not found at: {weights_path}")
            continue

        print(f"--------------------------------------------------")
        print(f"🔍 Running inference for regime: {regime}")
        print(f"--------------------------------------------------")

        model = YOLO(str(weights_path))

        model.predict(
            source=str(test_images_path),
            save=True,
            project=str(output_base_dir),
            name=regime,
            exist_ok=True,
            conf=0.25
        )

        print(f"✅ Detections successfully saved to: {output_base_dir / regime}\n")

if __name__ == "__main__":
    run_yolo_testing()