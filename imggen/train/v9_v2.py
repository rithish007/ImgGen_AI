"""Train yolo26x.pt on dataset/v9_flux2dev (base, no DR) with the same
heavy-augmentation/optimizer regime as imggen/train/v9_duo_v3.py - see that
module's docstring for full rationale. This is the base-only control run:
deliberately identical hyperparameters across all four v2/v3 runs so the
DR profile (or its absence) is the only varying factor between them.

    python -m imggen.train.v9_v2
"""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = "dataset/v9_flux2dev/data.yaml"
REAL_EVAL_DATA = "dataset/real_eval/data.yaml"
EPOCHS = 150
PATIENCE = 30
IMGSZ = 896
RUN_NAME = "v9_flux2dev_v2"


def main() -> None:
    project = Path("runs/train").resolve()

    model = YOLO("yolo26x.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=-1,
        device="0",
        project=str(project),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        optimizer="MuSGD",
        lr0=0.0004,
        lrf=0.5,
        cos_lr=True,
        momentum=0.948,
        weight_decay=0.00027,
        warmup_epochs=3.0,
        amp=True,
        mosaic=0.95,
        close_mosaic=10,
        mixup=0.35,
        copy_paste=0.35,
        scale=0.9,
        translate=0.1,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.4,
        hsv_h=0.03,
        hsv_s=0.9,
        hsv_v=0.6,
        erasing=0.4,
        freeze=10,
    )

    val_own = model.val(data=DATA_YAML, device="0", project=str(project), name=f"{RUN_NAME}_val_own", exist_ok=True)
    val_duo = model.val(data=REAL_EVAL_DATA, device="0", project=str(project), name=f"{RUN_NAME}_val_duo", exist_ok=True)

    summary = {
        "own_val": {
            "map50": float(val_own.box.map50),
            "map50_95": float(val_own.box.map),
            "precision": float(val_own.box.mp),
            "recall": float(val_own.box.mr),
        },
        "duo_test": {
            "map50": float(val_duo.box.map50),
            "map50_95": float(val_duo.box.map),
            "precision": float(val_duo.box.mp),
            "recall": float(val_duo.box.mr),
        },
    }
    summary_path = project / f"{RUN_NAME}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nsummary -> {summary_path}")
    print(f"own val:  mAP50={summary['own_val']['map50']:.4f}  mAP50-95={summary['own_val']['map50_95']:.4f}")
    print(f"DUO test: mAP50={summary['duo_test']['map50']:.4f}  mAP50-95={summary['duo_test']['map50_95']:.4f}  "
          f"P={summary['duo_test']['precision']:.4f}  R={summary['duo_test']['recall']:.4f}")


if __name__ == "__main__":
    main()
