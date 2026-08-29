"""Train yolo26x.pt on the flux2dev v9 base + placeholder-DR combined dataset
(Dataset B: dataset/v9_flux2dev_placeholder_dr, 1800 train / 200 val) and
score it against the real DUO test set. Same conventions as
imggen/train/v9_duo.py: mosaic/mixup off, fresh pretrained checkpoint, 100
epochs.

    python -m imggen.train.v9_placeholder
"""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = "dataset/v9_flux2dev_placeholder_dr/data.yaml"
REAL_EVAL_DATA = "dataset/real_eval/data.yaml"
EPOCHS = 100
IMGSZ = 640
RUN_NAME = "v9_flux2dev_placeholder_dr"


def main() -> None:
    project = Path("runs/train").resolve()

    model = YOLO("yolo26x.pt")
    model.train(
        data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=-1, device="0",
        project=str(project), name=RUN_NAME, exist_ok=True, plots=True,
        mosaic=0.0, mixup=0.0,
    )

    val_own = model.val(data=DATA_YAML, device="0", project=str(project), name=f"{RUN_NAME}_val_own", exist_ok=True)
    val_duo = model.val(data=REAL_EVAL_DATA, device="0", project=str(project), name=f"{RUN_NAME}_val_duo", exist_ok=True)

    summary = {
        "own_val": {
            "map50": float(val_own.box.map50), "map50_95": float(val_own.box.map),
            "precision": float(val_own.box.mp), "recall": float(val_own.box.mr),
        },
        "duo_test": {
            "map50": float(val_duo.box.map50), "map50_95": float(val_duo.box.map),
            "precision": float(val_duo.box.mp), "recall": float(val_duo.box.mr),
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
