"""Pilot-scale YOLO26x ablation driver - RunPod pipeline check, not the production result."""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

VARIANTS = ["pilot_base", "pilot_dr", "pilot_dr_anchored"]
EPOCHS = 100
IMGSZ = 640
REAL_EVAL_DATA = "dataset/real_eval/data.yaml"


def main() -> None:
    project = Path("runs/train").resolve()
    summary = {}

    for variant in VARIANTS:
        data_yaml = f"dataset/{variant}/data.yaml"
        print(f"\n{'=' * 70}\n{variant}  (data={data_yaml})\n{'=' * 70}")

        model = YOLO("yolo26x.pt")
        model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=-1,
            device="0",
            project=str(project),
            name=variant,
            exist_ok=True,
            plots=True,
            mosaic=0.0,
            mixup=0.0,
        )

        val_pilot = model.val(data=data_yaml, device="0", project=str(project), name=f"{variant}_val_pilot", exist_ok=True)
        val_duo = model.val(data=REAL_EVAL_DATA, device="0", project=str(project), name=f"{variant}_val_duo", exist_ok=True)

        summary[variant] = {
            "pilot_val": {
                "map50": float(val_pilot.box.map50),
                "map50_95": float(val_pilot.box.map),
                "precision": float(val_pilot.box.mp),
                "recall": float(val_pilot.box.mr),
            },
            "duo_test": {
                "map50": float(val_duo.box.map50),
                "map50_95": float(val_duo.box.map),
                "precision": float(val_duo.box.mp),
                "recall": float(val_duo.box.mr),
            },
        }

    summary_path = project / "pilot_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\nsummary -> {summary_path}\n{'=' * 70}")
    print(f"{'variant':<20}{'pilot mAP50':>13}{'DUO mAP50':>12}{'DUO mAP50-95':>14}{'DUO P':>8}{'DUO R':>8}")
    for name, m in summary.items():
        d = m["duo_test"]
        print(f"{name:<20}{m['pilot_val']['map50']:>13.3f}{d['map50']:>12.3f}{d['map50_95']:>14.3f}{d['precision']:>8.3f}{d['recall']:>8.3f}")


if __name__ == "__main__":
    main()
