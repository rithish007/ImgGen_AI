"""Detect + score the v9_flux2dev_duo_dr_v3 model (duo_calibrated DR, label-fix applied, trained with the heavier MuSGD/mosaic-0.95/mixup-0.35/ copy_paste-0.35/scale-0.9/freeze-10 regime - see imggen/train/v9_duo_v3.py) against the real DUO test set, and merge it into the same runs/eval_duo/comparison.json produced by imggen/eval/v9_on_duo.py and friends."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

VARIANT = "v9_flux2dev_duo_dr_v3"
WEIGHTS = f"runs/train/{VARIANT}/weights/best.pt"
REAL_EVAL_DATA = "dataset/real_eval/data.yaml"
REAL_EVAL_IMAGES = "dataset/real_eval/images"
REAL_EVAL_LABELS = Path("dataset/real_eval/labels")
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}
CONF_THRESHOLD = 0.25
COMPARISON_PATH = Path("runs/eval_duo/comparison.json")


def count_gt_instances() -> dict[str, int]:
    counts = Counter()
    for label_path in REAL_EVAL_LABELS.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return dict(counts)


def count_pred_instances(label_dir: Path) -> dict[str, int]:
    counts = Counter()
    if not label_dir.exists():
        return {}
    for label_path in label_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return dict(counts)


def main() -> None:
    predict_project = Path("runs/predict_duo").resolve()
    eval_project = Path("runs/eval_duo").resolve()
    gt_counts = count_gt_instances()

    model = YOLO(WEIGHTS)

    pred_name = f"{VARIANT}_detections"
    model.predict(
        source=REAL_EVAL_IMAGES,
        save=True,
        save_txt=True,
        save_conf=True,
        conf=CONF_THRESHOLD,
        project=str(predict_project),
        name=pred_name,
        exist_ok=True,
    )
    pred_counts = count_pred_instances(predict_project / pred_name / "labels")

    val_result = model.val(
        data=REAL_EVAL_DATA,
        device="0",
        project=str(eval_project),
        name=VARIANT,
        exist_ok=True,
        plots=True,
    )

    per_class = {}
    for idx, cid in enumerate(val_result.box.ap_class_index):
        cname = CLASS_NAMES[int(cid)]
        per_class[cname] = {
            "precision": float(val_result.box.p[idx]),
            "recall": float(val_result.box.r[idx]),
            "ap50": float(val_result.box.ap50[idx]),
            "ap50_95": float(val_result.box.ap[idx]),
            "gt_instances": gt_counts.get(cname, 0),
            "predicted_instances": pred_counts.get(cname, 0),
        }

    entry = {
        "overall": {
            "map50": float(val_result.box.map50),
            "map50_95": float(val_result.box.map),
            "precision": float(val_result.box.mp),
            "recall": float(val_result.box.mr),
        },
        "per_class": per_class,
        "detections_dir": str(predict_project / pred_name),
        "eval_dir": str(eval_project / VARIANT),
        "train_images": 1800,
    }

    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8")) if COMPARISON_PATH.exists() else {
        "ground_truth_instances": gt_counts, "models": {}
    }
    comparison["models"][VARIANT] = entry
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"\n{VARIANT}: mAP50={entry['overall']['map50']:.4f}  mAP50-95={entry['overall']['map50_95']:.4f}  "
          f"P={entry['overall']['precision']:.4f}  R={entry['overall']['recall']:.4f}")
    for cname, m in per_class.items():
        print(f"  {cname:<12} AP50={m['ap50']:.4f}  AP50-95={m['ap50_95']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  gt={m['gt_instances']}  pred={m['predicted_instances']}")
    print(f"\ncomparison.json updated -> {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
