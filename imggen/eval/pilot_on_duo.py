"""Detect + score the 3 pilot-ablation models (pilot_base, pilot_dr,
pilot_dr_anchored) against the real DUO test set (dataset/real_eval, 778
images, class-remapped to starfish/sea_urchin/scallop).

For each model:
    1. model.predict() over dataset/real_eval/images, saving annotated
       detection images + YOLO-format .txt predictions with confidences to
       a clearly named folder under runs/predict_duo/<variant>_detections/.
    2. model.val() with plots=True against dataset/real_eval/data.yaml to
       produce a confusion matrix, PR/F1 curves, and per-class P/R/AP50/
       AP50-95 under runs/eval_duo/<variant>/.

Writes runs/eval_duo/comparison.json - per-class + overall metrics for all
3 models side by side, plus per-class GT vs predicted instance counts (at
the predict() confidence threshold) to show over/under-detection bias.

    python -m imggen.eval.pilot_on_duo
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

VARIANTS = ["pilot_base", "pilot_dr", "pilot_dr_anchored"]
REAL_EVAL_DATA = "dataset/real_eval/data.yaml"
REAL_EVAL_IMAGES = "dataset/real_eval/images"
REAL_EVAL_LABELS = Path("dataset/real_eval/labels")
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}
CONF_THRESHOLD = 0.25


def count_gt_instances() -> dict[str, int]:
    counts = Counter()
    for label_path in REAL_EVAL_LABELS.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cid = int(line.split()[0])
                counts[CLASS_NAMES[cid]] += 1
    return dict(counts)


def count_pred_instances(label_dir: Path) -> dict[str, int]:
    counts = Counter()
    if not label_dir.exists():
        return {}
    for label_path in label_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cid = int(line.split()[0])
                counts[CLASS_NAMES[cid]] += 1
    return dict(counts)


def main() -> None:
    predict_project = Path("runs/predict_duo").resolve()
    eval_project = Path("runs/eval_duo").resolve()

    gt_counts = count_gt_instances()
    comparison = {"ground_truth_instances": gt_counts, "models": {}}

    for variant in VARIANTS:
        weights = f"runs/train/{variant}/weights/best.pt"
        print(f"\n{'=' * 70}\n{variant}  (weights={weights})\n{'=' * 70}")
        model = YOLO(weights)

        pred_name = f"{variant}_detections"
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
        pred_label_dir = predict_project / pred_name / "labels"
        pred_counts = count_pred_instances(pred_label_dir)

        val_result = model.val(
            data=REAL_EVAL_DATA,
            device="0",
            project=str(eval_project),
            name=variant,
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

        comparison["models"][variant] = {
            "overall": {
                "map50": float(val_result.box.map50),
                "map50_95": float(val_result.box.map),
                "precision": float(val_result.box.mp),
                "recall": float(val_result.box.mr),
            },
            "per_class": per_class,
            "detections_dir": str(predict_project / pred_name),
            "eval_dir": str(eval_project / variant),
        }

    comparison_path = eval_project / "comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\ncomparison -> {comparison_path}\n{'=' * 70}")
    print(f"ground truth instances: {gt_counts}")
    for variant, data in comparison["models"].items():
        print(f"\n{variant}:")
        print(f"  overall: mAP50={data['overall']['map50']:.4f}  mAP50-95={data['overall']['map50_95']:.4f}  "
              f"P={data['overall']['precision']:.4f}  R={data['overall']['recall']:.4f}")
        for cname, m in data["per_class"].items():
            print(f"  {cname:<12} AP50={m['ap50']:.4f}  AP50-95={m['ap50_95']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  gt={m['gt_instances']}  pred={m['predicted_instances']}")


if __name__ == "__main__":
    main()
