"""Evaluate a trained v9 "combined" model against the real 778-image DUO test set."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from imggen import REPO_ROOT

CONFIG = REPO_ROOT / "configs/experiments/v9_combined.json"
REAL_EVAL_DATA = "dataset/real_eval/data.yaml"
REAL_EVAL_IMAGES = "dataset/real_eval/images"
REAL_EVAL_LABELS = Path("dataset/real_eval/labels")
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}
CONF_THRESHOLD = 0.25


def load_variant(name: str) -> dict:
    variants = json.loads(CONFIG.read_text(encoding="utf-8"))["variants"]
    if name not in variants:
        raise SystemExit(f"unknown variant {name!r} - available: {', '.join(sorted(variants))}")
    return variants[name]


def _count(label_dir: Path) -> dict[str, int]:
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="variant key from configs/experiments/v9_combined.json")
    args = ap.parse_args()

    cfg = load_variant(args.variant)
    variant = cfg["run_name"]
    weights = f"runs/train/{variant}/weights/best.pt"

    predict_project = Path("runs/predict_duo").resolve()
    eval_project = Path("runs/eval_duo").resolve()
    gt_counts = _count(REAL_EVAL_LABELS)

    model = YOLO(weights)

    pred_name = f"{variant}_detections"
    model.predict(
        source=REAL_EVAL_IMAGES, save=True, save_txt=True, save_conf=True,
        conf=CONF_THRESHOLD, project=str(predict_project), name=pred_name, exist_ok=True,
    )
    pred_counts = _count(predict_project / pred_name / "labels")

    val_result = model.val(
        data=REAL_EVAL_DATA, device="0", project=str(eval_project), name=variant, exist_ok=True, plots=True,
    )

    per_class = {}
    for idx, cid in enumerate(val_result.box.ap_class_index):
        cname = CLASS_NAMES[int(cid)]
        per_class[cname] = {
            "precision": float(val_result.box.p[idx]), "recall": float(val_result.box.r[idx]),
            "ap50": float(val_result.box.ap50[idx]), "ap50_95": float(val_result.box.ap[idx]),
            "gt_instances": gt_counts.get(cname, 0), "predicted_instances": pred_counts.get(cname, 0),
        }

    entry = {
        "overall": {
            "map50": float(val_result.box.map50), "map50_95": float(val_result.box.map),
            "precision": float(val_result.box.mp), "recall": float(val_result.box.mr),
        },
        "per_class": per_class,
        "detections_dir": str(predict_project / pred_name),
        "eval_dir": str(eval_project / variant),
        "train_images": cfg["train_images"],
    }

    comparison_path = eval_project / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else {
        "ground_truth_instances": gt_counts, "models": {}
    }
    comparison["models"][variant] = entry
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"\n{variant}: mAP50={entry['overall']['map50']:.4f}  mAP50-95={entry['overall']['map50_95']:.4f}  "
          f"P={entry['overall']['precision']:.4f}  R={entry['overall']['recall']:.4f}")
    for cname, m in per_class.items():
        print(f"  {cname:<12} AP50={m['ap50']:.4f}  AP50-95={m['ap50_95']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  gt={m['gt_instances']}  pred={m['predicted_instances']}")
    print(f"\ncomparison.json updated -> {comparison_path}")


if __name__ == "__main__":
    main()
