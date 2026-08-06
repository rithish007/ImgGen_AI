"""Runs the 4 trained Stage 5 regimes against the real-image eval set built
by prepare_real_eval.py (dataset/real_eval/, from DUO_Dataset's actual test
split with real ground truth) - the sim-to-real number missing until now.

Stage 5's own runs/train/summary.json is train-set-only (no held-out val
split at pilot scale, see the plan doc's Stage 5 section) - optimistic by
construction. This is the first quantitative measurement of whether any of
these regimes actually generalizes to real underwater photos, using
Ultralytics' own mAP/precision/recall (same metric family as summary.json,
directly comparable).

    python src/eval_real.py
    python src/eval_real.py --regimes original_aug original_dr_aug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REGIMES = ["original_no_aug", "original_aug", "original_dr_no_aug", "original_dr_aug"]
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("dataset/real_eval/data.yaml"))
    ap.add_argument("--weights-root", type=Path, default=Path("runs/train"))
    ap.add_argument("--project", type=Path, default=Path("runs/real_eval"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    ap.add_argument("--out", type=Path, default=Path("reports/real_eval_summary.json"))
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found - run src/prepare_real_eval.py first")

    args.project = args.project.resolve()

    from ultralytics import YOLO

    summary = {}
    for name in args.regimes:
        weights = args.weights_root / name / "weights" / "best.pt"
        if not weights.exists():
            print(f"skipping {name}: {weights} not found")
            continue

        print(f"\n{'=' * 70}\n{name}  (weights={weights})\n{'=' * 70}")
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(args.data),
            device=args.device,
            project=str(args.project),
            name=name,
            exist_ok=True,
            plots=True,
        )

        per_class = {}
        for idx, cls_id in enumerate(metrics.box.ap_class_index):
            cls_id = int(cls_id)
            per_class[CLASS_NAMES.get(cls_id, str(cls_id))] = {
                "ap50": float(metrics.box.ap50[idx]),
                "ap50_95": float(metrics.box.ap[idx]),
                "precision": float(metrics.box.p[idx]),
                "recall": float(metrics.box.r[idx]),
            }

        summary[name] = {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class": per_class,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\nreal-image eval summary ({len(summary)} regime(s))\n{'=' * 70}")
    print(f"{'regime':<22}{'mAP50':>8}{'mAP50-95':>10}{'precision':>11}{'recall':>9}")
    for name, m in summary.items():
        print(f"{name:<22}{m['map50']:>8.3f}{m['map50_95']:>10.3f}{m['precision']:>11.3f}{m['recall']:>9.3f}")
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
