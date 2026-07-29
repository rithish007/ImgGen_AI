"""Stage 5 - YOLO26s training, the planned 2x2 ablation:
    {original, original+DR} x {no augmentation, with augmentation}

Runs locally (RTX 4060 Laptop GPU, 8GB VRAM) - no pod needed. The dataset is
tiny (20-40 images), so YOLO26s at 640px comfortably fits an 8GB card and
each run finishes in minutes, not hours.

Each regime starts from a FRESH yolo26s.pt checkpoint (no state carried over
between runs, so all 4 are a fair independent comparison) and lands in its
own clearly-named folder:

    runs/train/original_no_aug/       weights/best.pt, results.csv, results.png, ...
    runs/train/original_aug/
    runs/train/original_dr_no_aug/
    runs/train/original_dr_aug/

The "no augmentation" runs load dataset/hyp/no_aug.yaml (every augmentation
term zeroed - see assemble_dataset.py for why this is a training-time toggle
rather than pre-baked files); the "with augmentation" runs use Ultralytics'
own built-in defaults untouched.

data.yaml's val: currently points at images/train (Stage 4's pilot-scale
default, no held-out val set) - so these are train-set metrics, expected to
be optimistic. That's a known, flagged limitation at this pilot scale, not a
bug - see the plan doc's Stage 4 section.

    python src/train_yolo.py
    python src/train_yolo.py --epochs 50 --regimes original_no_aug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REGIMES = {
    "original_no_aug":    {"data": "dataset/original/data.yaml",    "hyp": "dataset/hyp/no_aug.yaml"},
    "original_aug":       {"data": "dataset/original/data.yaml",    "hyp": None},
    "original_dr_no_aug": {"data": "dataset/original_dr/data.yaml", "hyp": "dataset/hyp/no_aug.yaml"},
    "original_dr_aug":    {"data": "dataset/original_dr/data.yaml", "hyp": None},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = ultralytics auto-batch from available VRAM")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", type=Path, default=Path("runs/train"))
    ap.add_argument("--regimes", nargs="+", choices=sorted(REGIMES), default=sorted(REGIMES))
    args = ap.parse_args()

    # Resolve to absolute: this machine's global ultralytics settings.json has
    # its own runs_dir (and an unrelated project's datasets_dir) that silently
    # doubles up any relative project= path - e.g. "runs/train" landed at
    # "runs/detect/runs/train/<name>" in testing. Absolute sidesteps that.
    args.project = args.project.resolve()

    from ultralytics import YOLO

    summary = {}
    for name in args.regimes:
        cfg = REGIMES[name]
        print(f"\n{'=' * 70}\n{name}  (data={cfg['data']}  hyp={cfg['hyp'] or 'ultralytics defaults'})\n{'=' * 70}")

        model = YOLO(args.model)  # fresh pretrained checkpoint per regime - no cross-run state
        train_kwargs = dict(
            data=cfg["data"],
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.project),
            name=name,
            exist_ok=True,
            plots=True,
        )
        if cfg["hyp"]:
            train_kwargs["cfg"] = cfg["hyp"]
        model.train(**train_kwargs)

        metrics = model.val(data=cfg["data"], device=args.device, project=str(args.project), name=f"{name}_val", exist_ok=True)
        summary[name] = {
            "data": cfg["data"],
            "hyp": cfg["hyp"],
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

    summary_path = args.project / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\nsummary ({len(summary)} regime(s))\n{'=' * 70}")
    print(f"{'regime':<22}{'mAP50':>8}{'mAP50-95':>10}{'precision':>11}{'recall':>9}")
    for name, m in summary.items():
        print(f"{name:<22}{m['map50']:>8.3f}{m['map50_95']:>10.3f}{m['precision']:>11.3f}{m['recall']:>9.3f}")
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
