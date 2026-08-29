"""Train yolo26x.pt on a v9 "combined" dataset variant.

The combined base set is the original v8-prompt v9 run PLUS the
starfish-camouflage-prompt v9_starfish run (see
imggen/data/assemble_v9_combined_dataset.py and its DR siblings). Each variant
adds a different domain-randomization copy on top, or none for `base`.

Uses the heavy-augmentation/optimizer regime established by
imggen/train/v9_duo_v3.py - see that module's docstring for the full rationale
(MuSGD + lr0=0.0004 matching YOLO26's official X-scale training; mosaic=0.95 /
mixup=0.35 / copy_paste=0.35 to attack memorization; scale=0.9 targeting the
diagnosed object-scale mismatch; hsv well above stock).

Consolidated from four copy-pasted scripts (v9_combined, _duo, _duo_scatter,
_placeholder) that differed only in dataset path and run name. Those values now
live in configs/experiments/v9_combined.json.

    python -m imggen.train.v9_combined --variant base
    python -m imggen.train.v9_combined --variant duo
    python -m imggen.train.v9_combined --variant duo_scatter
    python -m imggen.train.v9_combined --variant placeholder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from imggen import REPO_ROOT

CONFIG = REPO_ROOT / "configs/experiments/v9_combined.json"
REAL_EVAL_DATA = "dataset/real_eval/data.yaml"
EPOCHS = 150
PATIENCE = 30
IMGSZ = 896


def load_variant(name: str) -> dict:
    variants = json.loads(CONFIG.read_text(encoding="utf-8"))["variants"]
    if name not in variants:
        raise SystemExit(f"unknown variant {name!r} - available: {', '.join(sorted(variants))}")
    return variants[name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="variant key from configs/experiments/v9_combined.json")
    args = ap.parse_args()

    cfg = load_variant(args.variant)
    data_yaml = f"{cfg['dataset']}/data.yaml"
    run_name = cfg["run_name"]
    project = Path("runs/train").resolve()

    model = YOLO("yolo26x.pt")
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=-1,
        device="0",
        project=str(project),
        name=run_name,
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

    val_own = model.val(data=data_yaml, device="0", project=str(project), name=f"{run_name}_val_own", exist_ok=True)
    val_duo = model.val(data=REAL_EVAL_DATA, device="0", project=str(project), name=f"{run_name}_val_duo", exist_ok=True)

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
    summary_path = project / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nsummary -> {summary_path}")
    print(f"own val:  mAP50={summary['own_val']['map50']:.4f}  mAP50-95={summary['own_val']['map50_95']:.4f}")
    print(f"DUO test: mAP50={summary['duo_test']['map50']:.4f}  mAP50-95={summary['duo_test']['map50_95']:.4f}  "
          f"P={summary['duo_test']['precision']:.4f}  R={summary['duo_test']['recall']:.4f}")


if __name__ == "__main__":
    main()
