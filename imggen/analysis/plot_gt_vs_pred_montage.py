"""GT-vs-prediction montage on real DUO test images, for the best run
(v9_flux2dev_duo_scatter_dr_v2, Calibrated+scatter / tuned -- Table
tab:transfer's top cell).

One panel per class in a 1x3 row. Each panel overlays, on the same real image,
the ground-truth boxes (solid) and the run's confidence-thresholded (>=0.25)
predictions (dashed), so a matched pair reads as two boxes on one object and a
miss reads as a ground-truth box with no prediction beside it. The frames are
chosen so every panel contains at least one miss: starfish GT 4/pred 3,
sea urchin GT 6/pred 5, scallop GT 8/pred 3 -- the dominant failure mode
Chapter 6 reports across all three classes.

Boxes are drawn as matplotlib patches (not baked into the raster) so they stay
crisp in the PDF; only the underwater photograph itself is raster.

    python -m imggen.analysis.plot_gt_vs_pred_montage
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = {0: "(a) Starfish", 1: "(b) Sea Urchins", 2: "(c) Scallops"}
GT_COLOR = "#FFE100"    # solid yellow -- high contrast on teal/green water
PRED_COLOR = "#FF3B30"  # dashed red

GT_LABELS = ROOT / "dataset/real_eval/labels"
GT_IMAGES = ROOT / "dataset/real_eval/images"
PRED_LABELS = ROOT / "runs/predict_duo/v9_flux2dev_duo_scatter_dr_v2_detections/labels"

# one representative frame per class (see module docstring)
# scallop: 6393_jpg... (GT=8, pred=3) -- chosen from slide 15 of the dissertation PPT
PANELS = [
    (0, "5264_jpg.rf.1a373f7b37cbdbb7022203416393e7e7"),
    (1, "3914_jpg.rf.9c5e8c49058599febd7dfac225b24cce"),
    (2, "6393_jpg.rf.71468a4319ec01c5dabdb14756ceacbe"),
]


def load_boxes(label_path: Path, class_id: int, with_conf: bool = False):
    boxes = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return boxes
    for line in lines:
        parts = line.split()
        if not parts or int(parts[0]) != class_id:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if with_conf and len(parts) > 5 else None
        boxes.append((cx, cy, w, h, conf))
    return boxes


def add_boxes(ax, boxes, w, h, color, dashed, show_conf):
    for cx, cy, bw, bh, conf in boxes:
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        ax.add_patch(Rectangle(
            (x1, y1), bw * w, bh * h, fill=False, edgecolor=color,
            linewidth=2.4, linestyle="--" if dashed else "-",
        ))
        if show_conf and conf is not None:
            ax.text(x1 + 2, y1 + 4, f"{conf:.2f}", color="white", fontsize=8.5,
                    va="top", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.8, alpha=0.9))


def main() -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 5.2), dpi=220)

    for ax, (class_id, stem) in zip(axes, PANELS):
        name = CLASS_NAMES[class_id]
        im = np.asarray(Image.open(GT_IMAGES / f"{stem}.jpg").convert("RGB"))
        h, w = im.shape[:2]
        ax.imshow(im)

        gt = load_boxes(GT_LABELS / f"{stem}.txt", class_id, with_conf=False)
        pred = load_boxes(PRED_LABELS / f"{stem}.txt", class_id, with_conf=True)
        add_boxes(ax, gt, w, h, GT_COLOR, dashed=False, show_conf=False)
        add_boxes(ax, pred, w, h, PRED_COLOR, dashed=True, show_conf=True)

        ax.set_title(name, fontsize=13)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [
        Line2D([0], [0], color=GT_COLOR, lw=2.6, linestyle="-", label="ground truth"),
        Line2D([0], [0], color=PRED_COLOR, lw=2.6, linestyle="--", label="prediction (with confidence)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))

    out = ROOT / "thesis/Writing/figures/fig_qualitative_montage"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"saved -> {out}.pdf / .png")


if __name__ == "__main__":
    main()
