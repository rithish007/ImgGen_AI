"""Confusion matrix on real DUO predictions -- the evidence every run's own
confusion_matrix.png is missing, because Ultralytics only ever computes that
plot against the synthetic held-out split (Chapter 6, "Synthetic Validation
Carries No Transfer Signal"). This script builds the real-domain equivalent:
IoU-matches one run's predictions on the 778-image DUO test split against
ground truth, and tallies a (class + background) x (class + background)
matrix, so misclassification (wrong class, right box) can be told apart from
a pure miss (false negative) and a pure false alarm (false positive).

Matching, per image: predictions are taken in confidence order (highest
first, thresholded at --conf); each is matched to the highest-IoU unmatched
ground-truth box of ANY class at IoU >= --iou. A matched pair with equal
classes is a diagonal hit; a matched pair with different classes is a
misclassification (off-diagonal). An unmatched prediction is a false
positive against the "background" column; an unmatched ground-truth box is a
false negative against the "background" row. This mirrors Ultralytics'
own ConfusionMatrix convention (rows = predicted, columns = true), so it
reads the same way as the synthetic-split matrix a reader may already know.

    python -m imggen.analysis.confusion_matrix_real \
        --pred-labels runs/predict_duo/v9_flux2dev_duo_scatter_dr_v2_detections/labels \
        --gt-labels dataset/real_eval/labels \
        --conf 0.25 --iou 0.5 \
        --out reports/analysis/confusion_matrix_real.json \
        --figure thesis/Writing/figures/fig_confusion_matrix_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imggen.prompts.base import class_names

CLASS_NAMES = class_names()
N = len(CLASS_NAMES)
BG = N  # background row/col index


def read_yolo(path: Path, with_conf: bool) -> list[tuple[int, list[float], float]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cid = int(parts[0])
        box = [float(x) for x in parts[1:5]]
        conf = float(parts[5]) if with_conf and len(parts) > 5 else 1.0
        out.append((cid, box, conf))
    return out


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two YOLO (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_image(preds: list[tuple[int, list[float], float]],
                 gts: list[tuple[int, list[float], float]],
                 iou_thresh: float, matrix: list[list[int]]) -> None:
    preds = sorted(preds, key=lambda p: -p[2])
    gt_used = [False] * len(gts)
    for pcid, pbox, _conf in preds:
        best_iou, best_j = 0.0, -1
        for j, (_gcid, gbox, _) in enumerate(gts):
            if gt_used[j]:
                continue
            i = iou(pbox, gbox)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_thresh and best_j >= 0:
            gt_used[best_j] = True
            gcid = gts[best_j][0]
            matrix[pcid][gcid] += 1  # row=predicted, col=true
        else:
            matrix[pcid][BG] += 1  # prediction with no matching GT -> false positive
    for used, (gcid, _, _) in zip(gt_used, gts):
        if not used:
            matrix[BG][gcid] += 1  # GT box no prediction matched -> false negative


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-labels", required=True, type=Path)
    ap.add_argument("--gt-labels", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--figure", type=Path, default=None,
                     help="if given, also render <figure>.pdf/.png (a heatmap)")
    args = ap.parse_args()

    size = N + 1
    matrix = [[0] * size for _ in range(size)]

    gt_stems = {p.stem for p in args.gt_labels.glob("*.txt")}
    for stem in sorted(gt_stems):
        gts = read_yolo(args.gt_labels / f"{stem}.txt", with_conf=False)
        preds_raw = read_yolo(args.pred_labels / f"{stem}.txt", with_conf=True)
        preds = [p for p in preds_raw if p[2] >= args.conf]
        match_image(preds, gts, args.iou, matrix)

    labels = [*CLASS_NAMES.values(), "background"]
    result = {
        "meta": {"pred_labels": str(args.pred_labels), "gt_labels": str(args.gt_labels),
                  "conf_threshold": args.conf, "iou_threshold": args.iou, "n_images": len(gt_stems)},
        "labels": labels,
        "matrix_rows_predicted_cols_true": matrix,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"confusion matrix ({len(gt_stems)} images, conf>={args.conf}, iou>={args.iou}) -> {args.out}\n")
    header = "predicted \\ true".ljust(16) + "".join(f"{l:>12}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        print(row_label.ljust(16) + "".join(f"{matrix[i][j]:>12}" for j in range(size)))

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        m = np.array(matrix, dtype=float)
        col_sums = m.sum(axis=0, keepdims=True)
        norm = np.divide(m, col_sums, out=np.zeros_like(m), where=col_sums > 0)

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(size)); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(size)); ax.set_yticklabels(labels)
        ax.set_xlabel("true"); ax.set_ylabel("predicted")
        for i in range(size):
            for j in range(size):
                if matrix[i][j] == 0:
                    continue
                colour = "white" if norm[i, j] > 0.6 else "black"
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=9, color=colour)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="column-normalised")
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(f"{args.figure}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"\nfigure -> {args.figure}.pdf / .png")


if __name__ == "__main__":
    main()
