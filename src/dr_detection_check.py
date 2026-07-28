"""Quantifies whether DR'd instances stay detectable, by matching the
ORIGINAL clean-image ground-truth boxes (outputs/1-pilot/labels/sam3/)
against SAM3's own re-detection on the DR'd images (outputs/1-pilot/labels/dr_sam3/).

This is a diagnostic only - it does NOT determine which labels ship with the
DR'd images. Those are always the original clean-image labels, unchanged
(see domain_randomize.py and AI_Pipeline_Test_Plan.md's Stage 3b section for
why re-detecting on the DR'd image and using THAT as ground truth would be
wrong - it would record false negatives for real, correctly-labeled
instances). What this answers instead: how much of the induced haze/noise/
vignette pushed a real instance past what a strong foundation model can still
find at all - i.e. is the DR set's difficulty calibrated sensibly, or has
some of it gone past "hard but learnable" into "erased."

For each ground-truth box, finds the best-IoU box among that image's DR
re-detections (regardless of class) and buckets it:
    matched        - same class, IoU >= threshold
    misclassified  - different class, IoU >= threshold (the wrong-class
                      concern from chat - a degraded region getting flagged
                      under the WRONG class's independent SAM3 pass)
    missed         - no DR detection reaches the IoU threshold at all

    python src/dr_detection_check.py
    python src/dr_detection_check.py --iou-threshold 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prompts import CLASSES


def load_yolo_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        cls, cx, cy, w, h = line.split()
        boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return boxes


def to_xyxy(box: tuple[int, float, float, float, float]) -> tuple[float, float, float, float]:
    _, cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = to_xyxy(box_a)
    bx1, by1, bx2, by2 = to_xyxy(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(gt_boxes, dr_boxes, iou_threshold: float) -> list[dict]:
    results = []
    for gt in gt_boxes:
        best_iou, best_dr = 0.0, None
        for dr in dr_boxes:
            i = iou(gt, dr)
            if i > best_iou:
                best_iou, best_dr = i, dr

        if best_dr is None or best_iou < iou_threshold:
            outcome = "missed"
        elif best_dr[0] == gt[0]:
            outcome = "matched"
        else:
            outcome = "misclassified"

        results.append({
            "gt_class": gt[0],
            "outcome": outcome,
            "best_iou": best_iou,
            "matched_class": best_dr[0] if best_dr is not None else None,
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-labels", type=Path, default=Path("outputs/1-pilot/labels/sam3"))
    ap.add_argument("--dr-labels", type=Path, default=Path("outputs/1-pilot/labels/dr_sam3"))
    ap.add_argument("--dr-suffix", default="_dr", help="suffix DR label stems have beyond the gt stem")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    args = ap.parse_args()

    gt_paths = sorted(args.gt_labels.glob("*.txt"))
    if not gt_paths:
        raise SystemExit(f"no .txt files found in {args.gt_labels}")

    per_class = {cid: {"matched": 0, "misclassified": 0, "missed": 0} for cid in CLASSES}
    per_image_missed = []

    for gt_path in gt_paths:
        dr_path = args.dr_labels / f"{gt_path.stem}{args.dr_suffix}.txt"
        gt_boxes = load_yolo_boxes(gt_path)
        dr_boxes = load_yolo_boxes(dr_path)
        if not gt_boxes:
            continue

        results = match_image(gt_boxes, dr_boxes, args.iou_threshold)
        image_missed = 0
        for r in results:
            per_class[r["gt_class"]][r["outcome"]] += 1
            if r["outcome"] != "matched":
                image_missed += 1

        if image_missed:
            per_image_missed.append((gt_path.stem, image_missed, len(gt_boxes)))

    print(f"{'class':<14}{'total':>7}{'matched':>10}{'misclass':>10}{'missed':>9}{'missed%':>9}")
    print("-" * 59)
    grand_total = {"matched": 0, "misclassified": 0, "missed": 0}
    for cid in sorted(CLASSES):
        c = per_class[cid]
        total = sum(c.values())
        for k in grand_total:
            grand_total[k] += c[k]
        missed_pct = c["missed"] / total * 100 if total else 0.0
        print(f"{CLASSES[cid]['short']:<14}{total:>7}{c['matched']:>10}{c['misclassified']:>10}{c['missed']:>9}{missed_pct:>8.1f}%")

    grand_total_n = sum(grand_total.values())
    overall_missed_pct = grand_total["missed"] / grand_total_n * 100 if grand_total_n else 0.0
    overall_misclass_pct = grand_total["misclassified"] / grand_total_n * 100 if grand_total_n else 0.0
    print("-" * 59)
    print(f"{'ALL':<14}{grand_total_n:>7}{grand_total['matched']:>10}{grand_total['misclassified']:>10}{grand_total['missed']:>9}{overall_missed_pct:>8.1f}%")

    if grand_total["misclassified"]:
        print(f"\n{grand_total['misclassified']} instance(s) matched a DR detection of the WRONG class ({overall_misclass_pct:.1f}%) - worth a manual look at these specific images.")

    if per_image_missed:
        per_image_missed.sort(key=lambda t: -t[1] / t[2])
        print("\nimages with the highest fraction of missed/misclassified instances:")
        for stem, missed, total in per_image_missed[:10]:
            print(f"  {stem:<20} {missed}/{total} ({missed/total*100:.0f}%)")


if __name__ == "__main__":
    main()
