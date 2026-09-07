"""Stage 2 annotation-quality audit -- the SAM3 *annotator ceiling*.

Every detector in this project is trained on SAM3-generated labels, never on
human annotation, so the whole experiment is bounded above by how good those
labels are (see Chapter 4, "Detector-generated labels" threat). That ceiling
is not something the pipeline measures for free: a generated image has no
ground truth, so the only way to quantify SAM3's precision/recall is to put a
human in the loop on a sample. This module runs that loop in two steps.

    build   sample N images, draw the SAM3 boxes with a per-box index, and
            write a review_template.json the reviewer fills in by hand.
    score   read the (filled) template back and compute per-class precision,
            recall, F1 and a localisation-quality rate -- the ceiling numbers
            that go into Chapter 5's annotation-quality section.

The reviewer never edits the label files. They only mark, per predicted box,
whether it is a true or false positive (and whether the box is tight), and
per image how many real instances of each class SAM3 *missed*. Precision comes
from tp/(tp+fp); recall from tp/(tp+missed); everything else follows.

    # 1. make a 60-image review pack from the v9 production labels
    python -m imggen.analysis.annotation_audit build \
        --images-dir outputs/flux2dev/v9 \
        --labels-dir outputs/flux2dev/v9/labels/sam3 \
        --n 60 --seed 7 \
        --out-dir reports/annotation_audit/v9

    # 2. (human fills reports/annotation_audit/v9/review_template.json)

    # 3. compute the ceiling
    python -m imggen.analysis.annotation_audit score \
        --review reports/annotation_audit/v9/review_template.json \
        --out reports/annotation_audit/v9_ceiling.json

The review_template.json schema (one entry per sampled image)::

    {
      "meta": {...},                 # written by build, do not edit
      "images": [
        {
          "image": "..._0007.png",
          "boxes": [
            {"id": 0, "class": "scallop", "box": [cx,cy,w,h],
             "verdict": null,        # reviewer -> "tp" | "fp"
             "localisation": null},  # reviewer -> "tight" | "loose"  (tp only)
            ...
          ],
          "missed": {                # reviewer -> count of real instances
            "starfish": null,        #            SAM3 gave NO box for
            "sea_urchin": null,
            "scallop": null
          }
        },
        ...
      ]
    }

Pure PIL/stdlib -- no GPU, no model weights, runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from imggen.prompts.base import class_names
from imggen.analysis.visualize_annotations import CLASS_COLOURS

CLASS_NAMES = class_names()  # {0: "starfish", 1: "sea_urchin", 2: "scallop"}
NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# label IO
# ---------------------------------------------------------------------------

def read_label(path: Path) -> list[tuple[int, list[float]]]:
    """YOLO txt -> [(class_id, [cx, cy, w, h]), ...]; empty/missing -> []."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        cid, cx, cy, w, h = line.split()
        boxes.append((int(cid), [float(cx), float(cy), float(w), float(h)]))
    return boxes


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        p = images_dir / f"{stem}{suf}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# build: sample + draw numbered overlays + emit the review template
# ---------------------------------------------------------------------------

def draw_numbered_boxes(image_path: Path, boxes: list[tuple[int, list[float]]]) -> "Image.Image":
    """Overlay every box with its per-image index, so the reviewer can point at
    box '3' in the template. Colour matches visualize_annotations (per class)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default()

    for idx, (cid, (cx, cy, bw, bh)) in enumerate(boxes):
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        colour = CLASS_COLOURS.get(cid, (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        tag = f"{idx}:{CLASS_NAMES.get(cid, cid)}"
        tw = 11 * len(tag) + 8
        draw.rectangle([x1, max(0, y1 - 28), x1 + tw, max(28, y1)], fill=colour)
        draw.text((x1 + 4, max(0, y1 - 28)), tag, fill=(0, 0, 0), font=font)
    return img


def cmd_build(args: argparse.Namespace) -> None:
    label_paths = sorted(args.labels_dir.glob("*.txt"))
    if not label_paths:
        raise SystemExit(f"no .txt labels in {args.labels_dir}")

    rng = random.Random(args.seed)
    sample = sorted(rng.sample(label_paths, min(args.n, len(label_paths))),
                    key=lambda p: p.stem)

    overlays_dir = args.out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    n_boxes = 0
    for lp in sample:
        img_path = find_image(args.images_dir, lp.stem)
        if img_path is None:
            print(f"  ! no image for {lp.stem}, skipping")
            continue
        boxes = read_label(lp)
        draw_numbered_boxes(img_path, boxes).save(overlays_dir / f"{lp.stem}.png")
        entries.append({
            "image": img_path.name,
            "boxes": [
                {"id": i, "class": CLASS_NAMES.get(cid, str(cid)), "box": box,
                 "verdict": None, "localisation": None}
                for i, (cid, box) in enumerate(boxes)
            ],
            "missed": {name: None for name in NAME_TO_ID},
        })
        n_boxes += len(boxes)

    template = {
        "meta": {
            "images_dir": str(args.images_dir),
            "labels_dir": str(args.labels_dir),
            "n_images": len(entries),
            "n_boxes": n_boxes,
            "seed": args.seed,
            "instructions": (
                "For each box set verdict to 'tp' (a real instance of that class "
                "is inside it) or 'fp' (nothing/other class). For a 'tp' also set "
                "localisation to 'tight' or 'loose'. For each image set missed[class] "
                "to the number of real instances of that class SAM3 gave NO box for."
            ),
        },
        "images": entries,
    }
    template_path = args.out_dir / "review_template.json"
    template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")

    print(f"built review pack: {len(entries)} images, {n_boxes} boxes")
    print(f"  overlays -> {overlays_dir}")
    print(f"  template -> {template_path}  (fill verdict/localisation/missed by hand)")


# ---------------------------------------------------------------------------
# score: read the filled template -> ceiling metrics
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def cmd_score(args: argparse.Namespace) -> None:
    data = json.loads(args.review.read_text(encoding="utf-8"))
    images = data["images"]

    per_class = {name: {"tp": 0, "fp": 0, "fn": 0, "tight": 0} for name in NAME_TO_ID}
    unreviewed_boxes = 0
    unreviewed_missed = 0

    for entry in images:
        for b in entry["boxes"]:
            cls = b["class"]
            if cls not in per_class:
                continue
            v = b.get("verdict")
            if v == "tp":
                per_class[cls]["tp"] += 1
                if b.get("localisation") == "tight":
                    per_class[cls]["tight"] += 1
            elif v == "fp":
                per_class[cls]["fp"] += 1
            else:
                unreviewed_boxes += 1
        for cls, m in entry.get("missed", {}).items():
            if cls not in per_class:
                continue
            if m is None:
                unreviewed_missed += 1
            else:
                per_class[cls]["fn"] += int(m)

    result = {"per_class": {}, "overall": {}}
    tot_tp = tot_fp = tot_fn = tot_tight = 0
    for name, c in per_class.items():
        m = _prf(c["tp"], c["fp"], c["fn"])
        m["localisation_tight_rate"] = round(c["tight"] / c["tp"], 4) if c["tp"] else 0.0
        result["per_class"][name] = m
        tot_tp += c["tp"]; tot_fp += c["fp"]; tot_fn += c["fn"]; tot_tight += c["tight"]

    overall = _prf(tot_tp, tot_fp, tot_fn)
    overall["localisation_tight_rate"] = round(tot_tight / tot_tp, 4) if tot_tp else 0.0
    # macro = unweighted mean over the 3 classes; robust to scallop's rarity
    overall["macro_precision"] = round(
        sum(result["per_class"][n]["precision"] for n in per_class) / len(per_class), 4)
    overall["macro_recall"] = round(
        sum(result["per_class"][n]["recall"] for n in per_class) / len(per_class), 4)
    result["overall"] = overall
    result["review_completeness"] = {
        "n_images": len(images),
        "unreviewed_boxes": unreviewed_boxes,
        "images_with_unreviewed_missed_counts": unreviewed_missed,
        "complete": unreviewed_boxes == 0 and unreviewed_missed == 0,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"ceiling -> {args.out}")
    if unreviewed_boxes or unreviewed_missed:
        print(f"  WARNING: review incomplete -- {unreviewed_boxes} boxes and "
              f"{unreviewed_missed} missed-count fields still null; numbers are partial.")
    print(f"  {'class':<12} {'P':>7} {'R':>7} {'F1':>7} {'tp':>4} {'fp':>4} {'fn':>4}")
    for name, m in result["per_class"].items():
        print(f"  {name:<12} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")
    o = overall
    print(f"  {'OVERALL':<12} {o['precision']:>7.3f} {o['recall']:>7.3f} {o['f1']:>7.3f} "
          f"{o['tp']:>4} {o['fp']:>4} {o['fn']:>4}")
    print(f"  macro P/R: {o['macro_precision']:.3f} / {o['macro_recall']:.3f}   "
          f"localisation-tight: {o['localisation_tight_rate']:.3f}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="sample images + draw overlays + emit review template")
    b.add_argument("--images-dir", required=True, type=Path)
    b.add_argument("--labels-dir", required=True, type=Path)
    b.add_argument("--n", type=int, default=60, help="number of images to sample")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--out-dir", required=True, type=Path)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="compute ceiling metrics from a filled review template")
    s.add_argument("--review", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
