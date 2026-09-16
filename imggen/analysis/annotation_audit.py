"""Stage 2 annotation-quality audit -- the SAM3 *annotator ceiling*."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from imggen.prompts.base import class_names
from imggen.analysis.visualize_annotations import CLASS_COLOURS

CLASS_NAMES = class_names()
NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def read_label(path: Path) -> list[tuple[int, list[float]]]:
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


def draw_numbered_boxes(image_path: Path, boxes: list[tuple[int, list[float]]]) -> "Image.Image":
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
