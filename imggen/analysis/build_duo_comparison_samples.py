"""Pick a handful of DUO test images (starfish-heavy, scallop-heavy, and
sea_urchin-heavy) and assemble GT-vs-3-models comparison images: the
ground-truth boxes drawn on the original photo, side by side with each
pilot model's already-rendered detection image (from
runs/predict_duo/<variant>_detections/).

Run on the pod after imggen/eval/pilot_on_duo.py has produced the
runs/predict_duo/*/  detection folders.

    python -m imggen.analysis.build_duo_comparison_samples
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}
CLASS_COLORS = {0: (255, 80, 80), 1: (80, 180, 255), 2: (255, 210, 60)}
REAL_EVAL_IMAGES = Path("dataset/real_eval/images")
REAL_EVAL_LABELS = Path("dataset/real_eval/labels")
VARIANTS = ["pilot_base", "pilot_dr", "pilot_dr_anchored", "v9_flux2dev"]
OUT_DIR = Path("runs/duo_comparison_samples")
N_PER_CLASS = 4
MANIFEST_PATH = OUT_DIR / "manifest.json"


def load_gt(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cid, cx, cy, w, h = line.split()
        boxes.append((int(cid), float(cx), float(cy), float(w), float(h)))
    return boxes


def draw_gt(image_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> Image.Image:
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for cid, cx, cy, bw, bh in boxes:
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        color = CLASS_COLORS[cid]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), CLASS_NAMES[cid], fill=color)
    return im


def main() -> None:
    if MANIFEST_PATH.exists():
        # reuse the same previously-selected images so new models slot into
        # an apples-to-apples comparison instead of a freshly randomized set
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        print(f"reusing {len(manifest)} previously-selected samples from {MANIFEST_PATH}")
    else:
        by_class: dict[int, list[Path]] = defaultdict(list)
        for label_path in sorted(REAL_EVAL_LABELS.glob("*.txt")):
            boxes = load_gt(label_path)
            for cid in {b[0] for b in boxes}:
                by_class[cid].append(label_path)
        print("images containing each class:", {CLASS_NAMES[k]: len(v) for k, v in by_class.items()})

        rng = random.Random(7)
        selected: dict[str, list[Path]] = {}
        for cid, name in CLASS_NAMES.items():
            pool = by_class.get(cid, [])
            rng.shuffle(pool)
            selected[name] = pool[:N_PER_CLASS]

        manifest = []
        for focus_class, label_paths in selected.items():
            for label_path in label_paths:
                stem = label_path.stem
                img_path = REAL_EVAL_IMAGES / f"{stem}.jpg"
                if not img_path.exists():
                    img_path = next(REAL_EVAL_IMAGES.glob(f"{stem}.*"), None)
                if img_path is None:
                    continue
                boxes = load_gt(label_path)
                manifest.append({
                    "focus_class": focus_class,
                    "image_id": stem,
                    "gt_box_count": len(boxes),
                    "gt_classes": sorted({CLASS_NAMES[b[0]] for b in boxes}),
                    "dir": str(OUT_DIR / focus_class / stem),
                })

    for entry in manifest:
        stem = entry["image_id"]
        img_path = REAL_EVAL_IMAGES / f"{stem}.jpg"
        if not img_path.exists():
            img_path = next(REAL_EVAL_IMAGES.glob(f"{stem}.*"), None)
        label_path = REAL_EVAL_LABELS / f"{stem}.txt"

        sample_dir = Path(entry["dir"])
        sample_dir.mkdir(parents=True, exist_ok=True)

        if not (sample_dir / "0_ground_truth.jpg").exists():
            boxes = load_gt(label_path)
            draw_gt(img_path, boxes).save(sample_dir / "0_ground_truth.jpg", quality=90)

        for i, variant in enumerate(VARIANTS, start=1):
            out_path = sample_dir / f"{i}_{variant}.jpg"
            if out_path.exists():
                continue
            pred_img = Path(f"runs/predict_duo/{variant}_detections") / img_path.name
            if pred_img.exists():
                Image.open(pred_img).convert("RGB").save(out_path, quality=90)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} comparison samples -> {OUT_DIR}")


if __name__ == "__main__":
    main()
