"""Draw YOLO-format label boxes on top of their source images, for Checkpoint
2.5 visual review. Pure PIL - no GPU, no model weights, runs anywhere.

    python src/visualize_annotations.py --images-dir outputs/1-pilot/klein --labels-dir outputs/1-pilot/labels/sam3 --out-dir outputs/1-pilot/viz/sam3

Also doubles as the Stage 4 pre-flight check: DR'd images always ship with
their ORIGINAL clean-image label (see domain_randomize.py and the plan doc's
Stage 3b section for why), but the DR'd PNGs are named "<stem>_dr.png" while
the label is "<stem>.txt" - use --strip-suffix to bridge that:

    python src/visualize_annotations.py --images-dir outputs/1-pilot/dr --labels-dir outputs/1-pilot/labels/sam3 --strip-suffix _dr --out-dir outputs/1-pilot/dr_labeled

(outputs/1-pilot/labels/gdino and viz/gdino are historical - see annotate.py's
module docstring for why Grounding DINO was dropped from the active workflow)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prompts import CLASSES

# One distinct, high-contrast colour per class_id.
CLASS_COLOURS = {
    0: (255, 60, 60),    # starfish - red
    1: (60, 220, 60),    # sea urchin - green
    2: (60, 140, 255),   # scallop - blue
}


def draw_boxes(image: Image.Image, label_path: Path) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
        return img

    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        class_id_s, cx_s, cy_s, w_s, h_s = line.split()
        class_id = int(class_id_s)
        cx, cy, w, h = float(cx_s), float(cy_s), float(w_s), float(h_s)

        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        colour = CLASS_COLOURS.get(class_id, (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)

        label = CLASSES.get(class_id, {}).get("short", f"class{class_id}")
        text_bg = [x1, max(0, y1 - 24), x1 + 9 * len(label) + 8, max(24, y1)]
        draw.rectangle(text_bg, fill=colour)
        draw.text((x1 + 4, max(0, y1 - 24)), label, fill=(0, 0, 0), font=font)

    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--strip-suffix", default="", help="strip this suffix from the image stem before looking up its label file")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"no .png files found in {args.images_dir}")

    for path in image_paths:
        stem = path.stem[: -len(args.strip_suffix)] if args.strip_suffix and path.stem.endswith(args.strip_suffix) else path.stem
        label_path = args.labels_dir / f"{stem}.txt"
        image = Image.open(path)
        annotated = draw_boxes(image, label_path)
        out_path = args.out_dir / path.name
        annotated.save(out_path)

        n_boxes = 0
        if label_path.exists():
            n_boxes = len([l for l in label_path.read_text(encoding="utf-8").strip().splitlines() if l])
        print(f"{path.name}: {n_boxes} boxes -> {out_path}")


if __name__ == "__main__":
    main()
