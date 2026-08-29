"""Build a single YOLO-format dataset from flux2dev v9 (the 1000-image
production run) + its fresh SAM3 labels. Straightforward 90/10 train/val
split, kept at native 1024x1024 (source is already square, so labels carry
over unchanged - see imggen/data/assemble_dataset.py for why that only holds for a
square source).

Stored at native resolution deliberately, NOT pre-downscaled to a fixed
training size - Ultralytics' own dataloader letterboxes/resizes to whatever
`imgsz` a given training run uses, so pre-shrinking here just throws away
detail the dataloader would otherwise have available. An earlier version of
this (and every other assemble_v9*.py script) hard-resized to 640, which was
harmless while every run also trained at imgsz=640, but became actively
counterproductive once the heavy-aug recipe moved to imgsz=896: every image
was being upsampled from an already-shrunk 640px copy instead of the
1024px original.

Run on the pod, from /workspace/ImgGen_AI, after imggen/data/annotate.py has
produced outputs/flux2dev/v9/labels/sam3/.

    python -m imggen.data.assemble_v9_dataset
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

RESOLUTION = 1024
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=Path("outputs/flux2dev/v9"))
    ap.add_argument("--labels-dir", type=Path, default=Path("outputs/flux2dev/v9/labels/sam3"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/v9_flux2dev"))
    ap.add_argument("--val-count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    label_paths = sorted(args.labels_dir.glob("*.txt"))
    items = []
    for label_path in label_paths:
        stem = label_path.stem
        img_path = args.images_dir / f"{stem}.png"
        if not img_path.exists():
            raise SystemExit(f"missing image for label {label_path} (expected {img_path})")
        items.append((img_path, label_path))

    rng = random.Random(args.seed)
    rng.shuffle(items)
    val_items = items[: args.val_count]
    train_items = items[args.val_count :]

    for split_name, split_items in [("train", train_items), ("val", val_items)]:
        img_dir = args.out_dir / "images" / split_name
        lbl_dir = args.out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, label_path in split_items:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                if im.size != (RESOLUTION, RESOLUTION):
                    im = im.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
                im.save(img_dir / f"{img_path.stem}.png")
            lbl_dir.joinpath(f"{label_path.stem}.txt").write_text(
                label_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        print(f"{split_name}: {len(split_items)} images -> {img_dir}")

    lines = [
        f"path: {args.out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for cid in sorted(CLASS_NAMES):
        lines.append(f"  {cid}: {CLASS_NAMES[cid]}")
    (args.out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\ndata.yaml -> {args.out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
