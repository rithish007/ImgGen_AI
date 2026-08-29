"""Build a YOLO-format dataset from flux2dev v9's base images COMBINED with
their duo_calibrated_scatter DR'd copies (base images get domain-randomized
into additional variants, not replaced - matching this project's established
DR convention, see imggen/data/assemble_dataset.py). 900/100 train/val split by base
image identity (a base image and its DR'd copy always land in the same
split, so a recoloured copy of a val image can never leak into train).

Mirrors imggen/data/assemble_v9_duo_dataset.py exactly, pointed at the new
duo_calibrated_scatter profile (imggen/dr/randomize_v2.py) instead of
duo_calibrated.

Run on the pod after imggen/data/annotate.py has produced
outputs/flux2dev/v9/labels/sam3/ (base, already exists).

The DR'd copy reuses the base image's label file verbatim rather than an
independent SAM3 re-annotation of the DR'd pixels - domain randomization is
pixel-only, so object positions never change and the base boxes are still
exactly correct. An earlier version of this script pointed at a separate
outputs/flux2dev/v9/dr_runs/v1/labels/sam3_duo_calibrated_scatter/ directory
built by re-running SAM3 on the recolored+blurred images; comparing the two
against the base labels (IoU>=0.5 per class, all 1000 images) showed SAM3
recovered only 85.6% of the base instances on the scatter pixels (14.4%
silently missing - the worst of the three DR profiles, consistent with this
being the most visually degraded transform) - a real detection-recall gap
between SAM3 on clean vs. hazy/recolored renders, not intentional label
diversity. Reusing the base label file removes that gap entirely.

    python -m imggen.data.assemble_v9_duo_scatter_dataset
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

RESOLUTION = 1024  # native - see assemble_v9_dataset.py's docstring for why this changed from 640
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-images-dir", type=Path, default=Path("outputs/flux2dev/v9"))
    ap.add_argument("--base-labels-dir", type=Path, default=Path("outputs/flux2dev/v9/labels/sam3"))
    ap.add_argument("--dr-images-dir", type=Path, default=Path("outputs/flux2dev/v9/dr_runs/v1/dr_duo_calibrated_scatter"))
    ap.add_argument("--dr-suffix", default="_duo_scatter_dr")
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/v9_flux2dev_duo_scatter_dr"))
    ap.add_argument("--val-count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_label_paths = sorted(args.base_labels_dir.glob("*.txt"))
    items = []
    missing_dr = 0
    for label_path in base_label_paths:
        stem = label_path.stem
        base_img = args.base_images_dir / f"{stem}.png"
        if not base_img.exists():
            raise SystemExit(f"missing base image for {label_path} (expected {base_img})")

        dr_img = args.dr_images_dir / f"{stem}{args.dr_suffix}.png"
        # DR is pixel-only, so the base label file is still exactly correct for it.
        dr_pair = (dr_img, label_path) if dr_img.exists() else None
        if dr_pair is None:
            missing_dr += 1

        items.append({"stem": stem, "base": (base_img, label_path), "dr": dr_pair})

    print(f"{len(items)} base images, {missing_dr} missing a DR counterpart")

    rng = random.Random(args.seed)
    rng.shuffle(items)
    val_items = items[: args.val_count]
    train_items = items[args.val_count :]

    def place(img_src: Path, lbl_src: Path, img_dir: Path, lbl_dir: Path) -> None:
        with Image.open(img_src) as im:
            im = im.convert("RGB")
            if im.size != (RESOLUTION, RESOLUTION):
                im = im.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            im.save(img_dir / f"{img_src.stem}.png")
        lbl_dir.joinpath(f"{img_src.stem}.txt").write_text(lbl_src.read_text(encoding="utf-8"), encoding="utf-8")

    for split_name, split_items in [("train", train_items), ("val", val_items)]:
        img_dir = args.out_dir / "images" / split_name
        lbl_dir = args.out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for item in split_items:
            place(*item["base"], img_dir, lbl_dir)
            count += 1
            if item["dr"]:
                place(*item["dr"], img_dir, lbl_dir)
                count += 1
        print(f"{split_name}: {count} images ({len(split_items)} base identities) -> {img_dir}")

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
