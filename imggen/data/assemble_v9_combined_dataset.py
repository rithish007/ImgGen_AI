"""Build a single YOLO-format dataset combining flux2dev v9's two independent 1000-image base sets - the original v8-prompt run (outputs/flux2dev/v9) and the starfish-camouflage-prompt run (outputs/flux2dev/v9_starfish) - into one 2000-image pool."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

RESOLUTION = 1024
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-images-dir", type=Path, default=Path("outputs/flux2dev/v9"))
    ap.add_argument("--base-labels-dir", type=Path, default=Path("outputs/flux2dev/v9/labels/sam3"))
    ap.add_argument("--starfish-images-dir", type=Path, default=Path("outputs/flux2dev/v9_starfish"))
    ap.add_argument("--starfish-labels-dir", type=Path, default=Path("outputs/flux2dev/v9_starfish/labels/sam3"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/v9_flux2dev_combined"))
    ap.add_argument("--val-count", type=int, default=100, help="val GROUPS (image_ids), not individual images - same 100-out-of-1000 group ratio as every other dataset in this project; since each group yields 2 images here, this holds the same 10-pct image-level val ratio (200 of 2000), not double it")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_label_paths = sorted(args.base_labels_dir.glob("*.txt"))
    sf_label_paths = {p.stem: p for p in args.starfish_labels_dir.glob("*.txt")}

    groups = []
    missing_starfish = 0
    for base_label_path in base_label_paths:
        stem = base_label_path.stem
        base_img = args.base_images_dir / f"{stem}.png"
        if not base_img.exists():
            raise SystemExit(f"missing base image for {base_label_path} (expected {base_img})")

        sf_label_path = sf_label_paths.get(stem)
        sf_img = args.starfish_images_dir / f"{stem}.png"
        sf_pair = (sf_img, sf_label_path) if sf_label_path and sf_img.exists() else None
        if sf_pair is None:
            missing_starfish += 1

        groups.append({"stem": stem, "base": (base_img, base_label_path), "starfish": sf_pair})

    print(f"{len(groups)} image_id groups, {missing_starfish} missing a starfish-set counterpart")

    rng = random.Random(args.seed)
    rng.shuffle(groups)
    val_groups = groups[: args.val_count]
    train_groups = groups[args.val_count :]

    def place(img_src: Path, lbl_src: Path, suffix: str, img_dir: Path, lbl_dir: Path) -> None:
        out_stem = f"{img_src.stem}_{suffix}"
        with Image.open(img_src) as im:
            im = im.convert("RGB")
            if im.size != (RESOLUTION, RESOLUTION):
                im = im.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            im.save(img_dir / f"{out_stem}.png")
        lbl_dir.joinpath(f"{out_stem}.txt").write_text(lbl_src.read_text(encoding="utf-8"), encoding="utf-8")

    for split_name, split_groups in [("train", train_groups), ("val", val_groups)]:
        img_dir = args.out_dir / "images" / split_name
        lbl_dir = args.out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for group in split_groups:
            place(*group["base"], "base", img_dir, lbl_dir)
            count += 1
            if group["starfish"]:
                place(*group["starfish"], "starfish", img_dir, lbl_dir)
                count += 1
        print(f"{split_name}: {count} images ({len(split_groups)} image_id groups) -> {img_dir}")

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
