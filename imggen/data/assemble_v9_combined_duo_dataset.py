"""Build a YOLO-format dataset combining BOTH flux2dev v9 base sets (the
original v8-prompt run and the starfish-camouflage-prompt run, see
assemble_v9_combined_dataset.py) with their duo_calibrated-profile DR'd
copies.

Four images per image_id group: v9 base, v9 base+duo_calibrated DR,
v9_starfish, v9_starfish+duo_calibrated DR - up to 4000 images total.
Grouped and split by image_id (not individual image) so no same-seed pair
(base<->its own DR, v9<->v9_starfish) can straddle train/val - see
assemble_v9_combined_dataset.py and assemble_v9_duo_dataset.py for the
identical leakage reasoning applied here across all four variants at once.

Each DR'd copy reuses its OWN source set's base label file verbatim (v9's DR
reuses v9's label, v9_starfish's DR reuses v9_starfish's label) rather than a
fresh SAM3 pass on the DR'd pixels - domain randomization is pixel-only, so
object positions never move and the base boxes stay exactly correct. See
assemble_v9_duo_dataset.py's docstring for the measured SAM3-recall-gap
finding that motivated this (12.4-14.4% instance loss re-annotating hazy/
recoloured pixels vs. reusing the clean-image labels).

Output filenames are always derived from the shared image_id stem plus an
explicit role suffix (_base / _base_dr / _starfish / _starfish_dr) - never
from the DR file's own already-suffixed name, to avoid ambiguous double
suffixes.

val-count is GROUPS (image_ids), not images - 100 groups matches every other
dataset's 10% ratio here (100/1000 groups -> 400/4000 images, still 10%,
since all four variants scale together).

Run after both sets have duo_calibrated DR'd copies + SAM3 labels:
    outputs/flux2dev/v9/dr_runs/v1/dr_duo_calibrated/                (v9 base)
    outputs/flux2dev/v9_starfish/dr_runs/v1/dr_duo_calibrated/       (v9_starfish)
    outputs/flux2dev/v9/labels/sam3/                                 (v9 base labels)
    outputs/flux2dev/v9_starfish/labels/sam3/                        (v9_starfish labels)

    python -m imggen.data.assemble_v9_combined_duo_dataset
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

RESOLUTION = 1024  # native - see assemble_v9_dataset.py's docstring for why this isn't a fixed downscale
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}
DR_SUFFIX = "_duo_dr"


def collect(images_dir: Path, labels_dir: Path, dr_images_dir: Path) -> dict:
    """Per image_id: base (image, label) always present; dr (image, label)
    present only if the DR'd copy exists - dr's label is the SAME base
    label_path, reused verbatim (see module docstring)."""
    entries = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        base_img = images_dir / f"{stem}.png"
        if not base_img.exists():
            raise SystemExit(f"missing image for {label_path} (expected {base_img})")
        dr_img = dr_images_dir / f"{stem}{DR_SUFFIX}.png"
        entries[stem] = {
            "base": (base_img, label_path),
            "dr": (dr_img, label_path) if dr_img.exists() else None,
        }
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v9-images-dir", type=Path, default=Path("outputs/flux2dev/v9"))
    ap.add_argument("--v9-labels-dir", type=Path, default=Path("outputs/flux2dev/v9/labels/sam3"))
    ap.add_argument("--v9-dr-images-dir", type=Path, default=Path("outputs/flux2dev/v9/dr_runs/v1/dr_duo_calibrated"))
    ap.add_argument("--starfish-images-dir", type=Path, default=Path("outputs/flux2dev/v9_starfish"))
    ap.add_argument("--starfish-labels-dir", type=Path, default=Path("outputs/flux2dev/v9_starfish/labels/sam3"))
    ap.add_argument("--starfish-dr-images-dir", type=Path, default=Path("outputs/flux2dev/v9_starfish/dr_runs/v1/dr_duo_calibrated"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/v9_flux2dev_combined_duo_dr"))
    ap.add_argument("--val-count", type=int, default=100, help="val GROUPS (image_ids) - 100-out-of-1000 holds the same 10-pct image-level val ratio here (400 of 4000) as every other dataset in this project")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v9_entries = collect(args.v9_images_dir, args.v9_labels_dir, args.v9_dr_images_dir)
    sf_entries = collect(args.starfish_images_dir, args.starfish_labels_dir, args.starfish_dr_images_dir)

    stems = sorted(set(v9_entries) & set(sf_entries))
    missing = sorted(set(v9_entries) ^ set(sf_entries))
    if missing:
        raise SystemExit(f"{len(missing)} image_ids present in only one source set (expected both): {missing[:5]}...")

    missing_v9_dr = sum(1 for s in stems if v9_entries[s]["dr"] is None)
    missing_sf_dr = sum(1 for s in stems if sf_entries[s]["dr"] is None)
    print(f"{len(stems)} image_id groups, {missing_v9_dr} missing v9 duo_calibrated DR, {missing_sf_dr} missing starfish duo_calibrated DR")

    rng = random.Random(args.seed)
    rng.shuffle(stems)
    val_stems = stems[: args.val_count]
    train_stems = stems[args.val_count :]

    def place(stem: str, role: str, img_src: Path, lbl_src: Path, img_dir: Path, lbl_dir: Path) -> None:
        out_stem = f"{stem}_{role}"
        with Image.open(img_src) as im:
            im = im.convert("RGB")
            if im.size != (RESOLUTION, RESOLUTION):
                im = im.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            im.save(img_dir / f"{out_stem}.png")
        lbl_dir.joinpath(f"{out_stem}.txt").write_text(lbl_src.read_text(encoding="utf-8"), encoding="utf-8")

    for split_name, split_stems in [("train", train_stems), ("val", val_stems)]:
        img_dir = args.out_dir / "images" / split_name
        lbl_dir = args.out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for stem in split_stems:
            for role_prefix, entries in (("base", v9_entries[stem]), ("starfish", sf_entries[stem])):
                img_src, lbl_src = entries["base"]
                place(stem, role_prefix, img_src, lbl_src, img_dir, lbl_dir)
                count += 1
                if entries["dr"]:
                    img_src, lbl_src = entries["dr"]
                    place(stem, f"{role_prefix}_dr", img_src, lbl_src, img_dir, lbl_dir)
                    count += 1
        print(f"{split_name}: {count} images ({len(split_stems)} image_id groups) -> {img_dir}")

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
