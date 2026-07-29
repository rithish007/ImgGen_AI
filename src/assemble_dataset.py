"""Stage 4 - dataset assembly. Produces two YOLO-format dataset variants from
the pilot outputs:

    dataset/original/     raw Stage 1 images only
    dataset/original_dr/  raw + DR'd images, the same label reused for both

Data-augmentation architecture (decided 2026-07-29): augmentation is applied
ON-THE-FLY during YOLO training via Ultralytics' built-in hyp config (HSV
jitter, flips, rotation, mosaic, etc.), not pre-baked into extra files here.
Ultralytics correctly transforms box labels alongside its own geometric
augmentation, so the "geometric ops corrupt labels" risk doesn't apply to
on-the-fly augmentation the way it does to a hand-rolled pre-baked transform -
Stage 3's DR pipeline stayed strictly non-geometric specifically because IT
bakes fixed files with a reused label, a different situation from training-
time augmentation. This means only 2 dataset variants need assembling here,
not 4: the intended 2x2 (DR x augmentation) becomes {this script's 2 variants}
x {this script's 2 hyp configs, applied at Stage 5 train time}.

Images are square (1024x1024) throughout this pipeline, so resizing to 640x640
is a pure isotropic scale - YOLO's normalized box coordinates (cx,cy,w,h as
fractions of width/height) are invariant to that, so labels are copied
unchanged rather than recomputed. This would NOT hold if source images were
non-square (a resize that changes aspect ratio, or any crop, does require
recomputing coordinates) - flagged since it's a real assumption baked into
this script, not a general-purpose resize utility.

A raw image and its DR'd copy always land in the same split - grouped by
image_id BEFORE splitting, so a recoloured copy of a training image can never
end up in val while its raw twin is in train (leakage).

    python src/assemble_dataset.py
    python src/assemble_dataset.py --val-count 4

Stage 5 (training) stays ON HOLD per the plan doc until explicitly resumed -
this script prepares what Stage 5 will need (both dataset variants, plus a
no-augmentation hyp override for two of the four planned runs) but does not
itself invoke any training.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

from prompts import class_names

RESOLUTION = 640

HYP_NO_AUG = """\
# Ultralytics hyp override - all augmentation disabled, for the two
# "original"/"original_dr" (no-augmentation) runs of the planned 2x2.
# Usage: yolo train data=dataset/original/data.yaml model=yolo26s.pt cfg=dataset/hyp/no_aug.yaml
# The other two runs (with augmentation) use Ultralytics' own built-in
# defaults directly - no override file needed for those.
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
erasing: 0.0
"""


def collect_items(klein_dir: Path, dr_dir: Path, labels_dir: Path) -> list[dict]:
    items = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        raw_path = klein_dir / f"{stem}.png"
        if not raw_path.exists():
            raise SystemExit(f"missing raw image for label {label_path} (expected {raw_path})")
        dr_path = dr_dir / f"{stem}_dr.png"
        items.append({
            "image_id": stem,
            "raw_path": raw_path,
            "dr_path": dr_path if dr_path.exists() else None,
            "label_path": label_path,
        })
    return items


def split_ids(image_ids: list[str], val_count: int, seed: int) -> tuple[set, set]:
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_ids = set(ids[:val_count])
    train_ids = set(ids[val_count:])
    return train_ids, val_ids


def place_image(src_image: Path, img_dir: Path, src_label: Path, lbl_dir: Path, stem: str) -> None:
    with Image.open(src_image) as im:
        im = im.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        im.save(img_dir / f"{stem}.png")
    shutil.copy(src_label, lbl_dir / f"{stem}.txt")


def write_split(items: list[dict], split_name: str, out_dir: Path, include_dr: bool) -> int:
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in items:
        place_image(item["raw_path"], img_dir, item["label_path"], lbl_dir, item["raw_path"].stem)
        count += 1
        if include_dr and item["dr_path"] is not None:
            place_image(item["dr_path"], img_dir, item["label_path"], lbl_dir, item["dr_path"].stem)
            count += 1
    return count


def write_data_yaml(out_dir: Path, has_val: bool) -> None:
    names = class_names()
    val_target = "images/val" if has_val else "images/train"  # no held-out val at pilot scale - see plan doc
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        f"val: {val_target}",
        "names:",
    ]
    for cid in sorted(names):
        lines.append(f"  {cid}: {names[cid]}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--klein-dir", type=Path, default=Path("outputs/1-pilot/klein"))
    ap.add_argument("--dr-dir", type=Path, default=Path("outputs/1-pilot/dr"))
    ap.add_argument("--labels-dir", type=Path, default=Path("outputs/1-pilot/labels/sam3"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--val-count", type=int, default=0, help="image_ids held out for val (raw+DR share a split); 0 = pilot-scale default, all train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = collect_items(args.klein_dir, args.dr_dir, args.labels_dir)
    if not items:
        raise SystemExit("no labeled images found")

    missing_dr = [it["image_id"] for it in items if it["dr_path"] is None]
    if missing_dr:
        print(f"warning: no DR'd copy for {len(missing_dr)} image(s), will be raw-only in original_dr too: {missing_dr}")

    train_ids, val_ids = split_ids([it["image_id"] for it in items], args.val_count, args.seed)
    train_items = [it for it in items if it["image_id"] in train_ids]
    val_items = [it for it in items if it["image_id"] in val_ids]

    for variant, include_dr in [("original", False), ("original_dr", True)]:
        out_dir = args.out_dir / variant
        n_train = write_split(train_items, "train", out_dir, include_dr)
        n_val = write_split(val_items, "val", out_dir, include_dr) if val_items else 0
        write_data_yaml(out_dir, has_val=bool(val_items))
        print(f"{variant}: train={n_train} val={n_val} -> {out_dir}/data.yaml")

    hyp_dir = args.out_dir / "hyp"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    (hyp_dir / "no_aug.yaml").write_text(HYP_NO_AUG, encoding="utf-8")
    print(f"\nno-augmentation hyp override -> {hyp_dir / 'no_aug.yaml'}")
    print("(the with-augmentation runs use ultralytics' own built-in defaults - no override file needed)")
    print(f"\nplanned Stage 5 2x2: (original | original_dr) x (hyp/no_aug.yaml | ultralytics defaults) = 4 runs")


if __name__ == "__main__":
    main()
