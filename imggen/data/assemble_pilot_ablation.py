"""Pilot-scale (50+50 image) YOLO dataset assembly for the 3-way DR ablation requested for the RunPod YOLO26x pipeline check: dataset/pilot_base/ flux2dev v8 + hunyuan v7 base images only dataset/pilot_dr/ base + non-anchored DR (dr_runs/v1/dr) dataset/pilot_dr_anchored/ base + anchored DR (dr_runs/v1/dr_anchored) This is a PIPELINE CHECK at 50-image-per-model scale, not the production dataset (that's the pending 1000-image/model SLURM job on Stanage)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

RESOLUTION = 640
CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}

SOURCES = {
    "flux2dev": {
        "base_img": Path("outputs/flux2dev/v8"),
        "base_lbl": Path("outputs/flux2dev/v8/labels/sam3"),
        "base_glob": "2-pilot_*_flux2dev_bf16.png",
        "dr_img": Path("outputs/flux2dev/v8/dr_runs/v1/dr"),
        "dr_lbl": Path("outputs/flux2dev/v8/dr_runs/v1/labels/sam3"),
        "dr_suffix": "_dr",
        "anchored_img": Path("outputs/flux2dev/v8/dr_runs/v1/dr_anchored"),
        "anchored_lbl": Path("outputs/flux2dev/v8/dr_runs/v1/labels/sam3_anchored"),
        "anchored_suffix": "_anchored_dr",
    },
    "hunyuan": {
        "base_img": Path("outputs/hunyuan/v7"),
        "base_lbl": Path("outputs/hunyuan/v7/labels/sam3"),
        "base_glob": "2-pilot_*_hunyuan.png",
        "dr_img": Path("outputs/hunyuan/v7/dr_runs/v1/dr"),
        "dr_lbl": None,
        "dr_suffix": "_dr",
        "anchored_img": Path("outputs/hunyuan/v7/dr_runs/v1/dr_anchored"),
        "anchored_lbl": None,
        "anchored_suffix": "_anchored_dr",
    },
}


def collect_items(model: str, cfg: dict) -> list[dict]:
    items = []
    for label_path in sorted(cfg["base_lbl"].glob("*.txt")):
        stem = label_path.stem
        base_img = cfg["base_img"] / f"{stem}.png"
        if not base_img.exists():
            raise SystemExit(f"missing base image for label {label_path} (expected {base_img})")

        dr_img = cfg["dr_img"] / f"{stem}{cfg['dr_suffix']}.png"
        dr_lbl = (cfg["dr_lbl"] / f"{stem}{cfg['dr_suffix']}.txt") if cfg["dr_lbl"] else label_path
        anchored_img = cfg["anchored_img"] / f"{stem}{cfg['anchored_suffix']}.png"
        anchored_lbl = (cfg["anchored_lbl"] / f"{stem}{cfg['anchored_suffix']}.txt") if cfg["anchored_lbl"] else label_path

        items.append({
            "model": model,
            "image_id": stem,
            "base": (base_img, label_path),
            "dr": (dr_img, dr_lbl) if dr_img.exists() else None,
            "anchored": (anchored_img, anchored_lbl) if anchored_img.exists() else None,
        })
    return items


def split_items(items: list[dict], val_per_model: int, seed: int) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for model in sorted({it["model"] for it in items}):
        model_items = sorted((it for it in items if it["model"] == model), key=lambda it: it["image_id"])
        rng = random.Random(seed)
        rng.shuffle(model_items)
        val.extend(model_items[:val_per_model])
        train.extend(model_items[val_per_model:])
    return train, val


def place_pair(img_src: Path, lbl_src: Path, img_dir: Path, lbl_dir: Path, stem: str) -> None:
    with Image.open(img_src) as im:
        im = im.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        im.save(img_dir / f"{stem}.png")
    lbl_dir.joinpath(f"{stem}.txt").write_text(lbl_src.read_text(encoding="utf-8"), encoding="utf-8")


def write_split(items: list[dict], split_name: str, out_dir: Path, include: list[str]) -> int:
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in items:
        for key in include:
            pair = item.get(key) if key != "base" else item["base"]
            if pair is None:
                continue
            img_src, lbl_src = pair
            place_pair(img_src, lbl_src, img_dir, lbl_dir, img_src.stem)
            count += 1
    return count


def write_data_yaml(out_dir: Path) -> None:
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for cid in sorted(CLASS_NAMES):
        lines.append(f"  {cid}: {CLASS_NAMES[cid]}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--val-per-model", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_items = []
    for model, cfg in SOURCES.items():
        model_items = collect_items(model, cfg)
        print(f"{model}: {len(model_items)} base images "
              f"({sum(1 for it in model_items if it['dr'])} with dr, "
              f"{sum(1 for it in model_items if it['anchored'])} with anchored)")
        all_items.extend(model_items)

    train_items, val_items = split_items(all_items, args.val_per_model, args.seed)
    print(f"\nsplit: train={len(train_items)} images (raw count), val={len(val_items)} images (raw count)")

    variants = {
        "pilot_base": ["base"],
        "pilot_dr": ["base", "dr"],
        "pilot_dr_anchored": ["base", "anchored"],
    }
    for variant, include in variants.items():
        out_dir = args.out_dir / variant
        n_train = write_split(train_items, "train", out_dir, include)
        n_val = write_split(val_items, "val", out_dir, include)
        write_data_yaml(out_dir)
        print(f"{variant}: train={n_train} val={n_val} -> {out_dir}/data.yaml")


if __name__ == "__main__":
    main()
