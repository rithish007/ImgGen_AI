"""Builds a real-image evaluation set from the full DUO_Dataset test split
(dataset/DUO_Dataset/test/), remapped to this pipeline's own 3-class scheme.

Why this exists: dataset/DUO_Dataset uses DUO's original 4-class numbering
and polygon-style label lines (Roboflow's YOLO segmentation export always
closes the polygon by repeating the first point - in this dataset every
instance is a plain axis-aligned rectangle expressed as a closed 5-point
polygon, verified by checking that only 4 unique (x,y) values exist per box).
This pipeline trained on a different 3-class scheme (sea cucumber/holothurian
dropped, scallop renumbered - see prompts.py's CLASSES and the plan doc's
Stage 1 section), so DUO's labels can't be used as-is.

Remap: DUO names=['echinus','holothurian','scallop','starfish'] (data.yaml)
    echinus (0)     -> 1 (sea_urchin)
    holothurian (1) -> dropped entirely (this pipeline has no class for it)
    scallop (2)     -> 2 (scallop)
    starfish (3)    -> 0 (starfish)

Images whose only instances were holothurian end up with an empty label
file - kept as background/negative images, not deleted, which is standard
YOLO practice and still gives useful signal (false-positive rate on scenes
with no target class present).

Output mirrors the existing dataset/original/ layout so Ultralytics' own
images->labels sibling-directory convention just works:
    dataset/real_eval/images/*.jpg   (copied from DUO_Dataset/test/images)
    dataset/real_eval/labels/*.txt   (remapped boxes, class cx cy w h)
    dataset/real_eval/data.yaml

    python src/prepare_real_eval.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# DUO_Dataset/test index -> this pipeline's class_id, or None to drop
DUO_TO_PIPELINE = {0: 1, 1: None, 2: 2, 3: 0}


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """(x1,y1,x2,y2,...) closed-polygon coords -> (cx, cy, w, h)."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min


def remap_label_file(src: Path) -> list[str]:
    out_lines = []
    text = src.read_text(encoding="utf-8").strip()
    if not text:
        return out_lines
    for line in text.splitlines():
        parts = line.split()
        duo_cls = int(parts[0])
        pipeline_cls = DUO_TO_PIPELINE.get(duo_cls)
        if pipeline_cls is None:
            continue
        coords = [float(v) for v in parts[1:]]
        cx, cy, w, h = polygon_to_bbox(coords)
        out_lines.append(f"{pipeline_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out_lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duo-test-dir", type=Path, default=Path("dataset/DUO_Dataset/test"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/real_eval"))
    args = ap.parse_args()

    src_images = args.duo_test_dir / "images"
    src_labels = args.duo_test_dir / "labels"
    if not src_images.exists() or not src_labels.exists():
        raise SystemExit(f"expected images/ and labels/ under {args.duo_test_dir}")

    out_images = args.out_dir / "images"
    out_labels = args.out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(src_labels.glob("*.txt"))
    if not label_paths:
        raise SystemExit(f"no label files found in {src_labels}")

    dropped_instances = 0
    empty_images = 0
    class_counts = {0: 0, 1: 0, 2: 0}

    for label_path in label_paths:
        stem = label_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = src_images / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            raise SystemExit(f"no matching image for label {label_path}")

        raw_line_count = len([l for l in label_path.read_text(encoding="utf-8").strip().splitlines() if l])
        remapped = remap_label_file(label_path)
        dropped_instances += raw_line_count - len(remapped)
        if not remapped:
            empty_images += 1
        for line in remapped:
            class_counts[int(line.split()[0])] += 1

        (out_labels / f"{stem}.txt").write_text("\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")
        shutil.copy2(image_path, out_images / image_path.name)

    data_yaml = args.out_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(args.out_dir.resolve()) + "\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: starfish\n"
        "  1: sea_urchin\n"
        "  2: scallop\n",
        encoding="utf-8",
    )

    print(f"images copied: {len(label_paths)}")
    print(f"holothurian instances dropped: {dropped_instances}")
    print(f"images left with zero instances (background/negative): {empty_images}")
    print(f"remapped class counts: starfish={class_counts[0]}  sea_urchin={class_counts[1]}  scallop={class_counts[2]}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
