"""Contact-sheet montage of sample images from a generation output directory.
No GPU, no model weights - Pillow is the only dependency.

Samples evenly across the sorted file list rather than taking the first N, so
a manifest ordered by density/framing block (dense/close-up first, then
moderate/mid, then sparse/wide - see manifests/2-pilot.json) doesn't produce a
montage skewed toward one scene-scale condition.

    python src/montage.py outputs/flux2dev/v8
    python src/montage.py outputs/flux2dev/v8 --n 20 --cols 5 --out outputs/reports/montage_flux2dev_v8.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

THUMB = 300
LABEL_H = 24
PAD = 4


def sample_evenly(paths: list[Path], n: int) -> list[Path]:
    if n >= len(paths):
        return paths
    step = len(paths) / n
    return [paths[int(i * step)] for i in range(n)]


def build_montage(image_paths: list[Path], cols: int, out_path: Path) -> None:
    rows = (len(image_paths) + cols - 1) // cols
    cell_w = THUMB + PAD * 2
    cell_h = THUMB + LABEL_H + PAD * 2

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#1c1c1c")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for i, path in enumerate(image_paths):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w + PAD, r * cell_h + PAD

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB, THUMB))
            paste_x = x0 + (THUMB - im.width) // 2
            paste_y = y0 + (THUMB - im.height) // 2
            sheet.paste(im, (paste_x, paste_y))

        label = path.stem.replace("_flux2dev_bf16", "").replace("_hunyuan", "").replace("_klein", "")
        draw.text((x0, y0 + THUMB + 2), label, fill="#e0e0e0", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"montage saved -> {out_path} ({len(image_paths)} images, {cols}x{rows} grid)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", type=Path, help="directory of generated PNGs")
    ap.add_argument("--n", type=int, default=20, help="number of images to sample (default 20)")
    ap.add_argument("--cols", type=int, default=5, help="grid columns (default 5)")
    ap.add_argument("--out", type=Path, default=None, help="output PNG path")
    args = ap.parse_args()

    paths = sorted(args.input_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNGs found in {args.input_dir}")

    sampled = sample_evenly(paths, args.n)
    # e.g. outputs/flux2dev/v8 -> "flux2dev_v8", not just "v8" (which would
    # collide across models sharing a version number)
    tag = "_".join(args.input_dir.parts[-2:]) if len(args.input_dir.parts) >= 2 else args.input_dir.name
    out_path = args.out or Path("outputs/reports") / f"montage_{tag}.png"
    build_montage(sampled, args.cols, out_path)


if __name__ == "__main__":
    main()
