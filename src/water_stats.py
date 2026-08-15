"""Tier 2 / statistical-similarity route for the open Akkaynak-Treibitz
reverse-engineering task (see AI_Pipeline_Test_Plan.md and the DA-V2 vs
Depth Pro vs YOLO26 comparison work this follows on from) - depth-free
descriptive water-appearance statistics, computed directly from pixels.

This does NOT fit the Akkaynak-Treibitz beta/B_inf equation (that needs a
real per-pixel METRIC range map, which neither DUO nor a generated image
has - see the CIRS/NTNU discussion for the route that does need it). Instead
it computes cheaper, depth-free descriptors that summarise "what underwater
water conditions look like" as plain pixel statistics:

    - mean_rgb            per-channel mean intensity, [0,1]
    - ratio_rg, ratio_bg  mean_R/mean_G, mean_B/mean_G - color-cast
                           direction/magnitude as a single number per image.
                           Same ratio SHAPE as jerlov.py's beta_bg/beta_br,
                           but NOT the same physical quantity - these are raw
                           pixel-statistic ratios of an already-attenuated
                           image, not attenuation-COEFFICIENT ratios. Don't
                           conflate the two when reading results.
    - dark_channel_mean   mean of the dark-channel map (min over R,G,B in a
                           local patch, He/Sun/Tang CVPR 2009) - a standard
                           haze-magnitude proxy, higher = hazier. Same paper
                           family depth_utils.py's guided filter already
                           comes from (He, Sun & Tang, ECCV 2010).
    - luminance_std        std of grayscale luminance - simple global-contrast
                           backup to dark_channel_mean.
    - veiling_light_rgb   airlight/veiling-light colour estimate: among the
                           haziest 0.1% of pixels (highest dark-channel
                           value), average the brightest few of those in the
                           original image - He/Sun/Tang's atmospheric-light
                           estimation step. Directly comparable in spirit to
                           domain_randomize.py's resolved B_inf_rgb.

Per-image stats are aggregated into distributions (mean/median/std/p10/p90)
across a whole directory - those aggregate numbers are the actual "values"
this script exists to produce, for later comparison against the same stats
run on generated or DR'd output. This script only computes and reports the
values for one directory at a time; it does not compare two directories.

Dark-channel patch size is expressed as a fraction of the image's shorter
side (not a fixed pixel count) so results stay comparable across datasets
at different resolutions - DUO's exports are 640x640, this pipeline's raw
generations are 1024x1024.

Pure numpy/scipy/PIL - no model weights, no GPU, matches domain_randomize.py
and depth_utils.py's existing "deterministic pixel-only" tooling.

    python src/water_stats.py --images-dir dataset/DUO_Dataset/train/images --out reports/duo_train_water_stats.json
    python src/water_stats.py --images-dir outputs/5-pilot/flux2dev_v4 --out reports/5pilot_flux2dev_v4_water_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter

DARK_CHANNEL_PATCH_FRAC = 15 / 640  # He/Sun/Tang's 15px default at DUO's 640px, expressed as a fraction so it scales to other resolutions
AIRLIGHT_TOP_DARK_FRAC = 0.001  # top 0.1% haziest pixels by dark-channel value, per He/Sun/Tang
AIRLIGHT_TOP_N = 25  # of those, average the N brightest - a single brightest pixel is noisy (specular highlights, sensor artifacts)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def dark_channel(img: np.ndarray, patch: int) -> np.ndarray:
    """(H,W,3) float [0,1] -> (H,W) dark-channel map (He, Sun & Tang, CVPR
    2009): min over R,G,B per pixel, then a local minimum over a patch x
    patch window.
    """
    channel_min = img.min(axis=-1)
    return minimum_filter(channel_min, size=patch, mode="nearest")


def estimate_veiling_light(img: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """Airlight/veiling-light colour: among the AIRLIGHT_TOP_DARK_FRAC
    haziest pixels (highest dark-channel value), average the AIRLIGHT_TOP_N
    most intense of those in the original image.
    """
    flat_dc = dc.reshape(-1)
    n_top = max(1, int(flat_dc.size * AIRLIGHT_TOP_DARK_FRAC))
    haziest_idx = np.argpartition(flat_dc, -n_top)[-n_top:]

    flat_rgb = img.reshape(-1, 3)
    intensities = flat_rgb[haziest_idx].sum(axis=1)
    n_bright = min(AIRLIGHT_TOP_N, len(haziest_idx))
    brightest_of_haziest = haziest_idx[np.argpartition(intensities, -n_bright)[-n_bright:]]

    return flat_rgb[brightest_of_haziest].mean(axis=0)


def image_stats(path: Path) -> dict:
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
    h, w = img.shape[:2]

    patch = max(3, int(round(DARK_CHANNEL_PATCH_FRAC * min(h, w))) | 1)  # odd, >=3
    dc = dark_channel(img, patch)
    veiling_light = estimate_veiling_light(img, dc)

    mean_rgb = img.reshape(-1, 3).mean(axis=0)
    luminance = img @ np.array([0.299, 0.587, 0.114])

    return {
        "id": path.stem,
        "mean_rgb": mean_rgb.tolist(),
        "ratio_rg": float(mean_rgb[0] / mean_rgb[1]),
        "ratio_bg": float(mean_rgb[2] / mean_rgb[1]),
        "dark_channel_mean": float(dc.mean()),
        "luminance_std": float(luminance.std()),
        "veiling_light_rgb": veiling_light.tolist(),
    }


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def summarize_rgb(values: list[list[float]]) -> dict:
    arr = np.asarray(values, dtype=np.float64)  # (N, 3)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "median": np.median(arr, axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap number of images processed (debug/speed)")
    args = ap.parse_args()

    image_paths = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise SystemExit(f"no images found in {args.images_dir}")
    if args.limit:
        image_paths = image_paths[: args.limit]

    per_image = []
    for i, path in enumerate(image_paths, start=1):
        per_image.append(image_stats(path))
        if i % 500 == 0 or i == len(image_paths):
            print(f"[{i}/{len(image_paths)}] {path.name}")

    aggregate = {
        "mean_rgb": summarize_rgb([r["mean_rgb"] for r in per_image]),
        "ratio_rg": summarize([r["ratio_rg"] for r in per_image]),
        "ratio_bg": summarize([r["ratio_bg"] for r in per_image]),
        "dark_channel_mean": summarize([r["dark_channel_mean"] for r in per_image]),
        "luminance_std": summarize([r["luminance_std"] for r in per_image]),
        "veiling_light_rgb": summarize_rgb([r["veiling_light_rgb"] for r in per_image]),
    }

    report = {
        "source": str(args.images_dir),
        "n_images": len(per_image),
        "aggregate": aggregate,
        "per_image": per_image,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mean_rgb = [round(v, 3) for v in aggregate["mean_rgb"]["mean"]]
    veiling = [round(v, 3) for v in aggregate["veiling_light_rgb"]["mean"]]
    print(f"\n{len(per_image)} images -> {args.out}")
    print(f"  mean_rgb (R,G,B)      {mean_rgb}")
    print(f"  ratio_rg  mean/median {aggregate['ratio_rg']['mean']:.3f} / {aggregate['ratio_rg']['median']:.3f}  (std {aggregate['ratio_rg']['std']:.3f})")
    print(f"  ratio_bg  mean/median {aggregate['ratio_bg']['mean']:.3f} / {aggregate['ratio_bg']['median']:.3f}  (std {aggregate['ratio_bg']['std']:.3f})")
    print(f"  dark_channel_mean     {aggregate['dark_channel_mean']['mean']:.3f}  (std {aggregate['dark_channel_mean']['std']:.3f})")
    print(f"  luminance_std         {aggregate['luminance_std']['mean']:.3f}  (std {aggregate['luminance_std']['std']:.3f})")
    print(f"  veiling_light_rgb     {veiling}")


if __name__ == "__main__":
    main()
