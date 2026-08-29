"""One-off Stage 3a variant: same pipeline as range_estimate.py, but with
Apple Depth Pro (apple/DepthPro-hf) swapped in for DA-V2 as the depth
backbone. NOT a pipeline stage - range_estimate.py's DA-V2 choice stays the
production default (see that file's docstring for why: DA-V2's relative
disparity gave more near-field spread in depth_compare.py's head-to-head).
This script exists to answer a narrower question: does swapping the depth
backbone change domain_randomize.py's downstream behaviour, and if so, how?

IMPORTANT ASYMMETRY vs DA-V2, output format compatible either way:
    DA-V2's raw output is disparity-like (approx 1/z, higher = NEARER), so
    range_estimate.py normalizes then does a LINEAR FLIP (1 - x) before
    remapping to [z_near, z_far] - see disparity_to_range()'s docstring.
    Depth Pro is a genuinely metric model (apple/DepthPro-hf, "metric": True
    in depth_compare.py's MODELS dict) - its raw output already increases
    with distance (higher = FARTHER), the normal depth-map convention. No
    flip needed here; see metric_to_range() below.

ALSO IMPORTANT - what this does and does NOT change downstream:
    domain_randomize.py's sample_params(seed, z_far, profile) draws every
    SCALAR parameter (water_type, beta_b, d, b_ref, noise/blur/vignette...)
    from `seed` and `z_far` alone. z_far itself comes from
    range_estimate.py's disparity_to_range() as
    random.Random(seed).uniform(*framing_range) - a draw that depends only on
    the seed and the image's framing, NOT on the depth model. This script
    reuses that exact same z_far draw (see main() below), so a v2 DR run
    built from this script's range maps will have BYTE-IDENTICAL scalar
    config values to the v1 (DA-V2) run for the same seed. What actually
    differs is the per-pixel z(x,y) map's SHAPE - Depth Pro vs DA-V2 disagree
    on which pixels are near/far and how sharply, which changes where in the
    frame the AT formula's exponential attenuation lands, not the global
    per-image physics parameters. This is the effect worth comparing, not a
    parameter-table diff.

    python -m imggen.analysis.range_estimate_depthpro --images-dir outputs/flux2dev/v8 --out-dir outputs/flux2dev/v8/dr_runs/range_depthpro
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from imggen.analysis.depth_utils import guided_filter, local_norm

MODEL_REPO = "apple/DepthPro-hf"

# Same constants as range_estimate.py - kept identical so z_far draws match
# byte-for-byte (see module docstring's "ALSO IMPORTANT" note).
Z_NEAR = 0.3
Z_FAR_RANGE_BY_FRAMING = {
    "close-up": (1.5, 2.5),
    "mid": (2.0, 4.0),
    "wide": (3.0, 6.0),
}


def load_sidecar(image_path: Path) -> dict:
    sidecar = image_path.with_suffix(".json")
    if not sidecar.exists():
        raise SystemExit(f"no sidecar JSON for {image_path} (expected {sidecar}) - run the matching imggen.generate module first")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def metric_to_range(depth, framing: str, seed: int):
    """Guided-filtered METRIC depth -> range map z(x,y) in metres.

    No flip (contrast with range_estimate.py's disparity_to_range()): Depth
    Pro's raw output already increases with distance, so normalizing to
    [0,1] and remapping directly preserves near=small/far=large without
    reversing anything. See module docstring.
    """
    depth_norm = local_norm(depth)

    lo, hi = Z_FAR_RANGE_BY_FRAMING[framing]
    z_far = random.Random(seed).uniform(lo, hi)  # identical draw to range_estimate.py, same seed
    z = Z_NEAR + depth_norm * (z_far - Z_NEAR)
    return z, z_far


def save_preview(z, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.imsave(out_path, local_norm(z), cmap="inferno")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=Path("outputs/1-pilot/klein"))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    image_paths = sorted(args.images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"no .png files found in {args.images_dir}")

    out_dir = args.out_dir or args.images_dir.parent / "range_depthpro"
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from transformers import pipeline
    from PIL import Image

    device = 0 if torch.cuda.is_available() else -1
    print(f"loading {MODEL_REPO} (device={'cuda' if device == 0 else 'cpu'})...")
    pipe = pipeline(task="depth-estimation", model=MODEL_REPO, device=device, torch_dtype=torch.float32)

    for i, path in enumerate(image_paths, start=1):
        sidecar = load_sidecar(path)
        framing = sidecar["framing"]
        seed = sidecar["seed"]

        image = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        depth = pipe(image)["predicted_depth"].squeeze().to(torch.float32).cpu().numpy()

        guide = np.asarray(image.convert("L")).astype(np.float64) / 255.0
        if guide.shape != depth.shape:
            guide = np.asarray(Image.fromarray((guide * 255).astype(np.uint8)).resize(depth.shape[::-1])).astype(np.float64) / 255.0
        depth_refined = guided_filter(guide, depth, radius=8, eps=1e-3)

        z, z_far = metric_to_range(depth_refined, framing, seed)
        elapsed = time.perf_counter() - t0

        stem = path.stem
        np.save(out_dir / f"{stem}_range.npy", z.astype(np.float32))
        save_preview(z, out_dir / f"{stem}_range.png")
        (out_dir / f"{stem}_range.json").write_text(
            json.dumps(
                {
                    "image_id": sidecar["image_id"],
                    "source_image": str(path),
                    "model_repo": MODEL_REPO,
                    "framing": framing,
                    "seed": seed,
                    "z_near": Z_NEAR,
                    "z_far": z_far,
                    "guided_filter": {"radius": 8, "eps": 1e-3, "guide": "rgb_luminance"},
                    "seconds": round(elapsed, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{i:>2}/{len(image_paths)}] {stem}  framing={framing}  z_far={z_far:.2f}m  {elapsed:.2f}s")

    print(f"\ndone: {len(image_paths)} range maps -> {out_dir}")


if __name__ == "__main__":
    main()
