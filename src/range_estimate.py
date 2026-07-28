"""Stage 3a - range estimation. DA-V2 Large, guided-filter refined.

Locked in after src/depth_compare.py's head-to-head against Apple Depth Pro -
see AI_Pipeline_Test_Plan.md's Stage 3a section for the comparison and why
DA-V2 (not an underwater-specific model) is the right choice: Stage 1 renders
clean, colour-neutral water on purpose, so these images have none of the
attenuation/backscatter domain shift underwater-specific depth models exist to
correct for.

Per image:
    1. DA-V2 Large -> raw relative inverse depth (disparity-like: higher = nearer)
    2. Guided-filter refinement (src/depth_utils.py) using the source RGB
       image's luminance as an edge guide - a training-free sharpening step,
       not a fix for the model's fundamental smoothness (see Stage 3a's
       "why not fine-tune" note in the plan doc)
    3. Normalize to [0,1], linearly flip (1-x - see disparity_to_range()'s
       docstring for why NOT a reciprocal, caught via an actual bug), then
       remap to [z_near, z_far] metres
    4. z_near is fixed at 0.3m; z_far is sampled per image, seeded by the
       image's own generation seed for reproducibility, from a framing-
       dependent range - the plan doc's original [2,4]m only covered
       close-up/mid framing, written before the pilot manifest grew "wide"
       framing rows (12-20); extended here so wide shots get a plausibly
       larger range:
           close-up: [1.5, 2.5] m
           mid:      [2.0, 4.0] m  (the plan doc's original default)
           wide:     [3.0, 6.0] m

Standing caveat (from the plan doc): this is an *estimated* range map of a
*generated* scene - a plausible randomization driver for Stage 3b, not ground
truth, and nothing downstream should treat it as measured.

Reads each image's sidecar JSON (written by generate.py) for its "framing"
and "seed" - both scripts must have already run.

    python src/range_estimate.py --images-dir outputs/1-pilot/klein
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from depth_utils import guided_filter, local_norm

MODEL_REPO = "depth-anything/Depth-Anything-V2-Large-hf"

Z_NEAR = 0.3
Z_FAR_RANGE_BY_FRAMING = {
    "close-up": (1.5, 2.5),
    "mid": (2.0, 4.0),
    "wide": (3.0, 6.0),
}


def load_sidecar(image_path: Path) -> dict:
    sidecar = image_path.with_suffix(".json")
    if not sidecar.exists():
        raise SystemExit(f"no sidecar JSON for {image_path} (expected {sidecar}) - run src/generate.py first")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def disparity_to_range(disp, framing: str, seed: int):
    """Guided-filtered disparity -> metric-ish range map z(x,y) in metres.

    Linear flip (1 - normalized_disparity), NOT a second reciprocal. DA-V2's
    raw disparity is already approximately proportional to 1/z, which is
    exactly what gives near objects more spread than far ones after percentile
    normalization (confirmed in depth_compare.py's comparison - that expanded
    near-field detail is why DA-V2 was picked over Depth Pro). Reciprocating
    an already-reciprocal quantity undoes that property: it re-compresses the
    near field and lets the far background dominate the range again - caught
    empirically (75th percentile of a first attempt landed at 0.33m, barely
    above z_near, with the jump to z_far only in the top few percent of
    pixels). A plain linear flip preserves the existing distribution and just
    orients it the right way (near = small z, far = large z).
    """
    disp_norm = local_norm(disp)
    rel_distance = 1.0 - disp_norm

    lo, hi = Z_FAR_RANGE_BY_FRAMING[framing]
    z_far = random.Random(seed).uniform(lo, hi)
    z = Z_NEAR + rel_distance * (z_far - Z_NEAR)
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

    out_dir = args.out_dir or args.images_dir.parent / "range"
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
        disp = pipe(image)["predicted_depth"].squeeze().to(torch.float32).cpu().numpy()

        guide = np.asarray(image.convert("L")).astype(np.float64) / 255.0
        if guide.shape != disp.shape:
            guide = np.asarray(Image.fromarray((guide * 255).astype(np.uint8)).resize(disp.shape[::-1])).astype(np.float64) / 255.0
        disp_refined = guided_filter(guide, disp, radius=8, eps=1e-3)

        z, z_far = disparity_to_range(disp_refined, framing, seed)
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
