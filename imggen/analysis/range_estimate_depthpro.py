"""One-off Stage 3a variant: same pipeline as range_estimate.py, but with Apple Depth Pro (apple/DepthPro-hf) swapped in for DA-V2 as the depth backbone."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from imggen.analysis.depth_utils import guided_filter, local_norm

MODEL_REPO = "apple/DepthPro-hf"

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
    depth_norm = local_norm(depth)

    lo, hi = Z_FAR_RANGE_BY_FRAMING[framing]
    z_far = random.Random(seed).uniform(lo, hi)
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
