"""Stage 3a candidate check: DA V2 Large vs Apple Depth Pro, head to head.

Not a pipeline stage itself - a one-off comparison to decide which model
Stage 3a's range estimation should use. Runs both models on the same clean
Stage 1 images and produces three figures:

  comparison.png       [original | DA-V2 | Depth Pro], whole-image, globally
                        normalized per image. Good for overall plausibility,
                        but Depth Pro's raw output is metric depth dominated
                        by the far background's range, so this view compresses
                        near-field detail (where the target classes live) into
                        a narrow band - don't read "less detail" from this one.
  detail_comparison.png  Cropped to the near-field/object region (bottom 65% -
                        skips the open-water backdrop at the top of frame) and
                        LOCALLY normalized, plus a Sobel edge-magnitude map per
                        model. This is the fair way to compare fine detail /
                        boundary sharpness, since it isn't diluted by the far
                        background's dynamic range.
  fine_tuned_comparison.png  Raw vs guided-filter-refined depth, both models.
                        Not fine-tuning (no training, no data problem) - a
                        classic edge-aware filter (He/Sun/Tang 2010) that uses
                        the RGB image as a guide to snap the depth map's
                        transitions to the guide's edges. Pure numpy/scipy,
                        no model weights.

Both models are general (non-underwater) by design: Stage 1 renders clean,
colour-neutral water on purpose (see prompts.py's SCENE_WATER_PHRASE comment),
so the images these models see have none of the attenuation/backscatter/
colour-cast domain shift that underwater-specific depth models exist to
correct for - see AI_Pipeline_Test_Plan.md's Stage 3 section for the full
reasoning on why underwater-fine-tuned candidates were not tried here.

    python src/depth_compare.py
    python src/depth_compare.py --images outputs/1-pilot/klein/pilot_001_klein.png ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from depth_utils import guided_filter, local_norm, sobel_magnitude

DEFAULT_IMAGES = [
    "outputs/1-pilot/klein/pilot_001_klein.png",  # single class, sparse, close-up
    "outputs/1-pilot/klein/pilot_007_klein.png",  # triple class, moderate, mid
    "outputs/1-pilot/klein/pilot_012_klein.png",  # single class, dense, wide (the starfish-overshoot row)
]

MODELS = {
    "da_v2_large": {
        "repo": "depth-anything/Depth-Anything-V2-Large-hf",
        "metric": False,
    },
    "depth_pro": {
        "repo": "apple/DepthPro-hf",
        "metric": True,
    },
}


def run_model(repo: str, image_paths: list[Path]) -> tuple[list, list[float]]:
    import torch
    from transformers import pipeline
    from PIL import Image

    device = 0 if torch.cuda.is_available() else -1
    print(f"loading {repo} (device={'cuda' if device == 0 else 'cpu'})...")
    pipe = pipeline(task="depth-estimation", model=repo, device=device, torch_dtype=torch.float32)

    depth_maps = []
    durations = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        result = pipe(image)
        elapsed = time.perf_counter() - t0
        depth_maps.append(result["predicted_depth"])
        durations.append(elapsed)
        print(f"  {path.name}: {elapsed:.2f}s")

    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return depth_maps, durations


def save_comparison(image_paths: list[Path], results: dict[str, list], out_path: Path) -> None:
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    model_keys = list(results.keys())
    n_rows = len(image_paths)
    n_cols = 1 + len(model_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=120)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.patch.set_facecolor("#fcfcfb")

    for row, path in enumerate(image_paths):
        axes[row, 0].imshow(Image.open(path).convert("RGB"))
        axes[row, 0].set_title(path.stem, fontsize=9)
        axes[row, 0].axis("off")

        for col, key in enumerate(model_keys, start=1):
            depth = results[key][row]
            depth_np = depth.squeeze().to(torch.float32).cpu().numpy()
            # Per-image normalization for display only - relative ordering
            # within the image is what Stage 3a actually needs, so this is
            # the right thing to eyeball, not absolute scale across models.
            axes[row, col].imshow(local_norm(depth_np), cmap="inferno")
            axes[row, col].set_title(key, fontsize=9)
            axes[row, col].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\ncomparison grid -> {out_path}")


def crop_bottom(arr, frac: float = 0.65):
    h = arr.shape[0]
    return arr[int(h * (1 - frac)):, ...]


def save_detail_comparison(image_paths: list[Path], results: dict[str, list], out_path: Path) -> None:
    """Fair fine-detail view: crop to the near-field region, normalize
    locally (not against the whole image's range), and show Sobel edge
    magnitude alongside the depth map itself.
    """
    import numpy as np
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    model_keys = list(results.keys())
    n_rows = len(image_paths)
    n_cols = 1 + 2 * len(model_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.6 * n_rows), dpi=120)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.patch.set_facecolor("#fcfcfb")

    for row, path in enumerate(image_paths):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        axes[row, 0].imshow(crop_bottom(rgb))
        axes[row, 0].set_title(f"{path.stem}\n(near-field crop)", fontsize=8)
        axes[row, 0].axis("off")

        col = 1
        for key in model_keys:
            depth_np = results[key][row].squeeze().to(torch.float32).cpu().numpy()
            crop = crop_bottom(depth_np)
            axes[row, col].imshow(local_norm(crop), cmap="inferno")
            axes[row, col].set_title(f"{key} (local norm)", fontsize=8)
            axes[row, col].axis("off")
            col += 1

            axes[row, col].imshow(sobel_magnitude(crop), cmap="gray")
            axes[row, col].set_title(f"{key} edges", fontsize=8)
            axes[row, col].axis("off")
            col += 1

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"detail comparison -> {out_path}")


def save_finetuned_comparison(image_paths: list[Path], results: dict[str, list], out_path: Path) -> None:
    """Raw vs guided-filter-refined depth (RGB luminance as the edge guide),
    both models, near-field crop. No training - see guided_filter()'s docstring.
    """
    import numpy as np
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    model_keys = list(results.keys())
    n_rows = len(image_paths)
    n_cols = 1 + 2 * len(model_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.6 * n_rows), dpi=120)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.patch.set_facecolor("#fcfcfb")

    for row, path in enumerate(image_paths):
        rgb = np.asarray(Image.open(path).convert("L")).astype(np.float64) / 255.0
        rgb_display = np.asarray(Image.open(path).convert("RGB"))
        axes[row, 0].imshow(crop_bottom(rgb_display))
        axes[row, 0].set_title(f"{path.stem}\n(guide: RGB luminance)", fontsize=8)
        axes[row, 0].axis("off")

        col = 1
        for key in model_keys:
            depth_np = results[key][row].squeeze().to(torch.float32).cpu().numpy()
            if rgb.shape == depth_np.shape:
                guide = rgb
            else:
                resized = Image.fromarray((rgb * 255).astype(np.uint8)).resize(depth_np.shape[::-1])
                guide = np.asarray(resized).astype(np.float64) / 255.0

            refined = guided_filter(guide, depth_np, radius=8, eps=1e-3)

            axes[row, col].imshow(local_norm(crop_bottom(depth_np)), cmap="inferno")
            axes[row, col].set_title(f"{key} raw", fontsize=8)
            axes[row, col].axis("off")
            col += 1

            axes[row, col].imshow(local_norm(crop_bottom(refined)), cmap="inferno")
            axes[row, col].set_title(f"{key} guided-filtered", fontsize=8)
            axes[row, col].axis("off")
            col += 1

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"guided-filter comparison -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_IMAGES])
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/depth_compare"))
    args = ap.parse_args()

    missing = [p for p in args.images if not p.exists()]
    if missing:
        raise SystemExit(f"missing image(s): {missing}")

    results = {}
    timings = {}
    for key, cfg in MODELS.items():
        depth_maps, durations = run_model(cfg["repo"], args.images)
        results[key] = depth_maps
        timings[key] = durations

    save_comparison(args.images, results, args.out_dir / "comparison.png")
    save_detail_comparison(args.images, results, args.out_dir / "detail_comparison.png")
    save_finetuned_comparison(args.images, results, args.out_dir / "fine_tuned_comparison.png")

    timing_report = {
        key: {"repo": MODELS[key]["repo"], "metric": MODELS[key]["metric"],
              "seconds_per_image": dict(zip([p.name for p in args.images], durations))}
        for key, durations in timings.items()
    }
    report_path = args.out_dir / "timings.json"
    report_path.write_text(json.dumps(timing_report, indent=2), encoding="utf-8")
    print(f"timings -> {report_path}")


if __name__ == "__main__":
    main()
