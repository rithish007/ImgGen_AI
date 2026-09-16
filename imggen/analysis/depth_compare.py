"""Stage 3a candidate check: DA V2 Large vs Apple Depth Pro, head to head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from imggen.analysis.depth_utils import guided_filter, local_norm, sobel_magnitude

DEFAULT_IMAGES = [
    "outputs/1-pilot/klein/pilot_001_klein.png",
    "outputs/1-pilot/klein/pilot_007_klein.png",
    "outputs/1-pilot/klein/pilot_012_klein.png",
]

MODELS = {
    "da_v2_large": {
        "repo": "depth-anything/Depth-Anything-V2-Large-hf",
        "metric": False,
        "kind": "transformers",
    },
    "depth_pro": {
        "repo": "apple/DepthPro-hf",
        "metric": True,
        "kind": "transformers",
    },
    "yolo26n_depth": {
        "repo": "yolo26n-depth.pt",
        "metric": True,
        "kind": "ultralytics",
    },
    "yolo26x_depth": {
        "repo": "yolo26x-depth.pt",
        "metric": True,
        "kind": "ultralytics",
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


def run_yolo26_depth(weights: str, image_paths: list[Path]) -> tuple[list, list[float]]:
    import torch
    from ultralytics import YOLO

    print(f"loading {weights} (ultralytics)...")
    model = YOLO(weights)

    depth_maps = []
    durations = []
    for path in image_paths:
        t0 = time.perf_counter()
        results = model(str(path), verbose=False)
        elapsed = time.perf_counter() - t0
        depth_maps.append(results[0].depth.data)
        durations.append(elapsed)
        print(f"  {path.name}: {elapsed:.2f}s")

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
        if cfg.get("kind") == "ultralytics":
            depth_maps, durations = run_yolo26_depth(cfg["repo"], args.images)
        else:
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
