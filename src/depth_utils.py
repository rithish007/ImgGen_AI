"""Shared depth-map helpers used by both depth_compare.py (Stage 3a candidate
evaluation) and range_estimate.py (Stage 3a itself). Pure numpy/scipy - no
model weights, no training.
"""

from __future__ import annotations

import numpy as np


def local_norm(arr: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Percentile-clip and rescale to [0, 1]. `lo`/`hi` are percentiles, not values."""
    d_min, d_max = np.percentile(arr, [lo, hi])
    return np.clip((arr - d_min) / max(d_max - d_min, 1e-6), 0, 1)


def sobel_magnitude(arr: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    gx = ndimage.sobel(arr, axis=1)
    gy = ndimage.sobel(arr, axis=0)
    mag = (gx**2 + gy**2) ** 0.5
    return local_norm(mag, lo=0, hi=99)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware filter (He, Sun & Tang, ECCV 2010): smooths `src` while
    snapping its transitions to `guide`'s edges, using local linear
    regression in windows of the given radius. Pure numpy/scipy - no
    training, no learned weights.

    Chosen over fine-tuning a depth model for sharper boundaries: no ground
    truth depth exists for images from a 2D diffusion model to fine-tune
    against. See AI_Pipeline_Test_Plan.md's Stage 3a section for the full
    comparison this was picked from (vs DA-V2 raw, vs Depth Pro).
    """
    from scipy.ndimage import uniform_filter

    def box(x):
        return uniform_filter(x, size=2 * radius + 1, mode="reflect")

    guide = guide.astype(np.float64)
    src = src.astype(np.float64)

    mean_g = box(guide)
    mean_s = box(src)
    corr_gg = box(guide * guide)
    corr_gs = box(guide * src)
    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return box(a) * guide + box(b)
