"""Shared depth-map helpers used by both depth_compare.py (Stage 3a candidate evaluation) and range_estimate.py (Stage 3a itself)."""

from __future__ import annotations

import numpy as np


def local_norm(arr: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    d_min, d_max = np.percentile(arr, [lo, hi])
    return np.clip((arr - d_min) / max(d_max - d_min, 1e-6), 0, 1)


def sobel_magnitude(arr: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    gx = ndimage.sobel(arr, axis=1)
    gy = ndimage.sobel(arr, axis=0)
    mag = (gx**2 + gy**2) ** 0.5
    return local_norm(mag, lo=0, hi=99)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
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
