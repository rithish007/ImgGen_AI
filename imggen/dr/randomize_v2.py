"""
Stage 3b v2 - DUO-calibrated underwater image formation + target-domain
scattering/detail degradation.

IMPORTANT:
    This is a NEW experimental calibration pipeline.
    Existing domain_randomize.py and all three original profiles remain untouched.

DESIGN:
    Starts from the existing "duo_calibrated" profile:
        - beta pinned to Jerlov 1C
        - DUO-measured veiling light
        - target-domain colour calibration

    Adds a second, explicitly separated camera/water-observation stage intended
    to close the remaining DUO gap in:
        - local contrast
        - high-frequency detail
        - underwater scattering softness

NEW PROFILE:
    duo_calibrated_scatter

The original colour/attenuation model is preserved.
Only the post-formation observation model is changed.

Pipeline:
    clean synthetic image
        -> Akkaynak/Treibitz colour + attenuation
        -> controlled scattering blur
        -> local contrast reduction
        -> sensor noise
        -> optional mild vignette

No neural network, no GPU, deterministic per-image seed.

The calibration target is DUO's measured image statistics.

Target values:
    ratio_rg            ~= 0.438
    ratio_bg            ~= 0.580
    dark_channel_mean   ~= 0.192
    luminance_std       ~= 0.093

The existing duo_calibrated profile already improves the colour statistics.
This profile specifically attacks the remaining sharpness/contrast gap.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from imggen.dr import jerlov_anchored


# ============================================================================
# EXISTING DUO CALIBRATION VALUES — FROZEN
# ============================================================================

DUO_CALIBRATED_WATER_TYPE = "1C"

DUO_VEILING_LIGHT_RGB = (0.417, 0.800, 0.535)

DUO_TARGET_JITTER_FRAC = 0.15

VISIBILITY_FLOOR_RANGE = (0.08, 0.45)

B_REF_RANGE = (0.5, 1.0)

D_RANGE = (2.0, 7.0)

MIN_BETA_B_ABSOLUTE = 0.01

ANCHOR_WIDTH_FRAC = 0.25


# ============================================================================
# NEW TARGET-DOMAIN IMAGE-DEGRADATION PARAMETERS
# ============================================================================

# Gaussian scattering blur at 1024 px.
#
# Kept deliberately mild: the purpose is not to make images obviously blurry,
# but to reduce the synthetic image's excessive high-frequency detail.
SCATTER_SIGMA_RANGE = (0.35, 1.25)

# Additional mild directional softness.
# 0 = no motion component.
MOTION_BLUR_LENGTH_RANGE = (0.0, 3.0)

# Contrast compression around image mean.
#
# 1.0 = unchanged
# <1.0 = reduced local/global contrast
CONTRAST_SCALE_RANGE = (0.72, 0.92)

# Slight brightness compression after contrast reduction.
BRIGHTNESS_SCALE_RANGE = (0.96, 1.02)

# Sensor noise.
SIGMA_READ_RANGE = (0.002, 0.008)
SIGMA_SHOT_RANGE = (0.008, 0.035)

NOISE_DEPTH_GAIN = 0.4

# Keep vignette secondary.
VIGNETTE_STRENGTH_RANGE = (0.0, 0.20)
VIGNETTE_CENTER_OFFSET_RANGE = (-0.12, 0.12)


# ============================================================================
# TARGET CALIBRATION
# ============================================================================

def z_far_beta_b_cap(
    z_far: float,
    beta_br: float,
    visibility_floor: float,
) -> float:
    if z_far <= 1e-6:
        return 0.25

    return -math.log(visibility_floor) * beta_br / z_far


def sample_params(seed: int, z_far: float) -> dict:
    rng = random.Random(seed)

    water_type = DUO_CALIBRATED_WATER_TYPE
    visibility_floor = rng.uniform(*VISIBILITY_FLOOR_RANGE)

    beta_br = jerlov_anchored.COASTAL_TYPES_ANCHORED[
        water_type
    ].beta_br

    anchor_floor, anchor_ceil = jerlov_anchored.beta_b_anchor_range(
        water_type
    )

    cap = max(
        min(
            anchor_ceil,
            z_far_beta_b_cap(
                z_far,
                beta_br,
                visibility_floor,
            ),
        ),
        MIN_BETA_B_ABSOLUTE,
    )

    beta_floor = max(
        min(anchor_floor, cap),
        MIN_BETA_B_ABSOLUTE,
    )

    params = {
        "profile": "duo_calibrated_scatter",
        "water_type": water_type,
        "visibility_floor": visibility_floor,
        "d": rng.uniform(*D_RANGE),
        "beta_b": rng.uniform(beta_floor, cap),
        "beta_b_floor_used": beta_floor,
        "beta_b_cap_used": cap,
        "b_ref": rng.uniform(*B_REF_RANGE),

        # Existing DUO colour calibration.
        "duo_target_jitter": tuple(
            rng.uniform(
                1.0 - DUO_TARGET_JITTER_FRAC,
                1.0 + DUO_TARGET_JITTER_FRAC,
            )
            for _ in range(3)
        ),

        # NEW observation-domain parameters.
        "scatter_sigma": rng.uniform(
            *SCATTER_SIGMA_RANGE
        ),
        "contrast_scale": rng.uniform(
            *CONTRAST_SCALE_RANGE
        ),
        "brightness_scale": rng.uniform(
            *BRIGHTNESS_SCALE_RANGE
        ),
        "motion_blur_length": rng.uniform(
            *MOTION_BLUR_LENGTH_RANGE
        ),
        "motion_blur_angle": rng.uniform(
            0.0,
            360.0,
        ),
        "sigma_read": rng.uniform(
            *SIGMA_READ_RANGE
        ),
        "sigma_shot": rng.uniform(
            *SIGMA_SHOT_RANGE
        ),
        "vignette_strength": rng.uniform(
            *VIGNETTE_STRENGTH_RANGE
        ),
        "vignette_center_offset": (
            rng.uniform(*VIGNETTE_CENTER_OFFSET_RANGE),
            rng.uniform(*VIGNETTE_CENTER_OFFSET_RANGE),
        ),
    }

    return params


# ============================================================================
# UNDERWATER FORMATION MODEL
# ============================================================================

def transform_image(
    j_img,
    z,
    params: dict,
):
    import numpy as np

    water_type = params["water_type"]

    beta_r, beta_g, beta_b = jerlov_anchored.beta_rgb(
        water_type,
        params["beta_b"],
    )

    kd_r, kd_g, kd_b = jerlov_anchored.kd_rgb(
        water_type
    )

    del kd_r, kd_g, kd_b

    beta = np.array(
        [beta_r, beta_g, beta_b],
        dtype=np.float64,
    )

    jitter = np.array(
        params["duo_target_jitter"],
        dtype=np.float64,
    )

    b_inf = np.array(
        DUO_VEILING_LIGHT_RGB,
        dtype=np.float64,
    ) * jitter

    z3 = z[..., None]

    attenuation = np.exp(
        -beta[None, None, :] * z3
    )

    dr = (
        j_img * attenuation
        + b_inf[None, None, :]
        * (1.0 - attenuation)
    )

    return (
        np.clip(dr, 0.0, 1.0),
        {
            "beta_D_rgb": [
                beta_r,
                beta_g,
                beta_b,
            ],
            "beta_B_rgb": [
                beta_r,
                beta_g,
                beta_b,
            ],
            "B_inf_rgb": b_inf.tolist(),
            "vertical_depth_d_m": params["d"],
        },
    )


# ============================================================================
# SCATTERING / DETAIL LOSS
# ============================================================================

def apply_scattering_blur(
    img,
    sigma: float,
):
    import cv2

    if sigma <= 0.05:
        return img

    k = max(
        3,
        int(math.ceil(sigma * 6)) | 1,
    )

    return cv2.GaussianBlur(
        img.astype("float32"),
        (k, k),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )


def apply_contrast_reduction(
    img,
    contrast_scale: float,
    brightness_scale: float,
):
    import numpy as np

    mean = np.mean(
        img,
        axis=(0, 1),
        keepdims=True,
    )

    out = (
        (img - mean)
        * contrast_scale
        + mean
    )

    out *= brightness_scale

    return np.clip(
        out,
        0.0,
        1.0,
    )


def apply_motion_blur(
    img,
    length: float,
    angle_deg: float,
):
    import cv2
    import numpy as np

    if length < 0.5:
        return img

    k = max(
        3,
        int(round(length)) | 1,
    )

    kernel = np.zeros(
        (k, k),
        dtype=np.float64,
    )

    kernel[k // 2, :] = 1.0

    matrix = cv2.getRotationMatrix2D(
        (k / 2 - 0.5, k / 2 - 0.5),
        angle_deg,
        1.0,
    )

    kernel = cv2.warpAffine(
        kernel,
        matrix,
        (k, k),
    )

    total = kernel.sum()

    if total <= 0:
        return img

    kernel /= total

    return cv2.filter2D(
        img.astype(np.float64),
        -1,
        kernel,
    )


def apply_sensor_noise(
    img,
    sigma_read: float,
    sigma_shot: float,
    depth_gain: float,
    seed: int,
):
    import numpy as np

    rng = np.random.default_rng(seed)

    sigma = np.sqrt(
        sigma_read ** 2
        + sigma_shot ** 2
        * np.clip(img, 0.0, None)
    )

    sigma *= depth_gain

    noise = rng.normal(
        0.0,
        1.0,
        size=img.shape,
    )

    return img + noise * sigma


def apply_vignette(
    img,
    strength: float,
    center_offset,
):
    import numpy as np

    h, w = img.shape[:2]

    cx = (
        w / 2
        * (1.0 + center_offset[0])
    )
    cy = (
        h / 2
        * (1.0 + center_offset[1])
    )

    yy, xx = np.mgrid[0:h, 0:w]

    radius = np.sqrt(
        ((xx - cx) / (w / 2)) ** 2
        + ((yy - cy) / (h / 2)) ** 2
    )

    gain = (
        1.0
        - strength
        * np.clip(radius, 0.0, 1.5) ** 2
    )

    return img * gain[..., None]


def apply_camera_effects(
    dr,
    params: dict,
    seed: int,
):
    d_max = D_RANGE[1]

    depth_gain = (
        1.0
        + NOISE_DEPTH_GAIN
        * (params["d"] / d_max)
    )

    out = apply_scattering_blur(
        dr,
        params["scatter_sigma"],
    )

    out = apply_contrast_reduction(
        out,
        params["contrast_scale"],
        params["brightness_scale"],
    )

    out = apply_motion_blur(
        out,
        params["motion_blur_length"],
        params["motion_blur_angle"],
    )

    out = apply_vignette(
        out,
        params["vignette_strength"],
        params["vignette_center_offset"],
    )

    out = apply_sensor_noise(
        out,
        params["sigma_read"],
        params["sigma_shot"],
        depth_gain,
        seed,
    )

    return np_clip(out)


def np_clip(img):
    import numpy as np
    return np.clip(img, 0.0, 1.0)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        "--images-dir",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--range-dir",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--config-dir",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    import numpy as np
    from PIL import Image

    image_paths = sorted(
        args.images_dir.glob("*.png")
    )

    if not image_paths:
        raise SystemExit(
            "No PNG images found."
        )

    for index, path in enumerate(
        image_paths,
        start=1,
    ):
        stem = path.stem

        sidecar = json.loads(
            path.with_suffix(".json")
            .read_text(encoding="utf-8")
        )

        seed = sidecar["seed"]

        range_path = (
            args.range_dir
            / f"{stem}_range.npy"
        )

        range_meta_path = (
            args.range_dir
            / f"{stem}_range.json"
        )

        if not range_path.exists():
            raise SystemExit(
                f"Missing range map: {range_path}"
            )

        z = np.load(range_path)

        z_meta = json.loads(
            range_meta_path.read_text(
                encoding="utf-8"
            )
        )

        image = (
            np.asarray(
                Image.open(path)
                .convert("RGB")
            )
            .astype(np.float64)
            / 255.0
        )

        params = sample_params(
            seed,
            z_meta["z_far"],
        )

        dr, resolved = transform_image(
            image,
            z,
            params,
        )

        dr = apply_camera_effects(
            dr,
            params,
            seed,
        )

        output_path = (
            args.out_dir
            / f"{stem}_duo_scatter_dr.png"
        )

        Image.fromarray(
            (dr * 255.0).astype(np.uint8)
        ).save(output_path)

        config = {
            "image_id": sidecar["image_id"],
            "source_image": str(path),
            "range_map": str(range_path),
            "seed": seed,
            "profile": "duo_calibrated_scatter",
            "water_type": params["water_type"],
            "visibility_floor": params["visibility_floor"],
            "beta_b": params["beta_b"],
            "vertical_depth_d_m": params["d"],
            "scatter_sigma": params["scatter_sigma"],
            "contrast_scale": params["contrast_scale"],
            "brightness_scale": params["brightness_scale"],
            "motion_blur_length_px": params["motion_blur_length"],
            "motion_blur_angle_deg": params["motion_blur_angle"],
            "sigma_read": params["sigma_read"],
            "sigma_shot": params["sigma_shot"],
            "vignette_strength": params["vignette_strength"],
            "vignette_center_offset": params["vignette_center_offset"],
            **resolved,
        }

        (
            args.config_dir
            / f"{stem}_duo_scatter_dr.json"
        ).write_text(
            json.dumps(
                config,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[{index:>4}/{len(image_paths)}] "
            f"{stem} -> {output_path}"
        )

    print(
        f"\nGenerated {len(image_paths)} "
        "DUO-calibrated + scattering images."
    )


if __name__ == "__main__":
    main()