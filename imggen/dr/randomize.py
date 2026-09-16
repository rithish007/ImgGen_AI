"""Stage 3b - physics transform (Akkaynak & Treibitz revised underwater image formation model, CVPR 2018) plus three camera-level effects Jerlov doesn't cover."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from imggen.dr import jerlov_anchored
from imggen.dr.jerlov import COASTAL_TYPES, beta_rgb, kd_rgb

BETA_B_FLOOR = 0.05
BETA_B_CEIL = 0.25
MIN_BETA_B_ABSOLUTE = 0.01
VISIBILITY_FLOOR_RANGE = (0.08, 0.45)
B_REF_RANGE = (0.5, 1.0)
D_RANGE = (0.0, 5.0)
D_RANGE_ANCHORED = (2.0, 7.0)

PROFILE_WATER_TYPES = {
    "placeholder": sorted(COASTAL_TYPES),
    "anchored": sorted(jerlov_anchored.COASTAL_TYPES_ANCHORED),
    "duo_calibrated": ["1C"],
}

DUO_CALIBRATED_WATER_TYPE = "1C"
DUO_VEILING_LIGHT_RGB = (0.417, 0.800, 0.535)
DUO_TARGET_JITTER_FRAC = 0.15

SIGMA_READ_RANGE = (0.002, 0.01)
SIGMA_SHOT_RANGE = (0.01, 0.05)
NOISE_DEPTH_GAIN = 0.4

MOTION_BLUR_LENGTH_RANGE = (0.0, 4.0)
VIGNETTE_STRENGTH_RANGE = (0.0, 0.35)
VIGNETTE_CENTER_OFFSET_RANGE = (-0.15, 0.15)


def z_far_beta_b_cap(z_far: float, beta_br: float, visibility_floor: float) -> float:
    return -math.log(visibility_floor) * beta_br / z_far


def sample_params(seed: int, z_far: float, profile: str = "placeholder") -> dict:
    rng = random.Random(seed)
    water_type = rng.choice(PROFILE_WATER_TYPES[profile])
    visibility_floor = rng.uniform(*VISIBILITY_FLOOR_RANGE)

    if profile == "placeholder":
        beta_br = COASTAL_TYPES[water_type].beta_br
        beta_floor, beta_ceil = BETA_B_FLOOR, BETA_B_CEIL
        cap = min(beta_ceil, z_far_beta_b_cap(z_far, beta_br, visibility_floor))
        cap = max(cap, beta_floor)
    else:
        beta_br = jerlov_anchored.COASTAL_TYPES_ANCHORED[water_type].beta_br
        anchor_floor, anchor_ceil = jerlov_anchored.beta_b_anchor_range(water_type)
        cap = max(min(anchor_ceil, z_far_beta_b_cap(z_far, beta_br, visibility_floor)), MIN_BETA_B_ABSOLUTE)
        beta_floor = max(min(anchor_floor, cap), MIN_BETA_B_ABSOLUTE)

    d_range = D_RANGE if profile == "placeholder" else D_RANGE_ANCHORED
    params = {
        "profile": profile,
        "water_type": water_type,
        "visibility_floor": visibility_floor,
        "d": rng.uniform(*d_range),
        "beta_b": rng.uniform(beta_floor, cap),
        "beta_b_floor_used": beta_floor,
        "beta_b_cap_used": cap,
        "b_ref": rng.uniform(*B_REF_RANGE),
        "sigma_read": rng.uniform(*SIGMA_READ_RANGE),
        "sigma_shot": rng.uniform(*SIGMA_SHOT_RANGE),
        "motion_blur_length": rng.uniform(*MOTION_BLUR_LENGTH_RANGE),
        "motion_blur_angle": rng.uniform(0, 360),
        "vignette_strength": rng.uniform(*VIGNETTE_STRENGTH_RANGE),
        "vignette_center_offset": (
            rng.uniform(*VIGNETTE_CENTER_OFFSET_RANGE),
            rng.uniform(*VIGNETTE_CENTER_OFFSET_RANGE),
        ),
    }
    if profile == "duo_calibrated":
        params["duo_target_jitter"] = tuple(
            rng.uniform(1 - DUO_TARGET_JITTER_FRAC, 1 + DUO_TARGET_JITTER_FRAC) for _ in range(3)
        )
    return params


def transform_image(j_img, z, params: dict):
    import numpy as np

    if params["profile"] == "placeholder":
        beta_r, beta_g, beta_b_ch = beta_rgb(params["water_type"], params["beta_b"])
        kd_r, kd_g, kd_b = kd_rgb(params["water_type"])
    else:
        beta_r, beta_g, beta_b_ch = jerlov_anchored.beta_rgb(params["water_type"], params["beta_b"])
        kd_r, kd_g, kd_b = jerlov_anchored.kd_rgb(params["water_type"])
    d = params["d"]
    b_ref = params["b_ref"]

    beta = np.array([beta_r, beta_g, beta_b_ch])
    kd = np.array([kd_r, kd_g, kd_b])
    if params["profile"] == "duo_calibrated":
        jitter = np.array(params["duo_target_jitter"])
        b_inf = np.array(DUO_VEILING_LIGHT_RGB) * jitter
    else:
        b_inf = b_ref * np.exp(-kd * d)

    z3 = z[..., None]
    attenuation = np.exp(-beta[None, None, :] * z3)
    dr = j_img * attenuation + b_inf[None, None, :] * (1 - attenuation)
    dr = np.clip(dr, 0.0, 1.0)

    resolved = {
        "profile": params["profile"],
        "water_type": params["water_type"],
        "visibility_floor": params["visibility_floor"],
        "vertical_depth_d_m": d,
        "beta_b_sampled": params["beta_b"],
        "beta_b_floor_used": params["beta_b_floor_used"],
        "beta_b_cap_used": params["beta_b_cap_used"],
        "beta_D_rgb": [beta_r, beta_g, beta_b_ch],
        "beta_B_rgb": [beta_r, beta_g, beta_b_ch],
        "kd_rgb": [kd_r, kd_g, kd_b],
        "b_ref": b_ref,
        "B_inf_rgb": b_inf.tolist(),
    }
    return dr, resolved


def apply_vignette(img, strength: float, center_offset: tuple[float, float]):
    import numpy as np

    h, w = img.shape[:2]
    cy = h / 2 * (1 + center_offset[1])
    cx = w / 2 * (1 + center_offset[0])
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    gain = 1.0 - strength * np.clip(r, 0, 1.5) ** 2
    return img * gain[..., None]


def apply_motion_blur(img, length: float, angle_deg: float):
    import numpy as np
    import cv2

    if length < 0.5:
        return img

    k = max(3, int(round(length)) | 1)
    kernel = np.zeros((k, k), dtype=np.float64)
    kernel[k // 2, :] = 1.0
    angle = angle_deg
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        return img
    kernel /= kernel_sum
    return cv2.filter2D(img.astype(np.float64), -1, kernel)


def apply_sensor_noise(img, sigma_read: float, sigma_shot: float, depth_gain: float, seed: int):
    import numpy as np

    np_rng = np.random.default_rng(seed)
    sigma = np.sqrt(sigma_read**2 + sigma_shot**2 * np.clip(img, 0, None)) * depth_gain
    return img + np_rng.normal(0.0, 1.0, size=img.shape) * sigma


def apply_camera_effects(dr, params: dict, seed: int):
    import numpy as np

    d_max = D_RANGE[1] if params["profile"] == "placeholder" else D_RANGE_ANCHORED[1]
    depth_gain = 1.0 + NOISE_DEPTH_GAIN * (params["d"] / d_max)

    out = apply_vignette(dr, params["vignette_strength"], params["vignette_center_offset"])
    out = apply_motion_blur(out, params["motion_blur_length"], params["motion_blur_angle"])
    out = apply_sensor_noise(out, params["sigma_read"], params["sigma_shot"], depth_gain, seed)
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=Path("outputs/1-pilot/klein"))
    ap.add_argument("--range-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument(
        "--profile", choices=["placeholder", "anchored", "duo_calibrated"], default="placeholder",
        help="placeholder = dataset B (frozen, current behaviour, default); "
             "anchored = dataset C (Solonenko & Mobley 2015-derived, see jerlov_anchored.py); "
             "duo_calibrated = dataset D (1C beta + DUO-measured veiling light, see module docstring)",
    )
    args = ap.parse_args()

    suffix = {"placeholder": "_dr", "anchored": "_anchored_dr", "duo_calibrated": "_duo_dr"}[args.profile]
    out_dir_name = {"placeholder": "dr", "anchored": "dr_anchored", "duo_calibrated": "dr_duo_calibrated"}[args.profile]
    range_dir = args.range_dir or args.images_dir.parent / "range"
    out_dir = args.out_dir or args.images_dir.parent / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image

    image_paths = sorted(args.images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"no .png files found in {args.images_dir}")

    for i, path in enumerate(image_paths, start=1):
        stem = path.stem
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        seed = sidecar["seed"]

        z_path = range_dir / f"{stem}_range.npy"
        z_json_path = range_dir / f"{stem}_range.json"
        if not z_path.exists():
            raise SystemExit(f"no range map for {path} (expected {z_path}) - run: python -m imggen.analysis.range_estimate")
        z = np.load(z_path)
        z_meta = json.loads(z_json_path.read_text(encoding="utf-8"))

        j_img = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0

        params = sample_params(seed, z_meta["z_far"], profile=args.profile)
        dr, resolved = transform_image(j_img, z, params)
        dr = apply_camera_effects(dr, params, seed)

        dr_path = out_dir / f"{stem}{suffix}.png"
        Image.fromarray((dr * 255).astype(np.uint8)).save(dr_path)

        camera_effects = {
            "sigma_read": params["sigma_read"],
            "sigma_shot": params["sigma_shot"],
            "motion_blur_length_px": params["motion_blur_length"],
            "motion_blur_angle_deg": params["motion_blur_angle"],
            "vignette_strength": params["vignette_strength"],
            "vignette_center_offset": params["vignette_center_offset"],
        }

        config = {
            "image_id": sidecar["image_id"],
            "source_image": str(path),
            "range_map": str(z_path),
            "seed": seed,
            "camera_effects": camera_effects,
            "z_near": z_meta["z_near"],
            "z_far": z_meta["z_far"],
            **resolved,
        }
        (args.config_dir / f"{stem}{suffix}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        print(f"[{i:>2}/{len(image_paths)}] {stem}  type={params['water_type']}  d={params['d']:.2f}m  -> {dr_path}")

    print(f"\ndone: {len(image_paths)} DR'd images ({args.profile}) -> {out_dir}, configs -> {args.config_dir}")


if __name__ == "__main__":
    main()
