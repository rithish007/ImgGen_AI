"""Stage 3b - physics transform (Akkaynak & Treibitz revised underwater image
formation model, CVPR 2018) plus three camera-level effects Jerlov doesn't
cover. Deterministic, pixel-only - no model weights, no GPU. Runs entirely on
the clean Stage 1 images plus Stage 3a's range maps.

TWO PROFILES, selected by --profile, for the A/B/C dataset-comparison plan:
    placeholder (default) - dataset B. Uses jerlov.py exactly as it has always
        been - {1C,3C,5C}, the borrowed danaberman/underwater-hl ratio table,
        one uncited Kd anchor, global BETA_B_FLOOR/CEIL. Frozen: zero real-
        world input from the AT-reverse-engineering research thread. Produces
        byte-identical output to pre-profile-flag behaviour for the same seed.
    anchored - dataset C. Uses jerlov_anchored.py - {1C,3C,5C,7C}, real
        per-type beta and Kd from Solonenko & Mobley 2015 (peer-reviewed,
        primary-sourced), beta_b's magnitude tied to the chosen water_type
        (+/- ANCHOR_WIDTH_FRAC) rather than one shared global range. See
        jerlov_anchored.py's module docstring for the full source trail and
        the one known gap (5C's red channel uses a 575nm proxy, not 600nm -
        the original paper's table has a printing error there, confirmed
        against the primary source, not an extraction issue on this end).

    I_c(x,y) = J_c(x,y) * exp(-beta_c^D * z(x,y))
             + B_c^inf   * (1 - exp(-beta_c^B * z(x,y)))
               |_______direct signal_______|   |_____backscatter_____|

then, in real-camera-pipeline order (lens -> optics -> sensor):
    vignette -> motion blur -> sensor noise

Per image, samples (seeded by the image's own generation seed, for
reproducibility):
    - Jerlov coastal type from {1C, 3C, 5C}
    - vertical depth d ~ U(0, 5) m - sets the veiling light B_c^inf via
      src/jerlov.py's kd_rgb() (see that file for the flagged simplification
      this rests on - it is NOT a primary-sourced Kd(lambda) table, confirmed
      with the user before use; see AI_Pipeline_Test_Plan.md's Stage 3b
      section for the full research trail)
    - visibility_floor ~ U(VISIBILITY_FLOOR_RANGE) - the min fraction of red-
      channel direct signal retained at THIS image's own z_far. Sampled per
      image, not fixed, specifically so the DR set spans mild-haze to
      notably-dark-and-murky rather than clustering at one "safe" look (the
      first recalibration fixed "too extreme" by capping tightly per-image,
      which as a side effect made every image similarly mild - see
      AI_Pipeline_Test_Plan.md). Low draws -> more haze allowed; high draws
      -> stays close to the clean image.
    - beam attenuation beta_b ~ U(BETA_B_FLOOR, cap) m^-1, cap = min(
      BETA_B_CEIL, z_far_beta_b_cap(z_far, water_type, visibility_floor)) -
      same z_far-coupling as before, now driven by the per-image floor above
      instead of a single global constant.
    - backscatter baseline b_ref ~ U(0.5, 1.0) - the veiling light's
      unattenuated (surface, d=0) achromatic intensity in the same [0,1]
      units as the scene image; all of B_c^inf's colour comes from the
      per-channel Kd exponential, not from this baseline being tinted.
      Lower bound dropped from 0.7 to 0.5 for the same darker-tail reason as
      visibility_floor above.
    - sensor noise: signal-dependent (shot + read) Gaussian noise, the
      standard heteroscedastic approximation of real camera noise (e.g.
      Brooks et al. "Unprocessing Images for Learned Raw Denoising", CVPR
      2019) - noise = N(0, sigma_read^2 + sigma_shot^2 * pixel_value), scaled
      up with d (deeper -> the ROV's camera would gain up -> more visible
      noise). sigma_read ~ U(0.002, 0.01), sigma_shot ~ U(0.01, 0.05) -
      plausible small-sensor orders of magnitude, not fitted to a specific
      camera.
    - motion blur: linear kernel, length ~ U(0, 4) px at 1024 resolution
      (0 = no blur for some images - not every frame has camera shake),
      angle ~ U(0, 360) deg. Mild by construction, matching the "slight
      motion softness" the Stage 1 prompts already ask for (prompts.py's
      CAMERA_MOTION) but can't reliably guarantee as a pixel effect.
    - vignette: radial darkening from a slightly off-centre point (simulating
      an ROV-mounted light, not perfectly centred on the lens axis).
      strength ~ U(0, 0.35) (0 = no vignette for some images), centre offset
      ~ U(-0.15, 0.15) of half-width/height in x and y independently.

beta_c^D and beta_c^B are treated as EQUAL (both = jerlov.beta_rgb()'s
output) - the revised model distinguishes them physically, but this pipeline
only has one attenuation-ratio table, not two. Flagged, not silently assumed;
see AI_Pipeline_Test_Plan.md.

    python src/domain_randomize.py --images-dir outputs/1-pilot/klein --range-dir outputs/1-pilot/range

Outputs:
    outputs/1-pilot/dr/<image_id>_dr.png   DR'd copy (label file unchanged -
                                            this is a pixel-only transform, so
                                            outputs/1-pilot/labels/sam3/<stem>.txt
                                            is still valid for it)
    configs/<image_id>_dr.json             water type, d, z_near/z_far, beta^D,
                                            beta^B, B^inf, noise/blur/vignette
                                            params, seed
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import jerlov_anchored
from jerlov import COASTAL_TYPES, beta_rgb, kd_rgb

BETA_B_FLOOR = 0.05  # m^-1 - placeholder profile only
BETA_B_CEIL = 0.25  # m^-1 - placeholder profile only, global cap even for the closest shots
MIN_BETA_B_ABSOLUTE = 0.01  # m^-1 - anchored profile only, true degenerate-case floor (see sample_params)
VISIBILITY_FLOOR_RANGE = (0.08, 0.45)  # sampled per image - see module docstring
B_REF_RANGE = (0.5, 1.0)
D_RANGE = (0.0, 5.0)  # m, per the plan doc's pilot value

PROFILE_WATER_TYPES = {
    "placeholder": sorted(COASTAL_TYPES),
    "anchored": sorted(jerlov_anchored.COASTAL_TYPES_ANCHORED),
}

SIGMA_READ_RANGE = (0.002, 0.01)
SIGMA_SHOT_RANGE = (0.01, 0.05)
NOISE_DEPTH_GAIN = 0.4  # noise scale at d=5m relative to d=0m: (1 + this)

MOTION_BLUR_LENGTH_RANGE = (0.0, 4.0)  # px at 1024 resolution
VIGNETTE_STRENGTH_RANGE = (0.0, 0.35)
VIGNETTE_CENTER_OFFSET_RANGE = (-0.15, 0.15)  # fraction of half-width/height


def z_far_beta_b_cap(z_far: float, beta_br: float, visibility_floor: float) -> float:
    """Largest beta_b that keeps the red channel (fastest-attenuating, via
    beta_br) at or above visibility_floor at this image's own z_far.

    exp(-beta_r * z_far) >= floor, beta_r = beta_b / beta_br
    => beta_b <= -ln(floor) * beta_br / z_far

    beta_br is passed in rather than looked up here, so this is shared between
    both profiles (placeholder's jerlov.COASTAL_TYPES and anchored's
    jerlov_anchored.COASTAL_TYPES_ANCHORED have different beta_br values).
    """
    return -math.log(visibility_floor) * beta_br / z_far


def sample_params(seed: int, z_far: float, profile: str = "placeholder") -> dict:
    rng = random.Random(seed)
    water_type = rng.choice(PROFILE_WATER_TYPES[profile])
    visibility_floor = rng.uniform(*VISIBILITY_FLOOR_RANGE)

    if profile == "placeholder":
        beta_br = COASTAL_TYPES[water_type].beta_br
        beta_floor, beta_ceil = BETA_B_FLOOR, BETA_B_CEIL
        cap = min(beta_ceil, z_far_beta_b_cap(z_far, beta_br, visibility_floor))
        cap = max(cap, beta_floor)  # degenerate-z_far safety net (unchanged - this branch
        # is untouched by the anchored-profile fix below; BETA_B_FLOOR=0.05 never exceeds
        # the visibility-safe cap in practice, so this line is a true edge-case fallback here)
    else:
        beta_br = jerlov_anchored.COASTAL_TYPES_ANCHORED[water_type].beta_br
        anchor_floor, anchor_ceil = jerlov_anchored.beta_b_anchor_range(water_type)
        # BUG FIXED HERE (2026-08-15, after the first anchored test run produced images
        # with no visible content): the anchor floor can legitimately exceed the
        # visibility-safe ceiling (e.g. 7C + a long z_far "wide" shot) - real 7C water
        # genuinely isn't visible that far. The visibility_floor constraint is the hard
        # physical invariant and must win; the type anchor is an aspirational centre that
        # yields when the two conflict, not the other way round. Previously `cap =
        # max(cap, beta_floor)` forced beta_b PAST the visibility-safe ceiling instead -
        # confirmed on the 50-image test set: 35/50 draws collapsed to beta_b == the
        # anchor floor exactly, 27/50 ended with <50% of visibility_floor's intended red
        # signal at z_far. Fix: shrink the floor down to meet the cap instead of pushing
        # the cap up past it.
        cap = max(min(anchor_ceil, z_far_beta_b_cap(z_far, beta_br, visibility_floor)), MIN_BETA_B_ABSOLUTE)
        beta_floor = max(min(anchor_floor, cap), MIN_BETA_B_ABSOLUTE)

    return {
        "profile": profile,
        "water_type": water_type,
        "visibility_floor": visibility_floor,
        "d": rng.uniform(*D_RANGE),
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


def transform_image(j_img, z, params: dict):
    """Apply the revised Akkaynak-Treibitz formula. j_img: (H,W,3) float [0,1].
    z: (H,W) metres. Returns (dr_image, resolved_params) where resolved_params
    records every derived per-channel coefficient for the config JSON.
    """
    import numpy as np

    if params["profile"] == "placeholder":
        beta_r, beta_g, beta_b_ch = beta_rgb(params["water_type"], params["beta_b"])
        kd_r, kd_g, kd_b = kd_rgb(params["water_type"])
    else:
        beta_r, beta_g, beta_b_ch = jerlov_anchored.beta_rgb(params["water_type"], params["beta_b"])
        kd_r, kd_g, kd_b = jerlov_anchored.kd_rgb(params["water_type"])
    d = params["d"]
    b_ref = params["b_ref"]

    beta = np.array([beta_r, beta_g, beta_b_ch])  # beta^D == beta^B, see module docstring
    kd = np.array([kd_r, kd_g, kd_b])
    b_inf = b_ref * np.exp(-kd * d)  # (3,)

    z3 = z[..., None]  # (H,W,1) to broadcast against (H,W,3)
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
    """Multiplicative radial darkening from an off-centre point - simulates
    an ROV-mounted light rather than a perfectly lens-centred one.
    """
    import numpy as np

    h, w = img.shape[:2]
    cy = h / 2 * (1 + center_offset[1])
    cx = w / 2 * (1 + center_offset[0])
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    gain = 1.0 - strength * np.clip(r, 0, 1.5) ** 2
    return img * gain[..., None]


def apply_motion_blur(img, length: float, angle_deg: float):
    """Mild linear motion-blur kernel. length in pixels; length < 0.5 is a
    no-op (avoids building a degenerate 1x1 "kernel" that does nothing but
    costs a convolution call).
    """
    import numpy as np
    import cv2

    if length < 0.5:
        return img

    k = max(3, int(round(length)) | 1)  # odd kernel size
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
    """Signal-dependent (shot + read) Gaussian noise - see module docstring
    for the Brooks et al. reference this approximation follows.
    """
    import numpy as np

    np_rng = np.random.default_rng(seed)
    sigma = np.sqrt(sigma_read**2 + sigma_shot**2 * np.clip(img, 0, None)) * depth_gain
    return img + np_rng.normal(0.0, 1.0, size=img.shape) * sigma


def apply_camera_effects(dr, params: dict, seed: int):
    import numpy as np

    depth_gain = 1.0 + NOISE_DEPTH_GAIN * (params["d"] / D_RANGE[1])

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
        "--profile", choices=["placeholder", "anchored"], default="placeholder",
        help="placeholder = dataset B (frozen, current behaviour, default); "
             "anchored = dataset C (Solonenko & Mobley 2015-derived, see jerlov_anchored.py)",
    )
    args = ap.parse_args()

    suffix = "_dr" if args.profile == "placeholder" else "_anchored_dr"
    range_dir = args.range_dir or args.images_dir.parent / "range"
    out_dir = args.out_dir or args.images_dir.parent / ("dr" if args.profile == "placeholder" else "dr_anchored")
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
            raise SystemExit(f"no range map for {path} (expected {z_path}) - run src/range_estimate.py first")
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
