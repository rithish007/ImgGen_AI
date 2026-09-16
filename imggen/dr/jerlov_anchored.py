"""Real-data counterpart to jerlov.py, for domain_randomize.py's --profile=anchored path only."""

from __future__ import annotations

from dataclasses import dataclass

_C_BLUE_GREEN_RED: dict[str, tuple[float, float, float]] = {
    "1C": (0.077 + 0.469, 0.068 + 0.395, 0.236 + 0.314),
    "3C": (0.105 + 1.36, 0.078 + 1.15, 0.239 + 0.916),
    "5C": (0.204 + 1.71, 0.127 + 1.44, 0.119 + 1.23),
    "7C": (0.388 + 3.01, 0.233 + 2.54, 0.301 + 2.03),
}

_KD_BLUE_GREEN_RED: dict[str, tuple[float, float, float]] = {
    "1C": (0.134, 0.122, 0.288),
    "3C": (0.223, 0.198, 0.342),
    "5C": (0.400, 0.315, 0.357),
    "7C": (0.693, 0.494, 0.478),
}

ANCHOR_WIDTH_FRAC = 0.25


@dataclass(frozen=True)
class AnchoredWaterIOP:

    name: str
    beta_bg: float
    beta_br: float
    beta_blue: float
    kd_bg: float
    kd_br: float
    kd_green: float


def _build(name: str) -> AnchoredWaterIOP:
    c_b, c_g, c_r = _C_BLUE_GREEN_RED[name]
    kd_b, kd_g, kd_r = _KD_BLUE_GREEN_RED[name]
    return AnchoredWaterIOP(
        name=name,
        beta_bg=c_b / c_g,
        beta_br=c_b / c_r,
        beta_blue=c_b,
        kd_bg=kd_b / kd_g,
        kd_br=kd_b / kd_r,
        kd_green=kd_g,
    )


COASTAL_TYPES_ANCHORED: dict[str, AnchoredWaterIOP] = {
    name: _build(name) for name in ("1C", "3C", "5C", "7C")
}


def beta_rgb(water_type: str, beta_b: float) -> tuple[float, float, float]:
    t = COASTAL_TYPES_ANCHORED[water_type]
    beta_g = beta_b / t.beta_bg
    beta_r = beta_b / t.beta_br
    return beta_r, beta_g, beta_b


def kd_rgb(water_type: str) -> tuple[float, float, float]:
    t = COASTAL_TYPES_ANCHORED[water_type]
    kd_blue = t.kd_green * t.kd_bg
    kd_red = kd_blue / t.kd_br
    return kd_red, t.kd_green, kd_blue


def beta_b_anchor_range(water_type: str) -> tuple[float, float]:
    center = COASTAL_TYPES_ANCHORED[water_type].beta_blue
    return center * (1 - ANCHOR_WIDTH_FRAC), center * (1 + ANCHOR_WIDTH_FRAC)
