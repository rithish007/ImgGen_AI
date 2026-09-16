"""Jerlov coastal-water attenuation ratios for the Stage 3b physics transform."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JerlovRatios:

    name: str
    beta_bg: float
    beta_br: float


COASTAL_TYPES: dict[str, JerlovRatios] = {
    "1C": JerlovRatios("1C", beta_bg=0.7937, beta_br=0.2773),
    "3C": JerlovRatios("3C", beta_bg=0.9539, beta_br=0.4051),
    "5C": JerlovRatios("5C", beta_bg=1.0930, beta_br=0.4642),
}


def beta_rgb(water_type: str, beta_b: float) -> tuple[float, float, float]:
    ratios = COASTAL_TYPES[water_type]
    beta_g = beta_b / ratios.beta_bg
    beta_r = beta_b / ratios.beta_br
    return beta_r, beta_g, beta_b


KD_GREEN_ANCHOR_1976 = 0.2763


def kd_rgb(water_type: str, kd_green: float = KD_GREEN_ANCHOR_1976) -> tuple[float, float, float]:
    ratios = COASTAL_TYPES[water_type]
    kd_blue = kd_green * ratios.beta_bg
    kd_red = kd_blue / ratios.beta_br
    return kd_red, kd_green, kd_blue
