"""Real-data counterpart to jerlov.py, for domain_randomize.py's --profile=anchored
path only. jerlov.py stays frozen as the placeholder profile's data source (dataset
B - zero real-world input, unmodified); this file is dataset C's data source (all
research from the AT-reverse-engineering thread applied).

SOURCE: Solonenko & Mobley, "Inherent optical properties of Jerlov water types,"
Applied Optics 54(17):5392-5401 (2015), doi:10.1364/AO.54.005392. Read in full
(user-supplied PDF) 2026-08-14/15. Tables 6-8 (Appendix A) give K^0_d(lambda),
a(lambda), b(lambda) for every Jerlov type at 25nm steps, 300-700nm - a real,
peer-reviewed, primary-sourced spectral IOP table, replacing BOTH simplifications
jerlov.py flags as weak (the borrowed beta-ratio shape from danaberman/underwater-hl,
and the single uncited Kd magnitude anchor).

WAVELENGTHS: 475/525/600nm (blue/green/red), matching the existing pipeline's
camera-agnostic peak-sensitivity convention (Jiang et al. WACV 2013), so results
stay comparable to jerlov.py's ratios in shape even though the underlying data
source differs.

BETA (beam attenuation c = a+b, for beta_c^D/beta_c^B - horizontal-range
attenuation): computed directly from Table 6 (1C, paired with Jerlov III),
Table 7 (3C, 5C), Table 8 (7C, 9C).

KD (diffuse attenuation, for B_c^infinity's depth dependence): K^0_d(lambda)
read directly from the same tables - real per-type values, not one anchor
applied uniformly (jerlov.py's second flagged simplification).

KNOWN GAP - 5C's red channel: Table 7's 600-700nm rows for 5C (and 675-700nm for
3C) are a printing/typesetting duplication in the original published paper, not
an extraction error on this project's end - confirmed by cross-checking a
screenshot of the actual page image against this file's text extraction
(2026-08-15): both show 5C's 600nm row as byte-identical to its own 300nm row,
which is not physically possible for a real Kd spectrum. 5C's red-channel values
below use 575nm (the nearest confirmed-clean wavelength) as a stand-in for
600nm - a real measured point, ~25nm off the other three types' target
wavelength, not an interpolated/invented number. Flagged here, not silently
absorbed into "the data."

Blue-channel c(475nm) values are also used as dataset C's beta_b MAGNITUDE
anchor (not just the ratio shape) - see beta_b_anchor_range(). This is a
structural difference from jerlov.py: in the anchored profile, beta_b is no
longer an independently-sampled free parameter, its central value is tied to
whichever water_type gets picked, because this source supplies real absolute
magnitudes, not just relative ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

# c(lambda) = a(lambda) + b(lambda), m^-1, at 475/525/600nm (5C uses 575nm for
# "red" - see module docstring). Read directly from Solonenko & Mobley Tables 6-8.
_C_BLUE_GREEN_RED: dict[str, tuple[float, float, float]] = {
    "1C": (0.077 + 0.469, 0.068 + 0.395, 0.236 + 0.314),
    "3C": (0.105 + 1.36, 0.078 + 1.15, 0.239 + 0.916),
    "5C": (0.204 + 1.71, 0.127 + 1.44, 0.119 + 1.23),  # red = 575nm proxy
    "7C": (0.388 + 3.01, 0.233 + 2.54, 0.301 + 2.03),
}

# K^0_d(lambda), m^-1, at the same three wavelengths - read directly, not derived.
_KD_BLUE_GREEN_RED: dict[str, tuple[float, float, float]] = {
    "1C": (0.134, 0.122, 0.288),
    "3C": (0.223, 0.198, 0.342),
    "5C": (0.400, 0.315, 0.357),  # red = 575nm proxy
    "7C": (0.693, 0.494, 0.478),
}

ANCHOR_WIDTH_FRAC = 0.25  # judgment call, NOT literature-sourced: sampling width
# around each type's literature-derived beta_b center, so C keeps some
# randomization width instead of collapsing to one point value per type
# (per "anchor the range, don't point-match" - see chat trail). Reconsider this
# number if C's images look too narrowly clustered or too erratic per type.


@dataclass(frozen=True)
class AnchoredWaterIOP:
    """Real per-channel ratios AND magnitudes for one Jerlov coastal type,
    from Solonenko & Mobley 2015 - see module docstring for what this replaces.
    """

    name: str
    beta_bg: float  # beta_B / beta_G, from c(lambda) = a(lambda)+b(lambda)
    beta_br: float  # beta_B / beta_R
    beta_blue: float  # c(475nm), m^-1 - beta_b's REAL magnitude anchor for this type
    kd_bg: float  # Kd_B / Kd_G
    kd_br: float  # Kd_B / Kd_R
    kd_green: float  # Kd(525nm), m^-1 - real per-type magnitude, not one shared anchor


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
    """Same signature/shape as jerlov.beta_rgb(), real-data-backed ratios."""
    t = COASTAL_TYPES_ANCHORED[water_type]
    beta_g = beta_b / t.beta_bg
    beta_r = beta_b / t.beta_br
    return beta_r, beta_g, beta_b


def kd_rgb(water_type: str) -> tuple[float, float, float]:
    """Same signature/shape as jerlov.kd_rgb(), but kd_green is this type's OWN
    real magnitude (not a shared anchor), and the R/G/B shape is Kd's own
    measured ratio (not borrowed from beta's ratio) - both of jerlov.py's
    flagged simplifications are resolved here, not just one.
    """
    t = COASTAL_TYPES_ANCHORED[water_type]
    kd_blue = t.kd_green * t.kd_bg
    kd_red = kd_blue / t.kd_br
    return kd_red, t.kd_green, kd_blue


def beta_b_anchor_range(water_type: str) -> tuple[float, float]:
    """(low, high) to sample beta_b from for this type in the anchored profile -
    centred on the real c(475nm) magnitude, +/- ANCHOR_WIDTH_FRAC. Replaces
    domain_randomize.py's single global BETA_B_FLOOR/BETA_B_CEIL, which doesn't
    make sense once beta_b's centre is type-dependent by an order of magnitude
    (1C ~0.5 m^-1 vs 7C ~3.4 m^-1).
    """
    center = COASTAL_TYPES_ANCHORED[water_type].beta_blue
    return center * (1 - ANCHOR_WIDTH_FRAC), center * (1 + ANCHOR_WIDTH_FRAC)
