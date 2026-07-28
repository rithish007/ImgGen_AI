"""Jerlov coastal-water attenuation ratios for the Stage 3b physics transform.

There is no universal, camera-independent 'Jerlov coefficient table'. Akkaynak
et al. (CVPR 2017, "What Is the Space of Attenuation Coefficients in Underwater
Computer Vision?") show that the RGB-domain attenuation coefficients depend on
the camera's spectral response, the imaging range, and scene reflectance
(their Eq. 9) - the numbers are always a projection at one operating point, not
a fixed physical constant. Both that paper and Berman et al. (BMVC 2017,
"Diving into Haze-Lines") present the per-water-type ratios only as a scatter
plot (Fig. 3a / Fig. 2 middle respectively), never as a printed table.

VALUES: taken verbatim from the 'peak' branch of get_water_types.m in
danaberman/underwater-hl (https://github.com/danaberman/underwater-hl,
BMVC 2017, cited by dozens of follow-on underwater vision papers). That
function evaluates the ratios at the peak sensitivity wavelengths of a
generic camera (475/525/600nm for B/G/R - a camera-agnostic Dirac-delta
approximation, per Jiang et al. WACV 2013), which is exactly the same
family of calculation as the CVPR17 paper's Eq. 9.

ONE UNVERIFIED INFERENCE, confirmed with the user before use: the source
file's water_types cell array names 10 Jerlov types (I, IA, IB, II, III, 1C,
3C, 5C, 7C, 9C - this ordering is independently confirmed by the legend of
Fig. 3a in the CVPR17 paper), but the 'peak' branch's beta_BG_pair /
beta_BR_pair arrays hold only 8 values, with no comment indicating which two
are omitted. We assume the array corresponds to I, II, III, 1C, 3C, 5C, 7C, 9C
(dropping the closely-spaced IA/IB oceanic subtypes, which are rarely
distinguished in practice) - the values increase monotonically as expected
with turbidity, which is consistent with but does not prove this reading.

Do not extend this table to other Jerlov types without re-checking that
inference against the primary source.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JerlovRatios:
    """beta_B/beta_G and beta_B/beta_R attenuation ratios for one water type."""

    name: str
    beta_bg: float  # beta_B / beta_G
    beta_br: float  # beta_B / beta_R


# Verbatim from get_water_types.m 'peak' branch (indices 3, 4, 5 of 8, 0-based),
# under the inferred I/II/III/1C/3C/5C/7C/9C mapping described above.
COASTAL_TYPES: dict[str, JerlovRatios] = {
    "1C": JerlovRatios("1C", beta_bg=0.7937, beta_br=0.2773),
    "3C": JerlovRatios("3C", beta_bg=0.9539, beta_br=0.4051),
    "5C": JerlovRatios("5C", beta_bg=1.0930, beta_br=0.4642),
}


def beta_rgb(water_type: str, beta_b: float) -> tuple[float, float, float]:
    """Convert (beta_B/beta_G, beta_B/beta_R) ratios + a chosen beta_B into
    absolute per-channel attenuation coefficients (beta_R, beta_G, beta_B).

    beta_B itself is not pinned by the ratios alone (Berman's method never
    needs it in absolute terms - see uw_restoration.m, which works entirely
    in ratio space). For Stage 3b we need an absolute scale to combine with
    the sampled vertical depth d and camera range z(x,y). Pick beta_B per
    image from a plausible physical range and derive the other two channels
    from it; do not treat beta_B itself as sourced from the same table.
    """
    ratios = COASTAL_TYPES[water_type]
    beta_g = beta_b / ratios.beta_bg
    beta_r = beta_b / ratios.beta_br
    return beta_r, beta_g, beta_b


# ============================================================================
# VEILING LIGHT (B_c^infinity) - Kd approximation
# ============================================================================
#
# STATUS: flagged simplification, not a primary-sourced Kd(lambda) table.
# Confirmed with the user before use (2026-07-28 chat) after an attempt to
# source one failed - see AI_Pipeline_Test_Plan.md's Stage 3b section for the
# full trail:
#
#   - Akkaynak & Treibitz's revised model needs B_c^infinity(d), the veiling
#     light colour/intensity as a function of VERTICAL DEPTH d (distinct from
#     beta_rgb's z-range attenuation above).
#   - The natural source is a Jerlov depth-irradiance chart (K_d(lambda) per
#     water type). Williamson & Hollins 2023 ("Depth profiles of Jerlov water
#     types", Limnol. Oceanogr. Lett. 8:781-788) was checked directly (PDF
#     read in full) and does NOT contain this: it studies whether a water
#     column's Jerlov TYPE classification drifts with depth, not Kd magnitude
#     by wavelength. Its own Table 3 (reconstructed from Jerlov 1976 fig. 71)
#     stops at type "1C" - no 3C/5C data exists there either. The actual
#     Kd(lambda) numbers live in a supplementary figshare dataset this PDF
#     references but does not contain.
#   - One thing that IS useful from that paper: its finest depth resolution
#     near the surface is a single 0-10m bucket. Our pilot's entire d range
#     (0-5m) sits inside that one bucket - there is no published evidence of
#     resolvable optical change within 0-5m specifically, so treating a
#     chosen type's attenuation as constant across our whole d range is not
#     cutting a corner the literature would otherwise resolve.
#
# SIMPLIFICATION USED (two stacked assumptions, both flagged):
#   1. Kd's per-channel SPECTRAL SHAPE reuses the SAME beta_bg/beta_br ratios
#      as beam attenuation above. Kd (diffuse, vertical) and beam attenuation
#      c=a+b (used for beta_rgb, horizontal range) are physically different
#      quantities - this treats them as sharing the same relative R/G/B shape
#      per water type, which is not verified, only plausible.
#   2. Kd's ABSOLUTE green-channel magnitude is anchored to one real citation
#      found during research: "a Kd value of 0.2763 m^-1 is compatible with
#      Jerlov's coastal water types 3C-5C for the wavelength range 500-550nm"
#      (secondary citation, exact primary source not confirmed). Applied
#      UNIFORMLY across 1C/3C/5C - i.e. this does NOT differentiate the
#      absolute Kd magnitude between the three coastal types, only their R/G/B
#      shape (via the existing ratios). That is a real loss of information
#      Jerlov's type ordering implies (1C should genuinely attenuate less than
#      5C) that this approximation does not capture.
#
# Likely fixable with Solonenko & Mobley 2015 ("Inherent optical properties of
# Jerlov water types", Appl. Opt. 54(17):5392-5401) if that becomes available -
# it is the primary IOP source this whole ratio table already wanted (see this
# file's top docstring) and would probably resolve both stacked assumptions at
# once, not just this one.

KD_GREEN_ANCHOR_1976 = 0.2763  # m^-1, ~500-550nm, cited as compatible with 3C-5C


def kd_rgb(water_type: str, kd_green: float = KD_GREEN_ANCHOR_1976) -> tuple[float, float, float]:
    """Per-channel diffuse attenuation coefficient (Kd_R, Kd_G, Kd_B), m^-1.

    Same ratio math as beta_rgb() but starting from a green anchor since that
    is what the one sourced Kd citation gives us (beta_rgb starts from blue
    because that is what beta_b happened to be free in). See this section's
    module-level comment for what is and isn't sourced here.
    """
    ratios = COASTAL_TYPES[water_type]
    kd_blue = kd_green * ratios.beta_bg
    kd_red = kd_blue / ratios.beta_br
    return kd_red, kd_green, kd_blue
