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
