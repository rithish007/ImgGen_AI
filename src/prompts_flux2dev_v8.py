"""
prompts_flux2dev_v8.py
=======================
Prompt engine v8 - ONE fix on top of v7: drops "shell" from
COLOR_PALETTE_GUARD. Nothing else changes from v7 - user's own visual
review of v7's output concluded it already has the upper hand over Hunyuan
(camera-to-object distance reads better than Hunyuan's "product photograph"
look, and the haziness gap is minor and acceptable) - so this file is
deliberately a single, narrow fix, not a broader rework.

THE BUG: cross-version SAM3 analysis (comparing requested vs detected counts
across every flux2dev v3-v7 and Hunyuan v1/v4 output, all 50-image full
runs) found scallop-leak (phantom scallop detections in scenes that
requested ZERO scallops) is near-zero in every version EXCEPT v7 (11/50
images) and Hunyuan v4 (19/50 images, prompts_hunyuan_v4.py) - the two
versions that introduced this exact COLOR_PALETTE_GUARD wording:
"seabed, rock and shell material in muted natural tones, not vivid or
saturated". No earlier version of either model's prompt engine mentions
"shell" in its colour guard. Visually confirmed via outputs/flux2dev/v7/
row 025 and row 043 - both requested zero scallops, both genuinely contain
dozens of individually-rendered scallop shells scattered across the seabed,
not a SAM3 detection artifact.

Likely mechanism: same "ironic rebound" pattern already documented once
this project (prompts_hunyuan_v2/v3's equipment-exclusion list backfiring)
- naming a concept in the prompt, even in a neutral regulatory context
("keep X's colour muted"), appears to prime the model to generate X into
the scene regardless of whether X was requested. This is the second time
this exact mechanism has caused a scallop-specific defect - the first was
flux2dev v4's "occasional shell fragments" scene-template phrase (fixed in
v5 by dropping the phrase; the fix measurably worked, v4's 4/50 leak rate
dropped to v5's 1/50).

THE FIX:
    COLOR_PALETTE_GUARD = "seabed, rock and shell material in muted natural tones, not vivid or saturated"
    -> "seabed and rock material in muted natural tones, not vivid or saturated"

Nothing else in this file differs from prompts_flux2dev_v7.py - same
CAMERA_HEIGHTS["far"] forcing, same SUBJECT_SCALE_GUARD, same materials-only
scoping rationale, same everything. See prompts_flux2dev_v7.py's own
docstring (reproduced below) for the full history this file doesn't touch.

NOT YET RUN. Needs a smoke test and then a full 50-image run before trusting
the fix actually closes the scallop-leak gap - this is a hypothesis backed
by strong correlational evidence (two independent cases of the same
mechanism), not a guarantee.

--- prompts_flux2dev_v7.py's own docstring follows for everything this file
didn't touch ---

Prompt engine v7 - narrows COLOR_PALETTE_GUARD to substrate/organism
materials only, on top of v6's camera-distance work.

WHY: real quantitative evidence, not a subjective call. water_stats.py
(built in a parallel session, computes pixel-level colour-cast stats -
mean_rgb, ratio_rg/ratio_bg, dark-channel haziness) compared against 5,448
real DUO training images found DUO's actual water is GREEN-dominant
(ratio_rg=0.451 - red is under half of green), not blue and not the
brown/beige v5's COLOR_PALETTE_GUARD aimed for. Worse: v5's guard measurably
moved the WRONG direction - ratio_rg went from v4's 1.024 to v5's 1.102 (MORE
red-dominant), while the Stage 3 domain-randomization pipeline's own output
(reports/5pilot_dr_placeholder_water_stats.json) already sits at 0.902,
closer to DUO's real value than anything Stage 1 has produced, with no
Stage-1 colour guard driving it there at all.

CONCLUSION: Stage 3 (Akkaynak-Treibitz domain randomization,
src/domain_randomize.py) is the component actually responsible for imposing
realistic water colour as a physically-motivated post-process - Stage 1
chasing DUO's water-tint statistics directly is redundant with, and
apparently working against, what Stage 3 already does starting from
SCENE_WATER_PHRASE's deliberately clear/colour-neutral render. What Stage 1
SHOULD keep constraining is the substrate/organism MATERIAL colour (sand,
rock tones) - not vivid/saturated, but not chasing a specific water cast
either, since that's not this stage's job.

v6's camera-distance work (CAMERA_HEIGHTS["far"], SUBJECT_SCALE_GUARD,
forced camera_height="far" in the generate script) carries forward
unchanged - user's own visual review (2026-08-15) of v7's full 50-image run
confirmed this combination reads as a proper survey-distance shot, not
Hunyuan's close "product photograph" look, and specifically re-checked the
3 rows (22, 28, 30) that were defective in v6 (robot-in-frame, blob
urchins) - all clean in v7.

Everything else (CLASSES, SCENE_TEMPLATES, framing, count ranges, guards)
is unchanged from v6/v5/v4/v3/v2 - see prompts_flux2dev_v7.py for the full
chained docstring history.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================================
# CLASS DEFINITIONS (unchanged from prompts_flux2dev_v7.py)
# ============================================================================

CLASSES: dict[int, dict[str, object]] = {

    0: {
        "duo_label": "starfish",
        "short": "starfish",

        "morphology": [
            "a small starfish, five arms, mottled brown-grey, rough texture",
            "a small five-armed starfish, muted brown-grey, irregular darker patches",
            "a small five-armed starfish, dark brown-grey mottling, irregular arm proportions",
            "a small starfish, five broad arms, muted reddish-brown and grey",
            "a small starfish, five arms, muted blue-grey, rough texture",
        ],

        "arrangements_solo": [
            "resting flat against rock and algae",
            "resting beside a low rock ledge, partly obscured",
            "lying on sediment near algae",
            "partially covered by fine sediment, resting against a rock",
        ],
        "arrangements_group": [
            "resting flat against rock and algae",
            "resting beside a low rock ledge, partly obscured",
            "lying on sediment near algae",
            "partially covered by fine sediment, resting against a rock",
        ],
    },

    1: {
        "duo_label": "echinus",
        "short": "sea urchin",

        "morphology": [
            "a sea urchin, low flattened dome body, dense short spines, dark brown-black",
            "a nearly black sea urchin, low flattened dome body, dense short spines",
            "a nearly black sea urchin, low wide body, dense short dark spines",
            "a dark brown sea urchin, flattened low dome body, dense short spines",
        ],

        "arrangements_solo": [
            "wedged into a rocky crevice, partly shielded by an overhang",
            "resting on a rocky ledge, spines catching the light",
        ],
        "arrangements_group": [
            "clustered in a rocky crevice, several touching or overlapping",
            "wedged against a rock face, at slightly different depths",
            "grouped along a ledge, some individuals partly hidden behind rocks",
        ],
    },

    2: {
        "duo_label": "scallop",
        "short": "scallop",

        "morphology": [
            "a scallop, ribbed fan shell, cream-brown, tightly closed, dusted with sediment",
            "a fan-shaped scallop shell, cream-brown, tightly closed, ribbed, partly buried in sediment",
            "a small scallop, ribbed shell, light brown-cream, tightly closed, thin sediment layer",
            "a scallop, textured ribbed shell, cream-brown, tightly closed, partly buried",
        ],

        "arrangements_solo": [
            "partially buried at an irregular spot on the seabed",
        ],
        "arrangements_group": [
            "sparsely scattered across open sediment, substantial irregular spacing between individuals",
            "loosely distributed across sand and gravel, individuals at different distances",
            "partially buried at irregular locations across the seabed",
        ],
    },
}


# ============================================================================
# SCENE / ENVIRONMENT (unchanged from prompts_flux2dev_v7.py)
# ============================================================================

SCENE_TEMPLATES = [
    "temperate coastal seabed - sand, coarse sediment, gravel, irregular rocks, shallow ledges",
    "open sandy plain - fine sand, only occasional scattered pebbles, very little exposed rock",
    "dense boulder field - large rounded boulders close together, narrow sediment gaps between them",
    "silty low-lying flat - fine beige sediment, sparse small rocks",
    "cobble and shingle bank - uniform fist-sized rounded stones covering most of the seabed",
    "eroded rock ledge and wall - a low rocky wall rising from the seabed, sediment at its base",
    "mixed rubble seabed - broken rock fragments, gravel and sand in irregular patches",
]

ALGAE_VARIATIONS = [
    "sparse turf algae on rocks",
    "dense green-brown algae covering exposed rock surfaces",
    "thin diatom film over sediment, faint brownish tinge",
    "little to no visible algae, bare rock and sediment",
]

SUBSTRATE_VARIATIONS = [
    "subtle variation in sediment grain size",
    "fine sediment accumulating around rocks",
    "small gravel mixed irregularly with sand",
    "exposed rock patches surrounded by fine sediment",
    "slightly uneven sediment with small stones and debris",
]

ROCK_FORMATIONS = [
    "rounded cobbles scattered loosely nearby",
    "a small pile of angular rubble nearby",
    "loose pebbles mixed into the sediment nearby",
    "small boulders resting nearby, spaced irregularly",
    "pebbles and small cobbles in a shallow depression nearby",
    "a low pile of rubble and small boulders nearby",
]


# ============================================================================
# ECOLOGICAL / COMPOSITIONAL CONDITIONS (unchanged from prompts_flux2dev_v7.py)
# ============================================================================

SCENE_DENSITIES = {
    "sparse": "relatively open seabed, substantial exposed sediment, limited clutter",
    "moderate": "moderately cluttered seabed, natural rocks/algae/sediment across foreground and mid-ground",
    "dense": "visually cluttered seabed, rocks/algae/gravel/sediment and closely spaced natural debris",
}

DETECTION_DIFFICULTY = {
    "easy": "objects mostly visible, limited occlusion",
    "moderate": "some objects partly obscured by rocks/algae/sediment",
    "hard": "several objects small or partly obscured, blending into substrate",
}

SCENE_WATER_PHRASE = "clear water, true-to-life colour"

LIGHTING_CONDITIONS = [
    "diffuse natural sunlight, soft uneven brightness, subtle shadows",
    "soft filtered daylight, gentle brightness variation, low-contrast shadows",
    "natural daylight, realistic underwater attenuation, soft illumination",
    "weak diffuse daylight, slightly uneven illumination",
    "bright shallow sunlight, dappled light patterns across the seabed",
    "flat overcast light, minimal shadows, slight greenish colour cast",
]

CAMERA_HEIGHTS = {
    "low": "camera ~0.5m up, angled down",
    "medium": "camera ~1m up, angled down",
    "high": "camera ~1.5-2m up, looking down",
    "far": "camera ~5-8m up, wide overview",
}

CAMERA_FOV = [
    "moderately wide view",
    "wide-angle view",
    "compact-camera wide view",
]

CAMERA_MOTION = [
    "slight motion softness from a moving robot",
    "very mild motion blur, slow robotic movement",
    "minimal motion softness, forward-moving camera",
    "stable capture, only subtle sensor/motion effects",
]

IMAGING_CONDITIONS = [
    "subtle sensor noise, mild detail loss with distance",
    "subtle image noise, natural exposure",
    "fine sensor noise, subtle contrast loss",
    "subtle sensor noise, restrained softness",
]

COMPOSITIONS = [
    "candid documentary style, off-centre framing",
    "irregular asymmetric composition",
    "unposed framing, layered depth",
    "documentary frame, no deliberate hero subject",
]

DEPTH_DISTRIBUTIONS = [
    "objects span foreground to background",
    "varied apparent sizes from camera distance",
    "some objects close, others farther away",
]


# ============================================================================
# POSITIVE GUARDS
# ============================================================================

COMPOSITION_GUARD = "natural asymmetric spacing, no decorative symmetry or cloned objects"
SPECIES_GUARD = "only these organisms and seabed material in frame, no other animals"
REALISM_GUARD = "plain documentary robot photo, no divers, boats, or CG rendering"
OPTICAL_GUARD = "full frame, no fisheye or vignette"

# v8 (this file): "shell" dropped - see module docstring. This was the only
# change made in this file. Every other guard/pool/structure is identical
# to prompts_flux2dev_v7.py.
COLOR_PALETTE_GUARD = "seabed and rock material in muted natural tones, not vivid or saturated"

SUBJECT_SCALE_GUARD = "organisms small in the wide view"

BIVALVE_GUARD = "bivalve shells fully closed and undisturbed"

FRAMING_COUNT_ANCHOR = "count is a strict total for the frame, not per unit area"


# ============================================================================
# FRAMING (unchanged from prompts_flux2dev_v7.py)
# ============================================================================

FRAMING = {
    "close-up": "close survey framing at natural working distance, camera near the seabed but not a close macro product shot, objects at different distances",
    "mid": "mid-distance framing, foreground and mid-ground objects visible",
    "wide": "wide framing, larger section of seabed, objects at different depths",
}


# ============================================================================
# OBJECT COUNT RANGES (unchanged from prompts_flux2dev_v7.py)
# ============================================================================

COUNT_RANGES = {
    0: {"sparse": (1, 2), "moderate": (1, 3), "dense": (2, 4)},
    1: {"sparse": (1, 2), "moderate": (2, 4), "dense": (3, 6)},
    2: {"sparse": (1, 2), "moderate": (2, 4), "dense": (3, 6)},
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PromptMetadata:
    """Metadata describing the synthetic scene requested by the prompt."""

    seed: int
    density: str
    difficulty: str
    camera_height: str
    framing: str
    class_counts: dict[int, int]
    scene_template_index: int
    algae_variation_index: int
    substrate_variation_index: int
    rock_formation_index: int
    lighting_index: int
    composition_index: int
    camera_fov_index: int
    camera_motion_index: int
    imaging_index: int


# ============================================================================
# RANDOM HELPERS (unchanged from prompts_flux2dev_v7.py)
# ============================================================================

def _choose(rng: random.Random, values: list[str]) -> tuple[str, int]:
    index = rng.randrange(len(values))
    return values[index], index


def _random_count(rng: random.Random, class_id: int, density: str) -> int:
    low, high = COUNT_RANGES[class_id][density]
    return rng.randint(low, high)


def _drop_leading_article(text: str) -> str:
    """Strip only a genuine leading 'a ' or 'an ', not every occurrence."""
    for article in ("an ", "a "):
        if text.startswith(article):
            return text[len(article):]
    return text


# ============================================================================
# CLASS PHRASE GENERATION (unchanged from prompts_flux2dev_v7.py)
# ============================================================================

def class_phrase(class_id: int, count: int, rng: random.Random) -> str:
    entry = CLASSES[class_id]
    morphology = rng.choice(entry["morphology"])
    arrangement = rng.choice(
        entry["arrangements_solo"] if count == 1 else entry["arrangements_group"]
    )

    if count == 1:
        return f"{morphology}, {arrangement}"

    return (
        f"{count} living {entry['short']}: "
        f"{_drop_leading_article(morphology)}; "
        f"{arrangement}"
    )


# ============================================================================
# SCENE OBJECT GENERATION (unchanged from prompts_flux2dev_v7.py)
# ============================================================================

def generate_class_counts(
    rng: random.Random,
    density: str,
    min_classes: int = 2,
    max_classes: int = 4,
) -> dict[int, int]:
    available = list(CLASSES.keys())
    number_of_classes = rng.randint(min_classes, min(max_classes, len(available)))
    selected = rng.sample(available, number_of_classes)
    return {cid: _random_count(rng, cid, density) for cid in sorted(selected)}


# ============================================================================
# PROMPT BUILDER
# ============================================================================

def build_prompt(
    counts: dict[int, int],
    *,
    seed: int = 0,
    density: Optional[str] = None,
    difficulty: Optional[str] = None,
    camera_height: Optional[str] = None,
    framing: Optional[str] = None,
) -> tuple[str, PromptMetadata]:
    """
    Construct a complete underwater survey image-generation prompt.

    Identical assembly logic to prompts_flux2dev_v7.py - only
    COLOR_PALETTE_GUARD's text differs (see module docstring).
    """

    rng = random.Random(seed)

    if density is None:
        density = rng.choice(list(SCENE_DENSITIES.keys()))
    if density not in SCENE_DENSITIES:
        raise ValueError(f"Invalid density {density!r}. Expected one of {list(SCENE_DENSITIES)}")

    if difficulty is None:
        difficulty = rng.choice(list(DETECTION_DIFFICULTY.keys()))
    if difficulty not in DETECTION_DIFFICULTY:
        raise ValueError(f"Invalid difficulty {difficulty!r}. Expected one of {list(DETECTION_DIFFICULTY)}")

    if camera_height is None:
        camera_height = rng.choice(list(CAMERA_HEIGHTS.keys()))
    if camera_height not in CAMERA_HEIGHTS:
        raise ValueError(f"Invalid camera height {camera_height!r}. Expected one of {list(CAMERA_HEIGHTS)}")

    if framing is None:
        framing = rng.choice(list(FRAMING.keys()))
    if framing not in FRAMING:
        raise ValueError(f"Invalid framing {framing!r}. Expected one of {list(FRAMING)}")

    scene_template, scene_idx = _choose(rng, SCENE_TEMPLATES)
    algae, algae_idx = _choose(rng, ALGAE_VARIATIONS)
    substrate, substrate_idx = _choose(rng, SUBSTRATE_VARIATIONS)
    rock_formation, rock_formation_idx = _choose(rng, ROCK_FORMATIONS)
    lighting, lighting_idx = _choose(rng, LIGHTING_CONDITIONS)
    composition, composition_idx = _choose(rng, COMPOSITIONS)
    camera_fov, camera_fov_idx = _choose(rng, CAMERA_FOV)
    camera_motion, camera_motion_idx = _choose(rng, CAMERA_MOTION)
    imaging, imaging_idx = _choose(rng, IMAGING_CONDITIONS)
    depth_distribution = rng.choice(DEPTH_DISTRIBUTIONS)

    include_rock_formation = not (len(counts) == 1 and density == "dense")

    subject_phrases = []
    for class_id in sorted(counts):
        if class_id not in CLASSES:
            raise ValueError(f"Unknown class ID {class_id}. Expected IDs: {list(CLASSES)}")
        count = counts[class_id]
        if count < 1:
            raise ValueError(f"Class {class_id} has invalid count {count}. Counts must be >= 1.")
        subject_phrases.append(class_phrase(class_id=class_id, count=count, rng=rng))

    if len(subject_phrases) == 1:
        subjects = subject_phrases[0]
    elif len(subject_phrases) == 2:
        subjects = f"{subject_phrases[0]} and {subject_phrases[1]}"
    else:
        subjects = ", ".join(subject_phrases[:-1]) + ", and " + subject_phrases[-1]

    guard_parts = [COMPOSITION_GUARD, SPECIES_GUARD, REALISM_GUARD, COLOR_PALETTE_GUARD, SUBJECT_SCALE_GUARD, OPTICAL_GUARD]
    if 2 in counts:
        guard_parts.append(BIVALVE_GUARD)
    if framing == "wide":
        guard_parts.append(FRAMING_COUNT_ANCHOR)
    guards = "; ".join(guard_parts) + "."

    environment = (
        f"{scene_template}, with {algae} and {substrate}"
        + (f", {rock_formation}" if include_rock_formation else "")
        + "."
    )

    context = (
        f"{composition}; {SCENE_DENSITIES[density]}; "
        f"{DETECTION_DIFFICULTY[difficulty]}; {depth_distribution}."
    )

    secondary = (
        f"{CAMERA_HEIGHTS[camera_height]}, {camera_fov}; "
        f"{lighting}; {camera_motion}, {imaging}; "
        f"{SCENE_WATER_PHRASE}; {FRAMING[framing]}."
    )

    prompt = " ".join([
        f"Photorealistic underwater robot-survey photograph. Scene contains: {subjects}.",
        guards,
        environment,
        context,
        secondary,
    ])

    metadata = PromptMetadata(
        seed=seed,
        density=density,
        difficulty=difficulty,
        camera_height=camera_height,
        framing=framing,
        class_counts=dict(counts),
        scene_template_index=scene_idx,
        algae_variation_index=algae_idx,
        substrate_variation_index=substrate_idx,
        rock_formation_index=rock_formation_idx,
        lighting_index=lighting_idx,
        composition_index=composition_idx,
        camera_fov_index=camera_fov_idx,
        camera_motion_index=camera_motion_idx,
        imaging_index=imaging_idx,
    )

    return prompt, metadata


# ============================================================================
# CLASS NAME HELPERS
# ============================================================================

def class_names() -> dict[int, str]:
    return {cid: entry["short"] for cid, entry in CLASSES.items()}


def detector_prompts() -> dict[int, str]:
    return {cid: entry["short"] for cid, entry in CLASSES.items()}


if __name__ == "__main__":
    counts = {0: 3, 1: 2, 2: 4}
    prompt, metadata = build_prompt(counts, seed=1, camera_height="far")
    print(prompt)
    print()
    print(f"word count: {len(prompt.split())}")
    from dataclasses import asdict as _asdict
    print(_asdict(metadata))
