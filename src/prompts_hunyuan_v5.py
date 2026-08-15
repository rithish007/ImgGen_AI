"""
prompts_hunyuan_v5.py
======================
Two changes on top of v4, both evidence-driven, deliberately kept narrow:

1. THE SHELL-LEAK FIX (same bug, same fix as prompts_flux2dev_v8.py).
   Cross-version SAM3 analysis (comparing requested vs detected counts
   across every flux2dev v3-v7 and Hunyuan v1/v4 50-image full run) found
   scallop-leak (phantom scallop detections in scenes that requested ZERO
   scallops) is near-zero everywhere EXCEPT v4 (19/50 images, the worst of
   any dataset checked) and flux2dev v7 (11/50) - the two prompt files that
   introduced this exact COLOR_PALETTE_GUARD wording:
       "seabed, rock and shell material in muted natural tones, not vivid
       or saturated"
   No earlier version of either model's prompt engine mentions "shell" in
   its colour guard. Visually confirmed on outputs/hunyuan/v4/2-pilot_016
   and _023 - both genuinely contain real, individually-rendered scallop
   shells scattered across the seabed as ambient scene detail, not a SAM3
   detection artifact, and not requested. Same "ironic rebound" mechanism
   already documented once for this project (prompts_hunyuan_v2/v3's
   equipment-exclusion list backfiring): naming a concept, even to regulate
   its colour, appears to prime the model to generate it.
   Fix: "shell" dropped from COLOR_PALETTE_GUARD.

2. THE CAMERA-DISTANCE / "PRODUCT PHOTOGRAPH" FIX.
   User's direct feedback (2026-08-15): "Every hunyuan images are
   beautiful, very product photograph like. I want the floor to be 5-8
   meters below the camera." Root cause, already correctly diagnosed
   earlier this session: CAMERA_HEIGHTS["far"] in prompts_hunyuan_v4.py
   already reads "camera ~5-8m up, wide overview survey view, far above
   the seabed" - exactly the requested distance - but
   generate_hunyuan_v4.py's build_prompt() calls never pass
   camera_height="far", so it falls back to a random choice among all four
   heights (~25% far, ~75% close/medium/high). v4 dropped the forced-far
   behaviour on purpose at the time, to isolate the equipment-hallucination
   revert as the only variable under test (see v4's own docstring) - that
   isolation already paid off (confirmed 0/50 equipment-hallucination at
   full scale), so this file re-adds the camera-distance fix on its own,
   now that it's safe to test as the next single variable.

   This is not a new, untested idea: flux2dev v6/v7 already forced
   camera_height="far" plus a short SUBJECT_SCALE_GUARD ("organisms small
   in the wide view"), and the user's own full 50-image visual review of
   flux2dev v7 (this session) confirmed clean survey-distance framing with
   no camera/robot-in-frame or blob-urchin regressions. This file applies
   the identical mechanism to Hunyuan - camera_height="far" forced in
   generate_hunyuan_v5.py, plus a matching subject-scale guard - rather
   than inventing a new untested lever. Worded a little fuller than
   flux2dev's terse version (Hunyuan has no FLUX.2-style 512-token cap and
   this file's existing guards - REALISM_GUARD, GROUND_CONTACT_GUARD - are
   already full sentences, not compressed phrases), but deliberately NOT
   stacked with additional new guards beyond this one restored pair -
   explicit user instruction: reiterate properly, but don't over-engineer
   it. In particular, no elaborate multi-clause exclusion language is
   added anywhere in this file - that exact pattern (prompts_hunyuan_v2/v3's
   equipment-exclusion list) already backfired once this project, and the
   shell-leak bug above is a second, independent case of the same
   mechanism, so this file stays deliberately conservative on guard
   wording everywhere, not just the two changes above.

NOT YET RUN. Needs a smoke test and then a full 50-image run before
trusting either fix - both are strong, evidence-backed hypotheses, not
guarantees.

--- prompts_hunyuan_v4.py's own docstring follows for everything this file
didn't touch (the equipment-hallucination revert, product-photo/
ground-contact guards, colour-palette materials scoping, etc. - all still
accurate) ---

REVERT + evidence-driven rewrite. Diagnosed why hunyuan_v2/v3's
equipment-hallucination "fix" wasn't working (both smoke tests still hit
1/3 = 33%, no better than doing nothing) by going back to the two full
50-image Hunyuan runs the user identified as "almost right"
(outputs/hunyuan/v1/3-pilot_promptfix and outputs/hunyuan/v1/5-pilot, both
using prompts_v2.py's plain wording) and actually counting the defect rate
there: 2/50 = 4%, scanned across all 50 images, not a smoke-test-sized
sample.

DIAGNOSIS: hunyuan_v2/v3's "fix" - an elaborate first-person-POV opening
("captured in first-person point-of-view... as though the viewer is
looking directly through the camera's own lens") plus an explicit
equipment-exclusion list ("no robotic arms, thrusters, camera housing,
cables...") - measurably made the defect WORSE, not better (33% vs the
baseline's 4%; two independent 1/3 hits against a true ~4% rate have
roughly 1-in-80 odds by chance, not proof but strong enough to act on).
Likely mechanism: naming the exact equipment to exclude, stacked with
camera/lens language elsewhere in the prompt, probably primed the model
toward the concept rather than suppressing it - an "ironic rebound" effect,
not unheard of in text-to-image models. Overspecifying an exclusion isn't
automatically safer than a plain description.

WHAT THIS FILE ACTUALLY DOES: reverts the opening line and REALISM_GUARD to
prompts_v2.py's original plain wording (the empirically-good 4% version),
while keeping everything else that was never implicated in the defect and
has its own independent justification: PRODUCT_PHOTO_GUARD, GROUND_CONTACT_GUARD
(thread 2's "product photography/hovering" fixes - about specimen-display
framing and substrate contact, no camera/equipment language, no reason to
suspect these), the no-kelp/DUO-consistent SCENE_TEMPLATES/ALGAE_VARIATIONS/
LIGHTING_CONDITIONS pools, the blue starfish variant, and the urchin
grey-blob fix. COLOR_PALETTE_GUARD further narrowed to materials-only (see
its own comment - real water_stats.py evidence, ported from prompts_v7.py).

EVERYTHING ELSE IS UNCHANGED FROM prompts_v3.py: SUBSTRATE_VARIATIONS/
ROCK_FORMATIONS/CAMERA_FOV/CAMERA_MOTION/IMAGING_CONDITIONS/COMPOSITIONS/
DEPTH_DISTRIBUTIONS, SCENE_DENSITIES, DETECTION_DIFFICULTY, CAMERA_HEIGHTS,
COUNT_RANGES, the count=1 arrangements_solo/group split, the urchin
grey-blob fix (carried forward though Hunyuan's own anatomy has been clean
throughout - kept identical to flux2dev/klein for a fair cross-model
comparison, not because Hunyuan needed it), the single-class+dense
rock-formation gate, the wide+sparse FRAMING_COUNT_ANCHOR, and the
BIVALVE_GUARD scallop-only gate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================================
# CLASS DEFINITIONS (unchanged from prompts_hunyuan_v4.py)
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
# SCENE / ENVIRONMENT (unchanged from prompts_hunyuan_v4.py)
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
# ECOLOGICAL / COMPOSITIONAL CONDITIONS (unchanged from prompts_hunyuan_v4.py)
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

# v5 (this file): wording expanded slightly - not a new idea, just made a
# little fuller given Hunyuan has no token-budget pressure. "far" is the
# ONLY height this file's generate script uses (forced, not random - see
# generate_hunyuan_v5.py), matching flux2dev v6/v7's already-proven
# approach to the same "product photography" complaint.
CAMERA_HEIGHTS = {
    "low": "camera ~0.5m up, angled down",
    "medium": "camera ~1m up, angled down",
    "high": "camera ~1.5-2m up, looking down",
    "far": "camera positioned roughly 5 to 8 meters above the seabed, a wide elevated survey view showing the full scene from a clear vertical distance",
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
# GUARDS
# ============================================================================

COMPOSITION_GUARD = "natural asymmetric spacing, no decorative symmetry or cloned objects"
SPECIES_GUARD = "only these organisms and seabed material in frame, no other animals"
REALISM_GUARD = "plain documentary robot photo, no divers, boats, or CG rendering"

PRODUCT_PHOTO_GUARD = "candid in-situ ecological documentation, not a specimen display or catalogue photograph"

GROUND_CONTACT_GUARD = "every organism rests directly on the substrate with genuine physical contact and a soft contact shadow, never elevated or hovering above the seabed"

# v5 (this file): "shell" dropped - see module docstring, this is the
# shell-leak fix. Everything else about this guard's scoping (materials
# only, not the whole frame's colour grading) is unchanged from v4.
COLOR_PALETTE_GUARD = "seabed and rock material in muted natural tones, not vivid or saturated"

# v5 (this file): restored, worded a little fuller than flux2dev's
# equivalent ("organisms small in the wide view") to match this file's
# existing sentence-length guards rather than switching styles mid-file.
# Paired with camera_height="far" being forced in generate_hunyuan_v5.py -
# same combination already confirmed clean on flux2dev v7's full 50-image
# visual review (this session). See module docstring point 2 for the full
# reasoning - this is the direct fix for the "product photograph" framing
# complaint.
SUBJECT_SCALE_GUARD = "organisms appear small within the wide elevated view, not enlarged or filling the frame, consistent with a genuine overhead survey distance rather than a close specimen shot"

OPTICAL_GUARD = "full frame, no fisheye or vignette"

BIVALVE_GUARD = "bivalve shells fully closed and undisturbed"

FRAMING_COUNT_ANCHOR = "count is a strict total for the frame, not per unit area"


# ============================================================================
# FRAMING (unchanged from prompts_hunyuan_v4.py)
# ============================================================================

FRAMING = {
    "close-up": "close survey framing at natural working distance, camera near the seabed but keeping full ecological context, not a tight macro product shot; objects at different distances",
    "mid": "mid-distance framing, foreground and mid-ground objects visible",
    "wide": "wide framing, larger section of seabed, objects at different depths",
}


# ============================================================================
# OBJECT COUNT RANGES (unchanged from prompts_hunyuan_v4.py)
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
# RANDOM HELPERS (unchanged from prompts_hunyuan_v4.py)
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
# CLASS PHRASE GENERATION (unchanged from prompts_hunyuan_v4.py)
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
# SCENE OBJECT GENERATION (unchanged from prompts_hunyuan_v4.py)
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
    Construct a Hunyuan-specific underwater survey image-generation prompt.

    Same rng draw sequence/order as prompts_hunyuan_v2/v3/v4.py, so a given
    seed selects the same environment/subject content across versions -
    only the surrounding scaffolding text differs (see module docstring:
    shell-leak fix + restored camera-distance/subject-scale guard).

    Assembly order follows Hunyuan's own recommended prompt structure (main
    subject/scene -> image quality/style -> composition/perspective ->
    lighting/atmosphere -> technical parameters).
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

    # ------------------------------------------------------------------
    # Select scene components (same rng sequence order as prompts_hunyuan_v2/
    # v3/v4, so a given seed draws the same environment even though the
    # assembled sentences below differ)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Build subject descriptions (unchanged from prompts_hunyuan_v4.py)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Compose the final prompt - Hunyuan's own recommended structure:
    # main subject/scene -> image quality/style -> composition/perspective
    # -> lighting/atmosphere -> technical parameters.
    # ------------------------------------------------------------------

    opening = f"Photorealistic underwater robot-survey photograph. Scene contains: {subjects}."

    guard_parts = [COMPOSITION_GUARD, SPECIES_GUARD, REALISM_GUARD, PRODUCT_PHOTO_GUARD, GROUND_CONTACT_GUARD, COLOR_PALETTE_GUARD, SUBJECT_SCALE_GUARD, OPTICAL_GUARD]
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

    perspective = (
        f"{composition}; {SCENE_DENSITIES[density]}; "
        f"{DETECTION_DIFFICULTY[difficulty]}; {depth_distribution}; "
        f"{FRAMING[framing]}."
    )

    atmosphere = f"{lighting}; {SCENE_WATER_PHRASE}."

    technical = (
        f"{CAMERA_HEIGHTS[camera_height]}, {camera_fov}; "
        f"{camera_motion}, {imaging}."
    )

    prompt = " ".join([
        opening,
        guards,
        environment,
        perspective,
        atmosphere,
        technical,
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
