"""
prompts_hunyuan_v2.py
======================
Hunyuan-only prompt engine, forked from prompts_v3.py to fix the two defect
threads found in the 5-pilot Hunyuan audit, using Hunyuan's own prompting
headroom instead of FLUX.2's constraints.

WHY A SEPARATE FILE (not folded into prompts_v3.py):
HunyuanImage-3.0's own docs/repo have no stated hard token/word cap - training
captions ranged 30-1000 words, unlike FLUX.2's confirmed hard 512-token
truncation (see prompts_v2.py's module docstring). prompts_v2.py/prompts_v3.py
are deliberately compressed and reordered (subject+guards first) specifically
to survive that FLUX.2-only truncation risk. Hunyuan has no such constraint,
so this file is free to spell things out explicitly instead of compressing -
and the two defects below are exactly the kind of thing that benefits from
being spelled out rather than implied. Also follows Hunyuan's own recommended
prompt structure (main subject/scene -> image quality/style -> composition/
perspective -> lighting/atmosphere -> technical parameters) instead of BFL's
FLUX.2-specific ordering, since front-loading for truncation safety is not a
concern here.

THE TWO DEFECT THREADS (found auditing 5-pilot/hunyuan, prompts_v2, 50 images):

1. SURVEY EQUIPMENT HALLUCINATING INTO FRAME (images 16, 19 - a robotic
   camera rig/arm visible in shot). Root cause: prompts_v2's opening line
   ("Photorealistic underwater robot-survey photograph") and REALISM_GUARD
   ("plain documentary robot photo") both describe the image as a photo OF a
   robot doing a survey, never disambiguating that the camera IS the robot
   (first-person POV) rather than a photo showing a robot from the outside.
   REALISM_GUARD's exclusion list (divers/boats/CG) never mentioned the
   equipment itself as something to exclude.
   Fix: explicit first-person-POV framing device in the opening line ("as
   though the viewer is looking directly through the camera's own lens") and
   an expanded REALISM_GUARD that explicitly excludes robotic arms,
   thrusters, camera housing, cables, and "any other part of the survey
   vehicle."

2. "PRODUCT PHOTOGRAPHY" LOOK - camera reads as too close, and organisms
   read as sitting ON TOP of the substrate with a slight gap rather than
   genuinely resting on it (reported directly: "Most starfish and scallop are
   hovering a little bit from the ground"). Re-verified 2026-08-10 by
   sampling 5-pilot/hunyuan images 16/19/25/40: all show organisms lined up
   frontally at close range with no visible ground contact/shadow - a
   catalogue-photo composition, not a candid survey frame. Two contributing
   causes, both addressed here:
     - No guard anywhere said organisms must show physical ground contact -
       "resting flat against rock" (starfish arrangement text) describes
       position but not contact/shadow.
     - FRAMING["close-up"]'s text ("close framing, camera near the seabed")
       has no anti-macro/anti-product-shot qualifier, unlike wide/mid.
   Fix: two new guards - PRODUCT_PHOTO_GUARD (explicitly rules out specimen-
   display/catalogue framing) and GROUND_CONTACT_GUARD (explicit contact +
   shadow requirement) - plus a reworded close-up FRAMING entry that keeps
   the close distance (still needed for the manifest's close-up rows) but
   qualifies it as survey-distance, not a macro product shot.

   NOTE - NOT fixed here, out of scope for this file: the manifest itself
   assigns close-up+dense framing to 21/50 rows (42%) in a rigid block
   pattern (rows 1-7, 22-28, 43-49 of every stage that reuses
   manifests/2-pilot.json), which structurally biases the whole dataset -
   all three models, not just Hunyuan - toward the tight/dense look on
   nearly half of all images. That's a manifest-level fix, not a prompt-text
   one; flagged in the 5-pilot cross-model audit, not addressed by this file.

UPDATE (same session, before this file was ever run): also picked up
prompts_v4.py's two fixes, keeping this file aligned with flux2dev's CLASSES/
pool content rather than freezing it at the v3 snapshot: starfish
recoloured orange/reddish-brown + camouflage language dropped from
arrangement text (flux2dev evidence: mean detection ratio 0.54x on
multi-class starfish rows vs 0.91x single-class - not yet measured on
Hunyuan specifically, but no reason to leave Hunyuan on wording already
shown to matter elsewhere), and SCENE_TEMPLATES/ALGAE_VARIATIONS/
LIGHTING_CONDITIONS expanded with genuinely different content instead of
near-synonymous paraphrases (5-pilot audit finding, applies to all three
models equally). See prompts_v4.py's module docstring for the full
evidence behind both.

EVERYTHING ELSE IS UNCHANGED FROM prompts_v3.py: SUBSTRATE_VARIATIONS/
ROCK_FORMATIONS/CAMERA_FOV/CAMERA_MOTION/IMAGING_CONDITIONS/COMPOSITIONS/
DEPTH_DISTRIBUTIONS, SCENE_DENSITIES, DETECTION_DIFFICULTY, CAMERA_HEIGHTS,
COUNT_RANGES, the count=1 arrangements_solo/group split, the urchin
grey-blob fix (carried forward though Hunyuan's own anatomy has been clean
throughout - kept identical to flux2dev/klein for a fair cross-model
comparison, not because Hunyuan needed it), the single-class+dense
rock-formation gate, the wide+sparse FRAMING_COUNT_ANCHOR, and the
BIVALVE_GUARD scallop-only gate.

NOT YET DONE: generate_hunyuan_v2.py exists and dry-runs cleanly, but no
smoke test has actually been run on Stanage yet. Do that (and re-verify
word counts if curious, though there's no hard cap to check against) before
treating this as validated - same rule as every other prompt file in this
project, don't trust a new prompt file just because it imports cleanly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================================
# CLASS DEFINITIONS (unchanged from prompts_v3.py, including the urchin fix)
# ============================================================================

CLASSES: dict[int, dict[str, object]] = {

    0: {
        "duo_label": "starfish",
        "short": "starfish",

        # Reverted to the original brown-grey/muted palette - the orange
        # recolour (meant to fix flux2dev's multi-class starfish undercount,
        # see prompts_v4.py's module docstring) read as too vibrant/artificial
        # per direct feedback on the flux2dev smoke test. Same caveat applies
        # here: reverting the colour brings back the low-contrast condition
        # the fix targeted. One variant changed to muted blue-grey (kept in
        # sync with prompts_v5.py) - real ground-truth feedback that DUO
        # includes blue starfish, not a contrast hack; kept muted rather
        # than vivid blue to match COLOR_PALETTE_GUARD below.
        "morphology": [
            "a small starfish, five arms, mottled brown-grey, rough texture",
            "a small five-armed starfish, muted brown-grey, irregular darker patches",
            "a small five-armed starfish, dark brown-grey mottling, irregular arm proportions",
            "a small starfish, five broad arms, muted reddish-brown and grey",
            "a small starfish, five arms, muted blue-grey, rough texture",
        ],

        # Dropped "blending into the substrate" / "naturally camouflaged" -
        # explicit low-contrast instructions implicated in the fix above.
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
# SCENE / ENVIRONMENT (unchanged from prompts_v3.py)
# ============================================================================

# Expanded from the original 5, but kelp/reef entries were removed and the
# silty entry reworded per direct ground-truth feedback on the real DUO
# dataset: "primary scene environment ... is sandy, small rocks here and
# there, boulders - all in sandy, beige or brownish tones ... no kelp
# fields, corals in DUO datasets." "Occasional shell fragments" was also
# independently found causing false-positive scallop detections (flux2dev_v4
# full run row 46: 0 scallops requested, 16 detected) - dropped for both
# reasons. 7 templates - all sandy/rocky/boulder terrain, matching DUO.
SCENE_TEMPLATES = [
    "temperate coastal seabed - sand, coarse sediment, gravel, irregular rocks, shallow ledges",
    "open sandy plain - fine sand, only occasional scattered pebbles, very little exposed rock",
    "dense boulder field - large rounded boulders close together, narrow sediment gaps between them",
    "silty low-lying flat - fine beige sediment, sparse small rocks",
    "cobble and shingle bank - uniform fist-sized rounded stones covering most of the seabed",
    "eroded rock ledge and wall - a low rocky wall rising from the seabed, sediment at its base",
    "mixed rubble seabed - broken rock fragments, gravel and sand in irregular patches",
]

# Dropped "pink coralline algae" (reads as coral - not in DUO) and "loose
# kelp debris" (kelp - not in DUO), same ground-truth feedback as above. The
# remaining 4 stay within DUO's actual beige/brown/muted-green palette.
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
# ECOLOGICAL / COMPOSITIONAL CONDITIONS (unchanged from prompts_v3.py)
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

# Expanded from the original 4, but "golden low-angle light" was dropped -
# warm/dramatic lighting pushes toward the "too vibrant" look corrected
# below (ground-truth DUO feedback: "current images ... are a bit too
# vibrant and colourful"). 6 remaining - diffuse/soft/overcast/dappled
# brightness variation without a colour-temperature swing. Only varies
# light quality/angle, not visibility - Stage 3 still owns turbidity,
# nothing here contradicts SCENE_WATER_PHRASE.
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
#
# COMPOSITION_GUARD/SPECIES_GUARD/OPTICAL_GUARD/BIVALVE_GUARD/
# FRAMING_COUNT_ANCHOR unchanged from prompts_v3.py. REALISM_GUARD expanded
# (thread 1). PRODUCT_PHOTO_GUARD and GROUND_CONTACT_GUARD are new (thread
# 2). No length budget to respect here (see module docstring), so no need to
# keep these as tight as prompts_v2/v3's versions.

COMPOSITION_GUARD = "natural asymmetric spacing, no decorative symmetry or cloned objects"
SPECIES_GUARD = "only these organisms and seabed material in frame, no other animals"

# Thread 1 fix: explicitly excludes the survey vehicle/camera rig itself,
# not just divers/boats/CG. prompts_v2/v3's version never named the
# equipment as something to exclude, which is the gap that let a robotic
# camera rig hallucinate into frame (5-pilot/hunyuan images 16, 19).
REALISM_GUARD = (
    "plain documentary survey photograph, not a posed wildlife photograph or "
    "product photograph; no divers, boats, or CG rendering; no robotic arms, "
    "thrusters, camera housing, cables, or any other part of the survey "
    "vehicle visible anywhere in frame"
)

# Thread 2 fix, part A: rules out the specimen-display/catalogue-photo
# framing directly (the "product photography" complaint).
PRODUCT_PHOTO_GUARD = "candid in-situ ecological documentation, not a specimen display or catalogue photograph"

# Thread 2 fix, part B: explicit ground-contact requirement. Nothing in
# prompts_v2/v3 ever said organisms must visibly touch the substrate -
# arrangement text describes position ("resting flat against rock") but not
# contact/shadow, which is exactly the gap behind "hovering a little bit
# from the ground."
GROUND_CONTACT_GUARD = "every organism rests directly on the substrate with genuine physical contact and a soft contact shadow, never elevated or hovering above the seabed"

# New, direct fix for ground-truth DUO feedback: "current images ... are a
# bit too vibrant and colourful" (real dataset is sandy/beige/brownish
# tones throughout). Nothing before this constrained overall colour
# saturation, only individual-class colour words. Applied unconditionally.
COLOR_PALETTE_GUARD = "muted natural colour palette, beige, tan and brown tones, no vivid or saturated colour grading"

OPTICAL_GUARD = "full frame, no fisheye or vignette"

BIVALVE_GUARD = "bivalve shells fully closed and undisturbed"

FRAMING_COUNT_ANCHOR = "count is a strict total for the frame, not per unit area"


# ============================================================================
# FRAMING
# ============================================================================
#
# mid/wide unchanged. close-up reworded (thread 2 fix, part C): keeps the
# close distance the manifest actually asks for on close-up rows, but adds
# an explicit anti-macro/anti-product-shot qualifier that prompts_v2/v3's
# version lacked.

FRAMING = {
    "close-up": "close survey framing at natural working distance, camera near the seabed but keeping full ecological context, not a tight macro product shot; objects at different distances",
    "mid": "mid-distance framing, foreground and mid-ground objects visible",
    "wide": "wide framing, larger section of seabed, objects at different depths",
}


# ============================================================================
# OBJECT COUNT RANGES (unchanged from prompts_v3.py)
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
# RANDOM HELPERS (unchanged from prompts_v3.py)
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
# CLASS PHRASE GENERATION (unchanged from prompts_v3.py)
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
# SCENE OBJECT GENERATION (unchanged from prompts_v3.py)
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

    Same rng draw sequence/order as prompts_v2.py/prompts_v3.py, so a given
    seed selects the same environment/subject content across all three
    prompt engines - only the surrounding scaffolding text differs (see
    module docstring for what changed and why: first-person POV + equipment
    exclusion, product-photography guard, ground-contact guard, reworded
    close-up framing).

    Assembly order follows Hunyuan's own recommended prompt structure (main
    subject/scene -> image quality/style -> composition/perspective ->
    lighting/atmosphere -> technical parameters) rather than prompts_v2/v3's
    FLUX.2-truncation-safe ordering, since there's no hard token cap to
    protect against here.
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
    # Select scene components (same rng sequence order as prompts_v2/v3, so
    # a given seed draws the same environment even though the assembled
    # sentences below differ)
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
    # Build subject descriptions (unchanged from prompts_v3.py)
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

    opening = (
        "Photorealistic underwater seabed photograph, captured in first-person "
        "point-of-view from a stationary benthic survey camera positioned just "
        "above the seafloor, as though the viewer is looking directly through "
        f"the camera's own lens. Scene contains: {subjects}."
    )

    guard_parts = [COMPOSITION_GUARD, SPECIES_GUARD, REALISM_GUARD, PRODUCT_PHOTO_GUARD, GROUND_CONTACT_GUARD, COLOR_PALETTE_GUARD, OPTICAL_GUARD]
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
