"""
prompt.py
=========
Prompt engine for AI-generated underwater synthetic datasets.

Purpose
-------
Generate scalable, varied prompts for photorealistic underwater robotic
survey imagery intended for object-detection dataset creation.

Target classes
--------------
Class IDs are kept fixed:

    0 -> starfish
    1 -> echinus / sea urchin
    2 -> scallop

NOTE: sea cucumber (DUO's holothurian, originally class 2) was removed after
repeated generation failures, and scallop was renumbered 3 -> 2 to keep IDs
contiguous for YOLO training (a gap at 2 either breaks training config or
wastes a class slot). This is now a DUO-DERIVED 3-CLASS SUBSET, not an exact
DUO match - a class_id of 2 here means scallop, not DUO's holothurian. Note
this explicitly in anything that compares results back to DUO.

Design philosophy
-----------------
Stage 1 should create:
    - ecological diversity
    - object morphology diversity
    - scene composition diversity
    - camera diversity
    - object-scale diversity
    - occlusion/clutter diversity

Stage 2 / domain randomization should primarily manipulate:
    - water colour
    - turbidity
    - suspended particles
    - contrast
    - haze
    - optical attenuation
    - sensor characteristics

Do NOT rely on the generator to create one particular "beautiful underwater
photography" aesthetic. The target distribution is underwater robotic survey
imagery.

The module supports:
    - deterministic random generation
    - parameterized prompt construction
    - models with negative-prompt support
    - models without negative-prompt support
    - metadata export for dataset manifests
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

# IMPORTANT:
# Do not renumber these IDs. They should remain consistent with the dataset
# annotation pipeline.

CLASSES: dict[int, dict[str, object]] = {

    0: {
        "duo_label": "starfish",
        "short": "starfish",

        "morphology": [
            (
                "a small living starfish with five arms, "
                "mottled brown and grey coloration, "
                "rough natural surface texture"
            ),
            (
                "a small five-armed starfish with muted brown-grey coloration, "
                "irregular darker patches and a rough textured surface"
            ),
            (
                "a small five-armed starfish with dark brown and grey mottling, "
                "subtle natural colour variation and irregular arm proportions"
            ),
            (
                "a small starfish with five broad arms, "
                "muted reddish-brown and grey coloration, "
                "natural irregular surface texture"
            ),
            (
                "a small dark brown-grey starfish with five arms, "
                "subtle blotchy markings and a naturally rough surface"
            ),
        ],

        "arrangements": [
            (
                "resting flat against rock and algae, partially blending into "
                "the substrate"
            ),
            (
                "resting beside a low rock ledge with portions of the body "
                "partially obscured"
            ),
            (
                "lying on mixed sediment near patches of algae, naturally "
                "camouflaged against the seabed"
            ),
            (
                "partially covered by fine sediment and resting against an "
                "irregular rock"
            ),
        ],
    },

    1: {
        "duo_label": "echinus",
        "short": "sea urchin",

        "morphology": [
            (
                "a living sea urchin with a low flattened dome-shaped body, "
                "dense short spines and very dark brown-black coloration"
            ),
            (
                "a dark grey-black sea urchin with a broad flattened dome-shaped "
                "body and dense short spines"
            ),
            (
                "a nearly black sea urchin with a low wide body, "
                "dense short dark spines and subtle natural surface variation"
            ),
            (
                "a dark brown sea urchin with a flattened low dome-shaped body "
                "and dense short spines"
            ),
        ],

        "arrangements": [
            (
                "clustered naturally inside a rocky crevice, with several "
                "individuals touching or partially overlapping"
            ),
            (
                "wedged tightly against a rock face inside a shallow crevice, "
                "with individuals at slightly different depths"
            ),
            (
                "grouped irregularly along a rocky ledge, with some individuals "
                "partially hidden behind rocks"
            ),
        ],
    },

    2: {
        # Renumbered from 3. Was DUO class_id 3; sea cucumber (DUO class_id 2,
        # "holothurian") was removed entirely after repeated generation
        # failures, so this ID shifted down to keep class IDs contiguous.
        "duo_label": "scallop",
        "short": "scallop",

        # "mostly closed with only a narrow edge of pale living tissue visible"
        # (previous wording) still rendered as a wide gaping shell showing a
        # large smooth pale interior - reads exactly like a plated/served
        # scallop dish. Rather than trying to constrain how far it opens,
        # the opening is removed from the description entirely: shells are
        # described as fully closed, exterior only, no interior/tissue
        # mentioned at all. A resting, undisturbed scallop is closed anyway -
        # this is not a less realistic description, it's a safer one.
        # 0.5-smoke audit (seed 42003, both models): the arrangement-only
        # burial cue ("partially buried at irregular locations") was ignored
        # 7/7 times - every rendered scallop across both models sat fully
        # exposed on top of the sand with zero sediment coverage. Moved the
        # cue into every morphology variant too (was only 2 of 4) so it's
        # stated twice per prompt regardless of which morphology gets drawn,
        # instead of depending on the arrangement clause alone.
        # No morphology variant specified shell colour at all (unlike
        # starfish/urchin, which both do) - left entirely to each model's own
        # prior, which is why flux2dev drew reddish-brown stripes and klein
        # drew near-white in the same 2-pilot run. Anchored to cream/light
        # brown (the real Pecten/Aequipecten range) in every variant below.
        "morphology": [
            (
                "a living scallop with a fan-shaped shell, cream to light "
                "brown in colour, tightly closed, clearly visible radiating "
                "ribs across the exterior, the lower edge dusted with fine "
                "sediment where it meets the seabed"
            ),
            (
                "a living fan-shaped scallop shell, cream to light brownish "
                "in colour, tightly closed, with strong radial ribs, "
                "partially buried in sediment"
            ),
            (
                "a small living scallop with a ribbed fan-shaped shell, "
                "light brown to cream in colour, tightly closed, naturally "
                "covered by a thin layer of sediment"
            ),
            (
                "a living scallop with a textured fan-shaped ribbed shell, "
                "cream or light brownish in colour, tightly closed, "
                "partially buried so that only part of the shell is exposed"
            ),
        ],

        "arrangements": [
            (
                "sparsely scattered across open sediment, with substantial "
                "irregular spacing between individuals"
            ),
            (
                "loosely distributed across sand and gravel, with individuals "
                "at different distances from the camera"
            ),
            (
                "partially buried at irregular locations across the seabed"
            ),
        ],
    },
}


# ============================================================================
# SCENE / ENVIRONMENT
# ============================================================================

SCENE_TEMPLATES = [

    (
        "temperate coastal seabed with a natural mixture of fine sand, "
        "coarse sediment, small gravel and irregular rocks, shallow depressions, "
        "low rocky ledges and occasional crevices"
    ),

    (
        "temperate coastal benthic habitat consisting of sandy sediment mixed "
        "with gravel and scattered irregular rocks, small depressions and "
        "natural rocky formations"
    ),

    (
        "natural temperate coastal seafloor with patches of fine sediment "
        "between scattered rocks, gravel, shallow grooves and small rocky "
        "crevices"
    ),

    (
        "mixed temperate marine substrate containing sand, gravel, small stones "
        "and irregular rocky patches with shallow crevices and uneven seabed "
        "topography"
    ),

    (
        "natural coastal seabed with exposed sandy areas, scattered rocks, "
        "small gravel deposits, shallow sediment depressions and irregular "
        "rock ledges"
    ),
]


ALGAE_VARIATIONS = [

    "sparse patches of natural turf algae attached to rocks",

    "small irregular patches of low algae growing across rocky surfaces",

    "sparse dark green and brown algae mixed with biological encrustation",

    "low natural algae and subtle biological growth attached to rocks",

    "small irregular algae-covered areas interspersed between exposed rock",
]


SUBSTRATE_VARIATIONS = [

    "subtle variation in sediment grain size and density",

    "fine sediment accumulating naturally around rocks",

    "small gravel mixed irregularly with fine sand",

    "patches of exposed rock surrounded by fine sediment",

    "slightly uneven sediment with small stones and natural debris",
]


# 2-pilot plan doc Q4 ("can the rock-formation pool be added to the prompt,
# and how?"), deliberately deferred out of manifests/2-pilot.json so that run
# tested only the bivalve-guard fix, not two changes at once. SCENE_TEMPLATES
# already gestures at rocks generically ("irregular rocks", "rocky ledges")
# and the large boulders/ledges already seen naturally in generated images are
# left alone - this pool is deliberately scoped to smaller-scale material
# (cobbles, rubble, pebbles, small boulders) as an independent axis on top of
# that, same pattern as ALGAE_VARIATIONS/SUBSTRATE_VARIATIONS, not a
# replacement for it. Not coupled to per-class arrangements yet (e.g.
# "resting beside a low rock ledge" doesn't reference which formation is
# actually in-scene) - a reasonable follow-up once this pool alone is
# verified not to break anything, not a day-one requirement.
ROCK_FORMATIONS = [

    "a cluster of rounded cobbles scattered loosely across the seabed",

    "a small pile of angular rubble formed from broken rock fragments",

    "a scatter of loose pebbles mixed into the surrounding sediment",

    "a group of small boulders resting on the seabed, spaced irregularly apart",

    "a mix of pebbles and small cobbles collected in a shallow depression",

    "a low pile of rubble and small boulders partially settled into the sediment",
]


# ============================================================================
# ECOLOGICAL / COMPOSITIONAL CONDITIONS
# ============================================================================

SCENE_DENSITIES = {

    "sparse": (
        "relatively open seabed with substantial exposed sediment between "
        "target objects and limited biological clutter"
    ),

    "moderate": (
        "moderately cluttered seabed with natural rocks, algae, sediment and "
        "target objects distributed across foreground and middle ground"
    ),

    "dense": (
        "visually cluttered natural seabed containing rocks, algae, gravel, "
        "sediment and overlapping environmental features, while maintaining "
        "realistic ecological structure"
    ),
}


DETECTION_DIFFICULTY = {

    "easy": (
        "target objects are mostly visible, with limited occlusion and "
        "moderate contrast against the surrounding substrate"
    ),

    "moderate": (
        "some target objects are partially obscured by rocks, algae or "
        "sediment, with moderate natural contrast and varying object sizes"
    ),

    "hard": (
        "several target objects are small or partially obscured by rocks, "
        "algae or sediment, with some objects naturally blending into the "
        "substrate and reduced contrast at greater distances"
    ),
}


# ============================================================================
# WATER CONDITIONS
# ============================================================================
#
# This used to be a randomly-chosen dict (clear / moderately_turbid / turbid /
# green_coastal / blue_coastal) injecting turbidity and colour-cast language
# directly into the Stage 1 prompt. That reopens the exact double-degradation
# bug fixed several rounds ago: Stage 3 (Akkaynak-Treibitz + Jerlov physics)
# needs clean scene radiance as its input. If Stage 1 already renders murk or
# a green cast, Stage 3 applies real optical physics on top of an image that
# is already degraded, and the two effects compound in a way that does not
# match any real water type. Water condition is Stage 3's job, full stop -
# Stage 1 always renders clear and colour-neutral. Kept as a single fixed
# phrase rather than a dict so there is no random selection to accidentally
# re-enable.
SCENE_WATER_PHRASE = (
    "clear seawater with good visibility, true-to-life natural colour and no "
    "artificial colour cast"
)


# ============================================================================
# LIGHTING CONDITIONS
# ============================================================================

LIGHTING_CONDITIONS = [

    (
        "natural underwater illumination from the surface, diffuse sunlight "
        "attenuated through the water column, soft uneven brightness across "
        "the seabed and subtle natural shadows"
    ),

    (
        "soft diffuse daylight filtered through the water column, with gentle "
        "brightness variation across the seabed and low-contrast natural shadows"
    ),

    (
        "natural daylight from above the water surface with realistic underwater "
        "attenuation, soft illumination and subtle directional brightness changes"
    ),

    (
        "weak diffuse underwater daylight with realistic attenuation and "
        "slightly uneven illumination across rocks and sediment"
    ),
]


# ============================================================================
# CAMERA CONDITIONS
# ============================================================================

CAMERA_HEIGHTS = {

    "low": (
        "camera approximately 0.5 metres above the seabed, looking slightly "
        "downward"
    ),

    "medium": (
        "camera approximately 1 metre above the seabed, looking slightly "
        "downward"
    ),

    "high": (
        "camera approximately 1.5 to 2 metres above the seabed, looking "
        "downward across the survey area"
    ),
}


CAMERA_FOV = [

    "moderately wide-angle field of view",

    "wide-angle underwater field of view",

    "natural wide field of view typical of a compact underwater survey camera",
]


CAMERA_MOTION = [

    "slight natural motion softness consistent with a moving underwater robot",

    "very mild motion blur consistent with slow robotic survey movement",

    "minimal motion softness from a forward-moving underwater camera",

    "stable robotic survey capture with only subtle sensor and motion effects",
]


# ============================================================================
# IMAGING / SENSOR CONDITIONS
# ============================================================================

IMAGING_CONDITIONS = [

    (
        "realistic underwater camera exposure, subtle sensor noise, "
        "natural optical response and mild reduction of fine detail with distance"
    ),

    (
        "realistic compact underwater camera characteristics, subtle image "
        "noise, natural exposure variation and mild distant detail loss"
    ),

    (
        "realistic digital underwater camera imagery with fine sensor noise, "
        "natural exposure and subtle loss of contrast with increasing distance"
    ),

    (
        "natural robotic-camera image characteristics with subtle sensor noise, "
        "realistic exposure and restrained optical softness"
    ),
]


# ============================================================================
# COMPOSITION
# ============================================================================

COMPOSITIONS = [

    (
        "candid marine survey photograph, documentary observation style, "
        "off-centre asymmetric framing and natural unposed composition"
    ),

    (
        "underwater robotic survey image, irregular asymmetric composition, "
        "natural spatial distribution of objects across the frame"
    ),

    (
        "unposed benthic survey image with foreground, middle-ground and "
        "background depth, natural off-centre framing"
    ),

    (
        "documentary-style underwater survey frame with no deliberate hero "
        "subject, natural ecological composition and uneven spatial distribution"
    ),
]


# ============================================================================
# OCCLUSION / DEPTH
# ============================================================================

DEPTH_DISTRIBUTIONS = [

    (
        "target objects occur at different distances from the camera, including "
        "foreground, middle-ground and background instances"
    ),

    (
        "objects have varied apparent sizes because of different distances "
        "from the camera"
    ),

    (
        "some objects are close to the camera while others are smaller and "
        "farther away in the scene"
    ),
]


# ============================================================================
# NEGATIVE PROMPTS
# ============================================================================

GLOBAL_NEGATIVE = (
    "text, watermark, logo, caption, human, diver, boat, submarine, "
    "water surface, sky, aquarium, fish tank, glass enclosure, "
    "tropical reef, coral reef, tropical fish, anemone, "
    "illustration, painting, drawing, cartoon, CGI, 3D render, "
    "artificial environment, studio photography, product photography, "
    "catalog photography, stock photography, cinematic underwater scene, "
    "dramatic spotlight, dramatic volumetric lighting"
)


COMPOSITION_NEGATIVE = (
    "centered subject, symmetrical composition, staged arrangement, "
    "posed wildlife, hero shot, isolated specimen, repeated objects, "
    "identical copies, cloned objects, grid arrangement, regular spacing, "
    "repeating pattern, artificial pattern, decorative arrangement"
)


OPTICAL_NEGATIVE = (
    "fisheye circle, circular distortion, circular vignette, black corners, "
    "black border, strong lens flare, excessive bloom, exaggerated light rays, "
    "hard-edged caustics, white geometric caustic patterns, polygonal light "
    "patterns, Voronoi patterns, artificial water texture"
)


BIOLOGICAL_NEGATIVE = (
    "cooked food, seafood dish, restaurant presentation, plate, kitchen, "
    "sashimi, cooked scallop meat, empty shell, dead shell, beach shell, "
    "shell litter, caterpillar, millipede, centipede, insect legs, larva, "
    "segmented insect body, articulated legs, "
    # "mostly closed with a narrow edge visible" still rendered as a wide
    # gaping shell showing smooth pale interior - a served-dish look. These
    # only help on a model with real negative-prompt support (not Klein);
    # the actual fix is describing the shell as closed with no interior
    # mentioned at all (see CLASSES[2]'s morphology entries).
    "gaping open shell, shucked shellfish, open bivalve interior, scallop meat, "
    "shell presentation, exposed shell interior"
)


NEGATIVE = ", ".join(
    [
        GLOBAL_NEGATIVE,
        COMPOSITION_NEGATIVE,
        OPTICAL_NEGATIVE,
        BIOLOGICAL_NEGATIVE,
    ]
)


# ============================================================================
# POSITIVE GUARDS
# ============================================================================
#
# These are used when the model does not support a negative prompt.

# Applies to every image regardless of requested classes.
POSITIVE_ONLY_GUARDS = (
    "true-to-life natural colour reproduction, realistic biological morphology, "
    "natural ecological habitat, full rectangular frame, no lens vignette, "
    "no artificial colour grading, realistic underwater camera imagery, "
    "natural irregular spatial distribution"
)

# Bug found via audit: this used to be baked into POSITIVE_ONLY_GUARDS above,
# which appends unconditionally to every prompt whenever supports_negative is
# False (always true for Klein - see build_prompt()). That meant every single
# image, including ones that never requested scallop, was explicitly told to
# render "bivalve shells... in their natural habitat" - and Klein complied by
# scattering small shell-shaped debris across the seabed as generic scene
# clutter. SAM3's scallop-concept pass then correctly detects these (they
# really do look like small closed scallop shells), which is why
# reports/class_counts.json showed scallop instances in far more images (17/20)
# than actually requested scallop (12/20): 33 of the 70 total SAM3-detected
# scallop instances across the pilot came from rows that requested zero
# scallops. Now only appended when scallop (class_id 2) is one of the row's
# requested classes.
BIVALVE_GUARD = (
    "bivalve shells fully closed and undisturbed as found in their natural habitat"
)


# ============================================================================
# FRAMING
# ============================================================================

FRAMING = {
    # Key is "close-up", not "close" - build_manifest.py's existing manifest
    # rows already use "close-up"/"mid" (from the original combinatorial pilot
    # design); renaming the dict key to match rather than regenerating every
    # manifest.
    "close-up": (
        "close survey framing, camera relatively near the seabed with several "
        "target objects visible at different distances"
    ),

    "mid": (
        "mid-distance survey framing, camera looking across the seabed with "
        "foreground and middle-ground objects"
    ),

    "wide": (
        "wide survey framing showing a larger section of seabed, with target "
        "objects distributed across different depths"
    ),
}


# 2-pilot audit (klein, manifests/2-pilot.json, 50 rows): every row combining
# framing="wide" with density="sparse" overshot its requested instance count
# badly (2x-11x measured; worst case row 20, 1 scallop requested -> 20
# detected), while every other framing/density combination stayed close to
# requested. SCENE_DENSITIES/arrangement text describes a spacing PATTERN
# ("sparsely scattered", "substantial irregular spacing"), not an absolute
# count, and "wide" framing exposes far more seabed area than close-up/mid -
# the model appears to apply that pattern per unit of visible area rather
# than honouring the explicit numeric count from class_phrase(). Scallop was
# worst hit because its arrangement text has no natural boundary ("scattered
# across open sediment"); starfish/urchin arrangements reference a specific
# rock/crevice, which caps them somewhat.
#
# Deliberately NOT dropping the wide+sparse combination: that would lose real
# scene-scale diversity and wouldn't fix anything, since the same weak
# numeracy is presumably still there at other density levels, just less
# visible. Reinforcing the absolute count only when framing is wide instead -
# UNVERIFIED as an actual fix (diffusion models have limited instruction-
# following for this kind of meta/negation instruction); re-run the specific
# rows that overshot in the 2-pilot (17, 19, 20, 21, 36, 39, 40, 41, 42) after
# this change and compare detected-vs-requested before trusting it at scale.
FRAMING_COUNT_ANCHOR = (
    "The stated number of individuals for each organism is a strict total "
    "for the entire frame, not a density to repeat across the visible "
    "seabed - a wider view must not add extra individuals beyond that total."
)


# ============================================================================
# OBJECT COUNT RANGES
# ============================================================================
#
# Counts are selected by the prompt generator.
#
# These ranges intentionally prevent every image from having identical object
# counts. The generator should record the selected count in the manifest.

COUNT_RANGES = {

    0: {
        "sparse": (1, 2),
        "moderate": (1, 3),
        "dense": (2, 4),
    },

    1: {
        "sparse": (1, 2),
        "moderate": (2, 4),
        "dense": (3, 6),
    },

    2: {  # scallop, renumbered from 3 - sea cucumber (was 2) removed entirely
        "sparse": (1, 2),
        "moderate": (2, 4),
        "dense": (3, 6),
    },
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PromptMetadata:
    """
    Metadata describing the synthetic scene requested by the prompt.

    This should be stored alongside the generated image.

    It is useful later for:
        - dataset auditing
        - train/validation splitting
        - reproducibility
        - domain-randomization analysis
        - dissertation methodology
    """

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
# RANDOM HELPERS
# ============================================================================

def _choose(rng: random.Random, values: list[str]) -> tuple[str, int]:
    """
    Select one item and return both its value and index.
    """
    index = rng.randrange(len(values))
    return values[index], index


def _random_count(
    rng: random.Random,
    class_id: int,
    density: str,
) -> int:
    """
    Generate a random instance count for a class.
    """
    low, high = COUNT_RANGES[class_id][density]
    return rng.randint(low, high)


# ============================================================================
# CLASS PHRASE GENERATION
# ============================================================================

def class_phrase(
    class_id: int,
    count: int,
    rng: random.Random,
) -> str:
    """
    Build a class description for the requested number of instances.

    The generator intentionally uses morphological descriptions rather than
    relying only on taxonomic names.
    """

    entry = CLASSES[class_id]

    morphology = rng.choice(entry["morphology"])
    arrangement = rng.choice(entry["arrangements"])

    if count == 1:
        quantity = "one"
    else:
        quantity = f"{count}"

    if count == 1:
        return (
            f"{morphology}, {arrangement}"
        )

    # We explicitly tell the generator that these are separate individuals.
    # Was morphology.replace('a ', '').replace('an ', ''), which strips EVERY
    # occurrence of "a "/"an " in the string, not just the leading article -
    # e.g. "a living sea urchin with a low flattened..." also loses the "a"
    # in "with a low", producing "with low flattened...". _drop_leading_article
    # only touches the front of the string.
    return (
        f"{quantity} separate living {entry['short']} individuals, "
        f"each showing {_drop_leading_article(morphology)}, "
        f"{arrangement}"
    )


def _drop_leading_article(text: str) -> str:
    """Strip only a genuine leading 'a ' or 'an ', not every occurrence."""
    for article in ("an ", "a "):
        if text.startswith(article):
            return text[len(article):]
    return text


# ============================================================================
# SCENE OBJECT GENERATION
# ============================================================================

def generate_class_counts(
    rng: random.Random,
    density: str,
    min_classes: int = 2,
    max_classes: int = 4,
) -> dict[int, int]:
    """
    Randomly select which target classes appear in a scene and how many
    instances of each are requested.

    Default:
        2 to 4 classes per scene.

    This is important because every generated frame does not need to contain
    every target class.
    """

    available = list(CLASSES.keys())

    number_of_classes = rng.randint(
        min_classes,
        min(max_classes, len(available)),
    )

    selected = rng.sample(available, number_of_classes)

    return {
        class_id: _random_count(rng, class_id, density)
        for class_id in sorted(selected)
    }


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
    supports_negative: bool = True,
) -> tuple[str, Optional[str], PromptMetadata]:
    """
    Construct a complete underwater survey image-generation prompt.

    Parameters
    ----------
    counts:
        Dictionary mapping class IDs to requested instance counts.

    seed:
        Random seed controlling all stochastic prompt choices.

    density:
        "sparse", "moderate", or "dense".

    difficulty:
        "easy", "moderate", or "hard".

    camera_height:
        "low", "medium", or "high".

    framing:
        "close", "mid", or "wide".

    supports_negative:
        True if the image model accepts a normal negative_prompt.
        False if exclusions need to be embedded into the positive prompt.

    Returns
    -------
    prompt:
        Positive generation prompt.

    negative_prompt:
        Negative prompt, or None for models without negative-prompt support.

    metadata:
        PromptMetadata object describing the generated scene.
    """

    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Validate / randomly choose high-level parameters
    # ------------------------------------------------------------------

    if density is None:
        density = rng.choice(list(SCENE_DENSITIES.keys()))

    if density not in SCENE_DENSITIES:
        raise ValueError(
            f"Invalid density {density!r}. "
            f"Expected one of {list(SCENE_DENSITIES)}"
        )

    if difficulty is None:
        difficulty = rng.choice(list(DETECTION_DIFFICULTY.keys()))

    if difficulty not in DETECTION_DIFFICULTY:
        raise ValueError(
            f"Invalid difficulty {difficulty!r}. "
            f"Expected one of {list(DETECTION_DIFFICULTY)}"
        )

    if camera_height is None:
        camera_height = rng.choice(list(CAMERA_HEIGHTS.keys()))

    if camera_height not in CAMERA_HEIGHTS:
        raise ValueError(
            f"Invalid camera height {camera_height!r}. "
            f"Expected one of {list(CAMERA_HEIGHTS)}"
        )

    if framing is None:
        framing = rng.choice(list(FRAMING.keys()))

    if framing not in FRAMING:
        raise ValueError(
            f"Invalid framing {framing!r}. "
            f"Expected one of {list(FRAMING)}"
        )

    # ------------------------------------------------------------------
    # Select scene components
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

    # ------------------------------------------------------------------
    # Build class descriptions
    # ------------------------------------------------------------------

    subject_phrases = []

    for class_id in sorted(counts):

        if class_id not in CLASSES:
            raise ValueError(
                f"Unknown class ID {class_id}. "
                f"Expected IDs: {list(CLASSES)}"
            )

        count = counts[class_id]

        if count < 1:
            raise ValueError(
                f"Class {class_id} has invalid count {count}. "
                "Counts must be >= 1."
            )

        subject_phrases.append(
            class_phrase(
                class_id=class_id,
                count=count,
                rng=rng,
            )
        )

    if len(subject_phrases) == 1:
        subjects = subject_phrases[0]
    elif len(subject_phrases) == 2:
        subjects = f"{subject_phrases[0]} and {subject_phrases[1]}"
    else:
        subjects = (
            ", ".join(subject_phrases[:-1])
            + ", and "
            + subject_phrases[-1]
        )

    # ------------------------------------------------------------------
    # Compose the final prompt
    # ------------------------------------------------------------------
    #
    # Global composition comes before detailed subjects so the model is
    # encouraged to establish a survey frame before rendering individual
    # organisms.

    prompt_parts = [

        # 1. GLOBAL IMAGE TYPE
        (
            "Photorealistic underwater robotic benthic survey image, "
            "realistic biological morphology and natural ecological habitat."
        ),

        # 2. ENVIRONMENT
        (
            f"Temperate coastal seabed consisting of {scene_template}. "
            f"The habitat contains {algae} and {substrate}. "
            f"{rock_formation.capitalize()} is present in the scene."
        ),

        # 3. COMPOSITION
        (
            f"{composition}. "
            f"{SCENE_DENSITIES[density]}. "
            f"{DETECTION_DIFFICULTY[difficulty]}."
        ),

        # 4. DEPTH / SPATIAL DISTRIBUTION
        # Was module-level random.choice(), bypassing the seeded rng - meant
        # the same seed could produce a different prompt on a re-run, breaking
        # the module's own "deterministic random generation" design goal.
        rng.choice(DEPTH_DISTRIBUTIONS) + ".",

        # 5. TARGET OBJECTS
        (
            f"The target organisms are naturally distributed within the habitat: "
            f"{subjects}."
        ),

        # 6. CAMERA
        (
            f"{CAMERA_HEIGHTS[camera_height]}, "
            f"{camera_fov}, "
            f"natural perspective."
        ),

        # 7. WATER OPTICS - always clear/neutral; see SCENE_WATER_PHRASE comment
        (
            f"{SCENE_WATER_PHRASE}."
        ),

        # 8. LIGHTING
        (
            f"{lighting}."
        ),

        # 9. MOTION
        (
            f"{camera_motion}."
        ),

        # 10. SENSOR
        (
            f"{imaging}."
        ),

        # 11. FRAMING
        (
            f"{FRAMING[framing]}."
            + (f" {FRAMING_COUNT_ANCHOR}" if framing == "wide" else "")
        ),

        # 12. FINAL REALISM ANCHOR
        (
            "The image should resemble a frame captured by a real underwater "
            "robot during an ecological survey rather than a posed wildlife "
            "photograph, cinematic scene or artificial 3D environment."
        ),
    ]

    if not supports_negative:
        guards = POSITIVE_ONLY_GUARDS
        if 2 in counts:  # scallop - see BIVALVE_GUARD's comment for why this must be conditional
            guards = f"{guards}, {BIVALVE_GUARD}"
        prompt_parts.append(guards)

    prompt = " ".join(prompt_parts)

    negative_prompt = NEGATIVE if supports_negative else None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

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

    return prompt, negative_prompt, metadata


# ============================================================================
# HIGH-LEVEL DATASET PROMPT GENERATOR
# ============================================================================

def generate_dataset_prompts(
    number_of_images: int,
    *,
    base_seed: int = 2026,
    supports_negative: bool = True,
    density_distribution: Optional[dict[str, float]] = None,
    difficulty_distribution: Optional[dict[str, float]] = None,
) -> list[dict[str, object]]:
    """
    Generate a complete list of prompts for a synthetic dataset.

    No manual prompting is required.

    Each returned item contains:

        image_id
        seed
        prompt
        negative_prompt
        metadata

    Example
    -------
    prompts = generate_dataset_prompts(
        1500,
        base_seed=1234,
    )

    The caller can then feed each prompt to the image-generation model.
    """

    if number_of_images < 1:
        raise ValueError("number_of_images must be >= 1")

    rng = random.Random(base_seed)

    # Default distributions.
    if density_distribution is None:
        density_distribution = {
            "sparse": 0.25,
            "moderate": 0.55,
            "dense": 0.20,
        }

    if difficulty_distribution is None:
        difficulty_distribution = {
            "easy": 0.25,
            "moderate": 0.55,
            "hard": 0.20,
        }

    def weighted_choice(
        distribution: dict[str, float],
    ) -> str:

        keys = list(distribution.keys())
        weights = list(distribution.values())

        return rng.choices(
            keys,
            weights=weights,
            k=1,
        )[0]

    results = []

    for index in range(number_of_images):

        # Each image receives a unique seed.
        seed = rng.randint(0, 2**31 - 1)

        density = weighted_choice(
            density_distribution
        )

        difficulty = weighted_choice(
            difficulty_distribution
        )

        camera_height = rng.choice(
            list(CAMERA_HEIGHTS.keys())
        )

        framing = rng.choice(
            list(FRAMING.keys())
        )

        counts = generate_class_counts(
            rng=rng,
            density=density,
        )

        prompt, negative_prompt, metadata = build_prompt(
            counts=counts,
            seed=seed,
            density=density,
            difficulty=difficulty,
            camera_height=camera_height,
            framing=framing,
            supports_negative=supports_negative,
        )

        results.append(
            {
                "image_id": f"synthetic_{index:05d}",
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "metadata": asdict(metadata),
            }
        )

    return results


# ============================================================================
# SIMPLE MANIFEST EXPORT
# ============================================================================

def save_prompt_manifest(
    prompts: list[dict[str, object]],
    output_path: str,
) -> None:
    """
    Save generated prompts and metadata to JSON.

    This manifest is extremely useful for reproducibility and dataset auditing.
    """

    import json

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            prompts,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================================
# DETECTOR PROMPTS
# ============================================================================

def detector_prompts() -> dict[int, str]:
    """
    Short concept prompts for a later detector / segmentation model.

    Keep these separate from the detailed image-generation descriptions.

    These are useful for:
        - SAM-style segmentation
        - Grounding DINO
        - open-vocabulary detection
        - annotation assistance
    """

    return {
        0: "starfish",
        1: "sea urchin",
        2: "scallop",
    }


# ============================================================================
# CLASS MAP
# ============================================================================

def class_names() -> dict[int, str]:
    """
    Return the fixed YOLO class mapping.
    """

    return {
        0: "starfish",
        1: "sea_urchin",
        2: "scallop",
    }


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":

    # Generate ten example prompts.
    prompts = generate_dataset_prompts(
        number_of_images=10,
        base_seed=2026,
        supports_negative=True,
    )

    for item in prompts[:3]:

        print("=" * 80)

        print(
            f"IMAGE: {item['image_id']}"
        )

        print(
            f"SEED: {item['seed']}"
        )

        print(
            "\nPROMPT:\n"
            f"{item['prompt']}"
        )

        print(
            "\nNEGATIVE:\n"
            f"{item['negative_prompt']}"
        )

        print(
            "\nMETADATA:\n"
            f"{item['metadata']}"
        )

        print()