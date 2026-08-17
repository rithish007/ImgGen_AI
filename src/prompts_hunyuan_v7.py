"""
prompts_hunyuan_v7.py
======================
One fix on top of v6: splits each class's single "placement" list into
"placement_solo" (used when count==1) and "placement_group" (used
otherwise), the same fix already applied once elsewhere in this project.

THE BUG: v6's CLASSES entries had one undifferentiated "placement" list per
class, picked at random regardless of requested count. Auditing the actual
phrases found most are plural-implying:
  - starfish: "distributed across...", "...in some locations" (2 of 5 risky)
  - sea urchin: "occurring in small numbers...", "...some individuals partly
    obscured", "clustered locally..." (4 of 5 risky - only "partly tucked
    against uneven rock surfaces" was singular-safe)
  - scallop: "...with unequal spacing between individuals" (a direct
    self-contradiction on count=1 - there is only one individual), plus
    "distributed sparsely...", "...scattered locations" (3 of 5 risky)

This is the exact same bug this project already found and fixed once,
documented in prompts.py's v2 rewrite: a single undifferentiated
arrangement/placement list caused count=1 rows to describe multiple
individuals while only one was requested, measured at the time as a
2.5x-6.44x count overshoot specifically on count=1 rows (sea urchin and
scallop respectively). v6 reintroduced the same shape of bug by moving away
from the solo/group split without carrying the split itself forward - not
run at meaningful scale yet, so no fresh measurement of the overshoot this
time, but the mechanism is identical and already proven to matter.

THE FIX: split "placement" into "placement_solo" / "placement_group" per
class, gated on count in class_phrase() exactly like arrangements_solo/
arrangements_group already work in prompts_hunyuan_v2.py through v5.py.
Sea urchin only had one genuinely singular-safe phrase in v6, so two new
solo-safe variants were added (matching the existing vocabulary/style) to
avoid every count=1 urchin row reading identically.

Also fixed in passing: a cosmetic double-space bug in the prompt assembly
(atmosphere_block had a trailing space that " ".join() then duplicated) -
harmless to the model, just untidy.

NOT fixed here, left as open findings from the v6 audit, not yet acted on:
- v6 dropped v2-v5's REALISM_GUARD/OPTICAL_GUARD-equivalent phrasing (no
  divers/boats/CG, no fisheye/vignette) as part of its "no exclusion-heavy
  wording" design principle. The 6-image test6 run showed a real fisheye
  distortion in one image (row 6) - possibly related, not confirmed at
  n=6. Flagging, not fixing, until there's a decision on whether to bring
  back a positive-framed equivalent.
- FRAMING_COUNT_ANCHOR's original trigger (framing="wide" + density="sparse"
  causing 2-11x overshoot) may not apply the same way now that wide framing
  text no longer exists in the prompt at all, but nothing explicitly
  replaced that reinforcement for sparse-density scenes generally.

NOT YET RUN AT FULL SCALE. test6 (v6, before this fix) confirmed the
camera-distance/habitat-first rewrite genuinely fixes the "product
photography" complaint - that mechanism is untouched here. This file only
changes placement-text count-safety; needs its own smoke test and full run
to confirm the fix holds.

--- prompts_hunyuan_v6.py's own docstring follows for everything this file
didn't touch ---

HunyuanImage-3.0 prompt engine for synthetic underwater benthic survey imagery.

v6 is a structural rewrite of v5. The goal is not to add more negative guards;
it changes the semantic hierarchy of the prompt so Hunyuan treats the seabed
habitat as the primary scene and the requested organisms as naturally occurring
parts of that habitat.

Design principles
-----------------
1. Habitat first, organisms second.
2. Camera geometry is fixed to an elevated 5-8 m seabed distance.
3. No random close/mid/wide framing variable. The old framing vocabulary could
   contradict the mandatory survey distance and encourage specimen-like shots.
4. Ecological placement is generated separately from morphology. Organisms are
   described as distributed through terrain rather than staged around rocks.
5. Avoid exclusion-heavy wording that names unwanted visual concepts. v2/v3
   showed that explicit lists of unwanted equipment could backfire, so v6 uses
   positive scene descriptions instead.
6. The image should contain a large continuous habitat with a substantial water
   column between camera and seabed. No single organism is the visual anchor.
7. Keep class-count semantics compatible with the existing manifest format.

This file is intentionally written for HunyuanImage-3.0 Pretrain. It does not
rely on automatic prompt recaptioning.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Optional


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================
#
# v7 (this file): "placement" split into placement_solo (count==1) /
# placement_group (count>1) - see module docstring. morphology unchanged.

CLASSES: dict[int, dict[str, object]] = {
    0: {
        "duo_label": "starfish",
        "short": "starfish",
        "morphology": [
            "a small five-armed starfish with muted brown-grey mottling and a rough natural surface",
            "a small five-armed starfish with irregular darker patches and subdued brown-grey coloration",
            "a small starfish with five broad arms, muted reddish-brown and grey tones, and a textured surface",
            "a small five-armed starfish with irregular arm proportions and dark brown-grey mottling",
            "a small five-armed starfish with subdued blue-grey and brown-grey variation and a rough surface",
        ],
        "placement_solo": [
            "occurring on exposed sediment within the surrounding habitat",
            "resting naturally on uneven seabed terrain with part of the body blending into nearby substrate",
        ],
        "placement_group": [
            "occurring on exposed sediment within the surrounding habitat",
            "distributed across sediment-rock transition zones",
            "resting naturally on uneven seabed terrain with part of the body blending into nearby substrate",
            "occurring beside small pieces of rock and algae without a deliberate spatial arrangement",
            "partly obscured by nearby substrate in some locations",
        ],
    },
    1: {
        "duo_label": "echinus",
        "short": "sea urchin",
        "morphology": [
            "a low flattened sea urchin with dense short dark brown-black spines",
            "a nearly black low-domed sea urchin with dense short spines",
            "a low wide sea urchin with tightly packed short dark spines and a subdued natural surface",
            "a dark brown flattened sea urchin with dense short spines and a compact body",
        ],
        # v6 had only one genuinely singular-safe phrase ("partly tucked
        # against uneven rock surfaces") - two new solo variants added here,
        # matching the existing vocabulary, so count=1 rows aren't all
        # identical.
        "placement_solo": [
            "partly tucked against uneven rock surfaces within the terrain",
            "wedged into a rocky crevice within the surrounding terrain",
            "resting against a stable rock surface with spines visible against the substrate",
        ],
        "placement_group": [
            "occurring in small numbers around stable rocky substrate",
            "distributed irregularly along rock-sediment boundaries",
            "partly tucked against uneven rock surfaces within the terrain",
            "occurring at different positions around low rocky areas, with some individuals partly obscured",
            "clustered locally only where the available rocky substrate provides suitable shelter",
        ],
    },
    2: {
        "duo_label": "scallop",
        "short": "scallop",
        "morphology": [
            "a closed ribbed fan-shaped scallop shell with muted cream-brown coloration and a weathered natural surface",
            "a small closed scallop with strong radial ribs, subdued cream-brown tones and traces of sediment",
            "a closed textured scallop shell with pale tan and dull reddish-brown variation and natural wear",
            "a ribbed fan-shaped scallop with muted beige-brown coloration, partly carrying fine sediment",
        ],
        "placement_solo": [
            "partly embedded in open sediment at an irregular location",
            "partly buried by sediment so that the local terrain remains visually continuous",
        ],
        "placement_group": [
            "partly embedded in open sediment at an irregular location",
            "distributed sparsely across exposed sediment patches",
            "occurring along sand-gravel transitions with unequal spacing between individuals",
            "partly buried by sediment so that the local terrain remains visually continuous",
            "appearing at scattered locations across suitable open seabed areas",
        ],
    },
}


# ============================================================================
# HABITAT CONSTRUCTION (unchanged from prompts_hunyuan_v6.py)
# ============================================================================

HABITAT_TEMPLATES = [
    "a temperate coastal seabed of sand, gravel, coarse sediment and irregular low rocks",
    "an open sandy seabed with broad sediment areas, occasional rounded stones and low rock patches",
    "a mixed sand-and-rock seabed with irregular exposed rock and gravel patches",
    "a gently uneven silty seabed with shallow depressions, sparse stones and exposed substrate",
    "a cobble seabed with fine sediment accumulated between rounded stones",
    "a low eroded rocky shelf crossing the seabed with sediment around its edges",
    "a mixed rubble habitat of broken rock, gravel, sand and scattered larger stones",
    "a gently sloping seabed transitioning between sand, gravel and small rocky outcrops",
    "a patchy coastal habitat with open sediment, scattered rocks, shallow depressions and low uneven terrain",
]

TOPOGRAPHY_VARIATIONS = [
    "shallow depressions interrupt the otherwise gently uneven sediment surface",
    "low rock edges create gradual substrate transitions",
    "fine sediment accumulates in low areas and around rocks",
    "erosion leaves irregular exposed patches of coarse substrate between smoother sediment areas",
    "small stones form loose, discontinuous concentrations rather than deliberate piles",
    "low ridges and shallow troughs create subtle changes in seabed elevation across the frame",
    "sediment thickness varies naturally around rocks, gravel and small depressions",
    "the terrain changes gradually across the view with no sharply bounded scene elements",
    "rock, gravel and sediment interleave across the seabed",
]

ALGAE_VARIATIONS = [
    "sparse turf algae attached mainly to stable exposed rock surfaces",
    "patchy green-brown algae concentrated around sheltered rocky areas",
    "a thin muted biological film across some sediment and rock surfaces",
    "very sparse algae with mostly bare sediment and exposed rock",
    "low algae growth following the contours of stable rocks and substrate transitions",
    "patches of subdued brown-green growth separated by broad areas of bare seabed",
]

MICROSUBSTRATE_VARIATIONS = [
    "fine and coarse sediment grains mix gradually across the seabed",
    "small gravel is scattered irregularly through the sand",
    "fine sediment accumulates between larger stones and around low rock edges",
    "coarse patches interrupt otherwise fine sediment without geometric boundaries",
    "small pebbles and fragments are distributed unevenly across open sediment",
    "local sediment texture changes from compact fine material to looser coarse grains",
]

BACKGROUND_DEBRIS_VARIATIONS = [
    "occasional isolated cobbles occur at irregular intervals across the habitat",
    "a few larger stones sit independently within the sediment",
    "broken rock fragments occur naturally within the substrate",
    "small pebbles and coarse debris are scattered with unequal spacing",
    "isolated low rocks break up otherwise open sections of seabed",
    "debris follows the underlying terrain",
]


# ============================================================================
# ECOLOGICAL CONDITIONS (unchanged from prompts_hunyuan_v6.py)
# ============================================================================

SCENE_DENSITIES = {
    "sparse": "a relatively open seabed with large areas of exposed substrate and low biological clutter",
    "moderate": "a naturally varied seabed with rocks, sediment, algae and organisms distributed across broad areas",
    "dense": "a biologically and physically varied seabed with frequent rocks, substrate changes and local organism clusters while retaining visible open habitat",
}

DETECTION_DIFFICULTY = {
    "easy": "most requested organisms are visible against nearby substrate with limited natural occlusion",
    "moderate": "some organisms are partly obscured by terrain, sediment or algae and others blend into similar-toned substrate",
    "hard": "several organisms are small within the broad view, partly obscured or close in colour and texture to the surrounding seabed",
}

ECOLOGICAL_FIELD_PATTERNS = [
    "organisms are spread across the habitat with substantial variation in spacing and image position",
    "organisms occur at different lateral positions and ranges across the continuous seabed",
    "individuals are distributed irregularly across suitable microhabitats rather than concentrated at the centre of the frame",
    "the biological community is spatially uneven, with some open areas and some small local concentrations",
    "organisms are embedded within the wider terrain and occur at different distances from one another",
    "distribution follows suitable sediment and rocky microhabitats",
]

GROUP_BEHAVIOUR = [
    "where several individuals of one class occur, spacing remains irregular and visibility varies naturally",
    "multiple individuals occupy different positions and ranges according to the surrounding terrain",
    "same-class individuals follow the local terrain with varied spacing and orientation",
    "local clusters remain small and environmentally plausible within surrounding open seabed",
]


# ============================================================================
# CAMERA / IMAGE GEOMETRY (unchanged from prompts_hunyuan_v6.py)
# ============================================================================

CAMERA_HEIGHTS = {
    "far_5": "camera approximately 5 meters above the seabed",
    "far_6": "camera approximately 6 meters above the seabed",
    "far_7": "camera approximately 7 meters above the seabed",
    "far_8": "camera approximately 8 meters above the seabed",
}

CAMERA_PITCHES = [
    "view directed gently downward across the seabed",
    "slightly downward-looking survey view",
    "broad oblique-downward view of the seabed",
    "moderate downward pitch with the seabed extending continuously through the image",
]

CAMERA_GEOMETRY = [
    "a substantial water column separates the camera from the seabed",
    "organisms occupy a small fraction of the image",
    "the view covers a broad continuous patch of seabed",
    "wide environmental context surrounds every organism",
]

SURVEY_HEADINGS = [
    "view aligned roughly along the natural direction of the seabed contours",
    "view rotated slightly across the local terrain rather than directly centred on one feature",
    "view oriented across a broad section of habitat with no single central focal area",
    "view oriented diagonally across the substrate to reveal spatial variation through the scene",
]

IMAGE_SCALE = [
    "organisms remain small relative to the overall frame",
    "the surrounding seabed occupies much more visual area than the organisms",
    "no organism dominates the composition; environmental context remains visually primary",
    "open habitat surrounds the organisms",
]


# ============================================================================
# ATMOSPHERE / WATER / IMAGING (unchanged from prompts_hunyuan_v6.py)
# ============================================================================

WATER_CONDITIONS = [
    "clear blue-green water with realistic distance-dependent attenuation",
    "natural blue-green underwater water column with mild suspended particulate matter",
    "temperate underwater water with realistic colour loss and gentle particulate scattering",
    "moderately clear seawater with subtle green-blue attenuation through distance",
    "natural underwater visibility with slight haze increasing gradually with range",
]

LIGHTING_CONDITIONS = [
    "diffuse natural daylight filtered through the water column with soft uneven illumination",
    "soft underwater daylight with restrained contrast and subtle brightness variation across the seabed",
    "natural daylight attenuation with gentle highlights on exposed terrain and muted shadow transitions",
    "weak diffuse daylight with mild blue-green atmospheric influence and soft terrain shading",
    "subtle shallow-water light variation across the seabed without theatrical illumination",
]

IMAGING_CONDITIONS = [
    "natural camera exposure, restrained contrast, subtle sensor noise and mild distance-related detail loss",
    "realistic underwater imaging with slight softness in distant terrain and fine sensor noise",
    "natural photographic response with modest dynamic range, subtle particulate softness and realistic exposure",
    "light natural image noise, gradual contrast reduction with distance and restrained sharpness",
]

PERSPECTIVE_VARIATIONS = [
    "continuous natural spatial depth across the seabed",
    "uneven asymmetric terrain filling the frame without deliberate graphic composition",
    "environment-led composition with no isolated hero subject",
    "broad field survey composition following the geometry of the terrain",
]


# ============================================================================
# POSITIVE-ONLY GUARDS (unchanged from prompts_hunyuan_v6.py)
# ============================================================================

SURVEY_IDENTITY = (
    "authentic benthic habitat survey image from an elevated underwater inspection camera"
)

HABITAT_PRIORITY = (
    "the seabed habitat is the primary visual subject, with organisms naturally embedded within it"
)

SPATIAL_CONTINUITY = (
    "one continuous seabed spans the image with gradual transitions between sediment, rock, gravel and algae"
)

NATURAL_PLACEMENT = (
    "organisms follow the underlying terrain and remain supported by the substrate"
)

NO_HERO_SUBJECT = (
    "visual emphasis remains on the habitat and its spatial context rather than any one organism"
)

COLOR_MATERIALS = (
    "sediment, rock and biological surfaces use muted natural underwater tones with restrained saturation"
)

SPECIES_CONTROL = (
    "the requested classes define the visible biological subjects within the natural seabed habitat"
)

BROAD_CONTEXT = (
    "broad environmental context surrounds each organism, with open seabed visible between them"
)


# ============================================================================
# METADATA
# ============================================================================

@dataclass
class PromptMetadata:
    seed: int
    density: str
    difficulty: str
    camera_height_key: str
    camera_height_m: int
    pitch_index: int
    heading_index: int
    habitat_index: int
    topography_index: int
    algae_index: int
    microsubstrate_index: int
    debris_index: int
    ecological_field_index: int
    group_behaviour_index: int
    water_index: int
    lighting_index: int
    imaging_index: int
    perspective_index: int
    image_scale_index: int
    class_counts: dict[int, int]
    legacy_framing: Optional[str]


# ============================================================================
# RANDOM HELPERS (unchanged from prompts_hunyuan_v6.py)
# ============================================================================

def _choose(rng: random.Random, values: list[str]) -> tuple[str, int]:
    index = rng.randrange(len(values))
    return values[index], index


def _choose_key(rng: random.Random, mapping: dict[str, str]) -> tuple[str, str]:
    key = rng.choice(list(mapping.keys()))
    return key, mapping[key]


def _random_count(rng: random.Random, class_id: int, density: str) -> int:
    low, high = COUNT_RANGES[class_id][density]
    return rng.randint(low, high)


def _drop_leading_article(text: str) -> str:
    for article in ("an ", "a "):
        if text.startswith(article):
            return text[len(article):]
    return text


# ============================================================================
# COUNT RANGES / CLASS FIELD GENERATION
# ============================================================================

COUNT_RANGES = {
    0: {"sparse": (1, 2), "moderate": (1, 3), "dense": (2, 4)},
    1: {"sparse": (1, 2), "moderate": (2, 4), "dense": (3, 6)},
    2: {"sparse": (1, 2), "moderate": (2, 4), "dense": (3, 6)},
}


def class_phrase(class_id: int, count: int, rng: random.Random) -> str:
    """
    v7 (this file): picks from placement_solo when count==1, placement_group
    otherwise - see module docstring for why this split exists (a single
    undifferentiated list produced singular/plural contradictions that
    caused a measured 2.5x-6.44x count overshoot on count=1 rows the first
    time this bug existed, in prompts.py before the v2 rewrite).
    """
    entry = CLASSES[class_id]
    morphology = rng.choice(entry["morphology"])
    placement = rng.choice(
        entry["placement_solo"] if count == 1 else entry["placement_group"]
    )

    if count == 1:
        return f"{morphology}, {placement}"

    morphology_short = _drop_leading_article(morphology)
    return f"{count} {entry['short']} individuals, {morphology_short}, {placement}"


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
    Construct a habitat-first HunyuanImage-3.0 prompt.

    `framing` is retained only for manifest/API compatibility with older
    generators. It is not used to control image composition because v6 makes
    the 5-8 m survey geometry a hard scene property.

    `camera_height` may be one of far_5/far_6/far_7/far_8. For compatibility,
    the legacy value "far" is accepted and sampled from the four survey heights.
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

    if camera_height in (None, "far"):
        camera_height_key, camera_height_text = _choose_key(rng, CAMERA_HEIGHTS)
    elif camera_height in CAMERA_HEIGHTS:
        camera_height_key = camera_height
        camera_height_text = CAMERA_HEIGHTS[camera_height]
    else:
        raise ValueError(
            f"Invalid camera height {camera_height!r}. Expected one of {list(CAMERA_HEIGHTS)} or 'far'"
        )

    # Select scene variables in a stable order so seeds remain reproducible.
    habitat, habitat_idx = _choose(rng, HABITAT_TEMPLATES)
    topography, topography_idx = _choose(rng, TOPOGRAPHY_VARIATIONS)
    algae, algae_idx = _choose(rng, ALGAE_VARIATIONS)
    microsubstrate, microsubstrate_idx = _choose(rng, MICROSUBSTRATE_VARIATIONS)
    debris, debris_idx = _choose(rng, BACKGROUND_DEBRIS_VARIATIONS)
    ecological_field, ecological_field_idx = _choose(rng, ECOLOGICAL_FIELD_PATTERNS)
    group_behaviour, group_behaviour_idx = _choose(rng, GROUP_BEHAVIOUR)
    pitch, pitch_idx = _choose(rng, CAMERA_PITCHES)
    heading, heading_idx = _choose(rng, SURVEY_HEADINGS)
    camera_context = rng.choice(CAMERA_GEOMETRY)
    water, water_idx = _choose(rng, WATER_CONDITIONS)
    lighting, lighting_idx = _choose(rng, LIGHTING_CONDITIONS)
    imaging, imaging_idx = _choose(rng, IMAGING_CONDITIONS)
    perspective, perspective_idx = _choose(rng, PERSPECTIVE_VARIATIONS)
    image_scale, image_scale_idx = _choose(rng, IMAGE_SCALE)

    # Build class descriptions only after the environment has been selected.
    subject_phrases: list[str] = []
    for class_id in sorted(counts):
        if class_id not in CLASSES:
            raise ValueError(f"Unknown class ID {class_id}. Expected IDs: {list(CLASSES)}")
        count = counts[class_id]
        if count < 1:
            raise ValueError(f"Class {class_id} has invalid count {count}. Counts must be >= 1.")
        subject_phrases.append(class_phrase(class_id=class_id, count=count, rng=rng))

    if len(subject_phrases) == 1:
        organism_field = subject_phrases[0]
    elif len(subject_phrases) == 2:
        organism_field = f"{subject_phrases[0]}; {subject_phrases[1]}"
    else:
        organism_field = "; ".join(subject_phrases)

    # ----------------------------------------------------------------------
    # Prompt hierarchy:
    # habitat -> camera geometry -> habitat microstructure -> organisms ->
    # ecology/composition -> water/light -> imaging.
    # ----------------------------------------------------------------------
    opening = f"{SURVEY_IDENTITY}. {HABITAT_PRIORITY}. {SPATIAL_CONTINUITY}."

    habitat_block = (
        f"Seabed habitat: {habitat}. {topography}. {microsubstrate}. {algae}. {debris}."
    )

    camera_block = (
        f"Camera geometry: {camera_height_text}, {pitch}, {heading}. "
        f"{camera_context}. {image_scale}."
    )

    organism_block = (
        f"Organisms within this habitat: {organism_field}. "
        f"{SPECIES_CONTROL}. {NATURAL_PLACEMENT}. {ecological_field}."
    )

    ecology_block = (
        f"Ecological structure: {SCENE_DENSITIES[density]}. {DETECTION_DIFFICULTY[difficulty]}. "
        f"{group_behaviour}. {perspective}."
    )

    # v7: dropped the trailing space that was here in v6 - " ".join() below
    # was adding a second one, producing a double space before "Imaging:".
    atmosphere_block = (
        f"Water and light: {water}. {lighting}."
    )

    imaging_block = (
        f"Imaging: {imaging}. {COLOR_MATERIALS}."
    )

    prompt = " ".join(
        [opening, habitat_block, camera_block, organism_block, ecology_block, atmosphere_block, imaging_block]
    )

    # Resolve the exact height in metres for metadata.
    camera_height_m = int(camera_height_key.rsplit("_", 1)[-1])

    metadata = PromptMetadata(
        seed=seed,
        density=density,
        difficulty=difficulty,
        camera_height_key=camera_height_key,
        camera_height_m=camera_height_m,
        pitch_index=pitch_idx,
        heading_index=heading_idx,
        habitat_index=habitat_idx,
        topography_index=topography_idx,
        algae_index=algae_idx,
        microsubstrate_index=microsubstrate_idx,
        debris_index=debris_idx,
        ecological_field_index=ecological_field_idx,
        group_behaviour_index=group_behaviour_idx,
        water_index=water_idx,
        lighting_index=lighting_idx,
        imaging_index=imaging_idx,
        perspective_index=perspective_idx,
        image_scale_index=image_scale_idx,
        class_counts=dict(counts),
        legacy_framing=framing,
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
    counts = {0: 1, 1: 1, 2: 1}
    prompt, metadata = build_prompt(counts, seed=1, camera_height="far")
    print("--- count=1 sanity check (this is what v6 got wrong) ---")
    print(prompt)
    print()
    print(f"word count: {len(prompt.split())}")
    print()
    print(asdict(metadata))
