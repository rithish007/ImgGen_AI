"""Prompt engine v7 - narrows COLOR_PALETTE_GUARD to substrate/organism materials only, on top of v6's camera-distance work."""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional


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


COMPOSITION_GUARD = "natural asymmetric spacing, no decorative symmetry or cloned objects"
SPECIES_GUARD = "only these organisms and seabed material in frame, no other animals"
REALISM_GUARD = "plain documentary robot photo, no divers, boats, or CG rendering"
OPTICAL_GUARD = "full frame, no fisheye or vignette"

COLOR_PALETTE_GUARD = "seabed, rock and shell material in muted natural tones, not vivid or saturated"

SUBJECT_SCALE_GUARD = "organisms small in the wide view"

BIVALVE_GUARD = "bivalve shells fully closed and undisturbed"

FRAMING_COUNT_ANCHOR = "count is a strict total for the frame, not per unit area"


FRAMING = {
    "close-up": "close survey framing at natural working distance, camera near the seabed but not a close macro product shot, objects at different distances",
    "mid": "mid-distance framing, foreground and mid-ground objects visible",
    "wide": "wide framing, larger section of seabed, objects at different depths",
}


COUNT_RANGES = {
    0: {"sparse": (1, 2), "moderate": (1, 3), "dense": (2, 4)},
    1: {"sparse": (1, 2), "moderate": (2, 4), "dense": (3, 6)},
    2: {"sparse": (1, 2), "moderate": (2, 4), "dense": (3, 6)},
}


@dataclass
class PromptMetadata:

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


def _choose(rng: random.Random, values: list[str]) -> tuple[str, int]:
    index = rng.randrange(len(values))
    return values[index], index


def _random_count(rng: random.Random, class_id: int, density: str) -> int:
    low, high = COUNT_RANGES[class_id][density]
    return rng.randint(low, high)


def _drop_leading_article(text: str) -> str:
    for article in ("an ", "a "):
        if text.startswith(article):
            return text[len(article):]
    return text


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


def build_prompt(
    counts: dict[int, int],
    *,
    seed: int = 0,
    density: Optional[str] = None,
    difficulty: Optional[str] = None,
    camera_height: Optional[str] = None,
    framing: Optional[str] = None,
) -> tuple[str, PromptMetadata]:

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


def class_names() -> dict[int, str]:
    return {cid: entry["short"] for cid, entry in CLASSES.items()}


def detector_prompts() -> dict[int, str]:
    return {cid: entry["short"] for cid, entry in CLASSES.items()}
