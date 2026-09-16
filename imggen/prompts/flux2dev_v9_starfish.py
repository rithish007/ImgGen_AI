"""Flux2Dev v9 - starfish-focused synthetic underwater prompt engine."""


from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


CLASSES: dict[int, dict[str, object]] = {


    0: {
        "duo_label": "starfish",
        "short": "starfish",

        "morphology": [
            "a small five-armed starfish, muted tan-brown and grey, irregular arm proportions, rough mottled surface",
            "a small brown-grey starfish with uneven five arms, broken mottled coloration and a dull granular surface",
            "a small low-contrast starfish, dusty tan-brown and grey, irregular arms, natural substrate staining",
            "a small five-armed starfish, dull brown-grey, rough textured body, slightly asymmetrical arms",
            "a small tan-grey starfish with darker mottling, irregular arm widths and a weathered rough surface",
            "a small subdued brown starfish, five irregular arms, patchy natural coloration, partially stained by sediment",
            "a small muted brown-grey starfish, uneven five-arm geometry, rough granular texture, low visual contrast",
            "a small starfish, dull tan-brown with grey patches, irregular arms, natural sediment adhering to the surface",
        ],

        "arrangements_solo": [
            "lying partly buried in fine sediment, with one or more arm tips obscured",
            "resting against low rock and partly covered by sediment",
            "partially hidden against a rock edge, blending into the surrounding substrate",
            "lying flat on mixed sand and gravel, slightly covered by loose sediment",
            "partly buried beside a small rock, with only part of the body clearly visible",
            "resting in a shallow sediment depression with its coloration blending into the seabed",
            "partially obscured by algae and loose sediment near a rock",
            "lying at an irregular angle on the seabed, difficult to distinguish from nearby substrate",
        ],

        "arrangements_group": [
            "sparsely scattered across sediment with natural spacing and different apparent sizes",
            "distributed at different distances, some partly obscured by rock or sediment",
            "several individuals partially buried at irregular locations across the seabed",
            "loosely distributed across mixed sand and gravel, blending into the surrounding substrate",
            "some individuals fully visible while others are partly covered by sediment or algae",
            "irregularly positioned on the seabed, with no repeated orientation or arrangement",
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
    "temperate coastal seabed with fine sand, coarse sediment, gravel and irregular low rocks",
    "open sandy seabed with scattered pebbles and small patches of exposed rock",
    "mixed sand and gravel seabed with shallow irregular rocks and natural sediment deposits",
    "silty low-lying seabed with small rocks, sediment depressions and scattered natural debris",
    "cobble and shingle seabed with irregular gaps containing fine sediment",
    "eroded low rock ledge with sediment accumulating around the base",
    "mixed rubble seabed with sand, gravel, small rocks and irregular sediment patches",
]

ALGAE_VARIATIONS = [
    "sparse low algae on exposed rock surfaces",
    "thin green-brown algae on rocks and small patches of substrate",
    "faint diatom film across parts of the sediment",
    "little visible algae, mostly bare rock and sediment",
]

SUBSTRATE_VARIATIONS = [
    "subtle variation in sediment grain size",
    "fine sediment accumulating around rocks and object edges",
    "small gravel mixed irregularly with sand",
    "exposed rock patches surrounded by fine sediment",
    "slightly uneven sediment with small stones and biological debris",
]

ROCK_FORMATIONS = [
    "rounded cobbles scattered nearby",
    "a small pile of angular rubble nearby",
    "loose pebbles mixed into the sediment",
    "small boulders resting irregularly nearby",
    "pebbles and small cobbles in a shallow depression",
    "a low pile of rubble and small boulders nearby",
]


SCENE_DENSITIES = {
    "sparse": "relatively open seabed with substantial exposed sediment and limited clutter",
    "moderate": "moderately cluttered seabed with natural rocks, algae, sediment and gravel",
    "dense": "visually cluttered seabed with rocks, algae, gravel, sediment and irregular debris",
}

DETECTION_DIFFICULTY = {
    "easy": "objects mostly visible, limited occlusion",
    "moderate": "some objects partly obscured by rocks, algae or sediment",
    "hard": "several objects small, low contrast or partly obscured and blending into substrate",
}

SCENE_WATER_PHRASE = "natural underwater water appearance, realistic green-dominant coastal water"

LIGHTING_CONDITIONS = [
    "diffuse natural sunlight with soft uneven brightness",
    "soft filtered daylight with gentle brightness variation",
    "natural underwater daylight with realistic attenuation",
    "weak diffuse daylight with low-contrast shadows",
    "overcast underwater illumination with subdued contrast",
]

CAMERA_HEIGHTS = {
    "medium": "camera approximately 1m above the seabed, angled downward",
    "high": "camera approximately 1.5-2m above the seabed, looking downward",
    "far": "camera approximately 5-8m above the seabed, wide survey overview",
}

CAMERA_FOV = [
    "moderately wide field of view",
    "wide-angle survey view",
    "compact-camera wide view",
]

CAMERA_MOTION = [
    "slight motion softness from a moving underwater robot",
    "very mild motion blur from slow robotic movement",
    "minimal motion softness from forward motion",
    "stable survey capture with subtle sensor motion effects",
]

IMAGING_CONDITIONS = [
    "visible underwater particulate haze and mild detail loss with distance",
    "subtle scattering, reduced distant contrast and mild sensor noise",
    "natural underwater softness with reduced fine detail in the background",
    "mild suspended-particle haze and subdued local contrast",
]

COMPOSITIONS = [
    "candid benthic survey frame, off-centre composition",
    "irregular asymmetric documentary composition",
    "unposed ecological survey frame with layered depth",
    "documentary seabed frame with no deliberate hero subject",
]

DEPTH_DISTRIBUTIONS = [
    "objects span foreground to background with substantial apparent size variation",
    "target organisms occur at different camera distances",
    "some organisms are farther away and partly softened by water",
]


COMPOSITION_GUARD = (
    "natural asymmetric spacing, no decorative symmetry, "
    "no cloned or repeated-looking objects"
)

SPECIES_GUARD = (
    "only the requested organisms and natural seabed material are present, "
    "no other animals"
)

REALISM_GUARD = (
    "plain documentary underwater robot survey photograph, "
    "no divers, boats, aquarium presentation or CG-rendered appearance"
)

OPTICAL_GUARD = (
    "full-frame survey image, no fisheye distortion, no vignette artifact"
)

COLOR_PALETTE_GUARD = (
    "seabed and organisms use muted natural benthic colours, "
    "no vivid or highly saturated colours"
)

SUBJECT_SCALE_GUARD = (
    "target organisms are small relative to the full frame, "
    "with many individuals occupying only a small portion of the image"
)

STARFISH_GUARD = (
    "starfish are low-contrast benthic organisms, naturally camouflaged "
    "against the seabed, often partially buried or obscured, with irregular "
    "orientation and no clean studio-like silhouette"
)

BIVALVE_GUARD = (
    "bivalve shells fully closed and undisturbed"
)

FRAMING_COUNT_ANCHOR = (
    "count is the strict total for the entire frame, not per unit area"
)


FRAMING = {
    "mid": (
        "mid-distance benthic survey framing with foreground and "
        "mid-ground visible"
    ),
    "wide": (
        "wide benthic survey framing showing a large section of seabed "
        "with objects at different distances"
    ),
}


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
    2: {
        "sparse": (1, 2),
        "moderate": (2, 4),
        "dense": (3, 6),
    },
}


@dataclass
class PromptMetadata:
    seed: int
    density: str
    difficulty: str
    camera_height: str
    framing: str
    class_counts: dict[int, int]


def _drop_leading_article(text: str) -> str:
    for article in ("an ", "a "):
        if text.startswith(article):
            return text[len(article):]
    return text


def class_phrase(
    class_id: int,
    count: int,
    rng: random.Random,
) -> str:

    entry = CLASSES[class_id]

    morphology = rng.choice(
        entry["morphology"]
    )

    if class_id == 0:
        arrangement = rng.choice(
            entry["arrangements_solo"]
            if count == 1
            else entry["arrangements_group"]
        )
    else:
        arrangement = rng.choice(
            entry["arrangements_solo"]
            if count == 1
            else entry["arrangements_group"]
        )

    if count == 1:
        return (
            f"{morphology}, {arrangement}"
        )

    return (
        f"{count} living {entry['short']}: "
        f"{_drop_leading_article(morphology)}; "
        f"{arrangement}"
    )


def generate_class_counts(
    rng: random.Random,
    density: str,
    min_classes: int = 2,
    max_classes: int = 3,
) -> dict[int, int]:

    available = list(CLASSES.keys())

    number_of_classes = rng.randint(
        min_classes,
        min(max_classes, len(available)),
    )

    selected = rng.sample(
        available,
        number_of_classes,
    )

    return {
        cid: rng.randint(
            *COUNT_RANGES[cid][density]
        )
        for cid in sorted(selected)
    }


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
        density = rng.choice(
            list(SCENE_DENSITIES)
        )

    if difficulty is None:
        difficulty = rng.choice(
            list(DETECTION_DIFFICULTY)
        )

    if camera_height is None:
        camera_height = rng.choice(
            list(CAMERA_HEIGHTS)
        )

    if framing is None:
        framing = rng.choice(
            list(FRAMING)
        )

    scene = rng.choice(
        SCENE_TEMPLATES
    )

    algae = rng.choice(
        ALGAE_VARIATIONS
    )

    substrate = rng.choice(
        SUBSTRATE_VARIATIONS
    )

    rock = rng.choice(
        ROCK_FORMATIONS
    )

    lighting = rng.choice(
        LIGHTING_CONDITIONS
    )

    composition = rng.choice(
        COMPOSITIONS
    )

    fov = rng.choice(
        CAMERA_FOV
    )

    motion = rng.choice(
        CAMERA_MOTION
    )

    imaging = rng.choice(
        IMAGING_CONDITIONS
    )

    depth = rng.choice(
        DEPTH_DISTRIBUTIONS
    )

    subject_phrases = [
        class_phrase(
            class_id,
            count,
            rng,
        )
        for class_id, count in sorted(
            counts.items()
        )
    ]

    if len(subject_phrases) == 1:
        subjects = subject_phrases[0]
    elif len(subject_phrases) == 2:
        subjects = (
            f"{subject_phrases[0]} and "
            f"{subject_phrases[1]}"
        )
    else:
        subjects = (
            ", ".join(subject_phrases[:-1])
            + ", and "
            + subject_phrases[-1]
        )

    guards = [
        COMPOSITION_GUARD,
        SPECIES_GUARD,
        REALISM_GUARD,
        COLOR_PALETTE_GUARD,
        SUBJECT_SCALE_GUARD,
        OPTICAL_GUARD,
    ]

    if 0 in counts:
        guards.append(
            STARFISH_GUARD
        )

    if 2 in counts:
        guards.append(
            BIVALVE_GUARD
        )

    if framing == "wide":
        guards.append(
            FRAMING_COUNT_ANCHOR
        )

    prompt = " ".join([
        (
            "Photorealistic underwater robot-survey "
            "photograph. Scene contains: "
            f"{subjects}."
        ),
        "; ".join(guards) + ".",
        (
            f"{scene}, with {algae}, {substrate}, "
            f"{rock}."
        ),
        (
            f"{composition}; "
            f"{SCENE_DENSITIES[density]}; "
            f"{DETECTION_DIFFICULTY[difficulty]}; "
            f"{depth}."
        ),
        (
            f"{CAMERA_HEIGHTS[camera_height]}, "
            f"{fov}; {lighting}; {motion}, "
            f"{imaging}; {SCENE_WATER_PHRASE}; "
            f"{FRAMING[framing]}."
        ),
    ])

    metadata = PromptMetadata(
        seed=seed,
        density=density,
        difficulty=difficulty,
        camera_height=camera_height,
        framing=framing,
        class_counts=dict(counts),
    )

    return prompt, metadata


def class_names() -> dict[int, str]:
    return {
        cid: entry["short"]
        for cid, entry in CLASSES.items()
    }


def detector_prompts() -> dict[int, str]:
    return class_names()


if __name__ == "__main__":
    counts = {
        0: 3,
        1: 2,
        2: 3,
    }

    prompt, metadata = build_prompt(
        counts,
        seed=1,
        camera_height="far",
        framing="wide",
        difficulty="hard",
    )

    print(prompt)
    print()
    print(metadata)