"""Prompt engine for AI-generated underwater synthetic datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional


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
        "duo_label": "scallop",
        "short": "scallop",

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


ROCK_FORMATIONS = [

    "a cluster of rounded cobbles scattered loosely across the seabed",

    "a small pile of angular rubble formed from broken rock fragments",

    "a scatter of loose pebbles mixed into the surrounding sediment",

    "a group of small boulders resting on the seabed, spaced irregularly apart",

    "a mix of pebbles and small cobbles collected in a shallow depression",

    "a low pile of rubble and small boulders partially settled into the sediment",
]


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


SCENE_WATER_PHRASE = (
    "clear seawater with good visibility, true-to-life natural colour and no "
    "artificial colour cast"
)


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


POSITIVE_ONLY_GUARDS = (
    "true-to-life natural colour reproduction, realistic biological morphology, "
    "natural ecological habitat, full rectangular frame, no lens vignette, "
    "no artificial colour grading, realistic underwater camera imagery, "
    "natural irregular spatial distribution"
)

BIVALVE_GUARD = (
    "bivalve shells fully closed and undisturbed as found in their natural habitat"
)


FRAMING = {
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


FRAMING_COUNT_ANCHOR = (
    "The stated number of individuals for each organism is a strict total "
    "for the entire frame, not a density to repeat across the visible "
    "seabed - a wider view must not add extra individuals beyond that total."
)


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


def _random_count(
    rng: random.Random,
    class_id: int,
    density: str,
) -> int:
    low, high = COUNT_RANGES[class_id][density]
    return rng.randint(low, high)


def class_phrase(
    class_id: int,
    count: int,
    rng: random.Random,
) -> str:

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

    return (
        f"{quantity} separate living {entry['short']} individuals, "
        f"each showing {_drop_leading_article(morphology)}, "
        f"{arrangement}"
    )


def _drop_leading_article(text: str) -> str:
    for article in ("an ", "a "):
        if text.startswith(article):
            return text[len(article):]
    return text


def generate_class_counts(
    rng: random.Random,
    density: str,
    min_classes: int = 2,
    max_classes: int = 4,
) -> dict[int, int]:

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

    rng = random.Random(seed)


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


    scene_template, scene_idx = _choose(rng, SCENE_TEMPLATES)

    algae, algae_idx = _choose(rng, ALGAE_VARIATIONS)

    substrate, substrate_idx = _choose(rng, SUBSTRATE_VARIATIONS)

    rock_formation, rock_formation_idx = _choose(rng, ROCK_FORMATIONS)

    lighting, lighting_idx = _choose(rng, LIGHTING_CONDITIONS)

    composition, composition_idx = _choose(rng, COMPOSITIONS)

    camera_fov, camera_fov_idx = _choose(rng, CAMERA_FOV)

    camera_motion, camera_motion_idx = _choose(rng, CAMERA_MOTION)

    imaging, imaging_idx = _choose(rng, IMAGING_CONDITIONS)


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


    prompt_parts = [

        (
            "Photorealistic underwater robotic benthic survey image, "
            "realistic biological morphology and natural ecological habitat."
        ),

        (
            f"Temperate coastal seabed consisting of {scene_template}. "
            f"The habitat contains {algae} and {substrate}. "
            f"{rock_formation.capitalize()} is present in the scene."
        ),

        (
            f"{composition}. "
            f"{SCENE_DENSITIES[density]}. "
            f"{DETECTION_DIFFICULTY[difficulty]}."
        ),

        rng.choice(DEPTH_DISTRIBUTIONS) + ".",

        (
            f"The target organisms are naturally distributed within the habitat: "
            f"{subjects}."
        ),

        (
            f"{CAMERA_HEIGHTS[camera_height]}, "
            f"{camera_fov}, "
            f"natural perspective."
        ),

        (
            f"{SCENE_WATER_PHRASE}."
        ),

        (
            f"{lighting}."
        ),

        (
            f"{camera_motion}."
        ),

        (
            f"{imaging}."
        ),

        (
            f"{FRAMING[framing]}."
            + (f" {FRAMING_COUNT_ANCHOR}" if framing == "wide" else "")
        ),

        (
            "The image should resemble a frame captured by a real underwater "
            "robot during an ecological survey rather than a posed wildlife "
            "photograph, cinematic scene or artificial 3D environment."
        ),
    ]

    if not supports_negative:
        guards = POSITIVE_ONLY_GUARDS
        if 2 in counts:
            guards = f"{guards}, {BIVALVE_GUARD}"
        prompt_parts.append(guards)

    prompt = " ".join(prompt_parts)

    negative_prompt = NEGATIVE if supports_negative else None


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


def generate_dataset_prompts(
    number_of_images: int,
    *,
    base_seed: int = 2026,
    supports_negative: bool = True,
    density_distribution: Optional[dict[str, float]] = None,
    difficulty_distribution: Optional[dict[str, float]] = None,
) -> list[dict[str, object]]:

    if number_of_images < 1:
        raise ValueError("number_of_images must be >= 1")

    rng = random.Random(base_seed)

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


def save_prompt_manifest(
    prompts: list[dict[str, object]],
    output_path: str,
) -> None:

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


def detector_prompts() -> dict[int, str]:

    return {
        0: "starfish",
        1: "sea urchin",
        2: "scallop",
    }


def class_names() -> dict[int, str]:

    return {
        0: "starfish",
        1: "sea_urchin",
        2: "scallop",
    }


if __name__ == "__main__":

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