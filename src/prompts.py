"""Prompt construction for Stage 0.5 / Stage 1 generation.

Three things this module exists to get right:

1. Class phrasing. DUO's taxonomic labels (echinus, holothurian) are rare in
   web image-caption data, so they are used for class_id mapping only and never
   sent to a generator or detector. And bare "scallop"/"sea cucumber" resolve to
   the culinary sense, so the phrases below force the habitat sense.

2. Negative prompts. Flux2KleinPipeline accepts no text-level negative_prompt
   (only negative_prompt_embeds); StableDiffusion3Pipeline does. Rather than let
   the negative silently no-op on Klein, exclusions are folded into the positive
   prompt affirmatively for models that cannot take a negative.

3. Composition. Stage 0.5 smoke-test output (v6) came back with animals staged
   like product photography - centered, evenly lit - and multi-instance counts
   rendered as near-identical clones at regular spacing rather than naturally
   scattered individuals. Both are textbook diffusion failure modes: "evenly
   lit" / "colour-accurate" read as studio-photography language, and a bare
   "N of X" count gets resolved by tiling the same rendering. COMPOSITION and
   SCATTER_CUE below exist specifically to counter these two failures.
"""

from __future__ import annotations

# Appended whenever more than one instance of a class is requested, to break
# the "identical clones at regular spacing" failure mode.
SCATTER_CUE = (
    "individuals of varying sizes and orientations, scattered at different "
    "distances from the camera, unevenly spaced, some partially buried in "
    "sand or tucked among rocks"
)

# class_id matches DUO exactly. Do not renumber.
CLASSES: dict[int, dict[str, str]] = {
    0: {
        "duo_label": "starfish",
        "short": "starfish",
        "singular": "a starfish on the seabed",
        "plural": "{n} starfish on the seabed, {scatter}",
    },
    1: {
        "duo_label": "echinus",
        "short": "sea urchin",
        "singular": "a spiny sea urchin on the rocky seabed",
        "plural": "{n} spiny sea urchins on the rocky seabed, {scatter}",
    },
    2: {
        "duo_label": "holothurian",
        "short": "sea cucumber",
        "singular": "a live sea cucumber crawling on the sandy seabed",
        "plural": "{n} live sea cucumbers crawling on the sandy seabed, {scatter}",
    },
    3: {
        "duo_label": "scallop",
        "short": "scallop",
        "singular": "a live scallop, shell open, resting on the sandy seabed",
        "plural": "{n} live scallops, shells open, resting on the sandy seabed, {scatter}",
    },
}

# Names a real photographic condition that is both underwater and colour-neutral.
# Sun caustics are the load-bearing cue: dappled light on the seabed reads as
# unmistakably submerged while costing no colour cast. Output must approximate
# scene radiance J_c, which is what the Stage 3b physics transform consumes.
SCENE = (
    "underwater photograph, shallow tropical reef flat, bright sunlight, "
    "sun caustics rippling across the sandy seabed, clear water, high visibility, "
    "natural colour, neutral white balance"
)

# Counters the "product photo" staging seen in the v6 smoke test: centered
# subject, symmetrical arrangement, even studio-style lighting. Framed as a
# candid documentary/survey photograph, since that is the training-data
# distribution that actually contains off-center, naturally-lit wildlife shots.
COMPOSITION = (
    "candid marine survey photograph, documentary style, off-center asymmetric "
    "framing, natural unposed arrangement, not a product photo, not staged, "
    "not centered, not symmetrical"
)

CAMERA = "wide-angle ROV camera, slight fisheye distortion, photorealistic"

NEGATIVE = (
    "text, watermark, diver, boat, human, water surface, sky, "
    "green tint, murky, hazy, low visibility, colour cast, dark, "
    "aquarium, fish tank, glass, white background, studio, "
    "illustration, drawing, cartoon, 3d render, "
    "product photo, stock photo, catalog photo, studio lighting, "
    "centered composition, symmetrical, posed, staged, "
    "cloned, duplicated, identical copies, grid pattern, repeating pattern"
)

# Affirmative restatement of NEGATIVE, for pipelines with no negative_prompt.
# Deliberately avoids "evenly lit" / "colour-accurate" - both read as studio-
# photography language and contributed to the v6 staged-product-shot look.
POSITIVE_ONLY_GUARDS = (
    "natural uneven sunlight, true-to-life colour, no artificial colour grading, "
    "open natural habitat, unobstructed view of the seafloor"
)

FRAMING = {
    "close-up": "close-up shot, camera low near the seafloor",
    "mid": "mid-distance shot, seafloor receding into the background",
}

DENSITY_RANGE = {"sparse": (2, 3), "moderate": (4, 6)}

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _count_word(n: int) -> str:
    return _NUMBER_WORDS.get(n, str(n))


def class_phrase(class_id: int, count: int) -> str:
    entry = CLASSES[class_id]
    if count == 1:
        return entry["singular"]
    return entry["plural"].format(n=_count_word(count), scatter=SCATTER_CUE)


def build_prompt(
    counts: dict[int, int],
    framing: str,
    supports_negative: bool,
) -> tuple[str, str | None]:
    """Build (prompt, negative_prompt) for one manifest row.

    counts maps class_id -> requested instance count. Returns negative_prompt
    of None for pipelines that cannot take one, with the exclusions folded into
    the positive prompt instead.
    """
    if framing not in FRAMING:
        raise ValueError(f"unknown framing {framing!r}, expected one of {list(FRAMING)}")

    subjects = [class_phrase(cid, counts[cid]) for cid in sorted(counts)]
    if len(subjects) == 1:
        subject_text = subjects[0]
    else:
        subject_text = ", ".join(subjects[:-1]) + " and " + subjects[-1]

    # COMPOSITION is placed before the subject deliberately: with a causal
    # text encoder (e.g. Klein's Mistral-based one), earlier tokens have more
    # influence over global layout, later tokens over local detail. Stating
    # "off-center, unposed documentary style" before naming the subject count
    # gives it priority over the staged/centered look that a bare "N of X"
    # count otherwise defaults to.
    parts = [SCENE, COMPOSITION, subject_text, FRAMING[framing], CAMERA]
    if not supports_negative:
        parts.append(POSITIVE_ONLY_GUARDS)

    return ", ".join(parts), (NEGATIVE if supports_negative else None)


def detector_prompts() -> dict[int, str]:
    """Short noun-phrase concepts for Stage 2 (SAM3 / Grounding DINO).

    Deliberately NOT the Stage 1 scene sentences - detectors want a bare concept.
    """
    return {cid: entry["short"] for cid, entry in CLASSES.items()}
