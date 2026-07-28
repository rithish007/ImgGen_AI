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
# the "identical clones at regular spacing" failure mode. Verified effective in
# the v7 smoke test (starfish/sea cucumber rows came back naturally scattered).
SCATTER_CUE = (
    "individuals of varying sizes and orientations, scattered at different "
    "distances from the camera, unevenly spaced, some partially buried in "
    "sediment or tucked among rocks"
)

# class_id matches DUO exactly. Do not renumber.
#
# Phrasing is morphological rather than nominal. Naming the animal alone pulls
# the model toward the most photographed sense of the word, which for two of
# these four classes is a seafood dish. Describing the body - shape, texture,
# colour, posture on the substrate - anchors it to the live animal instead.
#
# The scallop entry in particular must NOT say "shell open": the v7 smoke test
# rendered that as cooked scallop meat presented in open shells. Live scallops
# in survey imagery are closed or barely gaped and usually partly buried, with
# only the ribbed upper valve showing.
CLASSES: dict[int, dict[str, str]] = {
    0: {
        "duo_label": "starfish",
        "short": "starfish",
        "singular": "a five-armed starfish lying flat against the sediment, mottled brown and grey",
        "plural": "{n} five-armed starfish lying flat against the sediment, mottled brown and grey, {scatter}",
    },
    1: {
        "duo_label": "echinus",
        "short": "sea urchin",
        "singular": "a dark purple-black sea urchin, a round test covered in short dense spines, sitting on the bottom",
        "plural": "{n} dark purple-black sea urchins, round tests covered in short dense spines, sitting on the bottom, {scatter}",
    },
    2: {
        "duo_label": "holothurian",
        "short": "sea cucumber",
        "singular": "a dark brown sea cucumber, elongated leathery body with blunt conical papillae, resting on the sediment",
        "plural": "{n} dark brown sea cucumbers, elongated leathery bodies with blunt conical papillae, resting on the sediment, {scatter}",
    },
    3: {
        "duo_label": "scallop",
        "short": "scallop",
        "singular": "a scallop, fan-shaped shell with radiating ribs, closed and half-buried in the sediment with only the upper valve showing",
        "plural": "{n} scallops, fan-shaped shells with radiating ribs, closed and half-buried in the sediment with only the upper valves showing, {scatter}",
    },
}

# Matches DUO's actual habitat: temperate Chinese coastal seabed - sand, gravel
# and scattered rock. NOT tropical reef. The v7 smoke test used "shallow
# tropical reef flat", which produced coral heads and tropical species and put
# the whole pilot in the wrong domain before Stage 3 even ran.
#
# Lighting is deliberately understated. v7 used "sun caustics rippling across
# the sandy seabed" and every image came back with a hard-edged white polygonal
# web across the sand, like a Voronoi diagram drawn in marker pen. Real caustics
# in survey imagery are low-contrast brightness variation, not white lines - so
# they are described softly here and the failure mode is pushed into NEGATIVE.
SCENE = (
    "underwater photograph, temperate coastal seabed, sand and fine gravel bottom "
    "with scattered rocks, soft diffuse daylight from above, gentle low-contrast "
    "variation in brightness across the bottom, clear water, good visibility, "
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

# "slight fisheye distortion" was removed after v7: it was applied as a full
# circular fisheye, leaving a black vignette ring around the frame. That is
# fatal for this pipeline - the black corners survive the Stage 4 640x640 crop
# and would be baked into the dataset as fake image content.
CAMERA = "wide-angle underwater survey camera, photorealistic, sharp focus"

NEGATIVE = (
    "text, watermark, diver, boat, human, water surface, sky, "
    "green tint, murky, hazy, low visibility, colour cast, dark, "
    "aquarium, fish tank, glass, white background, studio, "
    "illustration, drawing, cartoon, 3d render, "
    "product photo, stock photo, catalog photo, studio lighting, "
    "centered composition, symmetrical, posed, staged, "
    "cloned, duplicated, identical copies, grid pattern, repeating pattern, "
    # v7 failures, in priority order
    "caustics, light caustics, white lines on sand, polygonal light pattern, "
    "voronoi pattern, cracked pattern, net pattern, "
    "fisheye, circular vignette, black border, black corners, vignetting, "
    "coral reef, coral, tropical fish, anemone, "
    "cooked, food, seafood dish, plate, restaurant, sashimi, open shell, shellfish meat"
)

# Affirmative restatement of NEGATIVE, for pipelines with no negative_prompt.
# Deliberately avoids "evenly lit" / "colour-accurate" - both read as studio-
# photography language and contributed to the v6 staged-product-shot look.
POSITIVE_ONLY_GUARDS = (
    "true-to-life colour, no artificial colour grading, "
    "soft even ambient light with no harsh highlights on the sand, "
    "full rectangular frame, no lens vignette, "
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
