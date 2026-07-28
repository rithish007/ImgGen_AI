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

# Per-class spatial arrangement, appended whenever more than one instance is
# requested. Originally a single global SCATTER_CUE ("unevenly spaced, some
# distance from the others") applied to every class alike - that fixed the
# v6 clone-at-regular-spacing problem, but it is biologically wrong for
# species that naturally aggregate. Real DUO reference photos (user-supplied)
# show sea urchins wedged tightly into rock crevices, several individuals
# touching or overlapping - the opposite of "spaced apart". Applying one
# spacing rule to all four classes was the bug; arrangement is now per-class.
ARRANGEMENTS: dict[int, str] = {
    0: (  # starfish: solitary, camouflaged against rock/algae (v10 - was "open
          # sediment", which fought against the CLASSES entry's "blending into
          # rock and algae" and pulled the substrate back toward open sand)
        "individuals of varying sizes and orientations, scattered among rocks and "
        "patches of algae some distance apart from each other, unevenly spaced"
    ),
    1: (  # sea urchin: aggregates in crevices and rock faces - matches DUO reference photos
        "clustered tightly together in a rocky crevice or wedged against a rock "
        "face, several individuals touching or overlapping one another, packed "
        "into the gap between rocks rather than spaced apart"
    ),
    2: (  # sea cucumber: solitary, spread across open sediment
        "individuals resting alone, spread apart across the open sediment, each "
        "some distance from the others"
    ),
    3: (  # scallop: loosely scattered, well separated. v9 had "a few resting close
          # together near a rock" here, which over-triggered a ~2 -> ~11 instance
          # blowout - the only class of the four that overshot its requested count
          # by more than one or two. The vague quantifier "a few" was read as an
          # invitation to keep adding scallops; the fix is to remove it, not to
          # rephrase it, since every other class lacks that clause and did not
          # over-generate.
        "individuals of varying sizes, sparsely and loosely scattered across the "
        "open sediment, each well separated from the others"
    ),
}

# class_id matches DUO exactly. Do not renumber.
#
# Phrasing is morphological rather than nominal. Naming the animal alone pulls
# the model toward the most photographed sense of the word, which for two of
# these four classes is a seafood dish. Describing the body - shape, texture,
# colour, posture on the substrate - anchors it to the live animal instead.
#
# The scallop entry must NOT say "shell open": the v7 smoke test rendered that
# as cooked scallop meat presented in open shells. Live scallops in survey
# imagery are closed or barely gaped and usually partly buried, with only the
# ribbed upper valve showing.
#
# The sea cucumber entry avoids ANY word that names the wrong anatomy, even
# negated. v8's "elongated leathery body with blunt conical papillae" rendered
# as a toy caterpillar/millipede with neat symmetric rows of leg-like cones -
# diffusion models routinely embed a concept whether or not it is negated, so
# "not a caterpillar, no legs" would likely still draw a caterpillar. The fix
# is to give it a *correct* positive anchor instead: real holothurians read as
# legless cylinders, avoiding the word entirely does that work without ever
# naming the wrong animal.
#
# v9's "slug" anchor fixed the caterpillar problem but overcorrected: it
# produced a straight, uniformly smooth, uniformly dark body. A real reference
# photo (user-supplied) shows a curved, bent posture and mottled grey/black/
# white blotchy skin with a rough warty granular texture - the opposite of
# smooth and uniform. Rewritten directly from that reference. "Slug" dropped:
# it was pulling toward smooth skin, which this reference shows is wrong.
CLASSES: dict[int, dict[str, str]] = {
    0: {
        "duo_label": "starfish",
        "short": "starfish",
        # v9 rendered starfish large, crisp and centered on open light sand -
        # visually a "hero shot", clearly the dominant subject. A real DUO
        # reference photo (user-supplied) shows starfish as small, low-contrast,
        # camouflaged blobs blending into rock and algae - easy for even a human
        # annotator to miss. That's not a colour-cast difference (Stage 3's job),
        # it's a Stage 1 framing/substrate choice: making the subject always
        # large, sharp and prominent teaches Stage 2 annotation an easier task
        # than the real detector will face on real footage.
        "singular": "a small five-armed starfish, mottled brown and grey blending into the surrounding rock and algae, camouflaged and easy to overlook, resting flat against the substrate",
        "plural": "{n} small five-armed starfish, mottled brown and grey blending into the surrounding rock and algae, camouflaged and easy to overlook, resting flat against the substrate, {arrangement}",
    },
    1: {
        "duo_label": "echinus",
        "short": "sea urchin",
        # v9's "round test" rendered as ball-shaped/spherical. User feedback:
        # too round, needs a flattened dome shape, and darker colour.
        "singular": "a very dark, almost black sea urchin, a flattened dome-shaped test low and wide rather than spherical, covered in short dense spines, sitting on the bottom",
        "plural": "{n} very dark, almost black sea urchins, flattened dome-shaped tests low and wide rather than spherical, covered in short dense spines, {arrangement}",
    },
    2: {
        "duo_label": "holothurian",
        "short": "sea cucumber",
        "singular": "a sea cucumber, an elongated cylindrical body gently curved and bent like a hook, mottled grey-black-and-white blotchy skin with a rough warty granular texture, blunt rounded ends, lying motionless on the sand",
        "plural": "{n} sea cucumbers, elongated cylindrical bodies gently curved and bent like hooks, mottled grey-black-and-white blotchy skin with a rough warty granular texture, blunt rounded ends, lying motionless, {arrangement}",
    },
    3: {
        "duo_label": "scallop",
        "short": "scallop",
        "singular": "a scallop, a fan-shaped shell with radiating ribs, mostly closed with a thin sliver of pale living tissue and tiny tentacles visible at the shell's edge, half-buried in the sediment",
        "plural": "{n} scallops, fan-shaped shells with radiating ribs, mostly closed with a thin sliver of pale living tissue and tiny tentacles visible at the shell's edge, half-buried, {arrangement}",
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
#
# Rock crevices and turf-covered ledges are named explicitly (v9) because the
# sea urchin ARRANGEMENT above asks it to wedge into one - without a crevice in
# the scene description, there is nothing for it to cluster into. Still says
# "clear water, natural colour" deliberately: the green/murky look in real DUO
# footage is Stage 3's job (Jerlov physics transform on a clean scene), not
# something to bake in here - see the Stage 3 section of the plan for why.
SCENE = (
    "underwater photograph, temperate coastal seabed, mixed sand and rocky bottom "
    "with crevices, ledges, and ridges, patches of turf algae and encrustation on "
    "the rock surfaces, soft diffuse daylight from above, gentle low-contrast "
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
    "cooked, food, seafood dish, plate, restaurant, sashimi, open shell, shellfish meat, "
    # v8 failures: sea cucumber rendered as a caterpillar/millipede, scallop
    # shells read as empty/dead. SD3.5 has real negative-prompt suppression,
    # unlike Klein, so these terms only help here.
    "caterpillar, millipede, centipede, larva, insect legs, "
    "segmented body, articulated legs, rows of legs, worm with legs, "
    "empty shell, dead shell, beachcombed shell, shell litter, bleached shell"
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
    return entry["plural"].format(n=_count_word(count), arrangement=ARRANGEMENTS[class_id])


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
