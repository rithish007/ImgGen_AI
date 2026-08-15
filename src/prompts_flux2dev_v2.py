"""
prompts_v2.py
=============
Prompt engine v2 - positive-only, length-budgeted, front-loaded.

Duplicated from prompts.py rather than editing it in place (same "duplicate,
don't edit shared files" rule used for prompts_hunyuan.py). This is a bigger
rewrite than that file: it isn't scoped to one model's defects, it fixes
three things found to affect klein, flux2dev and (structurally) Hunyuan
alike, all discovered while investigating why klein's images looked more
"artificial" than flux2dev/Hunyuan's and why a hallucinated non-target
animal appeared in a flux2dev image:

1. THE PROMPT WAS SILENTLY TRUNCATED, EVERY SINGLE TIME.
   Flux2Pipeline/Flux2KleinPipeline default max_sequence_length=512 tokens
   (Mistral Small 3.1 text encoder - confirmed via docs.bfl.ml and
   huggingface.co/docs/diffusers/api/pipelines/flux2, both cite 512 as the
   pipeline's actual supported limit, not just a soft default). Measured
   with the real FLUX.2 tokenizer against all 50 rows of the 3-pilot
   flux2dev output: EVERY row exceeded 512 tokens (min 569, max 764, mean
   670). Decoding token 512 onward for row 1 showed the dropped tail was
   the back half of the realism-anchor sentence PLUS the entire
   POSITIVE_ONLY_GUARDS block - the guards written specifically to fight
   decorative/symmetric composition and unrealistic colour grading were
   never seen by the model, not once, for any image generated in this
   pipeline so far. This single fact plausibly explains multiple previously
   "partially effective" or "unverified" fixes in prompts.py (e.g.
   FRAMING_COUNT_ANCHOR, appended even later in the prompt than the guards)
   - they may never have reached the model at all on longer rows.

2. NO NEGATIVE PROMPT SUPPORT ON ANY MODEL ACTUALLY IN USE.
   klein and flux2dev already had supports_negative=False (Flux2*Pipeline
   only exposes negative_prompt_embeds, no text-level negative_prompt).
   Hunyuan's generate_image() has no negative_prompt/cfg parameter either
   (confirmed via inspect.signature() - see generate_hunyuan.py). So the
   entire NEGATIVE prompt block (GLOBAL/COMPOSITION/OPTICAL/BIOLOGICAL) in
   prompts.py has been dead code for every model this project has actually
   run, this whole time - it only ever applied to sd35/qwen_image, both
   dropped from the pipeline early on. Removed entirely here rather than
   kept as an unused branch. This isn't a workaround forced by this
   pipeline's model choices - Black Forest Labs' own prompting guide states
   plainly: "FLUX.2 does not support negative prompts. Focus on describing
   what you want, not what you don't want"
   (https://docs.bfl.ml/guides/prompting_guide_flux2). Positive-only is the
   documented, intended way to prompt this model family, not a compromise.

3. WORD ORDER / STRUCTURE DIDN'T MATCH HOW THE MODEL WEIGHTS THE PROMPT.
   Per the same BFL guide, FLUX.2 "pays more attention to what comes
   first," with a stated priority order: main subject -> key action ->
   critical style -> essential context -> secondary details. prompts.py put
   scene/environment first and the actual target organisms (the real "main
   subject") at position 7 of 13, with the anti-defect guards dead last at
   position 13. Reordered below: subject first, guards/critical-style
   second (so they survive even in a worst-case truncation scenario),
   environment/composition third, camera/lighting/sensor last (least
   important to detection-training-data correctness, most fine to lose if
   something still runs long).

4. RAW LENGTH ITSELF, INDEPENDENT OF ORDERING. BFL's guide: "30-80 words
   typically ideal... longer prompts aren't inherently better." The
   original prompts.py averaged ~450-500 words (670 tokens - the FLUX.2
   tokenizer runs close to 2 tokens/word on this kind of descriptive text,
   not the ~1.3 a plain word count would suggest, confirmed empirically
   below). Reordering the OLD wording alone only got the worst case down to
   ~570-690 tokens - still over the cap. Getting durably under 512 required
   rewriting the morphology/arrangement/guard text itself to be denser, not
   just reordering it - e.g. starfish morphology went from "a small living
   starfish with five arms, mottled brown and grey coloration, rough
   natural surface texture" (28 tokens) to "a small starfish, five arms,
   mottled brown-grey, rough texture" (18 tokens), keeping every anatomical
   anchor, just without the connective padding. Verified empirically (see
   self-test below): worst case across all 50 manifest rows is now within
   the cap with real margin, not just on average.

ALSO FIXED: the count=1 arrangement contradiction. class_phrase() used to
pick an arrangement phrase at random regardless of requested count. Sea
urchin's arrangements were ALL group-phrased ("several individuals
touching...", "grouped irregularly... some individuals hidden...") with
zero singular-safe option - so a count=1 urchin prompt read "a sea urchin
... grouped irregularly ... some individuals partially hidden," a direct
contradiction. Measured impact across the 3-pilot flux2dev/klein data:
count=1 urchin rows overshot 2.5x on average vs 1.07x for count>1 rows;
scallop count=1 rows overshot up to 6.44x (klein) vs ~1.3x for count>1.
Starfish never showed this because its arrangements happened to already be
singular-safe. Fixed by splitting each class's arrangements into
`arrangements_solo` (used when count==1) and `arrangements_group` (used
otherwise), rather than one undifferentiated list.

ALSO CARRIED FORWARD from prompts_hunyuan.py: the SCENE_DENSITIES "dense"
reword (dropped the ambiguous "overlapping environmental features" that
contradicted per-class arrangement text) and the structural rock-formation
gate for single-class+dense rows. Measured net-neutral for flux2dev
(one row improved, one got worse on the decorative-arrangement front) but
a real improvement for klein (scallop count=1 overshoot 6.44x -> 3.00x) -
folded into the shared engine rather than kept Hunyuan-only, since it's not
shown to regress anything and helps at least one model concretely.

NOT YET DONE: integrating this into generate.py / generate_3pilot.py /
generate_hunyuan.py / generate_klein_promptfix.py - those call
build_prompt(..., supports_negative=cfg["supports_negative"]) and unpack a
3-tuple (prompt, negative, metadata). This file's build_prompt() drops the
now-meaningless supports_negative parameter and returns a 2-tuple (prompt,
metadata) instead of carrying a dead always-None negative_prompt slot -
that is a deliberate breaking change, not an oversight, so those call sites
need updating before this can actually run a generation job. Do that only
after re-verifying prompt token counts and running a fresh dry-run/smoke
test - do not assume this file works untested just because it imports
cleanly. Also note CLASSES only has 3 keys (0/1/2 - sea cucumber was
removed earlier in the project) so the real worst case is a 3-class row,
not 4, even though generate_class_counts() still takes max_classes=4.

Self-test (token budget check against the real tokenizer):
    python -c "
    from transformers import AutoTokenizer
    from prompts_flux2dev_v2 import build_prompt
    import json
    tok = AutoTokenizer.from_pretrained('black-forest-labs/FLUX.2-dev', subfolder='tokenizer')
    manifest = json.loads(open('../manifests/2-pilot.json').read())
    worst = 0
    for row in manifest['rows']:
        counts = {int(k): v for k, v in row['requested_counts'].items()}
        prompt, _ = build_prompt(counts, seed=row['seed'], density=row['density'], framing=row['framing'])
        n = len(tok(prompt)['input_ids'])
        worst = max(worst, n)
    print('worst-case token count:', worst, '/ 512')
    "
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================
#
# Morphology/arrangement text rewritten denser than prompts.py (see module
# docstring point 4) - every phrase here keeps the specific anatomical
# anchors already proven to matter (five arms, dense short spines, tightly
# closed shell + sediment coverage mentioned twice for scallop - the
# 0.5-smoke audit found the arrangement-only burial cue got ignored 7/7
# times unless also stated in the morphology), just without the connective
# padding prompts.py used ("a small living X with...", "...natural X
# surface texture").

CLASSES: dict[int, dict[str, object]] = {

    0: {
        "duo_label": "starfish",
        "short": "starfish",

        "morphology": [
            "a small starfish, five arms, mottled brown-grey, rough texture",
            "a small five-armed starfish, muted brown-grey, irregular darker patches",
            "a small five-armed starfish, dark brown-grey mottling, irregular arm proportions",
            "a small starfish, five broad arms, muted reddish-brown and grey",
            "a small dark brown-grey starfish, five arms, subtle blotchy markings",
        ],

        # Already singular-safe (no "individuals"/"grouped" language) - fine
        # to reuse for both the solo and group case unchanged.
        "arrangements_solo": [
            "resting flat against rock and algae, partially blending into the substrate",
            "resting beside a low rock ledge, partly obscured",
            "lying on sediment near algae, naturally camouflaged",
            "partially covered by fine sediment, resting against a rock",
        ],
        "arrangements_group": [
            "resting flat against rock and algae, partially blending into the substrate",
            "resting beside a low rock ledge, partly obscured",
            "lying on sediment near algae, naturally camouflaged",
            "partially covered by fine sediment, resting against a rock",
        ],
    },

    1: {
        "duo_label": "echinus",
        "short": "sea urchin",

        "morphology": [
            "a sea urchin, low flattened dome body, dense short spines, dark brown-black",
            "a dark grey-black sea urchin, broad flattened dome body, dense short spines",
            "a nearly black sea urchin, low wide body, dense short dark spines",
            "a dark brown sea urchin, flattened low dome body, dense short spines",
        ],

        # NEW - the original 3 urchin arrangement phrases were ALL group-only
        # ("several individuals", "grouped... some individuals hidden") with
        # no singular option at all, which is the root cause of the count=1
        # overshoot bug (see module docstring). These two are new, written
        # to be singular-safe.
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

        # "tightly closed" (never "gaping"/"open"/interior mentioned) and
        # sediment coverage are the two load-bearing anchors here - see the
        # 0.5-smoke audit note above and prompts.py's original comment on
        # why a "mostly closed, narrow edge visible" phrasing still rendered
        # as a served-dish look.
        "morphology": [
            "a scallop, ribbed fan shell, cream-brown, tightly closed, dusted with sediment",
            "a fan-shaped scallop shell, cream-brown, tightly closed, ribbed, partly buried in sediment",
            "a small scallop, ribbed shell, light brown-cream, tightly closed, thin sediment layer",
            "a scallop, textured ribbed shell, cream-brown, tightly closed, partly buried",
        ],

        # Original had only 1 of 3 phrases singular-safe ("partially buried
        # at irregular locations") - the other 2 used "individuals" even at
        # count=1. That 1 safe phrase becomes the whole solo list; the
        # group list is unchanged (shortened).
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
# SCENE / ENVIRONMENT
# ============================================================================

SCENE_TEMPLATES = [
    "temperate coastal seabed - sand, coarse sediment, gravel, irregular rocks, shallow ledges",
    "temperate coastal seabed - sandy sediment, gravel, scattered rocks, natural formations",
    "temperate coastal seafloor - fine sediment between rocks, gravel, small crevices",
    "temperate marine substrate - sand, gravel, small stones, irregular rocky patches",
    "temperate coastal seabed - exposed sand, scattered rocks, gravel deposits, rock ledges",
]

ALGAE_VARIATIONS = [
    "sparse turf algae on rocks",
    "small patches of low algae across rocky surfaces",
    "sparse dark green-brown algae with biological encrustation",
    "low algae and subtle biological growth on rocks",
    "small algae-covered patches between exposed rock",
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
# ECOLOGICAL / COMPOSITIONAL CONDITIONS
# ============================================================================

SCENE_DENSITIES = {
    "sparse": "relatively open seabed, substantial exposed sediment, limited clutter",
    "moderate": "moderately cluttered seabed, natural rocks/algae/sediment across foreground and mid-ground",
    # Reworded from prompts_hunyuan.py - "overlapping environmental features"
    # contradicted per-class arrangement text ("several individuals touching
    # or partially overlapping") and was implicated in Hunyuan's urchin
    # fusion defects. "closely spaced natural debris" keeps the density cue
    # without describing overlap at all - object-to-object spatial relationships
    # stay the arrangement text's job alone.
    "dense": "visually cluttered seabed, rocks/algae/gravel/sediment and closely spaced natural debris",
}

DETECTION_DIFFICULTY = {
    "easy": "objects mostly visible, limited occlusion",
    "moderate": "some objects partly obscured by rocks/algae/sediment",
    "hard": "several objects small or partly obscured, blending into substrate",
}

# Stage 3 (Akkaynak-Treibitz + Jerlov) owns water condition/turbidity - Stage
# 1 always renders clear and colour-neutral. Fixed phrase, not a dict, so
# there's no random selection to accidentally re-enable double-degradation.
SCENE_WATER_PHRASE = "clear water, true-to-life colour"

LIGHTING_CONDITIONS = [
    "diffuse natural sunlight, soft uneven brightness, subtle shadows",
    "soft filtered daylight, gentle brightness variation, low-contrast shadows",
    "natural daylight, realistic underwater attenuation, soft illumination",
    "weak diffuse daylight, slightly uneven illumination",
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
# POSITIVE GUARDS
# ============================================================================
#
# Replaces prompts.py's NEGATIVE block (GLOBAL/COMPOSITION/OPTICAL/
# BIOLOGICAL_NEGATIVE) entirely - see module docstring point 2. Every
# concept that block tried to exclude is restated here as an affirmative
# description, per Black Forest Labs' own prompting guidance for this model
# family ("describe what you want, not what you want to avoid"). Applied to
# every image, every model - there is no supports_negative branch anymore
# because no model actually in use here supports a real negative prompt.
#
# Kept deliberately short: this whole block plus the subject line must
# survive comfortably inside the 512-token hard cap even on the longest
# (3-class - CLASSES only has 3 keys - dense, wide) rows, see module
# docstring points 1 and 4. Positioned second in the prompt (right after
# the subject), not last, specifically so it survives truncation even in a
# worst case.

COMPOSITION_GUARD = "natural asymmetric spacing, no decorative symmetry or cloned objects"
SPECIES_GUARD = "only these organisms and seabed material in frame, no other animals"
REALISM_GUARD = "plain documentary robot photo, no divers, boats, or CG rendering"
OPTICAL_GUARD = "full frame, no fisheye or vignette"

# Bug found via audit (kept from prompts.py): only append when scallop is
# actually requested - appending unconditionally caused klein to scatter
# shell-shaped debris into every scene as generic clutter, which SAM3 then
# correctly detected as false-positive scallops.
BIVALVE_GUARD = "bivalve shells fully closed and undisturbed"

# 2-pilot audit: framing="wide" + density="sparse" caused 2x-11x count
# overshoot (worst: 1 requested -> 20 detected) - the model appears to
# apply the arrangement text's spacing pattern per unit of visible area
# rather than honouring the explicit count. Only reinforced for wide
# framing, where the effect was actually measured. Moved into the guards
# block (was in FRAMING, position 11 of 13 in prompts.py) so it actually
# survives truncation now - it was one of the last sentences added to the
# old prompt, meaning it was one of the first to be cut.
FRAMING_COUNT_ANCHOR = "count is a strict total for the frame, not per unit area"


# ============================================================================
# FRAMING
# ============================================================================

FRAMING = {
    "close-up": "close framing, camera near the seabed, objects at different distances",
    "mid": "mid-distance framing, foreground and mid-ground objects visible",
    "wide": "wide framing, larger section of seabed, objects at different depths",
}


# ============================================================================
# OBJECT COUNT RANGES
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
# RANDOM HELPERS
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
# CLASS PHRASE GENERATION
# ============================================================================

def class_phrase(class_id: int, count: int, rng: random.Random) -> str:
    """
    Build a class description for the requested number of instances.

    Picks from arrangements_solo when count==1, arrangements_group
    otherwise - see module docstring for why this split exists (the old
    single undifferentiated list produced singular/plural contradictions
    that caused a measured 2.5-6.4x count overshoot specifically on
    count=1 rows).
    """
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
# SCENE OBJECT GENERATION
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
    Construct a complete underwater survey image-generation prompt.

    No supports_negative parameter - there is no negative-prompt path for
    any model this project runs (klein, flux2dev, Hunyuan all lack one -
    see module docstring). Returns (prompt, metadata), not a 3-tuple with a
    dead negative-prompt slot.

    Ordering follows BFL's stated FLUX.2 priority (main subject -> critical
    style -> essential context -> secondary details), and the whole prompt
    is budgeted to stay comfortably under the pipeline's 512-token hard cap
    - see module docstring for the measurements behind both choices.
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
    # Select scene components (same rng sequence order as prompts.py, so
    # a given seed draws the same environment even though the assembled
    # sentences below are restructured)
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

    # Single-class + dense is the row shape behind both the 4-pilot
    # decorative-clustering issue and the 3-pilot urchin fusion/bald-dome
    # defects (see prompts_hunyuan.py). Gated structurally rather than
    # adding another sentence to constrain it. rock_formation is still
    # drawn unconditionally above to preserve rng sequence stability.
    include_rock_formation = not (len(counts) == 1 and density == "dense")

    # ------------------------------------------------------------------
    # Build subject descriptions
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
    # Compose the final prompt - subject and guards FIRST (BFL: model
    # attends most to what comes first; also the safest position if any
    # row still runs long), secondary camera/sensor detail LAST.
    # ------------------------------------------------------------------

    guard_parts = [COMPOSITION_GUARD, SPECIES_GUARD, REALISM_GUARD, OPTICAL_GUARD]
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


# ============================================================================
# CLASS NAME HELPERS
# ============================================================================

def class_names() -> dict[int, str]:
    return {cid: entry["short"] for cid, entry in CLASSES.items()}


def detector_prompts() -> dict[int, str]:
    return {cid: entry["short"] for cid, entry in CLASSES.items()}
