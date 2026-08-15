"""Ad hoc test script for tencent/HunyuanImage-3.0-Instruct - NOT part of the
generation pipeline. Aligns with prompts_hunyuan_v4.py's revert: the
original test (test_hunyuan_instruct.py) used prompts_hunyuan_v2's
elaborate first-person-POV wording for its structured prompt and measured
ZERO equipment hallucination in that one sample - the only Hunyuan output
so far (Pretrain or Instruct) with a clean record on this defect. That's
suggestive Instruct may just be robust to this regardless of prompt
wording (its own reasoning/rewrite step likely normalizes the prompt before
generating), but n=1 isn't confirmation either way. This version switches
the structured prompt to prompts_hunyuan_v4.py (the evidence-based revert -
see that file's docstring for the 4%-vs-33% diagnosis) for consistency with
what Pretrain is now using, and drops camera_height="far" forcing (v3's
separate, unrelated experiment) so this test isn't conflating two variables.
RAW_INSTRUCTION's hand-written wording is also reworded to drop the same
"as if the viewer is looking through the camera's own lens" elaborate
framing that's implicated in Pretrain's regression - untested whether this
matters for Instruct, but no reason to keep wording under suspicion when a
plainer version is just as easy to write.

Duplicated from test_hunyuan_instruct_v2.py rather than editing it in place
- that file was queued as a real job (cancelled before running, but still
following the "duplicate" convention rather than assuming that's fine to
skip).

Instruct is a different checkpoint from the Pretrain one this project
otherwise uses (tencent/HunyuanImage-3.0, see generate_hunyuan.py/
generate_hunyuan_v2.py's module docstrings for that distinction). This file
exists purely to test whether Instruct's own reasoning/rewrite pipeline
("Prompt Self-Rewrite") does a better job than this project's hand-engineered
prompt text - it is not wired into build_manifest/annotate/anything else.

Confirmed API (huggingface.co/tencent/HunyuanImage-3.0-Instruct, 2026-08-13):
    model.generate_image(
        prompt=prompt, seed=42, image_size="auto",
        use_system_prompt="en_unified", bot_task="think_recaption",
        diff_infer_steps=50, verbose=2,
    )
This is a DIFFERENT call signature from Pretrain's generate_image() (see
generate_hunyuan.py) in two ways: it takes use_system_prompt/bot_task (inert
no-ops on Pretrain - see the "DeepSeek and Instruct" explanation given in
chat), and it returns a (cot_text, samples) TUPLE, not a single PIL.Image -
cot_text is the model's own reasoning/rewritten prompt (only populated for
bot_task modes that include reasoning, i.e. "think_recaption"), samples is a
list of PIL.Image. This script prints cot_text specifically so you can see
what Instruct rewrote the prompt INTO before generating - that's the whole
point of testing this checkpoint.

use_system_prompt options: None, "dynamic", "en_vanilla", "en_recaption",
"en_think_recaption", "en_unified" (default/recommended), "custom".
bot_task options: "image" (direct generation, no rewrite), "recaption"
(rewrite then generate, no reasoning), "think_recaption" (reason, then
rewrite, then generate - the full-power path, used here).

TWO PROMPTS ARE TESTED, DELIBERATELY DIFFERENT IN STYLE:

1. RAW_INSTRUCTION - a short, natural-language instruction, not our usual
   guard-laden structured prompt. This is the point of testing Instruct at
   all: its "think_recaption" reasoning step is meant to do the elaboration
   this project currently hand-engineers in prompts_v4.py/prompts_hunyuan_v2.py
   (subject/environment/lighting/camera detail, positive-only guards, etc).
   If Instruct's own reasoning does that job well, most of this project's
   hand-written scaffolding may be unnecessary for this checkpoint - that is
   exactly the question this test is designed to answer, not assumed.

2. STRUCTURED_PROMPT - the SAME hand-engineered prompt prompts_hunyuan_v2.py
   would build for the identical request (5 starfish, 3 sea urchins, dense,
   close-up framing - matches manifests/2-pilot.json row 7's shape), run
   through the same think_recaption pipeline, for a direct side-by-side: does
   Instruct's reasoning step help, hurt, or do nothing when it's already
   given a fully-specified prompt instead of a short instruction?

Compare: does the RAW_INSTRUCTION version (short input, model does the work)
match or beat the STRUCTURED_PROMPT version (already-elaborated input) on
count fidelity / anatomy / camera-POV / ground-contact - the exact defects
prompts_hunyuan_v2.py was built to fix by hand. If Instruct's reasoning
handles those on its own from a short instruction, that's a real finding
worth acting on; if not, Pretrain + hand-engineered prompts remains the
right choice for this pipeline's batch generation.

NOT YET RUN. No VRAM/node requirements confirmed for the Instruct checkpoint
specifically - assume similar to Pretrain's ~181GB fp16 footprint (H100NVL/
A100 4-GPU node) until checked; Instruct's added reasoning path may need
more.

    python src/test_hunyuan_instruct.py --which raw
    python src/test_hunyuan_instruct.py --which structured
    python src/test_hunyuan_instruct.py --which both
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_REPO = "tencent/HunyuanImage-3.0-Instruct"

SEED = 42007  # matches manifests/2-pilot.json row 7's seed, for a like-for-like comparison

# Same request as manifest row 7: 5 starfish, 3 sea urchins... actually row 7
# is all-three-classes dense/close-up (2 starfish, 2 urchin, 3 scallop) - see
# manifests/2-pilot.json. Kept here as plain language, not fragments.
RAW_INSTRUCTION = (
    "Generate a photorealistic underwater seabed photograph, shot from a "
    "stationary benthic survey camera near the seafloor. Show exactly 2 "
    "starfish, 2 sea urchins and 3 scallops, resting naturally and in "
    "direct physical contact with the substrate, on a rocky, moderately "
    "algae-covered temperate coastal seabed with sand and gravel. "
    "Documentary survey style, candid and unposed, not a specimen display "
    "- no divers, boats, or CG rendering. Natural diffuse daylight, "
    "true-to-life colour, full frame with no vignette."
)


def _build_structured_prompt() -> str:
    """The same request, built through prompts_hunyuan_v4.py's pipeline
    (the equipment-hallucination revert, random camera_height - matches
    what generate_hunyuan_v4.py now sends Pretrain) - for a direct
    side-by-side against RAW_INSTRUCTION above."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from prompts_hunyuan_v4 import build_prompt

    counts = {0: 2, 1: 2, 2: 3}
    prompt, _ = build_prompt(counts, seed=SEED, density="dense", framing="close-up")
    return prompt


def _generate(model, prompt: str, label: str, out_dir: Path) -> None:
    print(f"\n=== {label} ===")
    print(f"prompt ({len(prompt.split())} words): {prompt}\n")

    cot_text, samples = model.generate_image(
        prompt=prompt,
        seed=SEED,
        image_size="auto",
        use_system_prompt="en_unified",
        bot_task="think_recaption",
        diff_infer_steps=50,
        verbose=2,
    )

    if cot_text:
        print(f"--- Instruct's own reasoning/rewrite (cot_text) ---\n{cot_text}\n")
    else:
        print("(no cot_text returned)")

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(samples):
        path = out_dir / f"instruct_{label}_{i}.png"
        img.save(path)
        print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["raw", "structured", "both"], default="both")
    ap.add_argument("--out", type=Path, default=Path("outputs") / "hunyuan" / "instruct_v3")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    print(f"loading {MODEL_REPO} (AutoModelForCausalLM, auto dtype, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        moe_impl="eager",
    )
    # Confirmed bug in the Instruct checkpoint's shipped config.json (not our
    # code): it's simply missing "model_version", which the Pretrain
    # checkpoint's config.json HAS ("HunyuanImage-3.0") and which this
    # model's own load_tokenizer() reads unconditionally
    # (self.config.model_version, no fallback - see modeling_hunyuan_image_3.py
    # line ~1793). Confirmed on Stanage 2026-08-14: crashes with
    # "AttributeError: 'HunyuanImage3Config' object has no attribute
    # 'model_version'" without this patch. HunyuanImage3TokenizerFast.__init__
    # takes **kwargs generically with no validation on this value, so setting
    # it to match the Pretrain checkpoint's own naming convention is a safe,
    # targeted workaround - not a guess at what "should" work, matched to
    # what Tencent's own Pretrain config actually contains.
    if not hasattr(model.config, "model_version"):
        print("model.config missing 'model_version' (known gap in the Instruct "
              "checkpoint's config.json) - patching to 'HunyuanImage-3.0-Instruct'")
        model.config.model_version = "HunyuanImage-3.0-Instruct"

    model.load_tokenizer(MODEL_REPO)

    if args.which in ("raw", "both"):
        _generate(model, RAW_INSTRUCTION, "raw_instruction_v4", args.out)
    if args.which in ("structured", "both"):
        _generate(model, _build_structured_prompt(), "structured_prompt_v4", args.out)


if __name__ == "__main__":
    main()
