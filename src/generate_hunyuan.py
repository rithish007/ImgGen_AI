"""Stage 1 generation - 3-pilot: HunyuanImage-3.0 (tencent/HunyuanImage-3.0).

New integration, not an extension of generate.py's MODELS-dict/diffusers-
pipeline pattern - HunyuanImage-3.0 is NOT a diffusers pipeline. It's an
autoregressive native-multimodal model loaded via
transformers.AutoModelForCausalLM (confirmed from the model's own HF card),
so its call signature and available knobs are fundamentally different from
every other model in this pipeline.

Confirmed from https://huggingface.co/tencent/HunyuanImage-3.0 (2026-08-07):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="sdpa", trust_remote_code=True,
        torch_dtype="auto", device_map="auto", moe_impl="eager",
    )
    model.load_tokenizer(model_id)
    image = model.generate_image(prompt=prompt, stream=True)
    image.save("image.png")

Confirmed via inspect.signature() on the live model during the 3-pilot smoke
test (2026-08-08/09), not just from docs:
    model.generate_image signature: (prompt, seed=None, image_size='auto',
    use_system_prompt=None, system_prompt=None, bot_task=None, stream=False,
    **kwargs)
    - "seed" IS a real, directly-supported parameter - confirmed, not a
      guess. _generate_one() uses it directly now.
    - No negative_prompt / cfg parameter anywhere in this signature, nor in
      Tencent's own run_image_gen.py usage script (checked both). The
      underlying diffusion decoder does have real classifier-free guidance
      internally (ClassifierFreeGuidance class, guidance_scale, conditional/
      unconditional prediction blending - confirmed by reading
      hunyuan_image_3_pipeline.py in their repo) but nothing suggests the
      unconditional branch is user-customizable through generate_image() -
      almost certainly a fixed empty-string default. Exclusions stay folded
      into the positive prompt via POSITIVE_ONLY_GUARDS, same as klein/
      flux2dev without real negative-prompt support.

Still UNVERIFIED / assumed - check against reality, do not trust blindly:
    - Whether height/width/resolution are meaningfully controllable beyond
      image_size='auto' - NOT passed here since guessing wrong parameter
      names would just crash. run_startup_introspection() prints the real
      signature on every run specifically so each invocation doubles as API
      discovery, before assuming anything further.
    - What type generate_image() returns - assumed PIL.Image (matches the
      model card's own image.save(...) usage) until proven otherwise.
    - No official INT8/quantization support exists (checked, not present in
      docs) - this project's decision is full bf16/fp16 across multiple GPUs
      via the device_map="auto" shown above, which IS Tencent's own default,
      not a workaround. Needs a single Stanage node with enough combined
      VRAM for the ~181GB (fp16) footprint + activation/KV-cache headroom -
      H100NVL (4x94GB=376GB/node) or A100 (4x80GB=320GB/node). Plain H100
      nodes (2x80GB=160GB/node) do NOT have enough capacity in one node and
      device_map="auto" cannot span separate nodes.

Prompt engine: uses prompts_v2.py (klein and flux2dev's generate.py /
generate_3pilot.py were switched to it too, same run). Supersedes
prompts_hunyuan.py - carries its two subtractive fixes forward (dropped
density phrase, rock-formation gate) plus a much larger rewrite: every
prompt in prompts.py silently exceeded FLUX.2's 512-token hard cap (mean
670 tokens measured against the real tokenizer), so the positive guards
appended at the end were never actually reaching the model. prompts_v2.py
is restructured (subject/guards first, per BFL's own prompting guidance)
and rewritten denser specifically to fit under that cap with margin. See
prompts_v2.py's module docstring for the full rationale and the token
measurements behind it.

Needs its OWN conda env, separate from the rest of this pipeline - Tencent's
docs specify Python 3.12+, CUDA 12.8, torch==2.8.0/torchvision==0.23.0,
which may not match the pinned diffusers==0.39.0 stack the rest of this
project uses. trust_remote_code=True also means this model ships and
executes its own custom modeling code on load - be aware of that, it is not
a standard transformers architecture.

    # smoke test first - same practice as every other model in this pipeline,
    # doubly important here since this has never run in this pipeline at all
    python src/generate_hunyuan.py --manifest manifests/2-pilot.json --limit 3 --out outputs/3-pilot/hunyuan_smoke

    # full run
    python src/generate_hunyuan.py --manifest manifests/2-pilot.json --out outputs/3-pilot/hunyuan

Outputs PNG + sidecar JSON per image under outputs/<out>/ - same file-naming
convention as generate.py/generate_3pilot.py, so annotate.py works on these
outputs unmodified.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path

from prompts_flux2dev_v2 import build_prompt

MODEL_REPO = "tencent/HunyuanImage-3.0"


def run_startup_introspection(model) -> dict:
    """Print generate_image()'s real signature before trusting any assumption
    about its parameters. This is the first thing that happens after load,
    specifically so a smoke-test run doubles as API discovery rather than us
    guessing parameter names and getting a wall of stack trace instead.
    """
    sig = inspect.signature(model.generate_image)
    print(f"model.generate_image signature: {sig}")
    accepted = set(sig.parameters.keys())
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    print(f"  accepts **kwargs (params beyond the listed ones may still work): {has_var_kwargs}")
    return {"accepted_params": sorted(accepted), "accepts_var_kwargs": has_var_kwargs}


def _generate_one(model, prompt: str, seed: int):
    """seed is a confirmed, directly-supported kwarg - see module docstring
    (inspect.signature() captured on the live model during the 3-pilot smoke
    test). No detection dance needed."""
    return model.generate_image(prompt=prompt, seed=seed, stream=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="stop after N rows")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, load nothing")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"][: args.limit] if args.limit else manifest["rows"]

    out_dir = args.out or Path("outputs") / "3-pilot" / "hunyuan"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model=hunyuan  stage=3-pilot  rows={len(rows)}\n")

    if args.dry_run:
        for row in rows:
            counts = {int(k): v for k, v in row["requested_counts"].items()}
            prompt, metadata = build_prompt(
                counts,
                seed=row["seed"],
                density=row["density"],
                framing=row["framing"],
            )
            print(f"--- row {row['row']}  {row['image_id']}  seed={row['seed']}")
            print(f"    classes: {row['class_names']}  counts: {counts}")
            print(f"    prompt: {prompt}")
            print(f"    metadata: {asdict(metadata)}")
            print()
        return

    from transformers import AutoModelForCausalLM
    import torch

    print(f"loading {MODEL_REPO} (AutoModelForCausalLM, bf16/fp16 auto, device_map=auto, unquantized)...")
    t_load0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        moe_impl="eager",
    )
    model.load_tokenizer(MODEL_REPO)
    print(f"load complete in {time.perf_counter() - t_load0:.1f}s")

    if hasattr(model, "hf_device_map"):
        print(f"device_map: {model.hf_device_map}")

    run_startup_introspection(model)  # kept for visibility - flags upstream API changes early

    durations: list[float] = []
    for row in rows:
        counts = {int(k): v for k, v in row["requested_counts"].items()}
        prompt, metadata = build_prompt(
            counts,
            seed=row["seed"],
            density=row["density"],
            framing=row["framing"],
        )

        t0 = time.perf_counter()
        image = _generate_one(model, prompt, row["seed"])
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)

        stem = f"{row['image_id']}_hunyuan"
        image.save(out_dir / f"{stem}.png")
        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image_id": row["image_id"],
                    "row": row["row"],
                    "model": "hunyuan_image_3_bf16_unquantized",
                    "model_repo": MODEL_REPO,
                    "prompt": prompt,
                    "seed": row["seed"],
                    "class_ids": row["class_ids"],
                    "requested_counts": row["requested_counts"],
                    "density": row["density"],
                    "framing": row["framing"],
                    "quantization": None,
                    "device_map": getattr(model, "hf_device_map", None),
                    "seconds": round(elapsed, 2),
                    "prompt_metadata": asdict(metadata),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"[{row['row']:>2}/{len(rows)}] {stem}.png  {elapsed:.1f}s  {row['class_names']}")

    total = sum(durations)
    if durations:
        print(f"\ndone: {len(durations)} images in {total/60:.1f} min "
              f"({total/len(durations):.1f}s/image) -> {out_dir}")


if __name__ == "__main__":
    main()
