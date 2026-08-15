"""Stage 1 generation - 5-pilot: HunyuanImage-3.0 with the Hunyuan-specific
prompt fix (prompts_hunyuan_v2.py).

Duplicated from generate_hunyuan.py rather than editing it in place (same
"duplicate, don't edit shared files" rule used throughout this project) - the
only real difference is the prompt engine import and the default output
directory. Everything about the model loading/call path is identical and
still applies unchanged; see generate_hunyuan.py's module docstring for the
full HunyuanImage-3.0 integration notes (AutoModelForCausalLM, confirmed
generate_image() signature, VRAM/node requirements, separate conda env).

Prompt engine: uses prompts_hunyuan_v2.py, not prompts_v2.py. Fixes the two
defect threads found auditing 5-pilot/hunyuan (prompts_v2 run): survey
equipment (a robotic camera rig/arm) hallucinating into frame on images 16
and 19, and a "product photography" look (camera reads too close, organisms
read as sitting on top of the substrate rather than resting on it - reported
directly: "Most starfish and scallop are hovering a little bit from the
ground"). See prompts_hunyuan_v2.py's module docstring for the full
root-cause analysis and exactly what changed. Unlike prompts_v2/v3, this
engine is NOT length-budgeted against a hard token cap - HunyuanImage-3.0 has
no confirmed hard limit (training captions ranged 30-1000 words), so this
file's prompts run longer (~250-320 words) and more explicit than
flux2dev/klein's.

NOT YET RUN - no smoke test yet. Same practice as every other model in this
pipeline: smoke test (--limit 3) before a full run, and do not assume the
prompt fix works just because it imports cleanly and dry-run output reads
sensibly.

    # smoke test first
    python src/generate_hunyuan_v2.py --manifest manifests/2-pilot.json --limit 3 --out outputs/hunyuan/v2/smoke

    # full run
    python src/generate_hunyuan_v2.py --manifest manifests/2-pilot.json --out outputs/hunyuan/v2

Outputs PNG + sidecar JSON per image under outputs/<out>/ - same file-naming
convention as generate_hunyuan.py, so annotate.py works on these outputs
unmodified.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path

from prompts_hunyuan_v2 import build_prompt

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
    """seed is a confirmed, directly-supported kwarg - see generate_hunyuan.py's
    module docstring (inspect.signature() captured on the live model during
    the 3-pilot smoke test). No detection dance needed."""
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

    out_dir = args.out or Path("outputs") / "hunyuan" / "v2"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model=hunyuan  prompt_variant=hunyuan_v2  rows={len(rows)}\n")

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
                    "prompt_variant": "hunyuan_v2_pov_ground_contact_fix",
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
