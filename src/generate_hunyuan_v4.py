"""Stage 1 generation - HunyuanImage-3.0 v4 - reverts the equipment-hallucination
"fix" back to prompts_v2.py's plain wording, which measured 4% (2/50) vs
hunyuan_v2/v3's 33% (1/3, twice). See prompts_hunyuan_v4.py's module
docstring for the full diagnosis.

Duplicated from generate_hunyuan_v3.py rather than editing it in place (same
"duplicate, don't edit files that produced real output" rule used throughout
this project - v3 already ran a smoke test). Imports build_prompt() from
prompts_hunyuan_v4.py. Does NOT force camera_height="far" - v3's camera-
distance experiment is deliberately not part of this file, so the
equipment-hallucination fix is the only variable under test. camera_height
is drawn randomly per-row, matching the proven-good 5-pilot/hunyuan
baseline this reverts to.

Everything else (model loading/call path, VRAM/node requirements, separate
conda env) is unchanged from generate_hunyuan.py - see that file's
docstring for the full HunyuanImage-3.0 integration notes. No token cap to
manage here (Hunyuan has none).

NOT YET RUN. This is a hypothesis based on comparing two full 50-image runs
against two 3-image smoke tests - needs its own run at meaningful scale
(not another 3-image smoke test) before trusting the 4% rate actually
holds for this exact prompt content.

    # smoke test first
    python src/generate_hunyuan_v4.py --manifest manifests/2-pilot.json --limit 3 --out outputs/hunyuan/v4/smoke

    # full run
    python src/generate_hunyuan_v4.py --manifest manifests/2-pilot.json --out outputs/hunyuan/v4

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

from prompts_hunyuan_v4 import build_prompt

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

    out_dir = args.out or Path("outputs") / "hunyuan" / "v4"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model=hunyuan  prompt_variant=hunyuan_v4_equipment_revert  rows={len(rows)}\n")

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
                    "prompt_variant": "hunyuan_v4_equipment_revert",
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
