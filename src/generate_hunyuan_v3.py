"""Stage 1 generation - HunyuanImage-3.0, EXPERIMENTAL - camera_height forced
to "far" (5-8m above the seabed) plus a subject-scale guard.

Duplicated from generate_hunyuan_v2.py rather than editing it in place (that
file already produced real output - smoke_hunyuan_povfix - same
"duplicate, don't edit files that produced real output" rule used throughout
this project). Imports build_prompt() from prompts_hunyuan_v3.py - see that
file's module docstring for the full rationale, same as flux2dev's
prompts_v6.py: two prior camera-distance experiments (framing swap,
camera_height="high") both came back visually negative, both stayed in a
narrow 0.5-2m band. This forces camera_height="far" (~5-8m,
CAMERA_HEIGHTS["far"] in prompts_hunyuan_v3.py) on every row, plus
SUBJECT_SCALE_GUARD (explicit "small in frame" language, untested lever).
NOT touching water colour/clarity - Stage 3 owns that separately.

Everything else (model loading/call path, VRAM/node requirements, separate
conda env) is unchanged from generate_hunyuan.py/generate_hunyuan_v2.py -
see those files' docstrings for the full HunyuanImage-3.0 integration
notes. No token cap to re-verify here (Hunyuan has none), unlike flux2dev's
prompts_v6.py which needed real trimming to stay under FLUX.2's 512-token
cap after the same two additions.

NOT YET RUN.

    # smoke test first
    python src/generate_hunyuan_v3.py --manifest manifests/2-pilot.json --limit 3 --out outputs/hunyuan/v3/smoke

    # full run
    python src/generate_hunyuan_v3.py --manifest manifests/2-pilot.json --out outputs/hunyuan/v3

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

from prompts_hunyuan_v3 import build_prompt

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

    out_dir = args.out or Path("outputs") / "hunyuan" / "v3"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model=hunyuan  prompt_variant=hunyuan_v3_far  rows={len(rows)}\n")

    if args.dry_run:
        for row in rows:
            counts = {int(k): v for k, v in row["requested_counts"].items()}
            prompt, metadata = build_prompt(
                counts,
                seed=row["seed"],
                density=row["density"],
                framing=row["framing"],
                camera_height="far",
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
            camera_height="far",
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
                    "prompt_variant": "hunyuan_v3_far_camera_5to8m",
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
