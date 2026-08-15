"""Stage 1 generation - klein, prompt-fix variant.

Duplicated from generate.py rather than editing it in place, so the already-
verified klein baseline (outputs/2-pilot/klein) stays untouched and
reproducible. This file's only real difference: it imports build_prompt()
from prompts_hunyuan.py instead of the shared prompts.py, and only keeps
klein's MODELS entry (flux2dev/sd35/qwen_image/qwen_image_lightning dropped -
not relevant here, see generate.py for those).

Why: same rationale as generate_3pilot_promptfix.py (flux2dev). Compare this
run's output against outputs/2-pilot/klein (identical manifest/seeds/model
config, only the prompt text differs) to check whether prompts_hunyuan.py's
two subtractive fixes (reworded SCENE_DENSITIES "dense" phrasing, structural
rock-formation gate for single-class+dense rows) are neutral-or-better for
klein too, or Hunyuan-specific. flux2dev's own comparison (outputs/3-pilot/
flux2dev vs flux2dev_promptfix) came back net neutral - one row improved, one
row got worse on the decorative-arrangement front, no anatomical difference
either way - so don't assume this run will show a clean win either; it's a
genuine open question, not a formality.

klein is single-GPU and unquantized (~29GB bf16, fits one A100/H100 with
room to spare) - no multi-gpu/quantization complexity like flux2dev needed.

    # smoke test first, same as every other model in this pipeline
    python src/generate_klein_promptfix.py --manifest manifests/2-pilot.json --limit 3 --out outputs/3-pilot/klein_promptfix_smoke

    # full run
    python src/generate_klein_promptfix.py --manifest manifests/2-pilot.json --out outputs/3-pilot/klein_promptfix

Outputs PNG + sidecar JSON per image under outputs/<out>/ - same convention
as generate.py, so annotate.py works on these outputs unmodified.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from prompts_hunyuan import build_prompt

MODEL_KEY = "klein"
MODEL_CFG = {
    "repo": "black-forest-labs/FLUX.2-klein-base-9B",
    "pipeline": "Flux2KleinPipeline",
    "steps": 50,
    "guidance": 4.0,
    "guidance_param": "guidance_scale",
    # Flux2KleinPipeline takes negative_prompt_embeds but no text-level
    # negative_prompt. Exclusions are folded into the positive prompt.
    "supports_negative": False,
    "approx_vram_gb": 29,
}


def load_pipeline():
    import torch
    import diffusers

    pipe_cls = getattr(diffusers, MODEL_CFG["pipeline"])
    print(f"loading {MODEL_CFG['repo']} ({MODEL_CFG['pipeline']}, bf16, unquantized)...")
    pipe = pipe_cls.from_pretrained(MODEL_CFG["repo"], torch_dtype=torch.bfloat16)

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"gpu: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")

    pipe = pipe.to("cuda")
    print(f"weights fully resident, ~{MODEL_CFG['approx_vram_gb']} GB")
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None, help="override model default")
    ap.add_argument("--guidance", type=float, default=None, help="override model default")
    ap.add_argument("--limit", type=int, default=None, help="stop after N rows")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, load nothing")
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else MODEL_CFG["steps"]
    guidance = args.guidance if args.guidance is not None else MODEL_CFG["guidance"]

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"][: args.limit] if args.limit else manifest["rows"]

    out_dir = args.out or Path("outputs") / "3-pilot" / "klein_promptfix"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={MODEL_KEY}  stage=3-pilot-promptfix  rows={len(rows)}")
    print(f"steps={steps}  guidance={guidance}  resolution={args.resolution}\n")

    if args.dry_run:
        for row in rows:
            counts = {int(k): v for k, v in row["requested_counts"].items()}
            prompt, negative, metadata = build_prompt(
                counts,
                seed=row["seed"],
                density=row["density"],
                framing=row["framing"],
                supports_negative=MODEL_CFG["supports_negative"],
            )
            print(f"--- row {row['row']}  {row['image_id']}  seed={row['seed']}")
            print(f"    classes: {row['class_names']}  counts: {counts}")
            print(f"    prompt: {prompt}")
            print(f"    metadata: {asdict(metadata)}")
            print()
        return

    import torch

    pipe = load_pipeline()

    durations: list[float] = []
    for row in rows:
        counts = {int(k): v for k, v in row["requested_counts"].items()}
        prompt, negative, metadata = build_prompt(
            counts,
            seed=row["seed"],
            density=row["density"],
            framing=row["framing"],
            supports_negative=MODEL_CFG["supports_negative"],
        )
        generator = torch.Generator(device="cpu").manual_seed(row["seed"])

        call_kwargs = {
            "prompt": prompt,
            "height": args.resolution,
            "width": args.resolution,
            "num_inference_steps": steps,
            MODEL_CFG["guidance_param"]: guidance,
            "generator": generator,
        }
        if MODEL_CFG["supports_negative"]:
            call_kwargs["negative_prompt"] = negative

        t0 = time.perf_counter()
        image = pipe(**call_kwargs).images[0]
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)

        stem = f"{row['image_id']}_{MODEL_KEY}"
        image.save(out_dir / f"{stem}.png")
        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image_id": row["image_id"],
                    "row": row["row"],
                    "model": MODEL_KEY,
                    "model_repo": MODEL_CFG["repo"],
                    "prompt_variant": "hunyuan_fix",
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "seed": row["seed"],
                    "steps": steps,
                    "guidance": guidance,
                    "height": args.resolution,
                    "width": args.resolution,
                    "class_ids": row["class_ids"],
                    "requested_counts": row["requested_counts"],
                    "density": row["density"],
                    "framing": row["framing"],
                    "cpu_offload": False,
                    "seconds": round(elapsed, 2),
                    "prompt_metadata": asdict(metadata),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{row['row']:>2}/{len(rows)}] {stem}.png  {elapsed:.1f}s  {row['class_names']}")

    total = sum(durations)
    print(f"\ndone: {len(durations)} images in {total/60:.1f} min "
          f"({total/len(durations):.1f}s/image) -> {out_dir}")


if __name__ == "__main__":
    main()
