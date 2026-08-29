"""Stage 1 generation - flux2dev v9, starfish-focused prompt engine
(prompts_flux2dev_v9_starfish.py) instead of v8's. Everything else - crash-
resume, multi-GPU bf16 loading, manifest/row handling - is identical to
generate_flux2dev_v9.py; only the prompt import changed. Duplicated rather
than parameterizing generate_flux2dev_v9.py's import, matching this
pipeline's established convention of one file per prompt-engine version
(see prompts_flux2dev_v8.py's predecessors).

Targets the same production manifest as the v9 base run
(manifests/benthic-survey-1000-flux2dev.json) so this is a direct,
apples-to-apples regeneration of the same 1000 (seed, counts, density,
framing) requests under the new starfish-camouflage prompt wording - not a
new sample.

    # smoke test first
    python -m imggen.generate.flux2dev_v9_starfish --model flux2dev --manifest manifests/benthic-survey-1000-flux2dev.json --limit 3 --out outputs/flux2dev/v9_starfish/smoke

    # full run (safe to re-run after an interruption - already-done rows are skipped)
    python -m imggen.generate.flux2dev_v9_starfish --model flux2dev --manifest manifests/benthic-survey-1000-flux2dev.json --out outputs/flux2dev/v9_starfish

Outputs PNG + sidecar JSON per image under outputs/<stage>/<model>/ - same
convention as generate.py, so annotate.py works on these outputs unmodified.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from imggen.prompts.flux2dev_v9_starfish import build_prompt

# The production manifest (built for v8's 3-way {close-up, mid, wide}
# FRAMING vocabulary) has 345/1000 rows requesting "close-up" - but the
# starfish prompt engine deliberately dropped "close-up" from its own
# FRAMING dict as part of the SUBJECT_SCALE_GUARD fix (diagnosis finding:
# generated objects sit too large/close in frame vs real DUO photos), so it
# only defines {mid, wide}. Remap here rather than reintroducing "close-up"
# text into the prompt engine, which would undo that fix. "mid" is the
# closer of the two remaining options, so it's the natural downgrade target.
FRAMING_COMPAT = {"close-up": "mid"}

MODELS = {
    "flux2dev": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "pipeline": "Flux2Pipeline",
        "steps": 50,
        "guidance": 4.0,
        "guidance_param": "guidance_scale",
        # ~106-112GB combined bf16 (transformer + text_encoder) - no
        # quantization this round, split across 2 GPUs instead. See
        # _load_flux2dev_multi_gpu().
        "approx_vram_gb": 112,
        "multi_gpu": True,
        "lora": None,
    },
}


def _load_flux2dev_multi_gpu(cfg: dict):
    """Load flux2dev at full bf16 precision across 2 GPUs, no quantization.

    Primary path: diffusers' device_map="balanced", which lets diffusers'
    own accelerate-backed dispatch decide the split and - critically -
    correctly handles moving intermediate tensors between devices during the
    forward pass.

    Fallback: manual placement (text_encoder -> cuda:1, transformer/vae ->
    cuda:0). This is NOT guaranteed correct - see generate_flux2dev_v7.py's
    identical comment for the full explanation of why, unchanged here.
    """
    import torch
    import diffusers

    pipe_cls = getattr(diffusers, cfg["pipeline"])
    print(f"loading {cfg['repo']} ({cfg['pipeline']}, bf16, unquantized, multi-GPU)...")

    try:
        pipe = pipe_cls.from_pretrained(
            cfg["repo"], torch_dtype=torch.bfloat16, device_map="balanced",
            local_files_only=True,
        )
        print("multi-gpu: device_map='balanced' (diffusers-managed split)")
        return pipe, "balanced"
    except (TypeError, ValueError, NotImplementedError) as e:
        print(f"device_map='balanced' failed ({type(e).__name__}: {e}) - trying manual placement")

    pipe = pipe_cls.from_pretrained(cfg["repo"], torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.text_encoder.to("cuda:1")
    pipe.transformer.to("cuda:0")
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.to("cuda:0")
    print("multi-gpu: manual placement (text_encoder->cuda:1, transformer/vae->cuda:0) - UNVERIFIED, watch for device-mismatch errors")
    return pipe, "manual"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None, help="override model default")
    ap.add_argument("--guidance", type=float, default=None, help="override model default")
    ap.add_argument("--limit", type=int, default=None, help="stop after N rows")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, load nothing")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    steps = args.steps if args.steps is not None else cfg["steps"]
    guidance = args.guidance if args.guidance is not None else cfg["guidance"]

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"][: args.limit] if args.limit else manifest["rows"]
    stage = manifest.get("stage", "unknown")

    out_dir = args.out or Path("outputs") / "flux2dev" / "v9_starfish"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}  stage={stage}  rows={len(rows)}")
    print(f"steps={steps}  guidance={guidance}  resolution={args.resolution}  multi_gpu={cfg['multi_gpu']}\n")

    if args.dry_run:
        for row in rows:
            counts = {int(k): v for k, v in row["requested_counts"].items()}
            framing = FRAMING_COMPAT.get(row["framing"], row["framing"])
            prompt, metadata = build_prompt(
                counts,
                seed=row["seed"],
                density=row["density"],
                framing=framing,
                camera_height="far",
            )
            print(f"--- row {row['row']}  {row['image_id']}  seed={row['seed']}")
            print(f"    classes: {row['class_names']}  counts: {counts}")
            print(f"    prompt: {prompt}")
            print(f"    metadata: {asdict(metadata)}")
            print()
        return

    import torch

    pipe, placement_mode = _load_flux2dev_multi_gpu(cfg)

    durations: list[float] = []
    skipped = 0
    for row in rows:
        stem = f"{row['image_id']}_{args.model}_bf16"
        png_path = out_dir / f"{stem}.png"
        json_path = out_dir / f"{stem}.json"
        if png_path.exists() and json_path.exists():
            skipped += 1
            print(f"[{row['row']:>2}/{len(rows)}] {stem}.png  SKIP (already exists)")
            continue

        counts = {int(k): v for k, v in row["requested_counts"].items()}
        framing = FRAMING_COMPAT.get(row["framing"], row["framing"])
        prompt, metadata = build_prompt(
            counts,
            seed=row["seed"],
            density=row["density"],
            framing=framing,
            camera_height="far",
        )
        generator = torch.Generator(device="cpu").manual_seed(row["seed"])

        call_kwargs = {
            "prompt": prompt,
            "height": args.resolution,
            "width": args.resolution,
            "num_inference_steps": steps,
            cfg["guidance_param"]: guidance,
            "generator": generator,
        }

        t0 = time.perf_counter()
        image = pipe(**call_kwargs).images[0]
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)

        image.save(png_path)
        json_path.write_text(
            json.dumps(
                {
                    "image_id": row["image_id"],
                    "row": row["row"],
                    "model": f"{args.model}_bf16_unquantized",
                    "model_repo": cfg["repo"],
                    "prompt_variant": "v9_starfish",
                    "prompt": prompt,
                    "seed": row["seed"],
                    "steps": steps,
                    "guidance": guidance,
                    "height": args.resolution,
                    "width": args.resolution,
                    "class_ids": row["class_ids"],
                    "requested_counts": row["requested_counts"],
                    "density": row["density"],
                    "framing": framing,
                    "manifest_framing": row["framing"],
                    "quantization": None,
                    "multi_gpu_placement": placement_mode,
                    "seconds": round(elapsed, 2),
                    "prompt_metadata": asdict(metadata),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{row['row']:>2}/{len(rows)}] {stem}.png  {elapsed:.1f}s  {row['class_names']}")

    total = sum(durations)
    print(f"\ndone: {len(durations)} generated, {skipped} skipped (already existed), "
          f"{total/60:.1f} min ({total/len(durations):.1f}s/image if any generated) -> {out_dir}"
          if durations else f"\ndone: 0 generated, {skipped} skipped (already existed) -> {out_dir}")


if __name__ == "__main__":
    main()
