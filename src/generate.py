"""Stage 0.5 / Stage 1 image generation on the RunPod RTX 5090.

Runs one manifest through one model. Never load both models at once - Klein base
is ~29GB and SD3.5-Large ~26GB in bf16, which does not fit together on 32GB.

    # no GPU needed - check prompts before you pay for a pod
    python src/generate.py --model klein --manifest manifests/smoke.json --dry-run

    # on the pod
    python src/generate.py --model klein --manifest manifests/smoke.json
    python src/generate.py --model sd35  --manifest manifests/pilot.json

Outputs PNG + sidecar JSON per image under outputs/<stage>/<model>/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from prompts import build_prompt

MODELS = {
    "klein": {
        "repo": "black-forest-labs/FLUX.2-klein-base-9B",
        "pipeline": "Flux2KleinPipeline",
        "steps": 50,
        "guidance": 4.0,
        # Flux2KleinPipeline takes negative_prompt_embeds but no text-level
        # negative_prompt. Exclusions are folded into the positive prompt.
        "supports_negative": False,
        "approx_vram_gb": 29,
    },
    "sd35": {
        "repo": "stabilityai/stable-diffusion-3.5-large",
        "pipeline": "StableDiffusion3Pipeline",
        "steps": 32,
        "guidance": 4.0,
        "supports_negative": True,
        "approx_vram_gb": 26,
    },
}


def load_pipeline(model_key: str, offload_mode: str):
    """Load the pipeline, falling back to CPU offload if it will not fit.

    Weights land in CPU RAM first, so the OOM fallback costs a device transfer
    rather than a re-download.
    """
    import torch
    import diffusers

    cfg = MODELS[model_key]
    pipe_cls = getattr(diffusers, cfg["pipeline"])

    print(f"loading {cfg['repo']} ({cfg['pipeline']}, bf16)...")
    pipe = pipe_cls.from_pretrained(cfg["repo"], torch_dtype=torch.bfloat16)

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"gpu: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")

    if offload_mode == "on":
        print("cpu offload: forced on")
        pipe.enable_model_cpu_offload()
        return pipe, True

    try:
        pipe = pipe.to("cuda")
        print(f"cpu offload: off (weights fully resident, ~{cfg['approx_vram_gb']} GB)")
        return pipe, False
    except torch.cuda.OutOfMemoryError:
        if offload_mode == "off":
            raise
        print(f"OOM placing weights on GPU ({total_gb:.1f} GB card) - enabling cpu offload")
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        return pipe, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None, help="override model default")
    ap.add_argument("--guidance", type=float, default=None, help="override model default")
    ap.add_argument("--offload", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--limit", type=int, default=None, help="stop after N rows")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, load nothing")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    steps = args.steps if args.steps is not None else cfg["steps"]
    guidance = args.guidance if args.guidance is not None else cfg["guidance"]

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"][: args.limit] if args.limit else manifest["rows"]

    out_dir = args.out or Path("outputs") / manifest["stage"] / args.model
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}  stage={manifest['stage']}  rows={len(rows)}")
    print(f"steps={steps}  guidance={guidance}  resolution={args.resolution}")
    print(f"negative_prompt supported: {cfg['supports_negative']}\n")

    if args.dry_run:
        for row in rows:
            counts = {int(k): v for k, v in row["requested_counts"].items()}
            prompt, negative = build_prompt(counts, row["framing"], cfg["supports_negative"])
            print(f"--- row {row['row']}  {row['image_id']}  seed={row['seed']}")
            print(f"    classes: {row['class_names']}  counts: {counts}")
            print(f"    prompt: {prompt}")
            if negative:
                print(f"    negative: {negative}")
            print()
        return

    import torch

    pipe, offloaded = load_pipeline(args.model, args.offload)

    durations: list[float] = []
    for row in rows:
        counts = {int(k): v for k, v in row["requested_counts"].items()}
        prompt, negative = build_prompt(counts, row["framing"], cfg["supports_negative"])
        generator = torch.Generator(device="cpu").manual_seed(row["seed"])

        call_kwargs = dict(
            prompt=prompt,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if cfg["supports_negative"]:
            call_kwargs["negative_prompt"] = negative

        t0 = time.perf_counter()
        image = pipe(**call_kwargs).images[0]
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)

        stem = f"{row['image_id']}_{args.model}"
        image.save(out_dir / f"{stem}.png")
        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image_id": row["image_id"],
                    "row": row["row"],
                    "model": args.model,
                    "model_repo": cfg["repo"],
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
                    "cpu_offload": offloaded,
                    "seconds": round(elapsed, 2),
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
