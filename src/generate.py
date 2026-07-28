"""Stage 0.5 / Stage 1 image generation.

klein, plus flux2dev as a future candidate. SD3.5 was dropped after the
smoke-test comparison: it botched sea urchin spine anatomy (short blunt bumps
rather than thin sharp spines, a real shape defect) and consistently shot
every class as an isolated macro product photo rather than the wide-angle
survey-camera framing this pipeline needs - Klein respected that framing and
SD3.5 did not.

flux2dev and qwenimage were added after the 20-image klein pilot showed a
visible instance-count overshoot (pilot_012 asked for 7 starfish, rendered
~9-10), to smoke-test as possible replacements. qwenimage was dropped after
that test and its weights deleted from the pod: it rendered only ~5 starfish
for the same 7-instance prompt (undershoot, not a fix) and additionally put a
visible robot/rover in frame - an out-of-distribution object klein never
produces, and worse than klein's problem, not better. flux2dev remains a
**future** candidate, not yet tested: its transformer+text-encoder combo needs
`enable_model_cpu_offload()` to fit at all (see below), and that offload path
already crashed this pod once on qwenimage's pre-quantization attempt by
exceeding the container's real RAM cap (~58GB via /sys/fs/cgroup/memory.max,
not the much larger host total `free -h` reports) - worth retrying on a
higher-RAM pod. See the plan doc's Stage 1 section for the full comparison.

  - flux2dev: black-forest-labs/FLUX.2-dev, the 32B undistilled model Klein
    was distilled from. Its Mistral Small 3.1 (~24B) text encoder makes the
    combined footprint too large for a 47GB card in plain bf16, so both the
    transformer and text encoder are fp8-quantized on load (see
    `_build_quantization_config`) - needs the `torchao` dependency and a GPU
    with compute capability >= 8.9 (RTX 4090/6000 Ada and newer) for native
    fp8 tensor cores. Same non-commercial BFL license as klein.

    # no GPU needed - check prompts before you pay for a pod
    python src/generate.py --model klein --manifest manifests/smoke.json --dry-run

    # on the pod
    python src/generate.py --model klein    --manifest manifests/smoke.json
    python src/generate.py --model flux2dev --manifest manifests/smoke.json
    python src/generate.py --model klein --manifest manifests/pilot.json

Outputs PNG + sidecar JSON per image under outputs/<stage>/<model>/.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
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
        "quantize_components": [],
    },
    "flux2dev": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "pipeline": "Flux2Pipeline",
        "steps": 50,
        "guidance": 4.0,
        # Same situation as klein: no text-level negative_prompt on this
        # pipeline either (checked via inspect.signature(Flux2Pipeline.__call__)).
        "supports_negative": False,
        # fp8 transformer (~32GB) + fp8 text encoder (~24GB) never coexist on
        # GPU at once - load_pipeline's offload fallback keeps peak usage near
        # whichever single component is larger, not their sum. BUT: RunPod
        # containers can cap system RAM well below what `free -h` reports (see
        # /sys/fs/cgroup/memory.max - one pod measured ~58GB against a 503GB
        # host). enable_model_cpu_offload() needs BOTH components resident in
        # CPU RAM at once (each is swapped to GPU in turn, not deleted), which
        # can exceed that cap even though neither alone would. Verify
        # memory.max before running this on a new pod.
        "approx_vram_gb": 32,
        "quantize_components": ["transformer", "text_encoder"],
    },
}


def _build_quantization_config(components: list[str]):
    """fp8 weight-only quantization for the given pipeline component names.

    Requires torchao and a >=8.9 compute capability GPU (RTX 4090/6000 Ada,
    Hopper) for native fp8 tensor cores; falls back to (slower) emulated fp8
    on older cards.
    """
    from diffusers import PipelineQuantizationConfig, TorchAoConfig
    from torchao.quantization import Float8WeightOnlyConfig

    return PipelineQuantizationConfig(
        quant_mapping={name: TorchAoConfig(Float8WeightOnlyConfig()) for name in components}
    )


def load_pipeline(model_key: str, offload_mode: str):
    """Load the pipeline, falling back to CPU offload if it will not fit.

    Weights land in CPU RAM first, so the OOM fallback costs a device transfer
    rather than a re-download. Note this fallback is not a safe last resort for
    every model - see flux2dev's RAM-cap comment above.
    """
    import torch
    import diffusers

    cfg = MODELS[model_key]
    pipe_cls = getattr(diffusers, cfg["pipeline"])

    from_pretrained_kwargs = {"torch_dtype": torch.bfloat16}
    if cfg["quantize_components"]:
        from_pretrained_kwargs["quantization_config"] = _build_quantization_config(cfg["quantize_components"])

    quant_label = f"  fp8-quantized({','.join(cfg['quantize_components'])})" if cfg["quantize_components"] else ""
    print(f"loading {cfg['repo']} ({cfg['pipeline']}, bf16{quant_label})...")
    pipe = pipe_cls.from_pretrained(cfg["repo"], **from_pretrained_kwargs)

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
            prompt, negative, metadata = build_prompt(
                counts,
                seed=row["seed"],
                density=row["density"],
                framing=row["framing"],
                supports_negative=cfg["supports_negative"],
            )
            print(f"--- row {row['row']}  {row['image_id']}  seed={row['seed']}")
            print(f"    classes: {row['class_names']}  counts: {counts}")
            print(f"    prompt: {prompt}")
            if negative:
                print(f"    negative: {negative}")
            print(f"    metadata: {asdict(metadata)}")
            print()
        return

    import torch

    pipe, offloaded = load_pipeline(args.model, args.offload)

    durations: list[float] = []
    for row in rows:
        counts = {int(k): v for k, v in row["requested_counts"].items()}
        prompt, negative, metadata = build_prompt(
            counts,
            seed=row["seed"],
            density=row["density"],
            framing=row["framing"],
            supports_negative=cfg["supports_negative"],
        )
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
