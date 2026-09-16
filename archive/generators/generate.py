"""Stage 0.5 / Stage 1 image generation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from imggen.prompts.legacy.flux2dev_v2 import build_prompt

MODELS = {
    "klein": {
        "repo": "black-forest-labs/FLUX.2-klein-base-9B",
        "pipeline": "Flux2KleinPipeline",
        "steps": 50,
        "guidance": 4.0,
        "guidance_param": "guidance_scale",
        "approx_vram_gb": 29,
        "quantize_components": [],
        "lora": None,
    },
    "flux2dev": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "pipeline": "Flux2Pipeline",
        "steps": 50,
        "guidance": 4.0,
        "guidance_param": "guidance_scale",
        "approx_vram_gb": 32,
        "quantize_components": ["transformer", "text_encoder"],
        "lora": None,
    },
    "sd35": {
        "repo": "stabilityai/stable-diffusion-3.5-large",
        "pipeline": "StableDiffusion3Pipeline",
        "steps": 28,
        "guidance": 4.5,
        "guidance_param": "guidance_scale",
        "approx_vram_gb": 16,
        "quantize_components": [],
        "lora": None,
    },
    "qwen_image": {
        "repo": "Qwen/Qwen-Image",
        "pipeline": "DiffusionPipeline",
        "steps": 50,
        "guidance": 4.0,
        "guidance_param": "true_cfg_scale",
        "approx_vram_gb": 40,
        "quantize_components": [],
        "lora": None,
    },
    "qwen_image_lightning": {
        "repo": "Qwen/Qwen-Image",
        "pipeline": "DiffusionPipeline",
        "steps": 8,
        "guidance": 1.0,
        "guidance_param": "true_cfg_scale",
        "approx_vram_gb": 40,
        "quantize_components": [],
        "lora": {
            "repo": "lightx2v/Qwen-Image-Lightning",
            "weight_name": "Qwen-Image-Lightning-8steps-V1.0.safetensors",
        },
    },
}


def _build_quantization_config(components: list[str]):
    from diffusers import PipelineQuantizationConfig, TorchAoConfig
    from torchao.quantization import Float8WeightOnlyConfig

    def _config():
        cfg = TorchAoConfig(Float8WeightOnlyConfig())
        cfg.include_input_output_embeddings = False
        cfg.untie_embedding_weights = False
        return cfg

    return PipelineQuantizationConfig(
        quant_mapping={name: _config() for name in components}
    )


def load_pipeline(model_key: str, offload_mode: str):
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

    if cfg["lora"]:
        print(f"loading LoRA {cfg['lora']['repo']} ({cfg['lora']['weight_name']})...")
        pipe.load_lora_weights(cfg["lora"]["repo"], weight_name=cfg["lora"]["weight_name"])

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
    print(f"steps={steps}  guidance={guidance}  resolution={args.resolution}\n")

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

    import torch

    pipe, offloaded = load_pipeline(args.model, args.offload)

    durations: list[float] = []
    for row in rows:
        counts = {int(k): v for k, v in row["requested_counts"].items()}
        prompt, metadata = build_prompt(
            counts,
            seed=row["seed"],
            density=row["density"],
            framing=row["framing"],
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
