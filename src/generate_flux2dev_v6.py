"""Stage 1 generation - flux2dev, EXPERIMENTAL - camera_height forced to
"far" (5-8m above the seabed), a much bigger jump than the earlier "high"
(1.5-2m) test, plus a new subject-scale guard.

Duplicated from generate_3pilot_v5.py rather than editing it in place (same
"duplicate, don't edit shared files" rule used throughout this project).
Imports build_prompt() from prompts_v6.py - see that file's module
docstring for the full rationale: two prior attempts at fixing the "product
photography" look (framing label swap, camera_height="high") both came back
visually negative, both stayed in a narrow 0.5-2m band. This test escalates
camera_height to "far" (~5-8m, CAMERA_HEIGHTS["far"] in prompts_v6.py) and
adds SUBJECT_SCALE_GUARD (explicit "small in frame" language) as a second,
previously-untested lever, since camera-position text alone has failed
twice.

NOT touching water colour/clarity - SCENE_WATER_PHRASE stays "clear water,
true-to-life colour" unchanged. Stage 3's Akkaynak-Treibitz transform
already owns depth-based colour degradation as a separate post-process; see
prompts_v6.py's docstring for why doing it here too would double-degrade.

Token budget re-verified after both additions - see prompts_v6.py's
docstring. Worst case over manifests/2-pilot.json (what this script
actually runs) is 479/512 - real margin, comparable to v5's own 478/512.
A broader 5000-seed brute force (3-class, dense, wide - the worst-case row
shape) came in at 504/512, so the true ceiling is closer than earlier
versions but still under the cap; watch this if extending prompts_v6.py
further rather than assuming there's lots of room left.

Defaults output to outputs/5-pilot/flux2dev_v6_far.

flux2dev's combined bf16 footprint (~32B transformer + ~24B Mistral Small 3.1
text encoder) is ~106-112GB - too large for any single Stanage GPU (A100/H100
80GB, H100NVL 94GB), so it needs 2 GPUs. Tries diffusers' device_map="balanced"
first (the diffusers-managed split, most likely to correctly handle moving
intermediate tensors between devices); falls back to manual per-component
placement if that pipeline class doesn't support it in the pinned
diffusers==0.39.0. UNVERIFIED which path actually works until tested on
Stanage - 2-pilot never needed multi-GPU, this is new territory for this
pipeline. The manual fallback in particular is a best-effort attempt, not a
guaranteed-correct path: diffusers pipelines only handle cross-device tensor
transfer automatically when device_map's own dispatch machinery is what did
the placement, not when components are just individually .to()'d after the
fact. If it errors with a "tensors on different devices" RuntimeError, that's
this fallback being wrong, not a mystery bug - see the comment on
_load_flux2dev_multi_gpu() below before debugging from scratch.

    # smoke test first, same as every other model in this pipeline
    python src/generate_flux2dev_v6.py --model flux2dev --manifest manifests/2-pilot.json --limit 3 --out outputs/flux2dev/v6/smoke

    # full run
    python src/generate_flux2dev_v6.py --model flux2dev --manifest manifests/2-pilot.json --out outputs/flux2dev/v6

Outputs PNG + sidecar JSON per image under outputs/<stage>/<model>/ - same
convention as generate.py, so annotate.py works on these outputs unmodified.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from prompts_flux2dev_v6 import build_prompt

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
    forward pass. This is a generic DiffusionPipeline.from_pretrained()
    capability for pipelines built from standard ModelMixin/PreTrainedModel
    components (which Flux2Pipeline's transformer and text_encoder are), not
    a pipeline-specific opt-in - expected to work, not confirmed for this
    exact diffusers==0.39.0 + Flux2Pipeline combination until run for real.

    Fallback: manual placement (text_encoder -> cuda:1, transformer/vae ->
    cuda:0). This is NOT guaranteed correct - diffusers pipelines only
    auto-handle cross-device tensor transfer when device_map's own dispatch
    machinery did the placement. Manually .to()-ing components after loading
    can crash mid-forward-pass with a "tensors on different devices"
    RuntimeError if Flux2Pipeline.__call__ doesn't already move intermediate
    hidden states to whatever device the next component expects. If that
    happens, the fix is tracing where in Flux2Pipeline.__call__ the
    text_encoder's output feeds into the transformer and adding an explicit
    .to(transformer.device) there (or patching a subclass) - not a sign this
    approach is unsalvageable, just that it needs that one explicit hop.
    """
    import torch
    import diffusers

    pipe_cls = getattr(diffusers, cfg["pipeline"])
    print(f"loading {cfg['repo']} ({cfg['pipeline']}, bf16, unquantized, multi-GPU)...")

    try:
        pipe = pipe_cls.from_pretrained(
            cfg["repo"], torch_dtype=torch.bfloat16, device_map="balanced",
        )
        print("multi-gpu: device_map='balanced' (diffusers-managed split)")
        return pipe, "balanced"
    except (TypeError, ValueError, NotImplementedError) as e:
        print(f"device_map='balanced' failed ({type(e).__name__}: {e}) - trying manual placement")

    pipe = pipe_cls.from_pretrained(cfg["repo"], torch_dtype=torch.bfloat16)
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

    out_dir = args.out or Path("outputs") / "flux2dev" / "v6"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}  stage=3-pilot  rows={len(rows)}")
    print(f"steps={steps}  guidance={guidance}  resolution={args.resolution}  multi_gpu={cfg['multi_gpu']}\n")

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

    import torch

    pipe, placement_mode = _load_flux2dev_multi_gpu(cfg)

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

        stem = f"{row['image_id']}_{args.model}_bf16"
        image.save(out_dir / f"{stem}.png")
        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image_id": row["image_id"],
                    "row": row["row"],
                    "model": f"{args.model}_bf16_unquantized",
                    "model_repo": cfg["repo"],
                    "prompt_variant": "v6_far_camera_5to8m",
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
    print(f"\ndone: {len(durations)} images in {total/60:.1f} min "
          f"({total/len(durations):.1f}s/image) -> {out_dir}")


if __name__ == "__main__":
    main()
