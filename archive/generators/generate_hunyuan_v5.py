"""Stage 1 generation - HunyuanImage-3.0 v5 - two fixes on top of v4: drops "shell" from COLOR_PALETTE_GUARD (closes the scallop-leak bug), and forces camera_height="far" (closes the "product photograph" framing complaint by actually using the CAMERA_HEIGHTS["far"] text that v4 wrote but never applied)."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path

from imggen.prompts.legacy.hunyuan_v5 import build_prompt

MODEL_REPO = "tencent/HunyuanImage-3.0"


def run_startup_introspection(model) -> dict:
    sig = inspect.signature(model.generate_image)
    print(f"model.generate_image signature: {sig}")
    accepted = set(sig.parameters.keys())
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    print(f"  accepts **kwargs (params beyond the listed ones may still work): {has_var_kwargs}")
    return {"accepted_params": sorted(accepted), "accepts_var_kwargs": has_var_kwargs}


def _generate_one(model, prompt: str, seed: int):
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

    out_dir = args.out or Path("outputs") / "hunyuan" / "v5"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model=hunyuan  prompt_variant=hunyuan_v5_shell_fix_camera_far  rows={len(rows)}\n")

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

    run_startup_introspection(model)

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
                    "prompt_variant": "hunyuan_v5_shell_fix_camera_far",
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
