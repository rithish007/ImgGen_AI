"""
Stage 1 generation - HunyuanImage-3.0 v6
==========================================

Generates the v6 habitat-first underwater survey dataset.

Key changes from v5
-------------------
- Imports prompts_hunyuan_v6.py.
- Keeps the camera-to-seabed distance strictly in the 5-8 m range.
- Uses camera_height='far' only as a compatibility alias; the prompt engine
  resolves it to an exact 5, 6, 7 or 8 m survey height per seed.
- Removes the legacy close/mid/wide framing influence. Manifest `framing`
  remains readable and is preserved as metadata, but no longer controls prompt
  geometry.
- Saves the exact generated prompt and full prompt metadata beside each PNG.

The model/API path follows the working HunyuanImage-3.0 v5 driver: startup
introspection is retained, `seed` is passed directly, and generation uses
stream=True.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path

from prompts_hunyuan_v6 import build_prompt

MODEL_REPO = "tencent/HunyuanImage-3.0"
PROMPT_VARIANT = "hunyuan_v6_habitat_first_survey_5_8m"


def run_startup_introspection(model) -> dict:
    """Print the live generate_image() signature before generation."""
    sig = inspect.signature(model.generate_image)
    print(f"model.generate_image signature: {sig}")
    accepted = set(sig.parameters.keys())
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    print(
        "  accepts **kwargs (params beyond the listed ones may still work): "
        f"{has_var_kwargs}"
    )
    return {
        "accepted_params": sorted(accepted),
        "accepts_var_kwargs": has_var_kwargs,
    }


def _generate_one(model, prompt: str, seed: int):
    """Generate one image using the confirmed Hunyuan API."""
    return model.generate_image(prompt=prompt, seed=seed, stream=True)


def _build_for_row(row: dict) -> tuple[str, object]:
    counts = {int(k): v for k, v in row["requested_counts"].items()}

    # Force the mandatory survey-distance regime. The prompt engine converts
    # "far" into a deterministic 5/6/7/8 m selection based on the row seed.
    return build_prompt(
        counts,
        seed=row["seed"],
        density=row["density"],
        framing=row.get("framing"),
        camera_height="far",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: outputs/hunyuan/v6)",
    )
    ap.add_argument("--limit", type=int, default=None, help="stop after N rows")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print prompts and metadata without loading the model",
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"][: args.limit] if args.limit else manifest["rows"]

    out_dir = args.out or Path("outputs") / "hunyuan" / "v6"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"model=hunyuan  prompt_variant={PROMPT_VARIANT}  rows={len(rows)}\n"
    )

    if args.dry_run:
        for row in rows:
            prompt, metadata = _build_for_row(row)
            print(f"--- row {row['row']}  {row['image_id']}  seed={row['seed']}")
            print(f"    classes: {row['class_names']}  counts: {row['requested_counts']}")
            print(
                f"    survey height: {metadata.camera_height_m} m  "
                f"density: {metadata.density}  difficulty: {metadata.difficulty}"
            )
            print(f"    legacy framing (ignored): {metadata.legacy_framing}")
            print(f"    prompt: {prompt}")
            print(f"    metadata: {asdict(metadata)}")
            print()
        return

    from transformers import AutoModelForCausalLM

    print(
        f"loading {MODEL_REPO} (AutoModelForCausalLM, bf16/fp16 auto, "
        "device_map=auto, unquantized)..."
    )
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

    api_info = run_startup_introspection(model)

    durations: list[float] = []
    heights: list[int] = []

    for row in rows:
        prompt, metadata = _build_for_row(row)
        heights.append(metadata.camera_height_m)

        t0 = time.perf_counter()
        image = _generate_one(model, prompt, row["seed"])
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)

        stem = f"{row['image_id']}_hunyuan"
        image.save(out_dir / f"{stem}.png")

        sidecar = {
            "image_id": row["image_id"],
            "row": row["row"],
            "model": "hunyuan_image_3_bf16_unquantized",
            "model_repo": MODEL_REPO,
            "prompt_variant": PROMPT_VARIANT,
            "prompt": prompt,
            "seed": row["seed"],
            "class_ids": row["class_ids"],
            "requested_counts": row["requested_counts"],
            "density": row["density"],
            "framing": row.get("framing"),
            "legacy_framing_used": False,
            "camera_height_m": metadata.camera_height_m,
            "quantization": None,
            "device_map": getattr(model, "hf_device_map", None),
            "generate_image_api": api_info,
            "seconds": round(elapsed, 2),
            "prompt_metadata": asdict(metadata),
        }

        (out_dir / f"{stem}.json").write_text(
            json.dumps(sidecar, indent=2, default=str),
            encoding="utf-8",
        )

        print(
            f"[{row['row']:>2}/{len(rows)}] {stem}.png  "
            f"{elapsed:.1f}s  height={metadata.camera_height_m}m  "
            f"{row['class_names']}"
        )

    total = sum(durations)
    if durations:
        height_counts = {height: heights.count(height) for height in sorted(set(heights))}
        print(
            f"\ndone: {len(durations)} images in {total/60:.1f} min "
            f"({total/len(durations):.1f}s/image) -> {out_dir}"
        )
        print(f"survey heights used: {height_counts}")


if __name__ == "__main__":
    main()
