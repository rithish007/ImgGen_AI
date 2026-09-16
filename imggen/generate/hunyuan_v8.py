"""Identical to v7 except it fixes a real gap found by measuring v7's actual output: `model.generate_image()` accepts an `image_size` parameter (confirmed via run_startup_introspection's live signature dump, and already noted as 'auto' by default all the way back in generate_hunyuan.py's docstring), but no version of this script has ever passed it."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path

from imggen.prompts.hunyuan_v7 import build_prompt

MODEL_REPO = "tencent/HunyuanImage-3.0"
PROMPT_VARIANT = "hunyuan_v7_habitat_first_survey_placement_fix"
IMAGE_SIZE = "1024x1024"


def run_startup_introspection(model) -> dict:
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
    return model.generate_image(
        prompt=prompt, seed=seed, image_size=IMAGE_SIZE, stream=True
    )


def _build_for_row(row: dict) -> tuple[str, object]:
    counts = {int(k): v for k, v in row["requested_counts"].items()}

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
        help="output directory (default: outputs/hunyuan/v8)",
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

    out_dir = args.out or Path("outputs") / "hunyuan" / "v8"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"model=hunyuan  prompt_variant={PROMPT_VARIANT}  image_size={IMAGE_SIZE}  "
        f"rows={len(rows)}\n"
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO, trust_remote_code=True, local_files_only=True
    )
    model.load_tokenizer(tokenizer)
    print(f"load complete in {time.perf_counter() - t_load0:.1f}s")

    if hasattr(model, "hf_device_map"):
        print(f"device_map: {model.hf_device_map}")

    api_info = run_startup_introspection(model)
    if "image_size" not in api_info["accepted_params"] and not api_info["accepts_var_kwargs"]:
        raise SystemExit(
            "model.generate_image() no longer accepts image_size and has no "
            "**kwargs fallback - the IMAGE_SIZE fix in this script would "
            "silently no-op or crash. Stop and re-check the live signature "
            "printed above before generating anything."
        )

    durations: list[float] = []
    heights: list[int] = []
    sizes_seen: dict[str, int] = {}
    skipped = 0

    for row in rows:
        stem = f"{row['image_id']}_hunyuan"
        png_path = out_dir / f"{stem}.png"
        json_path = out_dir / f"{stem}.json"
        if png_path.exists() and json_path.exists():
            skipped += 1
            print(f"[{row['row']:>2}/{len(rows)}] {stem}.png  SKIP (already exists)")
            continue

        prompt, metadata = _build_for_row(row)
        heights.append(metadata.camera_height_m)

        t0 = time.perf_counter()
        image = _generate_one(model, prompt, row["seed"])
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)

        actual_size = f"{image.size[0]}x{image.size[1]}"
        sizes_seen[actual_size] = sizes_seen.get(actual_size, 0) + 1

        image.save(png_path)

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
            "image_size_requested": IMAGE_SIZE,
            "image_size_actual": actual_size,
            "seconds": round(elapsed, 2),
            "prompt_metadata": asdict(metadata),
        }

        json_path.write_text(
            json.dumps(sidecar, indent=2, default=str),
            encoding="utf-8",
        )

        print(
            f"[{row['row']:>2}/{len(rows)}] {stem}.png  "
            f"{elapsed:.1f}s  size={actual_size}  height={metadata.camera_height_m}m  "
            f"{row['class_names']}"
        )

    total = sum(durations)
    print(f"\ngenerated: {len(durations)}  skipped (already existed): {skipped}  -> {out_dir}")
    if durations:
        height_counts = {height: heights.count(height) for height in sorted(set(heights))}
        print(
            f"{total/60:.1f} min ({total/len(durations):.1f}s/image)"
        )
        print(f"survey heights used: {height_counts}")
        print(f"image sizes produced: {sizes_seen}")
        if len(sizes_seen) > 1:
            print(
                "  WARNING: more than one size was produced - image_size is "
                "not pinning the resolution as expected, investigate before "
                "using this output for training."
            )


if __name__ == "__main__":
    main()
