"""
Stage 1 generation - HunyuanImage-3.0 v8
==========================================

Identical to v7 except it fixes a real gap found by measuring v7's actual
output: `model.generate_image()` accepts an `image_size` parameter (confirmed
via run_startup_introspection's live signature dump, and already noted as
'auto' by default all the way back in generate_hunyuan.py's docstring), but
no version of this script has ever passed it. Left at 'auto', the model
silently picks its own aspect-ratio bucket per image based on internal
heuristics - measuring the v7 50-image run's actual PNGs found FIVE distinct
sizes: 1024x1024 (2), 1152x896 (25), 1216x832 (6), 1280x768 (16), 1344x704 (1).
flux2dev v8, by contrast, is 1024x1024 on all 50 images (it explicitly passes
height/width). That asymmetry means ~96% of Hunyuan's images pick up letterbox
padding when resized for YOLO training while flux2dev's never do - a
source-correlated artifact worth removing before the next full run.

Fix: pass image_size=IMAGE_SIZE ("1024x1024") explicitly, matching flux2dev's
format. NOT independently confirmed against the model beyond the fact that
"1024x1024" is one of the bucket strings the model already produced under
'auto' (so it's very likely a valid literal, not a guess out of nowhere) -
per this project's own established rule (see generate_hunyuan.py), don't
trust an unverified API parameter at full scale. Run with --limit 2 first and
check the printed size= field before committing to a 50+ image run.

Everything else (prompt engine, camera-height forcing, seed handling,
sidecar fields) is unchanged from v7. Duplicated rather than editing
generate_hunyuan_v7.py in place - v7 already produced real output via the
full 50-image run.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path

from prompts_hunyuan_v7 import build_prompt

MODEL_REPO = "tencent/HunyuanImage-3.0"
PROMPT_VARIANT = "hunyuan_v7_habitat_first_survey_placement_fix"
IMAGE_SIZE = "1024x1024"


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
    """Generate one image using the confirmed Hunyuan API, pinned to a fixed
    square resolution instead of leaving image_size='auto' to pick a
    different aspect-ratio bucket per image (see module docstring)."""
    return model.generate_image(
        prompt=prompt, seed=seed, image_size=IMAGE_SIZE, stream=True
    )


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
        local_files_only=True,
    )
    model.load_tokenizer(MODEL_REPO, local_files_only=True)
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
