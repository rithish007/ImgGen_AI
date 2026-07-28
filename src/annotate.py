"""Stage 2 - auto-annotation. SAM3 and Grounding DINO, one engine per run.

Both engines run one class per forward pass, never a period-joined multi-class
prompt - see the plan doc's Stage 2 section for why (SAM3 only accepts a single
noun phrase per call anyway; Grounding DINO's multi-class prompts collide on
shared tokens like "sea").

Annotates the CLEAN Stage 1 image. Stage 3's domain-randomization transform is
pixel-only, so the same label file is valid for the DR'd copy.

    python src/annotate.py --engine sam3  --images-dir outputs/1-pilot/klein --dry-run
    python src/annotate.py --engine sam3  --images-dir outputs/1-pilot/klein
    python src/annotate.py --engine gdino --images-dir outputs/1-pilot/klein

Outputs:
    outputs/1-pilot/labels/<engine>/<image_id>.txt   (YOLO format)
    reports/class_counts.json                        (merged across engines)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompts import CLASSES, detector_prompts

ENGINES = {
    "sam3": {
        "repo": "facebook/sam3",
        # Raw per-instance score threshold before the presence-head gate.
        "threshold": 0.5,
        "mask_threshold": 0.5,
        # final_score = instance_score * presence_score must clear this too.
        "presence_threshold": 0.5,
    },
    "gdino": {
        "repo": "IDEA-Research/grounding-dino-base",
        "threshold": 0.4,
        "text_threshold": 0.3,
    },
}


def xyxy_to_yolo_line(class_id: int, box: tuple[float, float, float, float], img_w: int, img_h: int) -> str:
    """Absolute-pixel xyxy -> YOLO 'class_id cx cy w h' (normalized 0-1, clipped)."""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(x1, img_w))
    x2 = max(0.0, min(x2, img_w))
    y1 = max(0.0, min(y1, img_h))
    y2 = max(0.0, min(y2, img_h))
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def update_class_counts(report_path: Path, engine: str, per_class: dict[int, dict[str, float]]) -> None:
    """Merge this engine's per-class stats into the shared class_counts.json.

    per_class: {class_id: {"instances": int, "images_with_detection": int,
                            "confidence_sum": float}}
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    engine_report = {}
    for class_id, stats in per_class.items():
        instances = stats["instances"]
        images = stats["images_with_detection"]
        engine_report[str(class_id)] = {
            "short_name": CLASSES[class_id]["short"],
            "instance_count": instances,
            "image_count": images,
            "mean_instances_per_image_present": (instances / images) if images else 0.0,
            "mean_confidence": (stats["confidence_sum"] / instances) if instances else 0.0,
        }

    report[engine] = engine_report
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_sam3(image_paths: list[Path], out_dir: Path, cfg: dict) -> dict[int, dict[str, float]]:
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    print(f"loading {cfg['repo']} (Sam3Model)...")
    model = Sam3Model.from_pretrained(cfg["repo"], device_map="auto")
    processor = Sam3Processor.from_pretrained(cfg["repo"])

    prompts = detector_prompts()
    per_class = {cid: {"instances": 0, "images_with_detection": 0, "confidence_sum": 0.0} for cid in prompts}

    for i, path in enumerate(image_paths, start=1):
        image = Image.open(path).convert("RGB")
        img_w, img_h = image.size

        img_inputs = processor(images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

        lines = []
        for class_id, prompt in prompts.items():
            text_inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(vision_embeds=vision_embeds, **text_inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=cfg["threshold"],
                mask_threshold=cfg["mask_threshold"],
                target_sizes=img_inputs.get("original_sizes").tolist(),
            )[0]

            # Presence head: pred_logits are per-query instance scores, but a
            # single concept can be entirely absent from the image. Most
            # manifest rows only contain a subset of the 3 classes, so this
            # gate matters a lot - without it, absent classes still produce
            # confident-looking false positives.
            presence = outputs.presence_logits.sigmoid().item()

            hit_this_class = False
            for box, score in zip(results["boxes"], results["scores"]):
                final_score = score.item() * presence
                if final_score < cfg["presence_threshold"]:
                    continue
                box = [float(v) for v in box.tolist()]
                lines.append(xyxy_to_yolo_line(class_id, box, img_w, img_h))
                per_class[class_id]["instances"] += 1
                per_class[class_id]["confidence_sum"] += final_score
                hit_this_class = True

            if hit_this_class:
                per_class[class_id]["images_with_detection"] += 1

        (out_dir / f"{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"[{i:>2}/{len(image_paths)}] {path.name}  {len(lines)} instances")

    return per_class


def run_gdino(image_paths: list[Path], out_dir: Path, cfg: dict) -> dict[int, dict[str, float]]:
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"loading {cfg['repo']} (GroundingDinoForObjectDetection)...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["repo"], device_map="auto")
    processor = AutoProcessor.from_pretrained(cfg["repo"])

    prompts = detector_prompts()
    per_class = {cid: {"instances": 0, "images_with_detection": 0, "confidence_sum": 0.0} for cid in prompts}

    for i, path in enumerate(image_paths, start=1):
        image = Image.open(path).convert("RGB")
        img_w, img_h = image.size

        lines = []
        for class_id, prompt in prompts.items():
            # One class per call, not a period-joined multi-class prompt -
            # GDINO's matched-span parsing can otherwise collide on shared
            # tokens between class names.
            inputs = processor(images=image, text=[[prompt]], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                threshold=cfg["threshold"],
                text_threshold=cfg["text_threshold"],
                target_sizes=[(img_h, img_w)],
            )[0]

            hit_this_class = False
            for box, score in zip(results["boxes"], results["scores"]):
                box = [float(v) for v in box.tolist()]
                score = float(score.item())
                lines.append(xyxy_to_yolo_line(class_id, box, img_w, img_h))
                per_class[class_id]["instances"] += 1
                per_class[class_id]["confidence_sum"] += score
                hit_this_class = True

            if hit_this_class:
                per_class[class_id]["images_with_detection"] += 1

        (out_dir / f"{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"[{i:>2}/{len(image_paths)}] {path.name}  {len(lines)} instances")

    return per_class


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=Path("reports/class_counts.json"))
    ap.add_argument("--dry-run", action="store_true", help="list images/prompts, load nothing")
    args = ap.parse_args()

    cfg = ENGINES[args.engine]
    image_paths = sorted(args.images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"no .png files found in {args.images_dir}")

    out_dir = args.out_dir or args.images_dir.parent / "labels" / args.engine
    prompts = detector_prompts()

    print(f"engine={args.engine}  images={len(image_paths)}  classes={prompts}")

    if args.dry_run:
        for path in image_paths:
            print(f"  would annotate: {path}")
        print(f"  output dir: {out_dir}")
        print(f"  report: {args.report}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.engine == "sam3":
        per_class = run_sam3(image_paths, out_dir, cfg)
    else:
        per_class = run_gdino(image_paths, out_dir, cfg)

    update_class_counts(args.report, args.engine, per_class)

    print(f"\ndone: labels -> {out_dir}, report -> {args.report}")
    for class_id, stats in per_class.items():
        print(f"  {CLASSES[class_id]['short']:<12} instances={stats['instances']:>3}  "
              f"images={stats['images_with_detection']:>2}/{len(image_paths)}")


if __name__ == "__main__":
    main()
