"""Stage 2 - auto-annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imggen.prompts.base import CLASSES, detector_prompts

ENGINES = {
    "sam3": {
        "repo": "facebook/sam3",
        "threshold": 0.5,
        "mask_threshold": 0.5,
        "presence_threshold": 0.5,
    },
}


def xyxy_to_yolo_line(class_id: int, box: tuple[float, float, float, float], img_w: int, img_h: int) -> str:
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"loading {cfg['repo']} (Sam3Model) on {device}...")
    model = Sam3Model.from_pretrained(cfg["repo"]).to(device)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="sam3", choices=sorted(ENGINES))
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=Path("reports/class_counts/class_counts.json"))
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

    per_class = run_sam3(image_paths, out_dir, cfg)

    update_class_counts(args.report, args.engine, per_class)

    print(f"\ndone: labels -> {out_dir}, report -> {args.report}")
    for class_id, stats in per_class.items():
        print(f"  {CLASSES[class_id]['short']:<12} instances={stats['instances']:>3}  "
              f"images={stats['images_with_detection']:>2}/{len(image_paths)}")


if __name__ == "__main__":
    main()
