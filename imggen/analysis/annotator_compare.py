"""Automated annotator comparison -- SAM3 vs Grounding DINO on the SAME images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imggen.prompts.base import class_names

CLASS_NAMES = class_names()
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def label_stats(labels_dir: Path, stems: set[str]) -> dict:
    per_class = {cid: {"instances": 0, "images": 0} for cid in CLASS_NAMES}
    for stem in stems:
        lp = labels_dir / f"{stem}.txt"
        seen = set()
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                cid = int(line.split()[0])
                if cid in per_class:
                    per_class[cid]["instances"] += 1
                    seen.add(cid)
        for cid in seen:
            per_class[cid]["images"] += 1
    return per_class


def parse_engine(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--engine expects name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", action="append", required=True,
                    help="name=labels_dir (repeatable; give at least two)")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="optional: restrict to stems that have a real image here")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    engines = [parse_engine(s) for s in args.engine]
    if len(engines) < 2:
        raise SystemExit("give at least two --engine name=dir specs to compare")

    stem_sets = [{p.stem for p in path.glob("*.txt")} for _, path in engines]
    stems = set.intersection(*stem_sets)
    if args.images_dir is not None:
        img_stems = {p.stem for p in args.images_dir.iterdir()
                     if p.suffix.lower() in IMAGE_SUFFIXES}
        stems &= img_stems
    if not stems:
        raise SystemExit("no common image stems across the given engines")

    report = {"n_images": len(stems), "engines": {}}
    for name, path in engines:
        stats = label_stats(path, stems)
        report["engines"][name] = {
            CLASS_NAMES[cid]: stats[cid] for cid in CLASS_NAMES
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    names = [n for n, _ in engines]
    print(f"compared over {len(stems)} shared images -> {args.out}\n")
    header = f"  {'class':<12}" + "".join(f"{n+' inst':>12}{n+' imgs':>12}" for n in names)
    print(header)
    for cid, cname in CLASS_NAMES.items():
        row = f"  {cname:<12}"
        for name in names:
            e = report["engines"][name][cname]
            row += f"{e['instances']:>12}{e['images']:>12}"
        print(row)


if __name__ == "__main__":
    main()
