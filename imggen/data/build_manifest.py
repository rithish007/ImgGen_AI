"""Build the Stage 1 pilot manifest (and the Stage 0.5 smoke manifest)."""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

from imggen.prompts.base import CLASSES

DENSITY_RANGE = {"sparse": (2, 3), "moderate": (4, 6), "dense": (5, 7)}

BASE_SEED = 42
ALL_CLASSES = sorted(CLASSES)


def allocate_counts(class_ids: list[int], density: str, rng: random.Random) -> dict[int, int]:
    lo, hi = DENSITY_RANGE[density]
    total = max(rng.randint(lo, hi), len(class_ids))

    counts = {cid: 1 for cid in class_ids}
    for _ in range(total - len(class_ids)):
        counts[rng.choice(class_ids)] += 1
    return counts


def build_rows() -> list[tuple[list[int], str, str]]:
    rows: list[tuple[list[int], str, str]] = []

    rows += [([c], "sparse", "close-up") for c in ALL_CLASSES]
    rows += [(list(p), "moderate", "mid") for p in combinations(ALL_CLASSES, 2)]
    rows.append((list(ALL_CLASSES), "moderate", "mid"))
    rows += [([c], "moderate", "mid") for c in ALL_CLASSES]
    rows.append((list(ALL_CLASSES), "sparse", "close-up"))

    rows += [([c], "dense", "wide") for c in ALL_CLASSES]
    rows += [(list(p), "dense", "wide") for p in combinations(ALL_CLASSES, 2)]
    rows.append((list(ALL_CLASSES), "dense", "wide"))
    rows.append((list(ALL_CLASSES), "sparse", "wide"))
    rows.append((list(ALL_CLASSES), "moderate", "wide"))

    assert len(rows) == 20, f"expected 20 rows, got {len(rows)}"
    return rows


def build_smoke_rows() -> list[tuple[list[int], str, str]]:
    return [([c], "sparse", "close-up") for c in ALL_CLASSES]


def build_balanced_rows(n_images: int, dr_rng: random.Random) -> list[tuple[list[int], str, str]]:
    combos = (
        [[c] for c in ALL_CLASSES]
        + [list(p) for p in combinations(ALL_CLASSES, 2)]
        + [list(ALL_CLASSES)]
    )
    densities = sorted(DENSITY_RANGE)
    framings = ["close-up", "mid", "wide"]

    rows: list[tuple[list[int], str, str]] = []
    combo_i = 0
    while len(rows) < n_images:
        combo = combos[combo_i % len(combos)]
        density = dr_rng.choice(densities)
        framing = dr_rng.choice(framings)
        rows.append((combo, density, framing))
        combo_i += 1
    return rows


def build_manifest(smoke: bool = False, balanced_n: int | None = None, stage_name: str = "1-pilot") -> dict:
    rng = random.Random(BASE_SEED)
    if balanced_n is not None:
        dr_rng = random.Random(BASE_SEED + 1)
        rows = build_balanced_rows(balanced_n, dr_rng)
        stage = stage_name
        id_prefix = stage_name
    else:
        rows = build_smoke_rows() if smoke else build_rows()
        stage = "0.5-smoke" if smoke else "1-pilot"
        id_prefix = "smoke" if smoke else "pilot"

    entries = []
    for i, (class_ids, density, framing) in enumerate(rows, start=1):
        counts = allocate_counts(class_ids, density, rng)
        lo, hi = DENSITY_RANGE[density]
        entries.append(
            {
                "row": i,
                "image_id": f"{id_prefix}_{i:03d}",
                "class_ids": class_ids,
                "class_names": [CLASSES[c]["short"] for c in class_ids],
                "requested_counts": {str(c): counts[c] for c in class_ids},
                "total_instances": sum(counts.values()),
                "density": density,
                "density_floor_applied": sum(counts.values()) > hi,
                "framing": framing,
                "seed": BASE_SEED * 1000 + i,
            }
        )

    return {
        "stage": stage,
        "base_seed": BASE_SEED,
        "class_map": {
            str(c): {"duo_label": CLASSES[c]["duo_label"], "short": CLASSES[c]["short"]}
            for c in ALL_CLASSES
        },
        "rows": entries,
    }


def summarize(manifest: dict) -> None:
    rows = manifest["rows"]
    print(f"stage={manifest['stage']}  rows={len(rows)}")

    img_counts = {c: 0 for c in ALL_CLASSES}
    inst_counts = {c: 0 for c in ALL_CLASSES}
    for r in rows:
        for cid_s, n in r["requested_counts"].items():
            img_counts[int(cid_s)] += 1
            inst_counts[int(cid_s)] += n

    print(f"{'class':<14} {'images':>7} {'instances':>10}")
    for c in ALL_CLASSES:
        print(f"{CLASSES[c]['short']:<14} {img_counts[c]:>7} {inst_counts[c]:>10}")

    floored = [r["row"] for r in rows if r["density_floor_applied"]]
    if floored:
        print(f"\ndensity floor raised the total on row(s): {floored}")
        print("(requested class count exceeded the density band - expected on all-four sparse rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="build the 4-row Stage 0.5 manifest")
    ap.add_argument("--balanced", type=int, default=None, metavar="N",
                     help="build an N-row class-balanced manifest (cycles the 7 combinations) instead of the fixed pilot design")
    ap.add_argument("--stage-name", default="2-pilot", help="stage/id-prefix for --balanced manifests (default: 2-pilot)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest = build_manifest(smoke=args.smoke, balanced_n=args.balanced, stage_name=args.stage_name)
    if args.balanced is not None:
        default_name = f"{args.stage_name}.json"
    else:
        default_name = "smoke.json" if args.smoke else "pilot.json"
    out = args.out or Path("manifests") / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summarize(manifest)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
