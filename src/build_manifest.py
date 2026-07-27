"""Build the Stage 1 pilot manifest (and the Stage 0.5 smoke manifest).

The 20-row design covers all 15 non-empty class combinations exactly once, then
reinforces single-class exemplars. That makes image-level class balance exact by
construction: every class appears in exactly 10 of 20 rows.

Instance-level balance is only balanced in expectation - the generator may not
comply with requested counts, and the Stage 2 detector has its own per-class
recall. Both are measured downstream, not enforced here. `requested_count` is
recorded per row so measured-vs-requested can be compared at Checkpoint 2.5.

Usage:
    python src/build_manifest.py                 # writes manifests/pilot.json
    python src/build_manifest.py --smoke         # writes manifests/smoke.json
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

from prompts import CLASSES, DENSITY_RANGE

BASE_SEED = 42
ALL_CLASSES = sorted(CLASSES)


def allocate_counts(class_ids: list[int], density: str, rng: random.Random) -> dict[int, int]:
    """Split a per-image instance budget across the requested classes.

    Density bands are per-image totals. A floor of one instance per requested
    class applies: row 20 asks for all four classes at 'sparse' (2-3), which is
    infeasible, so the total is raised to 4. The caller can detect this by
    comparing sum(counts) against DENSITY_RANGE[density].
    """
    lo, hi = DENSITY_RANGE[density]
    total = max(rng.randint(lo, hi), len(class_ids))

    counts = {cid: 1 for cid in class_ids}
    for _ in range(total - len(class_ids)):
        counts[rng.choice(class_ids)] += 1
    return counts


def build_rows() -> list[tuple[list[int], str, str]]:
    """(class_ids, density, framing) for each of the 20 pilot rows."""
    rows: list[tuple[list[int], str, str]] = []

    # 1-4: each class alone, sparse, close-up
    rows += [([c], "sparse", "close-up") for c in ALL_CLASSES]
    # 5-10: all 6 pairs, moderate, mid
    rows += [(list(p), "moderate", "mid") for p in combinations(ALL_CLASSES, 2)]
    # 11-14: all 4 triples, moderate, close-up
    rows += [(list(t), "moderate", "close-up") for t in combinations(ALL_CLASSES, 3)]
    # 15: all four, moderate, mid
    rows.append((list(ALL_CLASSES), "moderate", "mid"))
    # 16-19: each class alone again, moderate, mid
    rows += [([c], "moderate", "mid") for c in ALL_CLASSES]
    # 20: all four again, sparse, close-up
    rows.append((list(ALL_CLASSES), "sparse", "close-up"))

    assert len(rows) == 20, f"expected 20 rows, got {len(rows)}"
    return rows


def build_smoke_rows() -> list[tuple[list[int], str, str]]:
    """Stage 0.5: one single-class sparse close-up per class."""
    return [([c], "sparse", "close-up") for c in ALL_CLASSES]


def build_manifest(smoke: bool = False) -> dict:
    rng = random.Random(BASE_SEED)
    rows = build_smoke_rows() if smoke else build_rows()
    stage = "0.5-smoke" if smoke else "1-pilot"

    entries = []
    for i, (class_ids, density, framing) in enumerate(rows, start=1):
        counts = allocate_counts(class_ids, density, rng)
        lo, hi = DENSITY_RANGE[density]
        entries.append(
            {
                "row": i,
                "image_id": f"{'smoke' if smoke else 'pilot'}_{i:03d}",
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
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest = build_manifest(smoke=args.smoke)
    out = args.out or Path("manifests") / ("smoke.json" if args.smoke else "pilot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summarize(manifest)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
