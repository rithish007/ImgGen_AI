"""Build the Stage 1 pilot manifest (and the Stage 0.5 smoke manifest).

3 classes (starfish, sea urchin, scallop - sea cucumber removed, see
prompts.py's module docstring) = 7 non-empty combinations. The first 11 rows
cover all 7 exactly once, then reinforce single-class exemplars - every class
appears in exactly 5 of those 11 rows. (The original 4-class design used 20
rows to cover its 15 combinations the same way; 11 is the proportionate
equivalent for 3.)

Rows 12-20 are a second reinforcement round added to bring the pilot to 20
images total: the same 7-combination coverage again, this time at "dense"
density and "wide" framing (both unused by rows 1-11), plus two extra
all-three-classes rows at "sparse" and "moderate" density (also paired with
"wide", which none of rows 1-11 use) so the triple-class case gets one
exemplar per density band across the whole 20-row set. Rows 1-11 are
untouched - same rng draw order, same seeds, same counts - so existing
outputs/labels for pilot_001..011 remain valid without regeneration.

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

from prompts import CLASSES

# Local to this file, not imported from prompts.py: this is a per-IMAGE total
# instance budget for the deliberate combinatorial design below, a different
# concept from prompts.py's COUNT_RANGES (which is per-class, used only by
# prompts.py's own independent random generate_dataset_prompts() path).
DENSITY_RANGE = {"sparse": (2, 3), "moderate": (4, 6), "dense": (5, 8)}

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
    """(class_ids, density, framing) for each of the 20 pilot rows.

    Rows 1-11: 3 classes -> 7 non-empty subsets (3 singles + 3 pairs + 1
    triple), covered once, then singles and the all-three case are reinforced
    once more - the same design shape as the original 4-class/20-row/
    15-combination plan, scaled down to match 3 classes' smaller combination
    space.

    Rows 12-20: second reinforcement round, added later to bring the pilot to
    20 images total. Repeats the same 7-combination coverage at "dense"
    density / "wide" framing (neither used by rows 1-11), then adds two more
    all-three-classes rows at "sparse" and "moderate" density so the triple
    case has one wide-framing exemplar per density band.
    """
    rows: list[tuple[list[int], str, str]] = []

    # 1-3: each class alone, sparse, close-up
    rows += [([c], "sparse", "close-up") for c in ALL_CLASSES]
    # 4-6: all 3 pairs, moderate, mid
    rows += [(list(p), "moderate", "mid") for p in combinations(ALL_CLASSES, 2)]
    # 7: all three together, moderate, mid
    rows.append((list(ALL_CLASSES), "moderate", "mid"))
    # 8-10: each class alone again, moderate, mid
    rows += [([c], "moderate", "mid") for c in ALL_CLASSES]
    # 11: all three together again, sparse, close-up
    rows.append((list(ALL_CLASSES), "sparse", "close-up"))

    # 12-14: each class alone again, dense, wide
    rows += [([c], "dense", "wide") for c in ALL_CLASSES]
    # 15-17: all 3 pairs again, dense, wide
    rows += [(list(p), "dense", "wide") for p in combinations(ALL_CLASSES, 2)]
    # 18: all three together, dense, wide
    rows.append((list(ALL_CLASSES), "dense", "wide"))
    # 19: all three together, sparse, wide
    rows.append((list(ALL_CLASSES), "sparse", "wide"))
    # 20: all three together, moderate, wide
    rows.append((list(ALL_CLASSES), "moderate", "wide"))

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
