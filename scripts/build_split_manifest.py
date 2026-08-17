"""One-off: build a single 2000-row manifest (one continuous RNG sequence, so
no accidental content/seed duplication between halves - see the chat record
for why two independent build_manifest() calls would have produced identical
files), then split it into two standalone 1000-row manifests, one per model,
so each model's generate script can run from its own file with zero
coordination and zero row overlap. Rows keep their original global seed
(42001-44000) even after the split, so seeds stay unique across BOTH files
combined, not just within each one.

    python scripts/build_split_manifest.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_manifest import build_balanced_rows, allocate_counts, BASE_SEED, ALL_CLASSES
from prompts import CLASSES

STAGE = "benthic-survey-1000"
N_TOTAL = 2000
N_PER_MODEL = 1000

rng = random.Random(BASE_SEED)
dr_rng = random.Random(BASE_SEED + 1)
row_shapes = build_balanced_rows(N_TOTAL, dr_rng)

all_rows = []
for i, (class_ids, density, framing) in enumerate(row_shapes, start=1):
    counts = allocate_counts(class_ids, density, rng)
    from build_manifest import DENSITY_RANGE
    lo, hi = DENSITY_RANGE[density]
    all_rows.append({
        "global_row": i,
        "class_ids": class_ids,
        "class_names": [CLASSES[c]["short"] for c in class_ids],
        "requested_counts": {str(c): counts[c] for c in class_ids},
        "total_instances": sum(counts.values()),
        "density": density,
        "density_floor_applied": sum(counts.values()) > hi,
        "framing": framing,
        "seed": BASE_SEED * 1000 + i,
    })

assert len(all_rows) == N_TOTAL
assert len({r["seed"] for r in all_rows}) == N_TOTAL, "seed collision across the 2000-row sequence"

def make_manifest(rows_slice: list[dict], model_tag: str) -> dict:
    entries = []
    for local_i, r in enumerate(rows_slice, start=1):
        entries.append({
            "row": local_i,
            "image_id": f"{STAGE}-{model_tag}_{local_i:04d}",
            "class_ids": r["class_ids"],
            "class_names": r["class_names"],
            "requested_counts": r["requested_counts"],
            "total_instances": r["total_instances"],
            "density": r["density"],
            "density_floor_applied": r["density_floor_applied"],
            "framing": r["framing"],
            "seed": r["seed"],
        })
    return {
        "stage": f"{STAGE}-{model_tag}",
        "base_seed": BASE_SEED,
        "class_map": {
            str(c): {"duo_label": CLASSES[c]["duo_label"], "short": CLASSES[c]["short"]}
            for c in ALL_CLASSES
        },
        "rows": entries,
    }

flux_rows = all_rows[:N_PER_MODEL]
hunyuan_rows = all_rows[N_PER_MODEL:]

flux_manifest = make_manifest(flux_rows, "flux2dev")
hunyuan_manifest = make_manifest(hunyuan_rows, "hunyuan")

out_dir = Path(__file__).resolve().parent.parent / "manifests"
flux_path = out_dir / f"{STAGE}-flux2dev.json"
hunyuan_path = out_dir / f"{STAGE}-hunyuan.json"
flux_path.write_text(json.dumps(flux_manifest, indent=2), encoding="utf-8")
hunyuan_path.write_text(json.dumps(hunyuan_manifest, indent=2), encoding="utf-8")

# Sanity: zero content overlap between the two files (species+counts+density)
def content_set(manifest):
    return {
        (tuple(sorted(r["class_ids"])), tuple(sorted(r["requested_counts"].items())), r["density"])
        for r in manifest["rows"]
    }

flux_content = content_set(flux_manifest)
hunyuan_content = content_set(hunyuan_manifest)
overlap = flux_content & hunyuan_content
all_seeds = {r["seed"] for r in flux_manifest["rows"]} | {r["seed"] for r in hunyuan_manifest["rows"]}

print(f"wrote {flux_path} ({len(flux_manifest['rows'])} rows)")
print(f"wrote {hunyuan_path} ({len(hunyuan_manifest['rows'])} rows)")
print(f"seed uniqueness across BOTH files: {len(all_seeds)} / {N_TOTAL}")
print(f"exact (species+counts+density) content shared between the two files: {len(overlap)} (rows can still coincidentally share this triple by chance - not a bug, just means those specific rows happen to overlap; each row's seed and full render will still differ)")
