"""Object box-size comparison -- synthetic v9 training labels vs. real DUO test labels.

Chapter 6's content-component argument (RQ4) claims the generator produces
objects that are too large relative to the frame, based on visual inspection
of the rendered montage; the number that would turn that inference into a
measurement was not available because the v9 label files lived only on the
RunPod workspace at the time (see reports/analysis/v9_yolo_sim2real_diagnosis.json,
finding 3). Both label sets are now on disk locally, so this script computes
it directly: box area (w*h, already frame-normalised in YOLO format) per
class and overall, for the delivered v9 synthetic corpus and the 778-image
DUO test split, and reports the median and mean of each.

    python -m imggen.analysis.box_size_compare \
        --synthetic-labels outputs/flux2dev/v9/labels/sam3 \
        --real-labels dataset/real_eval/labels \
        --out reports/analysis/box_size_compare.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from imggen.prompts.base import class_names

CLASS_NAMES = class_names()


def read_areas(labels_dir: Path) -> dict[int, list[float]]:
    """class_id -> [box area fraction of frame, ...] over every .txt in labels_dir."""
    areas: dict[int, list[float]] = {cid: [] for cid in CLASS_NAMES}
    for lp in labels_dir.glob("*.txt"):
        for line in lp.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cid = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            areas.setdefault(cid, []).append(w * h)
    return areas


def summarise(areas: list[float]) -> dict:
    if not areas:
        return {"n": 0, "median_pct": None, "mean_pct": None, "p10_pct": None, "p90_pct": None}
    s = sorted(areas)
    return {
        "n": len(areas),
        "median_pct": round(statistics.median(s) * 100, 3),
        "mean_pct": round(statistics.fmean(s) * 100, 3),
        "p10_pct": round(s[max(0, int(0.10 * len(s)) - 1)] * 100, 3),
        "p90_pct": round(s[min(len(s) - 1, int(0.90 * len(s)))] * 100, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-labels", required=True, type=Path)
    ap.add_argument("--real-labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    syn = read_areas(args.synthetic_labels)
    real = read_areas(args.real_labels)

    result = {"synthetic": {}, "real": {}}
    syn_all: list[float] = []
    real_all: list[float] = []
    for cid, name in CLASS_NAMES.items():
        result["synthetic"][name] = summarise(syn.get(cid, []))
        result["real"][name] = summarise(real.get(cid, []))
        syn_all += syn.get(cid, [])
        real_all += real.get(cid, [])
    result["synthetic"]["overall"] = summarise(syn_all)
    result["real"]["overall"] = summarise(real_all)
    result["ratio_median_overall"] = (
        round(result["synthetic"]["overall"]["median_pct"] / result["real"]["overall"]["median_pct"], 2)
        if result["real"]["overall"]["median_pct"] else None
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"box-size comparison -> {args.out}\n")
    print(f"  {'class':<12} {'synthetic median%':>18} {'real median%':>14} {'ratio':>7}")
    for name in [*CLASS_NAMES.values(), "overall"]:
        sm = result["synthetic"][name]["median_pct"]
        rm = result["real"][name]["median_pct"]
        ratio = f"{sm / rm:.1f}x" if sm and rm else "n/a"
        print(f"  {name:<12} {sm if sm is not None else '--':>18} {rm if rm is not None else '--':>14} {ratio:>7}")
    print(f"\n  overall: synthetic boxes are ~{result['ratio_median_overall']}x the frame-area of real DUO boxes (median).")


if __name__ == "__main__":
    main()
