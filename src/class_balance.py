"""View per-class balance from reports/class_counts.json. No GPU, no model
weights, runs anywhere - matplotlib is the only non-stdlib dependency, and
only needed for --plot.

Reads whatever engines annotate.py has written into the report (currently
just "sam3" - see annotate.py's module docstring for why gdino was dropped;
a stale "gdino" key from before that decision is still shown if present,
since this script doesn't know which engines are "current").

    python src/class_balance.py
    python src/class_balance.py --report reports/class_counts.json --engine sam3
    python src/class_balance.py --plot
    python src/class_balance.py --plot --plot-out outputs/reports/class_balance.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BAR_WIDTH = 30

# dataviz skill's validated categorical palette, slots 1-2 (light mode) -
# re-run scripts/validate_palette.js if more than 2 engines ever appear here.
ENGINE_COLOURS = {
    "sam3": "#2a78d6",
    "gdino": "#eb6834",
}
FALLBACK_COLOURS = ["#1baf7a", "#eda100", "#e87ba4"]


def print_engine_balance(engine: str, per_class: dict[str, dict]) -> None:
    total_instances = sum(c["instance_count"] for c in per_class.values())
    max_instances = max((c["instance_count"] for c in per_class.values()), default=0)

    print(f"\n=== {engine} ===")
    if total_instances == 0:
        print("  no instances detected for any class")
        return

    header = f"{'class':<14} {'instances':>9} {'share':>7} {'images':>7} {'mean/img':>9} {'conf':>6}  bar"
    print(header)
    print("-" * len(header))

    for class_id in sorted(per_class, key=int):
        c = per_class[class_id]
        share = c["instance_count"] / total_instances
        bar_len = round(BAR_WIDTH * c["instance_count"] / max_instances) if max_instances else 0
        bar = "#" * bar_len
        print(
            f"{c['short_name']:<14} {c['instance_count']:>9} {share:>6.1%} "
            f"{c['image_count']:>7} {c['mean_instances_per_image_present']:>9.2f} "
            f"{c['mean_confidence']:>6.2f}  {bar}"
        )

    # Balance ratio: weakest class's instance count as a fraction of the
    # strongest. 1.0 = perfectly even; low values flag a class worth
    # reinforcing in the next manifest round.
    min_instances = min(c["instance_count"] for c in per_class.values())
    ratio = min_instances / max_instances if max_instances else 0.0
    weakest = min(per_class.values(), key=lambda c: c["instance_count"])
    print(f"\nbalance ratio (weakest/strongest instance count): {ratio:.2f}")
    if ratio < 0.5:
        print(f"  flag: '{weakest['short_name']}' is under half the strongest class - consider reinforcing it")


def plot_balance(report: dict, engines: list[str], out_path: Path) -> None:
    """Grouped bar chart: instance count per class, one bar per engine.

    Color encodes engine (the series/legend dimension) - class identity is
    already on the x-axis, so re-using visualize_annotations.py's per-class
    colours here would encode the same thing twice and nothing for engine.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_ids = sorted({cid for e in engines for cid in report[e]}, key=int)
    class_names = [report[engines[0]][cid]["short_name"] for cid in class_ids]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    n_groups = len(class_ids)
    n_series = len(engines)
    group_width = 0.7
    bar_width = group_width / n_series
    x = range(n_groups)

    for i, engine in enumerate(engines):
        colour = ENGINE_COLOURS.get(engine, FALLBACK_COLOURS[i % len(FALLBACK_COLOURS)])
        offsets = [xi - group_width / 2 + bar_width * (i + 0.5) for xi in x]
        counts = [report[engine].get(cid, {}).get("instance_count", 0) for cid in class_ids]
        bars = ax.bar(offsets, counts, width=bar_width * 0.9, color=colour, label=engine)
        ax.bar_label(bars, padding=2, fontsize=9, color="#0b0b0b")

    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names)
    ax.set_ylabel("instances")
    ax.set_title("Class balance - instance counts by detector")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if n_series > 1:
        ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nplot saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, default=Path("reports/class_counts.json"))
    ap.add_argument("--engine", default=None, help="show only this engine (default: all present)")
    ap.add_argument("--plot", action="store_true", help="also save a grouped bar chart")
    ap.add_argument("--plot-out", type=Path, default=Path("outputs/reports/class_balance.png"))
    args = ap.parse_args()

    if not args.report.exists():
        raise SystemExit(f"{args.report} not found - run src/annotate.py first")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    engines = [args.engine] if args.engine else sorted(report)

    for engine in engines:
        if engine not in report:
            raise SystemExit(f"engine {engine!r} not found in {args.report} (present: {sorted(report)})")
        print_engine_balance(engine, report[engine])

    if args.plot:
        plot_balance(report, engines, args.plot_out)


if __name__ == "__main__":
    main()
