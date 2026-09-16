"""Chapter 5 result figures -- turns the run/stat JSON artefacts into the publication figures the results chapter references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "thesis" / "Writing" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

REGIMES = [
    ("Base",               "v9_flux2dev",                  "v9_flux2dev_v2"),
    ("Placeholder",        "v9_flux2dev_placeholder_dr",   "v9_flux2dev_placeholder_dr_v2"),
    ("Calibrated",         "v9_flux2dev_duo_dr",           "v9_flux2dev_duo_dr_v3"),
    ("Calibrated\n+scatter", "v9_flux2dev_duo_scatter_dr", "v9_flux2dev_duo_scatter_dr_v2"),
]
CLASSES = ["starfish", "sea_urchin", "scallop"]

STAT_FILES = {
    "Base":               "flux2dev_v9_water_stats.json",
    "Placeholder":        "v9_dr_placeholder_water_stats.json",
    "Calibrated":         "v9_dr_duo_calibrated_water_stats.json",
    "Calibrated+scatter": "v9_dr_duo_calibrated_scatter_water_stats.json",
}
REAL_STAT_FILE = "duo_test_water_stats.json"
STATS = [
    ("ratio_rg", "R/G ratio"),
    ("ratio_bg", "B/G ratio"),
    ("dark_channel_mean", "dark-channel mean"),
    ("luminance_std", "luminance std"),
]

C_DEFAULT, C_TUNED = "#8aa9c9", "#2f4b6e"
C_REGIME = ["#b0b7bd", "#8aa9c9", "#3d7a63", "#c0743a"]
C_OWNVAL, C_REAL = "#c0743a", "#2f4b6e"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig_transfer() -> None:
    models = _load(ROOT / "runs/eval_duo/comparison.json")["models"]
    labels = [r[0] for r in REGIMES]
    default = [models[r[1]]["overall"]["map50"] for r in REGIMES]
    tuned = [models[r[2]]["overall"]["map50"] for r in REGIMES]

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    b1 = ax.bar([i - w / 2 for i in x], default, w, label="default config", color=C_DEFAULT)
    b2 = ax.bar([i + w / 2 for i in x], tuned, w, label="tuned config", color=C_TUNED)

    ax.axhline(default[0], ls="--", lw=0.9, color=C_DEFAULT, zorder=0)
    ax.axhline(tuned[0], ls="--", lw=0.9, color=C_TUNED, zorder=0)

    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("real DUO test mAP$_{50}$")
    ax.set_ylim(0, max(tuned) * 1.25)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig_transfer_bars")


def fig_per_class() -> None:
    models = _load(ROOT / "runs/eval_duo/comparison.json")["models"]
    x = range(len(CLASSES))
    n = len(REGIMES)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for j, (label, _default, tuned_key) in enumerate(REGIMES):
        pc = models[tuned_key]["per_class"]
        vals = [pc[c]["ap50"] for c in CLASSES]
        off = (j - (n - 1) / 2) * w
        ax.bar([i + off for i in x], vals, w,
               label=label.replace("\n", " "), color=C_REGIME[j])
    ax.set_xticks(list(x)); ax.set_xticklabels([c.replace("_", " ") for c in CLASSES])
    ax.set_ylabel("AP$_{50}$ (tuned config)")
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    _save(fig, "fig_per_class_ap")


def fig_domain_stats() -> None:
    real = _load(ROOT / "reports/water_stats" / REAL_STAT_FILE)["aggregate"]
    regimes = list(STAT_FILES)
    stat_by_regime = {
        r: _load(ROOT / "reports/water_stats" / STAT_FILES[r])["aggregate"] for r in regimes
    }
    fig, axes = plt.subplots(1, len(STATS), figsize=(11, 3.0))
    for ax, (key, title) in zip(axes, STATS):
        vals = [stat_by_regime[r][key]["mean"] for r in regimes]
        ax.bar(range(len(regimes)), vals, color=C_REGIME)
        target = real[key]["mean"]
        ax.axhline(target, ls="--", lw=1.2, color="#b03030")
        ax.annotate(f"real\n{target:.3f}", (len(regimes) - 0.5, target),
                    color="#b03030", fontsize=8, va="bottom", ha="right")
        ax.set_title(title)
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels(["base", "plac.", "calib.", "scat."], rotation=0, fontsize=8)
    axes[0].set_ylabel("per-image mean")
    fig.suptitle("Water statistics per regime vs the real DUO target (dashed)", y=1.02)
    _save(fig, "fig_domain_stats")


def fig_synthetic_vs_real() -> None:
    rows = []
    for label, default_key, tuned_key in REGIMES:
        for cfg, key in (("default", default_key), ("tuned", tuned_key)):
            s = _load(ROOT / f"runs/train/{key}_summary.json")
            rows.append((f"{label.replace(chr(10), ' ')} ({cfg})",
                         s["own_val"]["map50"], s["duo_test"]["map50"]))
    rows.reverse()

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ys = range(len(rows))
    for y, (_lab, ov, rl) in zip(ys, rows):
        ax.plot([rl, ov], [y, y], color="#c9c9c9", lw=2, zorder=1)
    ax.scatter([r[2] for r in rows], list(ys), color=C_REAL, zorder=2, label="real DUO test")
    ax.scatter([r[1] for r in rows], list(ys), color=C_OWNVAL, zorder=2, label="own synthetic val")
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("mAP$_{50}$")
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    _save(fig, "fig_synthetic_vs_real")


def fig_count() -> None:
    data = _load(ROOT / "reports/analysis/cross_version_analysis.json")
    versions = [k for k in ("flux2dev_v3", "flux2dev_v4", "flux2dev_v5",
                            "flux2dev_v6", "flux2dev_v7") if k in data]
    exact_rate = [data[v]["n_exact"] / data[v]["n_images"] for v in versions]
    scallop_ratio = [data[v]["detected"].get("2", 0) / data[v]["requested"].get("2", 1)
                     for v in versions]
    short = [v.replace("flux2dev_", "") for v in versions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(short, exact_rate, "o-", color=C_TUNED)
    ax1.set_ylabel("exact-count image rate")
    ax1.set_title("Prompt-count compliance")
    ax1.set_ylim(0, 1)

    ax2.bar(short, scallop_ratio, color=C_REGIME[3])
    ax2.axhline(1.0, ls="--", lw=1.0, color="#555")
    ax2.set_ylabel("scallop detected / requested")
    ax2.set_title("Scallop over-generation")
    _save(fig, "fig_count_compliance")


FIGURES = {
    "transfer": fig_transfer,
    "per_class": fig_per_class,
    "domain_stats": fig_domain_stats,
    "synthetic_vs_real": fig_synthetic_vs_real,
    "count": fig_count,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", default="all", choices=["all", *FIGURES])
    args = ap.parse_args()

    todo = FIGURES if args.figure == "all" else {args.figure: FIGURES[args.figure]}
    for name, fn in todo.items():
        print(f"[{name}]")
        fn()
    print(f"\nfigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
