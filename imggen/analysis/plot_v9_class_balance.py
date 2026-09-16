"""Simple matplotlib bar chart of flux2dev v9's (Dataset A) training-set class balance."""
from __future__ import annotations

import matplotlib.pyplot as plt

classes = ["starfish", "sea_urchin", "scallop"]
counts = [1142, 1330, 1674]

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
bars = ax.bar(classes, counts, color=["#1f77b4", "#2ca02c", "#9467bd"])

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
             str(count), ha="center", va="bottom", fontsize=11)

ax.set_ylabel("instances")
ax.set_title("Flux2Dev v9 (Dataset A) — Training Set Class Balance")
ax.set_ylim(0, max(counts) * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out_path = "assets/flux2dev_v9_class_balance_matplotlib.png"
fig.savefig(out_path)
print(f"saved -> {out_path}")
