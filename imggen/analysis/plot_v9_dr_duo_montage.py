"""Simple matplotlib montage: the same flux2dev v9 base image (#0007 - chosen
because it has all three classes visible: starfish, sea_urchin, scallop) shown
across its two DR variants, next to a real DUO test photo for reference.

    python -m imggen.analysis.plot_v9_dr_duo_montage
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from PIL import Image

PANELS = [
    ("Flux2Dev v9 (base)", "outputs/flux2dev/v9/benthic-survey-1000-flux2dev_0007_flux2dev_bf16.png"),
    ("DR (placeholder)", "outputs/flux2dev/v9/dr_runs/v1/dr/benthic-survey-1000-flux2dev_0007_flux2dev_bf16_dr.png"),
    ("DR Calibrated (duo_calibrated)", "outputs/flux2dev/v9/dr_runs/v1/dr_duo_calibrated/benthic-survey-1000-flux2dev_0007_flux2dev_bf16_duo_dr.png"),
    ("Real DUO", "dataset/real_eval/images/1002_jpg.rf.32c6af3d666b8a7b0f6ebf76fbecac7a.jpg"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)

for ax, (title, path) in zip(axes, PANELS):
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Flux2Dev v9 image #0007 across DR profiles, vs. a real DUO photo", fontsize=14)
fig.tight_layout()
out_path = "assets/v9_dr_duo_montage_matplotlib.png"
fig.savefig(out_path)
print(f"saved -> {out_path}")
