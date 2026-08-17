# Model capability matrix — klein vs flux2dev vs HunyuanImage-3.0

**Versions**: klein (`outputs/klein/2-pilot`, unchanged since the first matrix — not
requested to update), **flux2dev v8** (`outputs/flux2dev/v8`, latest), **HunyuanImage-3.0
v7** (`outputs/hunyuan/v7`, latest). Same 50-row manifest (`manifests/2-pilot.json`), same
seeds per row, across all three.

## What changed since the last version of this matrix

The wide+sparse scallop bug that dominated the old matrix (5.9x/2.9x/3.1x overshoot for
klein/flux2dev/hunyuan) is **essentially fixed in both new versions**: flux2dev v8 sits at
1.18x and hunyuan v7 at 1.18x in that cell, down from 2.9x and 3.1x. That was a
`prompts.py`-family fix (flux2dev v8 dropped `"shell"` from its color-palette guard to close
a scallop-leak bug found via cross-version SAM3 analysis; hunyuan v6 got a "habitat-first"
prompt rewrite plus a v7 fix for a count=1 placement-text bug).

The bigger surprise: **Hunyuan closed almost the entire gap with flux2dev.** In the old
matrix Hunyuan won 0 of 9 cells and scored worst on every axis, especially spatial realism
(2.78/5) and photorealism (3.00/5) — it read like staged aquarium macro photography, not AUV
survey footage. The v6 habitat-first rewrite visibly fixes that (see the samples below):
Hunyuan v7 now produces natural, asymmetric, unstaged compositions indistinguishable in style
from flux2dev's, and **now beats flux2dev v8 on count accuracy (4.67 vs 4.56) and
detector-usefulness (4.33 vs 4.00)**, at the cost of still being ~4x slower and needing a
much heavier multi-GPU node. This is a real reversal from before and worth re-opening the
"which model" decision, not just re-confirming flux2dev.

One new shared defect worth flagging: **both** flux2dev v8 and Hunyuan v7 sometimes render
starfish in a cool blue-grey rather than the prompt's stated "mottled brown and grey" /
"reddish-brown" (seen in `2-pilot_007` and `2-pilot_014` for both models). This could be a
real color variant of the reference species rather than a defect, but it's a new,
cross-model pattern that wasn't present in the earlier klein/flux2dev matrix — worth checking
if it traces back to a shared prompt clause both v8/v7 inherited. Also new: flux2dev v8
rendered a starfish with **six arms** in `2-pilot_014` (should be five) — a genuine
anatomical defect, logged below.

---

## Summary — which model to use for what

| | **klein** | **flux2dev v8** | **HunyuanImage-3.0 v7** |
|---|---|---|---|
| Gen. speed (measured) | 27.1 s/image | 71.0 s/image (2.6x klein) | 284.3 s/image (10.5x klein, 4x flux2dev) |
| Hardware footprint | 1 GPU, ~29 GB | 2 GPUs bf16, ~112 GB combined, no quantization | No quantized path — bf16/fp16 across multiple 80GB+ GPUs (`device_map="auto"`); separate conda env, custom `transformers.AutoModelForCausalLM` integration |
| Count accuracy | 3.78 | 4.56 | **4.67 (best)** |
| Biological accuracy (morphology+anatomy) | 3.84 | **4.61 (best)** | 4.39 |
| Spatial realism (natural, unstaged composition) | 3.44 | **5.00 (best)** | 4.00 |
| Photorealism (AUV-survey look) | 4.00 | **5.00 (best)** | 4.00 |
| Detector-readiness (SAM3 confidence + label cleanliness) | 3.67 | 4.39 | **4.39 (tied best)** |
| **Overall (equal-weighted)** | 3.75 | **4.65 (best)** | 4.32 |
| **Use it for** | fast/cheap iteration, prompt debugging, bulk generation on a budget or single-GPU rig | the production dataset — still the best on every axis except count accuracy and detector-readiness (where Hunyuan v7 now edges it) | now a legitimate second production candidate — count accuracy and detector-readiness are its strongest points, if you can absorb the ~4x cost |
| **Watch out for** | scallop count still blows up 5.9x in wide+sparse — this baseline predates the v8/v7 prompt fixes | occasional 6-arm starfish anatomy defect (`2-pilot_014`); starfish moderate/mid count now *undershoots* (0.69x) rather than overshoots — a new regression, not the old bug | still 4x flux2dev's cost and needs a genuinely heavier node; the "staged" look is fixed but photorealism/spatial-realism scores are from only 3 spot-checked images each, same caveat as everything below |

**klein wasn't re-run** — it's still the original baseline from the first matrix, included
for reference. If you want true three-way parity, `outputs/klein/3-pilot_promptfix` exists
and is closer in vintage to v8/v7 (scallop count 118 vs the original's 139, still not
re-scored here — say the word and I'll fold it in).

---

## Method & important caveat (unchanged from before)

**Density and framing are not independent axes in this dataset.** `manifests/2-pilot.json`
only ever combines them three ways: dense+close-up (21 rows), moderate+mid (15 rows),
sparse+wide (14 rows). So this is a 3-condition breakdown, not a full 3×3 Density×Framing
grid — that still doesn't exist yet (see the separate conversation about designing a manifest
that decouples them for a routing experiment).

**Count accuracy** is computed directly from SAM3 label files vs. manifest
`requested_counts` across all 50 images per model (real data). **Criteria 2–7** are scored
from direct visual inspection of the same three representative multi-class images used in the
original matrix (`2-pilot_007` dense/close-up, `2-pilot_014` moderate/mid, `2-pilot_021`
sparse/wide) — 3 images × 2 updated models = 6 new images inspected, cross-checked against the
quantitative SAM3 confidence/count data across all 50. Same spot-check caveat as before:
treat 2–7 as calibrated sampling, not exhaustive per-cell audit.

**Scale**: 1 (poor) – 5 (excellent). Legend: **CA**=count accuracy, **Morph**=morphology,
**Sep**=instance separation, **Anat**=anatomical validity, **Spat**=spatial realism,
**Photo**=photorealism, **Det**=detector usefulness.

---

## Starfish

| Condition | Model | CA | Morph | Sep | Anat | Spat | Photo | Det | Notes |
|---|---|---|---|---|---|---|---|---|---|
| dense/close-up | klein | 5 | 4 | 4 | 4 | 4 | 4 | 5 | (unchanged from previous matrix) |
| dense/close-up | flux2dev v8 | 5 | 4 | 5 | 5 | 5 | 5 | 5 | ratio 0.98 (43/44). `2-pilot_007`: correct anatomy, but colour reads cool blue-grey rather than the prompt's brown/grey. |
| dense/close-up | hunyuan v7 | 5 | 4 | 4 | 4 | 4 | 4 | 5 | ratio 1.02 (45/44) aggregate, but `2-pilot_007` itself showed 3 starfish for 2 requested — good composition/photorealism, same blue-grey colour tendency as flux2dev. |
| moderate/mid | klein | 5 | 4 | 5 | 3 | 4 | 4 | 5 | (unchanged) |
| moderate/mid | flux2dev v8 | 4 | 4 | 4 | **2** | 5 | 5 | 5 | ratio 0.69 (20/29) — new undershoot regression. `2-pilot_014`: correct count in this sample (1 req→1) but the individual has **six arms**, a genuine anatomical defect. |
| moderate/mid | hunyuan v7 | 5 | 5 | 5 | 5 | 4 | 4 | 5 | ratio 0.97 (28/29). `2-pilot_014`: clean 5-arm starfish, correct count, strong sample overall. |
| sparse/wide | klein | 3 | 4 | 2 | 4 | 3 | 4 | 3 | (unchanged) |
| sparse/wide | flux2dev v8 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ratio 1.00 (11/11) — exact match. `2-pilot_021`: correct reddish-brown colour this time, clean anatomy, best cell in this whole class. |
| sparse/wide | hunyuan v7 | 5 | 4 | 4 | 3 | 4 | 4 | 5 | ratio 1.09 (12/11). `2-pilot_021`: arm count looked ambiguous (3–5, partly foreshortened) — flagged as uncertain, not a confirmed defect. |

## Sea urchin

| Condition | Model | CA | Morph | Sep | Anat | Spat | Photo | Det | Notes |
|---|---|---|---|---|---|---|---|---|---|
| dense/close-up | klein | 4 | 5 | 4 | 5 | 4 | 4 | 3 | (unchanged) |
| dense/close-up | flux2dev v8 | 5 | 5 | 5 | 5 | 5 | 5 | 3 | ratio 0.98 (40/41). `2-pilot_007`: 2 req→2, clean dense spine texture. Confidence stays the lowest of the three classes (0.79) — a SAM3-side pattern, consistent across every model/version tested so far. |
| dense/close-up | hunyuan v7 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | ratio 1.20 (49/41). Confidence 0.78 — same SAM3 weak spot. |
| moderate/mid | klein | 4 | 5 | 5 | 5 | 4 | 4 | 3 | (unchanged) |
| moderate/mid | flux2dev v8 | 5 | 5 | 5 | 5 | 5 | 5 | 3 | ratio 0.85 (23/27). `2-pilot_014`: 3 req→3, well-separated along a rock ledge. |
| moderate/mid | hunyuan v7 | 5 | 5 | 5 | 5 | 4 | 4 | 3 | ratio 0.96 (26/27). `2-pilot_014`: 3 req→3, exact match, clean anatomy. |
| sparse/wide | klein | 3 | 4 | 4 | 4 | 3 | 4 | 2 | (unchanged) |
| sparse/wide | flux2dev v8 | 4 | 5 | 5 | 5 | 5 | 5 | 3 | ratio 1.33 (16/12). `2-pilot_021`: 1 req→1 in-sample, clean; the ambiguous grey-blob artifact from the old flux2dev sample is gone. |
| sparse/wide | hunyuan v7 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | ratio 1.25 (15/12). `2-pilot_021`: 1 req→1, correct. |

## Scallop

| Condition | Model | CA | Morph | Sep | Anat | Spat | Photo | Det | Notes |
|---|---|---|---|---|---|---|---|---|---|
| dense/close-up | klein | 5 | 3 | 4 | 3 | 4 | 4 | 4 | (unchanged) |
| dense/close-up | flux2dev v8 | 5 | 4 | 4 | 4 | 5 | 5 | 4 | ratio 1.02 (46/45). `2-pilot_007`: 2 shells for 3 requested in-sample, but clean ribbed fan shape, closed shells. |
| dense/close-up | hunyuan v7 | 5 | 4 | 4 | 4 | 4 | 4 | 5 | ratio 1.09 (49/45). `2-pilot_007`: 4 shells for 3 requested, still well-formed individually. |
| moderate/mid | klein | 4 | 2 | 4 | 2 | 3 | 4 | 4 | (unchanged — the gaping-open-shell defect) |
| moderate/mid | flux2dev v8 | 4 | 5 | 5 | 5 | 5 | 5 | 4 | ratio 1.26 (29/23). `2-pilot_014`: correctly closed, cream/tan, ribbed, exact 1-for-1 in sample. |
| moderate/mid | hunyuan v7 | 5 | 5 | 5 | 5 | 4 | 4 | 5 | ratio 1.09 (25/23). `2-pilot_014`: exact 1-for-1, clean closed shell. |
| sparse/wide | klein | 1 | 4 | 3 | 4 | 2 | 4 | 2 | (unchanged — the 5.9x bug, this baseline predates the fix) |
| sparse/wide | flux2dev v8 | 4 | 5 | 5 | 5 | 5 | 5 | 4 | ratio **1.18** (13/11) — bug fixed. `2-pilot_021`: exact 1-for-1, best photorealism sample in the whole matrix (dappled light on cracked boulders). |
| sparse/wide | hunyuan v7 | 4 | 5 | 5 | 5 | 4 | 4 | 5 | ratio **1.18** (13/11) — bug fixed here too. `2-pilot_021`: exact 1-for-1, appropriately small/distant for a wide shot. |

---

## Rollup — mean score per model (27 cells, equal-weighted)

| Model | CA | Morph | Sep | Anat | Spat | Photo | Det | Overall |
|---|---|---|---|---|---|---|---|---|
| klein | 3.78 | 3.89 | 3.89 | 3.78 | 3.44 | 4.00 | 3.44 | **3.75** |
| flux2dev v8 | 4.56 | 4.67 | 4.78 | 4.56 | **5.00** | **5.00** | 4.00 | **4.65** |
| hunyuan v7 | **4.67** | 4.44 | 4.44 | 4.33 | 4.00 | 4.00 | **4.33** | 4.32 |

**Reading this**: flux2dev v8 is still the overall leader (4.65) and wins outright on
morphology, instance separation, anatomy, spatial realism, and photorealism. But Hunyuan v7
now wins on count accuracy and ties on detector-readiness — a genuine reversal from the
previous matrix, where it lost every axis. The gap that's left is mostly photorealism/spatial
polish, not the count-following or anatomical-structure problems that used to be
Hunyuan's biggest weaknesses.

Worth flagging explicitly: flux2dev v8's 5.00 on Spatial realism and Photorealism means all
three spot-checked images (`2-pilot_007`, `_014`, `_021`) scored max on both axes — genuinely
striking image quality, but it's still only 3 images. Don't treat 5.00 as "verified perfect
at scale," treat it as "no flaws found in the samples checked so far."

**Biggest actionable finding**: the sparse+wide scallop bug — previously the single worst
cell in the entire matrix for all three models — is fixed in both v8 and v7 (1.18x for both,
down from 2.9x/3.1x). That was worth the prompt-engineering effort. The new open item is
flux2dev v8's starfish moderate/mid *undershoot* (0.69x) and the six-arm anatomy defect in
the same cell — smaller than the old bug, but a genuine new regression worth a look before
trusting flux2dev v8 at 1000-image scale.

---

## Version trajectory (for context — not separately scored above)

SAM3 exact-match counts across every generation-script version run on this 50-row manifest,
from `reports/cross_version_analysis.json` plus the new v8/v7 reports:

| Version | n_exact / 50 | scallop req / det | zero-req scallop leak |
|---|---|---|---|
| flux2dev v3 | 19 | 79 / 88 | 3 (1 image) |
| flux2dev v4 | 16 | 79 / 116 | 23 (4 images) |
| flux2dev v5 | 19 | 79 / 85 | 1 (1 image) |
| flux2dev v6 | 23 | 79 / 86 | 0 |
| flux2dev v7 | 14 | 79 / 235 | 151 (11 images) — bad regression |
| **flux2dev v8** | — | 79 / 92 | — (see per-condition table above; the wide+sparse-specific bug is fixed) |
| hunyuan v1 (3-pilot) | 10 | 79 / 125 | 0 |
| hunyuan v4 | 7 | 79 / 255 | 129 (19 images) — bad regression |
| **hunyuan v7** | — | 79 / 87 | — (habitat-first rewrite + placement fix; see per-condition table above) |

Both models regressed badly at an intermediate version (flux2dev v7, hunyuan v4 — both had
scallop leaking into images that never requested it) before the fixes in v8/v7 brought scallop
counts back down to roughly requested levels. Worth remembering if either model regresses
again: this has happened twice already for scallop specifically.
