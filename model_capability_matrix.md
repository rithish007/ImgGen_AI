# Model capability matrix — klein vs flux2dev vs HunyuanImage-3.0

## Summary — which model to use for what

Scores are 1 (poor)–5 (excellent), averaged across the 27-cell breakdown below (real SAM3
count data for Count acc.; visual spot-checks for the rest). Speed/hardware are measured, not
estimated (`seconds` field averaged over 50 sidecar JSONs per model).

| | **klein** | **flux2dev** | **HunyuanImage-3.0** |
|---|---|---|---|
| Gen. speed (measured) | 27.1 s/image | 66.5 s/image (2.5x klein) | 269.9 s/image (10x klein, 4x flux2dev) |
| Hardware footprint | 1 GPU, ~29 GB | 1 GPU fp8 (~32+24 GB, offloaded) or 2 GPUs bf16 (~106-112 GB) | No quantized path — needs bf16/fp16 across multiple 80GB+ GPUs (device_map="auto"); separate conda env, custom `transformers.AutoModelForCausalLM` integration |
| Count accuracy | 3.78 | **4.11 (best)** | 3.22 (worst) |
| Biological accuracy (morphology+anatomy) | 3.84 | 4.28 | 4.06 |
| Spatial realism (natural, unstaged composition) | 3.44 | 3.33 | **2.78 (worst)** — reads staged/symmetric |
| Photorealism (AUV-survey look, not documentary macro) | 4.00 | **4.67 (best)** | 3.00 (worst) — bright, sharp, saturated, aquarium-like |
| Detector-readiness (SAM3 confidence + label cleanliness) | 3.67 | **3.89 (best)** | 3.61 |
| **Overall (equal-weighted)** | 3.75 | **4.06 (best)** | 3.48 (worst) |
| **Use it for** | fast/cheap iteration, prompt debugging, bulk generation on a budget or single-GPU rig | the production dataset — best on every axis this pipeline is actually optimized for (count discipline, photorealism, detector-readiness) | not recommended as a production source yet at this integration/compute cost for what it delivers; keep as a dissertation comparison/ablation point only |
| **Watch out for** | scallop count blows up 5.9x in wide+sparse scenes (worst of the three) | same wide+sparse scallop bug, 2.9x (mildest of the three, still bad) | same bug, 3.1x; plus needs its own env/integration work and a multi-80GB-GPU node just to run at all |

**One caveat that applies to all three rows**: the wide+sparse scallop overshoot is a
`prompts.py` prompt-following bug (see the bottom of this doc), not something any of these
three models fixes on its own — budget time to fix it, or drop wide+sparse scallop rows,
before trusting any of these numbers at 1000-image scale.

---

## Method & important caveat before reading the detailed table

**Density and framing are not independent axes in this dataset.** `manifests/2-pilot.json`
(the one shared run all three models were compared on) only ever combines them three ways:

| density | framing | rows |
|---|---|---|
| dense | close-up | 21 |
| moderate | mid | 15 |
| sparse | wide | 14 |

So a true 3×3 Density×Framing grid doesn't exist yet — there are 3 realized *scene-scale
conditions*, not 9. The table below uses those 3 conditions. If you actually want the full
3×3 grid scored, `prompts.py`/`build_manifest.py` would need a manifest that decouples the
two (e.g. `dense`+`wide`, `sparse`+`close-up`) before there's anything real to score.

**Data sources** (all three models ran on the identical 50-row manifest, same seeds per row):
- klein: `outputs/2-pilot/klein/` + `reports/2pilot_class_counts_klein.json`
- flux2dev: `outputs/4-pilot/flux2dev/` (fp8, post scallop-prompt-fix, latest working run) + `reports/4pilot_class_counts_flux2dev.json`
- HunyuanImage-3.0: `outputs/3-pilot/hunyuan/` (bf16, unquantized — no quantized path exists yet) + `reports/3pilot_class_counts_hunyuan.json`

None of the three models actually receives a text-level negative prompt (`supports_negative:
False` for all three in `generate_flux.py`/`generate_hunyuan.py` — confirmed via
`inspect.signature`) — all three rely solely on `POSITIVE_ONLY_GUARDS` +
`BIVALVE_GUARD`. So differences below reflect model behavior, not one model getting a
negative-prompt advantage over another.

**Count accuracy** (col 1) is computed directly from SAM3 label files vs. manifest
`requested_counts` across all 50 images per model (real data, not sampled) — see ratios in
the Notes column. **Criteria 2–7** are scored from direct visual inspection of one
representative multi-class image per (class, condition) cell per model — 9 images per model,
27 total — cross-checked against the broader-sample findings already on record in
[pilot2.md](pilot2.md) (~15 starfish anatomy-audited, boulder/algae, occlusion checks across
all 50). Treat 2–7 as a calibrated spot-check, not an exhaustive per-cell audit.

**Scale**: 1 (poor) – 5 (excellent) for all 7 criteria.

Legend: **CA**=count accuracy, **Morph**=morphology, **Sep**=instance separation,
**Anat**=anatomical validity, **Spat**=spatial realism, **Photo**=photorealism,
**Det**=detector usefulness (SAM3 confidence + label cleanliness for training).

---

## Starfish

| Condition | Model | CA | Morph | Sep | Anat | Spat | Photo | Det | Notes |
|---|---|---|---|---|---|---|---|---|---|
| dense/close-up | klein | 5 | 4 | 4 | 4 | 4 | 4 | 5 | ratio 0.95 (42/44). `2-pilot_007`: 2 req→2 clean, well-separated 5-arm individuals. |
| dense/close-up | flux2dev | 5 | 5 | 4 | 4 | 4 | 5 | 5 | ratio 0.95 (42/44). Same scene, cleanest photoreal rendering of the three. |
| dense/close-up | hunyuan | 4 | 4 | 4 | 4 | 3 | 3 | 5 | ratio 1.18 (52/44). Correct anatomy but brighter/flatter lighting, less attenuation with depth. |
| moderate/mid | klein | 5 | 4 | 5 | 3 | 4 | 4 | 5 | ratio 1.14 (33/29). `2-pilot_014`: 1 req→1, clean single individual. Anatomy audit in pilot2.md puts limb-malformation risk at ~1-in-10 for this class/model pair generally. |
| moderate/mid | flux2dev | 4 | 5 | 5 | 3 | 3 | 5 | 5 | ratio 1.17 (34/29). Same ~1-in-10 malformed-limb risk per pilot2.md; sample itself was clean. Rock-pile prop in this frame reads slightly staged/geometric. |
| moderate/mid | hunyuan | 5 | 3 | 5 | 4 | 3 | 3 | 5 | ratio 1.07 (31/29). `2-pilot_014`: correct 5-arm topology but arm texture rendered as small hard spines/nubs rather than the requested "rough natural texture" — a morphology miss, not a structural one. |
| sparse/wide | klein | 3 | 4 | 2 | 4 | 3 | 4 | 3 | ratio 1.64 (18/11). `2-pilot_021`: only 1 starfish visible on inspection, but SAM3 detected 3 boxes on it — likely arm-silhouette over-segmentation, worth checking the raw viz overlay before trusting this class's wide-shot labels. |
| sparse/wide | flux2dev | 4 | 4 | 4 | 4 | 3 | 4 | 4 | ratio 1.18 (13/11). Clean 1:1 in this sample; small/distant instances start losing detail. |
| sparse/wide | hunyuan | 3 | 4 | 3 | 4 | 2 | 3 | 3 | ratio 1.55 (17/11). `2-pilot_021` had 2 starfish for 1 requested (one far background) — same overshoot pattern as the other classes, just milder. |

## Sea urchin

| Condition | Model | CA | Morph | Sep | Anat | Spat | Photo | Det | Notes |
|---|---|---|---|---|---|---|---|---|---|
| dense/close-up | klein | 4 | 5 | 4 | 5 | 4 | 4 | 3 | ratio 1.24 (51/41). `2-pilot_007`: 2 req→2, dense spine texture, correctly clustered in crevice. Confidence is the lowest of the three classes for every model (~0.77-0.78) — likely SAM3 struggling with spine texture/partial occlusion, not a generation defect. |
| dense/close-up | flux2dev | 4 | 5 | 4 | 5 | 4 | 5 | 3 | ratio 1.27 (52/41). Same scene; best photoreal integration into the crevice shadow. |
| dense/close-up | hunyuan | 4 | 4 | 4 | 5 | 3 | 3 | 3 | ratio 1.22 (50/41). Anatomy clean; urchins read slightly oversized/flattened relative to the crevice they sit in. |
| moderate/mid | klein | 4 | 5 | 5 | 5 | 4 | 4 | 3 | ratio 1.19 (32/27). `2-pilot_014`: 3 req→3, well-separated along a rock ledge. |
| moderate/mid | flux2dev | 5 | 5 | 5 | 5 | 3 | 5 | 3 | ratio 1.07 (29/27). Same scene, best count match of the three here. |
| moderate/mid | hunyuan | 3 | 4 | 3 | 5 | 3 | 3 | 3 | ratio 1.37 (37/27). `2-pilot_014` rendered 4-5 urchins for 3 requested — overshoot visible on inspection, not just an aggregate artifact. |
| sparse/wide | klein | 3 | 4 | 4 | 4 | 3 | 4 | 2 | ratio 1.58 (19/12). `2-pilot_021`: 1 req→1 in this sample, clean; aggregate ratio driven by other rows. |
| sparse/wide | flux2dev | 3 | 3 | 3 | 3 | 3 | 4 | 2 | ratio 1.67 (20/12). `2-pilot_021`: a second, ambiguous smooth grey dome-shaped object sits beside the real urchin — reads as neither a clean urchin nor a rock, an anatomical/morphology miss worth a closer look. |
| sparse/wide | hunyuan | 2 | 4 | 4 | 4 | 2 | 3 | 2 | ratio 2.58 (31/12) — worst urchin overshoot of the three models, though the one sample checked (`2-pilot_021`, 1 req→1) didn't show it; concentrated elsewhere in the 50-image set. |

## Scallop

| Condition | Model | CA | Morph | Sep | Anat | Spat | Photo | Det | Notes |
|---|---|---|---|---|---|---|---|---|---|
| dense/close-up | klein | 5 | 3 | 4 | 3 | 4 | 4 | 4 | ratio 0.96 (43/45). `2-pilot_007`: shells read a bit flatter/rounder than a true fan-ribbed scallop — closer to a cockle silhouette. |
| dense/close-up | flux2dev | 5 | 4 | 4 | 4 | 4 | 5 | 4 | ratio 1.00 (45/45) — exact match. Clear radial ribs, tightly closed, correct count. Best cell in the whole matrix. |
| dense/close-up | hunyuan | 4 | 4 | 4 | 4 | 3 | 3 | 4 | ratio 1.24 (56/45). Shells well-formed but composition reads posed (near-symmetric spacing across the frame). |
| moderate/mid | klein | 4 | 2 | 4 | 2 | 3 | 4 | 4 | ratio 1.35 (31/23). `2-pilot_014`: shell rendered bright white and visibly gaping open — the "shucked/plated" look `BIVALVE_GUARD` was added specifically to prevent, recurring here despite the guard being present (scallop was requested, so it fired). |
| moderate/mid | flux2dev | 5 | 5 | 5 | 5 | 3 | 5 | 4 | ratio 1.09 (25/23). Same scene: correctly closed, cream/tan, ribbed, resting naturally — the guard held here. |
| moderate/mid | hunyuan | 3 | 4 | 3 | 4 | 3 | 3 | 4 | ratio 1.52 (35/23). `2-pilot_014` had 2 shells for 1 requested. |
| sparse/wide | klein | 1 | 4 | 3 | 4 | 2 | 4 | 2 | ratio **5.91** (65/11) — worst cell in the matrix. `2-pilot_021`: 5 shells visible for 1 requested; matches the wide+sparse scallop-overshoot bug already documented in pilot2.md (arrangement text has no natural boundary, "wide" framing exposes more area, model scatters per unit area not per total count). |
| sparse/wide | flux2dev | 2 | 4 | 3 | 4 | 3 | 4 | 3 | ratio 2.91 (32/11) — same bug, clearly less severe than klein (best-controlled of the three here, consistent with pilot2.md's "flux2dev has meaningfully better count discipline" finding). |
| sparse/wide | hunyuan | 1 | 4 | 4 | 4 | 3 | 3 | 2 | ratio 3.09 (34/11) — same bug, roughly midway between klein and flux2dev in severity. |

---

## Rollup — mean score per model (all 27 cells, equal-weighted)

| Model | CA | Morph | Sep | Anat | Spat | Photo | Det | Overall |
|---|---|---|---|---|---|---|---|---|
| klein | 3.78 | 3.89 | 3.89 | 3.78 | 3.44 | 4.00 | 3.44 | **3.75** |
| flux2dev | 4.11 | 4.44 | 4.11 | 4.11 | 3.33 | 4.67 | 3.67 | **4.06** |
| hunyuan | 3.22 | 3.89 | 3.78 | 4.22 | 2.78 | 3.00 | 3.44 | **3.48** |

**Reading this**: flux2dev wins on the axis this whole comparison was built to measure
(count discipline, especially scallop) and on photorealism/morphology; the gap is narrowest
on structural anatomy (all three are comparable, hunyuan even edges ahead there) and widest
on photorealism and spatial-composition naturalness, where Hunyuan's brighter, more
symmetric "documentary macro" look consistently reads less like raw AUV survey footage than
either FLUX.2 variant. This is consistent with — and adds condition-level and per-class
granularity to — the class-count finding already written up in
[pilot2.md §10](pilot2.md).

**Biggest actionable finding**: the sparse/wide scallop cell is catastrophic for all three
models (ratio 2.9×–5.9×) and is a `prompts.py` prompt-following issue, not a model-quality
issue — worth fixing (or explicitly excluding wide+sparse scallop rows from a production run)
before scaling to 1000 images, independent of which generator gets chosen.
