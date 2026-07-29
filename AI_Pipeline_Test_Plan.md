# AI-Pipeline Smoke Test — Technical Build Plan (v8)

**Purpose:** validate an all-AI alternative to the UE5.8 pipeline (generation -> annotation -> DR -> YOLO26) on a small pilot batch before committing to full-scale dataset generation. This is a pipeline-mechanics check, not a scientific result — sample size is deliberately tiny at this stage.

**Compute:** RunPod Pod, **32GB+ VRAM**, On-Demand, **Secure Cloud** with a persistent volume. Nothing runs on the local machine — local GPUs are for editing only. Pods have moved between RTX 5090 (32GB) and RTX 6000 Ada (48GB) across sessions; the pipeline is GPU-model-agnostic as long as VRAM and sm-arch checks in `scripts/pod_preflight.py` pass. Sync via Git: edit in PyCharm -> `git push` -> SSH to pod -> `git pull` -> run -> pull results back. See `POD_RUNBOOK.md` for the exact per-session commands.

**Repo:** `https://github.com/rithish007/ImgGen_AI.git` (branch `main`)

---

## Changelog

**v7 -> v8** (n=20 pilot: Stage 2 finalized, Stage 3 built end-to-end)
1. **Pilot extended from 11 to 20 images.** Rows 12-20 repeat the same 7-combination coverage at "dense" density / "wide" framing (new since v7) - see `src/build_manifest.py`. Rows 1-11 verified byte-identical to the original 11-row manifest, so nothing already generated needed redoing.
2. **Grounding DINO dropped; SAM3 is the sole annotation engine.** Measured on all 20 images: starfish and sea urchin were close enough to call either way, but SAM3 found 70 scallop instances across 17/20 images vs. GDINO's 4 across 3/20 - that gap alone made GDINO unusable. `src/annotate.py` now runs SAM3 only; GDINO's pilot outputs are kept as historical record, not deleted. See Checkpoint 2.5.
3. **Two image-gen alternatives smoke-tested against klein's instance-count overshoot, neither adopted.** qwenimage (20B, Apache 2.0) undershot the same count and additionally rendered an out-of-distribution robot/rover in frame - dropped, weights deleted. flux2dev (32B) was never tested - its offload requirements don't fit this pod's real container RAM cap (~58GB via `/sys/fs/cgroup/memory.max`, not the much larger host total `free -h` reports) and that same offload path already crashed the pod once. Kept as a future candidate. Klein remains the sole generation model. See Stage 1.
4. **Stage 3 built end-to-end: range estimation (3a) and the physics transform (3b).** DA-V2 Large chosen over Apple Depth Pro after a head-to-head (`src/depth_compare.py`) - roughly tied on fine detail, but DA-V2's disparity representation gives more usable dynamic range in the near-field where the target classes live, and it's 2-10x faster. A training-free guided-filter refinement (`src/depth_utils.py`) is applied on top - not fine-tuning, since no ground-truth depth exists for 2D-diffusion-generated scenes to fine-tune against.
5. **A real bug caught in range estimation:** reciprocal-inverting DA-V2's already-reciprocal disparity output re-compressed the near field, the opposite of the intended fix. Caught by inspecting the actual output distribution (75th percentile landing 0.03m above `z_near`), fixed to a linear flip. See `src/range_estimate.py`.
6. **Jerlov's veiling light `B_c^∞(d)` resolved as a flagged simplification, not a sourced table.** No usable `K_d(λ)` table for the coastal types was found - the closest candidate (Williamson & Hollins 2023, full PDF read) turned out to study Jerlov type-classification drift with depth, not attenuation magnitude, and only covers types I-1C. Reuses the existing beam-attenuation ratios for spectral shape, anchored to one real citation for absolute green-channel magnitude. See Stage 3b.
7. **Stage 3b's parameters recalibrated twice against real output**, not just derived from theory: first pass was "way too extreme" (β_b sampled independent of each image's own `z_far`, so wide shots blacked out); second pass over-corrected to "too similar to the original" (one fixed visibility floor made every image mildly hazy). Fixed by coupling β_b's cap to each image's own `z_far` and sampling the visibility floor itself per image, so the DR set spans mild to notably dark/murky. See Stage 3b.
8. **Three camera-level effects added** (vignette, motion blur, signal-dependent sensor noise) - Jerlov/Akkaynak-Treibitz covers turbidity/haze/backscatter and depth-dependent lighting, nothing about the camera itself. All pure numpy/opencv, no model weights. See Stage 3b.
9. **Box-alignment verified, and explicitly distinguished from detectability.** The DR transform is pixel-only by construction, so label alignment is guaranteed, not really "tested." Detectability is separate and was quantified (`src/dr_detection_check.py`): 0% misclassification, but scallop - already this pipeline's weakest class - loses ~36% of instances to the haze. DR'd images always ship with their original clean-image labels, never a re-detection on the degraded copy. See Checkpoint 3.

**v6 -> v7**
1. **SD3.5 dropped; Klein is the sole generation model.** Smoke-test comparison on identical prompts: SD3.5 rendered sea urchin spines as short blunt bumps (a real anatomical defect, not a style difference) and shot every class as an isolated macro product photo, ignoring the wide-angle survey-camera framing the whole pipeline depends on. Klein respected that framing; SD3.5 didn't, consistently, across all four smoke-test classes.
2. **Sea cucumber (DUO class_id 2, "holothurian") removed entirely** after repeated generation failures (toy caterpillar/millipede anatomy) that did not resolve even with a real reference photo. **Scallop renumbered 3 -> 2** to keep class IDs contiguous for YOLO training (a gap at 2 either breaks training config or wastes a class slot). **This is now a DUO-derived 3-class subset (starfish, sea urchin, scallop), not an exact DUO match** — a class_id of 2 in this pipeline's output means scallop, not DUO's holothurian. Anything comparing results back to DUO must account for this explicitly.
3. **`src/prompts.py` rewritten** with a genuine diversity engine — per-class morphology/arrangement variant lists, plus scene/algae/substrate/lighting/composition/camera/imaging variation, all seeded deterministically per row. Real improvement over the single fixed string per class used through v6, where every instance of a class looked stylistically identical.
4. **Caught and fixed a reopened regression during that rewrite**: the new module's `WATER_CONDITIONS` baked turbidity/colour-cast language directly into the Stage 1 prompt (`"turbid coastal seawater"`, `"green-blue coastal seawater... reduced red and orange visibility"`, etc.) — the exact double-degradation bug fixed in v4/v5. Stage 3's Akkaynak-Treibitz transform needs clean scene radiance as input; reverted to a fixed clear/colour-neutral phrase (`SCENE_WATER_PHRASE`), removed the parameter and its randomized selection entirely so there is nothing left to accidentally re-enable.
5. Fixed three bugs the rewrite introduced that would have crashed or silently corrupted output: a determinism bug (one field used unseeded `random.choice` instead of the seeded `rng`), a fragile global string `.replace('a ', '')` that stripped mid-sentence articles as well as the intended leading one, and a `FRAMING` key rename (`"close"` vs. the manifest's existing `"close-up"`) that would have raised `KeyError` on the first real run.
6. `build_manifest.py`'s combinatorial design reworked for 3 classes: **7 non-empty combinations, 11-row pilot manifest** (was 15 combinations / 20 rows for 4 classes) — same design shape (cover every combination once, then reinforce singles and the all-classes case), scaled down.

**v5 -> v6**
1. **Jerlov coefficients sourced.** Confirmed there is no universal, camera-independent Jerlov coefficient table — Akkaynak et al. (CVPR 2017) show the RGB-domain ratios depend on camera spectral response, imaging range, and reflectance (their Eq. 9); both that paper and Berman et al. (BMVC 2017) present the per-type ratios only as scatter plots, never a printed table. Adopted the `'peak'`-branch values from `danaberman/underwater-hl`'s `get_water_types.m` (cited, code-shipped, camera-agnostic evaluation) for 1C/3C/5C, with one flagged, user-confirmed inference resolving an 8-vs-10 mismatch between that file's coefficient arrays and its named type list. See `src/jerlov.py`.
2. **Pod deployed and verified.** RTX 5090, sm_120, all five gated HF repos accessible, all pipeline classes import clean — see `scripts/pod_preflight.py`.
3. Fixed a preflight false-failure on the not-yet-created `HF_HOME` cache directory (`huggingface_hub` creates it lazily on first download).

**v4 -> v5**
1. **Depth model corrected.** v4 conflated vertical depth with camera-to-object range and capped range at 5m. These are separate variables in the Akkaynak-Treibitz model and control different effects. Stage 3 rewritten — see Stage 3b.
2. Jerlov set narrowed to **1C / 3C / 5C** at vertical depth d ∈ [0, 5] m.
3. **Stage 0.5 smoke test** added — 4 images/model before building the full manifest, to answer "are these generators good enough" for ~10 min of GPU instead of a full pilot.
4. **Per-class prompt expansion** — "scallop" and "sea cucumber" resolve to culinary/dried senses in web image data and must be forced to the habitat sense.
5. **Class-balance measurement** made an explicit Stage 2 deliverable.
6. Deploy runbook + pod git auth added (Stage 0).

**v3 -> v4**
1. FLUX.2-dev (32B) replaced with **FLUX.2 klein base (9B)** — 32B does not fit on 32GB in bf16 and would have needed 4-bit quantization, confounding the Stage 1 comparison. Both Stage 1 models now run bf16.
2. Community Cloud -> Secure Cloud (network volume requirement).
3. Stage 1 generates **clear, colour-neutral** scenes; Stage 3 applies all the water. v3 baked green water into pixels and then applied an attenuation model on top, double-degrading.
4. DUO turbidity calibration deferred; pilot uses published Jerlov coastal coefficients.

---

## Stage 0 — Prerequisites

**Git.** Repo and remote exist. A root `.gitignore` has been added — without it, `git add -A` commits the whole `ImgGenEnv` venv including torch binaries. Verify `git status` is clean of `ImgGenEnv/` before the first real commit.

**Packages** (missing from `ImgGenEnv`; all verified to resolve on Python 3.14.3):

```bash
pip install accelerate ultralytics opencv-python-headless scipy
```

Present and version-verified: `torch` 2.13.0+cu132, `torchvision` 0.28.0, `diffusers` 0.39.0, `transformers` 5.14.1, `huggingface_hub` 1.25.1, `safetensors`, `pillow`, `numpy`.

Freeze `requirements.txt` and commit it — the pod builds from this file, not from a copied venv.

**Class availability** — verified present in the local env:
`Flux2KleinPipeline`, `StableDiffusion3Pipeline` (diffusers 0.39.0); `Sam3Model`, `Sam3Processor`, `GroundingDinoForObjectDetection`, `DepthAnythingForDepthEstimation` (transformers 5.14.1).

**Hugging Face.** Confirm license acceptance for every gated repo listed in `scripts/pod_preflight.py`'s `MODEL_REPOS` (grows as models are added/removed - run the preflight script rather than trusting a hardcoded count here), in particular `black-forest-labs/FLUX.2-klein-base-9B` — a FLUX.2-dev acceptance does **not** carry over to klein. Export `HF_TOKEN` on the pod. Set `HF_HOME` to the network volume so weights survive pod termination:

```bash
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=hf_...
```

**Pod git auth.** RunPod's GitHub integration builds Serverless containers from a repo; it does **not** give a Pod git credentials. If `ImgGen_AI` is public, `git clone https://...` needs no auth for pulls and nothing else is required. If private, switch the remote to SSH and use agent forwarding so no token is stored on the pod:

```bash
git remote set-url origin git@github.com:rithish007/ImgGen_AI.git
```

then connect with `ssh -A root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519`.

### Deploy runbook — historical (first deployment only)

The steps below are what first stood the pod up. **For every session since, `POD_RUNBOOK.md` is the authoritative reference** (exact copy-paste commands, what does/doesn't persist across a stop) - this section is kept for how the pod and volume were originally provisioned, not as a per-session guide.

1. Scripts written locally, `requirements.txt` frozen, everything pushed to `main`
2. RunPod -> Settings -> SSH Public Keys -> paste the `id_ed25519.pub` key
3. Create the network volume **first** (Storage -> Network Volume, note the datacenter) - grown from an initial ~150GB to 200GB as more candidate models (flux2dev, qwenimage before it was dropped, Depth Pro) were added to the comparison work
4. **Then click DEPLOY** — Secure Cloud, in the volume's datacenter, PyTorch template on **CUDA 12.8 or newer**, volume mounted at `/workspace`. GPU has varied by availability across sessions (RTX 5090 32GB, RTX 6000 Ada 48GB) - the pipeline is GPU-model-agnostic as long as `scripts/pod_preflight.py`'s VRAM/sm-arch checks pass.
5. SSH in, `git clone`, `pip install -r requirements.txt` (**note:** on at least one session `pip install --target=...` pulled in a mismatched torch as a transitive dependency and broke torchaudio; a plain `pip install -r requirements.txt --break-system-packages` was the working fallback - see `POD_RUNBOOK.md`), export `HF_HOME`/`HF_TOKEN`
6. Run Stage 0.5 first — 4 images/model, ~10 min. If output is unusable, stop and re-prompt before burning hours on the full manifest.

---

### Scripts

`src/` now holds more scripts than fit a table worth hand-maintaining (prompt/manifest generation, per-model image generation, SAM3 annotation, depth estimation, the domain-randomization transform, several comparison/diagnostic tools) - each has a module docstring with its own usage examples; that's the source of truth, not a table here. `python <script>.py --dry-run` (where supported) or `--help` costs nothing and needs no GPU.

---

## Stage 0.5 — Generator smoke test (NEW, do this first)

Before building the full pilot manifest, generate **4 images per model** — one per class, single-class, sparse, close-up — using the Stage 1 prompt template. ~10 minutes of GPU total.

Gate: is the object recognizable, on a seabed, in clear neutral water, photographic rather than illustrated? If a model fails here it will fail 20 times over, and no amount of downstream work recovers it.

Watch specifically for the **culinary failure**: "scallop" in web image-caption data is overwhelmingly a seared scallop on a plate, and "sea cucumber" is often the dried market product. If the expanded prompts below do not fix it, that is a finding worth acting on immediately (reword, or accept the class will be weak).

---

## Stage 1 — Base image generation (Klein only)

SD3.5 was dropped after the smoke-test comparison (v7 changelog) — real anatomical defect on sea urchin, and it consistently ignored the wide-angle survey-camera framing this pipeline depends on. FLUX.2 klein base is now the sole generation model:

| Model | Repo | Params | Steps | Guidance | VRAM (bf16, 1024²) |
|---|---|---|---|---|---|
| FLUX.2 klein base | `black-forest-labs/FLUX.2-klein-base-9B` | 9B | 50 | 4.0 | ~29GB |

Use the **base** klein variant, not the step-distilled `FLUX.2-klein-9B`. Distilled runs in 4 steps but has materially weaker prompt adherence and less sample diversity — and compositional obedience is exactly what Checkpoint 1.5 measures.

Native 1024×1024; resize/crop to 640×640 happens at Stage 4, not here.

**Pilot batch:** 11 images (see the reworked 3-class manifest below), extended to 20 at the n=20 checkpoint (rows 12-20: same 7-combination coverage at "dense" density / "wide" framing, see `src/build_manifest.py`).

**Instance-count overshoot, and a model comparison that didn't resolve it.** `pilot_012` (klein, dense, wide, requested 7 starfish) rendered roughly 9-10 — a real, visible count-compliance gap, not just an annotation-recall gap. `src/generate.py` was extended with two candidates to smoke-test against the same row (same seed, same prompt):

| Model | Params | Result on pilot_012 (7 requested) | Notes |
|---|---|---|---|
| klein (current) | 9B | ~9-10 starfish | overshoot |
| qwenimage | 20B, Apache 2.0 | ~5 starfish | undershoot, **and rendered a visible robot/rover in frame** - an out-of-distribution object this pipeline doesn't want, not present in any klein output. **Dropped** after this result; weights deleted from the pod, `--model qwenimage` removed from `src/generate.py`. |
| flux2dev | 32B + 24B text encoder | not tested | needs both fp8-quantized components in CPU RAM simultaneously under `enable_model_cpu_offload()`, which doesn't fit this pod's actual container RAM cap (~58GB, not the 503GB the host reports - see `/sys/fs/cgroup/memory.max`). Already crashed the pod once via a similar offload path on qwenimage's first (pre-quantization) attempt. **Kept as a future candidate** (`--model flux2dev` still in `src/generate.py`) - retry on a pod with more system RAM.

**Conclusion: klein stays the pilot model, qwenimage dropped, flux2dev deferred.** qwenimage traded one count error direction for another and added a new artifact - worse than klein's problem, not better. The count-overshoot itself is left as a documented Stage 1 limitation, same treatment as the scallop count-overshoot problem noted below; fixing it properly likely needs layout-conditioned generation (bounding-box conditioning, GLIGEN-style, or compositing), not just a bigger base model - that's a larger change than a model swap and is out of scope here.

### Class vocabulary — 3 classes, DUO-derived (not DUO-exact)

Sea cucumber (DUO `class_id 2`, "holothurian") was removed entirely after repeated generation failures that did not resolve even against a real reference photo. Scallop was renumbered `3 -> 2` to keep IDs contiguous for YOLO training. **This pipeline's `class_id 2` means scallop, not DUO's holothurian** — note this explicitly in anything that compares back to DUO.

| class_id (this pipeline) | DUO class_id | DUO label | short name |
|---|---|---|---|
| 0 | 0 | starfish | starfish |
| 1 | 1 | echinus | sea urchin |
| 2 | 3 | scallop | scallop |

Prompt text now lives entirely in `src/prompts.py`'s `CLASSES` dict as **morphological** descriptions (body shape, texture, colour, posture), not the animal's name — naming pulls a diffusion model toward the most-photographed sense of the word, which for scallop is a seafood dish. Each class has several hand-written morphology and arrangement variants, randomly selected per row but fully deterministic given the row's seed. See that file's inline comments for the failure each specific phrasing choice fixes (sea urchin: flattened dome not sphere, darker; scallop: closed shell with a hint of living tissue, never "shell open"; starfish: small and camouflaged against rock/algae, not a large high-contrast hero shot).

For Stage 2 detector prompts, use `detector_prompts()` from `src/prompts.py` (bare noun-phrase concepts: "starfish", "sea urchin", "scallop" — not the Stage 1 scene sentences).

### Scene construction

**Density and framing** are now sampled from richer variant lists in `src/prompts.py` (`SCENE_DENSITIES`, `FRAMING`, plus independent camera-height/FOV/motion/lighting/imaging axes) rather than a single fixed phrase per density band — see the v7 changelog for what this replaced.

**Water is always clear and colour-neutral, fixed, non-random** (`SCENE_WATER_PHRASE`): *"clear seawater with good visibility, true-to-life natural colour and no artificial colour cast."* This is deliberately not configurable per-row. The output must approximate **scene radiance** `J_c`, which is what Stage 3b's Akkaynak-Treibitz transform consumes — all water-column effects (attenuation, colour cast, backscatter, veiling light) are added at Stage 3, never here. A rewrite of this module briefly reintroduced randomized turbidity/colour-cast language at Stage 1 (see v7 changelog item 4); it was caught and reverted, and the parameter removed entirely rather than just neutralized, so there's nothing left to accidentally re-enable.

**Negative prompt exists but Klein can't use it.** `Flux2KleinPipeline.__call__` accepts `negative_prompt_embeds` but has **no `negative_prompt` parameter** — verified against the installed diffusers. `src/prompts.py`'s `NEGATIVE` string (aquarium/coral/product-photo/caterpillar/dead-shell exclusions, built up across every smoke-test failure so far) is therefore restated affirmatively via `POSITIVE_ONLY_GUARDS` and appended to the positive prompt whenever `supports_negative=False`, which is always true for the current single-model setup.

**Resolution:** 1024×1024 native. Resize/crop to 640×640 at Stage 4, not here.

### Manifest

3 classes = 7 non-empty combinations (3 singles + 3 pairs + 1 triple). The 11-row design covers every combination once, then reinforces singles and the all-three case:

| Row | Classes | Density | Framing |
|---|---|---|---|
| 1-3 | each class alone | sparse | close-up |
| 4-6 | all 3 pairs | moderate | mid |
| 7 | all three | moderate | mid |
| 8-10 | each class alone (repeat, new seed) | moderate | mid |
| 11 | all three (repeat, new seed) | sparse | close-up |

Sidecar JSON per image: `{image_id, model, prompt, negative_prompt, seed, steps, guidance, classes, density, framing, prompt_metadata}` — `prompt_metadata` is the full `PromptMetadata` from `src/prompts.py` (which of each diversity axis was selected), kept for dataset auditing even though the manifest itself only fixes class_ids/density/framing/seed.

### Class balance — how it is controlled

Image-level balance is exact, by construction:

| Rows | Images per class |
|---|---|
| 1-3 singles | 1 |
| 4-6 pairs (each class in 2 of 3) | 2 |
| 7 triple | 1 |
| 8-10 singles repeat | 1 |
| 11 triple repeat | 1 |
| **Total** | **6 of 11, every class** |

Instance-level is balanced *in expectation*, but **cannot be enforced**, because two uncontrolled stages intervene:

1. **Generator non-compliance** — request 2 scallops, get 7-11 (observed repeatedly across smoke tests).
2. **Detector recall bias** — if SAM3 finds 90% of starfish and 40% of scallops, the labelled distribution reflects the *detector*, not the scene.

(2) is the subtle one: measured class balance is partly a property of the annotator. So balance is **measured, not enforced** — Stage 2 emits the counts, and those drive over-sampling in the *next* manifest.

### Checkpoint 1.5 — generation comparison
Review all 11 (small enough at this scale):
- **Compositional obedience** — requested classes present, at requested density?
- **Sense correctness** — live animals on a seabed, not food, not illustrations?
- **Anatomical correctness** — does the class actually look like the real animal (this is what killed sea cucumber and, on a different model, sea urchin on SD3.5)?
- **Colour neutrality** — clean enough to serve as `J_c` for Stage 3b?
- **Scale/camouflage realism** — checked against real annotated DUO frames, not just "looks like a real animal" (this is what caught starfish being rendered as an oversized, high-contrast hero shot in an earlier round)
- Visual realism, artifact presence

---

## Stage 2 — Auto-annotation (SAM3 only - see resolution below; originally SAM3 vs Grounding DINO)

- **SAM 3**, one independent pass per class (originally run alongside Grounding DINO for comparison - see "Resolved at n=20 pilot" below for why GDINO was dropped)
- **Text prompts:** `detector_prompts()` from `src/prompts.py` — `starfish`, `sea urchin`, `scallop`. Not the Stage 1 scene sentences, and never `echinus` (sea cucumber/holothurian is gone entirely, see v7 changelog).
- **Output:** YOLO `.txt` from boxes (mask-to-box for SAM3), using this pipeline's own `class_id` (0=starfish, 1=sea urchin/echinus, 2=scallop — **not** DUO's numbering, see Stage 1) regardless of which prompt triggered the detection
- Annotate the **clean** Stage 1 image. Stage 3 is pixel-only, so the same label file is valid for the DR'd copy.

**SAM 3 is one concept per forward pass.** `Sam3Processor` takes a single noun phrase (it wraps a CLIPTokenizer) — three classes means three passes per image. Compute vision features once and reuse:

```python
img_inputs = processor(images=image, return_tensors="pt").to(model.device)
vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)
for prompt in ["starfish", "sea urchin", "scallop"]:
    text_inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    outputs = model(vision_embeds=vision_embeds, **text_inputs)
```

`post_process_instance_segmentation(...)` returns `boxes` in absolute pixel xyxy, plus `masks` and `scores`.

**The presence head is mandatory here:**

```python
final_scores = outputs.pred_logits.sigmoid() * outputs.presence_logits.sigmoid()
```

Most manifest rows contain only a subset of the three classes, so we constantly prompt for concepts that are absent. Without the presence multiplier, absent classes produce confident false positives and Checkpoint 2.5 wrongly rejects SAM 3.

**Grounding DINO: one class per pass, not a period-joined prompt.** GDINO returns matched text spans and routinely emits partial phrases, which is exactly how "sea urchin"/"sea cucumber" used to collide on the token "sea" — with sea cucumber gone this specific collision no longer applies, but one-prompt-per-pass stays the rule since new collisions could exist between remaining classes' text.

**Fallback** if SAM3 access is revoked: Grounding DINO (text->box) + SAM2 (box->mask). SAM2 cannot take a text prompt alone, so it must be paired with GDINO to stay concept-driven.

**Required output — `reports/class_counts.json`:** per engine, per class: instance count, image count, mean instances per image where present, and mean confidence. This is a pilot deliverable, not a nice-to-have — it is the only measurement of the class balance actually achieved, and it drives the full-scale manifest.

### Checkpoint 2.5 — annotation comparison
- Box/mask correctness per class
- SAM3 vs GDINO **per class**, not just overall — pick the stronger engine, or a per-class hybrid if there is a clear split
- Expect scallop to be weakest (low contrast against substrate, partial burial, and the persistent count-overshoot problem noted in Stage 1).
- Cross-check against Checkpoint 1.5: a low instance count means either the generator did not draw them or the detector did not find them. Distinguish these — they have opposite fixes.

**Resolved at n=20 pilot: SAM3 only, GDINO dropped.** Measured on all 20 klein
images (`reports/class_counts.json`): starfish 44 vs 41 instances (12/20 vs
12/20 images) and sea urchin 48 vs 29 (12/20 vs 15/20) were close enough to
call either way, but scallop was not — SAM3 found 70 instances across 17/20
images, GDINO found 4 across 3/20. That gap alone makes GDINO unusable for
this pipeline; no per-class hybrid was worth the added complexity. `src/annotate.py`
now runs SAM3 only. GDINO's Stage 2 outputs from the pilot
(`outputs/1-pilot/labels/gdino/`, `outputs/1-pilot/viz/gdino/`) are kept as
the historical record, not deleted.

---

## Stage 3 — Domain randomization (physics, not generative)

### The two-variable correction (v5)

The revised Akkaynak-Treibitz model is:

```
I_c = J_c · e^(−β_c^D · z)  +  B_c^∞ · (1 − e^(−β_c^B · z))
      └─ direct signal ─┘      └──── backscatter ────┘
```

`z` here is **camera-to-object range**. `B_c^∞`, the veiling light, is set by the ambient illumination in the water, which is a function of **vertical depth below the surface**. These are two independent variables, and v4 collapsed them into one:

| | symbol | pilot value | controls |
|---|---|---|---|
| Vertical depth | `d` | **0-5 m** | veiling-light colour and intensity — *the "shot at 5m" look* |
| Camera range | `z` | ~0.3-4 m | haze buildup, near-vs-far falloff within the frame |

"Images should look like they were taken 5m down" is achieved by deriving `B^∞` and the ambient spectrum from the Jerlov type integrated over `d = 0..5 m` — which is exactly what a Jerlov depth chart plots. It is **not** achieved by capping `z` at 5m, which is a physically different operation and would look wrong. This also matches the published finding that β_B depends more strongly on ambient light (monotonic with depth) than on camera-to-target distance.

### 3a — Range estimation

Depth Anything V2 on each **clean** Stage 1 image -> relative inverse depth.

- Normalize predicted disparity to [0,1], invert to relative distance, linearly map to `[z_near, z_far]` metres with `z_near ≈ 0.3`, `z_far` sampled per image in ~[2, 4] m to match close-up/mid ROV framing
- Record `z_near`, `z_far` and normalization bounds in the per-image config JSON

**Why DA-V2 at all, given it is out-of-domain underwater?** Two reasons, and they hold:

1. We need only **relative ordering** (which pixels are farther), then impose our own metric scale. Relative ordering is what monocular depth models are reliable at even out-of-domain; the metric error is discarded by construction.
2. Its output is what makes haze **spatially varying** — background hazier than foreground. Without it, Stage 3b degenerates to a flat per-channel colour shift, which is exactly what `ImageTranform/domain_randomize_batch.py` does and it looks obviously fake.

The v4 clean-generation decision also substantially reduces the domain gap: DA-V2 now runs on a bright, high-contrast, clear-water image rather than a murky one. If it still underperforms at Checkpoint 3, the fallback is Depth Pro, not a hand-rolled gradient prior.

**Resolved: DA-V2 Large chosen over Apple Depth Pro, plus a free guided-filter refinement.** `src/depth_compare.py` ran both models head to head on 3 pilot images (`outputs/depth_compare/`):

- **First pass** (`comparison.png`, whole-image, globally normalized) made Depth Pro look worse — its near-field region (where the target classes live) appeared to collapse to flat black. That turned out to be a normalization artifact, not a real gap: Depth Pro outputs metric depth (far = high value) while DA-V2 outputs inverse depth/disparity (near = high value), and Depth Pro's raw range is dominated by the distant background, so global percentile normalization crushed its near-field detail into a sliver of the display range.
- **Fair pass** (`detail_comparison.png`, cropped to the near-field region, normalized locally, plus Sobel edge maps) told a more honest story: Depth Pro genuinely resolves more fine-grained surface texture (individual gravel/pebble relief) than DA-V2, which is smoother at that scale — consistent with Depth Pro's own paper. But on the boundary that actually matters here — the target-object silhouettes (starfish/urchin/scallop) — both models produce comparably clean, closed contours. Depth Pro's extra detail is mostly ambient substrate texture, not sharper object edges specifically.
- **Guided-filter check** (`fine_tuned_comparison.png`): tested whether a training-free edge-aware filter (He/Sun/Tang 2010 guided filter, RGB luminance as guide, pure numpy/scipy, no model weights) could sharpen the "blobby" object boundaries either model produces. It helps modestly — edges snap a little tighter to the RGB guide — but doesn't turn blobby depth into segmentation-crisp depth. That's judged to be the honest ceiling of a training-free approach; true precision would need fine-tuning against real ground-truth depth, which we don't have for 2D-diffusion-generated scenes (see the "fine-tuning" question below).
- **Decision:** DA-V2 Large, with the guided filter applied as a cheap refinement step. Reasons beyond the roughly-tied detail comparison: DA-V2 is 2-10x faster (0.07-0.36s vs Depth Pro's 0.77-0.85s/image), matches this pipeline's explicit relative-ordering-only requirement (Depth Pro's metric output is a capability we don't need and can't validate for a synthetic scene anyway), and gave equally clean target-object contours.
- **Why not fine-tune a depth model for sharper boundaries instead?** Considered and set aside for now: fine-tuning needs ground-truth depth to train against, which doesn't exist for images produced by a 2D diffusion model (unlike a 3D-rendered scene). It would also be solving a problem that may not need solving — some of the "blobbiness" reflects real geometry (a starfish resting flush on a rock has a small true height difference, so smooth depth there isn't necessarily wrong), and Stage 3a's range map is explicitly a randomization driver, not ground truth (see the standing caveat below) - precision beyond what Stage 3b actually consumes would be effort spent on a metric nothing downstream uses.

Standing caveat: this is an *estimated* range map of a *generated* scene. It is a plausible randomization driver, not ground truth, and nothing downstream should treat it as measured.

### 3b — Physics transform

Deterministic Python script. Per image, sample:

- **Jerlov type** from **{1C, 3C, 5C}** — coastal types, per your specification
- **Vertical depth** `d ~ U(0, 5)` m — sets ambient spectrum and `B_c^∞`
- **Range map** `z(x,y)` from 3a
- `β_c^D` and `β_c^B` from the Jerlov type's IOPs

**Coefficient source — resolved.** There is no universal, camera-independent Jerlov coefficient table. Akkaynak et al. (CVPR 2017, "What Is the Space of Attenuation Coefficients in Underwater Computer Vision?") show the RGB-domain ratios (β_B/β_R, β_B/β_G) depend on camera spectral response, imaging range, and scene reflectance (their Eq. 9) — they are a projection at one operating point, not a fixed physical constant. Solonenko & Mobley (2015) is paywalled; both that CVPR17 paper and Berman et al. (BMVC 2017, "Diving into Haze-Lines") present the per-type ratios only as scatter plots (their Fig. 3a / Fig. 2-middle), never a printed table.

`src/jerlov.py` adopts the `'peak'`-branch values from **`danaberman/underwater-hl`'s `get_water_types.m`** (BMVC 2017, cited by dozens of follow-on underwater-vision papers), which evaluates exactly this projection at a camera-agnostic peak-sensitivity approximation (475/525/600nm for B/G/R):

| Type | β_B/β_G | β_B/β_R |
|---|---|---|
| 1C | 0.7937 | 0.2773 |
| 3C | 0.9539 | 0.4051 |
| 5C | 1.0930 | 0.4642 |

**One flagged inference, confirmed with the user before use:** the source file's `water_types` list names 10 Jerlov types (I, IA, IB, II, III, 1C, 3C, 5C, 7C, 9C — this ordering is independently confirmed by Fig. 3a's legend in the CVPR17 paper), but its `'peak'` coefficient arrays hold only 8 values with no comment on which two are omitted. We read the array as I, II, III, 1C, 3C, 5C, 7C, 9C (dropping the closely-spaced oceanic IA/IB) — the values increase monotonically with turbidity as physically expected, which is consistent with but does not prove this reading.

These are ratios only; `beta_rgb()` in `src/jerlov.py` derives absolute per-channel β from a sampled `β_B` — the ratios do not pin an absolute scale, and neither Berman's method nor ours needs one until this stage.

**β^D vs β^B — simplified to equal.** The revised model treats direct-signal and backscatter attenuation as physically distinct coefficients. This pipeline has only one attenuation-ratio table, not two, so `domain_randomize.py` uses `beta_rgb()`'s output for both (`β_c^D = β_c^B`). Flagged, not silently assumed.

**`B_c^∞(d)` — veiling light, resolved as a flagged simplification after a sourcing attempt that came up short.** `B_c^∞` needs the per-channel ambient/backscatter colour as a function of vertical depth `d`, normally read off a Jerlov depth-irradiance chart (`K_d(λ)` per water type). Two sources were checked directly before falling back to a simplification:

- **Williamson & Hollins 2023** ("Depth profiles of Jerlov water types," *Limnol. Oceanogr. Lett.* 8:781–788) — user provided the PDF, read in full. Turned out to study whether a water column's Jerlov *type classification* drifts with depth, not `K_d` magnitude by wavelength. Its own Table 3 (reconstructed from Jerlov 1976 fig. 71) stops at type "1C" — no 3C/5C data exists there either; the actual `K_d(λ)` numbers live in a supplementary figshare dataset the PDF references but doesn't contain.
- One genuinely useful thing from that paper: its finest depth resolution near the surface is a single **0–10m bucket**. This pilot's entire `d` range (0–5m) sits inside that one bucket — there is no published evidence of resolvable optical change within 0–5m specifically, so treating a chosen water type's attenuation as constant across the full `d` range (rather than deriving a within-band depth curve) isn't cutting a corner the literature would otherwise resolve.
- **Solonenko & Mobley 2015** ("Inherent optical properties of Jerlov water types," *Appl. Opt.* 54(17):5392–5401) is the primary IOP source this whole ratio table already wanted (see above) and would likely resolve this properly if it becomes available — not yet obtained.

**Simplification used** (`kd_rgb()` in `src/jerlov.py`, two stacked flagged assumptions):
1. `K_d`'s per-channel spectral **shape** reuses the same `β_B/β_G`, `β_B/β_R` ratios as beam attenuation above — diffuse and beam attenuation are physically different quantities; this treats them as sharing the same relative R/G/B shape per water type. Plausible, not verified.
2. `K_d`'s absolute green-channel **magnitude** is anchored to one real citation found during research — a secondary source stating a `K_d` of 0.2763 m⁻¹ is compatible with Jerlov coastal types 3C–5C at 500–550nm — applied **uniformly across 1C/3C/5C**. This does not differentiate absolute `K_d` magnitude between the three coastal types (only their R/G/B shape, via the ratios), which is a real loss of information Jerlov's type ordering implies (1C should genuinely attenuate less than 5C).

Then: `B_c^∞(d) = b_ref · exp(−K_{d,c} · d)`, with `b_ref ~ U(0.7, 1.0)` a per-image sampled achromatic baseline (the veiling light's colour comes entirely from the per-channel `K_d` exponential, not from `b_ref` being tinted — matching how real veiling light starts as ~white sunlight and gets tinted by the water column). `b_ref`'s range, and `β_b`'s sampling range `U(0.3, 1.5)` m⁻¹, are both plausible-order-of-magnitude choices, not sourced from a table.

Outputs:
- `outputs/1-pilot/dr/<image_id>_dr.png` — DR'd copy of each image, **label file unchanged** (pixel-only transform, verified — see Checkpoint 3)
- `configs/<image_id>_dr.json`: Jerlov type, `d`, `z_near`/`z_far`, per-channel β^D and β^B, `K_d`, `B^∞`, `b_ref`, camera-effect params (below), seed

DUO-specific calibration is **deferred to the full-scale run**, per decision.

**Recalibration: first pass was "way too extreme, none of the images close to DUO."** `β_b` was originally sampled from a single fixed `U(0.3, 1.5)` m⁻¹ range independent of the image's own scale — a merely-average sample on a wide shot (`z_far` up to 6m) fully blacked out the frame well before the far edge, while the same sample on a close-up shot looked fine. Fixed by coupling `β_b`'s sampling ceiling to each image's own `z_far` (`z_far_beta_b_cap()` in `domain_randomize.py`): the red channel (fastest-attenuating) is guaranteed to retain at least a `visibility_floor` fraction of direct signal at that image's own far edge.

**Then: DR and original looked too similar across the board.** The fix above, done with one fixed `visibility_floor`, made every image similarly (mildly) hazy - it eliminated the extreme tail but also the dark tail. Fixed by sampling `visibility_floor` itself per image from `U(0.08, 0.45)` (was a fixed 0.25) and widening `b_ref` to `U(0.5, 1.0)` (was `U(0.7, 1.0)`) - the DR set now spans mild to notably dark/murky rather than clustering at one "safe" look, while the z_far-coupling still prevents pure blackout regardless of how dark a given draw is.

**Three camera-level effects added, applied after the physics transform in real-pipeline order (lens → optics → sensor):** Jerlov/Akkaynak-Treibitz only covers turbidity/haze/backscatter and the depth-component of ambient lighting - it says nothing about the camera itself. Added, all pure numpy/opencv, no model weights:
- **Vignette** — multiplicative radial darkening from a slightly off-centre point (simulating an ROV-mounted light, not a lens-centred one). `strength ~ U(0, 0.35)`, centre offset `~ U(-0.15, 0.15)` of half-width/height.
- **Motion blur** — mild linear kernel, `length ~ U(0, 4)` px at 1024 resolution (0 = no blur for some images), `angle ~ U(0, 360)`°. Reinforces what the Stage 1 prompts already ask for textually (`prompts.py`'s `CAMERA_MOTION`) but can't reliably guarantee as an actual pixel effect.
- **Sensor noise** — signal-dependent shot+read Gaussian noise (`N(0, σ_read² + σ_shot²·pixel_value)`, the standard heteroscedastic approximation per Brooks et al. CVPR 2019), scaled up with `d` (deeper → the ROV's camera would gain up → more visible noise).

Not covered by any of the above, and not yet added: non-Jerlov-depth-dependent lighting variation (sun angle, caustics), that's a bigger scope decision than the three above.

Note: `C:\dev\ImageTranform\domain_randomize_batch.py` is an ad-hoc per-channel RGB shift, not this model. Reusable as batch-loop scaffolding only.

### Checkpoint 3 — DR verification
- **Overlay the unchanged label boxes on the DR'd image and confirm alignment.** Cheapest possible test that the transform did not resize, crop or flip — and it directly tests a pilot success criterion. **Done for 5 pilot images** (`outputs/1-pilot/viz/dr_check/`), re-run after the recalibration and the vignette/motion-blur/sensor-noise additions — boxes still land exactly on the starfish/sea urchin/scallop silhouettes (including `pilot_020`'s sea urchin and scallop classes, and `pilot_012`'s strong vignette case), confirming all three new effects stayed pixel-only.
- Eyeball against DUO: does 3C at d≈4m land in the right neighbourhood, or is it far too dark/green? Records whether Jerlov defaults are close enough to make explicit DUO calibration worthwhile at scale. **Not yet done** - worth a pass once more DUO reference frames are on hand.

**Box alignment ≠ detectability - a distinction worth being explicit about.** The box-alignment check above only proves the transform is geometric-free (no resize/crop/flip) - since `domain_randomize.py` never touches array shape or position, that alignment is guaranteed by construction, not really a discovery. It says nothing about whether the *object* is still visually recognizable inside that box after haze/vignette/noise, which is a real and separate risk now that `visibility_floor` deliberately has a dark tail (down to 0.08).

There are two distinct downstream failure modes this doesn't rule out:
1. **The object becomes undetectable** (count/confidence collapses toward zero on a given image) - expected to *some* degree since that's what makes DR useful for robustness training, but a full collapse on a given image would mean that image-label pair is effectively mislabeled (the "answer" isn't recoverable from the pixels), not useful augmentation.
2. **Wrong-class detection in degraded regions** - SAM3 runs one independent forward pass per class prompt (never a joined multi-class prompt), so a degraded region could plausibly trigger a false positive under the *wrong* class's pass (e.g. a hazy scallop read as sea urchin). This wouldn't show up as a simple count drop - it shows up as a wrong-coloured box on the wrong species, and needs an actual re-detection pass to check, not an assumption from the alignment test.

**Check performed: rerun SAM3 on the DR'd set** (effectively Checkpoint 2.5's methodology - the one that caught GDINO's scallop weakness - applied to the DR'd images instead of the clean ones):
- Labels: `outputs/1-pilot/labels/dr_sam3/` (kept separate from `labels/sam3/`, which holds the clean-image ground truth these DR'd images' *actual* training labels are copied from unchanged - Stage 3b's whole label-reuse scheme depends on that copy staying untouched)
- Report: `reports/class_counts_dr.json` (kept separate from `reports/class_counts.json`, the clean-image baseline this is compared against)
- Viz: `outputs/1-pilot/viz/dr_sam3/` - colour-coded per class (red=starfish, green=sea urchin, blue=scallop, same convention as every other viz in this pipeline), for manual review at n=20 rather than an automated cross-class-confusion metric

**Quantified with `src/dr_detection_check.py`** - matches each original ground-truth box against the DR re-detections by IoU (>=0.3 threshold, same class = matched, different class = misclassified, no match = missed):

| class | total | matched | misclassified | missed |
|---|---|---|---|---|
| starfish | 44 | 44 | 0 | 0.0% |
| sea urchin | 48 | 37 | 0 | 22.9% |
| scallop | 70 | 45 | 0 | 35.7% |
| **all** | **162** | **126** | **0** | **22.2%** |

**Zero misclassification** - the wrong-class-detection concern (Stage 2's independent-per-class-prompt design means a degraded region could plausibly fire under the wrong class) didn't materialize at this threshold; SAM3's presence gate abstains on degraded regions rather than firing a wrong-class false positive. **Real, class-specific detectability loss instead:** starfish survives DR perfectly, scallop - already this pipeline's weakest class throughout (low contrast, partial burial, camouflaged-by-design morphology) - loses over a third of its instances. This is a difficulty-calibration signal for `visibility_floor`'s dark tail (down to 0.08), not a tooling problem - **the fix, if one is needed, is narrowing that range in `domain_randomize.py`, not swapping or fine-tuning SAM3** (SAM3's actual job is annotating the *clean* images, where it already performs well - see Checkpoint 2.5).

**Architecture note:** the DR'd images that feed Stage 4 always carry the **original clean-image labels** (`outputs/1-pilot/labels/sam3/`), never the `dr_sam3` re-detections - using a detector's own predictions on the degraded image as ground truth would record false negatives for real, correctly-labeled instances and actively teach the wrong thing. `dr_sam3` is diagnostic-only. The actual DR'd-image + original-label training pairs are visualized at `outputs/1-pilot/dr_labeled/` (via `visualize_annotations.py --strip-suffix _dr`).

---

## Stage 4 — Dataset assembly

**Goal driving this stage:** train YOLO26s under a 2x2 ablation - {original, original+DR} x {no augmentation, with augmentation} - to measure whether DR and/or augmentation each independently help.

**Augmentation architecture, decided:** augmentation is applied **on-the-fly during YOLO training** via Ultralytics' built-in hyp config (HSV jitter, flips, rotation, mosaic, etc.), not pre-baked into extra image files. This matters for the geometric-augmentation concern raised when planning this stage: pre-baked geometric transforms (à la Stage 3's DR) would need their own box-coordinate recomputation, since a fixed saved file reuses the original label as-is. On-the-fly training-time augmentation doesn't have that problem - Ultralytics transforms box labels alongside its own geometric ops every step, which is exactly why on-the-fly geometric augmentation is normal practice in YOLO training. Consequence: this stage only assembles **2** dataset variants, not 4 - the augmentation axis becomes a Stage 5 training flag.

`src/assemble_dataset.py` produces:
- `dataset/original/` — 20 raw Stage 1 images
- `dataset/original_dr/` — raw + DR'd (40 pairs), the same label file reused for both copies
- `dataset/hyp/no_aug.yaml` — Ultralytics hyp override with every augmentation term zeroed, for the two no-augmentation runs; the with-augmentation runs use Ultralytics' own defaults directly, no file needed

**Resize:** 640×640, not center-crop. Every image in this pipeline is square (1024×1024) throughout, so a straight resize is a pure isotropic scale - YOLO's normalized `cx,cy,w,h` coordinates are invariant to that, so labels are copied byte-unchanged rather than recomputed. Verified, not just assumed: compared `dataset/original_dr/labels/train/pilot_012_klein_dr.txt` against the original `outputs/1-pilot/labels/sam3/pilot_012_klein.txt` (identical), and re-ran the box-overlay check at the new 640×640 resolution (`visualize_annotations.py`) - boxes still land exactly on the starfish. This resize logic would need reworking if a future stage introduces non-square images or an actual crop.

- **A raw image and its DR'd copy always land in the same split** — grouped by `image_id` before splitting (`split_ids()`), so a recoloured copy of a training image can never end up in val while its raw twin is in train (leakage).
- Pilot scale defaults to `--val-count 0` (all 40 pairs in train, `data.yaml`'s `val:` points at `images/train` as a fallback so Ultralytics doesn't choke on a missing/empty val split) — a handful of val images carries no statistical signal at this scale and this plan's criteria are explicitly qualitative. The split logic itself is real and reusable via `--val-count N` for the full-scale run.

**Stage 5 stays ON HOLD** (below) — this stage prepared what the 4 planned runs need but did not launch any training.

---

## Stage 5 — ON HOLD

Paused pending manual review of Stage 1-4 outputs. Do not start until explicitly resumed.

---

## Pilot success criteria (qualitative, not statistical at n=20)

- Klein produces visually plausible, class-diverse images with the **correct sense** (live animal, not food) and **correct anatomy** for all 3 remaining classes
- Stage 1 output is colour-neutral enough to serve as `J_c` input to Stage 3b
- At least one annotation engine gets usable boxes on all 3 classes
- DR transform runs without breaking label/box alignment (verified by overlay at Checkpoint 3)
- DR'd output is recognisably 0-5m coastal water, not black and not neon green
- Clear enough result to decide: which annotation engine

## Deliverables back into the main plan

- Generation model: **Klein** (decided — see v7 changelog for why SD3.5 was dropped)
- Class set: **3-class DUO-derived subset** (starfish, sea urchin, scallop — sea cucumber dropped, see v7 changelog)
- Chosen annotation engine/combo
- `reports/class_counts.json` — achieved instance balance per class per engine, to drive full-scale manifest over-sampling
- Estimated auto-annotation error rate from the full-batch review
- Cost/time-per-image on the pod GPU, to budget the full-scale batch
- Whether Jerlov 1C/3C/5C defaults are close enough to DUO to make explicit calibration worthwhile
