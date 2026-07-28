# AI-Pipeline Smoke Test — Technical Build Plan (v7)

**Purpose:** validate an all-AI alternative to the UE5.8 pipeline (generation -> annotation -> DR -> YOLO26) on a small pilot batch before committing to full-scale dataset generation. This is a pipeline-mechanics check, not a scientific result — sample size is deliberately tiny at this stage.

**Compute:** RunPod Pod, **32GB+ VRAM**, On-Demand, **Secure Cloud** with a persistent volume. Nothing runs on the local machine — local GPUs are for editing only. Pods have moved between RTX 5090 (32GB) and RTX 6000 Ada (48GB) across sessions; the pipeline is GPU-model-agnostic as long as VRAM and sm-arch checks in `scripts/pod_preflight.py` pass. Sync via Git: edit in PyCharm -> `git push` -> SSH to pod -> `git pull` -> run -> pull results back. See `POD_RUNBOOK.md` for the exact per-session commands.

**Repo:** `https://github.com/rithish007/ImgGen_AI.git` (branch `main`)

---

## Changelog

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

**Hugging Face.** Confirm license acceptance for **all five**, in particular `black-forest-labs/FLUX.2-klein-base-9B` — a FLUX.2-dev acceptance does **not** carry over to klein. Export `HF_TOKEN` on the pod. Set `HF_HOME` to the network volume so weights survive pod termination:

```bash
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=hf_...
```

**Pod git auth.** RunPod's GitHub integration builds Serverless containers from a repo; it does **not** give a Pod git credentials. If `ImgGen_AI` is public, `git clone https://...` needs no auth for pulls and nothing else is required. If private, switch the remote to SSH and use agent forwarding so no token is stored on the pod:

```bash
git remote set-url origin git@github.com:rithish007/ImgGen_AI.git
```

then connect with `ssh -A root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519`.

### Deploy runbook — when to click DEPLOY

**Do not deploy yet.** The pod bills from the moment it starts, so all code should be written, committed and pushed first. Order:

1. Scripts written locally, `requirements.txt` frozen, everything pushed to `main`
2. RunPod -> Settings -> SSH Public Keys -> paste the `id_ed25519.pub` key
3. Create the network volume **first** (Storage -> Network Volume, ~150GB, note the datacenter)
4. **Then click DEPLOY** — Secure Cloud, RTX 5090, in the volume's datacenter, PyTorch template on **CUDA 12.8 or newer** (Blackwell requires it), container disk 50GB, volume mounted at `/workspace`
5. SSH in, `git clone`, `pip install -r requirements.txt`, export `HF_HOME`/`HF_TOKEN`
6. Run Stage 0.5 first — 4 images/model, ~10 min. If output is unusable, stop and re-prompt before burning hours on the full manifest.

Sizing note: ~50GB of weights (klein-base ~18GB, SD3.5-L + T5 ~26GB, SAM3 ~3GB, GDINO ~1GB, DA-V2 ~1.5GB) plus cache overhead and outputs. 150GB is comfortable.

---

### Scripts (written, dry-run verified locally)

| File | Purpose |
|---|---|
| `src/prompts.py` | Class table, scene descriptor, prompt assembly, negative-prompt handling |
| `src/build_manifest.py` | Builds `manifests/pilot.json` (20 rows) and `manifests/smoke.json` (4 rows) |
| `src/generate.py` | Runs one manifest through one model; `--dry-run` prints prompts with no GPU |

Verified locally without a GPU:
- Image-level class balance is exact — 10 images per class across the 20 rows
- Instance-level balance lands at 23 / 24 / 19 / 22 (starfish / urchin / cucumber / scallop), i.e. within ~13% of the ~22 expected by symmetry
- The `all four classes @ sparse` row (row 20) is infeasible as specified — 4 classes cannot fit a 2-3 instance budget. `allocate_counts` applies a floor of one instance per class and flags the row with `density_floor_applied`, rather than silently dropping a class

**Iterate prompts with `--dry-run` before deploying.** It loads no model and needs no GPU, so prompt wording costs nothing to refine while the pod is off.

---

## Stage 0.5 — Generator smoke test (NEW, do this first)

Before building the 20-row manifest, generate **4 images per model** — one per class, single-class, sparse, close-up — using the Stage 1 prompt template. ~10 minutes of GPU total.

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

**Pilot batch:** 11 images (see the reworked 3-class manifest below)

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

## Stage 2 — Auto-annotation (SAM 3 vs Grounding DINO)

- **SAM 3** and **Grounding DINO**, same 11 images, two independent passes
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

Outputs:
- DR'd copy of each image, **label file unchanged** (pixel-only transform)
- `configs/<image_id>_dr.json`: Jerlov type, `d`, `z_near`/`z_far`, per-channel β^D and β^B, `B^∞`, seed

DUO-specific calibration is **deferred to the full-scale run**, per decision.

Note: `C:\dev\ImageTranform\domain_randomize_batch.py` is an ad-hoc per-channel RGB shift, not this model. Reusable as batch-loop scaffolding only.

### Checkpoint 3 — DR verification
- **Overlay the unchanged label boxes on the DR'd image and confirm alignment.** Cheapest possible test that the transform did not resize, crop or flip — and it directly tests a pilot success criterion.
- Eyeball against DUO: does 3C at d≈4m land in the right neighbourhood, or is it far too dark/green? Records whether Jerlov defaults are close enough to make explicit DUO calibration worthwhile at scale.

---

## Stage 4 — Dataset assembly

- Resize/center-crop to 640×640 here
- Ultralytics layout: `images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`
- **A raw image and its DR'd copy must land in the same split.** Splitting them puts a recoloured copy of a training image into val — leakage.
- At pilot scale, consider putting all 22 pairs in train and skipping val — an 11-image val set carries no statistical signal and this plan's criteria are explicitly qualitative. Build the split logic anyway; it is needed at full scale.
- Pilot scale: 11 raw + 11 DR'd = 22 image-label pairs

---

## Stage 5 — ON HOLD

Paused pending manual review of Stage 1-4 outputs. Do not start until explicitly resumed.

---

## Pilot success criteria (qualitative, not statistical at n=11)

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
