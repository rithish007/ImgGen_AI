# AI-Pipeline Smoke Test — Technical Build Plan (v6)

**Purpose:** validate an all-AI alternative to the UE5.8 pipeline (generation -> annotation -> DR -> YOLO26) on a small pilot batch before committing to full-scale dataset generation. This is a pipeline-mechanics check, not a scientific result — sample size is deliberately tiny at this stage.

**Compute:** RunPod Pod, **RTX 5090 (32GB VRAM)**, On-Demand, **Secure Cloud** with a network volume (volumes are unavailable on Community Cloud; pod and volume must share a datacenter). Nothing runs on the local machine — the local RTX 4060 Laptop (8GB) is for editing only. Sync via Git: edit in PyCharm -> `git push` -> SSH to pod -> `git pull` -> run -> pull results back.

**Repo:** `https://github.com/rithish007/ImgGen_AI.git` (branch `main`)

---

## Changelog

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

## Stage 1 — Base image generation (dual-model comparison)

Same manifest through both models, both bf16:

| Model | Repo | Params | Steps | Guidance | VRAM (bf16, 1024²) |
|---|---|---|---|---|---|
| FLUX.2 klein base | `black-forest-labs/FLUX.2-klein-base-9B` | 9B | 50 | 4.0 | ~29GB |
| SD 3.5 Large | `stabilityai/stable-diffusion-3.5-large` | 8B MMDiT | 28-40 | 3.5-4.5 | ~26GB |

Use the **base** klein variant, not the step-distilled `FLUX.2-klein-9B`. Distilled runs in 4 steps but has materially weaker prompt adherence and less sample diversity — and compositional obedience is exactly what Checkpoint 1.5 measures. At n=20 the extra steps are free.

**Do not hold both models in VRAM at once** (29 + 26 > 32). Two separate passes; release the pipeline between them. Klein base at ~29GB is close enough to the ceiling that `enable_model_cpu_offload()` may be needed — confirm on the first image, not at image 15.

Both models are native 1024×1024, so neither gets a resize advantage before Stage 4.

**Pilot batch:** 20 images per model (40 total)

### Class vocabulary and prompt text

| class_id | DUO label | short name | **prompt phrase (use this)** |
|---|---|---|---|
| 0 | starfish | starfish | `a starfish on the seabed` |
| 1 | echinus | sea urchin | `a spiny sea urchin on the rocky seabed` |
| 2 | holothurian | sea cucumber | `a live sea cucumber crawling on the sandy seabed` |
| 3 | scallop | scallop | `a live scallop, shell open, resting on the sandy seabed` |

Two separate reasons for this table, both load-bearing:

- **Common name over taxonomic name.** VLMs are trained on web captions where "sea urchin"/"sea cucumber" are common and "echinus"/"holothurian" are rare. The taxonomic terms are for `class_id` mapping only — never typed into a generator or detector.
- **Habitat sense over culinary sense.** Bare "scallop" and "sea cucumber" resolve to food in web image data. The qualifiers (`live`, `on the seabed`, `crawling`, `shell open`) force the biological sense. Starfish and sea urchin do not have this problem but are phrased consistently.

For Stage 2 detector prompts, use the **short name** column (detectors want a bare noun phrase concept, not a scene description).

### Scene construction

**Density:** sparse (2-3 instances) or moderate (4-6 instances) per image
**Framing:** close-up to mid, ROV forward-facing — cues: wide-angle lens, slight fisheye distortion

**Scene descriptor — the v4 change, refined.** Do not prompt the abstraction "clear water". Name a real photographic condition that is *both* underwater *and* colour-neutral:

> `underwater photograph, shallow tropical reef flat, bright sunlight, sun caustics rippling across the sandy seabed, clear water, high visibility, natural colour, neutral white balance, wide-angle ROV camera`

**Sun caustics are the critical cue** — dappled light on the seabed is unmistakably underwater and occurs only in clear shallow water. It buys "obviously submerged" without buying a colour cast. This is what prevents the two failure modes: aquarium drift and green-murk-anyway.

The output must approximate **scene radiance** `J_c`, which is what Stage 3b consumes. All water-column effects — attenuation, colour cast, backscatter, veiling light — are added at Stage 3, never baked in here.

**Negative prompt (SD 3.5 only):**
`text, watermark, diver, boat, human, water surface, sky, green tint, murky, hazy, low visibility, colour cast, dark, aquarium, fish tank, glass, white background, studio, illustration, drawing, cartoon, 3d render`

**Negative-prompt asymmetry — verified against the installed diffusers.** `Flux2KleinPipeline.__call__` accepts `negative_prompt_embeds` but has **no `negative_prompt` parameter**; `StableDiffusion3Pipeline` has both. Passing a negative to Klein would silently do nothing. For Klein the exclusions are therefore restated affirmatively and appended to the positive prompt (`colour-accurate, evenly lit, open natural habitat, unobstructed view of the seafloor`), handled automatically by `supports_negative` in `src/prompts.py`.

Record this at Checkpoint 1.5: SD 3.5 gets true negative guidance and Klein does not, so a difference in murk/aquarium artifacts between the two models is partly an artifact of the interface, not purely of model quality.

**Resolution:** 1024×1024 native both models. Resize/crop to 640×640 at Stage 4, not here.

### Manifest

4 classes = 15 non-empty combinations. At n=20 cover every combination once, then reinforce single-class exemplars:

| Row | Classes | Density | Framing |
|---|---|---|---|
| 1-4 | each class alone | sparse | close-up |
| 5-10 | all 6 pairs | moderate | mid |
| 11-14 | all 4 triples | moderate | close-up |
| 15 | all four | moderate | mid |
| 16-19 | each class alone (repeat, new seed) | moderate | mid |
| 20 | all four (repeat, new seed) | sparse | close-up |

Sidecar JSON per image: `{image_id, model, prompt, negative_prompt, seed, steps, guidance, classes, density, framing}`.

**Seeds do not transfer across models.** Flux and SD3.5 use different latent geometry and schedulers, so a row's seed gives different compositions per model. The manifest holds *prompts* constant; it does not produce paired images.

### Class balance — how it is controlled

Image-level balance is exact, by construction:

| Rows | Images per class |
|---|---|
| 1-4 singles | 1 |
| 5-10 pairs (each class in 3 of 6) | 3 |
| 11-14 triples (each class in 3 of 4) | 3 |
| 15 all-four | 1 |
| 16-19 singles repeat | 1 |
| 20 all-four | 1 |
| **Total** | **10 of 20, every class** |

Instance-level is balanced *in expectation* (~22 instances/class/model by symmetry), but **cannot be enforced**, because two uncontrolled stages intervene:

1. **Generator non-compliance** — request 3 scallops, get 1.
2. **Detector recall bias** — if SAM3 finds 90% of starfish and 40% of scallops, the labelled distribution reflects the *detector*, not the scene.

(2) is the subtle one: measured class balance is partly a property of the annotator. So balance is **measured, not enforced** — Stage 2 emits the counts, and those drive over-sampling in the *next* manifest. At n=20 the counts are too noisy to act on; the mechanism must exist now so it is ready at scale.

### Checkpoint 1.5 — generation comparison
Review all 40 (small enough at this scale):
- **Compositional obedience** — requested classes present, at requested density?
- **Sense correctness** — live animals on a seabed, not food, not illustrations?
- **Colour neutrality** — clean enough to serve as `J_c` for Stage 3b, or did the model impose a cast despite the negatives?
- Visual realism, artifact presence
- Decide: one model for scale-up, or keep both as parallel tracks

---

## Stage 2 — Auto-annotation (SAM 3 vs Grounding DINO)

- **SAM 3** and **Grounding DINO**, same 40 images, two independent passes
- **Text prompts:** the short names — `starfish`, `sea urchin`, `sea cucumber`, `scallop`. Not the Stage 1 scene sentences, and never `echinus`/`holothurian`.
- **Output:** YOLO `.txt` from boxes (mask-to-box for SAM3), using DUO-aligned `class_id` (0=starfish, 1=echinus, 2=holothurian, 3=scallop) regardless of which prompt triggered the detection
- Annotate the **clean** Stage 1 image. Stage 3 is pixel-only, so the same label file is valid for the DR'd copy.

**SAM 3 is one concept per forward pass.** `Sam3Processor` takes a single noun phrase (it wraps a CLIPTokenizer) — four classes means four passes per image. Compute vision features once and reuse:

```python
img_inputs = processor(images=image, return_tensors="pt").to(model.device)
vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)
for prompt in ["starfish", "sea urchin", "sea cucumber", "scallop"]:
    text_inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    outputs = model(vision_embeds=vision_embeds, **text_inputs)
```

`post_process_instance_segmentation(...)` returns `boxes` in absolute pixel xyxy, plus `masks` and `scores`.

**The presence head is mandatory here:**

```python
final_scores = outputs.pred_logits.sigmoid() * outputs.presence_logits.sigmoid()
```

Most manifest rows contain only a subset of the four classes, so we constantly prompt for concepts that are absent. Without the presence multiplier, absent classes produce confident false positives and Checkpoint 2.5 wrongly rejects SAM 3.

**Grounding DINO: one class per pass, not a period-joined prompt.** GDINO returns matched text spans and routinely emits partial phrases; `"sea urchin"` and `"sea cucumber"` collide on the token `"sea"`, silently corrupting the class mapping. One prompt per pass makes it unambiguous. At n=40 the cost is irrelevant.

**Fallback** if SAM3 access is revoked: Grounding DINO (text->box) + SAM2 (box->mask). SAM2 cannot take a text prompt alone, so it must be paired with GDINO to stay concept-driven.

**Required output — `reports/class_counts.json`:** per engine, per class: instance count, image count, mean instances per image where present, and mean confidence. This is a pilot deliverable, not a nice-to-have — it is the only measurement of the class balance actually achieved, and it drives the full-scale manifest.

### Checkpoint 2.5 — annotation comparison
- Box/mask correctness per class
- SAM3 vs GDINO **per class**, not just overall — pick the stronger engine, or a per-class hybrid if there is a clear split
- Expect scallop and sea cucumber to be weakest (low contrast against substrate, partial burial). If both engines fail on the same class, that is a finding about the class, not the engine.
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
- At pilot scale, consider putting all 80 pairs in train and skipping val — a 40-image val set carries no statistical signal and this plan's criteria are explicitly qualitative. Build the split logic anyway; it is needed at full scale.
- Pilot scale: up to 40 raw (both models) + 40 DR'd = up to 80 image-label pairs

---

## Stage 5 — ON HOLD

Paused pending manual review of Stage 1-4 outputs. Do not start until explicitly resumed.

---

## Pilot success criteria (qualitative, not statistical at n=20/model)

- Both models produce visually plausible, class-diverse images with the **correct sense** of each class (live animal, not food)
- Stage 1 output is colour-neutral enough to serve as `J_c` input to Stage 3b
- At least one annotation engine gets usable boxes on all 4 classes
- DR transform runs without breaking label/box alignment (verified by overlay at Checkpoint 3)
- DR'd output is recognisably 0-5m coastal water, not black and not neon green
- Clear enough result to decide: which generation model, which annotation engine

## Deliverables back into the main plan

- Chosen generation model (or reason to keep both)
- Chosen annotation engine/combo
- `reports/class_counts.json` — achieved instance balance per class per engine, to drive full-scale manifest over-sampling
- Estimated auto-annotation error rate from the full-batch review
- Cost/time-per-image on the RTX 5090, to budget the full-scale batch
- Whether Jerlov 1C/3C/5C defaults are close enough to DUO to make explicit calibration worthwhile
