# 2-Pilot — Cross-Model Generation Comparison (v1)

**Purpose:** test the bivalve-guard fix (`src/prompts.py`) across multiple image-gen candidates on a shared, class-balanced-by-construction manifest, to decide which model(s) go into the 1000-image HPC run and whether the prompt itself needs further work first. This is a comparison/decision pilot, not a training run — no YOLO training happens on this data.

**Compute:** RunPod, RTX A6000 48GB, 250GB network volume, ~$10 budget. Same Git-based sync workflow as `POD_RUNBOOK.md`.

**The 6 questions this test must answer** (user's list, verbatim):
1. Is the prompt able to generate a class-balanced set of dataset?
2. Can this be used for the massive 1000-image generation using HPC H100?
3. Which image-gen model performs best for quality?
4. Can the rock-formation pool be added to the prompt, and how?
5. Which non-automated processes can be automated without affecting the result?
6. How is SAM3 performing — are detected classes balanced relative to requested counts?

---

## Open decisions — resolve before spending pod time

1. **RTX A6000 vs RTX 6000 Ada.** `generate.py`'s flux2dev fp8 quantization path (`torchao`'s `Float8WeightOnlyConfig`) was written assuming compute capability >= 8.9 (RTX 4090/6000 Ada, Hopper). **The RTX A6000 is Ampere, compute capability 8.6** — a different, older card than the similarly-named RTX 6000 Ada, despite the name overlap. This is untested territory: torchao may fall back to slower emulated fp8, or may error outright. Recommendation: run flux2dev's Stage-0.5 smoke test (4 images, `manifests/smoke.json`) first and treat a crash or clearly broken output as "drop flux2dev from this round," not a bug to debug under budget pressure.
2. **Who provisions the pod/volume.** RunPod MCP tools are available this session and could create the volume + pod directly, or this can follow `POD_RUNBOOK.md`'s existing manual-deploy-then-SSH-in workflow. Not decided yet — see chat.
3. **Network volume storage bills continuously, independent of pod runtime.** Stopping the pod after this test is not enough to stop all charges — the 250GB volume itself keeps billing until it's deleted or shrunk. Budget for this explicitly when deciding how long to leave it provisioned.
4. **diffusers must move off the pinned 0.39.0.** Z-Image-Turbo's `ZImagePipeline` needs unreleased diffusers PRs (#12703, #12715); Qwen-Image's own model card also recommends installing from git source. `requirements.txt` now points at `diffusers @ git+https://github.com/huggingface/diffusers`. **This is a real risk to klein/flux2dev**, which were only ever verified against 0.39.0 — `scripts/pod_preflight.py` now checks all 5 pipeline classes import cleanly; do not skip it after this install.

---

## Models

Six confirmed for this round. All route through the same `src/prompts.py::build_prompt()` — no per-model prompt structure changes made (see Q5/Q4 discussion in chat for why not, yet).

| key | repo | steps | guidance | negative_prompt | approx VRAM | status |
|---|---|---|---|---|---|---|
| `klein` | `black-forest-labs/FLUX.2-klein-base-9B` | 50 | 4.0 | No | ~29GB | incumbent baseline, confirmed for inclusion |
| `flux2dev` | `black-forest-labs/FLUX.2-dev` | 50 | 4.0 | No (verified via signature) | ~32GB, fp8-quantized | **compute-capability risk on Ampere cards, see GPU section** |
| `sd35` | `stabilityai/stable-diffusion-3.5-large` | 28 | 4.5 | Expected Yes, **not pod-verified** | ~16GB, bf16 | new integration |
| `qwen_image` | `Qwen/Qwen-Image` (base, undistilled) | 50 | 4.0 (`true_cfg_scale`) | Yes (confirmed on model card) | ~40GB est., **unverified fit** | new integration |
| `qwen_image_lightning` | `Qwen/Qwen-Image` + `lightx2v/Qwen-Image-Lightning` LoRA | 8 | 1.0 (`true_cfg_scale`) | Yes | ~40GB est. (same base model) | new integration |
| `z_image_turbo` | `Tongyi-MAI/Z-Image-Turbo` | 9 | 0.0 (turbo — no CFG) | No | ~13GB, bf16 | new integration, lowest risk |

**Both Qwen-Image variants are in this round** — base (50-step, full quality) and Lightning (8-step, distilled) — now that VRAM headroom (pending GPU choice) makes a head-to-head between them affordable, rather than picking one blind for budget reasons.

**HunyuanImage-3.0 is NOT in this round — deferred, not just to a bigger card.** Two separate reasons, not one: (1) it has no integration code in `generate.py` at all yet — unlike the other five, its actual call signature/API surface has never been checked (it may not even be a standard `diffusers` pipeline, given it's autoregressive/native-multimodal rather than diffusion), so it needs real integration work first, not just a config entry. (2) It's an 80B-total MoE model (64 experts, ~13B active per token) — MoE inference needs *all* experts resident in memory regardless of how few are active per token, so even at aggressive int8 that's ~80GB, and at bf16 roughly 160GB — **not confidently guaranteed to fit even on a 96GB RTX PRO 6000** without quantization this pipeline has no existing support for. This isn't a GPU-size problem alone; it's an unverified-integration problem on top of a genuine capacity question. Recommend keeping the original plan (defer until both are resolved) rather than adding it to this round's scope.

---

## Manifest

`manifests/2-pilot.json` — 50 rows, built by `src/build_manifest.py --balanced 50`, **shared unmodified across every model** so the comparison is apples-to-apples (same 50 prompts, same seeds, same requested class distribution per model).

Deliberately different from `generate_dataset_prompts()`'s independent-random approach: rows cycle through all 7 non-empty class combinations repeatedly (rotating density/framing each cycle), which **guarantees requested-side balance by construction** rather than leaving it to chance:

| class | requested images | requested instances |
|---|---|---|
| starfish | 29 | 84 |
| sea_urchin | 28 | 80 |
| scallop | 28 | 79 |

This is intentional: Question 1 ("is the prompt able to produce a balanced dataset") is only a clean question if the input side is already balanced — cycling the combinations removes manifest randomness as a confound, so any imbalance measured downstream is attributable to generation/detection compliance, not to this particular random draw happening to favor one class.

Verified: the bivalve-guard fix holds across all 50 rows — 0 mismatches between "scallop requested" and "bivalve clause present" (checked programmatically, not just spot-checked).

---

## Execution sequence

1. `python scripts/pod_preflight.py` — must pass clean, including the new diffusers-from-git import checks, before downloading anything.
2. **Per model, smoke test first**: `python src/generate.py --model <key> --manifest manifests/smoke.json` (4 images, ~minutes). Gate: recognizable, photographic, correct class sense, no crash. This is doubly important this round — four of five models are brand-new integrations with unverified `negative_prompt`/VRAM assumptions (see table above).
3. **Per model, full run**: `python src/generate.py --model <key> --manifest manifests/2-pilot.json --out outputs/2-pilot/<key>`.
4. **Per model, annotate**: `python src/annotate.py --images-dir outputs/2-pilot/<key> --report reports/2pilot_class_counts_<key>.json` — separate report file per model (not the shared `reports/class_counts.json`), so results don't overwrite each other.
5. **Aggregate**: compare each model's `reports/2pilot_class_counts_<key>.json` against `manifests/2-pilot.json`'s requested counts, same requested-vs-measured methodology that found the bivalve-guard bug — answers Q1 and Q6.
6. **Manual quality pass** per model against Checkpoint-1.5-style criteria (compositional obedience, sense/anatomical correctness, colour neutrality, scale/camouflage realism) — answers Q3.
7. **Timing log** per model (seconds/image, total wall-clock, pod cost at $0.33/hr) — extrapolate to 1000 images and compare against expected H100 throughput to answer Q2. H100 is materially faster than an A6000 per step, but the honest answer here will be an estimate, not a measurement — this pod can't measure H100 throughput directly.

## Q4 — rock formation pool

Not included in `manifests/2-pilot.json` as generated (that manifest is already verified against the bivalve-fix check above; changing its prompts now would mean testing two changes at once). Proposal for a `ROCK_FORMATIONS` list in `prompts.py`, same pattern as `SCENE_TEMPLATES`, to be added and folded into the *next* manifest rather than this one — draft pending user input on whether specific formation types (from CIRS/real reference footage) should inform the content, or a first pass drafted independently.

## Q5 — automation

Answered in full in chat (this doc references it rather than duplicating): everything that scales mechanically (pool selection, prompt assembly, per-image metadata/sidecar files) is already automated; everything that determines diversity/balance quality (the pool *content*, manifest target distribution, and the measure-then-reweight balance loop) is still manual or doesn't exist in code yet. This pilot doesn't change that — it's a measurement pass, not an automation pass.

---

## Next stage after this

Domain randomization (Stage 3, Akkaynak-Treibitz) — per user, this follows directly once a model is chosen from this comparison.
