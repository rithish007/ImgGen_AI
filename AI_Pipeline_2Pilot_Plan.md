# 2-Pilot — Cross-Model Generation Comparison (v2)

**Purpose:** test the bivalve-guard fix (`src/prompts.py`) across multiple image-gen candidates on a shared, class-balanced-by-construction manifest, to decide which model(s) go into the 1000-image HPC run and whether the prompt itself needs further work first. This is a comparison/decision pilot, not a training run — no YOLO training happens on this data.

**Compute:** RunPod, RTX PRO 6000 Blackwell Server Edition (96GB), 250GB network volume, EUR-IS-1 datacenter. Deployed via console (user), operated via SSH agent-forwarded git access from there on.

**The 6 questions this test must answer** (user's list, verbatim):
1. Is the prompt able to generate a class-balanced set of dataset?
2. Can this be used for the massive 1000-image generation using HPC H100?
3. Which image-gen model performs best for quality?
4. Can the rock-formation pool be added to the prompt, and how?
5. Which non-automated processes can be automated without affecting the result?
6. How is SAM3 performing — are detected classes balanced relative to requested counts?

---

## v1 → v2 changelog

1. **GPU decided: RTX PRO 6000 Blackwell (96GB)**, not the originally-discussed RTX A6000. Blackwell (compute capability sm_120) has native fp8 tensor core support, removing the Ampere compute-capability risk v1 flagged for flux2dev.
2. **`qwen_image` (base, 50-step, undistilled) added** alongside `qwen_image_lightning` — the 96GB card makes a head-to-head between them affordable instead of picking one blind for budget reasons.
3. **`z_image_turbo` dropped entirely**, and with it, the reason to leave pinned `diffusers==0.39.0`. Sequence of events: Z-Image-Turbo's `ZImagePipeline` needed diffusers built from git source (unreleased PRs #12703/#12715) → that git build was applied to *all* models, not just Z-Image-Turbo → its `TorchAoConfig` was missing an attribute `transformers==5.14.1` reads unconditionally → **flux2dev's quantization crashed**, a model that never needed the newer diffusers at all. Rather than keep chasing that version-skew bug, Z-Image-Turbo is out and `requirements.txt` is back to the exact pinned `diffusers==0.39.0` this project already verified — which has had Qwen-Image support since 0.35.0, so nothing else in this round loses anything from the revert. `generate.py`'s `_build_quantization_config()` keeps a defensive attribute-set as a harmless no-op in case 0.39.0 has any version of the same gap (untested — flux2dev's quantization path was never actually GPU-verified even in the original Stage 0 setup).
4. HunyuanImage-3.0 remains out of scope — see below, unchanged from v1.

---

## Models

Five confirmed for this round. All route through the same `src/prompts.py::build_prompt()` — no per-model prompt structure changes made (see Q5/Q4 discussion in chat for why not, yet).

| key | repo | steps | guidance | negative_prompt | approx VRAM | status |
|---|---|---|---|---|---|---|
| `klein` | `black-forest-labs/FLUX.2-klein-base-9B` | 50 | 4.0 | No | ~29GB | incumbent baseline — **smoke-tested, pass** |
| `flux2dev` | `black-forest-labs/FLUX.2-dev` | 50 | 4.0 | No (verified via signature) | ~32GB, fp8-quantized | pending re-test after diffusers revert |
| `sd35` | `stabilityai/stable-diffusion-3.5-large` | 28 | 4.5 | Expected Yes, **not pod-verified** | ~16GB, bf16 | new integration, untested |
| `qwen_image` | `Qwen/Qwen-Image` (base, undistilled) | 50 | 4.0 (`true_cfg_scale`) | Yes (confirmed on model card) | ~40GB est., **unverified fit** | new integration, untested |
| `qwen_image_lightning` | `Qwen/Qwen-Image` + `lightx2v/Qwen-Image-Lightning` LoRA | 8 | 1.0 (`true_cfg_scale`) | Yes | ~40GB est. (same base model) | new integration, untested |

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

1. `python scripts/pod_preflight.py` — must pass clean before downloading anything. Passed on first real run (31/31 checks) after removing a stale system `torchaudio` left over from an earlier `pip install --target` transitively upgrading torch and breaking `torchaudio`'s compiled extension's ABI - unrelated to the diffusers version, still worth knowing about if it recurs.
2. **Per model, smoke test first**: `python src/generate.py --model <key> --manifest manifests/smoke.json` (3 images, one per class, ~1-2 min). Gate: recognizable, photographic, correct class sense, no crash. Doubly important this round — four of five models are brand-new integrations with unverified `negative_prompt`/VRAM assumptions (see table above).
3. **Per model, full run**: `python src/generate.py --model <key> --manifest manifests/2-pilot.json --out outputs/2-pilot/<key>`.
4. **Per model, annotate**: `python src/annotate.py --images-dir outputs/2-pilot/<key> --report reports/2pilot_class_counts_<key>.json` — separate report file per model (not the shared `reports/class_counts.json`), so results don't overwrite each other.
5. **Aggregate**: compare each model's `reports/2pilot_class_counts_<key>.json` against `manifests/2-pilot.json`'s requested counts, same requested-vs-measured methodology that found the bivalve-guard bug — answers Q1 and Q6.
6. **Manual quality pass** per model against Checkpoint-1.5-style criteria (compositional obedience, sense/anatomical correctness, colour neutrality, scale/camouflage realism) — answers Q3.
7. **Timing log** per model (seconds/image, total wall-clock, pod cost at ~$2.09/hr) — extrapolate to 1000 images and compare against expected H100 throughput to answer Q2. klein's smoke test measured 27.6s/image at 50 steps on this card (no offload, weights fully resident). H100 vs. this Blackwell card's throughput will still be an estimate, not a measurement — this pod can't measure H100 directly.

## Q4 — rock formation pool

Not included in `manifests/2-pilot.json` as generated (that manifest is already verified against the bivalve-fix check above; changing its prompts now would mean testing two changes at once). Proposal for a `ROCK_FORMATIONS` list in `prompts.py`, same pattern as `SCENE_TEMPLATES`, to be added and folded into the *next* manifest rather than this one — draft pending user input on whether specific formation types (from CIRS/real reference footage) should inform the content, or a first pass drafted independently.

## Q5 — automation

Answered in full in chat (this doc references it rather than duplicating): everything that scales mechanically (pool selection, prompt assembly, per-image metadata/sidecar files) is already automated; everything that determines diversity/balance quality (the pool *content*, manifest target distribution, and the measure-then-reweight balance loop) is still manual or doesn't exist in code yet. This pilot doesn't change that — it's a measurement pass, not an automation pass.

---

## Next stage after this

Domain randomization (Stage 3, Akkaynak-Treibitz) — per user, this follows directly once a model is chosen from this comparison.
