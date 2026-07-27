# AI-Pipeline Smoke Test — Technical Build Plan (v3)

**Purpose:** validate an all-AI alternative to the UE5.8 pipeline (generation -> annotation -> DR -> YOLO26) on a small pilot batch before committing to full-scale dataset generation. This is a pipeline-mechanics check, not a scientific result — sample size is deliberately tiny at this stage.

**Compute:** RunPod Pod, **RTX 5090 (32GB VRAM)**, On-Demand billing, Community Cloud to start. Sync via Git — write/edit locally in PyCharm, `git push`, SSH into the pod (PyCharm's built-in terminal), `git pull`, run remotely, pull results back. Use a RunPod network volume so datasets/checkpoints persist across sessions.

**Hugging Face setup — DONE:** license agreements accepted for `black-forest-labs/FLUX.2-dev`, `stabilityai/stable-diffusion-3.5-large`, and `facebook/sam3`. Confirm `HF_TOKEN` is exported in the pod environment before running download scripts.

---

## Stage 1 — Base image generation (dual-model comparison)

Run the **same manifest** through both models:
- **Flux.2 Dev** (open-weight, seed-reproducible)
- **Stable Diffusion 3.5 Large** (Stability AI's current flagship, MMDiT architecture, meaningfully better prompt adherence than SDXL)

**Pilot batch size:** 20 images per model (40 total) — confirmed

**Class vocabulary (matches DUO exactly):**

| class_id | DUO label | prompt text to use |
|---|---|---|
| 0 | starfish | starfish |
| 1 | echinus | sea urchin |
| 2 | holothurian | sea cucumber |
| 3 | scallop | scallop |

Use the **common-name prompt text** in both Stage 1 generation prompts and Stage 2 annotation text prompts — the taxonomic DUO labels (echinus, holothurian) are for the dataset's class_id mapping only, not for what gets typed into the generator or detector. Vision-language models are trained on web image-caption data where "sea urchin"/"sea cucumber" are common and "echinus"/"holothurian" are rare, so prompting with the taxonomic term risks materially worse generation and detection quality for the exact same object.

**Density:** sparse (2-3 instances) to moderate (4-6 instances) per image
**Framing:** close-up to mid, ROV forward-facing — prompt cues: wide-angle lens, slight fisheye distortion
**Scene descriptor:** greenish coastal waters, shallow (contextual cue only — max depth 5m is enforced properly in Stage 3, not baked into pixels here)
**Negative prompts:** no text, no watermark, no diver, no boat, no human, no surface/sky
**Resolution:** native resolution per model, resize/crop to 640x640 at Stage 4, not here

**Manifest generation** (script-built, no manual per-image prompting):

4 classes = 15 possible non-empty combinations. At n=20/model, cover every combination exactly once, then use the remaining rows to reinforce clean single-class exemplars (useful for the Stage 2 spot-check):

| Row | Classes | Density | Framing |
|---|---|---|---|
| 1-4 | each class alone | sparse | close-up |
| 5-10 | all 6 pairs | moderate | mid |
| 11-14 | all 4 triples | moderate | close-up |
| 15 | all four together | moderate | mid |
| 16-19 | each class alone (repeat, new seed) | moderate | mid |
| 20 | all four together (repeat, new seed) | sparse | close-up |

Each row: class subset, density, framing, unique seed. Output per image: PNG + JSON sidecar {prompt, seed, model, slot_values, image_id}.

### Checkpoint 1.5 - generation comparison
Spot-check all 40 images (small enough to review in full at this scale):
- Compositional obedience - did it place the requested classes?
- Visual realism, artifact presence
- Decide: pick one model for the next scale-up, or keep both as parallel tracks

---

## Stage 2 — Auto-annotation (SAM 3 vs Grounding DINO)

- **SAM 3** and **Grounding DINO**, same 40-image set, run in parallel for comparison
- **Text prompts:** "starfish", "sea urchin", "sea cucumber", "scallop" (common-name phrasing, see Stage 1 table — do not prompt with "echinus"/"holothurian")
- **Output format:** YOLO .txt directly from boxes (or mask-to-box for SAM3), using the DUO-aligned class_id (0=starfish, 1=echinus, 2=holothurian, 3=scallop) regardless of which prompt text triggered the detection
- Fallback if SAM3 access is ever revoked/unavailable: Grounding DINO (text to box) + SAM2 (box to mask) — SAM2 alone cannot take a text prompt, so it must be paired with Grounding DINO to stay concept-driven

### Checkpoint 2.5 - annotation comparison
Review all 40 annotated images:
- Box/mask correctness per class
- Compare SAM3 vs Grounding DINO per-class, not just overall — pick the stronger engine, or a per-class hybrid if there's a clear split

---

## Stage 3 — Domain randomization (depth + physics, not generative)

**3a - Depth estimation:** Depth Anything V2 on each raw image -> relative depth map. Cap the simulated depth range at **5m maximum**, matching your scene's stated bound.

**3b - Physics transform:** deterministic Python script, revised Akkaynak-Treibitz model + Jerlov water-type sampling, using the depth map for range-dependent attenuation plus backscatter/veiling-light term. **Calibrate turbidity/colour coefficient ranges specifically against DUO's image statistics.**

- Output: DR'd copy of each image, same label files as the raw version (pixel-only transform)
- Config JSON per DR'd image: sampled Jerlov type, turbidity/backscatter coefficients, seed

---

## Stage 4 — Dataset assembly

- Resize/center-crop to 640x640 here
- Standard Ultralytics layout: images/train, images/val, labels/train, labels/val, data.yaml
- Split by generation batch/seed, not per-image random
- Pilot scale: up to 40 raw images (both models) + 40 DR'd copies = up to 80 image-label pairs total

---

## Stage 5 — ON HOLD

Paused pending your manual review of Stage 1-4 outputs. Do not start until explicitly resumed.

---

## Pilot success criteria (qualitative, not statistical at n=20/model)

- Both models produce visually plausible, class-diverse images
- At least one annotation engine gets usable boxes on all 4 classes
- DR transform runs without breaking label/box alignment
- Clear enough result to decide: which generation model, which annotation engine

## Deliverables from this pilot back into the main plan

- Chosen generation model (or reason to keep testing both)
- Chosen annotation engine/combo
- Estimated auto-annotation error rate from the full-batch review
- Cost/time-per-image on the RTX 5090, to budget the real full-scale batch
