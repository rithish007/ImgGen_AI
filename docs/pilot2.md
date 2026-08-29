# PILOT 2
Let me set up a thorough quantitative pass first, then sample images for the visual checks (orientation, occlusion, artefacts, boulder/algae). Structure check is done — everything's actually complete and correctly organized (50/50 across images, JSON sidecars, labels, and viz for both models, no missing rows, no stray/duplicate files). Whatever felt off was probably just the nested folder layout itself, not actual corruption.

Good dataset for analysis. Let me add occlusion overlap stats, then move to visual sampling.

Now the visual checks. Let's look at the boulder/algae question first with a few samples.

Good — that confirms the pattern: near-camera boulders usually get algae, but mid/background ones often don't (visible in `2-pilot_017_klein`), while other samples (`2-pilot_050_flux2dev`) show good algae coverage throughout. The `ALGAE_VARIATIONS` text literally says "sparse patches ... attached to rocks" — patchy coverage is what was asked for, and it's ecologically plausible too (real reef algae distribution is uneven). I'd call this a minor polish item, not a defect — you're not being too picky to notice it, but it's not worth spending effort fixing.

Now checking anatomical artefacts across a larger sample:

Good — the wider sample shows the earlier malformed-limb issue is real but **occasional, not endemic** (roughly 1 clear case out of ~10 starfish closely inspected here, plus the one from the smoke test — call it a low single-digit-to-low-teens percentage, not "every starfish"). One ambiguous case in `2-pilot_033_klein` looks more like two overlapping individuals reading as one blob than true malformed anatomy. Also confirms the wide+sparse scallop overshoot is a genuine **generation** problem, not a detection false-positive problem — the tiny background scallops in `2-pilot_042_klein` really are small rendered scallop shapes, not misclassified gravel.

Now let me check Stage 3 (water-condition diversity) status and then move to the cost/scaling and Hunyuan questions.

Stage 3 (`domain_randomize.py`) hasn't been run on the 2-pilot data at all yet — Stage 1 deliberately renders water-neutral by design, and water-condition diversity is entirely Stage 3's job. But it's pure PIL/numpy, no GPU needed — let me actually run it locally so I can give real data instead of "not applicable."

You actually have a local GPU (RTX 4060 Laptop, 8GB) — Depth-Anything-V2-Large should fit. Let me try running Stage 3 locally instead of needing the pod.

Worked great, fast too (~0.4s/image locally). Doing flux2dev's, then running the actual domain-randomize physics transform on both.

Now let's tabulate water-type diversity and view a couple of samples.

Excellent — physically convincing haze/backscatter, confirms Stage 3 works well. Now researching HunyuanImage-3.0's current hardware requirements.

All 11 checks, done entirely from already-pulled data plus your local GPU (no additional RunPod cost this pass). Folder structure and boulder/algae questions are answered above — nothing wrong with the structure (50/50 complete on every artifact type for both models), and the algae patchiness is prompt-consistent and ecologically plausible, not a defect.

## 1-2. Class diversity & balance

Manifest is balanced by construction (`build_manifest.py --balanced`): 29/28/28 images touch starfish/urchin/scallop respectively, out of 50. Detected-vs-requested (SAM3 counts vs. manifest):

| class | requested (inst) | klein detected | flux2dev detected |
|---|---|---|---|
| starfish | 84 | 93 (1.11x) | 94 (1.12x) |
| sea urchin | 80 | 102 (1.27x) | 94 (1.18x) |
| scallop | 79 | 139 (**1.76x**) | 113 (1.43x) |

**flux2dev has meaningfully better count discipline on every class**, most clearly on scallop — this is the same axis the wide+sparse bug already explained: almost all of the excess concentrates in `framing=wide`+`density=sparse` rows for both models, flux2dev just overshoots less badly there too.

## 3. Object size distribution

| class | klein median area | flux2dev median area | klein "tiny(<0.5%)" | flux2dev "tiny" |
|---|---|---|---|---|
| starfish | 2.37% | 2.20% | 12/93 | 10/94 |
| sea urchin | 1.20% | 1.54% | 15/102 | 1/94 |
| scallop | 0.52% | 1.13% | **69/139** | 30/113 |

Scallops are systematically the smallest class in both models, and klein's scallop set is half tiny detections. I visually verified a sample of these (the `2-pilot_042_klein` wide-shot background scallops) — they're genuine small rendered shells, not SAM3 misfiring on gravel. So this is real: a lot of "scallop" instances are barely-there background specks, largely inflated by the same wide+sparse bug rather than reflecting genuine close-range size diversity.

## 4. Object orientation

YOLO boxes are axis-aligned, so there's no true orientation field to check — bbox aspect ratio (long/short side) is the closest proxy:

- sea urchin: median ~1.15-1.21 (near-circular, as expected for a radially symmetric animal)
- scallop: median ~1.34-1.45 (fan shape viewed at an angle)
- starfish: median ~1.6-1.65, max up to 5.63 (elongated — either genuine angled 5-arm silhouettes, or occasionally two overlapping individuals read as one box, per the ambiguous case in `2-pilot_033_klein`)

Visually, camera angle/height variety (low/medium/high x close-up/mid/wide) is doing real work — objects show up flat-on, angled, and partially side-on across the sample. No evidence of the "every object faces the camera" staged-photo problem.

## 5. Occlusion

~40% of images in both models show at least one overlapping box pair (klein 20/50, flux2dev 18/50, IoU>0.05) — consistent between models since it's the same manifest driving `moderate`/`hard` difficulty settings. Visually confirmed genuine occlusion patterns: starfish tucked partly under rocks, urchins overlapping in crevices, scallops partially behind boulders. This looks healthy, not manufactured.

## 6. Background diversity

Scene template/algae/substrate/lighting/composition/camera-fov/motion/imaging indices are all reasonably well-spread across their pools (no index dominating >~35% of the 50 images) — see the raw distributions if you want them, but nothing is clumped enough to worry about. `rock_formation_index` isn't in these sidecars since that field postdates this run.

## 7. Water-condition diversity

This was **zero in the 2-pilot data as generated** — Stage 1 always renders clear/neutral by design (`SCENE_WATER_PHRASE`), water condition is entirely Stage 3's job, and Stage 3 had never been run on this set. Rather than just tell you that, I ran it: Stage 3a (depth) runs fine on your laptop's RTX 4060 (0.4s/image, no RunPod needed), and Stage 3b then produced real diversity —

```
water types: {5C(murky): 25, 3C(moderate): 15, 1C(clear): 10}
depth: 0.07m - 4.79m (median 2.39m)
visibility_floor: 0.09 - 0.44 (median 0.25)
```

Both models identical (same seeds). I generated all 100 DR'd images — [outputs/2-pilot/dr/](outputs/2-pilot/dr) — visually confirmed convincing haze/backscatter/depth-darkening. **Useful side effect: Stage 3 costs you nothing on RunPod or Stanage — it runs on your own machine.**

## 8. AI anatomical artefacts

Sampled ~15 starfish across both models (beyond the original smoke test): the malformed/truncated-limb issue from the smoke test is **real but occasional**, not endemic — roughly 1-in-10-ish in this sample, not "every starfish." Most instances (`2-pilot_046_klein`, `2-pilot_046_flux2dev`, several others) show clean 5-arm anatomy. One case reads more like two overlapping individuals fused into one box than a true anatomical defect. Scallop/urchin anatomy looked consistently fine across every sample I checked — the artefact risk is specifically a starfish (radial-symmetry) issue.

## 9. Scaling to 1000 images — cost

Measured rates: klein 27.1s/image, flux2dev 67.0s/image (both include per-image save overhead, exclude one-time model load).

| | generation | + annotation (~est.) | total time | cost @ $2.09/hr |
|---|---|---|---|---|
| klein, 1000 img | 7.5 hr | ~45 min | ~8.3 hr | **~$17** |
| flux2dev, 1000 img | 18.6 hr | ~45 min | ~19.3 hr | **~$40** |

(Annotation estimate is rough — I don't have a clean per-image annotation-only timing log, extrapolated from how fast the 50-image passes felt.) Stage 3 (domain randomization) adds ~$0 since it runs locally. **Yes, this scales fine technically** — nothing in tonight's run suggests the pipeline itself breaks at 1000 images, just linear time/cost. The real risk at that scale is the same host-capacity/pod-migration friction we hit twice tonight, which is exactly why moving to Stanage for this run (no per-hour billing, no host-pinning) makes sense.

## 10. Which model to stick with

**flux2dev**, if the extra time/cost is acceptable — it wins on the one axis this whole pilot was built to measure (class-count compliance), most clearly on scallop, and it's the undistilled base model klein was themselves distilled from, so this isn't a surprising result. Caveat: this recommendation is driven by the quantitative compliance data, not a rigorous blind visual-quality study — I haven't done a systematic "which looks more real" pass beyond spot samples, which looked comparable between the two. Given the move to Stanage removes the per-hour cost pressure that made klein's speed attractive, I'd lean flux2dev for the production run unless Stanage queue time makes the 2.5x slower generation a real scheduling problem.

## 11. HunyuanImage-3.0 — GPU choice and batch-split question

Researched current state (this has moved since the plan doc's original note):

- **~83B params.** FP16 needs ~181GB VRAM (infeasible on a single card). **INT8 quantized needs only ~18-20GB** — a big change from the plan doc's original "80-160GB" worry, this now fits comfortably even on a mid-tier GPU. NF4/INT4 gets to ~45GB.
- **Still not a diffusers pipeline** — it's autoregressive/native-multimodal, loaded via `transformers.AutoModelForCausalLM`, confirmed via an open (unresolved) diffusers GitHub issue requesting official pipeline support. This means it needs genuinely different integration code in this project, not a `generate.py` `MODELS` dict entry — the plan doc's "needs real integration work first" caution is still accurate.
- Pre-quantized community checkpoints exist (INT8/NF4, by a third-party HF user "EricRollei") — usable, but verify license/integrity before trusting them for dissertation data; they're not the official Tencent release.

**GPU**: given INT8's ~20GB footprint, you don't need the RTX PRO 6000 96GB for this at all — and since Stanage's A100/H100 nodes (80GB) would run it trivially, I'd test Hunyuan on Stanage once you're set up there rather than spend more RunPod credits on it.

**Batch-splitting the 1000-image run three ways: no, not yet.** Hunyuan hasn't produced a single verified underwater-domain image yet, and it needs real integration code before it can. Splitting the production budget 1/3 to an untested model risks wasting a third of it if it turns out to not handle this domain well — autoregressive image models can fail in totally different ways than diffusion models, and we have zero evidence yet either way. Recommend the same sequence that worked for klein/flux2dev: get Hunyuan integrated → smoke-test (3 images) → small pilot (~20-50 images) → only then decide whether it earns a share of the 1000-image production run alongside or instead of flux2dev.

Sources:
- [HunyuanImage-3.0 VRAM Requirements & Cheapest GPU | Spheron](https://www.spheron.network/tools/gpu-recommender/tencent/HunyuanImage-3.0)
- [HunyuanImage 3.0 VRAM Requirements | WillItRunAI](https://willitrunai.com/image-models/hunyuan-image-3)
- [HunyuanImage-3.0 support · Issue #12412 · huggingface/diffusers](https://github.com/huggingface/diffusers/issues/12412)
- [tencent/HunyuanImage-3.0 · Hugging Face](https://huggingface.co/tencent/HunyuanImage-3.0)