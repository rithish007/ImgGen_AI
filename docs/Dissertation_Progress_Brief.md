# Sim-to-Real Transfer for Underwater Object Detection via Domain Randomization
## Project History & Simulator Decision Log — Presentation Reference Brief

**Rithish Ramamoorthy Sathya** | Reg. 250201079 | ELE416, MSc Robotics, University of Sheffield
**Supervisor:** Jonathan Aitken | **Personal completion target:** 24 Aug | **Oral presentation:** 26 Aug | **Formal submission:** 7 Sept 2026

---

## Summary

- **Research question:** can domain randomization (DR) applied to purely synthetic underwater imagery reduce the sim-to-real performance gap for YOLO-based object detection, without using any real annotated training images?
- **Ten named simulators/frameworks were formally evaluated and ruled out** — mostly for a hard ROS/ROS2 dependency, no domain-randomization tooling, or no annotation/ground-truth export — documented deliberately as comparative-justification evidence for the dissertation's literature review.
- **OceanSim (Isaac Sim)** — the starting point, on the supervisor's recommendation — was abandoned not on capability grounds but on hardware grounds: WSL2 cannot supply the Vulkan ICD Isaac Sim's renderer needs, and HPC/cloud-GPU workarounds were also exhausted.
- **HoloOcean and Unity + Perception** both reached a genuine final shortlist and were spike-tested, but each failed at the last mile — HoloOcean's packaged scene had no per-object semantic tags, and Unity's Perception package turned out to be officially discontinued.
- **Raw Unreal Engine 5.8** was chosen as the final simulator, with a fully custom stencil-ID annotation pipeline built from scratch — the largest engineering lift of any option, taken on only after every "free annotation" shortcut evaporated on inspection.
- The UE5.8 build hit several structural complications (an uneditable purchased blueprint, Foliage-instance bounding-box collapse, a water post-process that doesn't reach capture cameras) — all worked around, not fatal.
- A first pipeline-verification test (raw sim vs. a lightly domain-randomized variant, evaluated on 1,955 real CIRS frames) proved the mechanics work end-to-end but returned weak results, prompting a serious proposal to abandon UE5.8 for a fully AI-generative pipeline.
- **Resolution: a dual-pipeline strategy, not a replacement.** UE5.8 remains the primary track (its three concrete weaknesses are being fixed, not the whole engine swapped out), while a fully-scoped five-stage AI-generative pipeline has been built in parallel as a genuine contingency, sharing the same DUO-matched four-class taxonomy and the same evaluation framework.

---

## 1. Research question and scope

Train a detection model **exclusively on synthetic imagery** under a cumulative sequence of DR conditions, then measure performance on public real-world underwater benchmarks — with no real images used for training at any point.

- **Detector:** YOLO (Ultralytics), trained and evaluated in PyCharm on Windows
- **Pipeline constraint:** entirely image-based; no ROS / ROS2 dependency anywhere, by deliberate design
- **Evaluation datasets:** **DUO** (primary, 4-class), **RUOD** (secondary, 10-class), **Brackish** (out-of-distribution stress test)
- **Ablation design:** cumulative B0→B-full structure modelled on Alghonaim & Johns (ICRA 2021), minimum 3 seeds per condition
- **Why it's a genuine gap:** the literature bundles DR with domain adaptation, image enhancement, or novel architectures, obscuring which DR parameters actually matter underwater. Isolating clean DR from those confounds is the dissertation's core contribution — a framing that later became central to rejecting a full pivot to AI-generated data (Section 10).

---

## 2. Timeline at a glance

| Period (2026) | Milestone |
|---|---|
| Early June | OceanSim (Isaac Sim + Omniverse) selected on the supervisor's recommendation; Stonefish evaluated alongside it and dropped in favour of OceanSim alone |
| Early July | OceanSim/Isaac Sim installation fails across three separate architectures; HPC (Iceberg/Stanage) and cloud-GPU (NVIDIA Brev) workarounds pursued and also abandoned |
| 3–5 July | Full simulator re-evaluation from zero: BlenderProc2 floated and dropped; Unity + Perception, Stonefish (revisited), and HoloOcean compared head-to-head |
| 3–4 July | HoloOcean spike-tested on Windows — installs and runs cleanly, but its packaged scene turns out to have no per-object semantic tags |
| 4–5 July | Ten further candidate simulators individually fetched, analysed, and ruled out — built explicitly as comparative-justification evidence for the literature review |
| 5–6 July | Class taxonomy narrowed via a Track A / Track B split; starfish identified as the only class common to all three evaluation datasets |
| 6 July | Progress Review Form submitted: simulator shortlist stated as UE5.8 (raw) / Unity + Perception / UE5.8 + HoloOcean |
| 17–19 July | UE5.8 scene build-out in earnest: Foliage/stencil annotation problems discovered and designed around |
| 19 July | Unity's Perception package confirmed **officially discontinued** — removes Unity from contention for good |
| 26 July | First pipeline-verification test reviewed; full pivot to an AI-generative pipeline proposed and debated |
| 27 July | Dual-pipeline strategy adopted; five-stage AI pipeline fully specified and pilot batch queued on RunPod |

---

## 3. Simulator evaluation — the full rejection list

Ten named simulators/frameworks were individually investigated (via their repos, docs, or source papers) and ruled out. This was done deliberately in parallel with pipeline decision-making, to build a documented, comparative "why not X" justification for the dissertation's literature review.

| Simulator | What it actually is | Why it was ruled out |
|---|---|---|
| **OceanSim (Isaac Sim + Omniverse)** | Physics-based Akkaynak–Treibitz water model, Omniverse Replicator auto-annotation, GPU sonar — supervisor-recommended starting point | **Hardware/access, not capability.** WSL2's NVIDIA driver stack ships no native Vulkan ICD (`libGLX_nvidia.so.0`) for Isaac Sim's RTX renderer — confirmed across native WSL2, Docker headless streaming, and Docker+X11 attempts, plus a native Windows install that crashed on launch. HPC (Iceberg/Stanage) and cloud GPU (NVIDIA Brev) were then also pursued and abandoned. Would have been the *strongest* tool on annotation and physical fidelity — excluded purely on access grounds |
| **Stonefish (v1.5/1.6)** | Native wavelength-resolved Jerlov water types, GPU-accelerated ML annotation, ICRA-2025 scene generator (auto-populates coral/fish) | Capture is fundamentally **ROS-coupled**: output comes via rosbag recording from a ROS launch file, not a scriptable batch export. Built for high-fidelity single-vehicle real-time simulation, not bulk dataset generation; not updated for ROS 2. Directly conflicts with the project's no-ROS constraint |
| **URSim** | ROS + Unity3D underwater vehicle-control framework (SRM AUV Software, 2019) | Vehicle control/mission-planning focus, not dataset generation; no DR tooling; no annotation/ground-truth export; ROS-dependent; thin single-team maintenance since 2019 |
| **UWRoboticsSimulator** | Unity3D–ROS simulator via rosbridge websocket (NUS ARL, OCEANS 2021) | Hard rosbridge dependency; vehicle-dynamics/thruster-control focus; no DR hooks (lighting, turbidity, Jerlov, distractors); no bounding-box/mask export; low activity since ~2021 |
| **DAVE** | ROS Noetic + Gazebo 11 manipulation/mission simulator | No built-in annotation or detection tooling; underwater degradation effects (turbidity, backscatter, caustics) exist only as **roadmap proposals**, not shipped features; object catalogue (torpedoes, sonobuoys, glass spheres) has zero class overlap with the project |
| **UNav-Sim** | UE5-based navigation/SLAM research platform (ORB-SLAM3, TartanVO, PPO, MPC; Amer et al. 2023) | Built for vision-based navigation, not detection — no detection algorithms, no annotation pipeline, no reef-like scenes. Despite superior UE5 rendering, offers no annotation advantage over HoloOcean and needs a from-scratch scene plus a partially untested Windows build (known rpclib version-mismatch issue) |
| **SMaRCSim / smarc2** | KTH's full ROS 2 Humble multi-vehicle robotics stack (mission planning, C2, behaviour trees) wrapping a Unity sim as a submodule | Not a lightweight image generator — a full multi-vehicle middleware stack; ROS 2 baked in throughout; no off-the-shelf DR tooling; low adoption. Its comparison paper's simulator table was kept as a citable cross-simulator reference |
| **MARUS** | Unity3D marine-robotics simulator (LABUST, Univ. of Zagreb; OCEANS 2022) | Core strength is sensor/robot-side simulation, not photorealistic rendering; underwater rendering is basic HDRP fog only, no DR framework; no coral/monopile/biofouling asset content (a robotics framework, not an asset library); alpha-stage documentation |
| **SEANav-Sim** | Unity HDRP sonar-inspection simulator (Lab-STICC / DOLFIA project, OCEANS 2025) | Wrong sensor modality (side-scan sonar vs. optical RGB); wrong task (3-class cable/pipeline segmentation vs. biological/structural detection); no DR support; ROS 2 architecturally central (Unity↔ROS2↔Jetson Orin hardware-in-the-loop); commercial licensing ties |
| **UUV Simulator** (Manhães et al., 2016) | Original Gazebo underwater-robotics extension (SWARMs project); Fossen hydrodynamics, DVL/IMU/multibeam sensors | Strong on vehicle dynamics and ROS multi-robot control, but visually primitive — a 2016-era exponential fog/colour-attenuation camera model only, no DR capability. Simulates sonar bathymetry *output*; does not consume real bathymetry to build scenes. Kept as citable prior art for a simple underwater colour-attenuation equation |

**Also briefly considered and dropped without a full write-up:** BlenderProc2 (Python-native, Cycles volumetric rendering, automatic YOLO export) was floated early as an alternative to Isaac Sim, then dropped for lacking academic precedent in this specific application.

---

## 4. The two genuine finalists that also fell through

Unlike the ten above — which were fetched, assessed, and ruled out fairly quickly — **HoloOcean** and **Unity + Perception** were taken seriously as real candidates, spike-tested, and only dropped very late.

### 4.1 HoloOcean (Unreal Engine 5, BYU Field Robotic Systems Lab)

What made it attractive: confirmed **native Windows install** (no WSL2/Vulkan wall at all), a semantic-segmentation camera sensor for automatic ground-truth, UE5 rendering quality, ROS as *optional* rather than mandatory, and solid academic citability (ICRA 2022, IROS 2022).

A full spike test on Windows/RTX 4060 confirmed the installation worked end-to-end: the Python↔Unreal bridge functioned, and the semantic camera sensor produced usable output. **The disqualifying finding came at the last step**: the packaged `OpenWater` scene ships with only **two** semantic classes — terrain vs. everything else. There are no per-object tags for the classes that actually matter (marine life, debris, structures). Getting real per-object labels would require tagging every mesh by hand inside the Unreal Editor anyway — at which point the effort is comparable to building the annotation pipeline directly in raw UE5.8, without inheriting any of HoloOcean's shortcut.

### 4.2 Unity 3D + Perception package

What made it attractive: the **Perception package** ships automatic ground-truth labelling (bounding boxes, semantic/instance segmentation) generated directly from scene metadata, plus a built-in **Randomizer/Scenario framework** purpose-built for exactly the kind of B0–B5 DR ablation this project needed — no manual annotation step at all, and no ROS dependency.

The disqualifying finding: **Unity's Perception package is officially discontinued and no longer maintained.** This surfaced only when UE5.8 build-out was already underway, and it substantially changed Unity's risk profile relative to continuing the already-progressing custom UE5.8 annotation pipeline. (Unity's underwater rendering was, in any case, a hand-built shader approximation rather than anything physically grounded — the same gap UE5.8 has, just without Unity's annotation advantage once that advantage disappeared.)

---

## 5. Why raw Unreal Engine 5.8 won

Once HoloOcean's "free annotation" turned out to require per-mesh Editor work anyway, and Unity's Perception package turned out to be a dead project, **neither shortlisted alternative retained any real advantage over building directly in Unreal**. Raw UE5.8 offered the highest rendering ceiling (Lumen/Nanite), native Windows support, no ROS, and — critically — a fully custom, fully-controlled annotation pipeline with no third-party tooling risk. The trade-off accepted knowingly: **no first-party annotation/ground-truth tooling at all** — everything (stencil IDs, capture rig, bounding-box extraction) had to be built from scratch.

---

## 6. Class taxonomy: how the target classes evolved

1. **May 2026 (Aims & Objectives):** generic placeholder examples — "barnacle-deposited monopiles, clutter, coral reef" — not yet finalised.
2. **June 2026:** an offshore-wind-inspection framing was tried first — monopile + cracks/barnacles/corrosive deposits. This hit a hard methodological wall: **no public annotated real-world dataset exists for offshore-structure defects**, making quantitative sim-to-real evaluation against real inspection imagery impossible.
3. **Resolution — Track A / Track B split:** **Track A** = quantitative, synthetic reef-like classes overlapping DUO/RUOD/Brackish (fish, coral, urchins) for a real transfer-gap measurement; **Track B** = the monopile/defect scenes kept as a *qualitative* proof-of-concept only, with the missing real-world dataset explicitly framed as a citable research gap rather than a project failure.
4. **5–6 July:** Track A refined by cross-referencing all three evaluation datasets' actual class lists (DUO: holothurian/echinus/scallop/starfish; RUOD: 10 classes including fish/coral/starfish/echinus/holothurian/scallop; Brackish: fish/crab/jellyfish/shrimp/starfish). **Starfish** was identified as the only class common to all three, and adopted as the fourth synthetic class in place of a proposed "debris" class, which had no annotated counterpart anywhere. Working set: coral, urchin, fish (single + swarm), starfish, with rocks as unlabelled background/hard negatives.
5. **Mid–late July, during UE5.8 scene-building:** the labelled set was practically simplified during construction (coral/rock/fish appears in the build-phase pipeline notes), and at least one further variant (coral/kelp/rock/sponge) was tried and then reverted.
6. **Final, current state:** both the UE5.8 track and the AI-generated track converge on **DUO's exact four-class taxonomy — starfish, echinus, holothurian, scallop** — chosen specifically so scoring is a direct, exact match against DUO's ground truth. Coral, rock, and kelp are confirmed **out** as labelled classes; rock remains only as unlabelled background scatter/hard negatives.
7. **Naming convention locked:** the taxonomic DUO labels (echinus, holothurian) are used only for `class_id` mapping in the dataset. Actual generation prompts and VLM annotation text always use common names ("sea urchin", "sea cucumber", "starfish", "scallop"), since vision-language models are trained on web data where common names dominate and taxonomic terms perform worse.

> **Worth double-checking before the presentation:** the pipeline reference document states the dissertation is now scoped tightly to the Track A detection-transfer question, with "no secondary inspection/structural-defect track" — i.e. Track B (monopile) appears to have been descoped for focus, even though it's referenced elsewhere as part of the current design. Confirm which framing you want to present.

---

## 7. Building the UE5.8 pipeline

- **Scene:** a single continuous 1000 m × 1000 m, 5 m-deep shallow reef environment — not a multi-tile design (an earlier plan for several small 50–100 m tiles was superseded). Seabed relief is hand-sculpted/procedural; no real bathymetry data is used, since geographic accuracy isn't a goal.
- **Underwater rendering:** the visual system (water colour, depth-based attenuation, fog/turbidity, backscatter) was authored manually inside UE5.8 — but only to expose parameters for randomisation, not to be physically accurate. **All Jerlov/Akkaynak-Treibitz physics runs in Python post-processing** on exported RGB + linear-depth frames; UE5.8's job is geometry and label generation only.
- **Assets:** coral generated via Tripo3D and Hunyuan3D, supplemented with 3D-scanned Smithsonian Sketchfab models; rocks and fish sourced from the **Lynkolight** asset pack (Gumroad). An additional Fab asset pack was pending university purchase approval and treated as non-critical bonus content.
- **Population:** coral/rock colonies placed procedurally via UE5.8's **Foliage** and **Spline** tools (spline defines colony contour paths, foliage scatters instances along them with density/rotation/scale variation); fish handled separately as a distractor population, not via foliage.
- **DR parameter set — deliberately kept to three axes:**
  - *Photometric:* light intensity, light colour
  - *Underwater-specific:* turbidity, murkiness/visibility distance, water colour, backscatter
  - *Distractors:* random fish spawn (density/presence)
  - Object pose/scale jitter, camera FOV randomisation, and texture randomisation are explicitly **out of scope**.
- **Conditions:** B0 (clean baseline) → B1 (+ photometric) → B2 (+ underwater-specific) → B3/full DR (+ distractors). Camera viewpoint sampling across the scene happens for *all* conditions, including B0 — it's basic dataset-coverage sampling, not itself a DR ablation variable.
- **Annotation design:** per-actor Custom Depth/Stencil ID assigned by class at spawn time; a dual `SceneCaptureComponent2D` rig captures the lit RGB frame and a stencil-ID mask pass; offline Python/OpenCV connected-component analysis converts masks to bounding boxes; a validation overlay pass spot-checks a sample batch before scaling up.

---

## 8. Complications hit in UE5.8 — and the fixes

| Problem | What went wrong | Fix adopted |
|---|---|---|
| **Lynkolight blueprint can't be edited** | It's a purchased asset; its internal blueprint graph is locked | Foliage/Spline is placement-authoring only — converting instances to individual Static Mesh Actors (needed anyway, see below) breaks the blueprint dependency entirely, after which density and placement are fully Python-controllable without touching the asset internals |
| **No native one-click Foliage→actors conversion** | UE4's right-click "convert" option was removed in UE5; where present, it merges everything into one combined mesh — the opposite of what's needed | Two paths identified: a Python script via `unreal.InstancedFoliageActor` (`get_used_foliage_types`, `get_instance_transforms`, `spawn_actor_from_class`), or a ~£15 third-party "Instance Tool" plugin as a GUI fallback |
| **Foliage stencil IDs are per-component, not per-instance** | Custom Depth Stencil is a property of the shared Hierarchical Instanced Static Mesh Component (HISMC), not of each instance — adjacent same-class instances silently **merge into one bounding box** under connected-component analysis (a correctness bug, not a visible one) | Convert foliage instances to individual Static Mesh Actors before dataset generation, restoring per-actor stencil assignment. Where clusters remain visually merged, accept colony-level bounding boxes — arguably a better match to how DUO/RUOD annotate coral colonies anyway |
| **Movie Render Queue's Object ID pass shares the same limitation** | MRQ's Cryptomatte-based Object ID pass was evaluated as an alternative, but Epic's own bug tracker confirms foliage instances collapse to "default" in Cryptomatte output — same root cause | Not adopted as the primary path; per-instance custom data on HISM was also considered but has a documented history of unreliability across UE engine versions, judged too risky this close to deadline |
| **Water plugin's underwater post-process doesn't reach SceneCapture cameras** | UE5.8's built-in Water-plugin underwater effect is designed for the player camera, not `SceneCaptureComponent2D` | Replaced with an unbound Post Process Volume driven by a custom material reading from a Material Parameter Collection (`MPC_DR`), settable from Blueprint or Python |
| **Visual style mismatch** | Initial look was bright cyan, unlike the evaluation datasets | Recentred on a muted green-murky look, closer to DUO/RUOD/Brackish's actual visual style |

---

## 9. First pipeline-verification test

A small-batch end-to-end test was run mid-to-late July to validate pipeline mechanics before committing to full-scale generation:

- **Method:** rendered the (still Lynkolight-based) UE5.8 scene; applied histogram matching against a single CIRS reference image plus a domain-randomiser script varying water colour, grain, and turbidity
- **Data:** two conditions — raw sim and modified/DR'd — 400 train + 100 validation images each
- **Training/evaluation:** four YOLO variants trained, tested against **1,955 real CIRS frames** (extracted earlier from the CIRS Caves `sparus_camera.bag` using a pure-Python `rosbags`-based extraction tool, avoiding any ROS install)
- **Results:** the unmodified baseline **detected nothing at all**; the DR'd variants detected something but with significant class confusion (rocks classified as brain coral; kelp misidentified due to a colour mismatch between simulated green and real brownish kelp)
- **Framing:** explicitly self-characterised as **pipeline verification, not a scientific/publishable result** — the point was only to prove the mechanics worked end-to-end

This test surfaced three concrete, fixable weaknesses rather than a fundamental failure:
1. Histogram-matching to a single reference image is **domain matching**, not domain randomisation — it produces one consistent style shift, not a genuine distribution of plausible water conditions.
2. The Lynkolight/Foliage annotation-collapse issue (Section 8) was still present in this test run.
3. A lingering misconception that UE5.8's water system itself needed to be Jerlov-accurate — corrected: all of that physics belongs in Python post-processing on exported RGB + depth, with UE5.8 responsible only for geometry and labels.

---

## 10. The idea shift: proposing (and then scoping) an AI-generated pipeline

Given the weak verification-test results plus the accumulated UE5.8 engineering friction, a full pivot was proposed: **replace UE5.8 entirely** with an AI-generative pipeline — image generation via diffusion models, automatic zero-shot annotation via a tool like Grounding DINO, and generative domain randomisation — feeding the same three-condition evaluation framework (B0 / B-full / R0).

**The case made against a full replacement:**
- Simulation's core advantage is **free, exact, pixel-perfect labels** from stencil-ID rendering. Switching to AI-generated images with model-predicted auto-annotation trades this away for noisy labels, reintroducing exactly the annotation-quality problem the simulation route exists to avoid.
- If generated images are additionally run through an image-to-image "make it look more real" step, that becomes **domain adaptation**, not domain randomisation. Conflating the two would make it impossible to cleanly attribute any gap-closing to DR specifically — directly undermining the dissertation's central novelty claim (Section 1), which is precisely that existing literature already bundles DR with adaptation in a way that obscures which DR parameters matter.
- Two specific tools under consideration, **SLURPP** and **UDAN-CLIP**, were clarified to be underwater image *restoration* models (degraded → clear) — the opposite direction from what's needed (clean render → realistic murky photo).
- **NVIDIA Cosmos**-style world models were identified as infrastructure-scale and impractical for an MSc timeline (a single H100 taking roughly 30–45 minutes to generate one 10-second clip), and would still require the same 3D-scene-with-labels step UE5.8 already provides.
- Instead, fixing the three specific weaknesses exposed by the verification test (Section 9) was recommended over switching engines entirely: proper multi-preset, Jerlov-based randomised DR calibrated against published water-type coefficients (Solonenko & Mobley, 2015) rather than single-image histogram matching; completing the Foliage→Static-Mesh-Actor conversion plus programmatic density culling (days of work against an already-built scene, not a rebuild); and dropping the misconception that Jerlov water needs modelling inside Unreal at all.

**Resolution — a dual-pipeline strategy, not a replacement:**

UE5.8 remains the **primary** pipeline, with its known weaknesses being fixed rather than the engine being abandoned. Alongside it, a fully-scoped **AI-driven fallback pipeline** was built out in parallel as a genuine contingency — sharing the same DUO-matched four-class taxonomy and the same B0/B-full/R0 evaluation framework, so either pipeline's output is directly usable and comparable.

**AI pipeline architecture — five stages:**

| Stage | Content |
|---|---|
| **1. Base image generation** | **Flux.2 Dev** and **Stable Diffusion 3.5 Large** run in parallel for comparison; manifest-driven batch loop; pilot batch of 20 images per model (40 total) |
| **2. Zero-shot auto-annotation** | **SAM 3** and **Grounding DINO** run in parallel for comparison; outputs YOLO-format labels |
| **3. Domain randomisation** | **Depth Anything V2** for monocular depth estimation, feeding a **deterministic** Akkaynak-Treibitz physics script (not a generative model — preserves parametric control and bounding-box validity), calibrated against DUO's own image statistics; simulated depth capped at 5 m |
| **4. Dataset assembly** | Ultralytics-standard folder structure (`images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`); resized/cropped to 640×640; split by generation batch/seed, not per-image random |
| **5. YOLO26 training smoke test** | **Deferred / on hold**, pending manual review of Stages 1–4 outputs |

Two further clarifications locked in alongside this build:
- **Grounding DINO's actual role** was repositioned away from being a training-data labeller (which would reintroduce the domain-adaptation conflation problem above) to a **zero-shot inference-time comparison point**, evaluated only against the real held-out test set — an afternoon's work, not a pipeline dependency.
- **Compute:** a RunPod **On-Demand Pod with an RTX 5090**, chosen over Serverless specifically because the pipeline is still in active development and needs interactive SSH iteration; synced via Git through PyCharm's built-in terminal. Hugging Face gated-model licences accepted for Flux.2 Dev, SD3.5 Large, and SAM 3.

---

## 11. Where things stand now

- **AI pipeline:** fully specified through Stage 4; the 40-image pilot batch (20 per generation model) is queued/generated, with **manual review of Stages 1–2 outputs required before proceeding to Stage 3 (domain randomisation) and beyond.** Stage 5 (YOLO26 smoke test) stays on hold until that review is signed off.
- **UE5.8 track:** continues in parallel as the primary pipeline, with a concrete four-week execution plan agreed:
  - **Week 1:** complete the Foliage→Static-Mesh-Actor conversion, programmatic density culling, add the depth-capture rig, and build the DR script with Jerlov-preset sampling
  - **Week 2:** batch-render ~750–1,000 frames, apply DR, prepare the real evaluation data
  - **Week 3:** model training across the B0 (raw sim) / B-full (DR applied) / R0 (real-data trained) conditions
  - **Week 4:** evaluation, per-class AP analysis, and presentation preparation

---

## 12. Immediate next steps

1. Complete manual review of the AI-pipeline pilot batch (Stages 1–2) and decide: preferred generation model, preferred annotation engine/combo, whether to proceed to Stage 3.
2. Finish the UE5.8 Foliage-to-actor conversion and density culling.
3. Add the depth-capture rig and build the Jerlov-preset-sampling DR script, calibrated against Solonenko & Mobley (2015) coefficients.
4. Batch-render the full ~750–1,000 frame dataset and apply DR.
5. Train and evaluate across B0 / B-full / R0, with per-class AP breakdown and both transfer-gap metric variants.
6. Resolve the Track A/B scope question (Section 6) ahead of finalising the presentation narrative.

---

*This brief reflects the project history as documented through late July 2026. If anything has moved since — pilot-batch review outcome, UE5.8 progress, or a scope decision — slot the update in before presenting.*
