# Path migration map

Generated during the repository reorganisation. Every Python file that moved,
with its new location and how to invoke it. Historical references to old
`src/…` / `scripts/…` paths in the plan documents and in module docstrings were
deliberately left as written — they name files as they were called at the time,
which is part of the decision record. Use this table to resolve them.

All commands are run from the repository root.

| Old path | New path | Invoke as |
|---|---|---|
| `src/generate.py` | `archive/generators/generate.py` | `python -m archive.generators.generate` |
| `src/generate_3pilot.py` | `archive/generators/generate_3pilot.py` | `python -m archive.generators.generate_3pilot` |
| `src/generate_3pilot_promptfix.py` | `archive/generators/generate_3pilot_promptfix.py` | `python -m archive.generators.generate_3pilot_promptfix` |
| `src/generate_flux.py` | `archive/generators/generate_flux.py` | `python -m archive.generators.generate_flux` |
| `src/generate_flux2dev_v3.py` | `archive/generators/generate_flux2dev_v3.py` | `python -m archive.generators.generate_flux2dev_v3` |
| `src/generate_flux2dev_v4.py` | `archive/generators/generate_flux2dev_v4.py` | `python -m archive.generators.generate_flux2dev_v4` |
| `src/generate_flux2dev_v5.py` | `archive/generators/generate_flux2dev_v5.py` | `python -m archive.generators.generate_flux2dev_v5` |
| `src/generate_flux2dev_v5_highcam.py` | `archive/generators/generate_flux2dev_v5_highcam.py` | `python -m archive.generators.generate_flux2dev_v5_highcam` |
| `src/generate_flux2dev_v6.py` | `archive/generators/generate_flux2dev_v6.py` | `python -m archive.generators.generate_flux2dev_v6` |
| `src/generate_flux2dev_v7.py` | `archive/generators/generate_flux2dev_v7.py` | `python -m archive.generators.generate_flux2dev_v7` |
| `src/generate_flux2dev_v8.py` | `archive/generators/generate_flux2dev_v8.py` | `python -m archive.generators.generate_flux2dev_v8` |
| `src/generate_hunyuan.py` | `archive/generators/generate_hunyuan.py` | `python -m archive.generators.generate_hunyuan` |
| `src/generate_hunyuan_v2.py` | `archive/generators/generate_hunyuan_v2.py` | `python -m archive.generators.generate_hunyuan_v2` |
| `src/generate_hunyuan_v3.py` | `archive/generators/generate_hunyuan_v3.py` | `python -m archive.generators.generate_hunyuan_v3` |
| `src/generate_hunyuan_v4.py` | `archive/generators/generate_hunyuan_v4.py` | `python -m archive.generators.generate_hunyuan_v4` |
| `src/generate_hunyuan_v5.py` | `archive/generators/generate_hunyuan_v5.py` | `python -m archive.generators.generate_hunyuan_v5` |
| `src/generate_hunyuan_v6.py` | `archive/generators/generate_hunyuan_v6.py` | `python -m archive.generators.generate_hunyuan_v6` |
| `src/generate_hunyuan_v7.py` | `archive/generators/generate_hunyuan_v7.py` | `python -m archive.generators.generate_hunyuan_v7` |
| `src/generate_klein_promptfix.py` | `archive/generators/generate_klein_promptfix.py` | `python -m archive.generators.generate_klein_promptfix` |
| `scripts/analyze_all_versions.py` | `imggen/analysis/analyze_all_versions.py` | `python -m imggen.analysis.analyze_all_versions` |
| `scripts/build_duo_comparison_samples.py` | `imggen/analysis/build_duo_comparison_samples.py` | `python -m imggen.analysis.build_duo_comparison_samples` |
| `src/class_balance.py` | `imggen/analysis/class_balance.py` | `python -m imggen.analysis.class_balance` |
| `src/depth_compare.py` | `imggen/analysis/depth_compare.py` | `python -m imggen.analysis.depth_compare` |
| `src/depth_utils.py` | `imggen/analysis/depth_utils.py` | `python -m imggen.analysis.depth_utils` |
| `src/montage.py` | `imggen/analysis/montage.py` | `python -m imggen.analysis.montage` |
| `scripts/plot_gt_vs_pred_montage.py` | `imggen/analysis/plot_gt_vs_pred_montage.py` | `python -m imggen.analysis.plot_gt_vs_pred_montage` |
| `scripts/plot_v9_class_balance.py` | `imggen/analysis/plot_v9_class_balance.py` | `python -m imggen.analysis.plot_v9_class_balance` |
| `scripts/plot_v9_dr_duo_montage.py` | `imggen/analysis/plot_v9_dr_duo_montage.py` | `python -m imggen.analysis.plot_v9_dr_duo_montage` |
| `src/range_estimate.py` | `imggen/analysis/range_estimate.py` | `python -m imggen.analysis.range_estimate` |
| `src/range_estimate_depthpro.py` | `imggen/analysis/range_estimate_depthpro.py` | `python -m imggen.analysis.range_estimate_depthpro` |
| `src/visualize_annotations.py` | `imggen/analysis/visualize_annotations.py` | `python -m imggen.analysis.visualize_annotations` |
| `src/water_stats.py` | `imggen/analysis/water_stats.py` | `python -m imggen.analysis.water_stats` |
| `src/annotate.py` | `imggen/data/annotate.py` | `python -m imggen.data.annotate` |
| `src/assemble_dataset.py` | `imggen/data/assemble_dataset.py` | `python -m imggen.data.assemble_dataset` |
| `scripts/assemble_pilot_ablation.py` | `imggen/data/assemble_pilot_ablation.py` | `python -m imggen.data.assemble_pilot_ablation` |
| `scripts/assemble_v9_dataset.py` | `imggen/data/assemble_v9_dataset.py` | `python -m imggen.data.assemble_v9_dataset` |
| `scripts/assemble_v9_duo_dataset.py` | `imggen/data/assemble_v9_duo_dataset.py` | `python -m imggen.data.assemble_v9_duo_dataset` |
| `scripts/assemble_v9_duo_scatter_dataset.py` | `imggen/data/assemble_v9_duo_scatter_dataset.py` | `python -m imggen.data.assemble_v9_duo_scatter_dataset` |
| `src/build_manifest.py` | `imggen/data/build_manifest.py` | `python -m imggen.data.build_manifest` |
| `scripts/build_split_manifest.py` | `imggen/data/build_split_manifest.py` | `python -m imggen.data.build_split_manifest` |
| `src/prepare_real_eval.py` | `imggen/data/prepare_real_eval.py` | `python -m imggen.data.prepare_real_eval` |
| `src/dr_detection_check.py` | `imggen/dr/detection_check.py` | `python -m imggen.dr.detection_check` |
| `src/jerlov.py` | `imggen/dr/jerlov.py` | `python -m imggen.dr.jerlov` |
| `src/jerlov_anchored.py` | `imggen/dr/jerlov_anchored.py` | `python -m imggen.dr.jerlov_anchored` |
| `src/domain_randomize.py` | `imggen/dr/randomize.py` | `python -m imggen.dr.randomize` |
| `src/domain_randomize_v2.py` | `imggen/dr/randomize_v2.py` | `python -m imggen.dr.randomize_v2` |
| `src/eval_pilot_on_duo.py` | `imggen/eval/pilot_on_duo.py` | `python -m imggen.eval.pilot_on_duo` |
| `src/eval_real.py` | `imggen/eval/real.py` | `python -m imggen.eval.real` |
| `src/eval_v9_duo_on_duo.py` | `imggen/eval/v9_duo_on_duo.py` | `python -m imggen.eval.v9_duo_on_duo` |
| `src/eval_v9_duo_scatter_on_duo.py` | `imggen/eval/v9_duo_scatter_on_duo.py` | `python -m imggen.eval.v9_duo_scatter_on_duo` |
| `src/eval_v9_on_duo.py` | `imggen/eval/v9_on_duo.py` | `python -m imggen.eval.v9_on_duo` |
| `src/eval_v9_placeholder_on_duo.py` | `imggen/eval/v9_placeholder_on_duo.py` | `python -m imggen.eval.v9_placeholder_on_duo` |
| `src/generate_flux2dev_v9.py` | `imggen/generate/flux2dev_v9.py` | `python -m imggen.generate.flux2dev_v9` |
| `src/generate_flux2dev_v9_starfish.py` | `imggen/generate/flux2dev_v9_starfish.py` | `python -m imggen.generate.flux2dev_v9_starfish` |
| `src/generate_hunyuan_v8.py` | `imggen/generate/hunyuan_v8.py` | `python -m imggen.generate.hunyuan_v8` |
| `src/prompts.py` | `imggen/prompts/base.py` | `python -m imggen.prompts.base` |
| `src/prompts_flux2dev_v8.py` | `imggen/prompts/flux2dev_v8.py` | `python -m imggen.prompts.flux2dev_v8` |
| `src/prompts_flux2dev_v9_starfish.py` | `imggen/prompts/flux2dev_v9_starfish.py` | `python -m imggen.prompts.flux2dev_v9_starfish` |
| `src/prompts_hunyuan_v7.py` | `imggen/prompts/hunyuan_v7.py` | `python -m imggen.prompts.hunyuan_v7` |
| `src/prompts_flux2dev_v2.py` | `imggen/prompts/legacy/flux2dev_v2.py` | `python -m imggen.prompts.legacy.flux2dev_v2` |
| `src/prompts_flux2dev_v3.py` | `imggen/prompts/legacy/flux2dev_v3.py` | `python -m imggen.prompts.legacy.flux2dev_v3` |
| `src/prompts_flux2dev_v4.py` | `imggen/prompts/legacy/flux2dev_v4.py` | `python -m imggen.prompts.legacy.flux2dev_v4` |
| `src/prompts_flux2dev_v5.py` | `imggen/prompts/legacy/flux2dev_v5.py` | `python -m imggen.prompts.legacy.flux2dev_v5` |
| `src/prompts_flux2dev_v6.py` | `imggen/prompts/legacy/flux2dev_v6.py` | `python -m imggen.prompts.legacy.flux2dev_v6` |
| `src/prompts_flux2dev_v7.py` | `imggen/prompts/legacy/flux2dev_v7.py` | `python -m imggen.prompts.legacy.flux2dev_v7` |
| `src/prompts_hunyuan.py` | `imggen/prompts/legacy/hunyuan.py` | `python -m imggen.prompts.legacy.hunyuan` |
| `src/prompts_hunyuan_v2.py` | `imggen/prompts/legacy/hunyuan_v2.py` | `python -m imggen.prompts.legacy.hunyuan_v2` |
| `src/prompts_hunyuan_v3.py` | `imggen/prompts/legacy/hunyuan_v3.py` | `python -m imggen.prompts.legacy.hunyuan_v3` |
| `src/prompts_hunyuan_v4.py` | `imggen/prompts/legacy/hunyuan_v4.py` | `python -m imggen.prompts.legacy.hunyuan_v4` |
| `src/prompts_hunyuan_v5.py` | `imggen/prompts/legacy/hunyuan_v5.py` | `python -m imggen.prompts.legacy.hunyuan_v5` |
| `src/prompts_hunyuan_v6.py` | `imggen/prompts/legacy/hunyuan_v6.py` | `python -m imggen.prompts.legacy.hunyuan_v6` |
| `src/train_pilot_ablation.py` | `imggen/train/pilot_ablation.py` | `python -m imggen.train.pilot_ablation` |
| `src/train_v9.py` | `imggen/train/v9.py` | `python -m imggen.train.v9` |
| `src/train_v9_duo.py` | `imggen/train/v9_duo.py` | `python -m imggen.train.v9_duo` |
| `src/train_v9_duo_scatter.py` | `imggen/train/v9_duo_scatter.py` | `python -m imggen.train.v9_duo_scatter` |
| `src/train_v9_duo_v2.py` | `imggen/train/v9_duo_v2.py` | `python -m imggen.train.v9_duo_v2` |
| `src/train_v9_placeholder.py` | `imggen/train/v9_placeholder.py` | `python -m imggen.train.v9_placeholder` |
| `src/train_yolo.py` | `imggen/train/yolo.py` | `python -m imggen.train.yolo` |
| `scripts/pod_preflight.py` | `tools/pod_preflight.py` | `python tools/pod_preflight.py` |
| `src/test_models.py` | `tools/test_models.py` | `python tools/test_models.py` |

## Non-Python moves

| Old | New |
|---|---|
| `*.md` (repo root) | `docs/` |
| `HPC_RUNBOOK.md`, `POD_RUNBOOK.md` | `docs/runbooks/` |
| `Literature/` | `literature/` |
| `Yolo26x_v1` / `v2` / `v3` | `results/yolo26x/v1` / `v2` / `v3` |
| `*.pt` (repo root) | `weights/` |
| `src/yolo26x-depth.pt` | `weights/_duplicates/` (byte-identical to the root copy) |
| `presentation_assets/`, `*.pptx` | `assets/` |
| `scratch_tmp/` | `scratch/` |
| `reports/*class_counts*.json` | `reports/class_counts/` |
| `reports/*water_stats*.json` | `reports/water_stats/` |
| `reports/` (everything else) | `reports/analysis/` |

## Second pass (cleanup round)

Changes made after the initial reorganisation, once the working tree had
accumulated new material:

| Old | New |
|---|---|
| `*.log` (repo root, 26 files) | `logs/train/ eval/ annotate/ assemble/ misc/` (git-ignored) |
| `scratch_tmp/` | merged into `scratch/` (contents were byte-identical) |
| `scratch_tmp_compare{,_placeholder,_scatter}.py` | `scratch/compare_base_vs_dr_labels.py --profile {duo_calibrated,placeholder,scatter}` |
| `scratch_compare_0777.png` | `scratch/` |
| `reorganize_versions.sh` | `tools/export_run_bundle.sh <run_name> [dest] [repo_root]` |
| `results/` | removed - `runs/` is the source of truth; regenerate bundles with `tools/export_run_bundle.sh` |
| `imggen/train/v9_combined{,_duo,_duo_scatter,_placeholder}.py` | `imggen/train/v9_combined.py --variant {base,duo,duo_scatter,placeholder}` |
| `imggen/eval/v9_combined*_on_duo.py` (4 files) | `imggen/eval/v9_combined_on_duo.py --variant ...` |
| `reports/flux2dev_v9_starfish_class_counts.json` | `reports/class_counts/` |

The four consolidated pairs differed only in dataset path, run name and
train-image count; those values now live in
`configs/experiments/v9_combined.json`. `.idea/` was removed from git tracking
(files remain on disk) and added to `.gitignore`.

Added: `imggen/eval/v9_duo_v2_on_duo.py`, closing the gap where
`imggen/train/v9_duo_v2.py` had no matching evaluation script.

## Documentation pass

`docs/*.md` and `docs/runbooks/*.md` were updated to the post-reorg paths: 37 file
references, 4 runnable commands (now `python -m ...`), 3 cross-document links
(`POD_RUNBOOK.md` -> `runbooks/POD_RUNBOOK.md`), and 8 `reports/` paths moved into
their `class_counts/` / `water_stats/` / `analysis/` subfolders.

This file is deliberately excluded from that rewrite - its "Old path" column must
keep the pre-reorg names to stay useful.

### Dangling references that could NOT be repointed

These appear in the plan documents and no longer resolve, but **not** because of
the reorganisation - the underlying files were removed by later work. They are
left as written, since they are an accurate record of what existed at the time:

| Reference | Status |
|---|---|
| `dataset/original`, `dataset/original_dr`, `dataset/hyp/no_aug.yaml` | pilot-era datasets, deleted |
| `outputs/1-pilot/**`, `outputs/2-pilot/**` | pilot-era outputs, deleted |
| `outputs/model_compare/hidream_short_test2.png` | model-comparison output, deleted |
| `reports/class_counts/class_counts.json` | never regenerated; `imggen/data/annotate.py` writes it on a default run |
| `reports/class_counts/class_counts_dr.json` | superseded by per-variant report names |

## Known follow-up

~~`core.ignorecase=true` means git's index still records the literature PDFs
under the old capitalised `Literature/`.~~ **Resolved** in the cleanup round -
the index was rebuilt against the lowercase path (`git rm -r --cached
Literature && git add literature`); all 8 PDFs are tracked under
`literature/`.
