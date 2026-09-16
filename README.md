# ImgGen

Synthetic underwater imagery generation and sim-to-real object detection.
Generated benthic-survey scenes are domain-randomised through a Jerlov water-column
model, then used to train YOLO detectors evaluated against the real DUO dataset.

## Layout

| Path | Contents |
|---|---|
| `imggen/` | The package. All active pipeline code. |
| `imggen/prompts/` | Active prompt engines. Superseded ones in `prompts/legacy/`. |
| `imggen/generate/` | Image generation entry points (flux2dev v9, hunyuan v8). |
| `imggen/dr/` | Domain randomization and Jerlov water-column physics. |
| `imggen/train/` | YOLO training entry points. |
| `imggen/eval/` | Evaluation against DUO and real imagery. |
| `imggen/data/` | Manifest building, dataset assembly, annotation. |
| `imggen/analysis/` | Water statistics, class balance, depth/range, montages, plots. |
| `archive/generators/` | Superseded generators, kept as the reproducibility record. |
| `tools/` | Standalone dev utilities (preflight, model smoke tests). |
| `slurm/` | Stanage batch scripts. |
| `manifests/`, `configs/` | Generation inputs — the reproducibility record. |
| `configs/experiments/` | Experiment variant registries consumed by the consolidated runners. |
| `reports/` | Measured outputs: `class_counts/`, `water_stats/`, `analysis/`. |
| `docs/` | Plans, decision log, and `runbooks/` for HPC and RunPod. |
| `literature/` | Reference papers. |
| `runs/` | Trained models, evals and detections — **the source of truth** for results. |
| `logs/` | Run logs, grouped `train/ eval/ annotate/ assemble/ misc/`. Git-ignored. |
| `weights/`, `outputs/`, `dataset/`, `scratch/` | Local artefacts, git-ignored. |

## Running

Everything runs as a module from the repository root — no `PYTHONPATH`, no install step:

```bash
python -m imggen.generate.hunyuan_v8 --manifest manifests/benthic-survey-1000-hunyuan.json --out outputs/hunyuan/v8
```

Superseded generators still run, from the same root:

```bash
python -m archive.generators.generate_flux2dev_v5 --manifest manifests/pilot.json --out outputs/flux2dev/v5
```

Experiment families that differ only by dataset are driven by a variant flag rather than
duplicated scripts:

```bash
python -m imggen.train.v9_combined --variant duo_scatter
python -m imggen.eval.v9_combined_on_duo --variant duo_scatter
```

`tools/` scripts are standalone and take a plain path:

```bash
python tools/pod_preflight.py
tools/export_run_bundle.sh v9_flux2dev_duo_dr_v3
```

See [docs/PATH_MIGRATION.md](docs/PATH_MIGRATION.md) for the full old-to-new path map.

## Related resources

- **Dataset and model checkpoints:** [huggingface.co/datasets/ArcaneEvolution/MScDissertation](https://huggingface.co/datasets/ArcaneEvolution/MScDissertation) (CC-BY-NC-4.0)
- **Companion repository:** [Shallow_Seabed](https://github.com/rithish007/Shallow_Seabed) — the descoped Unreal Engine 5.8 simulator arm (private, access on request)

## License

Code in this repository is released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite it:

```bibtex
@misc{ramamoorthysathya2026imggen,
  author       = {Ramamoorthy Sathya, Rithish},
  title        = {{ImgGen\_AI: Synthetic Underwater Imagery Generation and Sim-to-Real Object Detection}},
  year         = {2026},
  howpublished = {\url{https://github.com/rithish007/ImgGen_AI}},
  note         = {Code repository. University of Sheffield MSc Robotics Dissertation}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable citation record (also surfaced by GitHub's
"Cite this repository" button).
