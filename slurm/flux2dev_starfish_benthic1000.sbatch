#!/bin/bash
#SBATCH --job-name=flux2dev-starfish-benthic1000
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=/mnt/parscratch/users/elp25rr/logs/flux2dev_starfish_%j.out
#SBATCH --error=/mnt/parscratch/users/elp25rr/logs/flux2dev_starfish_%j.err
#
# flux2dev v9, starfish-focused prompt engine (imggen/prompts/flux2dev_v9_starfish.py
# via imggen/generate/flux2dev_v9_starfish.py), same manifest as the base v9 run
# (benthic-survey-1000-flux2dev.json, 1000 rows) so this is a direct
# regeneration of the same (seed, counts, density, framing) requests under
# the camouflage-focused starfish wording, not a new sample - diagnosis
# finding: starfish AP50 near-zero across every model trained so far because
# generated starfish read as bright/smooth "hero" objects instead of the
# camouflaged, partially-buried real thing.
#
# Token budget verified against the real FLUX.2 tokenizer (not estimated):
# worst-case 497/512 across a 30,000-combo sweep (500 seeds x 3 densities x
# 2 framings, all-classes-max-count) plus the actual production seed range
# (42000-43000) - zero breaches. Tighter margin than v8's ~395/512 worst
# case, but confirmed safe, not just plausible.
#
# Same timing assumption as flux2dev_benthic1000.sbatch (71.0s/image on 2x
# A100 SXM -> ~19.7h for 1000 images, --time=30:00:00 leaves ~50% margin).
# imggen/generate/flux2dev_v9_starfish.py skips any row whose PNG+JSON already
# exist, so if walltime runs out, just resubmit - it resumes where it left
# off.
#
# Test first with: sbatch --export=LIMIT=3 slurm/flux2dev_starfish_benthic1000.sbatch
# Full run:         sbatch slurm/flux2dev_starfish_benthic1000.sbatch

set -euo pipefail

module load Anaconda3/2025.06-1
source activate imggen

export HF_HOME=/mnt/parscratch/users/elp25rr/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
# NOTE: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE deliberately NOT set globally -
# see flux2dev_benthic1000.sbatch's identical note. local_files_only=True is
# passed directly in imggen/generate/flux2dev_v9_starfish.py's from_pretrained()
# calls instead.

cd /mnt/parscratch/users/elp25rr/ImgGen_AI

LIMIT_FLAG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

python -m imggen.generate.flux2dev_v9_starfish \
    --model flux2dev \
    --manifest manifests/benthic-survey-1000-flux2dev.json \
    --out outputs/flux2dev/v9_starfish \
    ${LIMIT_FLAG}
