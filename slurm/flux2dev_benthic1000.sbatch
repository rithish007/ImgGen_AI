#!/bin/bash
#SBATCH --job-name=flux2dev-benthic1000
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=/mnt/parscratch/users/elp25rr/logs/flux2dev_%j.out
#SBATCH --error=/mnt/parscratch/users/elp25rr/logs/flux2dev_%j.err
#
# flux2dev v9, benthic-survey-1000-flux2dev.json (1000 rows, split manifest -
# hunyuan gets its own separate 1000-row file, zero row overlap between the
# two - see scripts/build_split_manifest.py).
#
# Measured on RunPod (2x A100 SXM): 71.0s/image -> ~19.7h for 1000 images.
# --time=30:00:00 leaves ~50% margin. generate_flux2dev_v9.py skips any row
# whose PNG+JSON already exist, so if this walltime is somehow still not
# enough, just resubmit the same job - it picks up where it left off, no
# wasted GPU time on already-completed rows.
#
# Test first with: sbatch --export=LIMIT=3 slurm/flux2dev_benthic1000.sbatch
# Full run:         sbatch slurm/flux2dev_benthic1000.sbatch

set -euo pipefail

module load Anaconda3/2025.06-1
source activate imggen

export HF_HOME=/mnt/parscratch/users/elp25rr/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1

cd /mnt/parscratch/users/elp25rr/ImgGen_AI

LIMIT_FLAG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

python src/generate_flux2dev_v9.py \
    --model flux2dev \
    --manifest manifests/benthic-survey-1000-flux2dev.json \
    --out outputs/flux2dev/v9 \
    ${LIMIT_FLAG}
