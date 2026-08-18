#!/bin/bash
#SBATCH --job-name=hunyuan-benthic1000
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/parscratch/users/elp25rr/logs/hunyuan_%j.out
#SBATCH --error=/mnt/parscratch/users/elp25rr/logs/hunyuan_%j.err
#
# hunyuan v8, benthic-survey-1000-hunyuan.json (1000 rows, split manifest -
# flux2dev gets its own separate 1000-row file, zero row overlap between the
# two - see scripts/build_split_manifest.py).
#
# Measured on RunPod (4x A100 SXM): 284.3s/image -> ~79h for 1000 images.
# --time=96:00:00 is the cluster's MaxTime for the gpu partition (confirmed
# via `scontrol show partition gpu`) - the absolute max available, only
# ~17h of margin over the estimate. generate_hunyuan_v8.py skips any row
# whose PNG+JSON already exist, so if 96h genuinely isn't enough, resubmit
# the same job and it continues from wherever it stopped - no wasted GPU
# time re-doing completed rows. This is expected to plausibly need at least
# one resubmission - do not treat a walltime kill as a failure, check
# progress first.
#
# Test first with: sbatch --export=LIMIT=3 slurm/hunyuan_benthic1000.sbatch
# Full run:         sbatch slurm/hunyuan_benthic1000.sbatch

set -euo pipefail

module load Anaconda3/2025.06-1
source activate hunyuan

export HF_HOME=/mnt/parscratch/users/elp25rr/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
# NOTE: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE deliberately NOT set globally -
# diffusers' offline-fallback path was unreliable with them set (see commit
# history). local_files_only=True is passed directly in
# generate_hunyuan_v8.py's from_pretrained() calls instead.

cd /mnt/parscratch/users/elp25rr/ImgGen_AI

LIMIT_FLAG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

python src/generate_hunyuan_v8.py \
    --manifest manifests/benthic-survey-1000-hunyuan.json \
    --out outputs/hunyuan/v8 \
    ${LIMIT_FLAG}
