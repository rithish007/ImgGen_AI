#!/bin/bash
#SBATCH --job-name=hunyuan-benthic1000-h100
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/parscratch/users/elp25rr/logs/hunyuan_h100_%j.out
#SBATCH --error=/mnt/parscratch/users/elp25rr/logs/hunyuan_h100_%j.err
#
# Parallel hedge alongside the queued A100 job (job 11341551, untouched).
# H100 nodes only have 2 GPUs each (160GB VRAM combined) vs A100's 4x80GB
# (320GB) - Hunyuan's ~160GB bf16 weight footprint barely fits with near-zero
# VRAM headroom for activations. Not treated as fatal: device_map="auto"
# automatically spills any overflow to system RAM (this node has 514GB,
# confirmed via `scontrol show node gpu21`) instead of crashing - just
# possibly slower per-image if real offload traffic occurs. True multi-node
# (2 separate H100 nodes = 4 GPUs) was considered and rejected: device_map=
# "auto" is single-process and can't place tensors across a network to a
# second physical node - that would need a genuine distributed rewrite of a
# trust_remote_code custom architecture, too risky to attempt blind here.
#
# Writes to a SEPARATE output dir (v8_h100) from the A100 run (v8) so the
# two attempts can never collide on the same files. generate_hunyuan_v8.py
# skips rows whose PNG+JSON already exist, so whichever of the two jobs
# ends up running, the other can be safely cancelled later with nothing
# lost or duplicated.
#
# Full run: sbatch slurm/hunyuan_benthic1000_h100.sbatch

set -euo pipefail

module load Anaconda3/2025.06-1
source activate hunyuan

export HF_HOME=/mnt/parscratch/users/elp25rr/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"

cd /mnt/parscratch/users/elp25rr/ImgGen_AI

LIMIT_FLAG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

python src/generate_hunyuan_v8.py \
    --manifest manifests/benthic-survey-1000-hunyuan.json \
    --out outputs/hunyuan/v8_h100 \
    ${LIMIT_FLAG}
