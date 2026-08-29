#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=annotate_flux2dev_starfish
#SBATCH --output=/mnt/parscratch/users/elp25rr/logs/annotate_flux2dev_starfish_%j.out
#SBATCH --error=/mnt/parscratch/users/elp25rr/logs/annotate_flux2dev_starfish_%j.err
#
# SAM3 auto-annotation for the 1000-image v9 starfish-camouflage generation
# run (outputs/flux2dev/v9_starfish, from slurm/flux2dev_starfish_benthic1000.sbatch).
# SAM3 is small (no sharding needed, per imggen/data/annotate.py's own
# comment) - one A100 is comfortably enough. 30 min budget vs the 45 min
# annotate_hunyuan.slurm template's 30 min for 50 images is generous;
# reference RunPod SAM3 run for the equivalent v9 base 1000-image batch was
# well under an hour.
set -e
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate imggen
source ~/.bashrc_hf_token
export HF_HOME=/mnt/parscratch/users/elp25rr/hf_cache
cd /mnt/parscratch/users/elp25rr/ImgGen_AI
python -m imggen.data.annotate \
    --images-dir outputs/flux2dev/v9_starfish \
    --out-dir outputs/flux2dev/v9_starfish/labels/sam3 \
    --report reports/class_counts/flux2dev_v9_starfish_class_counts.json
