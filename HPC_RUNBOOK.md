# Sheffield HPC (Stanage) Runbook

Companion to [POD_RUNBOOK.md](POD_RUNBOOK.md) - same pipeline, different compute
backend. Built from https://docs.hpc.shef.ac.uk/en/latest/ (2026-08-06). The
docs site itself flags that GPU nodes run EL9 while much of the documentation
still reflects EL7 - treat exact module names/versions below as a starting
point to verify with `module avail`, not gospel.

---

## Why Stanage instead of RunPod for the next run

RunPod's per-hour billing model directly caused two failure modes in the
2-pilot run: a stopped pod refusing to restart because its pinned physical
host was full, and a mid-generation cutoff when account balance hit zero.
Stanage is allocation-based (Slurm queue, not pay-as-you-go GPU-hours), which
removes both failure classes - but introduces queue wait time and campus
network/VPN requirements RunPod didn't have. Reserve Stanage for the real
production runs (the eventual 1000-image generation); the university's own
docs note GPU users should expect the H100/A100 fleet, not a single
consumer-style card.

## GPU fleet (confirm current queue depth with `sinfo`/`squeue` before assuming)

| Type | Nodes | GPUs/node | VRAM |
|---|---|---|---|
| A100 | 52 | 4 | 80 GB |
| H100 | 12 | 2 | 80 GB |
| H100 NVL | 16 | 4 | 94 GB |

A100 has by far the most nodes - start there for shorter queue wait unless a
specific run needs H100's extra throughput. Either comfortably fits klein
(~29GB) and flux2dev (~32GB fp8) with room to spare. Max 12 GPUs
concurrently per user across all three types.

---

## 1. Prerequisites (one-time, before any of this works)

- HPC account: requires passing the "HPC Driving License" test
  (https://infosecurity.shef.ac.uk/, needs VPN) **and** a supervisor-submitted
  request via the IT Service Desk Self Service Portal. If dissertation HPC
  access is already active, this is done - otherwise it's the actual first
  step, not SSH.
- DUO multi-factor auth enabled: https://www.sheffield.ac.uk/it-services/mfa/set-mfa
- Off-campus access needs the University SSL VPN (or the separately-approved
  HPC SSH gateway service for VPN-less access - ask research-it@sheffield.ac.uk
  if that's preferable to VPN for repeated sessions).

## 2. Connect

```bash
ssh -X YOUR_USERNAME@stanage.shef.ac.uk
```

Approve the DUO push (or enter a passcode) when prompted. Lands on a login
node - **login nodes are for editing/job prep/submission only**, never for
running generation directly (unlike the RunPod flow, where the SSH session
itself ran the GPU job).

## 3. Storage - where things go matters here more than on RunPod

| Area | Path | Quota | Backed up? |
|---|---|---|---|
| Home | `/users/$USER` | 50 GB / 300k files | No |
| Fastdata (parscratch) | `/mnt/parscratch/users/$USER/` | none enforced | No |
| Scratch | `$TMPDIR` | none, deleted at job end | No |
| Shared/project | `/shared/[project]` | 10 TB (needs separate request) | Yes, every 4h + nightly |

Tonight's `hf_cache` alone was 150-232GB - **that cannot go in home (50GB
cap)**. Point `HF_HOME` at `/mnt/parscratch/users/$USER/hf_cache`, exactly
analogous to RunPod's `/workspace/hf_cache` network volume. Clone the repo and
put venv/conda env there too if it grows past a few GB.

## 4. One-time environment setup

```bash
module load Anaconda3/2024.02-1
mkdir -p /mnt/parscratch/users/$USER/anaconda/.pkg-cache /mnt/parscratch/users/$USER/anaconda/.envs
```

Add to `~/.condarc` so packages don't fill the home quota:
```yaml
pkgs_dirs:
  - /mnt/parscratch/users/$USER/anaconda/.pkg-cache/
envs_dirs:
  - /mnt/parscratch/users/$USER/anaconda/.envs
```

```bash
conda create -n imggen python=3.12
source activate imggen
cd /mnt/parscratch/users/$USER/ && git clone git@github.com:rithish007/ImgGen_AI.git && cd ImgGen_AI
pip install -r requirements.txt
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache
export HF_TOKEN=hf_your_actual_token_here
```

Run `python scripts/pod_preflight.py` here too (it's not RunPod-specific -
same GPU/CUDA/HF-access/pipeline-import checks apply). **Watch for the same
stale-torchaudio ABI conflict** hit twice tonight on RunPod
(`pip uninstall -y --break-system-packages torchaudio` was the fix there) -
Stanage's base Python module stack is unverified against this project's pinned
`diffusers==0.39.0`/`torch` combination, so don't assume it's clean.

## 5. Generate - via `sbatch`, not a live SSH session

This is the actual structural difference from RunPod: no `nohup`/`disown`
juggling, no dropped-SSH-session risk. Submit a job, log out, come back later.

`generate_klein.slurm`:
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=imggen_klein
#SBATCH --output=/mnt/parscratch/users/%u/ImgGen_AI/logs/klein_%j.out

module load Anaconda3/2024.02-1
source activate imggen
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache
cd /mnt/parscratch/users/$USER/ImgGen_AI
python src/generate.py --model klein --manifest manifests/2-pilot.json --out outputs/2-pilot/klein
```

```bash
sbatch generate_klein.slurm       # submit
squeue --me                       # check status / queue position
squeue --me --start               # estimate start time
sacct --user=$USER                # job history
scancel <jobid>                   # cancel
```

For a quick interactive smoke test instead of a full queued job:
```bash
srun --partition=gpu --qos=gpu --gres=gpu:a100:1 --mem=82G --cpus-per-task=8 --pty bash -i
```

`--time=02:00:00` covers flux2dev's slower ~68s/image x 50 + load overhead
with buffer; klein needs far less but the extra headroom costs nothing if the
job finishes early.

## 6. Pull results back

Same syntax as RunPod, different host - requires VPN/campus network same as
SSH:
```powershell
scp -r YOUR_USERNAME@stanage.shef.ac.uk:/mnt/parscratch/users/YOUR_USERNAME/ImgGen_AI/outputs C:\dev\ImgGen\outputs
```
or `rsync -avzP` for large/repeated pulls (resumable, unlike plain scp).

## 7. Nothing to "stop"

No per-hour billing to cut off - once the sbatch job completes or its
`--time` limit expires, it just ends. No equivalent of RunPod's "click Stop or
it keeps billing" risk. Fair-share queue priority is the real constraint
instead of cost; heavy usage may just mean longer waits for the next job, not
a bill.

## Open questions before the first real run here

- Exact GPU-node module stack (`module avail` post-login) - EL9 vs EL7
  docs mismatch means the versions above need re-verification.
- Whether dissertation HPC access already covers Stanage specifically, and
  whether a `/shared/` project allocation exists or needs requesting for
  anything that should survive project end (home/fastdata are both
  unbacked-up).
- Actual queue wait time on the `gpu` partition/qos in practice - unknown
  until submitted once.
