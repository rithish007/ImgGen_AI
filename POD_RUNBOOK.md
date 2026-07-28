# RunPod Session Runbook

Commands to run every time you start the pod. Copy-paste top to bottom.

---

## What actually persists when you stop the pod

**Your model weights are safe.** A RunPod network volume persists across pod
**stop** *and* **termination** — that is the entire reason it exists as a
separate, separately-billed resource. As long as `HF_HOME` points inside
`/workspace`, the ~50GB of weights survives and you never re-download them.

What does **not** survive a stop:

| Thing | Survives? | Why |
|---|---|---|
| `/workspace/**` (network volume) | **Yes** | Separate persistent resource |
| Model weights in `/workspace/hf_cache` | **Yes** | Lives on the volume |
| Your code in `/workspace/ImgGen_AI` | **Yes** | Lives on the volume |
| Generated outputs in `/workspace/**` | **Yes** | Lives on the volume |
| `pip install` packages | **No** | Go to container disk `site-packages` |
| `export HF_TOKEN=...` etc. | **No** | Shell environment, not disk |
| Anything outside `/workspace` | **No** | Container disk is wiped |

So each session only needs to redo two cheap things: **export the env vars**
and **reinstall pip packages**. Step 2 below removes even the pip reinstall by
putting site-packages on the volume.

---

## 1. Connect

From **Windows PowerShell** (not the pod):

```powershell
ssh root@157.157.221.29 -p 46064 -i $HOME\.ssh\id_ed25519
```

Host and port change every time the pod restarts — copy the current ones from
the RunPod console's **Connect → SSH over exposed TCP** box.

---

## 2. One-time setup (first session on a fresh volume only)

Put Python packages on the network volume so they persist:

```bash
mkdir -p /workspace/hf_cache /workspace/pylibs
```

Set up git properly so you can `git pull` instead of re-copying files by hand.
Add `~/.ssh/id_ed25519.pub` to GitHub → Settings → SSH and GPG keys first, then
connect with `ssh -A` (agent forwarding) and run:

```bash
cd /workspace && git clone git@github.com:rithish007/ImgGen_AI.git && cd ImgGen_AI
```

If the repo directory already exists from an earlier `scp`, convert it in place:

```bash
cd /workspace/ImgGen_AI && git init && git remote add origin git@github.com:rithish007/ImgGen_AI.git && git fetch origin && git reset --hard origin/main
```

---

## 3. Every session — environment

```bash
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=hf_your_actual_token_here
export PYTHONPATH=/workspace/pylibs:$PYTHONPATH
export PATH=/workspace/pylibs/bin:$PATH
```

Install packages to the volume (first session installs, later sessions are a
fast no-op because the files are already there):

```bash
pip install --target=/workspace/pylibs -r /workspace/ImgGen_AI/requirements.txt
```

If `--target` gives you trouble with the preinstalled torch, fall back to a
plain `pip install -r requirements.txt` — it costs ~1 minute per session.

---

## 4. Every session — verify before spending GPU time

```bash
cd /workspace/ImgGen_AI && git pull && python scripts/pod_preflight.py
```

Do not continue past a non-zero exit. It checks GPU arch, VRAM, volume mount,
`HF_HOME` placement, gated-repo access, and pipeline imports — all without
downloading anything.

---

## 5. Generate

Smoke test first (4 images, ~4.5 min) whenever prompts have changed:

```bash
cd /workspace/ImgGen_AI && python src/generate.py --model klein --manifest manifests/smoke.json
```

Full pilot (20 images per model, ~22 min each) only once the smoke test looks right:

```bash
cd /workspace/ImgGen_AI && python src/generate.py --model klein --manifest manifests/pilot.json
```

```bash
cd /workspace/ImgGen_AI && python src/generate.py --model sd35 --manifest manifests/pilot.json
```

Iterate on prompt wording **locally** with `--dry-run` — it loads no model and
needs no GPU, so wording costs nothing while the pod is off.

---

## 6. Pull results back

From **Windows PowerShell** (not the pod):

```powershell
scp -P 46064 -i $HOME\.ssh\id_ed25519 -r root@157.157.221.29:/workspace/ImgGen_AI/outputs C:\dev\ImgGen\outputs
```

---

## 7. Stop the pod

**Click Stop in the RunPod console.** Closing the terminal does not stop
billing — an on-demand pod bills until explicitly stopped. Your weights, code
and outputs on `/workspace` are unaffected; see the table at the top.
