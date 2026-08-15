# HiDream-O1-Image — dropped, not integrated

**Status: abandoned 2026-08-08.** All code, cloned repo, isolated venv, and
model weights removed from the pod. This file documents why, so it doesn't
get re-attempted blind later.

## What it is

`HiDream-ai/HiDream-O1-Image` ("full", 8B, MIT-licensed, the only openly
downloadable non-API variant - "Pro" at 200B+ is API-only). A pixel-space
Unified Transformer built on `Qwen3VLForConditionalGeneration` - no VAE, no
separate text encoder, unlike every other model in this pipeline (klein,
flux2dev, Hunyuan all use diffusers-style pipelines or at least a
transformers/diffusers-native loading path). Not a diffusers pipeline at
all - requires cloning their own GitHub repo (`github.com/HiDream-ai/HiDream-O1-Image`)
and importing their custom `models/pipeline.py::generate_image()`.

## Why it didn't work for this project

**Root cause: this project's prompts are too long for HiDream's code.**
`models/utils.py::get_rope_index_fix_point()` anchors image-token rope
position IDs to a hardcoded `fix_point=4096` default - which is exactly the
patch count for HiDream's own 2048x2048 default resolution
(`2048/32 x 2048/32 = 64x64 = 4096`). This is very likely tuned/tested
against their own short, caption-style example prompts ("medium shot,
eye-level, front view..."). This project's prompts are long, structured,
multi-clause compound sentences (~200 words, ~439 tokens after their
chat-template wrapping) - well outside whatever envelope that hardcoded
constant assumes.

**Evidence, not speculation** - confirmed empirically, not just read from
source:
- Our actual prompt (long, ~439 tokens) -> crashes every time with
  `RuntimeError: The size of tensor a (439) must match the size of tensor b
  (4535) at non-singleton dimension 2` inside `apply_rotary_pos_emb`, at
  BOTH 1024x1024 and 2048x2048 (1024 silently snaps to 2048 - see
  `PREDEFINED_RESOLUTIONS` in their `utils.py` - so this wasn't actually two
  different resolution tests, it was the same one twice).
- `4535 - 4096 (image patches at 2048x2048) = 439` - matches our prompt's
  token count exactly, not a coincidence.
- Their own short example prompt ("a red apple on a wooden table") at the
  same 2048x2048 -> works cleanly, produces a real, high-quality image
  (`outputs/model_compare/hidream_short_test2.png`, kept as evidence).

So this is a real, reproducible upstream limitation tied specifically to
prompt length/structure, not a bug in this project's integration code.

## Other things fixed along the way (documented in case any of this recurs
## if HiDream - or a similarly-shaped model - is ever revisited)

1. **Their `requirements.txt` pins `transformers==4.57.1`**, this project's
   shared pod environment is pinned to `5.14.1` - installing their
   requirements into the shared `/workspace/pylibs` would have downgraded
   transformers for every other model (klein/flux2dev/Hunyuan/SAM3
   annotation) sharing that environment. Fixed with an isolated venv
   (`python3 -m venv --system-site-packages`) - but venv isolation alone
   wasn't enough: `pod_env.sh`'s `PYTHONPATH=/workspace/pylibs:...` export
   still shadowed the venv's own site-packages ahead of it on `sys.path`,
   silently using the wrong (5.14.1) transformers until `unset PYTHONPATH`
   was added before invoking the venv's python.
2. **Their default `use_flash_attn: True`** (`models/pipeline.py:341`)
   hard-asserts if `flash-attn` isn't installed, rather than falling back -
   exactly what their own README warns about. Patched to `False` on the
   cloned repo (not this project's code).
3. **Container root disk filled to 100%** (30GB total) from `pip`'s
   download cache (2.7GB) and, more significantly, the HF model download
   landing in `/root/.cache/huggingface` (28GB) instead of the persistent
   `/workspace` volume - happened because the first attempt ran before
   `source pod_env.sh` (which sets `HF_HOME=/workspace/hf_cache`). Also hit
   `/root/.triton`'s JIT cache defaulting to the same full disk -
   `TRITON_CACHE_DIR` needed redirecting to `/workspace` too. Both are
   generic "state defaults to container-local disk unless explicitly
   redirected" traps, not HiDream-specific, but HiDream's unusually large
   checkpoint (28GB) is what actually filled the disk.

## If this is ever revisited

The only path that showed any life was a short prompt. Options, roughly in
order of effort:
- Write a much shorter, HiDream-specific prompt variant (defeats the
  purpose of the shared `prompts.py` engine used by every other model here -
  a real cost, not a small one).
- Patch `get_rope_index_fix_point`'s `fix_point` handling to derive from
  actual text length instead of the hardcoded 4096 - requires understanding
  their position-id math well enough to not silently produce subtly wrong
  (not just crashing) results, which needs live tensor-shape debugging on a
  GPU, not just source reading.
- Wait and see if upstream fixes this - it reads like a real bug/unstated
  constraint in their own code, worth checking their GitHub issues before
  re-attempting.

Not recommended to re-attempt without a clear reason - klein, flux2dev, and
Hunyuan all accept this project's actual prompts without modification.
