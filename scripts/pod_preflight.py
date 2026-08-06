"""Pod preflight - run this BEFORE downloading any model weights.

Every check here is cheap (seconds, no large downloads) and each one catches a
failure that would otherwise surface only after tens of GB have been pulled and
paid for. In particular it verifies that the template's torch was actually built
for Blackwell (sm_120) and that every gated HF repo is accessible, which is the
single most common way this pipeline stalls on first run.

    python scripts/pod_preflight.py

Exit code 0 = clear to download weights. Non-zero = fix before proceeding.
"""

from __future__ import annotations

import os
import shutil
import sys

RESULTS: list[tuple[str, bool, str]] = []
WARNINGS: list[tuple[str, str]] = []

MODEL_REPOS = [
    "black-forest-labs/FLUX.2-klein-base-9B",
    "black-forest-labs/FLUX.2-dev",
    "stabilityai/stable-diffusion-3.5-large",
    "Qwen/Qwen-Image",
    "lightx2v/Qwen-Image-Lightning",
    "facebook/sam3",
    "IDEA-Research/grounding-dino-base",
    "depth-anything/Depth-Anything-V2-Large-hf",
]

# Weights + HF cache overhead + pilot outputs. For the 2-pilot 5-model
# comparison (Z-Image-Turbo dropped - see generate.py's MODELS comment), all
# cached simultaneously on one volume even though only one is ever
# GPU-resident at a time: klein ~18GB + flux2dev+text-encoder ~90GB + SD3.5
# ~16GB + Qwen-Image ~40GB (bf16, 20B; Lightning is a small LoRA on the same
# base, negligible extra), plus SAM3/GDINO/DA-V2. This is a rough sum, not
# measured - treat a free-space check that barely passes as a reason to
# verify actual usage partway through downloads, not as comfortable headroom.
REQUIRED_GB = 180


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def warn(name: str, detail: str = "") -> None:
    WARNINGS.append((name, detail))
    print(f"  [WARN] {name}" + (f" - {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")


def check_gpu() -> None:
    section("GPU / CUDA")
    try:
        import torch
    except ImportError as e:
        check("torch importable", False, str(e))
        return

    check("torch version", True, torch.__version__)

    if not check("cuda available", torch.cuda.is_available()):
        return

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    cc = f"sm_{props.major}{props.minor}"

    check("device", True, f"{props.name} ({vram_gb:.1f} GB, {cc})")

    # 32GB card is the whole basis of the model sizing in the plan.
    check("vram >= 30 GB", vram_gb >= 30, f"{vram_gb:.1f} GB")

    # The real Blackwell trap: a template built for cu121 has no sm_120 kernels
    # and every GPU op fails, despite cuda.is_available() returning True.
    # An exact arch_list match is not required - kernels for a lower minor
    # version of the same major arch run fine (sm_86 kernels work on sm_89) -
    # so this is informational and the matmul below is the authoritative test.
    arches = torch.cuda.get_arch_list()
    if cc in arches:
        check(f"torch built for {cc}", True)
    elif any(a.startswith(f"sm_{props.major}") for a in arches):
        warn(f"no exact {cc} kernels", f"same-major fallback available: {arches}")
    else:
        check(f"torch built for {cc}", False,
              f"no sm_{props.major}x kernels at all - wrong CUDA build. arch_list={arches}")

    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        (a @ a).sum().item()
        torch.cuda.synchronize()
        check("bf16 matmul on gpu (authoritative)", True)
    except Exception as e:
        check("bf16 matmul on gpu (authoritative)", False, f"{type(e).__name__}: {e}")


def check_storage() -> None:
    section("Storage")
    hf_home = os.environ.get("HF_HOME")
    check("HF_HOME set", bool(hf_home), hf_home or "unset - weights will NOT persist across pod stops")

    target = hf_home or "/workspace"
    if not os.path.isdir(target):
        # huggingface_hub creates this dir lazily on first download, so an
        # absent leaf directory just means "not downloaded yet", not "broken".
        # What actually matters is that the parent (the mounted volume) exists
        # and is writable.
        parent = os.path.dirname(target.rstrip("/")) or "/"
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            try:
                os.makedirs(target, exist_ok=True)
                check(f"{target} created", True, "did not exist yet, created it")
            except OSError as e:
                check(f"{target} exists", False, f"could not create: {e}")
                return
        else:
            check(f"{target} exists", False, f"parent {parent} missing/not writable - network volume not mounted?")
            return

    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1024**3
    check(f"{target} free space", free_gb >= REQUIRED_GB,
          f"{free_gb:.0f} GB free (need ~{REQUIRED_GB} GB)")

    if hf_home and not hf_home.startswith("/workspace"):
        check("HF_HOME on network volume", False,
              f"{hf_home} is on container disk - weights lost on pod stop")


def check_hf_access() -> None:
    section("Hugging Face access")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not check("HF_TOKEN set", bool(token)):
        return

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError as e:
        check("huggingface_hub importable", False, str(e))
        return

    api = HfApi(token=token)
    try:
        check("token valid", True, f"user={api.whoami()['name']}")
    except Exception as e:
        check("token valid", False, str(e))
        return

    # Metadata only - no weights pulled. Catches unaccepted licenses in seconds.
    for repo in MODEL_REPOS:
        try:
            api.model_info(repo)
            check(f"access {repo}", True)
        except GatedRepoError:
            check(f"access {repo}", False, "LICENSE NOT ACCEPTED - visit the model page and accept")
        except RepositoryNotFoundError:
            check(f"access {repo}", False, "repo not found (name changed?)")
        except Exception as e:
            check(f"access {repo}", False, f"{type(e).__name__}: {e}")


def check_libraries() -> None:
    section("Pipeline classes")
    try:
        import diffusers
        import transformers
    except ImportError as e:
        check("diffusers/transformers importable", False, str(e))
        return

    check("diffusers", True, diffusers.__version__)
    check("transformers", True, transformers.__version__)

    for mod, names in (
        (diffusers, ["Flux2KleinPipeline", "Flux2Pipeline", "StableDiffusion3Pipeline"]),
        (transformers, ["Sam3Model", "Sam3Processor",
                        "GroundingDinoForObjectDetection",
                        "DepthAnythingForDepthEstimation"]),
    ):
        for n in names:
            check(n, hasattr(mod, n))

    try:
        import accelerate
        check("accelerate", True, accelerate.__version__)
    except ImportError:
        check("accelerate", False, "required by enable_model_cpu_offload()")

    try:
        import torchao
        check("torchao", True, torchao.__version__)
    except ImportError:
        check("torchao", False, "required for flux2dev's fp8 quantization (not needed for klein)")


def main() -> int:
    print("=" * 62)
    print("POD PREFLIGHT - no weights are downloaded by this script")
    print("=" * 62)

    check_gpu()
    check_storage()
    check_hf_access()
    check_libraries()

    failures = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 62)
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s):")
        for name, detail in WARNINGS:
            print(f"  - {name}: {detail}")
    if failures:
        print(f"{len(failures)} FAILED - fix before downloading weights:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(RESULTS)} checks passed. Clear to run the smoke test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
