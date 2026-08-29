"""ImgGen: synthetic underwater imagery + sim-to-real detection pipeline.

Entry points are run as modules from the repository root, e.g.:

    python -m imggen.generate.hunyuan_v8 --manifest manifests/... --out outputs/...

REPO_ROOT is provided so modules can locate repo-relative assets without
counting ``parent`` hops, which silently break whenever a file is moved.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

__all__ = ["REPO_ROOT"]
