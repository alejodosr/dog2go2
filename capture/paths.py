"""Every machine-specific path the capture stack needs, in one place.

The pipeline depends on two things that are far too big to live in the working
tree: a checkout of AniMer (for its `amr` package and the SMAL model files) and
an 8.35 GB checkpoint. Both are located through environment variables so that
nothing under `capture/` contains a home directory.

Following the convention the rest of this repo already uses, bulk storage sits
under `$A2G2_SSD`, and the defaults here are derived from it. Set the variables
in `~/.bashrc` alongside the ones the README documents::

    export A2G2_SSD=/media/SHARED_DATA/postcapitalistrobots/a2g2
    export ANIMER_ROOT=$HOME/py_workspace/AniMer
    export ANIMER_CKPT=$A2G2_SSD/models/animer/checkpoint.ckpt

`resolve()` is deliberately not called at import time: the modules that only
need numpy must stay importable (and testable) on a machine that has neither
AniMer nor the checkpoint.
"""
import os
from pathlib import Path

#: Bulk storage root. Everything heavy — weights, caches, artifacts — hangs off
#: this, so a different machine only has to redefine one variable.
SSD = Path(os.environ.get("A2G2_SSD", "/media/SHARED_DATA/postcapitalistrobots/a2g2"))

#: Checkout of https://github.com/luoxue-star/AniMer. Stage 1 imports `amr.*`
#: and `demo_video` from it, and stage 3 reads `data/smal/*.pkl` under it.
ANIMER_ROOT = Path(os.environ.get("ANIMER_ROOT",
                                  Path.home() / "py_workspace" / "AniMer"))

#: The 8.35 GB vith checkpoint. The 2.7 GB one under `checkpoints/AniMer/`
#: declares `BACKBONE.TYPE=vit`, which AniMer's own loader rejects.
ANIMER_CKPT = Path(os.environ.get(
    "ANIMER_CKPT", SSD / "models/animer/gdrive_AniMer/checkpoints/checkpoint.ckpt"))

#: HuggingFace cache for the Depth Anything V2 weights (~1.3 GB, auto-downloaded).
HF_CACHE = Path(os.environ.get("HF_HOME", SSD / "models/hf"))

#: Where the pipeline writes per-clip artifacts, and where the ground-plane
#: calibrations live. `run_default.sh` mirrors these.
WORK = Path(os.environ.get("A2G2_WORK", SSD / "work/capture"))
CALIB = Path(os.environ.get("A2G2_CALIB", WORK / "calib_depth"))


def resolve(path, what):
    """Return `path` as an absolute Path, or explain which variable to set.

    A missing checkpoint otherwise surfaces deep inside torch as an unhelpful
    unpickling error, several minutes into a run.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(
            f"{what} not found at {p}\n"
            f"Set the matching environment variable (see capture/paths.py) "
            f"or pass the path explicitly on the command line.")
    return p
