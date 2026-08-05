"""Every machine-specific path the capture stack needs, in one place.

AniMer's `amr` package is vendored at the repo root, so no checkout of it is
needed. What cannot live in the working tree is the 8.35 GB checkpoint and the
34 MB of SMAL model files; both are located through environment variables so
that nothing under `capture/` contains a home directory.

Following the convention the rest of this repo already uses, bulk storage sits
under `$A2G2_SSD`, and the defaults here derive from it. Set these in
`~/.bashrc` alongside the ones the README documents::

    export A2G2_SSD=/media/SHARED_DATA/postcapitalistrobots/a2g2
    export ANIMER_CKPT=$A2G2_SSD/models/animer/checkpoint.ckpt

`resolve()` is deliberately not called at import time: the modules that only
need numpy must stay importable (and testable) on a machine that has neither
the checkpoint nor a GPU.
"""
import os
from pathlib import Path

#: This repository. The checkpoint's hydra config stores `SMAL.MODEL_PATH` as
#: a path relative to the working directory, so stage 1 chdirs here.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Bulk storage root. Everything heavy — weights, caches, artifacts — hangs off
#: this, so a different machine only has to redefine one variable.
SSD = Path(os.environ.get("A2G2_SSD", "/media/SHARED_DATA/postcapitalistrobots/a2g2"))

#: The SMAL model files (~34 MB), reached as `data/smal/` relative to REPO_ROOT
#: because that is the literal string inside the checkpoint's hydra config.
#: `data/` is a symlink into $A2G2_SSD, matching the mocap data's convention.
SMAL_DIR = REPO_ROOT / "data" / "smal"
SMAL_MODEL = SMAL_DIR / "my_smpl_00781_4_all.pkl"

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
