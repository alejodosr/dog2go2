# dog2go2

Dog motion → Unitree Go2 kinematic retargeting (Milestone 1). See `brief_claude.md`
for the full plan. Status: **all phases (0–5) done.**

> The uv project is still named `animal2go2` in `pyproject.toml`; only the
> checkout was renamed. Renaming the project would rewrite `uv.lock`, so it is
> left alone deliberately.

There are two ways in, and they meet at one npz contract:

    BVH mocap ── retarget/parse_mocap.py ──┐
                                           ├── processed/<clip>.npz ── retarget/retarget.py ── motions/<clip>.pkl
    video ───── capture/ (5 stages) ───────┘

[`capture/`](#capture--from-video-instead-of-mocap) is the video path, migrated
in from the AniMer repo — see [what changed](#what-changed-in-the-migration).

**Milestone 2** (RL tracking policy in Isaac Lab, `brief_claude_milestone2.md`)
lives in [`a2g2_tracking/`](a2g2_tracking/README.md) — status, measured
constants, and the train/replay workflow are documented there. Currently:
Phase 2 (tracking env + replay gate) done, Phase 3 (training) next.

## Reproduce everything (3 commands)

```bash
uv sync
curl -L -o data/MotionCapture.zip https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip && unzip -o data/MotionCapture.zip -d data/
MUJOCO_GL=egl uv run python export_all.py
```

The last command retargets every clip in `data/` to `motions/<clip>.pkl`
(brief §7 format) and renders `media/go2_<clip>.mp4` for each, then prints a
summary table (scale, clamp rate, foot-skate before/after) over the dataset.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync
```

The Menagerie Go2 model is vendored in `assets/unitree_go2/`
(BSD-3, see its `LICENSE`; pinned commit in `MENAGERIE_COMMIT.txt`).

### Bulk storage on the SSD

Code stays in this working tree; venvs, model weights, data, logs and caches
live on the SSD under `$A2G2_SSD`. `~/.bashrc` exports:

```bash
export A2G2_SSD=/media/SHARED_DATA/postcapitalistrobots/a2g2
export UV_CACHE_DIR="$A2G2_SSD/caches/uv"
export HF_HOME="$A2G2_SSD/models/hf"      # HuggingFace + DeepLabCut weights
export TORCH_HOME="$A2G2_SSD/models/torch"
export DLC_MODELS="$A2G2_SSD/models/dlc"
```

`data/` is a symlink into `$A2G2_SSD/data/animal2go2`.

### The capture environment

`capture/` needs torch + detectron2 + transformers, which conflict with this
project's mujoco env, so it runs on a separate interpreter. Point `PY_CAPTURE`
at it, and tell it where the AniMer checkpoint lives:

```bash
export PY_CAPTURE=$HOME/anaconda3/envs/animal/bin/python
export ANIMER_CKPT=/media/SHARED_DATA/postcapitalistrobots/animer/checkpoints/gdrive_AniMer/checkpoints/checkpoint.ckpt
```

No checkout of AniMer is needed — its `amr` package is vendored at
[`amr/`](amr/) (MIT, see `amr/LICENSE`). What stays outside the tree is the
8.35 GB checkpoint above, and the SMAL model files at `data/smal/`, which
arrive with the rest of `data/` via the SSD symlink.

That env is torch 2.5.1+cu121, numpy 2.2.6, timm 1.0.28, transformers ≥4.45,
detectron2, pytorch3d, opencv, matplotlib. `HF_HOME` is already exported above
and the Depth Anything V2 weights are cached there.

The dividing line is the package boundary: **`capture/` is torch, everything
else in the repo is uv.** The two exchange npz files and nothing else. The
pure-numpy half of `capture` (everything except stages 1–2) imports fine under
`uv run` too, and `tests/test_capture.py` exercises it there — that is a
standing check that the torch dependency stays confined to the function bodies
that need it.

Download the dog mocap dataset (AI4Animation, SIGGRAPH 2018 —
**CC BY-NC 4.0, not redistributed here**):

```bash
curl -L -o data/MotionCapture.zip https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip
unzip -o data/MotionCapture.zip -d data/
```

## Phase 0 — Go2 smoke test

```bash
MUJOCO_GL=egl uv run python viz/smoke_test_go2.py          # joint table + sweep mp4 in media/
uv run python viz/smoke_test_go2.py --interactive           # same sweep, live viewer
```

Verified from the MJCF (not hardcoded): 12 joints in qpos order FL, FR, RL, RR
× (hip, thigh, calf) — note this differs from our canonical output order
FR, FL, RR, RL, so all indexing goes through joint *names*. Limits: abduction
±1.047, front thigh [−1.571, 3.491], rear thigh [−0.524, 4.538], knee
[−2.723, −0.838]. Thigh = calf = 0.213 m. Home keyframe: z = 0.27 m,
(0, 0.9, −1.8) per leg. Perturbing each joint moves only its own foot.

## Phase 1 — parse & visualize the dog data

```bash
uv run python retarget/parse_mocap.py --scan                        # stats for all 52 clips
uv run python retarget/parse_mocap.py data/D1_007_KAN01_001.bvh     # -> data/processed/*.npz
uv run python viz/viz_source.py data/D1_007_KAN01_001.bvh           # -> media/source_*.mp4
```

Everything downstream of `retarget/skeleton.py` is meters, Z-up, leg order
FR, FL, RR, RL. Source data is Y-up, centimeters, 60 fps, ZXY euler channels.

## Phase 2 — analytic leg IK

```bash
uv run pytest tests/
```

`retarget/ik.py`: closed-form FK/IK for the 3-DOF Go2 leg (abduction from the
y–z geometry, then thigh+calf as a planar two-link arm via law of cosines),
knee-backward branch, pure numpy, vectorized over frames. Foot targets are in
the base frame; unreachable targets are clamped to the workspace so IK never
returns NaN, and `clamp_to_limits()` reports joint-limit violations for the
phase-4 clamp-rate log. Geometry/limit constants are transcribed from the
MJCF and a test asserts them against the loaded model so they can't drift.

FK(IK(p)) round-trip over 400k random reachable targets: max error 0.1 mm
(only at the foot-level-with-hip-axis boundary, where the workspace clamp's
ε engages; machine precision elsewhere). The analytic FK is also checked
against MuJoCo's own kinematics of the calf endpoint — which is what caught
the FL, FR, RL, RR qpos-order trap from phase 0 a second time: canonical dof
vectors must be scattered through `jnt_qposadr`, never block-copied.

## Phase 3 — retarget v0 + playback

```bash
uv run python retarget/retarget.py data/processed/D1_007_KAN01_001.npz   # -> motions/*.pkl
MUJOCO_GL=egl uv run python viz/playback.py motions/D1_007_KAN01_001.pkl # -> media/go2_*.mp4
uv run python viz/playback.py motions/D1_007_KAN01_001.pkl --interactive # live viewer
```

Viewer keys: Space pause, `.` single-step, `[`/`]` speed. Pipeline per the
brief §3: rigid trunk frame fitted to the dog's hip+shoulder segment (x from
pelvis→chest, y from left−right leg roots), uniform scale = 0.27 / mean dog
leg-root height, toes re-anchored from the dog's mean leg mounts to the Go2
leg-plane origins, analytic IK, resample to 50 Hz. Output pkls follow §7
(root_rot **xyzw**; playback converts to MuJoCo wxyz). Pass `--raw` to
`retarget.py` to see this v0 output — foot-skate, jitter, and some ground
penetration, all removed in phase 4. Clamp rates: walk 0.0%, trot 0.8%,
canter 1.3%.

## Phase 4 — post-processing

Runs by default inside `retarget/retarget.py` (`--raw` disables it).
`retarget/postprocess.py`, in order — order matters:

1. **Contact refinement**: stance/swing runs shorter than ~50 ms are detector
   flicker, merged away.
2. **Smoothing**: ~7 Hz Butterworth on foot targets and root trajectory
   *before* IK — smoothing joint angles after IK would drag stance feet and
   reintroduce skate; smoothing the targets cannot.
3. **Ground alignment**: global z-shift so the median stance-foot center sits
   at the foot-sphere radius (sphere touches z = 0).
4. **Foot-skate removal**: each stance segment's foot target is pinned to its
   touch-down xy at ground height, blended in/out over a few frames.
5. **IK + limit report**: solve, clamp to MJCF limits, log the clamp rate
   (>3% triggers a warning — fix scaling upstream, don't clamp harder).

Stance-foot skate drops to ~0 m/s (from ~0.2 m/s raw); per-clip numbers are
in the batch summary table.

## Phase 5 — batch export

```bash
MUJOCO_GL=egl uv run python export_all.py                    # all clips
MUJOCO_GL=egl uv run python export_all.py data/D1_007*.bvh   # a subset
uv run python export_all.py --no-video                       # pkls only, no GPU
```

One `motions/<clip>.pkl` + `media/go2_<clip>.mp4` per source clip, plus the
summary table. `--skip-existing` resumes an interrupted run. Failures don't
abort the batch; they're listed at the end (nonzero exit).

## Capture — from video instead of mocap

Turns a monocular video of an animal into the same npz the BVH path produces,
so everything from `retarget/retarget.py` onward is unchanged. Migrated from
`video2go2/` in the AniMer repo; see [what changed](#what-changed-in-the-migration).

    video ─1─ AniMer SMAL pose ─2─ ground plane (metric depth)
                    │                      │
                    └──3── contacts ───4── world placement (BA)
                                                  │
                                        5── processed/<clip>.npz

| # | stage | what it decides |
|---|---|---|
| 1 | `capture.animer_infer` | SMAL pose/shape per frame; shape frozen to the clip median |
| 2 | `capture.depth_calib` | plane normal + camera height, from metric depth |
| 3 | `capture.contacts_kine` | which paws are planted, per frame |
| 4 | `capture.world_place_ba` | clip-wide trajectory + metric scale (bundle adjustment) |
| 5 | `capture.parse_video` | the npz contract |

```bash
capture/run_default.sh media/dog_3.mp4 dog_3          # optional [trim_start,trim_end]
```

That runs all five stages, then retargets and renders, ending at
`motions/dog_3.pkl` and a side-by-side `media/sbs_dog_3.mp4`. Each stage is
skipped if its output exists, so a re-run resumes — to genuinely redo a clip,
delete its artifacts first, or you are measuring the old run. Every stage is
also a CLI of its own: `$PY_CAPTURE -m capture.<stage> --help`.

Artifacts land under `$A2G2_WORK` (default `$A2G2_SSD/work/capture`), not in
the working tree.

### The one input nothing can measure: focal length

Stage 2 needs a focal length to back-project depth. If no
`<clip>_seed.json` exists the script writes one assuming
`focal = FOCAL_RATIO × frame width`, and says so. The three clips with a real
four-point calibration give ratios 0.791 / 0.825 / 0.858, so the default is
their median, **0.825**. Override when you know better:

```bash
FOCAL_RATIO=0.86 capture/run_default.sh media/clip.mp4 clip
```

This is an assumption, not a measurement, and it propagates into every metre:
`tests/test_capture.py` pins the sensitivity at 11–13% of camera height for a
20–30% focal error. The `scale_mismatch` residual reveals it; `ortho` does
**not** — with the in-plane axes `depth_calib` builds, `ortho` is structurally
zero whatever the focal is.

### What "good" looks like

Verified end to end on `dog_3` (121 frames @ 24 fps) from a wiped state on
2026-08-05, in the AniMer repo, before the migration:

    ground plane      normal [-0.008 -0.996 -0.083], camera height 0.580 m
                      spread over 8 frames 0.550-0.609, round-trip error ~1e-15
    contacts          duty 0.77, zero-feet frames 1.7%
    placement         scale 0.895 m/unit -> shoulder 43.5 cm
                      path 2.30 m over 5.0 s, root z median 0.397 m
                      stance foot skate median 0.020 m
    contract          PASSED, toe z range -0.043 .. 0.104 m
    retarget          scale 0.615, clamp rate 0.28%, skate 0.037 -> 0.001 m/s

Clamp rate under 3% (it warns above), stance skate near zero after
post-processing, `contract checks passed`, plane round-trip ~1e-15. A toe-z
range outside (−0.15, 0.4) fails the contract and means the ground plane is
wrong. **Beware:** clamp rate and skate are both minimised by a *motionless*
robot, so neither on its own proves the motion is right. Watch the
side-by-side.

### Known limitations

**Generated video breaks the metric scale.** Depth models have no real geometry
to measure in AI-generated footage. Against four-point clicked calibrations:

| clip | clicked height | Depth Anything V2 | ZoeDepth (previous) |
|---|---|---|---|
| dog_1 (real) | 1.102 m | −4.6%, normal 1.0° | +18%, 5.2° |
| cat_1 (real) | 1.090 m | −9.3%, normal 2.7° | +14%, 2.3° |
| dog_2 (Veo)  | 1.143 m | −54.4%, normal 1.8° | −22%, 3.9° |

A biological-plausibility rejection test on the recovered plane is the intended
guard and **is not implemented**. Useful signal meanwhile: two independent
depth models disagreeing wildly is evidence of generated video. On the real
clips the two agree within ~25%; on dog_2 they differ by 71%, and on **dog_3 by
46%** (0.847 vs 0.580 m) — so treat dog_3's absolute metres as suspect, its
fitted 43.5 cm shoulder being short for a golden retriever (~55–60 cm).

**Stale calibrations are silently reused.** Stage 2 skips when the clip's
`_depth.json` exists. The script prints which depth model produced it and warns
on a mismatch, but will not overwrite it — delete the file to regenerate.

**Why ZoeDepth was replaced.** Unmaintained, and its BEiT-L checkpoint only
loads under `timm ≤0.6.x`; pinning that broke DeepLabCut, which needs
`timm.layers` (added in 0.9). Depth Anything V2 runs through `transformers`,
which carries its own DINOv2 and needs no `timm` at all — so the conflict is
deleted rather than worked around. It is also more accurate on real footage.

### Optional: the DeepLabCut quality check

Not part of the pipeline — nothing downstream reads it. It answers one
question, *is the mesh actually on the animal?*, by comparing an independent 2D
detector against the mesh's own paw projection. It needs its own environment:
DLC pins `numpy<2` and the capture env is on numpy 2.x, so the two can never be
merged.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. $PY_DLC -m capture.paw_detect_dlc \
  --video media/dog_3.mp4 --mesh $A2G2_WORK/dog_3_animer.npz \
  --out $A2G2_WORK/dog_3_dlc.npz --dest $A2G2_WORK/dog_3_dlc_raw
```

Expected on dog_3: 39 keypoints, all four paws assigned geometrically at
4.2–6.4 px median distance, mesh agreement median 5.7 px / p90 13.0 px,
coverage 90.1%. Agreement above ~15 px median means the mesh is drifting off
the animal and the placement will inherit that.

### Traps in the capture path

These cost real debugging time and are not visible in the code. They lived in
the AniMer repo's `STATUS.md`, which did not migrate.

- **The stages disagree about `PYTHONNOUSERSITE`, so there is no single right
  setting.** Stage 2 *requires* it: `~/.local` holds a broken `soundfile` that
  `transformers` imports, and without the guard you get
  `ModuleNotFoundError: _cffi_backend` surfacing as "Could not import module
  'AutoImageProcessor'". Stages 1, 3 and 8 require the *opposite*: they reach
  `amr/models/__init__.py`, whose chain hits `einops`, which is installed only
  in `~/.local` and not in the conda env. `run_default.sh` therefore has two
  runners; do not collapse them into one.
- **The retargeter names its output from the npz `source` FIELD, not the
  filename.** Feeding it an experiment npz with `source=dog_2` silently
  overwrites `motions/dog_2.pkl`. `run_default.sh` passes `--source $CLIP`;
  for any experimental branch pass `--source <name>_x`.
- **`cam_t` pairs with the RAW FK frame, not the root-centred one.** Adding it
  to root-centred vertices puts the animal in the wrong place, plausibly enough
  that it looks like a scale bug.
- **Scale must never be a least-squares variable.** Errors-in-variables
  attenuation shrinks it 8–25%. `world_place_ba` solves it outside the LS, on
  purpose; `--size-prior 0` is the default because an ablation showed the
  biological prior carried 6–18% of the weight while every sigma in it was
  hand-asserted.
- **`cv2.CAP_PROP_FPS` lies.** Use `ffprobe`; `animer_infer` does, and prints a
  note when the two disagree.
- **AniMer's `focal_full` is an assumed 5000 px** and must never be used
  geometrically. The geometric focal is the one in the calibration json.
- **The MJCF declares legs FL FR RL RR, not the canonical FR FL RR RL.** Never
  block-copy dof vectors between the two — this is the same trap Phase 0 and
  Phase 2 above each caught independently.

## What changed in the migration

`video2go2/` in the AniMer repo became `capture/` here. The stage logic — all
the numerics — is byte-for-byte unchanged; what changed is everything around it.

**Named `capture/`, not `video2go2/`.** Inside a repo already called dog2go2, a
directory named video2go2 that only does half the job reads as the whole
pipeline. `capture/` says what it is: the stage that produces motion, parallel
to `retarget/` and `viz/`, and the counterpart of `retarget/parse_mocap.py`.

**It is a package now.** Stages run as `python -m capture.<stage>` with
absolute `from capture.x import y` imports. Seven `sys.path.insert(__file__.parent)`
hacks are gone; scripts no longer depend on being invoked by path.

**Machine-specific paths moved into `capture/paths.py`.** Nothing under
`capture/` contains a home directory any more. Defaults derive from `$A2G2_SSD`,
matching the convention the rest of this repo already used, and every one is
overridable: `ANIMER_CKPT`, `HF_HOME`, `A2G2_WORK`, `A2G2_CALIB`, `PY_CAPTURE`.
A missing checkpoint now fails immediately with the variable to set, instead of
surfacing as an unpickling error minutes into a run.

**AniMer's `amr` package is vendored**, at [`amr/`](amr/) — 47 files (30 Python,
16 hydra yaml), 412 KB, MIT licensed, copied verbatim along with its `LICENSE`.
Verbatim on purpose: a pruned copy is one you have to re-prune on every upstream
bump. No checkout of AniMer is
needed and `ANIMER_ROOT` is gone. Taking a *subset* was considered and
rejected: `load_amr` calls `AMR.load_from_checkpoint`, which rebuilds the full
LightningModule, and `amr/models/amr.py` imports the discriminator and all six
loss classes at module level while `amr/utils/__init__.py` imports
`MeshRenderer` just to expose an 18-line `recursive_to`. Trimming those means
forking upstream to save ~60 KB of 412 KB. Not worth owning the divergence.

Two details made this safe to do. The checkpoint's hydra config bakes in no
`amr.*` module paths (its only `_target_` is `pytorch_lightning.Trainer`), so
the vendored copy keeps the top-level name `amr` and nothing in the checkpoint
cares. And `get_config(update_cachedir=True)` turns out to be a no-op upstream
— it defines an `update_path` helper and never calls it — so the config's
`SMAL.MODEL_PATH` stays the literal relative string `data/smal/...`.

That relative path is why stage 1 still calls `chdir`, but it now chdirs to
**this** repo instead of a foreign one, which is why `data/` must resolve here.
The stage also resolves `--video`/`--out`/`--debug-frames` to absolute paths
before the chdir; relative ones used to be silently reinterpreted against the
AniMer root.

**`demo_video` is gone**, replaced by [`capture/detector.py`](capture/detector.py).
Stage 1 used exactly two functions from it — `build_detector` and
`detect_animals`, about twenty lines of detectron2 glue — but importing the
module also pulled in trimesh and pyrender for renderer classes the stage never
calls.

**What vendoring did *not* buy.** The 8.35 GB checkpoint is still external, and
so is `data/smal/` (34 MB, arriving through the existing SSD symlink — its
license is unstated by AniMer, so it is not committed here). The environment is
untouched: `amr/` still needs torch, pytorch_lightning, smplx, einops, timm,
yacs, skimage, cv2, pytorch3d, trimesh, pyrender and more. Vendoring code does
not vendor dependencies. The win is a tidier repo, not an easier setup.

**`run_default.sh` goes all the way to the video.** It used to stop at the npz
and print a `cd ../animal2go2` instruction to run by hand. Both halves live
here now, so stages 6–8 (retarget, Go2 render, side-by-side) are part of the
script. It still switches interpreters between them — that boundary is real.

**Four things were deliberately left behind:**

- `calibrate_ground.py` — the interactive four-point clicked calibration that
  `depth_calib` replaced. Only its 6-line `to_ground` was still reachable; that
  now lives in `capture/contacts_ground.py`.
- `pose_refine_dlc.py` — opt-in pose refinement. Nothing creates the input file
  it needs, and the one time it ran on dog_3 it produced 39 m of travel in 5 s,
  1.97 m of foot skate and a violated contract, against 2.30 m and 0.02 m
  unrefined. It is dead code that is harmful when live. Its conditional block
  is gone from `run_default.sh`.
- The synthetic harness (`synth_harness.py`, `synth_eval.py`) and the
  exploratory scripts (`paw_track.py`, `paw_smoke_test.py`, `assess_contacts.py`,
  `check_phaseb.py`, `viz_skeleton.py`, `viz_mesh_overlay.py`) — measurement
  scaffolding for decisions already made and recorded.
- The design docs (`brief_claude.md`, `PLAN.md`, `STATUS.md`, `GUIDELINES.md`).
  Docstrings still cite them by name as provenance; the findings that still
  bind the code are restated above.

**Cleanup that came with it.** `v2k/` was already deleted in the working tree
but its three test modules remained, and they broke collection of the *whole*
suite — so 35 passing tests were invisible. Removed. `conftest.py` now puts the
repo root on `sys.path`, which is what the per-file boilerplate in
`tests/test_ik.py` was doing by hand.

**Verification status: done, twice.** `dog_3` was run end to end from a wiped
state — empty work directory, nothing skipped — and reproduced every number in
[what "good" looks like](#what-good-looks-like) exactly: plane normal
`[-0.008 -0.996 -0.083]`, camera height 0.580 m, round-trip 8.88e-16, duty 0.77,
zero-feet 1.7%, scale 0.8950 m/unit, path 2.30 m, root z 0.397 m, skate
0.020 m, contract passed with toe z −0.043..0.104, retarget scale 0.615, clamp
0.28%, skate 0.037 → 0.001 m/s.

It was then run **again after vendoring**, with `ANIMER_ROOT` unset and the
work directory wiped a second time. All five artifacts — `_animer.npz`,
`_contacts.npz`, `_world.npz`, `processed/dog_3.npz` and `motions/dog_3.pkl` —
came out **byte-identical** to the pre-vendoring run (sha256). The vendoring
changed nothing numerically, which is exactly what a copied-verbatim package
should do.

The test suite is green at 38: the 35 that already existed, plus
`tests/test_capture.py` covering the ground-plane round trip, the focal-length
sensitivity, the horizon mask and the npz contract validator.

## Notes / things that broke (article fodder)

These are from the mocap path (phases 0–5); the video path's equivalents are
under [traps in the capture path](#traps-in-the-capture-path).

- **Root OFFSET is a trap.** Standard BVH forward kinematics adds the root
  joint's `OFFSET` to its position channels. In this dataset the position
  channels are absolute: adding the offset floats every clip above the ground
  by exactly `OFFSET.y` (7.7 cm for most clips, 50 cm for some). Detected
  because `min(toe_z)` per clip matched each file's `OFFSET.y` to the mm.
- The dog skeleton's front legs are the *arm* chains (`...Shoulder→Arm→
  ForeArm→Hand`), rear legs the *leg* chains; toes are BVH end sites.
- **The dog's anatomy leaks into the robot's posture.** The trunk frame's
  pelvis→chest axis carries a constant pitch bias (withers sit ~10 cm higher
  than the hip balls; −18° mean on the trot clip), which made the Go2 play
  back permanently nose-down/up. Fixed by removing the per-clip *median*
  trunk tilt (median, not mean — the trot clip contains a crouch segment
  that would drag the mean). Dynamic pitch stays: the walk clip really does
  start with the dog sniffing the ground, and the retarget should keep that.
- Parsing sanity was verified quantitatively, not just visually: the dog
  moves head-first (heading·spine ≈ +1, heading·tail ≈ −1), and stance
  diagrams show real gaits — D1_007 is a lateral-sequence walk (duty 0.62),
  D1_009_..._002 a trot (diagonal-pair sync +0.74), D1_010_..._004 a canter
  (duty 0.38, flight phases).
