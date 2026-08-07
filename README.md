# Dog2go2: Using Monocular Videos to Teach a Unitree Go2 robot.

Dog2go2 turns monocular footage of a real dog into a trained motion policy on
a Unitree Go2. It lifts the dog's 3D pose from video, places it in a metric
world, retargets the motion onto the robot with analytic IK, and trains a
DeepMimic-style PPO policy in Isaac Lab to reproduce it under physics.

<img width="1280" height="720" alt="do2go2" src="https://github.com/user-attachments/assets/7a8d9432-7955-49d2-b706-f99b495b6a69" />

## Video Examples

https://github.com/user-attachments/assets/d0ef4e11-17cb-4c9d-8e76-7a74dc4ab268

## ⚡ Install

### Environment variables

```bash
export A2G2_SSD=/path/to/bulk/storage     # root for weights, work dirs, logs, caches
export PY_CAPTURE=$A2G2_SSD/venvs/env_capture/bin/python
export ANIMER_CKPT=/path/to/AniMer/checkpoint.ckpt
```

`capture/run_default.sh` derives everything else from `$A2G2_SSD` (work dirs,
`$HF_HOME`, calibration folder); each path can also be overridden
individually — see `capture/paths.py`.

### Three environments

The pipeline runs on three Python environments, all managed with
[uv](https://docs.astral.sh/uv/). They stay separate because their
dependencies genuinely conflict (detectron2 and pytorch3d are compiled
against one torch build, Isaac Sim pins its own runtime, and the MuJoCo half
needs neither).

**1 · The uv project** — retargeting, rendering, unit tests. Requires
[uv](https://docs.astral.sh/uv/) and Python ≥ 3.11:

```bash
git clone <this repo> && cd dog2go2
uv sync
```

**2 · The capture environment** — turns video into motion (`capture/`).
CUDA 12.1 toolkit (nvcc) must be installed; `ffmpeg` must be on PATH. 

```bash
uv venv --python 3.10 --seed $A2G2_SSD/venvs/env_capture
uv pip install --python $A2G2_SSD/venvs/env_capture/bin/python \
    -r requirements/capture.txt
uv pip install --python $A2G2_SSD/venvs/env_capture/bin/python \
    --no-build-isolation -r requirements/capture-src.txt   # compiles, ~30 min
```

**3 · The RL environment** — trains the policy (`a2g2_tracking/`). Python
3.11 with Isaac Sim 5.1.0 (NVIDIA's package index), Isaac Lab v2.3.1 from a
pinned checkout, and this repo's task package installed editable. `--seed`
matters: `isaaclab.sh` shells out to `python -m pip`, which a bare uv venv
does not have.

```bash
uv venv --python 3.11 --seed $A2G2_SSD/venvs/env_isaaclab
source $A2G2_SSD/venvs/env_isaaclab/bin/activate
uv pip install -r requirements/isaaclab.txt
git clone https://github.com/isaac-sim/IsaacLab.git ~/py_workspace/IsaacLab
(cd ~/py_workspace/IsaacLab && git checkout v2.3.1 && ./isaaclab.sh --install rsl_rl)
uv pip install -e a2g2_tracking/source/a2g2_tracking
```

### Models and data

| what | size | goes to | how |
|---|---|---|---|
| [AniMer](https://drive.google.com/drive/folders/1rr2dx8CPhVUoEASjxmjE0LJakrUYp0DQ?usp=sharing) ViT-H checkpoint | 8.35 GB | `$ANIMER_CKPT` | download from the AniMer repo (the 2.7 GB ViT-S variant will be rejected by the loader) |
| [SMAL](https://smal.is.tue.mpg.de/) model files | 34 MB | `data/smal/` | register on the SMAL site; its license forbids redistribution |
| Depth Anything V2 (metric, indoor, large) | ~1.3 GB | `$HF_HOME` | automatic on first run |
| Faster R-CNN COCO detector weights | ~430 MB | detectron2 cache | automatic on first run |
| AI4Animation dog mocap (only for the mocap path) | ~800 MB | `data/` | command below |

```bash
# only needed for the mocap entry point (CC BY-NC 4.0, not redistributed here)
curl -L -o data/MotionCapture.zip https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip
unzip -o data/MotionCapture.zip -d data/
```

## 🏃 Quick run

Turn a video of a dog into a Go2 motion. This one command runs the whole
capture-and-retarget pipeline and also writes a side-by-side video — source
footage on the left, the retargeted Go2 rendered in MuJoCo on the right:

```bash
capture/run_default.sh media/dog_3.mp4 dog_3
# -> motions/dog_3.pkl  +  media/sbs_dog_3.mp4
```

Then train the tracking policy on that motion and record it (training takes
roughly 2–3 hours on an RTX 3090). The resulting video shows the reference
motion on the left and the learned policy, running under physics, on the
right:

```bash
source $A2G2_SSD/venvs/env_isaaclab/bin/activate
python a2g2_tracking/scripts/gen_feet_cache.py --clips dog_3 --headless
python a2g2_tracking/scripts/rsl_rl/train.py \
    --task Template-A2g2-Tracking-Direct-v0 --headless --max_iterations 6000 \
    "env.motion_files=[dog_3.pkl]" "env.motion_cyclic=[false]"
python a2g2_tracking/scripts/rsl_rl/play.py \
    --task Template-A2g2-Tracking-Direct-v0 --headless --video \
    --motion dog_3 --start_at_zero
# -> logs/rsl_rl/a2g2_tracking_go2/<run>/videos/play/pip_dog_3.mp4
```

## From video to policy

### 1 · 🎥 Capture — video in, motion out

```bash
capture/run_default.sh media/dog_3.mp4 dog_3
```

One command, eight stages. AniMer regresses the dog's SMAL pose per frame,
Depth Anything V2 fits a metric ground plane, foot contacts are detected, and
a clip-wide bundle adjustment places the dog in the world. The result is
written to the npz contract, retargeted onto the Go2 with IK, rendered, and
composed into the side-by-side video. Watch that video first, and if you like it, continue. Re-running skips any stage whose output already
exists.

> [!IMPORTANT]
> **The video must show the floor.** Everything metric comes from the ground
> plane: its normal defines *up*, its distance gives the camera height that
> anchors every metre, and foot contacts are detected against it. The camera
> should be static the whole time.

### 2 · 🦿 Retarget — motion in, joint trajectories out

`run_default.sh` already did this; it also runs standalone:

```bash
uv run python retarget/retarget.py $A2G2_SSD/work/capture/processed/dog_3.npz
MUJOCO_GL=egl uv run python viz/playback.py motions/dog_3.pkl        # render
uv run python viz/playback.py motions/dog_3.pkl --interactive        # live viewer
```

A rigid trunk frame is fitted to the dog, the feet are re-anchored to the
Go2's leg geometry, and a closed-form 3-DOF leg IK (pure numpy, vectorized)
solves every frame. Post-processing: contact refinement, smoothing, ground
alignment, foot-skate removal, feasibility projection. Output pkls are 50 Hz
with root pose, joint angles and contact labels.

### 3 · 🧠 Track — train the policy in Isaac Lab

```bash
source $A2G2_SSD/venvs/env_isaaclab/bin/activate

# reference foot-position caches for the end-effector reward (once per clip)
python a2g2_tracking/scripts/gen_feet_cache.py --headless

# train (~150k steps/s on an RTX 3090), then film the result
python a2g2_tracking/scripts/rsl_rl/train.py \
    --task Template-A2g2-Tracking-Direct-v0 --headless --max_iterations 6000
python a2g2_tracking/scripts/rsl_rl/play.py \
    --task Template-A2g2-Tracking-Direct-v0 --num_envs 32 --video
```

The task tracks the reference with an 8-term DeepMimic reward, plus reference
state initialization, early termination and domain randomization. The clip set
lives in the env cfg; override it per run with `env.motion_files=[...]`.
Several clips train into **one policy** — the observations carry no clip ID,
the reference preview disambiguates. Follow training with
`tensorboard --logdir logs/rsl_rl/a2g2_tracking_go2`.

## 🧪 Preliminary tests with Mocap

In case you want to test RL learning first from mocap's ground-truth data.
Each layer has a cheap check, runnable in order (the mocap path needs the
AI4Animation dataset — download command in the install section):

```bash
# 0 · unit tests: IK round-trip, npz contract, capture numerics (no GPU)
uv run pytest tests/

# 1 · the robot model: joint table + a sweep video in media/
MUJOCO_GL=egl uv run python viz/smoke_test_go2.py

# 2 · the mocap path end to end
uv run python retarget/parse_mocap.py data/D1_007_KAN01_001.bvh
uv run python retarget/retarget.py data/processed/D1_007_KAN01_001.npz
MUJOCO_GL=egl uv run python viz/playback.py motions/D1_007_KAN01_001.pkl

# 3 · the RL env, no policy: kinematic replay must be clean before training
python a2g2_tracking/scripts/rsl_rl/play.py \
    --task Template-A2g2-Tracking-Direct-v0 --replay \
    --motion D1_007_KAN01_001 --headless --video

# 4 · training loop sanity: a short run must show reward climbing
python a2g2_tracking/scripts/rsl_rl/train.py \
    --task Template-A2g2-Tracking-Direct-v0 --headless --max_iterations 100
```

If the replay in step 3 skates or floats, fix the motion before training on
it — a policy will happily learn a broken reference.

## 📚 Built on

- **AniMer** — animal pose & shape from a single image.
  [paper](https://arxiv.org/abs/2412.00837) ·
  [code](https://github.com/luoxue-star/AniMer)
- **SMAL** — the 3D articulated animal model AniMer regresses.
  [project](https://smal.is.tue.mpg.de/) ·
  [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zuffi_3D_Menagerie_Modeling_CVPR_2017_paper.pdf)
- **Depth Anything V2** — metric depth for the ground plane.
  [paper](https://arxiv.org/abs/2406.09414) ·
  [code](https://github.com/DepthAnything/Depth-Anything-V2)
- **Detectron2** — Faster R-CNN dog detector feeding the crops.
  [code](https://github.com/facebookresearch/detectron2)
- **DeepMimic** — the tracking-reward formulation.
  [paper](https://arxiv.org/abs/1804.02717); end-effector term after
  [Peng et al. 2020](https://arxiv.org/abs/2004.00784)
- **Isaac Lab** — massively parallel RL environments.
  [code](https://github.com/isaac-sim/IsaacLab)
- **rsl_rl** — the PPO implementation.
  [code](https://github.com/leggedrobotics/rsl_rl)
- **MuJoCo Menagerie** — the Go2 model.
  [code](https://github.com/google-deepmind/mujoco_menagerie)
- **AI4Animation dog mocap** — the BVH entry point (Mode-Adaptive Neural
  Networks, SIGGRAPH 2018).
  [paper](https://doi.org/10.1145/3197517.3201366) ·
  [code](https://github.com/sebastianstarke/AI4Animation)

## ⚖️ Licenses

Code in this repo is the author's. Vendored: Unitree Go2 model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(BSD-3), AniMer's `amr` package (MIT). The AI4Animation dog mocap dataset
(CC BY-NC 4.0) and the SMAL model files are **not** redistributed here.

---

Made with ❤️ by [Postcapitalist Robots](https://postcapitalistrobots.substack.com)
for the Open Source Community.
