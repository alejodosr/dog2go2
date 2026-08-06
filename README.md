# Dog2go2: Teach a Unitree Go2 to move like a dog, starting from a video of one.

Dog2go2 turns monocular footage of a real dog into a trained motion policy on
a Unitree Go2. It lifts the dog's 3D pose from video, places it in a metric
world, retargets the motion onto the robot with analytic IK, and trains a
DeepMimic-style PPO policy in Isaac Lab to reproduce it under physics.

<img width="1280" height="720" alt="telegram-cloud-photo-size-4-6028323995846905170-y" src="https://github.com/user-attachments/assets/fa0d1193-f3c5-488b-8e96-fc7b3b77ec3d" />

## ⚡ Quick install

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
git clone <this repo> && cd dog2go2
uv sync
```

That covers the retargeting half (MuJoCo, numpy IK, rendering). Capture and RL
each need their own environment:

| what | needs | where |
|---|---|---|
| `capture/` (video → motion) | torch, detectron2, transformers, pytorch3d + the 8.35 GB [AniMer](https://github.com/luoxue-star/AniMer) checkpoint | `$PY_CAPTURE` interpreter |
| `a2g2_tracking/` (RL) | Isaac Lab v2.3.1, Isaac Sim 5.1.0 | its own venv |

```bash
export A2G2_SSD=/path/to/bulk/storage          # data, weights, logs, caches
export PY_CAPTURE=$HOME/anaconda3/envs/animal/bin/python
export ANIMER_CKPT=/path/to/AniMer/checkpoint.ckpt
```

## 🏃 Quick run

Video → retargeted Go2 motion, plus a side-by-side video (real dog | Go2 IK):

```bash
capture/run_default.sh media/dog_3.mp4 dog_3
# -> motions/dog_3.pkl  +  media/sbs_dog_3.mp4
```

Train the tracking policy on it and film the result (reference | policy
side-by-side):

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

One command, eight stages: per-frame SMAL pose (AniMer) → metric ground plane
(Depth Anything V2) → foot contacts → bundle-adjusted world placement → npz
contract → IK retarget → Go2 render → side-by-side video. Watch the
side-by-side: it is the one artifact that can't lie. Re-running skips any
stage whose output already exists.

> [!IMPORTANT]
> **The video must show the floor.** Everything metric comes from the ground
> plane: its normal defines *up*, its distance gives the camera height that
> anchors every metre, and foot contacts are detected against it. No visible
> floor, no world. AI-generated footage has no real geometry for the depth
> model to measure — expect its scale to be off by tens of percent.

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

Each layer has a cheap check, runnable in order (the mocap path needs the
AI4Animation dataset, see CHANGELOG):

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

Made with ❤️ from [Postcapitalist Robots](https://postcapitalistrobots.substack.com)
for the Open Source Community.
