# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--replay",
    action="store_true",
    default=False,
    help="Kinematic replay gate: force-set reference states every step, no policy. Prints ground z-offset stats.",
)
parser.add_argument(
    "--motion",
    type=str,
    default=None,
    help="Pin every episode to this clip (policy play and --replay; default: random RSI clip / env_idx %% num_clips).",
)
parser.add_argument("--replay_loops", type=float, default=2.0, help="Clip loops to replay.")
parser.add_argument("--ghost", action="store_true", default=False, help="Spawn the transparent reference ghost.")
parser.add_argument(
    "--pip_video",
    action="store_true",
    default=False,
    help="Record a side-by-side comparison video: reference ghost (left) | policy (right). Implies --ghost.",
)
parser.add_argument(
    "--no_pip",
    action="store_true",
    default=False,
    help="Opt out of the pip side-by-side that --video records by default (follow-cam video only).",
)
parser.add_argument(
    "--ghost_y_offset",
    type=float,
    default=None,
    help="Lateral ghost offset [m] (default: cfg value 1.0, or 3.0 in --pip_video so each camera frames its own robot).",
)
parser.add_argument(
    "--no_early_term",
    action="store_true",
    default=False,
    help="Disable ALL early terminations (drift bounds and falls) to watch free-running behavior to clip end.",
)
parser.add_argument(
    "--early_term",
    action="store_true",
    default=False,
    help="Re-enable early terminations when recording video (videos default to no-reset filming).",
)
parser.add_argument(
    "--start_at_zero",
    action="store_true",
    default=False,
    help="Start every episode at clip frame 0 (film one whole episode start-to-truncation).",
)
parser.add_argument(
    "--camera_world",
    type=str,
    default=None,
    help="Also record from the SOURCE video's solved camera (viz/source_cam.py): "
    "the clip's <clip>_world.npz. Needs --camera_calib and --motion; writes "
    "videos/play/source_<clip>.mp4.",
)
parser.add_argument(
    "--camera_calib",
    type=str,
    default=None,
    help="The clip's depth-calibration json (focal_px, img_size).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# video defaults (user policy, 2026-07-20): recordings are pip side-by-side
# and film without resets — a fall/drift plays out on camera instead of
# teleporting the robot. --no_pip / --early_term opt back out.
if args_cli.video and not args_cli.no_pip:
    args_cli.pip_video = True
if (args_cli.video or args_cli.pip_video or args_cli.camera_world) and not args_cli.early_term:
    args_cli.no_early_term = True
# always enable cameras to record video
if args_cli.video or args_cli.pip_video or args_cli.camera_world:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import a2g2_tracking.tasks  # noqa: F401

# animal2go2 repo root: logs/ and media/ are SSD-backed symlinks there
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _foot_collider_radius(env) -> float | None:
    """Radius of the foot collision sphere from the USD stage, if findable."""
    try:
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = stage_utils.get_current_stage()
        foot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/FR_foot")
        # collider geometry lives inside instanceable references — traverse proxies
        prims = Usd.PrimRange(foot_prim, Usd.TraverseInstanceProxies())
        spheres = [prim for prim in prims if prim.IsA(UsdGeom.Sphere)]
        colliders = [prim for prim in spheres if prim.HasAPI(UsdPhysics.CollisionAPI)]
        for prim in colliders or spheres:
            return float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[WARN] could not read foot collider radius: {exc}")
    return None


_PIP_PANES = {"ghost": "GHOST (reference)", "robot": "RL POLICY"}
_PIP_BANNER_H = 32  # px; keeps 480 + 32 = 512 divisible by the codec macro block
_pip_banners: dict[str, np.ndarray] = {}


def _pip_banner(width: int, text: str) -> np.ndarray:
    """Black strip with the centered pane label, rendered once and cached."""
    if text not in _pip_banners:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (width, _PIP_BANNER_H), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=20)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - x1 + x0) // 2 - x0, (_PIP_BANNER_H - y1 + y0) // 2 - y0), text, (255, 255, 255), font)
        _pip_banners[text] = np.asarray(img, dtype=np.uint8)
    return _pip_banners[text]


def _pip_frame(raw_env) -> np.ndarray:
    """Stitch the labelled ghost|robot chase-cam frames of env 0 into one RGB image."""
    halves = []
    for name, label in _PIP_PANES.items():
        rgb = raw_env._pip_cams[name].data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        halves.append(np.concatenate([_pip_banner(rgb.shape[1], label), rgb], axis=0))
    # 16 px keeps the stitched width a multiple of the codec macro block (no resize)
    divider = np.full((halves[0].shape[0], 16, 3), 255, dtype=np.uint8)
    return np.concatenate([halves[0], divider, halves[1]], axis=1)


def run_replay(env_cfg, task_name: str):
    """Phase 2 acceptance gate: physics kinematic puppet + z-offset measurement."""
    env_cfg.kinematic_replay = True
    env_cfg.replay_clip = args_cli.motion
    env_cfg.enable_ghost = args_cli.ghost
    env_cfg.events = None
    env_cfg.episode_length_s = 1.0e6  # let the loop decide when to stop

    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        clip_tag = args_cli.motion or "all"
        video_kwargs = {
            "video_folder": os.path.join(_REPO_ROOT, "media", f"replay_{clip_tag}"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    raw = env.unwrapped
    lib = raw._motion_lib
    num_steps = int(args_cli.replay_loops * lib.durations.max().item() / raw.step_dt)
    if args_cli.video:
        num_steps = min(num_steps, args_cli.video_length)
    zero_actions = torch.zeros(raw.num_envs, gym.spaces.flatdim(raw.single_action_space), device=raw.device)

    stance_foot_z, swing_foot_z = [], []
    env.reset()
    for _ in range(num_steps):
        with torch.inference_mode():
            env.step(zero_actions)
        foot_z = raw._robot.data.body_pos_w[:, raw._feet_ids_body, 2]
        ref_contacts = lib.get_frame(raw._clip_idx, raw._ref_t)["foot_contacts"].bool()
        stance_foot_z.append(foot_z[ref_contacts])
        swing_foot_z.append(foot_z[~ref_contacts])

    stance = torch.cat(stance_foot_z)
    swing = torch.cat(swing_foot_z)
    from a2g2_tracking.motion.motion_loader import FOOT_REST_CENTER_Z, GROUND_Z_OFFSET

    radius = _foot_collider_radius(raw)
    residual = FOOT_REST_CENTER_Z - stance.median().item()
    print("\n" + "=" * 70, flush=True)
    print(f"[REPLAY] clips: {lib.names} | envs: {raw.num_envs} | steps: {num_steps}", flush=True)
    print(f"[REPLAY] stance foot center z [m]: median {stance.median():.4f}  "
          f"p5 {stance.quantile(0.05):.4f}  p95 {stance.quantile(0.95):.4f}  n={stance.numel()}", flush=True)
    print(f"[REPLAY] swing  foot center z [m]: median {swing.median():.4f}  min {swing.min():.4f}", flush=True)
    if radius is not None:
        print(f"[REPLAY] foot collider sphere radius: {radius:.4f} m (geometric; effective "
              f"rest height of foot center is {FOOT_REST_CENTER_Z:.4f} m from the settle test)", flush=True)
    print(f"[REPLAY] applied GROUND_Z_OFFSET = {GROUND_Z_OFFSET:+.4f} m; residual vs. rest height "
          f"= {residual:+.4f} m (add this to GROUND_Z_OFFSET if |residual| is not mm-noise)", flush=True)
    print("=" * 70 + "\n", flush=True)
    env.close()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.enable_ghost = args_cli.ghost or args_cli.pip_video
    env_cfg.rsi_start_at_zero = args_cli.start_at_zero
    env_cfg.replay_clip = args_cli.motion
    if args_cli.no_early_term:
        # no deaths at all: only clip end / timeout truncates the episode
        env_cfg.term_root_pos_err = 1.0e9
        env_cfg.term_root_ori_err = 1.0e9
        env_cfg.term_joint_err = 1.0e9
        env_cfg.term_base_contact = False
    if args_cli.pip_video:
        env_cfg.pip_camera = True
        if args_cli.num_envs is None:
            env_cfg.scene.num_envs = 1
    if args_cli.ghost_y_offset is not None:
        env_cfg.ghost_y_offset = args_cli.ghost_y_offset
    elif args_cli.pip_video:
        env_cfg.ghost_y_offset = 3.0
    if args_cli.camera_world:
        if not (args_cli.camera_calib and args_cli.motion):
            raise SystemExit("--camera_world needs --camera_calib and --motion")
        import pickle

        from scipy.spatial.transform import Rotation

        sys.path.insert(0, _REPO_ROOT)
        from viz.source_cam import source_camera_pose

        with open(os.path.join(_REPO_ROOT, "motions", f"{args_cli.motion}.pkl"), "rb") as f:
            _motion = pickle.load(f)
        _pos, _R, _fovx, _fovy, (_w, _h) = source_camera_pose(
            args_cli.camera_world, args_cli.camera_calib, _motion
        )
        _qx, _qy, _qz, _qw = Rotation.from_matrix(_R).as_quat()
        env_cfg.source_camera = True
        env_cfg.source_cam_pos = tuple(float(v) for v in _pos)
        env_cfg.source_cam_quat = (float(_qw), float(_qx), float(_qy), float(_qz))
        env_cfg.source_cam_height = 480
        env_cfg.source_cam_width = int(round(480 * _w / _h)) // 2 * 2
        _fl = env_cfg.source_cam_focal_length
        env_cfg.source_cam_h_aperture = 2 * _fl * float(np.tan(np.radians(_fovx / 2)))
        env_cfg.source_cam_v_aperture = 2 * _fl * float(np.tan(np.radians(_fovy / 2)))
        if args_cli.num_envs is None:
            env_cfg.scene.num_envs = 1
    if args_cli.video:
        # follow camera on the robot root (not settable via hydra: asset_name
        # defaults to None and cfg overrides are type-checked against that)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (1.8, 1.8, 0.9)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)

    # kinematic replay gate (Phase 2): no checkpoint involved
    if args_cli.replay:
        if args_cli.num_envs is None:
            env_cfg.scene.num_envs = 1
        run_replay(env_cfg, args_cli.task)
        return

    # specify directory for logging experiments (repo logs/ symlink → SSD)
    log_root_path = os.path.join(_REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    pip_frames: list[np.ndarray] = []
    src_frames: list[np.ndarray] = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
        if args_cli.pip_video:
            pip_frames.append(_pip_frame(env.unwrapped))
        if args_cli.camera_world:
            src_frames.append(
                env.unwrapped._source_cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
            )
        if args_cli.video or args_cli.pip_video or args_cli.camera_world:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
            # with --start_at_zero the first done closes one whole episode —
            # stop there so the pip video never spans a reset (ghost teleports)
            if (args_cli.pip_video or args_cli.camera_world) and args_cli.start_at_zero and dones.any():
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if pip_frames:
        import imageio.v2 as imageio

        clip_tag = args_cli.motion or "mixed"
        pip_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(pip_dir, exist_ok=True)
        pip_path = os.path.join(pip_dir, f"pip_{clip_tag}.mp4")
        fps = round(1.0 / dt)
        imageio.mimwrite(pip_path, pip_frames, fps=fps, quality=8)
        print(f"[INFO] Side-by-side video ({len(pip_frames)} frames @ {fps} fps): {pip_path}", flush=True)

    if src_frames:
        import imageio.v2 as imageio

        clip_tag = args_cli.motion or "mixed"
        src_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(src_dir, exist_ok=True)
        src_path = os.path.join(src_dir, f"source_{clip_tag}.mp4")
        fps = round(1.0 / dt)
        imageio.mimwrite(src_path, src_frames, fps=fps, quality=8)
        print(f"[INFO] Source-camera video ({len(src_frames)} frames @ {fps} fps): {src_path}", flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
