# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate `<clip>_feet.npz` reference foot-position caches for the
end-effector reward (Peng 2020 Eq. 7).

Uses the kinematic-replay puppet, so the foot positions come from Isaac's own
forward kinematics — no analytic FK, no convention risk. Positions are stored
in the ROOT frame (canonical FR, FL, RR, RL order): root-relative quantities
are periodic across the loop wrap, so no unwrap bookkeeping is needed.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate reference foot-position caches via kinematic replay.")
parser.add_argument("--task", type=str, default="Template-A2g2-Tracking-Direct-v0")
parser.add_argument(
    "--clips", type=str, default=None,
    help="comma-separated clip names (with or without .pkl); default: the env cfg's motion_files",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import sys
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import a2g2_tracking.tasks  # noqa: F401
from a2g2_tracking.motion.motion_loader import LEG_ORDER
from a2g2_tracking.tasks.direct.a2g2_tracking.a2g2_tracking_env_cfg import MOTIONS_DIR


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=None)
    if args_cli.clips:
        env_cfg.motion_files = [
            c if c.endswith(".pkl") else c + ".pkl" for c in args_cli.clips.split(",")
        ]
        env_cfg.motion_cyclic = [False] * len(env_cfg.motion_files)
    num_clips = len(env_cfg.motion_files)
    env_cfg.scene.num_envs = num_clips  # env i replays clip i
    env_cfg.kinematic_replay = True
    env_cfg.rew_ee_w = 0.0  # the caches don't exist yet
    env_cfg.episode_length_s = 1.0e6  # never time out mid-collection

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    lib = raw._motion_lib
    num_frames = lib.num_frames.tolist()
    caches = [np.zeros((n, 4, 3), dtype=np.float32) for n in num_frames]
    filled = [np.zeros(n, dtype=bool) for n in num_frames]

    with torch.inference_mode():
        env.reset()
        # step k puts the puppet at frame k % N; k = N revisits frame 0
        for k in range(1, max(num_frames) + 1):
            env.step(torch.zeros((raw.num_envs, 12), device=raw.device))
            feet = raw._feet_pos_root().cpu().numpy()
            for i, n in enumerate(num_frames):
                if k <= n:
                    caches[i][k % n] = feet[i]
                    filled[i][k % n] = True

    print(f"\n## Reference foot caches (root frame, {'/'.join(LEG_ORDER)})\n")
    for i, path in enumerate(env_cfg.motion_files):
        assert filled[i].all(), f"clip {path}: {int((~filled[i]).sum())} frames not visited"
        out = MOTIONS_DIR / (path.removesuffix(".pkl") + "_feet.npz")
        np.savez(out, feet_pos_root=caches[i], leg_order=np.array(LEG_ORDER))
        span = caches[i].reshape(-1, 3)
        print(f"- {out.name}: {num_frames[i]} frames, z range [{span[:, 2].min():.3f}, {span[:, 2].max():.3f}] m")
    sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
