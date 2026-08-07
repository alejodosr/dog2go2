# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-tracking environment config for the Unitree Go2 (DeepMimic-style).

Load-bearing constants (see brief §3):
  - motions are 50 Hz → sim dt = 0.005 s, decimation = 4 → control at 50 Hz,
    exactly one reference frame per policy step.
  - observations/actions are SERIALIZED in canonical FR, FL, RR, RL order —
    the simulator-agnostic contract for Milestone 3.
"""

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# animal2go2 repo root (this file lives 7 levels below it); the motions dir can
# be overridden with A2G2_MOTIONS_DIR.
_REPO_ROOT = Path(__file__).resolve().parents[7]
MOTIONS_DIR = Path(os.environ.get("A2G2_MOTIONS_DIR", _REPO_ROOT / "motions"))

# Phase 2/3 curriculum clips (gait loops → all cyclic). Names match motions/.
WALK_CLIP = "D1_007_KAN01_001"
TROT_CLIP = "D1_009_KAN01_002"
CANTER_CLIP = "D1_010_KAN01_004"
# stage10 additions (2026-07-22): non-gait behaviors.
JUMP_CLIP = "D1_ex04_KAN02_003"
SIT_CLIP = "D1_ex03_KAN02_013"  # sit + foot lift, 76 s


@configclass
class EventCfg:
    """Basic randomization — cheap Milestone 3 insurance, ON from the start."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class A2g2TrackingEnvCfg(DirectRLEnvCfg):
    # env — 50 Hz control over 200 Hz physics (one reference frame per step)
    decimation = 4
    # must cover the longest acyclic clip (sit, 76.3 s); other clips truncate
    # at their own end well before this cap (stage10; was 10.0)
    episode_length_s = 80.0
    action_scale = 0.25
    # PD-target centering: "ref" = ref_dof_pos(t+1) + scale·a (Peng 2020-style,
    # |a|≈1 covers corrections; zero action holds the reference — stage2) vs
    # "default" = default_pose + scale·a (stage1: trot extremes need |a|≈5,
    # RESULTS.md action-reach analysis)
    action_center = "ref"
    # obs: gravity 3 + ang vel 3 + dof pos 12 + dof vel 12 + prev action 12
    #      + K×(ref dof 12 + ref root vel 6) + heading err 2 = 44 + 18K.
    # stage5 (phase-free tracker): NO phase input — the policy conditions
    # only on the reference preview, à la Peng 2020 (which has no phase
    # either; its 4-frame goal spans ~1 s). Makes the policy a general
    # reference-follower: any spliced/looped reference stream works at
    # deployment. Preview steps are time-faithful to the paper's [1,2,10,30]
    # @30 Hz → 20 ms, 40 ms, 0.3 s, 1.0 s @50 Hz. Near an acyclic clip end
    # the preview clamps to the final frame. Heading err (sin/cos of
    # ref-relative yaw, stage3) stays: it is a tracking-error signal, not a
    # clip-position signal; hardware source IMU yaw / odometry.
    ref_target_steps: list = [1, 2, 15, 50]  # control steps ahead
    action_space = 12
    observation_space = 116
    state_space = 124  # critic = actor obs + lin vel 3 + contacts 4 + height 1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot — stock Go2 actuator except: PD gains kp=100/kd=1.0 (kp=25 has a
    # 0.17 rad open-loop tracking floor on the trot, kp=100 → 0.09 with
    # diminishing returns beyond — pd_floor.py sweep, RESULTS.md; Peng 2020
    # uses kp=220 on the heavier Laikago), and velocity_limit 21 → 30 rad/s
    # per the Go2 datasheet (stock 21 derates swing torque: the kp-independent
    # front-calf floor error ~0.26 rad; ref dof_vel peaks 38). Gains + limits
    # go into the policy contract for the MuJoCo port.
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "base_legs": UNITREE_GO2_CFG.actuators["base_legs"].replace(
                stiffness=100.0, damping=1.0, velocity_limit=30.0
            )
        },
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # motion library — clips are ACYCLIC as of stage4: they are raw captures
    # with lead-ins/outs (trot = 36% crouch → stand → trot ending mid-flight;
    # canter contains a 167° turn), so the cyclic wrap seam teleports the
    # reference and killed 511/512 stage3 episodes (RESULTS.md root cause #2).
    # Episode = the clip; clip end is a truncation (bootstrapped), not a death.
    motion_files: list = [
        f"{name}.pkl" for name in (WALK_CLIP, TROT_CLIP, CANTER_CLIP, JUMP_CLIP, SIT_CLIP)
    ]
    motion_cyclic: list = [False, False, False, False, False]
    # equal expected env-step share per clip (clip ~ 1/duration, then uniform
    # frame) — without it the 76 s sit clip would own ~93% of training steps
    # (duration² weighting: most starts AND longest episodes). See MotionLib.
    rsi_equal_clip_steps = True

    # reference state initialization noise
    rsi_joint_pos_noise = 0.05  # rad, uniform ±
    rsi_root_z_noise = 0.02  # m, uniform [0, +z]
    # start every episode at clip frame 0 (play.py --start_at_zero: film one
    # whole episode, crouch → stand → trot → clean truncation); training keeps
    # uniform RSI
    rsi_start_at_zero = False

    # early termination bounds (loose — they truncate hopeless rollouts)
    # keep tight: this bound is itself the anti-drift learning signal — the
    # exp position kernel has no gradient beyond ~0.6 m, and relaxing to 2.0 m
    # quadrupled deterministic drift (RESULTS.md, stage1c)
    term_root_pos_err = 0.5  # m
    term_root_ori_err = 0.785  # rad (45°)
    term_joint_err = 1.0  # rad, mean over 12 dofs
    # falls (base contact) end the episode; play.py --no_early_term turns this
    # off too so a video keeps rolling through a fall
    term_base_contact = True

    # reward weights — Peng 2020 Eq. 4–9 structure and coefficients (stage2
    # alignment; stage1a–d used a different split, see RESULTS.md), plus our
    # contact-match term and the two penalties on top
    rew_joint_pos_w = 0.5
    rew_joint_pos_k = 5.0
    rew_joint_vel_w = 0.05
    rew_joint_vel_k = 0.1
    # end-effector: root-frame foot positions vs sim-generated ref cache
    # (paper Eq. 7 — its 2nd-largest tracking term; needs <clip>_feet.npz)
    rew_ee_w = 0.2
    rew_ee_k = 40.0
    # root pose, one kernel over global pos + ori (paper Eq. 8)
    rew_root_pose_w = 0.15
    rew_root_pose_kp = 20.0
    rew_root_pose_ko = 10.0
    # root velocity, lin + ang with separate coefficients (paper Eq. 9; the
    # 0.2 ang coefficient matters — a shared k=2 saturates on trot ang rates)
    rew_root_vel_w = 0.1
    rew_root_vel_kl = 2.0
    rew_root_vel_ka = 0.2
    rew_contact_match_w = 0.1
    rew_action_rate_w = -0.01
    rew_torque_w = -1.0e-4

    # contact detection threshold on the feet/base sensors
    contact_force_threshold = 1.0  # N

    # modes (set by play.py, not for training)
    kinematic_replay = False  # force-set reference states every step (Phase 2 gate)
    # PD-floor diagnostic (scripts/pd_floor.py): command ref dof_pos directly as
    # PD targets, no policy — measures the tracking error floor of the actuator
    pd_replay = False
    replay_clip: str | None = None  # clip name for replay; None → env_idx % num_clips
    enable_ghost = False  # transparent reference ghost robot (replay/videos)
    # ghost renders side-by-side, not overlaid: the Go2 USD is instanceable, so
    # collision_enabled=False fails on its instance proxies and an overlapped
    # ghost physically ejects the robot at every RSI reset
    ghost_y_offset = 1.0  # m, world +y
    # side-by-side comparison video (play.py --pip_video): two chase cameras,
    # one framing the ghost and one framing the robot; play.py stitches the
    # per-step frames into a single ghost|robot mp4. Requires enable_ghost.
    pip_camera = False
    pip_cam_width = 640  # px, per half
    pip_cam_height = 480
    pip_cam_eye_offset = (1.8, 1.8, 0.9)  # m, world-frame offset from tracked root
    pip_cam_lookat_offset = (0.0, 0.0, 0.2)
    # fixed camera at the SOURCE video's solved pose (play.py --camera-world):
    # renders the policy from the viewpoint of the footage the motion came
    # from, so a source|policy side-by-side reads as one scene. Pose is in the
    # env-origin frame, quat wxyz in the OpenGL/USD convention (+x right,
    # +y up, -z forward); play.py fills these from viz/source_cam.py.
    source_camera = False
    source_cam_pos = (0.0, 0.0, 1.0)
    source_cam_quat = (1.0, 0.0, 0.0, 0.0)
    source_cam_width = 832
    source_cam_height = 480
    source_cam_focal_length = 24.0  # mm; apertures set the FOV against this
    source_cam_h_aperture = 20.955
    source_cam_v_aperture = 12.09
