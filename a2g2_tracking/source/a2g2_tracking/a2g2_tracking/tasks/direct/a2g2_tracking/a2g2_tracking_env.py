# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DeepMimic-style motion tracking for the Unitree Go2 (DirectRLEnv).

Interface contract (Milestone 3 depends on this):
  - the actor observation and the action vector are serialized in canonical
    FR, FL, RR, RL leg order — never in Isaac's joint order;
  - reference targets are expressed relative to the robot's CURRENT base
    frame, never in world frame;
  - no base linear velocity and no world positions in the actor obs (the
    critic gets them as privileged extras).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, CameraCfg, ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, sample_uniform

from a2g2_tracking.motion.motion_lib import MotionLib
from a2g2_tracking.motion.motion_loader import LEG_ORDER, make_dof_index_map

from .a2g2_tracking_env_cfg import MOTIONS_DIR, A2g2TrackingEnvCfg


def _quat_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) of a wxyz quaternion batch (..., 4)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class A2g2TrackingEnv(DirectRLEnv):
    cfg: A2g2TrackingEnvCfg

    def __init__(self, cfg: A2g2TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # canonical FR,FL,RR,RL ↔ sim joint order maps (sim = canonical[perm])
        self._perm_c2s = make_dof_index_map(self._robot.joint_names).to(self.device)
        self._perm_s2c = torch.argsort(self._perm_c2s)

        # motion library (all conventions fixed inside the loader)
        self._motion_lib = MotionLib.from_files(
            [MOTIONS_DIR / f for f in self.cfg.motion_files],
            joint_names=self._robot.joint_names,
            cyclic=list(self.cfg.motion_cyclic),
            device=self.device,
            equal_clip_steps=self.cfg.rsi_equal_clip_steps,
        )
        needs_feet = self.cfg.rew_ee_w != 0.0 and not (self.cfg.kinematic_replay or self.cfg.pd_replay)
        if needs_feet and "feet_pos_root" not in self._motion_lib._fields:
            raise ValueError(
                "rew_ee_w != 0 but some clips lack a <clip>_feet.npz cache — "
                "generate them with scripts/gen_feet_cache.py (kinematic replay)"
            )
        if abs(self._motion_lib.fps * self.step_dt - 1.0) > 1e-6:
            raise ValueError(
                f"control rate {1.0 / self.step_dt:.1f} Hz != motion fps {self._motion_lib.fps} — "
                "the one-reference-frame-per-step invariant is broken"
            )

        # per-env reference cursor
        self._clip_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ref_t = torch.zeros(self.num_envs, device=self.device)

        # actions, canonical order (as the policy emits them)
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 12, device=self.device)

        # body/sensor indices — feet resolved per leg name so the canonical
        # order is guaranteed regardless of regex match order
        self._feet_ids_sensor = [self._contact_sensor.find_bodies(f"{leg}_foot")[0][0] for leg in LEG_ORDER]
        self._feet_ids_body = [self._robot.find_bodies(f"{leg}_foot")[0][0] for leg in LEG_ORDER]
        self._base_id_sensor, _ = self._contact_sensor.find_bodies("base")

        if self._pip_cams:
            self._pip_cam_eye = torch.tensor(self.cfg.pip_cam_eye_offset, device=self.device)
            self._pip_cam_lookat = torch.tensor(self.cfg.pip_cam_lookat_offset, device=self.device)

        # replay clip override
        if self.cfg.replay_clip is not None:
            if self.cfg.replay_clip not in self._motion_lib.names:
                raise ValueError(f"replay clip {self.cfg.replay_clip!r} not in {self._motion_lib.names}")
            self._replay_clip_idx = self._motion_lib.names.index(self.cfg.replay_clip)
        else:
            self._replay_clip_idx = None

        # action-saturation diagnostics (RESULTS.md open item): per-env
        # accumulators, logged per episode in _reset_idx
        self._steps_since_reset = torch.zeros(self.num_envs, device=self.device)
        self._action_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self._action_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._clamp_frac_sum = torch.zeros(self.num_envs, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "joint_pos_tracking",
                "joint_vel_tracking",
                "ee_tracking",
                "root_pose_tracking",
                "root_vel_tracking",
                "contact_match",
                "action_rate",
                "torque",
            ]
        }

    # -- scene ---------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._ghost = None
        if self.cfg.enable_ghost:
            ghost_cfg = self.cfg.robot.replace(prim_path="/World/envs/env_.*/Ghost")
            ghost_cfg.spawn = ghost_cfg.spawn.replace(
                activate_contact_sensors=False,
                rigid_props=ghost_cfg.spawn.rigid_props.replace(disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                # OPAQUE: de-instancing (below) makes this override live, and
                # a 0.35-opacity surface over the dark floor renders as
                # nothing but a drop shadow. Solid blue = visible reference.
                visual_material_path="ghost_material",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.55, 1.0), opacity=1.0),
            )
            self._ghost = Articulation(ghost_cfg)
            self.scene.articulations["ghost"] = self._ghost
            # The Go2 USD is instanceable, so the collision_props override
            # above never reaches the colliders inside the instanced
            # references (the RESULTS.md "solid ghost" gotcha). De-instance
            # the ghost subtree to make its prims authorable, then force
            # every collider off — the ghost must never push the robot.
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            ghost_prim = stage.GetPrimAtPath("/World/envs/env_0/Ghost")
            while True:  # nested instances become real one level at a time
                instanced = [p for p in Usd.PrimRange(ghost_prim) if p.IsInstance()]
                if not instanced:
                    break
                for p in instanced:
                    p.SetInstanceable(False)
            for p in Usd.PrimRange(ghost_prim):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
            # Rebind the ghost material to every mesh, stronger than the
            # bindings the de-instanced prototypes brought along — otherwise
            # per-mesh prototype materials win and the ghost renders with
            # missing/dark meshes.
            from pxr import UsdGeom, UsdShade

            mat_prim = stage.GetPrimAtPath("/World/envs/env_0/Ghost/ghost_material")
            if mat_prim.IsValid():
                mat = UsdShade.Material(mat_prim)
                for p in Usd.PrimRange(ghost_prim):
                    if p.IsA(UsdGeom.Gprim):
                        UsdShade.MaterialBindingAPI.Apply(p).Bind(
                            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                        )
        self._pip_cams = {}
        if self.cfg.pip_camera:
            if self._ghost is None:
                raise ValueError("pip_camera requires enable_ghost")
            for name in ("ghost", "robot"):
                cam = Camera(
                    CameraCfg(
                        prim_path=f"/World/envs/env_.*/PipCam{name.capitalize()}",
                        width=self.cfg.pip_cam_width,
                        height=self.cfg.pip_cam_height,
                        data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 200.0)),
                    )
                )
                self.scene.sensors[f"pip_cam_{name}"] = cam
                self._pip_cams[name] = cam
        self._source_cam = None
        if self.cfg.source_camera:
            cam = Camera(
                CameraCfg(
                    prim_path="/World/envs/env_.*/SourceCam",
                    width=self.cfg.source_cam_width,
                    height=self.cfg.source_cam_height,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=self.cfg.source_cam_focal_length,
                        horizontal_aperture=self.cfg.source_cam_h_aperture,
                        vertical_aperture=self.cfg.source_cam_v_aperture,
                        clipping_range=(0.1, 200.0),
                    ),
                )
            )
            self.scene.sensors["source_cam"] = cam
            self._source_cam = cam
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -- reference lookup ----------------------------------------------------

    def _ref_frame(self, dt_ahead: float = 0.0) -> dict[str, torch.Tensor]:
        return self._motion_lib.get_frame(self._clip_idx, self._ref_t + dt_ahead)

    def _ref_root_pos_w(self, ref: dict[str, torch.Tensor]) -> torch.Tensor:
        return ref["root_pos"] + self._terrain.env_origins

    # -- step ----------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        # advance the reference cursor: after this physics step the robot is
        # compared against ref(t + dt)
        self._ref_t += self.step_dt
        self._actions = actions.clone()
        if self.cfg.pd_replay:
            # command the reference pose the robot is tracked against this step
            self._processed_actions = self._ref_frame()["dof_pos"]
        elif not self.cfg.kinematic_replay:
            if self.cfg.action_center == "ref":
                # center on the pose being tracked this step: zero action holds
                # the reference, |a|≈1 covers corrections (vs |a|≈5 to reach
                # trot extremes from the static default — RESULTS.md)
                center = self._ref_frame()["dof_pos"]
            else:
                center = self._robot.data.default_joint_pos
            targets_sim = center + self.cfg.action_scale * self._actions[:, self._perm_c2s]
            limits = self._robot.data.soft_joint_pos_limits
            self._processed_actions = targets_sim.clamp(limits[..., 0], limits[..., 1])
            abs_a = self._actions.abs()
            self._steps_since_reset += 1.0
            self._action_abs_sum += abs_a.mean(dim=-1)
            self._action_abs_max = torch.maximum(self._action_abs_max, abs_a.amax(dim=-1))
            self._clamp_frac_sum += (self._processed_actions != targets_sim).float().mean(dim=-1)
        if self._ghost is not None:
            self._write_ref_state(
                self._ghost, self._ref_frame(), self._robot._ALL_INDICES, y_offset=self.cfg.ghost_y_offset
            )
        if self._pip_cams:
            # chase cams: fixed world-axis offset from each tracked root (same
            # framing as the viewer follow cam)
            for name, root_pos in (
                ("ghost", self._ghost.data.root_pos_w),
                ("robot", self._robot.data.root_pos_w),
            ):
                self._pip_cams[name].set_world_poses_from_view(
                    root_pos + self._pip_cam_eye, root_pos + self._pip_cam_lookat
                )
        if self._source_cam is not None:
            # fixed pose, re-set every step like the pip cams so resets can't
            # move it
            pos = torch.tensor(self.cfg.source_cam_pos, device=self.device) + self._terrain.env_origins
            quat = torch.tensor(self.cfg.source_cam_quat, device=self.device).expand(self.num_envs, 4)
            self._source_cam.set_world_poses(pos, quat, convention="opengl")

    def _apply_action(self):
        if self.cfg.kinematic_replay:
            # physics kinematic puppet: force the reference state every substep
            self._write_ref_state(self._robot, self._ref_frame(), self._robot._ALL_INDICES)
        else:
            self._robot.set_joint_position_target(self._processed_actions)

    def _write_ref_state(
        self, robot: Articulation, ref: dict[str, torch.Tensor], env_ids: torch.Tensor, y_offset: float = 0.0
    ):
        root_pos = self._ref_root_pos_w(ref)[env_ids]
        if y_offset != 0.0:
            root_pos = root_pos.clone()
            root_pos[:, 1] += y_offset
        root_state = torch.cat(
            [
                root_pos,
                ref["root_rot"][env_ids],
                ref["root_lin_vel"][env_ids],
                ref["root_ang_vel"][env_ids],
            ],
            dim=-1,
        )
        robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(ref["dof_pos"][env_ids], ref["dof_vel"][env_ids], None, env_ids)
        # PD holds the pose between state writes (ghost is written once per env step)
        robot.set_joint_position_target(ref["dof_pos"][env_ids], env_ids=env_ids)

    # -- observations --------------------------------------------------------

    def _foot_contacts(self) -> torch.Tensor:
        """Current foot contact flags (E, 4) float, canonical FR, FL, RR, RL."""
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids_sensor]
        return (forces.norm(dim=-1).max(dim=1)[0] > self.cfg.contact_force_threshold).float()

    def _feet_pos_root(self) -> torch.Tensor:
        """Foot link positions in the root frame (E, 4, 3), canonical order."""
        data = self._robot.data
        rel = data.body_pos_w[:, self._feet_ids_body] - data.root_pos_w.unsqueeze(1)
        quat = data.root_quat_w.unsqueeze(1).expand(-1, 4, -1)
        return quat_apply_inverse(quat, rel)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        data = self._robot.data
        quat = data.root_quat_w
        dof_pos_c = (data.joint_pos - data.default_joint_pos)[:, self._perm_s2c]
        dof_vel_c = data.joint_vel[:, self._perm_s2c]

        ref_targets = []
        for k in self.cfg.ref_target_steps:
            ref_k = self._ref_frame(dt_ahead=k * self.step_dt)
            ref_targets.append(ref_k["dof_pos"][:, self._perm_s2c])
            # reference root velocities in the robot's CURRENT base frame
            ref_targets.append(quat_apply_inverse(quat, ref_k["root_lin_vel"]))
            ref_targets.append(quat_apply_inverse(quat, ref_k["root_ang_vel"]))

        # heading error: ref-relative yaw (stage3) — the only world-frame-
        # derived scalar in the actor obs; hardware-realizable via IMU yaw.
        # NO phase input (stage5): the preview is the only clip conditioning.
        dyaw = _quat_yaw(self._ref_frame()["root_rot"]) - _quat_yaw(quat)
        obs = torch.cat(
            [
                data.projected_gravity_b,
                data.root_ang_vel_b,
                dof_pos_c,
                dof_vel_c,
                self._actions,
                *ref_targets,
                torch.sin(dyaw).unsqueeze(-1),
                torch.cos(dyaw).unsqueeze(-1),
            ],
            dim=-1,
        )
        # privileged critic extras: base lin vel, foot contacts, true root height
        critic = torch.cat(
            [obs, data.root_lin_vel_b, self._foot_contacts(), data.root_pos_w[:, 2:3]],
            dim=-1,
        )
        return {"policy": obs, "critic": critic}

    # -- rewards -------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        data = self._robot.data
        ref = self._ref_frame()

        joint_pos_err = (data.joint_pos - ref["dof_pos"]).square().sum(dim=-1)
        joint_vel_err = (data.joint_vel - ref["dof_vel"]).square().sum(dim=-1)
        ori_err = quat_error_magnitude(data.root_quat_w, ref["root_rot"])
        lin_vel_err = (data.root_lin_vel_w - ref["root_lin_vel"]).square().sum(dim=-1)
        ang_vel_err = (data.root_ang_vel_w - ref["root_ang_vel"]).square().sum(dim=-1)
        root_pos_err = (data.root_pos_w - self._ref_root_pos_w(ref)).square().sum(dim=-1)
        if "feet_pos_root" in ref:
            ee_err = (self._feet_pos_root() - ref["feet_pos_root"]).square().sum(dim=(-1, -2))
        else:  # replay/diagnostic modes may run without the feet caches
            ee_err = torch.zeros_like(joint_pos_err)
        contact_match = (self._foot_contacts() == ref["foot_contacts"]).float().mean(dim=-1)
        action_rate = (self._actions - self._previous_actions).square().sum(dim=-1)
        torque = data.applied_torque.square().sum(dim=-1)

        rewards = {
            "joint_pos_tracking": cfg.rew_joint_pos_w * torch.exp(-cfg.rew_joint_pos_k * joint_pos_err),
            "joint_vel_tracking": cfg.rew_joint_vel_w * torch.exp(-cfg.rew_joint_vel_k * joint_vel_err),
            "ee_tracking": cfg.rew_ee_w * torch.exp(-cfg.rew_ee_k * ee_err),
            "root_pose_tracking": cfg.rew_root_pose_w
            * torch.exp(-cfg.rew_root_pose_kp * root_pos_err - cfg.rew_root_pose_ko * ori_err.square()),
            "root_vel_tracking": cfg.rew_root_vel_w
            * torch.exp(-cfg.rew_root_vel_kl * lin_vel_err - cfg.rew_root_vel_ka * ang_vel_err),
            "contact_match": cfg.rew_contact_match_w * contact_match,
            "action_rate": cfg.rew_action_rate_w * action_rate,
            "torque": cfg.rew_torque_w * torque,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # -- termination ---------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._robot.data
        ref = self._ref_frame()

        # clip end (acyclic) is a truncation, not a failure
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out |= self._motion_lib.clip_done(self._clip_idx, self._ref_t)

        if self.cfg.kinematic_replay:
            return torch.zeros_like(time_out), time_out

        pos_err = (data.root_pos_w - self._ref_root_pos_w(ref)).norm(dim=-1)
        ori_err = quat_error_magnitude(data.root_quat_w, ref["root_rot"])
        joint_err = (data.joint_pos - ref["dof_pos"]).abs().mean(dim=-1)
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._base_id_sensor]
        base_contact = (forces.norm(dim=-1).max(dim=1)[0] > self.cfg.contact_force_threshold).any(dim=1)
        if not self.cfg.term_base_contact:
            base_contact = torch.zeros_like(base_contact)

        self._term_causes = {
            "root_pos": pos_err > self.cfg.term_root_pos_err,
            "root_ori": ori_err > self.cfg.term_root_ori_err,
            "joint_err": joint_err > self.cfg.term_joint_err,
            "base_contact": base_contact,
        }
        died = torch.stack(list(self._term_causes.values())).any(dim=0)
        return died, time_out

    # -- reset ---------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        if self._ghost is not None:
            self._ghost.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not (self.cfg.kinematic_replay or self.cfg.rsi_start_at_zero):
            # spread resets in time to avoid synchronized timeout spikes
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # reference state initialization: random (clip, frame), or clip start
        # in replay mode
        if self.cfg.kinematic_replay:
            if self._replay_clip_idx is not None:
                self._clip_idx[env_ids] = self._replay_clip_idx
            else:
                self._clip_idx[env_ids] = env_ids.to(self.device) % self._motion_lib.num_clips
            self._ref_t[env_ids] = 0.0
        else:
            clip_idx, t0 = self._motion_lib.sample(len(env_ids))
            if self._replay_clip_idx is not None:
                # pinned clip (play.py --motion): resample t0 within its duration
                clip_idx = torch.full_like(clip_idx, self._replay_clip_idx)
                t0 = torch.rand_like(t0) * self._motion_lib.durations[self._replay_clip_idx]
            self._clip_idx[env_ids] = clip_idx
            self._ref_t[env_ids] = 0.0 if self.cfg.rsi_start_at_zero else t0

        ref = self._motion_lib.get_frame(self._clip_idx[env_ids], self._ref_t[env_ids])
        root_pos = ref["root_pos"] + self._terrain.env_origins[env_ids]
        joint_pos = ref["dof_pos"]
        if not self.cfg.kinematic_replay:
            root_pos = root_pos.clone()
            root_pos[:, 2] += sample_uniform(0.0, self.cfg.rsi_root_z_noise, (len(env_ids),), self.device)
            joint_pos = joint_pos + sample_uniform(
                -self.cfg.rsi_joint_pos_noise, self.cfg.rsi_joint_pos_noise, joint_pos.shape, self.device
            )
            limits = self._robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

        self._robot.write_root_pose_to_sim(torch.cat([root_pos, ref["root_rot"]], dim=-1), env_ids)
        self._robot.write_root_velocity_to_sim(torch.cat([ref["root_lin_vel"], ref["root_ang_vel"]], dim=-1), env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, ref["dof_vel"], None, env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        steps = self._steps_since_reset[env_ids].clamp(min=1.0)
        extras["Action/abs_mean"] = (self._action_abs_sum[env_ids] / steps).mean()
        extras["Action/abs_max"] = self._action_abs_max[env_ids].max()
        extras["Action/clamp_frac"] = (self._clamp_frac_sum[env_ids] / steps).mean()
        self._steps_since_reset[env_ids] = 0.0
        self._action_abs_sum[env_ids] = 0.0
        self._action_abs_max[env_ids] = 0.0
        self._clamp_frac_sum[env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        if hasattr(self, "_term_causes"):
            for cause, mask in self._term_causes.items():
                extras[f"Episode_Termination/{cause}"] = torch.count_nonzero(mask[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
