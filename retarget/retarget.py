"""Retarget v0: dog keypoints -> Go2 root pose + 12 joint angles per frame.

Positions are retargeted, not joint angles (dog and Go2 topologies don't
correspond joint-by-joint):

  1. Trunk frame: a single rigid frame fitted to the dog's hip+shoulder
     segment per frame — origin midway between pelvis and chest, x along the
     spine, y from the left-right leg-root geometry (this captures roll
     without trusting the mocap pelvis orientation).
  2. Uniform scale = Go2 standing height / mean dog leg-root height.
  3. Foot targets: each dog toe expressed in the trunk frame, scaled, and
     re-anchored from the dog's mean leg-mount point to the matching Go2 leg
     plane origin (hip offset + lateral thigh offset).
  4. Analytic IK per leg (retarget/ik.py), clamped to joint limits with the
     clamp rate reported.

Foot contacts are *initially* detected on the source toes (height +
horizontal-speed thresholds) — good enough to seed ground alignment, but
the speed gate erases fast-gait stances, so post-processing
(retarget/postprocess.py: contact refinement, smoothing, ground alignment,
foot-skate removal, then contact relabel from the robot's realized feet
and a dof-velocity despike) replaces them and runs by default — pass
--raw to see the phase-3 output with its skate and jitter.

Usage:
    python retarget/retarget.py data/processed/D1_007_KAN01_001.npz
    python retarget/retarget.py data/D1_007_KAN01_001.bvh   # parses first

Writes motions/<clip>.pkl in the §7 output format (50 Hz, root_rot xyzw,
dof order FR, FL, RR, RL x hip/thigh/calf).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, sys.path[0] is retarget/ itself, and this file shadows the
# `retarget` package — swap it for the repo root so absolute imports work.
if sys.path and sys.path[0] == str(REPO_ROOT / "retarget"):
    sys.path[0] = str(REPO_ROOT)
else:
    sys.path.insert(0, str(REPO_ROOT))
from retarget import ik
from retarget.postprocess import postprocess
from retarget.skeleton import extract_keypoints, parse_bvh
MOTIONS_DIR = REPO_ROOT / "motions"

GO2_STANDING_HEIGHT = 0.27  # home keyframe root z, from the MJCF

# Go2 leg-plane origin per leg (hip joint + lateral thigh offset), base frame.
GO2_LEG_MOUNTS = ik.HIP_OFFSETS + np.stack(
    [np.zeros(4), ik.HIP_LATERAL, np.zeros(4)], axis=-1
)

# Source-side contact detection (dog scale, meters).
CONTACT_HEIGHT_M = 0.03
CONTACT_SPEED_MPS = 0.25


def load_keypoints(path):
    path = Path(path)
    if path.suffix == ".bvh":
        return extract_keypoints(parse_bvh(path))
    kp = dict(np.load(path))
    kp["fps"] = float(kp["fps"])
    kp["num_frames"] = int(kp["num_frames"])
    kp["source"] = str(kp["source"])
    return kp


def fit_trunk_frame(kp):
    """Rigid trunk frame per frame: origin (N,3) and Rotation (N,).

    x: pelvis -> chest (spine bending projected out by using the endpoints).
    y: left minus right leg roots, averaged front/rear (leg order FR FL RR RL).
    z: x cross y, re-orthogonalized.
    """
    origin = 0.5 * (kp["root_pos"] + kp["chest_pos"])
    x = _normalize(kp["chest_pos"] - kp["root_pos"])
    lr = kp["leg_root_pos"]
    y_seed = _normalize((lr[:, 1] - lr[:, 0]) + (lr[:, 3] - lr[:, 2]))
    z = _normalize(np.cross(x, y_seed))
    y = np.cross(z, x)
    rot = Rotation.from_matrix(np.stack([x, y, z], axis=-1))

    # Remove the constant anatomical tilt: the dog's withers sit higher than
    # its hip balls, so the hips->chest axis carries a standing pitch bias
    # that the equal-legged Go2 should not inherit; dynamic pitch/roll stays.
    # The bias must be measured STANDING. The old clip-wide median broke on
    # dog_3, which sits for more than half the clip: the median absorbed the
    # sit pitch (+20 deg) and every standing frame inherited it nose-down,
    # sweeping the rear feet into the air behind the robot. Standing frames =
    # all toes near the floor (excludes dog_4's flight, where the max-height
    # pose is mid-air and stretched) with the pelvis in the top height
    # quartile of those (excludes sit/crouch). Falls back to the plain median
    # when the mask starves (a clip that never stands).
    yaw = np.arctan2(x[:, 1], x[:, 0])
    tilt = (Rotation.from_euler("z", -yaw[:, None]) * rot).as_euler("ZYX")
    grounded = (kp["toe_pos"][..., 2] < 2 * CONTACT_HEIGHT_M).all(axis=1)
    if grounded.sum() >= 5:
        zr = kp["root_pos"][:, 2]
        standing = grounded & (zr >= np.quantile(zr[grounded], 0.75))
    else:
        standing = np.ones(len(x), bool)
    bias = Rotation.from_euler("YX", np.median(tilt[standing, 1:], axis=0))
    return origin, rot * bias.inv()


def detect_contacts(kp):
    """(N, 4) bool stance mask from source toe height + horizontal speed."""
    toe = kp["toe_pos"]
    speed = np.linalg.norm(np.gradient(toe[..., :2], axis=0), axis=-1) * kp["fps"]
    return (toe[..., 2] < CONTACT_HEIGHT_M) & (speed < CONTACT_SPEED_MPS)


def retarget_clip(kp, scale=None):
    """Dog keypoints -> Go2 motion dict at the source fps (§7 minus resample)."""
    origin, trunk_rot = fit_trunk_frame(kp)

    if scale is None:
        # Median, not mean: clips with crouch/lying segments (e.g. D1_009)
        # drag the mean leg-root height down and inflate the scale until the
        # Go2 legs can't reach the pinned stance feet.
        scale = GO2_STANDING_HEIGHT / np.median(kp["leg_root_pos"][..., 2])

    # Root: scaled trunk frame, xy shifted so the clip starts at the origin.
    root_pos = origin * scale
    root_pos[:, :2] -= root_pos[0, :2]

    # Foot targets in the Go2 base frame: dog toes in the trunk frame, scaled,
    # re-anchored from the dog's mean leg mount to the Go2 leg mount.
    trunk_inv = trunk_rot.inv()
    toe_local = _to_local(kp["toe_pos"], origin, trunk_inv)
    mount_local = _to_local(kp["leg_root_pos"], origin, trunk_inv)
    dog_mounts = mount_local.mean(axis=0)  # (4, 3), constant over the clip
    foot_targets = GO2_LEG_MOUNTS + scale * (toe_local - dog_mounts)

    dof_pos = ik.ik(foot_targets)
    dof_pos, violated = ik.clamp_to_limits(dof_pos)

    return {
        "fps": kp["fps"],
        "robot_type": "unitree_go2",
        "num_frames": kp["num_frames"],
        "root_pos": root_pos,
        "root_rot": trunk_rot.as_quat(),  # xyzw
        "dof_pos": dof_pos,
        "foot_contacts": detect_contacts(kp),
        "source": kp["source"],
    }, {"scale": scale, "clamp_rate": violated.mean(), "violated": violated,
        "foot_targets": foot_targets}


def resample(motion, target_fps=50.0):
    """Resample a motion dict to target_fps (linear; slerp for the root quat)."""
    n = motion["num_frames"]
    t_src = np.arange(n) / motion["fps"]
    t_out = np.arange(0.0, t_src[-1], 1.0 / target_fps)

    def lerp(x):
        flat = x.reshape(n, -1)
        out = np.stack([np.interp(t_out, t_src, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
        return out.reshape(len(t_out), *x.shape[1:])

    out = dict(motion)
    out["fps"] = target_fps
    out["num_frames"] = len(t_out)
    out["root_pos"] = lerp(motion["root_pos"])
    out["root_rot"] = Slerp(t_src, Rotation.from_quat(motion["root_rot"]))(t_out).as_quat()
    out["dof_pos"] = lerp(motion["dof_pos"])
    out["foot_contacts"] = lerp(motion["foot_contacts"].astype(np.float64)) > 0.5
    return out


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _to_local(points, origin, rot_inv):
    """World (N, 4, 3) points -> trunk-local, one leg at a time (scipy
    Rotation.apply pairs the i-th rotation with the i-th vector only)."""
    return np.stack(
        [rot_inv.apply(points[:, leg] - origin) for leg in range(points.shape[1])],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", type=Path,
                        help="processed .npz (or raw .bvh) clips to retarget")
    parser.add_argument("--fps", type=float, default=50.0, help="output fps")
    parser.add_argument("--scale", type=float, default=None,
                        help="override the automatic dog->Go2 scale factor")
    parser.add_argument("--raw", action="store_true",
                        help="skip post-processing (phase-3 output: skate + jitter)")
    args = parser.parse_args()

    MOTIONS_DIR.mkdir(exist_ok=True)
    for path in args.clips:
        kp = load_keypoints(path)
        motion, info = retarget_clip(kp, scale=args.scale)
        if not args.raw:
            motion, report = postprocess(motion, info["foot_targets"])
            info["clamp_rate"] = report["clamp_rate"]
            info["violated"] = report["violated"]
        motion = resample(motion, args.fps)
        out = MOTIONS_DIR / f"{motion['source']}.pkl"
        with open(out, "wb") as f:
            pickle.dump(motion, f)
        stance = motion["foot_contacts"].mean()
        print(f"wrote {out.relative_to(REPO_ROOT)}: {motion['num_frames']} frames "
              f"@ {motion['fps']:.0f} fps, scale {info['scale']:.3f}, "
              f"clamp rate {100 * info['clamp_rate']:.2f}%, "
              f"stance fraction {stance:.2f}")
        if not args.raw:
            print(f"  post-process: stance-foot skate "
                  f"{report['skate_before']:.3f} -> {report['skate_after']:.3f} m/s, "
                  f"ground offset {1000 * report['ground_offset']:+.0f} mm")
            print(f"  feasibility: contacts {report['contact_fraction_src']:.2f} -> "
                  f"{report['contact_fraction']:.2f} (relabeled from realized feet, "
                  f"{100 * report['contact_changed']:.1f}% foot-frames changed), "
                  f"dof-vel peak {report['dof_vel_peak_raw']:.0f} -> "
                  f"{report['dof_vel_peak']:.0f} rad/s "
                  f"(despike max dq {report['despike_max_dq']:.3f} rad)")
        if info["clamp_rate"] > 0.03:
            per_joint = info["violated"].mean(axis=0)
            worst = np.argsort(per_joint)[::-1][:3]
            names = [f"{ik.LEG_ORDER[i // 3]}_{['hip', 'thigh', 'calf'][i % 3]}"
                     f"={100 * per_joint[i]:.1f}%" for i in worst]
            print(f"  WARNING: clamp rate > 3% — scaling/offsets likely wrong "
                  f"(worst joints: {', '.join(names)})")


if __name__ == "__main__":
    main()
