"""Phase E — emit the Milestone 1 npz contract from a placed world skeleton.

This is the whole point of the milestone: a SECOND producer of the same npz
that retarget/parse_mocap.py emits, so everything downstream in this repo --
analytic IK, contact refinement, smoothing, ground alignment, foot-skate
removal, clamp reporting, batch export -- consumes it unchanged.

Schema, from retarget/skeleton.py::extract_keypoints. Nine keys, no more:
    fps, num_frames, source
    root_pos (N,3)      root_rot_xyzw (N,4)
    chest_pos (N,3)     chest_rot_xyzw (N,4)
    toe_pos (N,4,3)     leg_root_pos (N,4,3)
Metres, Z-up, ground at z~=0, leg order FR FL RR RL, quaternions xyzw.

Two notes on what downstream actually reads:

* `root_rot_xyzw` and `chest_rot_xyzw` are NEVER read by retarget.py -- trunk
  attitude is rebuilt from positions in fit_trunk_frame. Only the contract
  validator looks at them, and only to check unit norm. They are emitted for
  schema compliance; effort spent on them is wasted.

* The npz must be near-true-metric, NOT in arbitrary "Go2 units". The pose
  path is scale-invariant (retarget.py:118 divides by median leg-root height),
  but contacts are not: detect_contacts thresholds the SOURCE npz at an
  absolute 0.03 m and 0.25 m/s, so scaling the whole file by k silently
  reinterprets those as 0.03/k metres of real dog.

Diagnostics go to a sidecar json, never into the npz -- extra keys would be
ignored by retarget.py but they weaken the contract.
"""
from pathlib import Path
import argparse
import json

import numpy as np


def trunk_quaternion(root, chest, mounts):
    """Per-frame trunk rotation as xyzw, mirroring retarget.fit_trunk_frame."""
    from scipy.spatial.transform import Rotation

    def norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    x = norm(chest - root)
    y_seed = norm((mounts[:, 1] - mounts[:, 0]) + (mounts[:, 3] - mounts[:, 2]))
    z = norm(np.cross(x, y_seed))
    y = np.cross(z, x)
    return Rotation.from_matrix(np.stack([x, y, z], axis=-1)).as_quat()


def validate(kp):
    """The checks the retargeter's own contract validator applies."""
    problems = []
    n = int(kp["num_frames"])
    shapes = {"root_pos": (n, 3), "root_rot_xyzw": (n, 4),
              "chest_pos": (n, 3), "chest_rot_xyzw": (n, 4),
              "toe_pos": (n, 4, 3), "leg_root_pos": (n, 4, 3)}
    for k, s in shapes.items():
        a = np.asarray(kp[k])
        if a.shape != s:
            problems.append(f"{k} is {a.shape}, expected {s}")
        elif not np.isfinite(a).all():
            problems.append(f"{k} contains non-finite values")
    for k in ("root_rot_xyzw", "chest_rot_xyzw"):
        nrm = np.linalg.norm(np.asarray(kp[k]), axis=-1)
        if not np.allclose(nrm, 1.0, atol=1e-6):
            problems.append(f"{k} is not unit norm (max dev "
                            f"{np.abs(nrm-1).max():.2e})")

    trunk = np.median(np.linalg.norm(kp["chest_pos"] - kp["root_pos"], axis=-1))
    if not (0.15 < trunk < 1.5):
        problems.append(f"median trunk length {trunk:.3f} m outside (0.15, 1.5)"
                        " -- units are probably not metres")
    toe_z = kp["toe_pos"][..., 2]
    if not (-0.15 < toe_z.min() < 0.4):
        problems.append(f"min toe z {toe_z.min():.3f} outside (-0.15, 0.4) -- "
                        "ground plane is not at z~=0")
    if np.median(kp["root_pos"][:, 2]) <= np.median(toe_z):
        problems.append("median root z is not above median toe z -- not Z-up")

    # leg order FR FL RR RL, checked in the trunk frame
    x = kp["chest_pos"] - kp["root_pos"]
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    lr = kp["leg_root_pos"]
    y = (lr[:, 1] - lr[:, 0]) + (lr[:, 3] - lr[:, 2])
    y /= np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
    lat = np.einsum("nkc,nc->nk", lr - kp["root_pos"][:, None], y)
    if not (np.median(lat[:, 1]) > 0 and np.median(lat[:, 3]) > 0):
        problems.append("FL/RL mounts are not on the +y side -- FR/FL swapped?")
    if not (np.median(lat[:, 0]) < 0 and np.median(lat[:, 2]) < 0):
        problems.append("FR/RR mounts are not on the -y side -- FR/FL swapped?")
    fore = np.einsum("nkc,nc->nk", lr - kp["root_pos"][:, None], x)
    if not (np.median(fore[:, :2]) > np.median(fore[:, 2:])):
        problems.append("front mounts are not ahead of rear -- front/rear swapped?")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", required=True, help="Phase D npz")
    p.add_argument("--out", required=True, help="data/processed/<clip>.npz")
    p.add_argument("--source", default=None, help="clip name inside the npz")
    args = p.parse_args()

    w = np.load(args.world, allow_pickle=True)
    world = w["world"]
    n = len(world)
    source = args.source or str(w["source"])

    root, chest = world[:, 0], world[:, 1]
    mounts, toes = world[:, 2:6], world[:, 6:10]

    # Put the ground at z = 0 using the stance feet, and start the clip at the
    # xy origin the way parse_mocap's downstream expects.
    ct = w["contacts"]
    floor = np.median(toes[..., 2][ct]) if ct.any() else np.median(toes[..., 2])
    for a in (root, chest, mounts, toes):
        a[..., 2] -= floor

    q = trunk_quaternion(root, chest, mounts)
    kp = {
        "fps": float(w["fps"]),
        "num_frames": int(n),
        "source": source,
        "root_pos": root.astype(np.float64),
        "root_rot_xyzw": q.astype(np.float64),
        "chest_pos": chest.astype(np.float64),
        "chest_rot_xyzw": q.astype(np.float64),
        "toe_pos": toes.astype(np.float64),
        "leg_root_pos": mounts.astype(np.float64),
    }

    problems = validate(kp)
    print(f"clip {source}: {n} frames @ {kp['fps']:.3f} fps")
    print(f"  median trunk length   "
          f"{np.median(np.linalg.norm(chest-root, axis=-1)):.3f} m")
    print(f"  median leg-root z     {np.median(mounts[...,2]):.3f} m"
          f"   -> Go2 scale will be {0.27/np.median(mounts[...,2]):.3f}")
    print(f"  toe z range           {toes[...,2].min():.3f} .. "
          f"{toes[...,2].max():.3f} m")
    print(f"  median root z         {np.median(root[:,2]):.3f} m")

    if problems:
        print("\nCONTRACT VIOLATIONS")
        for m in problems:
            print("  -", m)
        raise SystemExit(1)
    print("\ncontract checks passed")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **kp)
    print(f"wrote {out}")

    side = out.with_suffix(".video.json")
    side.write_text(json.dumps({
        "source": source, "producer": "capture/parse_video.py",
        "fps": kp["fps"], "num_frames": n,
        "camera_height_m": float(w["camera_pos"][2]),
        "ortho_resid": float(w["ortho_resid"]),
        "metres_per_unit": float(w["metres_per_unit"]),
        "anchored_fraction": float(np.mean(w["anchored"])),
        "stance_fraction": float(ct.mean()),
    }, indent=2))
    print(f"wrote {side}  (diagnostics stay out of the npz)")


if __name__ == "__main__":
    main()
