"""Kinematic playback of a retargeted motion on the Go2 in MuJoCo.

Sets qpos directly every frame (root pose + 12 dofs) — mj_forward only, the
dynamics are never stepped. Renders an mp4 by default; --interactive opens
the passive viewer instead.

Viewer keys: Space = pause/resume, . = single-step while paused,
[ / ] = half / double playback speed.

Usage:
    MUJOCO_GL=egl python viz/playback.py motions/D1_007_KAN01_001.pkl
    python viz/playback.py motions/D1_007_KAN01_001.pkl --interactive
"""

import argparse
import pickle
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from retarget.ik import LEG_ORDER  # noqa: E402
from retarget.skeleton import quat_xyzw_to_wxyz  # noqa: E402

GO2_SCENE_XML = REPO_ROOT / "assets" / "unitree_go2" / "scene.xml"


def load_model(width=1920, height=1080):
    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE_XML))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    return model, mujoco.MjData(model)


def qpos_trajectory(model, motion):
    """(N, nq) qpos array: free-joint root pose + canonical dofs scattered
    through the joint addresses (the MJCF declares legs in FL FR RL RR order,
    not the canonical FR FL RR RL — never block-copy dof vectors)."""
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    adr = np.array([
        model.joint(f"{leg}_{suffix}_joint").qposadr[0]
        for leg in LEG_ORDER
        for suffix in ["hip", "thigh", "calf"]
    ])
    n = motion["num_frames"]
    qpos = np.zeros((n, model.nq))
    qpos[:, :3] = motion["root_pos"]
    qpos[:, 3:7] = quat_xyzw_to_wxyz(motion["root_rot"])
    qpos[:, adr] = motion["dof_pos"]
    return qpos


def tracking_camera(model):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("base").id
    cam.distance = 1.6
    cam.elevation = -15
    cam.azimuth = 125
    return cam


def matched_camera_model(world_npz, calib_json, motion, height):
    """Model with a fixed camera at the SOURCE video's viewpoint.

    The capture pipeline solves the real camera (world_place: R_cw and
    camera_pos, in dog-world metres) and the calibration knows its focal
    length. The retargeter maps dog world -> pkl frame by a uniform scale
    about the origin plus an xy shift (fit in retarget_clip; no rotation),
    and uniform scaling preserves the projected image as long as the camera
    position is scaled the same way — so placing the camera at
    scale * camera_pos - shift with the SAME orientation reproduces the
    source framing exactly. Scale and shift are recovered from the data
    rather than re-deriving retarget's internals: scale from the z-median
    ratio (robust to the 50 Hz resample), shift from frame 0.

    Returns (model, data, camera_name, render_width).
    """
    from scipy.spatial.transform import Rotation

    from viz.source_cam import source_camera_pose

    campos, R_mj, _, fovy, (W_, H_) = source_camera_pose(world_npz, calib_json, motion)
    qx, qy, qz, qw = Rotation.from_matrix(R_mj).as_quat()
    width = int(round(height * W_ / H_)) // 2 * 2

    spec = mujoco.MjSpec.from_file(str(GO2_SCENE_XML))
    cam = spec.worldbody.add_camera()
    cam.name = "source_cam"
    cam.pos = campos
    cam.quat = [qw, qx, qy, qz]
    cam.fovy = fovy
    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    print(f"source camera: pos ({campos[0]:.2f}, {campos[1]:.2f}, {campos[2]:.2f}) "
          f"fovy {fovy:.1f} deg  render {width}x{height}")
    return model, mujoco.MjData(model), "source_cam", width


def render_video(model, data, motion, out_path, width=1920, height=1080, cam=None):
    qpos = qpos_trajectory(model, motion)
    renderer = mujoco.Renderer(model, height=height, width=width)
    if cam is None:
        cam = tracking_camera(model)
    with imageio.get_writer(out_path, fps=motion["fps"]) as writer:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            writer.append_data(renderer.render())
    renderer.close()
    print(f"wrote {out_path} ({len(qpos)} frames @ {motion['fps']:.0f} fps)")


def run_interactive(model, data, motion):
    import time

    import mujoco.viewer

    qpos = qpos_trajectory(model, motion)
    state = {"paused": False, "step": False, "speed": 1.0}

    def key_callback(keycode):
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
        elif keycode == ord("."):
            state["step"] = True
        elif keycode == ord("["):
            state["speed"] = max(state["speed"] / 2, 0.125)
        elif keycode == ord("]"):
            state["speed"] = min(state["speed"] * 2, 8.0)

    print("Space = pause, . = single-step, [ / ] = slower / faster")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        i = 0
        while viewer.is_running():
            if not state["paused"] or state["step"]:
                state["step"] = False
                data.qpos[:] = qpos[i % len(qpos)]
                mujoco.mj_forward(model, data)
                viewer.sync()
                i += 1
            time.sleep(1.0 / (motion["fps"] * state["speed"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="retargeted motion .pkl")
    parser.add_argument("--interactive", action="store_true",
                        help="loop in the live passive viewer instead of rendering an mp4")
    parser.add_argument("--out", type=Path, default=None,
                        help="output mp4 path (default media/go2_<clip>.mp4)")
    parser.add_argument("--camera-world", type=Path, default=None,
                        help="render from the SOURCE video's solved camera: the "
                             "clip's <clip>_world.npz (needs --camera-calib)")
    parser.add_argument("--camera-calib", type=Path, default=None,
                        help="the clip's depth-calibration json (focal_px, img_size)")
    args = parser.parse_args()

    with open(args.motion, "rb") as f:
        motion = pickle.load(f)
    assert motion["robot_type"] == "unitree_go2"
    print(f"{motion['source']}: {motion['num_frames']} frames @ {motion['fps']:.0f} fps")

    if bool(args.camera_world) != bool(args.camera_calib):
        parser.error("--camera-world and --camera-calib go together")
    if args.camera_world and args.interactive:
        parser.error("--camera-world only applies to rendering")

    if args.camera_world:
        model, data, cam_name, width = matched_camera_model(
            args.camera_world, args.camera_calib, motion, height=1080)
        out = args.out or REPO_ROOT / "media" / f"go2_{motion['source']}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        render_video(model, data, motion, out, width=width, cam=cam_name)
        return

    model, data = load_model()
    if args.interactive:
        run_interactive(model, data, motion)
    else:
        out = args.out or REPO_ROOT / "media" / f"go2_{motion['source']}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        render_video(model, data, motion, out)


if __name__ == "__main__":
    main()
