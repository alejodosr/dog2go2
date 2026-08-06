"""Render the AniMer SMAL *mesh* over the source video, frame for frame.

The 10-point skeleton (viz_debug) shows where the fit's joints are; this
shows the whole body surface, which is what AniMer actually optimizes. When
a fit fails, the two fail differently — a skeleton misses the rear legs
quietly, a mesh visibly not covering the animal is unambiguous evidence the
failure is stage 1 and nothing downstream.

Rebuilds vertices exactly the way animer_infer's FK re-run does (frozen
betas + smoothed pose, SMALLayer.forward) and projects through the same
weak-perspective camera (cam_t, focal_full, principal point at centre), so
what is drawn is what every later stage consumed — not a fresh inference.

    PYOPENGL_PLATFORM=egl PYTHONPATH=$REPO $PY_CAPTURE -m capture.viz_smal \
        --infer $WORK/<clip>_animer.npz --video <video> --out media/smal_<clip>.mp4
"""
from pathlib import Path
import argparse
import pickle

import numpy as np

from capture import paths
from capture.viz_world import open_writer


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True, help="Phase B npz")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--smal", default=None,
                   help="SMAL model pkl (default: capture.paths.SMAL_MODEL)")
    args = p.parse_args()

    import cv2
    import torch
    import trimesh
    import pyrender
    from amr.models.smal_warapper import SMALLayer
    from amr.utils.renderer import create_raymond_lights

    b = np.load(args.infer, allow_pickle=True)
    N = int(b["num_frames"])
    W, H = [int(v) for v in b["img_size"]]
    focal = float(b["focal_full"])
    fps = float(b["fps"])
    valid = b["valid"]
    cam_t = b["cam_t"]

    smal_path = Path(args.smal) if args.smal else paths.resolve(
        paths.SMAL_MODEL, "SMAL model file (data/smal/)")
    with open(smal_path, "rb") as f:
        smal_cfg = pickle.load(f, encoding="latin1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smal = SMALLayer(**smal_cfg).to(device).eval()
    faces = smal.faces.cpu().numpy()

    verts = np.empty((N, 3889, 3), dtype=np.float32)
    with torch.no_grad():
        bt = torch.from_numpy(b["betas_frozen"]).float()[None].to(device)
        for s in range(0, N, 64):
            e = min(N, s + 64)
            o = SMALLayer.forward(
                smal, betas=bt.expand(e - s, -1),
                global_orient=torch.from_numpy(b["global_orient"][s:e]).float().to(device),
                pose=torch.from_numpy(b["pose"][s:e]).float().to(device),
                pose2rot=False)
            verts[s:e] = o.vertices.cpu().numpy()

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H,
                                          point_size=1.0)
    flip_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    lights = create_raymond_lights()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    writer = None
    idx, written = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx >= N or idx % args.stride:
            continue
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        if valid[idx]:
            # amr.utils.renderer.render_rgba's convention: mesh flipped 180
            # about x, camera at cam_t with x negated, same intrinsics the
            # projection used
            mesh = trimesh.Trimesh(verts[idx], faces.copy(), process=False)
            mesh.apply_transform(flip_x)
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0],
                                   ambient_light=(0.3, 0.3, 0.3))
            scene.add(pyrender.Mesh.from_trimesh(
                mesh, material=pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0, alphaMode="OPAQUE",
                    baseColorFactor=(0.4, 0.75, 1.0, 1.0))), "mesh")
            pose = np.eye(4)
            t = cam_t[idx].copy()
            t[0] *= -1.0
            pose[:3, 3] = t
            scene.add_node(pyrender.Node(
                camera=pyrender.IntrinsicsCamera(fx=focal, fy=focal,
                                                 cx=W / 2.0, cy=H / 2.0,
                                                 zfar=1e12),
                matrix=pose))
            for ln in lights:
                scene.add_node(ln)
            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            a = args.alpha * (rgba[..., 3:4].astype(np.float32) / 255.0)
            rgb_bgr = rgba[..., 2::-1].astype(np.float32)
            frame = (frame * (1 - a) + rgb_bgr * a).astype(np.uint8)
        else:
            cv2.putText(frame, "no detection", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"frame {idx}", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if writer is None:
            writer = open_writer(args.out, W, H, fps / args.stride)
        writer.stdin.write(np.ascontiguousarray(rgb).tobytes())
        written += 1
        if written % 60 == 0:
            print(f"  {written} frames", flush=True)

    cap.release()
    renderer.delete()
    if writer:
        writer.stdin.close()
        writer.wait()
    print(f"wrote {args.out} ({written} frames @ {fps / args.stride:.1f} fps)")


if __name__ == "__main__":
    main()
