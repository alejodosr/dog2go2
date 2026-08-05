"""Can a metric depth model replace the clicked ground-plane calibration?

The clicked calibration fixes the plane's SHAPE well but its absolute metres
are provisional — they came from counting floor tiles and assuming a tile
size, and its own note says so. A metric depth model outputs METRES directly,
so if a plane fitted to its floor pixels reproduces the clicked calibration's
camera height and orientation, the clicking step can go, and with it the
tile-size assumption.

Test, per clip:
  1. run ZoeDepth on a spread of frames
  2. back-project the lower image region to a metric point cloud using the
     CALIBRATED focal (not the depth model's assumed one)
  3. robustly fit a plane (RANSAC) to the floor points
  4. compare camera height and plane normal with the clicked calibration

The comparison is only meaningful where the clicked calibration is itself
trustworthy, so the two corridor clips (which share a rig and agree on camera
height to 1 cm) are the reference; dog_2's room is the harder case.

Note on what this can and cannot settle: agreement would show the two agree,
not that either is RIGHT — both could share a bias. An absolute check needs
an object of known size in the scene (dog_1's floor is covered in pens and
markers, which is the cheapest true ruler available).
"""
from pathlib import Path
import argparse
import json

import numpy as np

from capture import paths



def sample_frames(video, n, W, H):
    import cv2
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = np.linspace(0, max(total - 2, 0), n).astype(int)
    out, got = [], []
    for i in range(total):
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            out.append(cv2.cvtColor(cv2.resize(f, (W, H),
                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            got.append(i)
    cap.release()
    return np.array(out), got


def fit_plane_ransac(P, iters=600, tol=0.04, rng=None):
    """Plane through a metric point cloud. Returns (normal, d, inlier mask)."""
    rng = rng or np.random.default_rng(0)
    best = (None, None, np.zeros(len(P), bool))
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        d = -n @ s[0]
        inl = np.abs(P @ n + d) < tol
        if inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    if n is None:
        return None, None, None
    Q = P[inl]                                   # refine on inliers
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c)
    n = Vt[-1] / np.linalg.norm(Vt[-1])
    return n, float(-n @ c), inl


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--floor-frac", type=float, default=0.45,
                   help="use the bottom fraction of the image as floor candidates")
    p.add_argument("--stride", type=int, default=8, help="pixel subsampling")
    p.add_argument("--cache", default=paths.HF_CACHE,
                   help="HuggingFace cache for the depth weights (default $HF_HOME)")
    args = p.parse_args()

    import os
    os.environ["TORCH_HOME"] = args.cache
    import torch
    from capture.world_place import camera_to_world

    cal = json.loads(Path(args.calib).read_text())
    W, H = cal["img_size"]
    focal = float(cal["focal_px"])
    R_cw, C, ortho, _ = camera_to_world(np.array(cal["H_inv"]), focal,
                                        W / 2.0, H / 2.0)
    print(f"clicked calibration: camera height {C[2]:.3f} m, focal {focal:.0f} px")

    frames, idx = sample_frames(args.video, args.frames, W, H)
    print(f"running ZoeDepth on {len(frames)} frames", flush=True)
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True,
                           trust_repo=True).cuda().eval()

    ys, xs = np.mgrid[0:H:args.stride, 0:W:args.stride]
    keep = ys.ravel() > H * (1 - args.floor_frac)
    u = xs.ravel()[keep].astype(np.float64)
    v = ys.ravel()[keep].astype(np.float64)

    heights, tilts, scales = [], [], []
    for k, f in enumerate(frames):
        with torch.no_grad():
            t = torch.from_numpy(f).permute(2, 0, 1)[None].float().cuda() / 255.0
            d = model.infer(t)[0, 0].cpu().numpy()
        z = d[ys.ravel()[keep], xs.ravel()[keep]]
        # back-project with the CALIBRATED focal, so any focal error in the
        # depth model's own assumptions does not enter
        X = (u - W / 2.0) * z / focal
        Y = (v - H / 2.0) * z / focal
        P = np.stack([X, Y, z], 1)
        n, dd, inl = fit_plane_ransac(P)
        if n is None:
            continue
        if n[1] < 0:                              # make normal point up-ish in cam frame
            n, dd = -n, -dd
        h = abs(dd)                               # camera distance to the plane
        # angle between the depth plane's normal and the clicked plane's normal
        n_click = R_cw.T @ np.array([0.0, 0.0, 1.0])
        ang = np.degrees(np.arccos(np.clip(abs(n @ n_click), -1, 1)))
        heights.append(h)
        tilts.append(ang)
        scales.append(h / C[2])
        print(f"  frame {idx[k]:4d}: depth-plane camera height {h:5.3f} m   "
              f"normal vs clicked {ang:5.1f} deg   inliers {100*inl.mean():.0f}%")

    if not heights:
        raise SystemExit("no plane could be fitted")
    heights = np.array(heights)
    print(f"\nZoeDepth camera height: median {np.median(heights):.3f} m "
          f"(spread {heights.min():.3f}-{heights.max():.3f})")
    print(f"clicked calibration    : {C[2]:.3f} m")
    print(f"ratio depth/clicked    : {np.median(scales):.3f}  "
          f"-> the clicked metres would be off by "
          f"{100*(np.median(scales)-1):+.0f}% if ZoeDepth is right")
    print(f"plane orientation disagreement: median {np.median(tilts):.1f} deg")
    if args.out:
        np.savez(args.out, heights=heights, tilts=np.array(tilts),
                 clicked_height=C[2], frames=np.array(idx))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
