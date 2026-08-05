"""Source video beside the recovered world skeleton, frame for frame.

The honest visual check: it ties world placement back to the pixels it came
from. Rendering the 3D panel per source frame rather than compositing two
finished videos means the two sides cannot drift out of sync.

The skeleton is drawn in WORLD coordinates on the calibrated floor, so what
you see on the right is where the pipeline thinks the dog actually was, not a
root-centred dog walking on the spot.
"""
from pathlib import Path
import argparse
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from capture.contacts_2d import LEGS

BONES = [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5),
         (2, 6), (3, 7), (4, 8), (5, 9)]
LEG_COLOR = {"FR": "tab:red", "FL": "tab:orange",
             "RR": "tab:green", "RL": "tab:blue"}
TOE_OF = {6: "FR", 7: "FL", 8: "RR", 9: "RL"}


def open_writer(path, w, h, fps, crf=18):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
         "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium",
         "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--infer", default=None,
                   help="Phase B npz; draws the tracked paws on the source "
                        "panel in the same colours as the 3D panel, so the "
                        "two sides can be compared paw by paw")
    p.add_argument("--right-video", default=None,
                   help="use this video as the right panel (e.g. the Go2 "
                        "render) instead of drawing the 3D skeleton")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-72.0)
    args = p.parse_args()

    import cv2
    w = np.load(args.world, allow_pickle=True)
    world = w["world"]
    contacts = w["contacts"]
    anchored = w["anchored"]
    fps = float(w["fps"])
    N = len(world)

    lo = world.reshape(-1, 3).min(axis=0)
    hi = world.reshape(-1, 3).max(axis=0)
    pad = 0.12
    xlim = (lo[0] - pad, hi[0] + pad)
    ylim = (lo[1] - pad, hi[1] + pad)
    zlim = (-0.05, max(1.0, hi[2] + 0.2))

    paw_uv = infer_wh = None
    if args.infer:
        bi = np.load(args.infer, allow_pickle=True)
        # prefer the paw point the contacts were actually built from -- with
        # --refine-paws that is the sole, not the vertex-group centroid, and
        # drawing the centroid here would mislabel what the pipeline saw
        paw_uv = w["paw_uv"] if "paw_uv" in w.files else bi["paw_uv"]
        infer_wh = [int(v) for v in bi["img_size"]]

    right_frames = None
    if args.right_video:
        rcap = cv2.VideoCapture(args.right_video)
        if not rcap.isOpened():
            raise SystemExit(f"could not open {args.right_video}")
        rfps = rcap.get(cv2.CAP_PROP_FPS) or fps
        right_frames = []
        while True:
            ok, fr = rcap.read()
            if not ok:
                break
            right_frames.append(fr)
        rcap.release()
        print(f"right panel: {len(right_frames)} frames @ {rfps:.1f} fps")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")

    H = args.height
    # match the source panel's aspect so the 3D box fills the frame instead of
    # sitting in a square surrounded by white
    fig = plt.figure(figsize=(H * 1.6 / 100.0, H / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94)

    writer = None
    idx = -1
    written = 0
    root_trail = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx >= N or idx % args.stride:
            continue

        # left: source frame, scaled to the panel height
        fh, fw = frame.shape[:2]
        left = cv2.resize(frame, (int(round(fw * H / fh)) // 2 * 2, H),
                          interpolation=cv2.INTER_AREA)
        if paw_uv is not None:
            # inference ran on a uniformly resized copy, so pixels map by a
            # single ratio -- no need to redo the crop maths
            sx = left.shape[1] / infer_wh[0]
            sy = left.shape[0] / infer_wh[1]
            for li, leg in enumerate(LEGS):
                u, v = paw_uv[idx, li]
                down = bool(contacts[idx, li])
                col = tuple(int(255 * x) for x in
                            plt.matplotlib.colors.to_rgb(LEG_COLOR[leg]))[::-1]
                cv2.circle(left, (int(round(u * sx)), int(round(v * sy))),
                           11 if down else 8, col, -1 if down else 2,
                           cv2.LINE_AA)
                if down:
                    cv2.circle(left, (int(round(u * sx)), int(round(v * sy))),
                               11, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(left, leg, (int(u * sx) + 13, int(v * sy) - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            cv2.putText(left, "filled = planted", (14, left.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                        cv2.LINE_AA)
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

        if right_frames is not None:
            j = min(int(round(idx / fps * rfps)), len(right_frames) - 1)
            rf = right_frames[j]
            right = cv2.cvtColor(cv2.resize(
                rf, (int(round(rf.shape[1] * H / rf.shape[0])) // 2 * 2, H),
                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            both = np.concatenate([left, right], axis=1)
            if writer is None:
                writer = open_writer(args.out, both.shape[1], both.shape[0],
                                     fps / args.stride)
            writer.stdin.write(np.ascontiguousarray(both).tobytes())
            written += 1
            continue

        # right: world skeleton
        pts = world[idx]
        root_trail.append(pts[0].copy())
        ax.clear()
        # floor grid at z = 0
        for xg in np.arange(np.floor(xlim[0] * 2) / 2, xlim[1], 0.5):
            ax.plot([xg, xg], [ylim[0], ylim[1]], [0, 0], c="0.85", lw=.7)
        for yg in np.arange(np.floor(ylim[0] * 2) / 2, ylim[1], 0.5):
            ax.plot([xlim[0], xlim[1]], [yg, yg], [0, 0], c="0.85", lw=.7)
        tr = np.array(root_trail)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], c="0.35", lw=1.2)
        for a, b in BONES:
            seg = pts[[a, b]]
            leg = TOE_OF.get(b)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    c=LEG_COLOR[leg] if leg else "0.2", lw=3.2)
        ax.scatter(*pts[:2].T, c="k", s=42)
        for i, leg in TOE_OF.items():
            down = contacts[idx, i - 6]   # TOE_OF keys are point ids 6..9
            ax.scatter(*pts[i], c=LEG_COLOR[leg], s=150 if down else 45,
                       edgecolors="k" if down else "none",
                       linewidths=1.5 if down else 0, depthshade=False)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0],
                           max(0.6, zlim[1] - zlim[0])))
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_zticks([0, 0.5])
        ax.set_title(f"t = {idx/fps:5.2f} s    "
                     f"{'anchored' if anchored[idx] else 'INTERPOLATED'}    "
                     f"{int(contacts[idx].sum())} feet down\n"
                     f"filled paw = planted", fontsize=10)
        fig.canvas.draw()
        right = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        right = cv2.resize(right, (int(round(right.shape[1] * H / right.shape[0]))
                                   // 2 * 2, H), interpolation=cv2.INTER_AREA)

        both = np.concatenate([left, right], axis=1)
        if writer is None:
            writer = open_writer(args.out, both.shape[1], both.shape[0],
                                 fps / args.stride)
        writer.stdin.write(np.ascontiguousarray(both).tobytes())
        written += 1
        if written % 60 == 0:
            print(f"  {written} frames", flush=True)

    cap.release()
    if writer:
        writer.stdin.close()
        writer.wait()
    plt.close(fig)
    print(f"wrote {args.out} ({written} frames @ {fps/args.stride:.1f} fps)")


if __name__ == "__main__":
    main()
