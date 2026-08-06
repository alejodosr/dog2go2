"""Debug view: every intermediate the placement consumed, on one screen.

The sbs video only shows the END of the pipeline, so when it looks wrong it
cannot say which stage lied. This view overlays the intermediates on the
source pixels, where they can disagree visibly:

  source panel   * calibrated floor grid, projected through H_inv -- if the
                   grid does not lie on the visible floor, the ground plane
                   (stage 2) is wrong and nothing downstream can be right
                 * AniMer skeleton reprojected through its own weak camera
                   (cyan) -- must sit ON the animal; if not, stage 1 is wrong
                 * solved world skeleton reprojected through the calibrated
                   camera (magenta) -- where placement (stage 4) thinks the
                   dog is. The magenta-vs-cyan gap IS the placement error,
                   in pixels, per frame.
                 * paw markers, filled while the detector says stance
  3D panel       the world skeleton on the calibrated floor (viz_world's view)
  timeline       gait diagram, root height and lowest-toe height vs time,
                 with a playhead -- a jump that the contact detector missed
                 shows up here as an unbroken stance bar under a rising cyan
                 root, and a flattened magenta one.

Reads only artifacts the default pipeline already wrote; changes nothing.

    PYTHONPATH=$REPO $PY_CAPTURE -m capture.viz_debug \
        --infer $WORK/<clip>_animer.npz --contacts $WORK/<clip>_contacts.npz \
        --world $WORK/<clip>_world.npz --calib $CALIB/<clip>_depth.json \
        --video <video> --out media/debug_<clip>.mp4
"""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from capture.contacts_2d import LEGS, runs
from capture.viz_world import BONES, LEG_COLOR, TOE_OF, open_writer

TOE0 = 6


def project_points(pts_cam, focal, W, H):
    z = np.maximum(pts_cam[..., 2], 1e-6)
    return np.stack([focal * pts_cam[..., 0] / z + W / 2.0,
                     focal * pts_cam[..., 1] / z + H / 2.0], axis=-1)


def draw_skeleton_2d(img, uv, color, thickness, sx, sy):
    import cv2

    def pt(i):
        u = float(np.clip(np.nan_to_num(uv[i, 0]) * sx, -1e4, 1e4))
        v = float(np.clip(np.nan_to_num(uv[i, 1]) * sy, -1e4, 1e4))
        return int(round(u)), int(round(v))

    for a, b in BONES:
        cv2.line(img, pt(a), pt(b), color, thickness, cv2.LINE_AA)
    for i in range(len(uv)):
        cv2.circle(img, pt(i), 4 if i >= TOE0 else 3, color, -1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True)
    p.add_argument("--contacts", required=True)
    p.add_argument("--world", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--trim", default=None,
                   help="start,end video frames the world npz covers. "
                        "Defaults to the npz's own record (world_place stores "
                        "it); only needed for npz files older than that field")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--grid-half", type=float, default=2.0,
                   help="half-extent (m) of the projected floor grid around "
                        "the dog's median floor position")
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-72.0)
    args = p.parse_args()

    import cv2
    b = np.load(args.infer, allow_pickle=True)
    c = np.load(args.contacts, allow_pickle=True)
    w = np.load(args.world, allow_pickle=True)
    cal = json.loads(Path(args.calib).read_text())

    fps = float(w["fps"])
    world = w["world"]                            # (N,10,3) metres, world
    contacts = w["contacts"]
    anchored = w["anchored"]
    R_cw = w["R_cw"]
    C = w["camera_pos"]
    N = len(world)
    W, H_ = [int(v) for v in b["img_size"]]
    focal = float(cal["focal_px"])
    F_conv = float(b["focal_full"])
    H_inv = np.array(cal["H_inv"], float)

    # The world npz may be a TRIMMED view of the video (world_place --trim);
    # the infer/contacts npz are always untrimmed. a0 aligns the three:
    # world index = video frame - a0.
    if args.trim:
        a0, a1 = [int(x) for x in args.trim.split(",")]
    elif "trim" in w.files:
        a0, a1 = [int(v) for v in w["trim"]]
    else:
        a0, a1 = 0, N
    if a1 - a0 != N:
        raise SystemExit(f"trim {a0},{a1} covers {a1 - a0} frames but the "
                         f"world npz has {N}")
    sl = slice(a0, a0 + N)

    # AniMer's own camera-frame skeleton (model units, weak camera F_conv)
    pts_cam_animer = (b["points_local"] + b["root_model"][:, None, :]
                      + b["cam_t"][:, None, :])[sl]
    uv_animer = project_points(pts_cam_animer, F_conv, W, H_)

    # solved world skeleton back through the CALIBRATED camera; the stored
    # R_cw is used exactly as world_place_ba's proj(): cam = (p - C) @ R_cw
    pts_cam_world = np.einsum("nkj,ji->nki", world - C, R_cw)
    uv_world = project_points(pts_cam_world, focal, W, H_)

    paw_uv = w["paw_uv"] if "paw_uv" in w.files else c["paw_uv"][sl]

    # floor grid in world metres around where the dog actually is
    toe_med = np.median(world[:, TOE0:TOE0 + 4, :2].reshape(-1, 2), axis=0)
    g0 = np.floor((toe_med - args.grid_half) * 2) / 2
    g1 = np.ceil((toe_med + args.grid_half) * 2) / 2
    grid_lines = []
    for x in np.arange(g0[0], g1[0] + 1e-6, 0.5):
        grid_lines.append(np.stack([np.full(24, x),
                                    np.linspace(g0[1], g1[1], 24)], 1))
    for y in np.arange(g0[1], g1[1] + 1e-6, 0.5):
        grid_lines.append(np.stack([np.linspace(g0[0], g1[0], 24),
                                    np.full(24, y)], 1))

    def floor_to_px(xy):
        h = np.concatenate([xy, np.ones((len(xy), 1))], 1) @ H_inv.T
        return h[:, :2] / np.maximum(np.abs(h[:, 2:]), 1e-9) * np.sign(h[:, 2:]), h[:, 2]

    # ---- static timeline raster -------------------------------------------
    t = np.arange(N) / fps
    root_z = world[:, 0, 2]
    toe_z = world[:, TOE0:TOE0 + 4, 2]
    # AniMer's root height, in the same world frame, at the solved scale but
    # WITHOUT the stance anchoring: rotate its camera-frame root into world
    # orientation and remove the per-clip mean offset. Shape-only comparison:
    # if this rises during a jump and root_z does not, placement ate the jump.
    s = float(w["metres_per_unit"])
    root_w_animer = (R_cw @ (s * pts_cam_animer[:, 0]).T).T[:, 2]
    root_w_animer -= np.median(root_w_animer - root_z)

    TW, TH = 0, 220
    fig_t = plt.figure(figsize=(12.0, TH / 100.0), dpi=100)
    axg = fig_t.add_axes([0.055, 0.55, 0.93, 0.40])
    axz = fig_t.add_axes([0.055, 0.14, 0.93, 0.38])
    for i, leg in enumerate(LEGS):
        for s_, e_, v_ in runs(contacts[:, i]):
            if v_:
                axg.barh(i, (e_ - s_) / fps, left=s_ / fps, height=0.72,
                         color=LEG_COLOR[leg])
    nfeet = contacts.sum(1)
    axg.fill_between(t, -0.6, 3.6, where=nfeet == 0, color="0.85", step="mid")
    axg.set_yticks(range(4)); axg.set_yticklabels(LEGS, fontsize=7)
    axg.set_ylim(-0.6, 3.6); axg.invert_yaxis()
    axg.set_xlim(0, t[-1]); axg.tick_params(labelbottom=False, labelsize=7)
    axg.set_title("stance bars (grey = zero feet down)", fontsize=8, pad=2)
    axz.plot(t, root_z, c="m", lw=1.4, label="root z (solved)")
    axz.plot(t, root_w_animer, c="c", lw=1.4, label="root z (AniMer, shape only)")
    axz.plot(t, toe_z.min(1), c="0.3", lw=1.0, label="lowest toe z")
    axz.axhline(0, c="k", lw=0.6)
    axz.set_xlim(0, t[-1]); axz.legend(fontsize=7, ncol=3, loc="upper right")
    axz.set_xlabel("time (s)", fontsize=8); axz.tick_params(labelsize=7)
    axz.set_ylabel("m", fontsize=8)
    fig_t.canvas.draw()
    timeline = np.asarray(fig_t.canvas.buffer_rgba())[:, :, :3].copy()
    # time -> pixel column, for the playhead
    x0 = axg.get_window_extent().x0
    x1 = axg.get_window_extent().x1
    plt.close(fig_t)

    # ---- 3D panel figure ---------------------------------------------------
    lo = world.reshape(-1, 3).min(0); hi = world.reshape(-1, 3).max(0)
    pad = 0.12
    xlim = (lo[0] - pad, hi[0] + pad); ylim = (lo[1] - pad, hi[1] + pad)
    zlim = (-0.05, max(0.8, hi[2] + 0.2))
    Hp = args.height
    fig = plt.figure(figsize=(Hp * 1.35 / 100.0, Hp / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")

    writer = None
    vfr, written = -1, 0
    root_trail = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vfr += 1
        idx = vfr - a0            # index into the trimmed world/infer arrays
        if not (0 <= idx < N) or idx % args.stride:
            continue

        fh, fw = frame.shape[:2]
        left = cv2.resize(frame, (int(round(fw * Hp / fh)) // 2 * 2, Hp),
                          interpolation=cv2.INTER_AREA)
        sx = left.shape[1] / W; sy = left.shape[0] / H_

        # floor grid (thin white)
        for gl in grid_lines:
            px, wgt = floor_to_px(gl)
            seg = px[(wgt > 1e-6)]
            seg = seg[(seg[:, 0] > -W) & (seg[:, 0] < 2 * W)
                      & (seg[:, 1] > -H_) & (seg[:, 1] < 2 * H_)]
            for a, bb in zip(seg[:-1], seg[1:]):
                cv2.line(left, (int(a[0] * sx), int(a[1] * sy)),
                         (int(bb[0] * sx), int(bb[1] * sy)),
                         (240, 240, 240), 1, cv2.LINE_AA)

        # skeletons: cyan = AniMer (its own camera), magenta = solved world
        draw_skeleton_2d(left, uv_animer[idx], (220, 220, 40), 2, sx, sy)
        draw_skeleton_2d(left, uv_world[idx], (200, 40, 200), 2, sx, sy)

        # tracked paw pixels, filled while stance
        for li, leg in enumerate(LEGS):
            u, v = paw_uv[idx, li]
            down = bool(contacts[idx, li])
            col = tuple(int(255 * x) for x in
                        plt.matplotlib.colors.to_rgb(LEG_COLOR[leg]))[::-1]
            cv2.circle(left, (int(u * sx), int(v * sy)), 9 if down else 6,
                       col, -1 if down else 2, cv2.LINE_AA)
        cv2.putText(left, "cyan=AniMer  magenta=solved  filled paw=stance",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(left, f"frame {idx}  {int(contacts[idx].sum())} feet  "
                    f"{'anchored' if anchored[idx] else 'INTERP'}",
                    (10, left.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

        # 3D panel
        pts = world[idx]
        root_trail.append(pts[0].copy())
        ax.clear()
        for xg in np.arange(np.floor(xlim[0] * 2) / 2, xlim[1], 0.5):
            ax.plot([xg, xg], [ylim[0], ylim[1]], [0, 0], c="0.85", lw=.7)
        for yg in np.arange(np.floor(ylim[0] * 2) / 2, ylim[1], 0.5):
            ax.plot([xlim[0], xlim[1]], [yg, yg], [0, 0], c="0.85", lw=.7)
        tr = np.array(root_trail)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], c="0.35", lw=1.2)
        for a, bb in BONES:
            seg = pts[[a, bb]]
            leg = TOE_OF.get(bb)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    c=LEG_COLOR[leg] if leg else "0.2", lw=3.0)
        for i, leg in TOE_OF.items():
            down = contacts[idx, i - TOE0]
            ax.scatter(*pts[i], c=LEG_COLOR[leg], s=140 if down else 40,
                       edgecolors="k" if down else "none",
                       linewidths=1.4 if down else 0, depthshade=False)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0],
                           max(0.6, zlim[1] - zlim[0])))
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(f"t = {idx / fps:5.2f} s  root z {root_z[idx]:.3f} m",
                     fontsize=9)
        fig.canvas.draw()
        right = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        right = cv2.resize(right, (int(round(right.shape[1] * Hp
                           / right.shape[0])) // 2 * 2, Hp),
                           interpolation=cv2.INTER_AREA)

        top = np.concatenate([left, right], axis=1)
        # timeline, stretched to the top row's width, playhead at t
        tl = cv2.resize(timeline, (top.shape[1], timeline.shape[0]),
                        interpolation=cv2.INTER_AREA)
        xr = top.shape[1] / 1200.0
        px = int((x0 + (x1 - x0) * (idx / fps) / max(t[-1], 1e-9)) * xr)
        cv2.line(tl, (px, 0), (px, tl.shape[0]), (30, 30, 30), 2)
        both = np.concatenate([top, tl], axis=0)
        both = both[:both.shape[0] // 2 * 2, :both.shape[1] // 2 * 2]

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
    print(f"wrote {args.out} ({written} frames @ {fps / args.stride:.1f} fps)")


if __name__ == "__main__":
    main()
