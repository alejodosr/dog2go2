"""Phase 1 (PLAN.md) — stance/swing from signals that never touch the
homography.

contacts_ground.py measures paw speed ON THE FLOOR, which multiplies every
centimetre of paw-height error by distance/camera-height. This detector votes
across three signals that carry the 2-5 cm mesh noise but NOT that multiplier:

  1. forward-relative toe velocity — the toe's speed relative to the root
     along the body's own forward axis, from FK alone. During stance the body
     passes OVER the planted foot (strongly negative, about minus the walking
     speed); during swing the leg sweeps forward (strongly positive, about
     twice it). The signal amplitude is a leg-length per second against 2-5 cm
     of mesh noise — unlike toe height, whose 3-6 cm of swing lift sits AT the
     noise floor (the same reason STATUS.md rejected height relabelling).
  2. body-frame toe height — weak alone (see above), kept as a gate: a toe at
     swing apex has near-zero velocity but is visibly lifted.
  3. pixel speed — the camera is static, so a planted paw is stationary in
     the image in both axes. Depth-normalised via cam_t the way contacts_2d.py
     does. Its known blind spot (motion along the viewing ray compresses to
     ~1 px per 10 cm) is why it is one vote of three, not the detector.

All thresholds are in leg lengths and seconds — scale-free and clip-free by
construction. The gate (PLAN.md Phase 1) is that ONE set of constants works
across every camera height on the harness and across the three real clips,
where contacts_ground needed per-clip tuning (0.20 / 0.35 / 0.25).

Output contract matches contacts_ground.py, so world_place.py and
world_place_ba.py consume either interchangeably. `ground` (homography
positions) is still emitted — placement needs floor MEASUREMENTS; this file
only stops floor error from deciding TIMING.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from capture import paths
from capture.contacts_2d import lowpass, refine_contacts, runs, MIN_SEGMENT_S, LEGS
from capture.contacts_ground import (ground_positions, refine_paw_pixels, in_polygon)
from capture.world_place import camera_to_world

TOE0 = 6
MOUNT0 = 2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--height-frac", type=float, default=0.10,
                   help="planted if toe is within this fraction of leg length "
                        "of the lowest toe (body-frame height vote)")
    p.add_argument("--vfwd-frac", type=float, default=0.20,
                   help="planted if the toe's forward velocity RELATIVE to "
                        "the root is below this (leg lengths/s): stance is "
                        "backward relative motion, swing is strongly forward")
    p.add_argument("--px-frac", type=float, default=0.45,
                   help="planted if depth-normalised pixel speed is below "
                        "this, in leg lengths per second")
    p.add_argument("--votes", type=int, default=2, help="votes needed of 3")
    p.add_argument("--cutoff", type=float, default=4.0,
                   help="Hz low-pass before differentiating")
    p.add_argument("--refine-paws", action="store_true", default=True)
    p.add_argument("--no-refine-paws", dest="refine_paws", action="store_false")
    p.add_argument("--plot", default=None)
    args = p.parse_args()

    b = np.load(args.infer, allow_pickle=True)
    cal = json.loads(Path(args.calib).read_text())
    fps = float(b["fps"])
    N = int(b["num_frames"])
    valid = b["valid"].copy()
    W, H_ = [int(v) for v in b["img_size"]]
    if [W, H_] != cal["img_size"]:
        raise SystemExit(f"pixel frames disagree: inference {W}x{H_} vs "
                         f"calibration {cal['img_size']}")

    R_cw, _, _, _ = camera_to_world(np.array(cal["H_inv"]),
                                    float(cal["focal_px"]), W / 2.0, H_ / 2.0)

    pts = b["points_local"]                       # (N,10,3) model units, cam frame
    pts_or = np.einsum("ij,nkj->nki", R_cw, pts)  # world-ORIENTED, unplaced
    leg_len = float(np.median(np.linalg.norm(
        pts[:, MOUNT0:MOUNT0 + 4] - pts[:, TOE0:TOE0 + 4], axis=-1)))

    # ---- vote 1: forward-relative toe velocity ----------------------------
    fwd = pts_or[:, 1, :2] - pts_or[:, 0, :2]        # chest - root, horizontal
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    rel = np.einsum("nkc,nc->nk",
                    pts_or[:, TOE0:TOE0 + 4, :2] - pts_or[:, 0:1, :2], fwd)
    v_fwd_rel = np.gradient(lowpass(rel, fps, args.cutoff), axis=0) * fps
    v_stance = v_fwd_rel < args.vfwd_frac * leg_len

    # ---- vote 2: body-frame toe height ------------------------------------
    z = lowpass(pts_or[:, TOE0:TOE0 + 4, 2], fps, args.cutoff)
    z_rel = z - z.min(axis=1, keepdims=True)
    v_height = z_rel < args.height_frac * leg_len

    # ---- vote 3: depth-normalised pixel speed -----------------------------
    paw_uv_raw = b["paw_uv"]
    depth = np.abs(b["cam_t"][:, 2])              # model units, per frame
    px_v = np.linalg.norm(np.gradient(lowpass(paw_uv_raw, fps, args.cutoff),
                                      axis=0), axis=-1) * fps
    speed_u = px_v * depth[:, None] / float(b["focal_full"])   # units/s
    v_pix = speed_u < args.px_frac * leg_len

    votes = v_stance.astype(int) + v_height.astype(int) + v_pix.astype(int)
    contacts = refine_contacts((votes >= args.votes) & valid[:, None],
                               max(2, int(round(MIN_SEGMENT_S * fps))))

    # ---- floor measurements for the placement stage (compat contract) -----
    paw_uv = paw_uv_raw
    if args.refine_paws:
        paw_uv = refine_paw_pixels(b, R_cw,
                                   float(b["focal_full"]), W, H_)
    Hm = np.array(cal["H"])
    ground, ok = ground_positions(Hm, paw_uv)
    inside = in_polygon(cal["validity_polygon"], paw_uv)
    valid &= ok.all(axis=1)
    contacts &= valid[:, None]

    print(f"clip {str(b['source'])}: {N} frames @ {fps:.3f} fps  "
          f"(kinematic contacts, no floor speed)")
    print(f"  leg length {leg_len:.3f} units; votes needed {args.votes}/3")
    for i, leg in enumerate(LEGS):
        print(f"  {leg} duty {contacts[valid, i].mean():.3f}   "
              f"votes fwd/h/px "
              f"{v_stance[valid, i].mean():.2f}/"
              f"{v_height[valid, i].mean():.2f}/{v_pix[valid, i].mean():.2f}")
    nfeet = contacts.sum(axis=1)
    print(f"  duty overall {contacts[valid].mean():.2f}   zero-feet frames "
          f"{100 * float((nfeet[valid] == 0).mean()):.1f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(N) / fps
        fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        for i, leg in enumerate(LEGS):
            ax[0].plot(t, z_rel[:, i] / leg_len, lw=.9, label=leg)
        ax[0].axhline(args.height_frac, color="k", ls="--", lw=.8)
        ax[0].set_ylabel("toe height / leg length")
        ax[0].legend(ncol=4, fontsize=8)
        for i, leg in enumerate(LEGS):
            for s, e, v in runs(contacts[:, i]):
                if v:
                    ax[1].barh(i, (e - s) / fps, left=s / fps, height=.7,
                               color="tab:green")
        ax[1].set_yticks(range(4)); ax[1].set_yticklabels(LEGS)
        ax[1].invert_yaxis(); ax[1].set_xlabel("time (s)")
        ax[1].set_title("gait diagram")
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, source=str(b["source"]), fps=fps, num_frames=N,
                        contacts=contacts, valid=valid,
                        speed=speed_u / leg_len,      # leg-lengths/s, for viz
                        ground=ground, on_plane=ok, inside_polygon=inside,
                        paw_uv=paw_uv, refined_paws=bool(args.refine_paws),
                        threshold=float("nan"), calib=str(args.calib),
                        votes_stance=v_stance, votes_height=v_height,
                        votes_pix=v_pix)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
