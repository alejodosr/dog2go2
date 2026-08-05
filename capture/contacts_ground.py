"""Phase C, on the ground plane — stance/swing from paw speed in metres.

Supersedes the pixel-speed detector in contacts_2d.py, which was structurally
blind on this clip: 10 cm of paw travel toward the camera moves the pixel about
1 px, against 37 px for the same travel sideways, and this dog walks almost
entirely along the viewing direction. The consequence was measurable -- 42% of
detected stance runs had the paw drifting the wrong way relative to the body,
close to a coin flip, which cancelled the recovered depth travel.

The homography removes that at the root. A pixel known to lie on the ground has
exactly one position on it, so speed measured on the plane is correct in every
direction, including straight away from the camera.

A useful side effect: a LIFTED paw is not on the plane, so its ray strikes the
ground far beyond where the paw really is, and its apparent ground position
races away. Swing is therefore louder here than in pixels, not quieter. The
separation between stance and swing improves from both ends.

Guards this needs that the pixel version did not:
  * rays close to parallel with the ground (near the horizon) put a paw at a
    huge or negative distance -- flagged, not silently used;
  * paws outside the calibration's validity polygon are extrapolation --
    counted and reported per the brief's §5.2.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from capture import paths
from capture.contacts_2d import lowpass, refine_contacts, runs, MIN_SEGMENT_S, LEGS


def to_ground(H, uv):
    """(...,2) pixels -> (...,2) ground metres."""
    uv = np.asarray(uv, np.float64)
    flat = uv.reshape(-1, 2)
    h = np.concatenate([flat, np.ones((len(flat), 1))], axis=1) @ H.T
    return (h[:, :2] / h[:, 2:3]).reshape(uv.shape)


def ground_positions(H, uv, max_range=60.0):
    """Pixels -> ground metres, with a validity mask.

    The mask catches rays that graze the plane: the projective denominator
    goes through zero at the horizon, so points near it fly off to infinity
    and would otherwise dominate every velocity statistic.
    """
    flat = np.asarray(uv, np.float64).reshape(-1, 2)
    h = np.concatenate([flat, np.ones((len(flat), 1))], axis=1) @ H.T
    w = h[:, 2]
    ok = np.abs(w) > 1e-9
    g = np.full((len(flat), 2), np.nan)
    g[ok] = h[ok, :2] / w[ok, None]
    ok &= np.isfinite(g).all(axis=1)
    ok[ok] &= np.linalg.norm(g[ok] - np.nanmedian(g[ok], axis=0), axis=1) < max_range
    return g.reshape(np.asarray(uv).shape), ok.reshape(np.asarray(uv).shape[:-1])


def refine_paw_pixels(infer_npz, R_cw, focal_full, W, H):
    """Re-derive the paw pixel as the BOTTOM of the foot, not the middle of it.

    The landmark used elsewhere is the centroid of a SMAL vertex group, which
    on dog_2 sits 1.8-3.0 cm above the sole. The homography assumes whatever
    pixel it is given lies on the floor, so that offset is projected out along
    the viewing ray and lands 4.9x further away -- 9 to 15 cm of ground error,
    as large as the noise we are chasing. Worse, the offset differs per leg
    and changes as the paw rotates, so it distorts rather than cancels.

    Fixed by taking the lowest vertex of each foot along the world up
    direction, which needs the calibration and is why it happens here rather
    than in animer_infer. FK is re-run from the stored pose; the 8 GB
    checkpoint is not needed.
    """
    import pickle
    import torch
    smal_pkl = paths.resolve(paths.SMAL_MODEL, "SMAL model file (data/smal/)")
    from amr.models.smal_warapper import SMALLayer

    b = infer_npz
    with open(smal_pkl, "rb") as f:
        smal = SMALLayer(**pickle.load(f, encoding="latin1"))
    wgt = smal.lbs_weights.numpy()
    FOOT_J = {"FR": 14, "FL": 10, "RR": 24, "RL": 20}
    foot_idx = {k: np.where(wgt[:, j] > 0.5)[0] for k, j in FOOT_J.items()}

    up_cam = R_cw.T @ np.array([0.0, 0.0, 1.0])
    N = int(b["num_frames"])
    out = np.empty((N, 4, 2))
    with torch.no_grad():
        bt = torch.from_numpy(b["betas_frozen"]).float()[None]
        for s0 in range(0, N, 32):
            e = min(N, s0 + 32)
            o = SMALLayer.forward(
                smal, betas=bt.expand(e - s0, -1),
                global_orient=torch.from_numpy(b["global_orient"][s0:e]).float(),
                pose=torch.from_numpy(b["pose"][s0:e]).float(), pose2rot=False)
            v = o.vertices.numpy() + b["cam_t"][s0:e][:, None, :]
            for li, leg in enumerate(LEGS):
                fv = v[:, foot_idx[leg], :]
                k = np.argmin(fv @ up_cam, axis=1)          # lowest = the sole
                pt = fv[np.arange(len(fv)), k]
                z = np.maximum(pt[:, 2], 1e-6)
                out[s0:e, li, 0] = focal_full * pt[:, 0] / z + W / 2.0
                out[s0:e, li, 1] = focal_full * pt[:, 1] / z + H / 2.0
    return out


def local_metres_per_pixel(H, uv):
    """Ground metres subtended by one pixel, at each pixel location.

    Lets the animal's size be measured in metres without knowing the SMAL
    scale factor, which is not fitted until after contacts exist.
    """
    a = to_ground(H, uv)
    b = to_ground(H, uv + np.array([1.0, 0.0]))
    c = to_ground(H, uv + np.array([0.0, 1.0]))
    return 0.5 * (np.linalg.norm(b - a, axis=-1) + np.linalg.norm(c - a, axis=-1))


def body_length_m(H, paw_uv):
    """The animal's own size in metres, from its paw spread on the ground.

    Self-contained: needs no contacts and no metres-per-unit, so it can be used
    to make the threshold scale-aware before either exists.
    """
    spread_px = np.linalg.norm(paw_uv.max(axis=1) - paw_uv.min(axis=1), axis=-1)
    centre = paw_uv.mean(axis=1)
    mpp = local_metres_per_pixel(H, centre)
    return float(np.median(spread_px * mpp))


def windowed_excursion(g, half):
    """Peak deviation from the window median, per frame and leg (metres).

    TRIED AND REJECTED, kept so it is not re-invented. The reasoning was sound:
    the paw error is smooth 1-3 Hz drift, the same band as real paw motion, so
    no low-pass separates them, and excursion should discriminate on amplitude
    instead of frequency. Measured end to end it was clearly worse -- clamp
    rate 0.34% -> 2.21% (dog) and 4.42% (cat), and the cat's skate rose through
    post-processing instead of falling.

    Why: to reach a duty factor of 0.65 the threshold has to sit ABOVE the
    drift, which means admitting ~7 cm of paw wander as "planted". It relocates
    the problem rather than fixing it. Speed thresholding with heavier position
    smoothing accepts less slip for the same duty factor.
    """
    n = len(g)
    out = np.zeros(g.shape[:2])
    for t in range(n):
        a, b = max(0, t - half), min(n, t + half + 1)
        w = g[a:b]
        med = np.median(w, axis=0)
        out[t] = np.linalg.norm(w - med, axis=-1).max(axis=0)
    return out


def in_polygon(poly, uv):
    import cv2
    p = np.asarray(poly, np.float32)
    flat = np.asarray(uv, np.float64).reshape(-1, 2)
    r = np.array([cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0
                  for x, y in flat])
    return r.reshape(np.asarray(uv).shape[:-1])


def solve_threshold(speed, valid, fps, target_duty, lo=0.01, hi=5.0):
    """Pick the threshold that yields a given stance fraction, by bisection.

    A FIXED threshold cannot serve two clips, and normalising by body length
    does not fix it -- measured end to end, the dog wants 0.20 body-lengths/s
    and the cat 0.55, in opposite directions. The reason is not scale: the dog
    stands still for most of its clip and the cat walks for most of its, so the
    same threshold lands at very different points on each speed distribution.

    Duty factor is the scale-free, clip-free invariant. Walking quadrupeds keep
    each foot down 60-75% of the time, which is biomechanics rather than
    anything fitted here, so solving for it self-calibrates per clip.
    """
    min_len = max(2, int(round(MIN_SEGMENT_S * fps)))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ct = refine_contacts((speed < mid) & valid[:, None], min_len)
        if ct[valid].mean() < target_duty:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def drop_slipping_runs(contacts, ground, budget_m):
    """Reject individual stance runs whose paw wanders too far while planted.

    A global threshold cannot see this. On the dog, moving the threshold from
    0.20 to 0.30 changes the duty factor by 0.01 but quintuples the downstream
    clamp rate -- a handful of individually bad contacts, too few to shift any
    aggregate, each dragging the placement. The defect is per-run, so the
    filter has to be per-run: a stance claims the paw is still, and a run that
    wanders more than the budget has falsified its own claim.
    """
    out = contacts.copy()
    dropped = 0
    for leg in range(out.shape[1]):
        for s_, e_, v in runs(out[:, leg]):
            if not v:
                continue
            t = ground[s_:e_, leg]
            if not np.isfinite(t).all():
                continue
            if np.linalg.norm(t - np.median(t, axis=0), axis=-1).max() > budget_m:
                out[s_:e_, leg] = False
                dropped += 1
    return out, dropped


def evaluate(speed, valid, fps, thresh, ground):
    raw = (speed < thresh) & valid[:, None]
    ct = refine_contacts(raw, max(2, int(round(MIN_SEGMENT_S * fps))))
    slips = []
    for leg in range(4):
        for s, e, v in runs(ct[:, leg]):
            if not v or e - s < 3 or not valid[s:e].all():
                continue
            t = ground[s:e, leg]
            if not np.isfinite(t).all():
                continue
            slips.append(np.linalg.norm(t - t[0], axis=-1).max())
    return ct, (np.median(slips) if slips else np.nan), len(slips)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--thresh", type=float, default=None,
                   help="stance if paw ground speed is below this, in BODY "
                        "LENGTHS PER SECOND; omit to sweep. Scale-aware, so "
                        "one value means the same thing for a cat and a "
                        "wolfhound -- previously this needed 0.20 m/s for the "
                        "dog and 0.30 for the cat to mean the same thing.")
    p.add_argument("--slip-budget", type=float, default=0.0,
                   help="OFF by default: drop stance runs wandering more than "
                        "this fraction of a body length. Measured end to end "
                        "it PINS the animal -- on the cat, budget 0.015 gave a "
                        "clamp rate of 0.02%% and a skate of 0.001 m/s, which "
                        "looks ideal until you notice the recovered traverse "
                        "collapsed from 1.43 m to 0.30 m against a true ~1.3 m. "
                        "Clamp and skate are both minimised by a motionless "
                        "robot, so neither can be optimised without also "
                        "checking that the trajectory survives.")
    p.add_argument("--target-duty", type=float, default=0.68,
                   help="solve the threshold to hit this stance fraction "
                        "instead of fixing it. Self-calibrating per clip.")
    p.add_argument("--speed-cutoff", type=float, default=8.0)
    p.add_argument("--pos-cutoff", type=float, default=2.0,
                   help="Hz; low-pass on ground positions before "
                        "differentiating. This DOES overlap the 1-3 Hz band "
                        "where the paw error lives, so it costs real signal -- "
                        "but measured end to end it still beats the "
                        "alternatives (see windowed_excursion).")
    p.add_argument("--refine-paws", action="store_true", default=True,
                   help="ON by default: use the bottom of the foot rather than "
                        "the vertex-group centroid as the contact point. The "
                        "centroid sits 1.8-3.0 cm above the sole, and since the "
                        "homography assumes the pixel is ON the floor, that "
                        "offset is projected out along the viewing ray and "
                        "lands (distance/camera-height) times further away -- "
                        "9-15 cm on dog_2. Needs torch.")
    p.add_argument("--no-refine-paws", dest="refine_paws", action="store_false")
    p.add_argument("--plot", default=None)
    p.add_argument("--segments", default="0,174,330,660,807",
                   help="frame boundaries for the per-segment report")
    args = p.parse_args()

    b = np.load(args.infer, allow_pickle=True)
    cal = json.loads(Path(args.calib).read_text())
    fps = float(b["fps"])
    N = int(b["num_frames"])
    valid = b["valid"].copy()
    paw_uv = b["paw_uv"]
    W, H_ = [int(v) for v in b["img_size"]]
    if [W, H_] != cal["img_size"]:
        raise SystemExit(f"pixel frames disagree: inference {W}x{H_} vs "
                         f"calibration {cal['img_size']}")

    Hm = np.array(cal["H"])
    if args.refine_paws:
        from capture.world_place import camera_to_world
        R_cw, _, _, _ = camera_to_world(np.array(cal["H_inv"]),
                                        float(cal["focal_px"]), W / 2.0, H_ / 2.0)
        moved = paw_uv.copy()
        paw_uv = refine_paw_pixels(b, R_cw,
                                   float(b["focal_full"]), W, H_)
        print(f"  paw point moved to the sole: median "
              f"{np.median(np.linalg.norm(paw_uv - moved, axis=-1)):.1f} px")
    ground, ok = ground_positions(Hm, paw_uv)
    inside = in_polygon(cal["validity_polygon"], paw_uv)

    print(f"clip {str(b['source'])}: {N} frames @ {fps:.3f} fps")
    print(f"  paw samples off the plane / near horizon : "
          f"{100*(~ok).mean():5.2f}%")
    print(f"  paw samples outside validity polygon     : "
          f"{100*(~inside).mean():5.2f}%   (extrapolation; flagged not dropped)")

    # A frame is usable if every paw resolved onto the plane.
    valid &= ok.all(axis=1)
    g = np.where(np.isfinite(ground), ground, 0.0)
    g = lowpass(g, fps, args.pos_cutoff)
    body = body_length_m(Hm, paw_uv)
    speed = lowpass(np.linalg.norm(np.gradient(g, axis=0), axis=-1) * fps,
                    fps, args.speed_cutoff) / max(body, 1e-6)
    print(f"  animal size on the ground   {body:.3f} m"
          f"   -> speeds reported in body lengths/s")

    bounds = [int(x) for x in args.segments.split(",")]
    names = ["standing", "WALKING", "milling", "turn+away"][:len(bounds) - 1]
    segs = dict(zip(names, zip(bounds[:-1], bounds[1:])))

    if args.thresh is None and args.target_duty:
        args.thresh = solve_threshold(speed, valid, fps, args.target_duty)
        print(f"  solved threshold            {args.thresh:.3f} body-lengths/s"
              f"  for a target duty of {args.target_duty:.2f}")

    if args.thresh is None:
        print("\nthreshold sweep (body lengths per second)\n")
        print("%8s %9s %9s %9s %9s %9s" % ("bodylen", "walk duty", "walk 0ft",
                                           "walk slip", "stand", "milling"))
        for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55]:
            ct, slip, _ = evaluate(speed, valid, fps, th, g)
            row = []
            for nm in ("WALKING", "standing", "milling"):
                a, bb = segs[nm]
                m = valid[a:bb]
                row.append(ct[a:bb][m].mean() if m.any() else np.nan)
            a, bb = segs["WALKING"]
            m = valid[a:bb]
            zero = (ct[a:bb][m].sum(1) == 0).mean() if m.any() else np.nan
            _, wslip, _ = evaluate(speed[a:bb], valid[a:bb], fps, th, g[a:bb])
            print("%8.2f %9.2f %9.2f %9.3f %9.2f %9.2f"
                  % (th, row[0], zero, wslip, row[1], row[2]))
        print("\npick with --thresh; a dog walk has duty 0.6-0.75")
        return

    contacts, slip, nrun = evaluate(speed, valid, fps, args.thresh, g)
    if args.slip_budget:
        contacts, ndrop = drop_slipping_runs(contacts, g, args.slip_budget * body)
        print(f"  dropped {ndrop} stance runs exceeding the slip budget "
              f"({args.slip_budget:.3f} body-lengths = "
              f"{args.slip_budget*body*100:.1f} cm)")
    print(f"\nthreshold {args.thresh} m/s, flicker merge < "
          f"{max(2, int(round(MIN_SEGMENT_S*fps)))} frames")
    print("\nduty factor per leg")
    for i, leg in enumerate(LEGS):
        print(f"  {leg}  {contacts[valid, i].mean():.3f}")
    nfeet = contacts.sum(axis=1)
    print("\nfeet down per frame")
    for k in range(5):
        print(f"  {k}  {100*float((nfeet[valid]==k).mean()):5.1f}%")
    print(f"\nstance slip on the ground: median {slip:.3f} m over {nrun} runs")
    print("  (this is now a real distance, not a pixel count scaled by a "
          "single\n   isotropic factor -- the old 1.8 cm was optimistic by "
          "up to 38x in depth)")

    print("\nper segment")
    print("%-12s %8s %8s %8s" % ("segment", "duty", "0 feet", "slip m"))
    for nm, (a, bb) in segs.items():
        m = valid[a:bb]
        if not m.any():
            continue
        _, sl, _ = evaluate(speed[a:bb], valid[a:bb], fps, args.thresh, g[a:bb])
        print("%-12s %8.2f %8.2f %8.3f"
              % (nm, contacts[a:bb][m].mean(),
                 (contacts[a:bb][m].sum(1) == 0).mean(), sl))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(N) / fps
        fig, ax = plt.subplots(1, 3, figsize=(19, 5.2),
                               gridspec_kw={"width_ratios": [2.1, 1.5, 1.2]})
        for i, leg in enumerate(LEGS):
            ax[0].plot(t, speed[:, i], lw=0.8, label=leg)
        ax[0].axhline(args.thresh, color="k", ls="--", lw=1)
        ax[0].set_ylim(0, float(np.nanpercentile(speed[valid], 99)))
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("paw ground speed (m/s)")
        ax[0].legend(ncol=4, fontsize=8)
        ax[0].set_title(f"{str(b['source'])} — ground-plane contacts")
        for i, leg in enumerate(LEGS):
            for s, e, v in runs(contacts[:, i]):
                if v:
                    ax[1].barh(i, (e - s) / fps, left=s / fps, height=.7,
                               color="tab:green")
        ax[1].set_yticks(range(4)); ax[1].set_yticklabels(LEGS)
        ax[1].invert_yaxis(); ax[1].set_xlabel("time (s)")
        ax[1].set_title("gait diagram")
        col = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
        for i, leg in enumerate(LEGS):
            m = contacts[:, i] & valid
            ax[2].scatter(g[m, i, 0], g[m, i, 1], s=4, c=col[i], label=leg,
                          alpha=.55)
        ax[2].set_aspect("equal")
        ax[2].set_xlabel("ground X (m)"); ax[2].set_ylabel("ground Y (m)")
        ax[2].set_title("stance footfalls, top-down")
        ax[2].legend(fontsize=7)
        for a_ in ax:
            a_.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"\nwrote {args.plot}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, source=str(b["source"]), fps=fps, num_frames=N,
                        contacts=contacts, valid=valid, speed=speed,
                        ground=ground, on_plane=ok, inside_polygon=inside,
                        paw_uv=paw_uv, refined_paws=bool(args.refine_paws),
                        threshold=float(args.thresh), calib=str(args.calib))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
