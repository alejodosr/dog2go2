"""Phase C — stance/swing detection from paw pixel tracks.

Contacts are detected in IMAGE space, not from 3D foot height (brief §4.3).
Depth carries most of the monocular error, so a 3D height threshold is least
reliable exactly where precision matters; paw motion measured in pixels never
passes through the depth ambiguity at all.

Two deliberate departures from the brief, both because the camera is static:

  * It says "paw vertical velocity". We use the full 2D pixel speed. With a
    fixed camera a planted paw is stationary in the world and therefore
    stationary in the image, in BOTH axes -- so horizontal motion is signal
    too, and discarding it throws away half the evidence. Vertical-only is
    still reported for comparison.

  * Pixel speed is not scale-free: the same real motion produces fewer pixels
    at the far end of the corridor, and this dog walks most of the clip's
    length. So pixel speed is converted to a physical speed using the dog's
    depth: at distance Z a model unit subtends focal/Z pixels, so
    speed_units = speed_px * Z / focal, and metres follow from one constant.

    Do NOT normalise by a projected body length instead. The obvious choice,
    pelvis->chest, is foreshortened by exactly the amount the dog points at
    the camera -- on this clip it reads 76 px where the unforeshortened value
    would be ~186 px, and it breathes as the dog turns. That injects a
    spurious oscillation into every paw at once and buries the gait. Depth is
    foreshortening-free by construction.

Bias is deliberately toward missing contacts rather than inventing them: a
false stance pins a foot to a place it is not and jumps the root, which is
much harder to recover from downstream than a missing one. But not without
limit: a frame with no foot down has no anchor at all, so pushing the
threshold down until stance disappears just moves the failure somewhere less
visible.

Choosing the threshold on dog_1, by sweep (m/s -> duty on the walking segment,
fraction of walking frames with zero feet down):

    0.25 -> 0.42, 22%      0.40 -> 0.65,  3%
    0.30 -> 0.50, 10%      0.45 -> 0.71,  1%
    0.35 -> 0.56,  7%      0.60 -> 0.82,  0%

0.40 is the default. A walking dog's duty factor is 0.6-0.75 (below ~0.5 is a
trot), so 0.25 was badly under-detecting -- 22% of walking frames with nothing
planted is not conservatism, it is a broken anchor. 0.60 gives 0.82, which is
slower than this dog is moving. The standing and milling segments do not
discriminate here: raising the threshold always pushes them toward 1.0.

Caveat worth keeping in view: the speed noise floor is ~0.11 m/s, measured
where the dog is provably standing still, so 0.40 is only ~3.6x the floor.
This detector is marginal on this clip by construction. The real fix is better
paw localisation -- an independent 2D detector -- not a better threshold.
"""
from pathlib import Path
import argparse

import numpy as np


LEGS = ["FR", "FL", "RR", "RL"]
MIN_SEGMENT_S = 0.05     # matches retarget/postprocess.py::MIN_SEGMENT_S


def lowpass(x, fps, cutoff):
    from scipy.signal import butter, filtfilt
    if cutoff <= 0 or cutoff >= 0.5 * fps:
        return x.copy()
    b, a = butter(2, cutoff / (0.5 * fps), btype="low")
    flat = x.reshape(len(x), -1)
    if len(x) <= 3 * max(len(a), len(b)):
        return x.copy()
    return filtfilt(b, a, flat, axis=0).reshape(x.shape)


def runs(mask):
    """Yield (start, end, value) for consecutive runs in a 1-D bool array."""
    if len(mask) == 0:
        return
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(mask)]])
    for s, e in zip(bounds[:-1], bounds[1:]):
        yield int(s), int(e), bool(mask[s])


def refine_contacts(contacts, min_len):
    """Merge away stance/swing runs shorter than min_len frames.

    Same semantics and same order as retarget/postprocess.py::refine_contacts:
    short swing gaps are filled first (a one-frame liftoff inside a stance is
    detector noise), then short stance blips are dropped.
    """
    out = contacts.copy()
    for leg in range(out.shape[1]):
        col = out[:, leg]
        for value in (False, True):
            for s, e, val in runs(col):
                if val == value and e - s < min_len:
                    col[s:e] = not value
    return out


def project(points_cam, focal, W, H):
    z = np.maximum(points_cam[..., 2], 1e-6)
    return np.stack([focal * points_cam[..., 0] / z + W / 2.0,
                     focal * points_cam[..., 1] / z + H / 2.0], axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True, help="Phase B npz")
    p.add_argument("--out", required=True, help="contacts npz")
    p.add_argument("--thresh", type=float, default=0.40,
                   help="stance if paw speed is below this, in m/s; see the "
                        "module docstring for how this was calibrated")
    p.add_argument("--metres-per-unit", type=float, default=1.133,
                   help="metres per SMAL unit, fitted to a breed-typical "
                        "shoulder height. Only used by this module's own CLI; "
                        "the pipeline solves scale in capture.world_place_ba")
    p.add_argument("--speed-cutoff", type=float, default=8.0,
                   help="Hz; low-pass on the speed signal before thresholding")
    p.add_argument("--plot", type=str, default=None, help="diagnostic png")
    p.add_argument("--video", type=str, default=None,
                   help="source video, for stance-coloured overlay frames")
    p.add_argument("--overlay-dir", type=str, default=None)
    p.add_argument("--overlay-every", type=int, default=120)
    args = p.parse_args()

    d = np.load(args.infer, allow_pickle=True)
    fps = float(d["fps"])
    N = int(d["num_frames"])
    valid = d["valid"]
    paw_uv = d["paw_uv"]                                  # (N,4,2)
    W, H = [int(v) for v in d["img_size"]]
    focal = float(d["focal_full"])

    # Pixels per model unit at the dog's depth. Smoothed: the weak-perspective
    # depth is noisy per frame, and it only needs to track the dog walking
    # toward the camera, which is a sub-Hz signal.
    pts_cam = d["points_local"] + d["root_model"][:, None, :] + d["cam_t"][:, None, :]
    depth = np.maximum(lowpass(pts_cam[:, 0, 2][:, None], fps, 1.0)[:, 0], 1e-3)
    px_per_unit = focal / depth

    # Paw velocity: pixels/s -> model units/s -> m/s.
    vel = np.gradient(paw_uv, axis=0) * fps                # (N,4,2) px/s
    speed_px = np.linalg.norm(vel, axis=-1)
    vspeed_px = np.abs(vel[..., 1])
    to_mps = args.metres_per_unit / px_per_unit[:, None]
    speed = lowpass(speed_px * to_mps, fps, args.speed_cutoff)
    vspeed = lowpass(vspeed_px * to_mps, fps, args.speed_cutoff)

    raw = (speed < args.thresh) & valid[:, None]
    min_len = max(2, int(round(MIN_SEGMENT_S * fps)))
    contacts = refine_contacts(raw, min_len)

    # ---- report ---------------------------------------------------------
    print(f"clip {str(d['source'])}: {N} frames @ {fps:.3f} fps, "
          f"flicker merge < {min_len} frames ({MIN_SEGMENT_S*1000:.0f} ms)")
    q = np.percentile(speed[valid], [5, 25, 50, 75, 95])
    print(f"\npaw speed (m/s) percentiles over valid frames")
    print(f"  5% {q[0]:.3f}   25% {q[1]:.3f}   50% {q[2]:.3f}   "
          f"75% {q[3]:.3f}   95% {q[4]:.3f}      threshold {args.thresh:.3f}")

    print(f"\nduty factor (fraction of valid frames in stance)")
    for i, leg in enumerate(LEGS):
        print(f"  {leg}   raw {raw[valid, i].mean():.3f}   "
              f"merged {contacts[valid, i].mean():.3f}")

    nfeet = contacts.sum(axis=1)
    print(f"\nfeet down per frame -- decides which solver branch Phase D takes")
    for nf in range(5):
        frac = float((nfeet[valid] == nf).mean())
        note = {0: "flight: interpolate", 1: "under-determined",
                2: "reduced yaw+translation solve", 3: "full Kabsch",
                4: "full Kabsch"}[nf]
        print(f"  {nf} feet  {100*frac:5.1f}%   {note}")

    changed = int((raw != contacts).sum())
    print(f"\nflicker merge changed {changed} labels "
          f"({100.0*changed/max(1, raw.size):.2f}%)")

    # ---- plot -----------------------------------------------------------
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(N) / fps
        fig, ax = plt.subplots(3, 1, figsize=(14, 9),
                               gridspec_kw={"height_ratios": [2, 2, 1.2]})
        for i, leg in enumerate(LEGS):
            ax[0].plot(t, speed[:, i], lw=0.8, label=leg)
        ax[0].axhline(args.thresh, color="k", ls="--", lw=1,
                      label=f"threshold {args.thresh}")
        ax[0].set_ylim(0, float(np.percentile(speed[valid], 99.0)))
        ax[0].set_ylabel("paw speed (m/s)")
        ax[0].legend(ncol=5, fontsize=8)
        ax[0].set_title(f"{str(d['source'])} — Phase C contact detection")

        for i, leg in enumerate(LEGS):
            for s, e, v in runs(contacts[:, i]):
                if v:
                    ax[1].barh(i, (e - s) / fps, left=s / fps, height=0.7,
                               color="tab:green")
        ax[1].set_yticks(range(4))
        ax[1].set_yticklabels(LEGS)
        ax[1].set_ylim(-0.6, 3.6)
        ax[1].invert_yaxis()
        ax[1].set_ylabel("stance (gait diagram)")

        ax[2].fill_between(t, 0, ~valid, step="mid", color="tab:red", alpha=0.6,
                           label="no detection")
        ax[2].plot(t, nfeet / 4.0, lw=0.8, color="tab:blue", label="feet down / 4")
        ax[2].set_ylim(0, 1.05)
        ax[2].set_xlabel("time (s)")
        ax[2].legend(fontsize=8, ncol=2)
        for a in ax:
            a.set_xlim(0, t[-1])
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")

    # ---- overlay --------------------------------------------------------
    if args.video and args.overlay_dir:
        import cv2
        outdir = Path(args.overlay_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(args.video)
        idx, written = -1, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx >= N or idx % args.overlay_every != 0:
                continue
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            for i, leg in enumerate(LEGS):
                u, v = paw_uv[idx, i]
                down = contacts[idx, i]
                col = (0, 220, 0) if down else (0, 0, 235)
                cv2.circle(frame, (int(round(u)), int(round(v))), 9, col,
                           -1 if down else 2)
                cv2.putText(frame, f"{leg}{'*' if down else ''}",
                            (int(u) + 11, int(v) - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(frame, "green filled = stance", (12, H - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(outdir / f"contact_{idx:05d}.png"), frame)
            written += 1
        cap.release()
        print(f"wrote {written} overlay frames to {outdir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        source=str(d["source"]), fps=fps, num_frames=N,
        contacts=contacts, contacts_raw=raw, valid=valid,
        speed=speed, vspeed=vspeed, px_per_unit=px_per_unit, depth=depth,
        threshold=float(args.thresh), min_segment_frames=min_len,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
