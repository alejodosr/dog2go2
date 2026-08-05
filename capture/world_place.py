"""Phase D — put the dog in the world.

For a STATIC camera the usual per-frame rigid fit collapses. AniMer already
gives body->camera orientation every frame (`global_orient`, applied in
points_local), and the camera->world rotation is a single constant that the
ground calibration determines. So the only per-frame unknown is a 3-vector:
where the dog is. That is far better conditioned than running Kabsch on three
noisy foot correspondences, and it cannot silently absorb orientation error
into position the way a full fit can.

    p_world(i,t) = R_cw . points_local(i,t) + T(t)

with R_cw recovered from the homography by Zhang's construction: the plane
homography factors as K[r1 r2 t], so K^-1 H^-1 gives the first two columns of
the rotation up to scale and r3 follows from the cross product. How far r1 and
r2 are from orthonormal is a direct check on the focal length -- it is not
enforced, it is measured, and then reported.

T(t) comes from the stance feet: each planted paw says "the dog is positioned
such that this toe sits at that spot on the floor". With three or four feet
down the worst-fitting one is rejected rather than averaged in, per the
brief's §5.2 -- one bad contact corrupts a least-squares fit over three points
completely. Frames with nothing planted get constant-velocity interpolation.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from capture.contacts_2d import lowpass, runs, LEGS

# points_local layout: 0 root, 1 chest, 2-5 mounts FR FL RR RL, 6-9 toes
TOE0 = 6


def camera_to_world(H_inv, focal, cx, cy):
    """R_cw, camera world position, and the orthonormality residual."""
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ np.asarray(H_inv, float)
    n1, n2 = np.linalg.norm(M[:, 0]), np.linalg.norm(M[:, 1])
    lam = 2.0 / (n1 + n2)
    # The homography carries an arbitrary overall sign. Fix it by requiring the
    # plane to lie IN FRONT of the camera: the world origin, whose camera-frame
    # position is t, must have positive depth. Flipping columns instead (the
    # obvious-looking guard) is a no-op for the camera height -- it negates
    # R and t together and leaves -R^T t unchanged.
    if M[2, 2] < 0:
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    # How badly do the two recovered columns violate orthonormality? This is
    # the focal length's error made visible; 0 would mean f is exactly right.
    ortho = float(abs(np.dot(r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2))))
    scale_mismatch = float(abs(n1 - n2) / max(n1, n2))
    # nearest true rotation
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R_wc = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    C = -R_wc.T @ t
    if C[2] < 0:
        raise SystemExit(
            f"camera solved to {C[2]:.2f} m, i.e. below the floor. The world "
            "ground plane's X,Y axes are left-handed, so world Z comes out "
            "pointing down. Delete the clip's calibration json and re-run "
            "capture.depth_calib.")
    return R_wc.T, C, ortho, scale_mismatch


def solve_translation(pts_w, ground, contacts, reject=True):
    """T(t) from planted feet, dropping the single worst-fitting one.

    pts_w is the body already rotated into world orientation but not placed.
    Each stance foot i proposes T = ground_i - pts_w[toe_i]; with 3+ feet the
    proposals should agree, and the outlier is a bad contact or a bad
    keypoint rather than something to average in.
    """
    N = len(pts_w)
    T = np.full((N, 3), np.nan)
    nused = np.zeros(N, int)
    rejected = 0
    for t in range(N):
        s = np.flatnonzero(contacts[t])
        if len(s) == 0:
            continue
        prop = np.stack([
            np.array([ground[t, i, 0], ground[t, i, 1], 0.0]) - pts_w[t, TOE0 + i]
            for i in s])
        if reject and len(prop) >= 3:
            d = np.linalg.norm(prop - np.median(prop, axis=0), axis=1)
            keep = np.ones(len(prop), bool)
            keep[int(np.argmax(d))] = False
            prop = prop[keep]
            rejected += 1
        T[t] = prop.mean(axis=0)
        nused[t] = len(prop)
    return T, nused, rejected


def fill_flight(T, fps):
    """Constant-velocity interpolation through frames with nothing planted."""
    n = len(T)
    good = np.flatnonzero(np.isfinite(T[:, 0]))
    if len(good) < 2:
        raise SystemExit("fewer than two anchored frames -- contacts are unusable")
    out = T.copy()
    idx = np.arange(n)
    for c in range(3):
        out[:, c] = np.interp(idx, good, T[good, c])
    return out, n - len(good)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True)
    p.add_argument("--contacts", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metres-per-unit", type=float, default=1.133)
    p.add_argument("--focal", type=float, default=None,
                   help="calibrated focal px; defaults to calib['focal_px']. "
                        "NOT b['focal_full'] -- that is AniMer's assumed 5000 "
                        "px (14.6 deg FOV) and using it puts the camera below "
                        "the floor.")
    p.add_argument("--trim", default=None,
                   help="start,end frames to keep (the tail of dog_1 is unusable)")
    p.add_argument("--root-cutoff", type=float, default=4.0,
                   help="Hz low-pass on the recovered root path")
    p.add_argument("--no-penetration", action="store_true", default=True,
                   help="on unanchored frames, lift the root so no toe goes "
                        "below the floor (on by default)")
    p.add_argument("--allow-penetration", dest="no_penetration",
                   action="store_false")
    p.add_argument("--floor-tol", type=float, default=-0.02,
                   help="metres a toe may dip below z=0 before lifting")
    p.add_argument("--relabel-iters", type=int, default=1,
                   help="after placing the dog, every paw's HEIGHT above the "
                        "floor is known -- which is a direct test of contact, "
                        "unlike the speed proxy used to bootstrap it. Relabel "
                        "from height and re-place. 1 = no relabelling.")
    p.add_argument("--contact-height", type=float, default=0.06,
                   help="planted if the toe is below this fraction of leg "
                        "length above the floor")
    p.add_argument("--plot", default=None)
    args = p.parse_args()

    b = np.load(args.infer, allow_pickle=True)
    c = np.load(args.contacts, allow_pickle=True)
    cal = json.loads(Path(args.calib).read_text())
    fps = float(b["fps"])
    W, H_ = [int(v) for v in b["img_size"]]
    focal = args.focal or cal.get("focal_px")
    if not focal:
        raise SystemExit("no calibrated focal length: pass --focal, or put "
                         "focal_px in the clip's seed json and re-run "
                         "capture.depth_calib")
    focal = float(focal)

    pts = b["points_local"] * args.metres_per_unit          # (N,10,3) camera-oriented
    ground = c["ground"]                                     # (N,4,2) metres on floor
    contacts = c["contacts"].copy()
    valid = c["valid"].copy()

    paw_uv = c["paw_uv"] if "paw_uv" in c.files else b["paw_uv"]
    if args.trim:
        a, bb = [int(x) for x in args.trim.split(",")]
        sl = slice(a, bb)
        pts, ground, contacts, valid = pts[sl], ground[sl], contacts[sl], valid[sl]
        paw_uv = paw_uv[sl]
    N = len(pts)
    contacts &= valid[:, None]

    # --- constant camera->world rotation, from the calibration ------------
    R_cw, C, ortho, smis = camera_to_world(np.array(cal["H_inv"]), focal,
                                           W / 2.0, H_ / 2.0)
    pitch = np.degrees(np.arcsin(-R_cw[2, 2]))
    print("camera->world from the homography (Zhang)")
    print(f"  camera position      ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}) m")
    print(f"  height above floor   {C[2]:.2f} m")
    print(f"  orthonormality resid {ortho:.4f}   column-scale mismatch {smis:.4f}")
    print( "    (both would be 0 for a perfect focal length; this is the "
           "22% aspect\n     uncertainty showing up as a measurable quantity)")

    # --- place the dog ----------------------------------------------------
    pts_w = np.einsum("ij,nkj->nki", R_cw, pts)      # rotated, not yet placed
    leg_len = float(np.median(np.linalg.norm(pts[:, 2:4] - pts[:, 6:8], axis=-1)))
    h_tol = args.contact_height * leg_len

    for it in range(max(1, args.relabel_iters)):
        T, nused, nrej = solve_translation(pts_w, ground, contacts)
        anchored = np.isfinite(T[:, 0])
        if not anchored.any():
            raise SystemExit("no anchored frames -- contacts are unusable")
        T, nflight = fill_flight(T, fps)
        T = lowpass(T, fps, args.root_cutoff)
        if it == max(1, args.relabel_iters) - 1:
            break
        # The dog is now placed, so each toe's height above the floor is known.
        # That is a direct measurement of contact rather than the speed proxy
        # used to bootstrap, and it does not inherit the homography's
        # amplification of a lifted paw.
        toe_z = (pts_w + T[:, None, :])[:, TOE0:TOE0 + 4, 2]
        toe_z = toe_z - np.median(toe_z[contacts]) if contacts.any() else toe_z
        relab = (toe_z < h_tol) & valid[:, None]
        changed = float((relab != contacts).mean())
        print(f"  relabel pass {it+1}: contacts from height < {h_tol*100:.1f} cm"
              f"  ({100*changed:.1f}% of foot-frames changed, "
              f"duty {contacts.mean():.2f} -> {relab.mean():.2f})")
        contacts = relab

    # No-penetration, on unanchored stretches only. Where nothing is planted the
    # height is pure interpolation and nothing holds it up, so on dog_1's tail
    # (the dog turning away, 52% of frames with no foot down) feet sank 19 cm
    # through the floor. A foot below the floor is never right, so lift the root
    # just enough to stop it. Anchored frames are left alone -- there the height
    # is measured, and small negative excursions are honest noise that
    # retarget/postprocess.py::ground_align is built to absorb.
    lifted = 0.0
    if args.no_penetration:
        toe_z = (pts_w + T[:, None, :])[:, TOE0:TOE0 + 4, 2]
        deficit = np.maximum(0.0, args.floor_tol - toe_z.min(axis=1))
        deficit = np.where(anchored, 0.0, deficit)
        deficit = lowpass(deficit[:, None], fps, 2.0)[:, 0].clip(min=0.0)
        T = T + np.stack([np.zeros_like(deficit), np.zeros_like(deficit),
                          deficit], axis=-1)
        lifted = float((deficit > 1e-4).mean())

    world = pts_w + T[:, None, :]

    print(f"\nplacement over {N} frames")
    print(f"  anchored by >=1 planted foot   {100*anchored.mean():5.1f}%")
    print(f"  interpolated (nothing down)    {100*(1-anchored.mean()):5.1f}%"
          f"   ({nflight} frames)")
    print(f"  frames where a foot was rejected as an outlier  {nrej}")
    if args.no_penetration:
        print(f"  root lifted to stop floor penetration on {100*lifted:5.1f}% "
              f"of frames")

    # --- diagnostics ------------------------------------------------------
    root = world[:, 0]
    path = float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum())
    print(f"\ntrajectory")
    print(f"  path length (horizontal) {path:.2f} m over {N/fps:.1f} s"
          f"  -> mean speed {path/(N/fps):.2f} m/s")
    for ax, nm in zip(range(3), ["X across", "Y along", "Z up"]):
        print(f"  {nm:<9} range {world[:,0,ax].max()-world[:,0,ax].min():5.2f} m")

    toe_z = world[:, TOE0:TOE0 + 4, 2]
    st = contacts
    print(f"\nheights (metres)")
    print(f"  median root z            {np.median(root[:,2]):.3f}")
    print(f"  median stance toe z      {np.median(toe_z[st]) if st.any() else np.nan:.3f}"
          f"   (should be ~0 by construction)")
    print(f"  median swing toe z       "
          f"{np.median(toe_z[~st]) if (~st).any() else np.nan:.3f}"
          f"   (should be above stance)")

    # residual skate: how far a planted foot moves in world while planted
    skate = []
    for leg in range(4):
        for s, e, v in runs(contacts[:, leg]):
            if not v or e - s < 3:
                continue
            w = world[s:e, TOE0 + leg, :2]
            skate.append(np.linalg.norm(w - w[0], axis=1).max())
    print(f"  stance foot skate        "
          f"{np.median(skate) if skate else np.nan:.3f} m median over "
          f"{len(skate)} runs")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
        ax[0].plot(root[:, 0], root[:, 1], lw=1.4, color="k", label="root path")
        col = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
        for i, leg in enumerate(LEGS):
            m = contacts[:, i]
            ax[0].scatter(world[m, TOE0 + i, 0], world[m, TOE0 + i, 1], s=5,
                          c=col[i], alpha=.5, label=leg)
        ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
        ax[0].set_xlabel("X across (m)"); ax[0].set_ylabel("Y along (m)")
        ax[0].set_title("top-down: root path and stance footfalls")
        t = np.arange(N) / fps
        for ax_i, nm in zip(range(3), ["X across", "Y along", "Z up"]):
            ax[1].plot(t, root[:, ax_i], lw=1.1, label=nm)
        ax[1].legend(fontsize=8); ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("root position (m)"); ax[1].set_title("root over time")
        for i, leg in enumerate(LEGS):
            ax[2].plot(t, world[:, TOE0 + i, 2], lw=.9, c=col[i], label=leg)
        ax[2].axhline(0, color="k", lw=.8, ls="--")
        ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("toe height (m)")
        ax[2].set_title("toe height above the floor"); ax[2].legend(fontsize=8)
        for a_ in ax:
            a_.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"\nwrote {args.plot}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, source=str(b["source"]), fps=fps, num_frames=N,
                        world=world, contacts=contacts, valid=valid,
                        anchored=anchored, R_cw=R_cw, camera_pos=C,
                        paw_uv=paw_uv,   # carried through so the overlays draw
                                         # the point the contacts were built on

                        ortho_resid=ortho, metres_per_unit=args.metres_per_unit)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
