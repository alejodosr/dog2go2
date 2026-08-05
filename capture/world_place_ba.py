"""Phase 2 (PLAN.md) — joint placement: one clip-wide least squares instead
of per-frame homography projection.

world_place.py treats ray-intersect-floor as a measurement: at a grazing
angle that asserts the along-ray position with full confidence in exactly the
direction the image barely constrains, and the error scales with
distance/camera-height (measured: 9.3 cm per unit d/h on the harness).

Here the floor positions are VARIABLES, and the image contributes what it
actually knows, in pixels:

  variables   root trajectory knots T (3K), one anchor (x,y) per stance run
              (z = 0 by construction), and log metric scale
  residuals   1. pixel reprojection of every paw observation through the
                 calibrated camera — a large along-ray floor shift is a tiny
                 pixel change, so the optimizer KNOWS that direction is weakly
                 measured and lets the priors carry it; the anisotropic
                 weighting nobody tunes falls out here
              2. stance: FK toe must sit at its run's anchor (and on z=0)
              3. smoothness: acceleration penalty on the root knots
              4. depth cue: the frozen shape makes the animal its own depth
                 ruler — AniMer's cam_t encodes apparent size, so
                 depth ~ s * (f/focal_full) * cam_t_z, an observation that is
                 INDEPENDENT of camera height and constrains exactly the
                 direction a grazing homography cannot. (focal_full is still
                 never used geometrically — it is only the known convention
                 cam_t is expressed in.) Measured on dog_1's standing segment,
                 the real cue is PRECISE BUT BIASED: 1% relative std, ~+35%
                 absolute — the dangerous kind, and a stability check alone
                 would have blessed it. So the cue enters as RELATIVE depth: a
                 per-clip bias factor is estimated jointly under a weak prior,
                 the cue shapes the trajectory in depth, and absolute scale
                 comes from the stance anchors and FK.
              5. no-penetration hinge on every toe

Scale is fitted inside the same problem (Phase 3): the baseline fitted it
from paw spread on the floor, which inherits the full amplification
(+184% at d/h 7.5 on the harness — the dominant failure).

Input/output contracts match world_place.py, so parse_video.py and the viz
tools work unchanged.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from capture.contacts_2d import lowpass, runs, LEGS
from capture.world_place import camera_to_world, solve_translation, fill_flight

TOE0 = 6


def knot_matrix(n, k):
    """(n, k) sparse-ish linear interpolation from k evenly spaced knots."""
    t = np.linspace(0.0, k - 1.0, n)
    i0 = np.clip(t.astype(int), 0, k - 2)
    w1 = t - i0
    A = np.zeros((n, k))
    A[np.arange(n), i0] = 1.0 - w1
    A[np.arange(n), i0 + 1] = w1
    return A, i0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True)
    p.add_argument("--contacts", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--knot-dt", type=float, default=0.15,
                   help="root knot spacing, seconds")
    p.add_argument("--sigma-px", type=float, default=8.0)
    p.add_argument("--sigma-fk", type=float, default=0.04, help="metres")
    p.add_argument("--sigma-fkz", type=float, default=0.015,
                   help="metres; stance-toe-on-floor tie. Much stiffer than "
                        "the xy tie: the body must RIDE ON its planted feet "
                        "(world_place.py enforces this exactly, per frame). "
                        "At 0.04 the depth cue, smoothness and hinge floated "
                        "the body 7-11 cm off its stance feet on tail frames "
                        "and the retargeter clamped the calves pinning them "
                        "back to the floor.")
    p.add_argument("--sigma-acc", type=float, default=4.0, help="m/s^2")
    p.add_argument("--sigma-depth", type=float, default=0.05,
                   help="relative; the apparent-size depth cue")
    p.add_argument("--sigma-depth-bias", type=float, default=0.3,
                   help="log-prior width on the per-clip depth-cue bias "
                        "factor; monocular absolute depth is not trusted")
    p.add_argument("--no-depth-cue", action="store_true")
    p.add_argument("--floor-tol", type=float, default=0.02)
    p.add_argument("--size-prior", type=float, default=0.50,
                   help="prior mean shoulder height in metres for a dog "
                        "(0 disables). Generic, not per-clip.")
    p.add_argument("--size-sigma", type=float, default=0.30,
                   help="log-space width of the size prior; 0.30 spans "
                        "roughly 27-90 cm at one sigma")
    p.add_argument("--min-lat-info", type=float, default=60.0,
                   help="frames*units^2 of lateral stance information below "
                        "which the lateral-anchor scale is not trusted")
    p.add_argument("--split-budget", type=float, default=0.25,
                   help="leg lengths of floor travel before a stance run is "
                        "split into a new footfall/anchor")
    p.add_argument("--falsify-z", type=float, default=0.0,
                   help="metres; a stance sample whose toe sits further than "
                        "this from the floor at the round-1 solution is a "
                        "false contact and is dropped for round 2")
    p.add_argument("--max-pin-frac", type=float, default=0.12,
                   help="skip output-pinning a run whose correction exceeds "
                        "this fraction of leg length (the leg cannot absorb "
                        "it; leave FK)")
    p.add_argument("--trim", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    b = np.load(args.infer, allow_pickle=True)
    c = np.load(args.contacts, allow_pickle=True)
    cal = json.loads(Path(args.calib).read_text())
    fps = float(b["fps"])
    W, H_ = [int(v) for v in b["img_size"]]
    focal = float(cal["focal_px"])
    F_conv = float(b["focal_full"])

    pl = b["points_local"]                      # (N,10,3) units, cam-oriented
    uv = c["paw_uv"]                            # refined sole pixels
    ground = c["ground"]
    contacts = (c["contacts"] & c["valid"][:, None]).copy()
    valid = c["valid"].copy()
    ct_obs = b["cam_t"] + b["root_model"]       # raw-frame root, units
    if args.trim:
        a0, a1 = [int(x) for x in args.trim.split(",")]
        sl = slice(a0, a1)
        pl, uv, ground, contacts, valid, ct_obs = (
            pl[sl], uv[sl], ground[sl], contacts[sl], valid[sl], ct_obs[sl])
    N = len(pl)

    R_cw, C, ortho, _ = camera_to_world(np.array(cal["H_inv"]), focal,
                                        W / 2.0, H_ / 2.0)
    R_wc = R_cw.T
    print(f"camera: height {C[2]:.2f} m, ortho resid {ortho:.4f}")

    pl_w = np.einsum("ij,nkj->nki", R_cw, pl)   # world-ORIENTED, units
    toe_w_u = pl_w[:, TOE0:TOE0 + 4]            # (N,4,3)

    # ---- initialization: the baseline path --------------------------------
    T0, _, _ = solve_translation(pl_w * 1.0, ground, contacts)  # unit scale
    # crude initial scale from the depth cue at the anchored frames
    anch = np.isfinite(T0[:, 0])
    depth_pred_u = ((T0[anch] - C) @ R_wc.T)[:, 2]
    depth_obs_u = ct_obs[anch, 2] * focal / F_conv
    s0 = float(np.median(depth_pred_u / np.maximum(depth_obs_u, 1e-6)))
    s0 = float(np.clip(s0, 0.3, 3.0))
    T0, _ = fill_flight(T0 * s0, fps)
    T0 = lowpass(T0, fps, 4.0)

    # ---- stance runs -> anchor index per (frame, leg) ----------------------
    # One anchor assumes one FOOTFALL. The kinematic detector over-merges
    # stance on slow gaits (a milling cat), so a single run can contain real
    # toe travel — pinning that to one anchor stretched the emitted leg to
    # 1.05x its own FK length (p99) and the retargeter clamped the calves at
    # 10%. Split a run whenever its floor track wanders more than a fraction
    # of the leg length (converted to metres with the rough depth-cue scale;
    # a +26% scale error just loosens the budget by the same amount, which
    # is tolerable) from the current segment's start.
    leg_len_u = float(np.median(np.linalg.norm(
        pl[:, 2:6] - pl[:, TOE0:TOE0 + 4], axis=-1)))
    budget_m = args.split_budget * leg_len_u * s0
    run_id = np.full((N, 4), -1, int)
    anchors0 = []
    nsplit = 0
    for leg in range(4):
        for s_, e_, v_ in runs(contacts[:, leg]):
            if not v_:
                continue
            seg = s_
            for t_ in range(s_, e_ + 1):
                split_now = False
                if t_ < e_:
                    gt = ground[t_, leg]
                    ref = ground[seg, leg]
                    if (np.isfinite(gt).all() and np.isfinite(ref).all()
                            and np.linalg.norm(gt - ref) > budget_m):
                        split_now = True
                if t_ == e_ or split_now:
                    g0 = ground[seg:t_, leg]
                    g0 = g0[np.isfinite(g0).all(axis=1)]
                    if len(g0) > 0:
                        run_id[seg:t_, leg] = len(anchors0)
                        anchors0.append(np.median(g0, axis=0))
                    else:
                        contacts[seg:t_, leg] = False
                    if split_now:
                        nsplit += 1
                    seg = t_
    S = len(anchors0)
    if nsplit:
        print(f"  split {nsplit} over-merged stance runs "
              f"(> {args.split_budget:.2f} leg lengths = {budget_m:.2f} m "
              f"of floor travel)")
    anchors0 = np.asarray(anchors0, float).reshape(S, 2)
    if S == 0:
        raise SystemExit("no stance runs -- contacts are unusable")

    K = max(4, int(round(N / (args.knot_dt * fps))) + 1)
    A, i0 = knot_matrix(N, K)
    dt_k = (N - 1) / (K - 1) / fps
    # least-squares init of knots from T0
    knots0, *_ = np.linalg.lstsq(A, T0, rcond=None)

    st_t, st_leg = np.nonzero(contacts)
    st_run = run_id[st_t, st_leg]
    sw_mask = (~contacts) & valid[:, None] & np.isfinite(uv).all(axis=-1)
    sw_t, sw_leg = np.nonzero(sw_mask)
    print(f"{N} frames, {K} knots, {S} anchors; "
          f"{len(st_t)} stance obs, {len(sw_t)} swing obs")

    def unpack(x):
        kn = x[:3 * K].reshape(K, 3)
        an = x[3 * K:3 * K + 2 * S].reshape(S, 2)
        bd = np.exp(x[-1])                       # depth-cue bias factor
        return kn, an, bd

    def proj(pw):
        cam = (pw - C) @ R_wc.T
        z = np.maximum(cam[..., 2], 1e-6)
        return np.stack([focal * cam[..., 0] / z + W / 2.0,
                         focal * cam[..., 1] / z + H_ / 2.0], axis=-1)

    hinge_w = 1.0 / 0.03      # softer than the stance-z tie: penetration is
                              # honest noise downstream absorbs (world_place
                              # docstring); a stiff hinge levitates the body
    d_obs_u = ct_obs[:, 2] * focal / F_conv     # depth in units of s
    # No-penetration on EVERY frame. A flight-only hinge (world_place.py's
    # choice) was tried and measured worse on all three real clips through
    # the retargeter (dog_1 clamp 0.61% -> 5.83%): with the hinge off during
    # stance, swing toes near stance boundaries dive below the floor and the
    # retargeter's ground alignment chases them.
    flight_t = np.arange(N)

    # Scale is deliberately NOT a least-squares variable. Every residual it
    # multiplies also multiplies the FK noise, so the optimizer can always
    # shave cost by shrinking the animal — measured as a stable -8..-25%
    # scale bias (classic errors-in-variables attenuation, worsened by the
    # Huber knee weakening the geometric restoring force). Instead the LS is
    # solved with s FROZEN, and s re-estimated between solves from
    # TIME-AVERAGED stance geometry: mean anchor separation over mean FK
    # separation per leg pair. Averaging before the ratio is what removes the
    # attenuation; the loop converges in 2-3 rounds.
    # per-stance-sample weight; round 2 zeroes samples round 1 falsified
    w_st = np.ones(len(st_t))

    def residuals(x, s):
        kn, an, bd = unpack(x)
        T = A @ kn                               # (N,3)
        toes = s * toe_w_u + T[:, None, :]       # (N,4,3) world, metres

        r = []
        # 1. pixel reprojection: stance toes at their anchors, swing toes at FK
        a3 = np.concatenate([an[st_run], np.zeros((len(st_run), 1))], axis=1)
        r.append((w_st[:, None] * (proj(a3) - uv[st_t, st_leg])
                  / args.sigma_px).ravel())
        r.append(((proj(toes[sw_t, sw_leg]) - uv[sw_t, sw_leg])
                  / args.sigma_px).ravel())
        # 2. stance FK: toe at anchor, sole on the floor
        r.append((w_st[:, None] * (toes[st_t, st_leg, :2] - an[st_run])
                  / args.sigma_fk).ravel())
        r.append((w_st * toes[st_t, st_leg, 2] / args.sigma_fkz).ravel())
        # 3. smoothness
        acc = (kn[:-2] - 2 * kn[1:-1] + kn[2:]) / dt_k ** 2
        r.append((acc / args.sigma_acc).ravel())
        # 4. apparent-size depth cue, relative (bias estimated jointly)
        if not args.no_depth_cue:
            d_pred = ((T - C) @ R_wc.T)[:, 2]
            r.append((d_pred - s * bd * d_obs_u)
                     / (args.sigma_depth * np.maximum(d_pred, 0.5)))
            r.append(np.array([x[-1] / args.sigma_depth_bias]))
        # 5. no-penetration, flight frames only
        r.append(np.maximum(0.0, -toes[flight_t, :, 2] - args.floor_tol)
                 .ravel() * hinge_w)
        return np.concatenate(r)

    # Lateral direction: horizontal, perpendicular to the viewing direction.
    # Pixel rays pin floor points STIFFLY in this direction (f/z px per metre)
    # and softly along the view (f*h/z^2) — at grazing geometry the FK ties
    # overpower the rays along-view and drag the anchors inward, so along-view
    # separations carry the very bias being estimated. Scale is therefore
    # read from the lateral projection only.
    view2 = R_wc[2, :2] / max(np.linalg.norm(R_wc[2, :2]), 1e-9)
    lat2 = np.array([-view2[1], view2[0]])

    def estimate_scale(an):
        """s from time-averaged stance geometry, lateral projection only.

        Also returns the lateral information (frames * units^2): a walk ALONG
        the viewing axis leaves only the left-right paw separation in the
        lateral direction and the estimate degenerates — measured on cat_1 /
        dog_2 (info 33 / 19 vs dog_1's 114), where it landed +20% and the
        retargeter's clamp rate called it out."""
        num = den = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                m = (run_id[:, i] >= 0) & (run_id[:, j] >= 0)
                if m.sum() < 5:
                    continue
                a_l = float((an[run_id[m, i]] - an[run_id[m, j]]).mean(0) @ lat2)
                d_l = float((toe_w_u[m, i, :2] - toe_w_u[m, j, :2]).mean(0) @ lat2)
                n = float(m.sum())
                num += n * a_l * d_l
                den += n * d_l * d_l
        return (num / den if den > 0 else None), den

    # ---- sparsity pattern --------------------------------------------------
    from scipy.sparse import lil_matrix
    npar = 3 * K + 2 * S + 1                     # ... , log_bd
    rows = []

    def dep_T(t):
        j = i0[t]
        return [3 * j + c_ for c_ in range(3)] + [3 * (j + 1) + c_ for c_ in range(3)]

    def dep_anchor(rid):
        return [3 * K + 2 * rid, 3 * K + 2 * rid + 1]

    for t, rid in zip(st_t, st_run):             # stance px (2 rows each)
        d = dep_anchor(rid)
        rows += [d, d]
    for t in sw_t:                               # swing px
        d = dep_T(t)
        rows += [d, d]
    for t, rid in zip(st_t, st_run):             # stance FK xy
        d = dep_T(t) + dep_anchor(rid)
        rows += [d, d]
    for t in st_t:                               # stance FK z
        rows.append(dep_T(t))
    for k_ in range(K - 2):                      # smoothness
        d = [3 * (k_ + o) + c_ for o in range(3) for c_ in range(3)]
        rows += [d, d, d]
    if not args.no_depth_cue:
        for t in range(N):
            rows.append(dep_T(t) + [npar - 1])
        rows.append([npar - 1])                  # bias prior
    for t in flight_t:                           # hinge, 4 toes
        d = dep_T(t)
        rows += [d, d, d, d]

    Jsp = lil_matrix((len(rows), npar), dtype=bool)
    for ri, cols in enumerate(rows):
        Jsp[ri, cols] = True
    Jsp = Jsp.tocsr()

    from scipy.optimize import least_squares
    x0 = np.concatenate([knots0.ravel(), anchors0.ravel(), [0.0]])
    r0 = residuals(x0, s0)
    assert len(r0) == len(rows), (len(r0), len(rows))

    # Scale from the RAY-ONLY anchors, before the LS ever touches them: the
    # homography's lateral coordinates are unbiased by the height/grazing
    # amplification (that error is along-view), and fitted anchors would
    # already be dragged toward the FK prediction, feeding the very bias
    # being estimated back in. Measured on the harness: this estimator lands
    # within -11%..+2% of truth across every camera, where the baseline
    # paw-spread fit spans +17%..+184%. Iterating on fitted anchors gives
    # the SAME value on good geometry and drifts on bad — so: once, ray-only.
    s_hat, lat_info = estimate_scale(anchors0)
    # fallback for lateral-starved clips: the floor paw-spread fit. It
    # carries the d/h amplification (harness: +17% at 2.5x), but on the
    # along-view real clips it beat the starved lateral estimate by the
    # retargeter's own clamp check. The right fix is an independent 2D paw
    # detector (PLAN.md, out of scope); until then the divergence between
    # the two estimates is reported as the scale's honesty bound.
    from capture.contacts_ground import body_length_m
    body_m = body_length_m(np.array(cal["H"]), uv)
    paws_u = pl[:, TOE0:TOE0 + 4]
    spread_u = float(np.median(np.linalg.norm(
        paws_u.max(axis=1) - paws_u.min(axis=1), axis=-1)))
    s_spread = body_m / max(spread_u, 1e-9)
    # ---- scale as a 1-D MAP in log space --------------------------------
    # Three estimators that disagree (dog_2: 1.17 / 3.08 / 1.27) plus a weak
    # BIOLOGICAL prior. The prior is what humans get for free — a person is
    # 1.7 m +/- 10%, which is why their pipelines need no floor. A dog is
    # 25-75 cm, far broader, but broad still beats a 3x disagreement, and it
    # is generic: no fine-tuning, no per-clip tuning, true of any dog.
    #
    # It also arbitrates between CALIBRATIONS: on dog_2 a ZoeDepth-derived
    # plane implies a 40 cm shoulder for an obviously large retriever, and
    # the prior rejects it; on dog_1 every calibration implies ~54 cm and the
    # prior stays silent. Sigmas come from measured behaviour: lateral-anchor
    # was -12%..+1% on the harness, floor-spread +17%..+184%, and the depth
    # cue carried a +35% bias on dog_1.
    shoulder_u = float(np.median(pl_w[:, 2:6, 2] - pl_w[:, TOE0:TOE0 + 4, 2]))
    obs = []
    if s_hat and s_hat > 0:
        obs.append((np.log(s_hat),
                    0.08 * np.sqrt(max(args.min_lat_info * 2.0, 1.0)
                                   / max(lat_info, 1.0))))
    if s_spread > 0:
        obs.append((np.log(s_spread), 0.60))
    if s0 > 0:
        obs.append((np.log(s0), 0.35))
    if args.size_prior > 0 and shoulder_u > 1e-6:
        obs.append((np.log(args.size_prior / shoulder_u), args.size_sigma))
    num = sum(m / sg ** 2 for m, sg in obs)
    den = sum(1.0 / sg ** 2 for m, sg in obs)
    s = float(np.clip(np.exp(num / den), 0.2, 4.0))
    print("scale MAP inputs (log-space mean +/- sigma):")
    for (m, sg), nm in zip(obs, ["lateral-anchor", "floor-spread", "depth-cue",
                                 "size-prior"][:len(obs)]):
        print(f"    {nm:<15} {np.exp(m):6.3f}  sigma {sg:.2f}"
              f"   -> shoulder {np.exp(m) * shoulder_u * 100:5.1f} cm")
    print(f"    MAP            {s:6.3f}         "
          f"   -> shoulder {s * shoulder_u * 100:5.1f} cm"
          f"   (lateral info {lat_info:.0f})")
    res = least_squares(residuals, x0, args=(s,), jac_sparsity=Jsp,
                        method="trf", loss="huber", f_scale=2.5,
                        max_nfev=120, verbose=2 if args.verbose else 0)
    kn, an, bd = unpack(res.x)
    T = A @ kn

    # A stance claim the solve itself contradicts (toe far off the floor at
    # the solution) is a FALSE contact — the cat's 3-6 cm swing lift sits at
    # the mesh-noise floor, so the detector over-labels. Drop those samples
    # and solve once more; the freed toes follow FK (higher, faster) and the
    # retargeter's own contact gate then correctly ignores them.
    toe_z1 = (s * toe_w_u + T[:, None, :])[st_t, st_leg, 2]
    bad = (np.abs(toe_z1) > args.falsify_z) if args.falsify_z > 0 \
        else np.zeros(len(toe_z1), bool)
    if bad.any():
        w_st[bad] = 0.0
        contacts[st_t[bad], st_leg[bad]] = False
        run_id[st_t[bad], st_leg[bad]] = -1
        print(f"  round 2: dropped {bad.sum()} of {len(bad)} stance samples "
              f"(toe further than {args.falsify_z * 100:.0f} cm from the "
              f"floor at the round-1 solution)")
        res = least_squares(residuals, res.x, args=(s,), jac_sparsity=Jsp,
                            method="trf", loss="huber", f_scale=2.5,
                            max_nfev=120, verbose=2 if args.verbose else 0)
        kn, an, bd = unpack(res.x)
        T = A @ kn
    print(f"solver: {res.status} ({res.nfev} evals), "
          f"cost {0.5 * (r0 ** 2).sum():.0f} -> {res.cost:.0f}")
    print(f"scale: {s:.4f} m/unit   depth-cue bias {bd:.3f}")

    world = s * pl_w + T[:, None, :]
    # Pin stance toes to their anchors in the OUTPUT as well, crossfaded a
    # few frames into swing. The solver already believes the anchors; leaving
    # the raw FK drift in the emitted stance toes hands the retargeter's
    # post-processing 4 cm of 1-3 Hz wander to re-litigate (measured: its
    # ground alignment shifted +51 mm and clamp exploded on cat_1).
    fade = max(2, int(round(0.05 * fps)))
    npin_skipped = 0
    for leg in range(4):
        r_ = run_id[:, leg]
        t0 = 0
        while t0 < N:                            # blocks of constant run id
            if r_[t0] < 0:
                t0 += 1
                continue
            t1 = t0
            while t1 < N and r_[t1] == r_[t0]:
                t1 += 1
            s_, e_, rid = t0, t1, r_[t0]
            t0 = t1
            tgt = np.array([an[rid, 0], an[rid, 1], 0.0])
            delta = tgt[None, :] - world[s_:e_, TOE0 + leg]
            # a correction bigger than the leg can absorb means the anchor
            # and the FK disagree fundamentally — stretching the leg to the
            # anchor hands the IK an unreachable target; leave FK alone
            if np.median(np.linalg.norm(delta, axis=1)) > \
                    args.max_pin_frac * leg_len_u * s:
                npin_skipped += 1
                continue
            world[s_:e_, TOE0 + leg] += delta
            for k_, t_ in enumerate(range(max(0, s_ - fade), s_)):
                w_ = (k_ + 1) / (fade + 1)
                world[t_, TOE0 + leg] += w_ * delta[0]
            for k_, t_ in enumerate(range(e_, min(N, e_ + fade))):
                w_ = 1.0 - (k_ + 1) / (fade + 1)
                world[t_, TOE0 + leg] += w_ * delta[-1]
    if npin_skipped:
        print(f"  left {npin_skipped} runs un-pinned (correction beyond "
              f"{args.max_pin_frac:.2f} leg lengths)")
    anchored = contacts.any(axis=1)

    root = world[:, 0]
    path = float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum())
    print(f"trajectory: path {path:.2f} m over {N / fps:.1f} s; "
          f"root z median {np.median(root[:, 2]):.3f} m")
    toe_z = world[:, TOE0:TOE0 + 4, 2]
    print(f"stance toe z median {np.median(toe_z[contacts]):.3f} m; "
          f"swing {np.median(toe_z[~contacts]):.3f} m")
    skate = []
    for leg in range(4):
        for s_, e_, v_ in runs(contacts[:, leg]):
            if v_ and e_ - s_ >= 3:
                w_ = world[s_:e_, TOE0 + leg, :2]
                skate.append(np.linalg.norm(w_ - w_[0], axis=1).max())
    print(f"stance foot skate median {np.median(skate) if skate else np.nan:.3f} m "
          f"over {len(skate)} runs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, source=str(b["source"]), fps=fps, num_frames=N,
                        world=world, contacts=contacts, valid=valid,
                        anchored=anchored, R_cw=R_cw, camera_pos=C,
                        paw_uv=uv, ortho_resid=ortho,
                        metres_per_unit=float(s),
                        anchors=an, run_id=run_id, solver="ba")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
