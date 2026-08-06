"""Post-processing of retargeted motions (phase 4): what separates "demo"
from "usable".

Takes the raw retarget output (root trajectory + world-frame foot targets +
source-side contacts) and produces clean joint trajectories:

  1. Contact refinement: stance/swing segments shorter than a few frames are
     flicker from the threshold detector, not gait — merge them away.
  2. Smoothing: low-pass (Butterworth, ~7 Hz) on the foot targets and the
     root trajectory *before* IK. Smoothing joint angles after IK would drag
     stance feet around and reintroduce foot-skate; smoothing the targets and
     solving IK against the already-smooth root cannot.
  3. Ground alignment: shift everything in z so the median stance foot
     center sits at the foot-sphere radius (i.e. the sphere touches z=0).
  4. Foot-skate removal: during each stance segment the foot target is
     pinned to its touch-down xy at ground height, blended in/out over a few
     swing frames to avoid pops. Runs no single stance could produce
     (longer than MAX_PIN_RUN_S with more than PIN_SPLIT_M of target travel
     — footfall sequences the height-only relabel merged) are split into
     re-steps instead of held to one pin, which wrapped the leg around the
     moving trunk until the calf sat at its limit (dog_5: 38.6% clamped).
  5. IK + limit report: solve, clamp to the MJCF limits, report the clamp
     rate (a high rate means the scaling/offsets upstream are wrong).
  6. Contact relabel from the ROBOT's realized (post-IK) foot heights, then
     re-pin + re-IK with the corrected labels. The source-side detector's
     horizontal-speed gate fails at fast gaits (canter D1_010_KAN01_004:
     13% stance vs a real ~30-40%, the whole gallop burst labeled airborne
     while retargeted feet sat at ground height) — and those labels feed
     both the stance pinning here and the contact_match reward downstream.
  7. Despike: minimal-deformation velocity clamp on the final dof_pos. IK
     branch flips can jump a joint in a single frame (same clip: 1.33 rad
     in 20 ms = 66 rad/s); target smoothing (step 2) runs *before* IK, so
     nothing else catches them.
  8. Support-aware re-grounding (reground.py): sustained all-feet-elevated
     segments are hidden terrain flattened away by the retarget (jump clip:
     the dog stood on a ~0.1 m object for 1.7 s and the segment came out as
     impossible "flight"); root z is projected down so the lowest foot is
     back on the ground, contacts relabeled. No-op without hidden terrain.
  9. Speed-aware time warp (timewarp.py, feasibility projection v1): planar
     root speed above the Go2 tracking ceiling is projected out by local
     playback slowdown instead of trimming. No-op for clips under the cap.

Order matters: smooth first (a filter would smear the pins), then align,
then pin, then IK; relabel needs realized feet, so it triggers one more
pin + IK pass; despike runs on the joint trajectory the tracker sees; the
re-ground and warp go last, on the final §7 dict — re-ground before warp
so the warp's speed envelope sees the corrected support.

(A 10th stage — kinematic clearance projection for morphology-gap poses,
knees below the floor in deep crouches — was built and reverted 2026-07-23
after stage12: it fixed the sit posture but regressed canter 66→43% and
did not improve the jump. Design, measurements and the pose-gate idea are
documented in RESULTS.md "Stage12" sections for a future revisit.)
"""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from retarget import ik

FOOT_RADIUS = 0.022        # foot sphere size in the MJCF: center z at contact
CUTOFF_HZ = 7.0            # Butterworth low-pass cutoff (brief: ~6-8 Hz)
MIN_SEGMENT_S = 0.05       # stance/swing runs shorter than this are flicker
BLEND_S = 0.05             # skate-removal blend in/out duration (~3 frames @ 60)
# Realized-foot contact threshold: stance foot centers sit at FOOT_RADIUS
# (0.022) after pinning; the Isaac replay of accepted clips puts labeled
# stances at z <= 0.028 (2026-07-20 audit), so 0.030 splits cleanly.
CONTACT_RELABEL_Z = 0.030
# Reference joint-velocity clamp (rad/s). Not the actuator limit (30): the
# trot clip peaks at 40.8 rad/s raw yet tracks at 99.5% survival, so 40 is
# an empirically trackable *reference* demand; anything above it here has
# been an IK artifact, not motion.
DOF_VEL_CLAMP = 40.0
# Re-pin a stance foot once its free target has walked this far (m) from the
# pin. The relabel (step 6) is height-only, so a slow-stepping dog whose swing
# clearance sits under CONTACT_RELABEL_Z gets its swings relabeled as stance
# and whole footfall sequences merge into one run (dog_5: RR/RL became single
# 10 s runs while the trunk yawed ~50 deg; the pinned target ended 0.65 m from
# the free one, and the calf sat clamped at its straight limit 38.6% of
# frames). Genuine stances never approach this budget — their free-target
# drift is skate-sized (<5 cm) — so splitting only fires where a single pin
# is a lie. 0.08 m is ~0.2 leg lengths, matching world_place_ba's
# --split-budget reasoning for the identical footfall-merging problem.
PIN_SPLIT_M = 0.08
# ... but only in runs no single stance could be (same reasoning as
# reground.py's REGROUND_MIN_S: quadruped support phases are short). The
# canter's relabeled stances drift past the budget in <0.2 s — those are the
# skating feet the relabel+pin exists to freeze, and splitting them turned
# the pin into a glide (stance skate 0.067 -> 0.380 m/s). Measured split by
# duration: every footfall-merged run (dog_1/2/5) is >0.5 s, 10 of 12 fast
# canter drifters are <=0.5 s, and the walk's 0.33-0.43 s stances stay solid.
MAX_PIN_RUN_S = 0.5


def foot_world_positions(root_pos, root_rot, foot_base):
    """Base-frame foot points (N, 4, 3) -> world frame."""
    return root_pos[:, None, :] + _apply_per_leg(root_rot, foot_base)


def foot_base_positions(root_pos, root_rot, foot_world):
    """World-frame foot points (N, 4, 3) -> base frame."""
    return _apply_per_leg(root_rot.inv(), foot_world - root_pos[:, None, :])


def refine_contacts(contacts, min_len):
    """Merge away stance/swing runs shorter than min_len frames.

    Short swing gaps are filled first (a 1-frame liftoff inside a stance is
    detector noise, and filling it keeps the foot pinned), then short stance
    blips are dropped.
    """
    out = contacts.copy()
    for leg in range(out.shape[1]):
        col = out[:, leg]
        for value in (False, True):  # fill short swings, then short stances
            for start, end, val in _runs(col):
                if val == value and end - start < min_len:
                    col[start:end] = not value
    return out


def lowpass(x, fps, cutoff=CUTOFF_HZ):
    """Zero-phase Butterworth low-pass along axis 0."""
    if cutoff >= 0.5 * fps:
        return x.copy()
    b, a = butter(2, cutoff / (0.5 * fps))
    return filtfilt(b, a, x, axis=0)


def smooth_rotations(rot, fps, cutoff=CUTOFF_HZ):
    """Low-pass a Rotation sequence via its quaternions.

    Sign-continuous quaternions are filtered componentwise and renormalized —
    valid for smoothing because consecutive frames are close on the sphere.
    """
    q = rot.as_quat()
    flip = np.cumprod(np.where(np.sum(q[1:] * q[:-1], axis=-1) < 0, -1.0, 1.0))
    q[1:] *= flip[:, None]
    q = lowpass(q, fps, cutoff)
    return Rotation.from_quat(q / np.linalg.norm(q, axis=-1, keepdims=True))


def ground_align(root_pos, foot_world, contacts):
    """Shift root + feet in z so the median stance foot touches the ground.

    Returns the offset that was subtracted (positive = motion was floating).
    """
    stance_z = foot_world[..., 2][contacts]
    offset = np.median(stance_z) - FOOT_RADIUS
    root_pos = root_pos.copy()
    foot_world = foot_world.copy()
    root_pos[:, 2] -= offset
    foot_world[..., 2] -= offset
    return root_pos, foot_world, offset


def pin_stance_feet(foot_world, contacts, blend, split=PIN_SPLIT_M,
                    split_min=None):
    """Remove foot-skate: pin each stance segment to its touch-down point.

    During stance the target is (touch-down xy, ground height); the `blend`
    swing frames on either side interpolate between the free target and the
    pin so liftoff/touch-down don't pop. Blending only touches swing frames,
    so neighboring stance segments stay exactly pinned.

    With `split_min` set (frames), a run longer than that whose free target
    drifts more than `split` from the pin is split into a new footfall there,
    gliding to the new pin over `blend` frames at ground height (a quick
    re-step). See PIN_SPLIT_M / MAX_PIN_RUN_S for why both gates: one pin
    across a footfall sequence the relabel merged does not remove skate, it
    wraps the leg around the trunk until the IK clamps — but a short
    fast-drifting run is a skating foot the pin exists to freeze. The default
    (split_min=None) never splits.
    """
    out = foot_world.copy()
    n = len(out)
    for leg in range(out.shape[1]):
        stance = contacts[:, leg]
        for start, end, val in _runs(stance):
            if not val:
                continue
            bounds = [start]  # sub-run starts: one per footfall
            if split_min is not None and end - start > split_min:
                pin_xy = foot_world[start, leg, :2]
                for t in range(start + 1, end):
                    if np.linalg.norm(foot_world[t, leg, :2] - pin_xy) > split:
                        bounds.append(t)
                        pin_xy = foot_world[t, leg, :2]
            bounds.append(end)
            prev = None
            for s, e in zip(bounds[:-1], bounds[1:]):
                pin = np.array([*foot_world[s, leg, :2], FOOT_RADIUS])
                out[s:e, leg] = pin
                if prev is not None:  # glide between pins, staying grounded
                    for k in range(min(blend, e - s)):
                        w = (k + 1) / (blend + 1)
                        out[s + k, leg] = (1 - w) * prev + w * pin
                prev = pin
            pin_first = np.array([*foot_world[bounds[0], leg, :2], FOOT_RADIUS])
            for k in range(1, blend + 1):  # ease in before touch-down
                i = start - k
                if i < 0 or stance[i]:
                    break
                w = (blend + 1 - k) / (blend + 1)
                out[i, leg] = w * pin_first + (1 - w) * foot_world[i, leg]
            for k in range(1, blend + 1):  # ease out after liftoff
                i = end - 1 + k
                if i >= n or stance[i]:
                    break
                w = (blend + 1 - k) / (blend + 1)
                out[i, leg] = w * prev + (1 - w) * foot_world[i, leg]
    return out


def relabel_contacts(foot_realized, min_len):
    """Stance mask from the robot's realized foot heights (world frame).

    Height-only on purpose: the sim contact sensor fires on any touch, so
    a label derived from where the retargeted feet actually are matches
    what the sensor will report — unlike the source-side speed gate, which
    erases fast-gait stances (see module docstring, step 6).
    """
    return refine_contacts(foot_realized[..., 2] < CONTACT_RELABEL_Z, min_len)


def despike_dof(dof_pos, fps, vmax=DOF_VEL_CLAMP):
    """Clamp per-frame joint velocity to vmax with minimal deformation.

    Per joint, solves min ||q - q_ref||^2 s.t. |q[t+1] - q[t]| <= vmax/fps
    (BVLS on the frame-to-frame steps), so a spike is spread over adjacent
    frames while joints already under the clamp are returned bit-identical.
    """
    dt_max = vmax / fps
    out = dof_pos.copy()
    steps = np.diff(dof_pos, axis=0)
    lower = None
    for j in np.flatnonzero(np.abs(steps).max(axis=0) > dt_max):
        if lower is None:
            n = len(dof_pos) - 1
            lower = np.tril(np.ones((n, n)))
        res = lsq_linear(
            lower, dof_pos[1:, j] - dof_pos[0, j],
            bounds=(-dt_max, dt_max), method="bvls",
        )
        out[1:, j] = dof_pos[0, j] + np.cumsum(res.x)
    return out


def skate_speed(foot_world, contacts, fps):
    """Mean horizontal speed (m/s) of feet during stance — 0 means no skate.

    Only interior stance frames count: the central difference at a
    touch-down/liftoff frame picks up legitimate swing speed, not skate.
    """
    interior = contacts.copy()
    interior[1:] &= contacts[:-1]
    interior[:-1] &= contacts[1:]
    if not interior.any():
        return 0.0
    vel = np.gradient(foot_world[..., :2], axis=0) * fps
    return float(np.linalg.norm(vel, axis=-1)[interior].mean())


def postprocess(motion, foot_targets_base):
    """Full phase-4 pipeline on a raw retargeted motion (at the source fps).

    motion: §7-format dict from retarget_clip.
    foot_targets_base: (N, 4, 3) raw IK targets in the base frame.
    Returns (motion, report) with cleaned root/dof/contacts; report carries
    the clamp rate, ground offset and stance-foot skate speeds before/after.
    """
    fps = motion["fps"]
    rot = Rotation.from_quat(motion["root_rot"])
    foot_world = foot_world_positions(motion["root_pos"], rot, foot_targets_base)

    min_len = max(2, round(MIN_SEGMENT_S * fps))
    blend = max(2, round(BLEND_S * fps))
    contacts_src = refine_contacts(motion["foot_contacts"], min_len)
    skate_before = skate_speed(foot_world, contacts_src, fps)

    foot_world = lowpass(foot_world, fps)
    root_pos = lowpass(motion["root_pos"], fps)
    rot = smooth_rotations(rot, fps)

    root_pos, foot_world, ground_offset = ground_align(
        root_pos, foot_world, contacts_src
    )

    def solve(contacts):
        pinned = pin_stance_feet(foot_world, contacts, blend,
                                 split_min=round(MAX_PIN_RUN_S * fps))
        pinned[..., 2] = np.maximum(pinned[..., 2], FOOT_RADIUS)  # no swing dips
        dof_pos, violated = ik.clamp_to_limits(
            ik.ik(foot_base_positions(root_pos, rot, pinned))
        )
        return dof_pos, violated

    dof_pos, violated = solve(contacts_src)
    contacts = relabel_contacts(
        foot_world_positions(root_pos, rot, ik.fk(dof_pos)), min_len
    )
    if (contacts != contacts_src).any():  # corrected labels change the pins
        dof_pos, violated = solve(contacts)

    dof_vel_peak_raw = np.abs(np.diff(dof_pos, axis=0)).max() * fps
    despiked = despike_dof(dof_pos, fps)
    foot_realized = foot_world_positions(root_pos, rot, ik.fk(despiked))
    skate_after = skate_speed(foot_realized, contacts, fps)

    out = dict(motion)
    out["root_pos"] = root_pos
    out["root_rot"] = rot.as_quat()
    out["dof_pos"] = despiked
    out["foot_contacts"] = contacts

    # at call site: both modules import this module's helpers
    from retarget import reground, timewarp

    out, ground = reground.reground(out)
    out, warp = timewarp.timewarp(out)
    report = {
        "reground_segments": ground["segments"],
        "reground_max_offset": ground["max_offset"],
        "warp_min_rate": warp["min_rate"],
        "warp_slowed_fraction": warp["slowed_fraction"],
        "warp_duration": (warp["duration_before"], warp["duration_after"]),
        "warp_speed_peak_smooth": (
            warp["planar_speed_peak_before"][1],
            warp["planar_speed_peak_after"][1],
        ),
        "clamp_rate": violated.mean(),
        "violated": violated,
        "ground_offset": ground_offset,
        "skate_before": skate_before,
        "skate_after": skate_after,
        "contact_fraction_src": contacts_src.mean(),
        "contact_fraction": contacts.mean(),
        "contact_changed": (contacts != contacts_src).mean(),
        "dof_vel_peak_raw": dof_vel_peak_raw,
        "dof_vel_peak": np.abs(np.diff(despiked, axis=0)).max() * fps,
        "despike_max_dq": np.abs(despiked - dof_pos).max(),
    }
    return out, report


def _apply_per_leg(rot, points):
    """Apply one Rotation per frame to (N, 4, 3) points (scipy pairs the i-th
    rotation with the i-th vector only)."""
    return np.stack(
        [rot.apply(points[:, leg]) for leg in range(points.shape[1])], axis=1
    )


def _runs(mask):
    """Consecutive runs of a 1-D bool array as (start, end, value) triples."""
    edges = np.flatnonzero(np.diff(mask))
    bounds = np.concatenate([[0], edges + 1, [len(mask)]])
    return [(bounds[i], bounds[i + 1], bool(mask[bounds[i]]))
            for i in range(len(bounds) - 1)]
