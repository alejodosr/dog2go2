"""Unit tests for phase-4 post-processing (retarget/postprocess.py).

Covers the acceptance criteria of §10 at the unit level: pinned stance feet
do not skate or penetrate the ground, contacts are de-flickered, smoothing
is applied before IK, and the world<->base foot transforms round-trip.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retarget import ik  # noqa: E402
from retarget import postprocess as pp  # noqa: E402

RNG = np.random.default_rng(0)


def test_runs():
    mask = np.array([1, 1, 0, 0, 0, 1, 0], dtype=bool)
    assert pp._runs(mask) == [(0, 2, True), (2, 5, False), (5, 6, True), (6, 7, False)]
    assert pp._runs(np.ones(3, dtype=bool)) == [(0, 3, True)]


def test_refine_contacts_removes_flicker():
    stance = np.ones(20, dtype=bool)
    stance[5] = False           # 1-frame liftoff inside a stance: filled
    stance[10:12] = False       # 2-frame gap: also below min_len, filled
    contacts = np.zeros((20, 4), dtype=bool)
    contacts[:, 0] = stance
    contacts[8, 1] = True       # isolated 1-frame stance blip: dropped
    refined = pp.refine_contacts(contacts, min_len=3)
    assert refined[:, 0].all()
    assert not refined[:, 1].any()
    assert not refined[:, 2:].any()


def test_refine_contacts_keeps_real_segments():
    contacts = np.zeros((30, 4), dtype=bool)
    contacts[5:15, 2] = True
    refined = pp.refine_contacts(contacts, min_len=3)
    np.testing.assert_array_equal(refined, contacts)


def test_foot_world_base_roundtrip():
    n = 50
    root_pos = RNG.normal(size=(n, 3))
    rot = Rotation.from_quat(_random_quats(n))
    foot_base = RNG.normal(size=(n, 4, 3))
    world = pp.foot_world_positions(root_pos, rot, foot_base)
    np.testing.assert_allclose(
        pp.foot_base_positions(root_pos, rot, world), foot_base, atol=1e-12
    )


def test_lowpass_attenuates_jitter_keeps_mean():
    fps = 60.0
    t = np.arange(300) / fps
    clean = np.sin(2 * np.pi * 1.0 * t)          # 1 Hz gait-scale signal
    noisy = clean + 0.1 * np.sin(2 * np.pi * 25.0 * t)  # 25 Hz jitter
    filtered = pp.lowpass(noisy[:, None], fps)[:, 0]
    # filtfilt has edge transients; judge the interior.
    assert np.abs(filtered - clean)[20:-20].max() < 0.02
    # Above-Nyquist cutoff degenerates to a copy.
    np.testing.assert_array_equal(pp.lowpass(noisy, fps, cutoff=40.0), noisy)


def test_smooth_rotations_handles_quat_sign_flips():
    fps = 60.0
    angles = np.linspace(0, np.pi / 4, 200)[:, None]
    rot = Rotation.from_euler("z", angles)
    q = rot.as_quat()
    q[::2] *= -1.0  # alternating sign convention, same rotations
    smoothed = pp.smooth_rotations(Rotation.from_quat(q), fps)
    err = (smoothed * rot.inv()).magnitude()
    assert err.max() < 1e-3


def test_ground_align_puts_stance_feet_on_ground():
    n = 40
    float_height = 0.05
    root_pos = np.tile([0.0, 0.0, 0.27 + float_height], (n, 1))
    foot_world = RNG.normal(scale=0.1, size=(n, 4, 3))
    contacts = RNG.random((n, 4)) < 0.5
    foot_world[..., 2][contacts] = pp.FOOT_RADIUS + float_height
    aligned_root, aligned_feet, offset = pp.ground_align(root_pos, foot_world, contacts)
    np.testing.assert_allclose(offset, float_height, atol=1e-12)
    np.testing.assert_allclose(aligned_feet[..., 2][contacts], pp.FOOT_RADIUS)
    np.testing.assert_allclose(aligned_root[:, 2], 0.27)


def test_pin_stance_feet_removes_skate():
    n, fps = 60, 60.0
    foot_world = np.zeros((n, 4, 3))
    foot_world[:, 0, 0] = np.linspace(0.0, 0.3, n)  # FR drifts forward
    foot_world[:, 0, 2] = 0.01
    contacts = np.zeros((n, 4), dtype=bool)
    contacts[20:40, 0] = True
    pinned = pp.pin_stance_feet(foot_world, contacts, blend=3)
    stance = pinned[20:40, 0]
    np.testing.assert_allclose(stance - stance[0], 0.0)    # no motion in stance
    np.testing.assert_allclose(stance[:, :2] - foot_world[20, 0, :2], 0.0)  # touch-down xy
    np.testing.assert_allclose(stance[:, 2], pp.FOOT_RADIUS)          # on the ground
    assert pp.skate_speed(pinned, contacts, fps) < 1e-9
    # Blend frames move monotonically toward/away from the pin; frames outside
    # the blend window are untouched.
    np.testing.assert_array_equal(pinned[:17], foot_world[:17])
    np.testing.assert_array_equal(pinned[43:], foot_world[43:])
    assert not np.array_equal(pinned[18, 0], foot_world[18, 0])


def test_pin_stance_feet_blend_respects_neighbor_stance():
    """A short swing between two stances never overwrites pinned frames."""
    n = 30
    foot_world = RNG.normal(size=(n, 4, 3))
    contacts = np.zeros((n, 4), dtype=bool)
    contacts[5:12, 1] = True
    contacts[14:22, 1] = True  # 2-frame swing < blend width
    pinned = pp.pin_stance_feet(foot_world, contacts, blend=3)
    np.testing.assert_allclose(pinned[5:12, 1] - pinned[5, 1], 0.0)
    np.testing.assert_allclose(pinned[14:22, 1] - pinned[14, 1], 0.0)


def test_pin_stance_feet_splits_merged_run():
    """A long stance run with real target travel re-steps instead of holding
    one pin (the dog_5 failure: relabel merged a footfall sequence into one
    10 s run and the single pin clamped the calf for 38.6% of frames)."""
    n = 100
    foot_world = np.zeros((n, 4, 3))
    foot_world[:, 0, 0] = np.linspace(0.0, 0.4, n)   # 0.4 m of travel
    contacts = np.zeros((n, 4), dtype=bool)
    contacts[:, 0] = True
    pinned = pp.pin_stance_feet(foot_world, contacts, blend=3, split_min=30)
    # the pin follows the free target within the split budget (+ glide)
    err = np.abs(pinned[:, 0, 0] - foot_world[:, 0, 0])
    assert err.max() < pp.PIN_SPLIT_M + 1e-9
    # several distinct footfalls, each held constant at ground height
    pins = np.unique(pinned[:, 0, 0])
    assert len(pins) > 3
    np.testing.assert_allclose(pinned[:, 0, 2], pp.FOOT_RADIUS)


def test_pin_stance_feet_short_run_never_splits():
    """Runs at or under split_min keep the single pin no matter the drift —
    a fast-skating foot in a short relabeled stance (canter) must freeze."""
    n = 30
    foot_world = np.zeros((n, 4, 3))
    foot_world[:, 0, 0] = np.linspace(0.0, 1.0, n)   # violent drift
    contacts = np.zeros((n, 4), dtype=bool)
    contacts[5:25, 0] = True
    pinned = pp.pin_stance_feet(foot_world, contacts, blend=3, split_min=20)
    stance = pinned[5:25, 0]
    np.testing.assert_allclose(stance - stance[0], 0.0)


def test_relabel_contacts_from_realized_height():
    """Low feet become stance regardless of speed; high feet stay swing."""
    n = 30
    feet = np.full((n, 4, 3), 0.10)
    feet[:, 0, 2] = pp.FOOT_RADIUS       # FR on the ground the whole clip
    feet[10:20, 1, 2] = 0.025            # FL touches for 10 frames
    relabeled = pp.relabel_contacts(feet, min_len=3)
    assert relabeled[:, 0].all()
    np.testing.assert_array_equal(relabeled[10:20, 1], True)
    assert not relabeled[:10, 1].any() and not relabeled[20:, 1].any()
    assert not relabeled[:, 2:].any()


def test_relabel_contacts_drops_flicker():
    feet = np.full((30, 4, 3), 0.10)
    feet[15, 2, 2] = 0.01                # 1-frame ground graze: not a stance
    assert not pp.relabel_contacts(feet, min_len=3).any()


def test_despike_dof_spreads_spike_minimally():
    fps, n = 50.0, 60
    q = np.zeros((n, 12))
    q[:, 3] = np.sin(np.arange(n) / fps * 2 * np.pi)      # feasible joint
    q[30:, 5] = 1.2                                       # 1-frame step: 60 rad/s
    out = pp.despike_dof(q, fps, vmax=40.0)
    vel = np.abs(np.diff(out, axis=0)) * fps
    assert vel.max() <= 40.0 + 1e-6
    # untouched joints come back bit-identical
    np.testing.assert_array_equal(out[:, 3], q[:, 3])
    # deformation is local and small: same endpoints, tiny total change
    np.testing.assert_allclose(out[0, 5], q[0, 5], atol=1e-9)
    np.testing.assert_allclose(out[-1, 5], q[-1, 5], atol=1e-6)
    assert np.abs(out[:, 5] - q[:, 5]).max() < 0.4
    assert (np.abs(out[:, 5] - q[:, 5]) > 1e-6).sum() <= 3


def test_despike_dof_noop_below_clamp():
    q = RNG.normal(scale=0.01, size=(40, 12)).cumsum(axis=0)
    np.testing.assert_array_equal(pp.despike_dof(q, fps=50.0), q)


def test_postprocess_end_to_end():
    """Synthetic stepping motion: post-processing kills skate, keeps limits."""
    n, fps = 120, 60.0
    root_pos = np.zeros((n, 3))
    # Small drift: feet stay reachable while pinned for the whole clip.
    root_pos[:, 0] = np.linspace(0.0, 0.12, n)
    root_pos[:, 2] = 0.30  # slightly floating: ground align must fix this
    root_rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))

    # Feet under the hips, world-fixed except a drift on FR to create skate.
    foot_base = np.tile(ik.HIP_OFFSETS + [0.0, 0.0, -0.26], (n, 1, 1))
    foot_world = root_pos[:, None, :] + foot_base
    foot_world[:, 1:, 0] = foot_world[0, 1:, 0]  # FL/RR/RL truly pinned in world
    contacts = np.ones((n, 4), dtype=bool)

    motion = {
        "fps": fps, "robot_type": "unitree_go2", "num_frames": n,
        "root_pos": root_pos, "root_rot": root_rot,
        "dof_pos": np.zeros((n, 12)), "foot_contacts": contacts,
        "source": "synthetic",
    }
    rot = Rotation.from_quat(root_rot)
    out, report = pp.postprocess(
        motion, pp.foot_base_positions(root_pos, rot, foot_world)
    )

    # FR's 0.12 m of drift inside one 2 s all-stance run exceeds the pin
    # split budget, so it re-steps once instead of freezing (the run is
    # longer than MAX_PIN_RUN_S); the other three feet stay exactly pinned.
    # Skate is therefore small but not zero.
    assert report["skate_after"] < 0.02
    # FR drifts 0.06 m/s; averaged over the 4 stance feet -> 0.015 m/s.
    assert report["skate_before"] > 0.01
    assert report["skate_after"] < report["skate_before"]
    assert report["clamp_rate"] < 0.02   # acceptance §10
    # Realized stance feet sit on the ground.
    feet = pp.foot_world_positions(
        out["root_pos"], Rotation.from_quat(out["root_rot"]), ik.fk(out["dof_pos"])
    )
    np.testing.assert_allclose(feet[..., 2], pp.FOOT_RADIUS, atol=1e-3)
    _, violated = ik.clamp_to_limits(out["dof_pos"])
    assert not violated.any()


def _random_quats(n):
    q = RNG.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)
