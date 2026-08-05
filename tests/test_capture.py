"""Tests for the capture stack — the pure-numpy parts of it.

Stages 1 and 2 need torch and a GPU, so they are not testable here. What IS
testable is the geometry every later stage rests on, and the contract the
retargeter consumes. Both run under this repo's own uv env, which is also a
standing check that `capture`'s torch dependency really is confined to the
function bodies that need it (see capture/__init__.py).
"""
import numpy as np
import pytest

from capture.contacts_ground import ground_positions, to_ground
from capture.parse_video import validate
from capture.world_place import camera_to_world


def synthetic_calibration(focal=686.4, W=832, H=468, height=0.58, tilt_deg=5.0):
    """Build H / H_inv exactly the way capture.depth_calib does.

    Same construction, from a plane we choose instead of one a depth model
    fitted, so the answer is known in advance.
    """
    t = np.radians(tilt_deg)
    n = np.array([0.0, -np.cos(t), -np.sin(t)])     # camera y is DOWN, so up is -y
    n /= np.linalg.norm(n)
    fwd = np.array([0.0, 0.0, 1.0])
    x_w = fwd - (fwd @ n) * n
    x_w /= np.linalg.norm(x_w)
    y_w = np.cross(n, x_w)
    o_c = height * (-n)
    K = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]])
    H_inv = K @ np.stack([x_w, y_w, o_c], axis=1)
    H_inv = H_inv / H_inv[2, 2]
    return np.linalg.inv(H_inv), H_inv, focal, W, H, height


def test_ground_homography_round_trips():
    """Metres -> pixels -> metres. GUIDELINES quotes ~1e-15 as healthy."""
    Hm, H_inv, *_ = synthetic_calibration()
    world = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [-0.7, 3.2]])
    px = (H_inv @ np.c_[world, np.ones(len(world))].T).T
    px = px[:, :2] / px[:, 2:3]
    assert np.abs(to_ground(Hm, px) - world).max() < 1e-9


def test_camera_to_world_recovers_the_height_it_was_built_from():
    """The whole metric scale hangs off this number."""
    Hm, H_inv, focal, W, H, height = synthetic_calibration()
    R_cw, C, ortho, mismatch = camera_to_world(H_inv, focal, W / 2.0, H / 2.0)
    assert C[2] == pytest.approx(height, abs=1e-6)
    assert mismatch < 1e-9
    assert np.abs(R_cw @ R_cw.T - np.eye(3)).max() < 1e-9


@pytest.mark.parametrize("k, expect", [(0.8, 0.6425), (1.3, 0.5056)])
def test_a_wrong_focal_length_moves_the_camera_height(k, expect):
    """FOCAL_RATIO is assumed, never measured, and it propagates into every
    metre. A 20-30% focal error moves the recovered camera height by 11-13%.
    `scale_mismatch` is the diagnostic that reveals it."""
    _, H_inv, focal, W, H, true_h = synthetic_calibration()
    _, C, ortho, mismatch = camera_to_world(H_inv, focal * k, W / 2.0, H / 2.0)
    assert C[2] == pytest.approx(expect, abs=1e-3)
    assert mismatch > 0.15
    # ...but `ortho` does NOT reveal it. depth_calib builds the in-plane axes
    # from the camera forward direction, which leaves the second axis exactly
    # along image x; scaling the focal then rescales both columns without
    # tilting them, so their dot product stays 0 whatever the focal is. Do not
    # reach for `ortho` to validate a focal length -- it is structurally blind.
    assert ortho < 1e-9


def test_ground_positions_masks_paws_near_the_horizon():
    """A paw near the horizon back-projects to tens of metres away; it must be
    masked, not silently averaged into a velocity. The guard is RELATIVE --
    points further than max_range from the batch median -- so it needs a batch,
    which is how the pipeline calls it (all frames, all four paws)."""
    Hm, _, _, W, H, _ = synthetic_calibration()
    uv = np.array([[W / 2.0, 400.0], [W * 0.4, 430.0],
                   [W * 0.6, 450.0], [W / 2.0, 170.0]])
    g, ok = ground_positions(Hm, uv)
    assert ok[:3].all()
    assert not ok[3]
    assert abs(g[3, 0]) > 60.0


def plausible_clip(n=40):
    """A minimal but contract-legal motion: standing still, feet on z=0."""
    root = np.zeros((n, 3))
    root[:, 0] = np.linspace(0, 1.5, n)
    root[:, 2] = 0.40
    chest = root + np.array([0.35, 0.0, 0.02])
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    mounts = np.array([[0.30, -0.10], [0.30, 0.10], [-0.20, -0.10], [-0.20, 0.10]])
    leg_root = np.zeros((n, 4, 3))
    toe = np.zeros((n, 4, 3))
    for i, (dx, dy) in enumerate(mounts):
        leg_root[:, i] = root + np.array([dx, dy, 0.0])
        toe[:, i, :2] = leg_root[:, i, :2]
    return {"fps": 30.0, "num_frames": n, "source": "unit_test",
            "root_pos": root, "root_rot_xyzw": quat,
            "chest_pos": chest, "chest_rot_xyzw": quat,
            "toe_pos": toe, "leg_root_pos": leg_root}


def test_validate_accepts_a_plausible_clip():
    assert validate(plausible_clip()) == []


def test_validate_rejects_a_bad_ground_plane():
    """The failure GUIDELINES names explicitly: toe z outside (-0.15, 0.4)
    means the recovered plane is wrong, and every metre downstream with it."""
    kp = plausible_clip()
    kp["toe_pos"] = kp["toe_pos"] + np.array([0.0, 0.0, -0.6])
    assert any("toe z" in p for p in validate(kp))


def test_validate_rejects_non_metric_units():
    kp = plausible_clip()
    for k in ("root_pos", "chest_pos", "toe_pos", "leg_root_pos"):
        kp[k] = kp[k] * 100.0                      # centimetres
    assert any("units" in p for p in validate(kp))
