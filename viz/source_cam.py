"""Recover the SOURCE video's camera in a retargeted clip's pkl frame.

The capture pipeline solves the real camera (world_place: R_cw, camera_pos,
in dog-world metres) and the calibration knows the focal length. The
retargeter maps dog world -> pkl frame by a uniform scale about the origin
plus an xy shift (retarget_clip; no rotation), and uniform scaling preserves
the projected image when the camera position is scaled the same way. Scale
and shift are recovered from the data rather than re-deriving retarget's
internals: scale from the z-median ratio (robust to the 50 Hz resample),
shift from frame 0.

Pure numpy — imported by both viz/playback.py (MuJoCo render, uv env) and
a2g2_tracking's play.py (Isaac source camera, isaaclab env).
"""
import json
from pathlib import Path

import numpy as np


def source_camera_pose(world_npz, calib_json, motion):
    """-> (pos (3,), R (3,3), fovx_deg, fovy_deg, (W, H)).

    R is the camera rotation in the OpenGL/USD/MuJoCo convention: +x right,
    +y up, looking along -z. The capture convention is +x right, +y down,
    +z forward, hence the diag(1,-1,-1) flip.
    """
    w = np.load(world_npz, allow_pickle=True)
    cal = json.loads(Path(calib_json).read_text())
    origin_w = 0.5 * (w["world"][:, 0] + w["world"][:, 1])   # trunk origin, metres
    s = float(np.median(motion["root_pos"][:, 2]) / np.median(origin_w[:, 2]))
    t_xy = s * origin_w[0, :2] - motion["root_pos"][0, :2]

    C = w["camera_pos"]
    pos = np.array([s * C[0] - t_xy[0], s * C[1] - t_xy[1], s * C[2]])
    R = np.asarray(w["R_cw"]) @ np.diag([1.0, -1.0, -1.0])

    W_, H_ = cal["img_size"]
    f = float(cal["focal_px"])
    fovx = float(np.degrees(2.0 * np.arctan(W_ / (2.0 * f))))
    fovy = float(np.degrees(2.0 * np.arctan(H_ / (2.0 * f))))
    return pos, R, fovx, fovy, (int(W_), int(H_))
