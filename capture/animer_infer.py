"""Phase B — AniMer video inference -> canonical body-frame skeleton + paw pixel tracks.

Milestone 2, first stage. Reads a video, runs AniMer per frame, and writes an
intermediate npz that the rest of the video path (contacts_2d, world_place,
parse_video) consumes as pure numpy. This is NOT the Milestone 1 npz contract —
that is produced later by parse_video.py.

Runs under the perception env (`$PY_CAPTURE`), not this repo's uv env, because
it needs torch + detectron2 + smplx, which the mujoco env deliberately has
none of. That makes this stage a subprocess boundary whose output is a plain
npz:

    $PY_CAPTURE -m capture.animer_infer \
        --video media/dog_1.mov --out <outdir>/dog_1_animer.npz

AniMer's `amr` package is vendored at the repo root, so no checkout of it is
needed; only the checkpoint ($ANIMER_CKPT) and `data/smal/` stay external.
Everything downstream of this stage is pure numpy/scipy and needs no GPU.

Three things happen to the raw AniMer output here, the first two from brief §4.1:

  1. beta is frozen to the per-clip median. AniMer regresses shape per frame
     and it wobbles; a varying beta is a varying skeleton, which makes the
     world-placement scale term jitter. Measured before/after — see the
     `beta_wobble_*` diagnostics.
  2. theta is smoothed in 6D rotation representation before FK. Not axis-angle:
     wraparound produces artifacts.
  3. cam_t is re-fitted so the projected mesh bbox tracks the DETECTOR bbox
     (see bbox_align_camt). AniMer's pred_cam scale fails on foreshortened
     poses — on dog_4's vertical jump tz spiked +23% over 6 frames while the
     detector held the dog throughout, which downstream reads as the dog
     teleporting 1.5 m away mid-jump (cam_t_z is world_place_ba's depth cue,
     and paw_uv is projected through cam_t). The correction is normalized to
     the clip median, so frames where AniMer and the detector agree are
     untouched; measured on dog_4 the corrected tz is flat through the jump.

The joints we need are NOT what the model hands back. `pred_keypoints_3d` is 26
*surface landmarks* (SMAL.forward overwrites smal_output.joints with them), so
the skeletal points come from J_regressor @ vertices, and the paws come from
the unnamed vertex-group table. See FK_* and PAW_KP below.
"""
from pathlib import Path
import argparse
import contextlib
import io
import os
import sys

import numpy as np

from capture import paths


# --- Canonical skeletal points, by name, from name2id35 in smal_warapper.py ---
# The J_regressor is (35, 3889); these index its rows.
FK_ROOT = 0     # 'root'   -- pelvis. Matches Milestone 1's BVH "Hips".
FK_CHEST = 6    # 'spine3' -- trunk front: parent of Neck and of both front legs.
                #             Matches Milestone 1's BVH "Spine1".
# Leg roots in canonical FR, FL, RR, RL order. Right is -y in the SMAL frame.
FK_MOUNTS = [
    11,  # FR <- 'RLeg1'      right shoulder
    7,   # FL <- 'LLeg1'      left shoulder
    21,  # RR <- 'RLegBack1'  right hip
    17,  # RL <- 'LLegBack1'  left hip
]

# Paw tips are NOT joints -- the regressor terminates at hock/carpus. They are
# hand-picked vertex groups in smal_warapper.keypoint_vertices_idx, which ships
# with no names attached. Identified by nearest rest joint, LBS weight
# dominance, and the 26-keypoint skeleton edge list in mesh_renderer.py:79-81,
# where the chains read shoulder->elbow->wrist->paw (12->8->14->3) and
# hip->stifle->hock->paw (7->10->16->5). Left is +y.
#
# NOTE the front pair is transposed relative to natural reading order. This is
# exactly the FR/FL swap the npz contract validator exists to catch.
PAW_KP = [
    4,  # FR -- vertices 3188, 3156, 2327, 3183   template (+0.453, -0.134, -0.458)
    3,  # FL -- vertices  360, 1203, 1235, 1230   template (+0.453, +0.134, -0.458)
    6,  # RR -- vertices 3854, 2820, 3852, 3858   template (-0.321, -0.122, -0.512)
    5,  # RL -- vertices 1976, 1974, 1980,  856   template (-0.321, +0.123, -0.512)
]

LEG_ORDER = ["FR", "FL", "RR", "RL"]

# Output point order, matching the flat 10-point convention the npz contract
# validator uses: root, chest, then the four mounts, then the four toes.
POINT_NAMES = (["root", "chest"]
               + [f"{leg}_mount" for leg in LEG_ORDER]
               + [f"{leg}_toe" for leg in LEG_ORDER])

# Bones used for the beta-wobble diagnostic: rigid segments of the FK skeleton.
# (parent, child) row pairs into J_regressor @ V.
DIAG_BONES = [
    (0, 6), (6, 15), (15, 16),          # pelvis->chest->neck->head
    (7, 8), (8, 9), (9, 10),            # left front:  shoulder/elbow/wrist/foot
    (11, 12), (12, 13), (13, 14),       # right front
    (17, 18), (18, 19), (19, 20),       # left rear
    (21, 22), (22, 23), (23, 24),       # right rear
]


# --------------------------------------------------------------------------
# 6D rotation representation. Matches amr.utils.geometry.rot6d_to_rotmat:
# the 6 numbers are the first two COLUMNS of R, recovered by Gram-Schmidt.
# Implemented here rather than imported so the pair round-trips by
# construction and cannot drift from AniMer's convention.
# --------------------------------------------------------------------------
def rotmat_to_6d(R):
    """(..., 3, 3) -> (..., 6)"""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_rotmat(x):
    """(..., 6) -> (..., 3, 3), re-orthonormalized."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def probe_fps(path):
    """True frame rate from the container's r_frame_rate.

    cv2.CAP_PROP_FPS is not trustworthy here: on dog_1.mov it reports 58.861
    (the container's avg_frame_rate, skewed by an edit list) while every actual
    inter-frame delta is 0.016667 s -- a real 60.0. A 1.9% error would bias
    every paw velocity and the final 50 Hz resample, so prefer ffprobe and fall
    back to OpenCV only if it is unavailable.
    """
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        num, den = r.stdout.strip().split("/")
        fps = float(num) / float(den)
        return fps if fps > 0 else None
    except Exception:
        return None


def interp_gaps(x, valid):
    """Linearly fill invalid rows of (N, ...) along axis 0. Edges held."""
    x = x.copy()
    if valid.all():
        return x
    idx = np.arange(len(x))
    good = idx[valid]
    if len(good) == 0:
        raise SystemExit("no frame produced a detection")
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(idx, good, flat[good, c])
    return flat.reshape(x.shape)


def lowpass(x, fps, cutoff):
    """Zero-phase Butterworth along axis 0, on a (N, ...) array."""
    from scipy.signal import butter, filtfilt
    if cutoff <= 0 or cutoff >= 0.5 * fps:
        return x.copy()
    b, a = butter(2, cutoff / (0.5 * fps), btype="low")
    flat = x.reshape(len(x), -1)
    if len(x) <= 3 * max(len(a), len(b)):
        return x.copy()
    return filtfilt(b, a, flat, axis=0).reshape(x.shape)


def bbox_align_camt(verts, camt, det_box, valid, focal, W, H, fps, cutoff,
                    max_scale=1.4):
    """Re-fit cam_t so the projected mesh bbox tracks the detector bbox.

    The detector and AniMer see the same crop, but the detector's box stays on
    the animal in exactly the poses where pred_cam's scale drifts (strong
    foreshortening, rearing). Matching absolute boxes would be wrong — the
    mesh bbox systematically overshoots the detector's (tail, ears), and that
    offset is pose-dependent — so both the height ratio and the centre offset
    are normalized by their clip medians: the correction is IDENTITY on the
    frames where the two agree, and only the per-frame DEVIATION from the
    typical relationship is corrected. Targets are gap-filled over undetected
    frames and low-passed at the theta cutoff before solving, so the
    correction cannot re-introduce per-frame jitter.

    Scale moves tz (height ~ focal/tz), centre moves tx, ty. Solved by
    fixed-point iteration on the true perspective projection; three rounds
    converge to sub-pixel. Returns (camt, tz_ratio) with tz_ratio clipped to
    [1/max_scale, max_scale].
    """
    camt = camt.copy()
    tz0 = camt[:, 2].copy()

    def mesh_bbox(ct):
        vc = verts + ct[:, None, :]
        z = np.maximum(vc[..., 2], 1e-6)
        u = focal * vc[..., 0] / z + W / 2.0
        v = focal * vc[..., 1] / z + H / 2.0
        return (np.stack([(u.min(1) + u.max(1)) / 2,
                          (v.min(1) + v.max(1)) / 2], axis=-1),
                v.max(1) - v.min(1))

    ctr, h = mesh_bbox(camt)
    det_ctr = (det_box[:, :2] + det_box[:, 2:]) / 2.0
    det_h = det_box[:, 3] - det_box[:, 1]
    ratio = h / np.maximum(det_h, 1e-6)
    ratio_norm = np.median(ratio[valid])
    h_tgt = det_h * ratio_norm
    ctr_tgt = det_ctr + np.median((ctr - det_ctr)[valid], axis=0)
    h_tgt = lowpass(interp_gaps(h_tgt, valid), fps, cutoff)
    ctr_tgt = lowpass(interp_gaps(ctr_tgt, valid), fps, cutoff)

    for _ in range(3):
        ctr, h = mesh_bbox(camt)
        camt[:, 2] = np.clip(camt[:, 2] * h / np.maximum(h_tgt, 1e-6),
                             tz0 / max_scale, tz0 * max_scale)
        ctr, h = mesh_bbox(camt)
        zbar = camt[:, 2] + verts[..., 2].mean(axis=1)
        camt[:, :2] += (ctr_tgt - ctr) * zbar[:, None] / focal
    return camt, camt[:, 2] / tz0


def hf_power_fraction(x, fps, split_hz):
    """Fraction of variance above split_hz, averaged over channels.

    Ground-truth-free jitter measure. Real dog motion is a few Hz; anything
    well above that is per-frame regression noise.
    """
    flat = x.reshape(len(x), -1)
    flat = flat - flat.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(flat, axis=0)) ** 2
    freq = np.fft.rfftfreq(len(flat), d=1.0 / fps)
    total = spec.sum(axis=0)
    hi = spec[freq > split_hz].sum(axis=0)
    ok = total > 0
    return float(np.mean(hi[ok] / total[ok])) if ok.any() else 0.0


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="intermediate npz path")
    p.add_argument("--checkpoint", default=paths.ANIMER_CKPT,
                   help="the 8.35 GB vith checkpoint (default $ANIMER_CKPT); "
                        "the 2.7 GB one declares BACKBONE.TYPE=vit and the "
                        "loader rejects it")
    p.add_argument("--max-side", type=int, default=1280,
                   help="downscale so the longest side is at most this (0 = native)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--fps", type=float, default=None,
                   help="override the source frame rate (see probe_fps)")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--det-thresh", type=float, default=0.7)
    p.add_argument("--theta-cutoff", type=float, default=6.0,
                   help="Butterworth cutoff (Hz) for theta smoothing in 6D")
    p.add_argument("--no-bbox-align", dest="bbox_align", action="store_false",
                   default=True,
                   help="disable re-fitting cam_t to the detector bbox "
                        "(see bbox_align_camt)")
    p.add_argument("--jitter-split", type=float, default=10.0,
                   help="frequency above which power counts as jitter")
    p.add_argument("--debug-frames", type=str, default=None,
                   help="directory to write paw-overlay frames for eyeballing")
    p.add_argument("--debug-every", type=int, default=60)
    args = p.parse_args()

    paths.resolve(args.checkpoint, "AniMer checkpoint (--checkpoint)")
    paths.resolve(paths.SMAL_MODEL, "SMAL model file (data/smal/)")
    # chdir below would reinterpret any relative path the caller gave us.
    args.video = str(Path(args.video).resolve())
    args.out = str(Path(args.out).resolve())
    if args.debug_frames:
        args.debug_frames = str(Path(args.debug_frames).resolve())

    # The checkpoint's hydra config stores SMAL.MODEL_PATH as "data/smal/..."
    # relative to the working directory, so the loader only finds it from the
    # repo root. run_default.sh already cds here; this makes the stage work
    # when invoked from anywhere else too.
    os.chdir(paths.REPO_ROOT)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import cv2
    import torch
    import torch.utils.data
    from amr.models import load_amr
    from amr.utils import recursive_to
    from amr.utils.renderer import cam_crop_to_full
    from amr.datasets.vitdet_dataset import ViTDetDataset
    from amr.models.smal_warapper import keypoint_vertices_idx

    from capture.detector import build_detector, detect_animals

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading AniMer from {args.checkpoint}", flush=True)
    model, model_cfg = load_amr(args.checkpoint)
    model = model.to(device).eval()
    detector = build_detector(args.det_thresh)
    J_reg = model.smal.J_regressor.detach().cpu().numpy()      # (35, 3889)
    assert J_reg.shape[0] == 35, f"unexpected J_regressor shape {J_reg.shape}"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    src_fps = args.fps or probe_fps(args.video) or cap.get(cv2.CAP_PROP_FPS) or 30.0
    cv_fps = cap.get(cv2.CAP_PROP_FPS)
    if cv_fps and abs(cv_fps - src_fps) / src_fps > 0.005:
        print(f"note: using {src_fps:.3f} fps; OpenCV reported {cv_fps:.3f}", flush=True)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0
    if args.max_side > 0 and max(src_w, src_h) > args.max_side:
        scale = args.max_side / max(src_w, src_h)
    W = int(round(src_w * scale)) // 2 * 2
    H = int(round(src_h * scale)) // 2 * 2
    focal_full = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(W, H)
    fps = src_fps / max(1, args.frame_stride)

    debug_dir = Path(args.debug_frames) if args.debug_frames else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    go_raw, pose_raw, betas_raw, camt_raw, valid, frame_idx = [], [], [], [], [], []
    box_raw = []
    debug_cache = {}

    idx = -1
    n_read = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.frame_stride != 0:
            continue
        if args.max_frames and n_read >= args.max_frames:
            break
        n_read += 1

        if (W, H) != (src_w, src_h):
            frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes = detect_animals(detector, frame_rgb, args.det_thresh, 1)
        frame_idx.append(idx)

        if len(boxes) == 0:
            valid.append(False)
            go_raw.append(np.eye(3)[None])
            pose_raw.append(np.tile(np.eye(3), (34, 1, 1)))
            betas_raw.append(np.zeros(41))
            camt_raw.append(np.zeros(3))
            box_raw.append(np.full(4, np.nan))
            continue
        box_raw.append(boxes[0].astype(np.float64))

        ds = ViTDetDataset(model_cfg, frame_rgb, boxes)
        with contextlib.redirect_stdout(io.StringIO()):   # it prints per crop
            batch = torch.utils.data.default_collate([ds[0]])
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        sp = out["pred_smal_params"]
        go_raw.append(sp["global_orient"][0].detach().cpu().numpy().reshape(1, 3, 3))
        pose_raw.append(sp["pose"][0].detach().cpu().numpy().reshape(34, 3, 3))
        betas_raw.append(sp["betas"][0].detach().cpu().numpy().reshape(-1))
        camt_raw.append(cam_crop_to_full(out["pred_cam"],
                                         batch["box_center"].float(),
                                         batch["box_size"].float(),
                                         batch["img_size"].float(),
                                         focal_full)[0].detach().cpu().numpy())
        valid.append(True)

        if debug_dir is not None and (n_read - 1) % args.debug_every == 0:
            debug_cache[n_read - 1] = frame_bgr.copy()

        if n_read % 50 == 0:
            print(f"{Path(args.video).name}: {n_read} frames, "
                  f"{int(np.sum(valid))} detected", flush=True)
    cap.release()

    valid = np.asarray(valid, dtype=bool)
    N = len(valid)
    if N < 4:
        raise SystemExit(f"only {N} frames read")
    print(f"read {N} frames at {fps:.3f} fps, {valid.sum()} with a detection "
          f"({100.0 * valid.mean():.1f}%)", flush=True)

    go_raw = np.asarray(go_raw)            # (N, 1, 3, 3)
    pose_raw = np.asarray(pose_raw)        # (N, 34, 3, 3)
    betas_raw = np.asarray(betas_raw)      # (N, 41)
    camt_raw = np.asarray(camt_raw)        # (N, 3)
    box_raw = np.asarray(box_raw)          # (N, 4) x0 y0 x1 y1, nan on miss

    # ---- diagnostic: how much was beta wobbling? -------------------------
    # Bone lengths depend only on beta (they are rest-skeleton distances), so
    # their spread across frames is a direct read on shape instability -- the
    # thing freezing beta is meant to remove. After freezing it is exactly 0 by
    # construction, which is why this has to be measured on the raw output.
    import torch as _t
    with _t.no_grad():
        Jrest = _t.einsum("jv,nvc->njc",
                          model.smal.J_regressor.cpu(),
                          model.smal.v_template.cpu()[None]
                          + _t.einsum("vcb,nb->nvc",
                                      model.smal.shapedirs.cpu(),
                                      _t.from_numpy(betas_raw[valid]).float()))
    Jrest = Jrest.numpy()
    bone_len = np.stack([np.linalg.norm(Jrest[:, c] - Jrest[:, pa], axis=-1)
                         for pa, c in DIAG_BONES], axis=1)      # (Nvalid, B)
    cv = bone_len.std(axis=0) / np.maximum(bone_len.mean(axis=0), 1e-9)
    beta_wobble_med = float(np.median(cv))
    beta_wobble_max = float(cv.max())

    # ---- freeze beta to the per-clip median ------------------------------
    betas_frozen = np.median(betas_raw[valid], axis=0)

    # ---- smooth theta in 6D, after filling detection gaps ----------------
    go6 = interp_gaps(rotmat_to_6d(go_raw), valid)
    pose6 = interp_gaps(rotmat_to_6d(pose_raw), valid)
    camt = interp_gaps(camt_raw, valid)

    jit_before = hf_power_fraction(pose6, fps, args.jitter_split)
    go6_s = lowpass(go6, fps, args.theta_cutoff)
    pose6_s = lowpass(pose6, fps, args.theta_cutoff)
    camt_s = lowpass(camt, fps, args.theta_cutoff)
    jit_after = hf_power_fraction(pose6_s, fps, args.jitter_split)

    go_s = rot6d_to_rotmat(go6_s)          # (N, 1, 3, 3)
    pose_s = rot6d_to_rotmat(pose6_s)      # (N, 34, 3, 3)

    # ---- re-run FK with frozen beta and smoothed theta -------------------
    print("re-running FK with frozen beta", flush=True)
    from amr.models.smal_warapper import SMALLayer
    verts = np.empty((N, 3889, 3), dtype=np.float32)
    joints = np.empty((N, 35, 3), dtype=np.float32)
    with _t.no_grad():
        bt = _t.from_numpy(betas_frozen).float()[None].to(device)
        for s in range(0, N, 64):
            e = min(N, s + 64)
            # Call SMALLayer.forward unbound, NOT model.smal(...). The SMAL
            # subclass overwrites smal_output.joints with the 26 surface
            # landmarks and throws the rigid LBS joints away, and
            # `J_regressor @ posed_vertices` is not a substitute: posed
            # vertices carry pose-corrective blendshapes and skinning, which
            # leaked ~1% of length variation into supposedly rigid bones.
            o = SMALLayer.forward(model.smal,
                                  betas=bt.expand(e - s, -1),
                                  global_orient=_t.from_numpy(go_s[s:e]).float().to(device),
                                  pose=_t.from_numpy(pose_s[s:e]).float().to(device),
                                  pose2rot=False)
            verts[s:e] = o.vertices.cpu().numpy()
            joints[s:e] = o.joints.cpu().numpy()
    paws = np.stack([verts[:, keypoint_vertices_idx[k], :].mean(axis=1)
                     for k in PAW_KP], axis=1)                      # (N, 4, 3)
    skel = joints[:, [FK_ROOT, FK_CHEST] + FK_MOUNTS, :]            # (N, 6, 3)
    points = np.concatenate([skel, paws], axis=1)                   # (N, 10, 3)

    # Root-centre: separate the pose from the (meaningless, weak-perspective)
    # translation. Phase D recovers world placement; it does not come from here.
    root = points[:, 0:1, :].copy()
    points_local = points - root

    # ---- re-fit cam_t to the detector bbox (docstring point 3) -----------
    tz_ratio = np.ones(N)
    if args.bbox_align:
        camt_s, tz_ratio = bbox_align_camt(verts, camt_s, box_raw, valid,
                                           focal_full, W, H, fps,
                                           args.theta_cutoff)

    # ---- paw pixel tracks ------------------------------------------------
    # Project the same paw points that Phase D will use in the body frame, so
    # the world side and body side of the rigid fit are the SAME physical
    # point. Plain pinhole, principal point at image centre, +z forward --
    # matching amr.utils.geometry.perspective_projection.
    paw_cam = paws + camt_s[:, None, :]
    z = np.maximum(paw_cam[..., 2], 1e-6)
    paw_uv = np.stack([focal_full * paw_cam[..., 0] / z + W / 2.0,
                       focal_full * paw_cam[..., 1] / z + H / 2.0], axis=-1)

    # ---- debug overlay ---------------------------------------------------
    if debug_dir is not None:
        colors = [(0, 0, 255), (0, 200, 255), (255, 0, 0), (255, 200, 0)]
        for fi, img in debug_cache.items():
            vis = img.copy()
            for li in range(4):
                u, v = paw_uv[fi, li]
                cv2.circle(vis, (int(round(u)), int(round(v))), 7, colors[li], -1)
                cv2.putText(vis, LEG_ORDER[li], (int(u) + 9, int(v) - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[li], 2)
            cv2.imwrite(str(debug_dir / f"paws_{fi:05d}.png"), vis)
        print(f"wrote {len(debug_cache)} debug frames to {debug_dir}", flush=True)

    # ---- report ----------------------------------------------------------
    print()
    print("=== Phase B diagnostics " + "=" * 40)
    print(f"  frames                     {N} at {fps:.3f} fps")
    print(f"  detection rate             {100.0 * valid.mean():.1f}%")
    print(f"  beta wobble (bone CV)      median {100 * beta_wobble_med:.2f}%  "
          f"max {100 * beta_wobble_max:.2f}%   <- removed by freezing")
    print(f"  theta jitter >{args.jitter_split:.0f} Hz       "
          f"{100 * jit_before:.2f}% -> {100 * jit_after:.2f}% of variance")
    if args.bbox_align:
        dev = np.abs(tz_ratio - 1.0)
        print(f"  bbox tz correction         median {100 * np.median(dev):.1f}%  "
              f"max {100 * dev.max():.1f}%   "
              f"({int((dev > 0.05).sum())} frames beyond 5%)")
    # FK sanity: a SINGLE bone is rigid once beta is frozen, so its spread must
    # be ~0. pelvis->chest is NOT a valid check -- it spans joints 0..6 through
    # the articulated spine, so it legitimately varies as the dog flexes.
    rigid = np.linalg.norm(joints[:, 8] - joints[:, 7], axis=-1)   # LLeg1->LLeg2
    tl = np.linalg.norm(points[:, 1] - points[:, 0], axis=-1)
    print(f"  rigid bone (shoulder-elbow) {rigid.mean():.4f}  "
          f"std {rigid.std():.1e}   <- FK sanity, must be ~0")
    print(f"  pelvis->chest span          {tl.mean():.4f}  "
          f"+/-{tl.std():.4f}   (spans the spine; flexes by design)")
    print("=" * 64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        source=Path(args.video).stem,
        fps=fps,
        num_frames=N,
        frame_idx=np.asarray(frame_idx),
        valid=valid,
        point_names=np.asarray(POINT_NAMES),
        points_local=points_local.astype(np.float64),
        root_model=root[:, 0, :].astype(np.float64),
        paw_uv=paw_uv.astype(np.float64),
        global_orient=go_s.astype(np.float64),
        pose=pose_s.astype(np.float64),
        betas_frozen=betas_frozen.astype(np.float64),
        cam_t=camt_s.astype(np.float64),
        det_box=box_raw.astype(np.float64),
        bbox_tz_ratio=tz_ratio.astype(np.float64),
        img_size=np.asarray([W, H]),
        focal_full=float(focal_full),
        theta_cutoff=float(args.theta_cutoff),
        beta_wobble_median=beta_wobble_med,
        beta_wobble_max=beta_wobble_max,
        theta_jitter_before=jit_before,
        theta_jitter_after=jit_after,
        detection_rate=float(valid.mean()),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
