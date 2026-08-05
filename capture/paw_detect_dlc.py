"""Independent 2D paw pixels from DeepLabCut SuperAnimal-Quadruped.

Why a detector rather than the mesh projection or a point tracker:

  * measured on the harness, paw pixels that are CORRELATED with the mesh's
    own FK error cost 2-3x more placement error than INDEPENDENT pixels of
    the same magnitude, because a wrong body that is self-consistent is
    invisible to the solver. 8 px of independent error is as good as perfect
    pixels; precision past a few px buys nothing.
  * a generic point tracker (CoTracker3) was tried and rejected: it needs a
    correct seed and continuous appearance, and the mesh sole lands 7-21 px
    off the animal, so it locked onto floor texture.

A detector supplies exactly what is needed — a per-frame estimate from a
different model, on different evidence, with errors independent of the SMAL
fit both across frames and from the mesh.

Runs in its own environment (dlcenv) behind a subprocess boundary, like
animer_infer.py, so DLC's dependency tree cannot disturb the AniMer env.

TWO CONVENTIONS THIS FILE RECONCILES

  * RESOLUTION. DLC runs on the source video; the pipeline works at max-side
    1280 (dog_1: 1280x800). Coordinates are rescaled by the exact per-axis
    factor rather than transcoding, so no resampling error is introduced.
  * LEG ORDER. Do NOT trust the bodypart names. AniMer's own front pair is
    transposed relative to reading order (STATUS.md Traps), and left/right in
    an animal-anatomy schema may or may not match ours. The assignment is
    therefore solved GEOMETRICALLY against the mesh paws (Hungarian on median
    pixel distance) and the name-based mapping is printed alongside as a
    cross-check. If the two disagree, that disagreement is the finding.
"""
from pathlib import Path
import argparse
import glob
import sys

import numpy as np

LEGS = ["FR", "FL", "RR", "RL"]
# what the SuperAnimal-Quadruped schema calls the paws, for the cross-check
NAME_HINTS = {
    "FR": ("front_right_paw", "right_front_paw", "RF_paw"),
    "FL": ("front_left_paw", "left_front_paw", "LF_paw"),
    "RR": ("back_right_paw", "right_back_paw", "hind_right_paw", "RB_paw"),
    "RL": ("back_left_paw", "left_back_paw", "hind_left_paw", "LB_paw"),
}


def frame_size(src_w, src_h, max_side):
    """Matches the pipeline's own downscaling, so pixels mean the same."""
    s = 1.0
    if max_side > 0 and max(src_w, src_h) > max_side:
        s = max_side / max(src_w, src_h)
    return int(round(src_w * s)) // 2 * 2, int(round(src_h * s)) // 2 * 2


def run_dlc(video, dest, max_frames):
    import deeplabcut
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fn = deeplabcut.video_inference_superanimal
    import inspect
    sig = inspect.signature(fn)
    print("video_inference_superanimal signature:", sig, flush=True)
    kw = dict(videos=[str(video)], superanimal_name="superanimal_quadruped",
              dest_folder=str(dest))
    for k, v in (("model_name", "hrnet_w32"), ("detector_name", "fasterrcnn_mobilenet_v3_large_fpn"),
                 ("video_adapt", False), ("max_individuals", 1)):
        if k in sig.parameters:
            kw[k] = v
    print("calling with:", {k: v for k, v in kw.items() if k != "videos"}, flush=True)
    fn(**kw)
    return dest


def load_dlc(dest):
    """Find and parse whatever DLC wrote; return (N,K,3) xy+conf and names."""
    import pandas as pd
    cands = sorted(glob.glob(str(Path(dest) / "*.h5"))) + \
        sorted(glob.glob(str(Path(dest) / "*.csv")))
    if not cands:
        raise SystemExit(f"no DLC output found under {dest}")
    path = cands[0]
    print(f"reading {path}")
    df = pd.read_hdf(path) if path.endswith(".h5") else pd.read_csv(
        path, header=[0, 1, 2], index_col=0)
    cols = df.columns
    bp_level = 1 if cols.nlevels == 3 else 0
    if cols.nlevels == 4:                       # scorer, individual, bp, coord
        bp_level = 2
    names = list(dict.fromkeys(cols.get_level_values(bp_level)))
    arr = np.full((len(df), len(names), 3), np.nan)
    for i, nm in enumerate(names):
        sub = df.xs(nm, axis=1, level=bp_level)
        take = lambda c: (sub[c].to_numpy() if c in sub.columns
                          else sub.xs(c, axis=1, level=-1).to_numpy().ravel())
        try:
            arr[:, i, 0] = take("x")
            arr[:, i, 1] = take("y")
            arr[:, i, 2] = take("likelihood")
        except Exception:
            pass
    return arr, names


def assign_legs(arr, names, mesh_uv, sx, sy, min_conf):
    """Hungarian assignment of DLC keypoints to our FR/FL/RR/RL."""
    from scipy.optimize import linear_sum_assignment
    n = min(len(arr), len(mesh_uv))
    xy = arr[:n, :, :2] * np.array([sx, sy])
    conf = arr[:n, :, 2]
    K = xy.shape[1]
    cost = np.full((4, K), 1e6)
    for p in range(4):
        for k in range(K):
            m = np.isfinite(xy[:, k, 0]) & (conf[:, k] > min_conf)
            if m.sum() > 20:
                cost[p, k] = np.median(np.linalg.norm(
                    xy[m, k] - mesh_uv[:n][m, p], axis=1))
    rows, cols = linear_sum_assignment(cost)
    return {LEGS[r]: (names[c], c, cost[r, c]) for r, c in zip(rows, cols)}, xy, conf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--mesh", required=True,
                   help="npz with the mesh paw_uv, for the geometric leg "
                        "assignment and the agreement report")
    p.add_argument("--out", required=True)
    p.add_argument("--dest", default=None, help="DLC scratch dir")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--min-conf", type=float, default=0.4)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--reuse", action="store_true",
                   help="skip inference, parse an existing DLC output")
    args = p.parse_args()

    import cv2
    cap = cv2.VideoCapture(args.video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    W, H = frame_size(src_w, src_h, args.max_side)
    sx, sy = W / src_w, H / src_h
    print(f"source {src_w}x{src_h} -> pipeline {W}x{H}  (scale {sx:.4f},{sy:.4f})")

    dest = Path(args.dest or (Path(args.out).parent / "dlc_raw"))
    if not args.reuse:
        run_dlc(args.video, dest, args.max_frames)
    arr, names = load_dlc(dest)
    print(f"DLC returned {arr.shape[1]} keypoints over {len(arr)} frames")
    paw_names = [n for n in names if "paw" in n.lower() or "foot" in n.lower()]
    print(f"  keypoints with 'paw'/'foot' in the name: {paw_names}")

    mesh = np.load(args.mesh, allow_pickle=True)["paw_uv"]
    amap, xy, conf = assign_legs(arr, names, mesh, sx, sy, args.min_conf)

    print("\ngeometric assignment (Hungarian vs mesh paws):")
    for leg in LEGS:
        nm, k, c = amap[leg]
        hint = [h for h in NAME_HINTS[leg] if h in names]
        flag = "" if (hint and nm == hint[0]) else "   <-- name-based guess " \
            f"would be {hint[0] if hint else 'n/a'}"
        print(f"  {leg} <- '{nm}'  median distance {c:.1f} px{flag}")

    n = min(len(xy), len(mesh))
    out = np.full((n, 4, 2), np.nan)
    cf = np.zeros((n, 4))
    for i, leg in enumerate(LEGS):
        k = amap[leg][1]
        out[:, i] = xy[:n, k]
        cf[:, i] = conf[:n, k]
    lowc = cf < args.min_conf
    out[lowc] = np.nan
    print(f"\nconfidence: mean {cf.mean():.2f}; dropped {100 * lowc.mean():.1f}% "
          f"of paw-frames below {args.min_conf}")
    good = np.isfinite(out[..., 0])
    dev = np.linalg.norm(out - mesh[:n], axis=-1)
    print(f"agreement with mesh: median {np.nanmedian(dev):.1f} px, "
          f"p90 {np.nanpercentile(dev, 90):.1f} px")
    print(f"coverage after confidence gate: {100 * good.mean():.1f}%")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o, paw_uv=out, confidence=cf, paw_uv_mesh=mesh[:n],
                        num_frames=n, detector="superanimal_quadruped",
                        assignment=np.array([amap[l][0] for l in LEGS]),
                        scale=np.array([sx, sy]))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
