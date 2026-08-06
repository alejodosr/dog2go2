#!/usr/bin/env bash
# DEFAULT video -> Go2 pipeline, end to end, in this repo.
#
#   usage:  capture/run_default.sh <video> <clip-name> [trim_start,trim_end]
#
# Stages 1-5 turn the video into the npz contract; stages 6-8 are the existing
# retargeter, unchanged. Splitting them is still meaningful because they run on
# DIFFERENT INTERPRETERS -- see "environments" below -- but you no longer have
# to drive the halves from two repositories by hand.
#
# The ground plane comes from Depth Anything V2 metric depth, not from clicking
# four tile corners. That removes both the interactive step and the tile-size
# assumption that made the old calibration's metres provisional. Measured
# against the clicked calibration on real footage: -4.6% (dog_1) and -9.3%
# (cat_1). It diverges on AI-generated video (dog_2, -54%), which is the known
# limitation -- there is no real geometry in generated video to measure.
#
# Scale is a log-space combination of three estimators with NO biological
# prior: an ablation showed the prior carried 6-18% of the weight and moved the
# answer by at most 4.6%, while every sigma in it was hand-asserted.
#
#   1  capture.animer_infer     SMAL pose per frame            (GPU, $PY_CAPTURE)
#   2  capture.depth_calib      ground plane from metric depth (GPU, $PY_CAPTURE)
#   3  capture.contacts_kine    stance/swing, no floor speed        ($PY_CAPTURE)
#   4  capture.world_place_ba   clip-wide placement, --size-prior 0 ($PY_CAPTURE)
#   5  capture.parse_video      the npz contract                    ($PY_CAPTURE)
#   6  retarget/retarget.py     -> motions/<clip>.pkl                     (uv)
#   7  viz/playback.py          -> Go2 render                             (uv)
#   8  capture/viz_world.py     -> side-by-side mp4                 ($PY_CAPTURE)
#
# ENVIRONMENTS. Two, and they can never be merged: stages 1-5 need torch +
# detectron2 + transformers, stages 6-7 need mujoco and this repo's uv env.
# Set $PY_CAPTURE to the perception interpreter (the 'animal' conda env).
#
# AniMer's `amr` package is vendored at the repo root, so no checkout of it is
# needed. Only $ANIMER_CKPT (8.35 GB) and data/smal/ stay outside the tree.
#
# Trap: the stages DISAGREE about user site-packages, so there is no single
# right setting -- see the two runners defined below. Stage 2 dies without
# PYTHONNOUSERSITE=1; stages 1, 3 and 8 die with it. Both directions have been
# observed, so do not "simplify" them into one.
#
# DeepLabCut is NOT part of this pipeline. Its 2D skeleton is never read by
# contacts_kine, world_place_ba or parse_video -- those take only --infer,
# --contacts and --calib. capture/paw_detect_dlc.py remains an OPTIONAL,
# separate quality check answering "is the mesh actually on the animal?" with
# an independent detector (dog_3: median 5.7 px, 90.1% coverage). It needs its
# own environment: DLC pins numpy<2 and the perception env is on numpy 2.x.
#
#   PYTHONNOUSERSITE=1 $PY_DLC -m capture.paw_detect_dlc \
#     --video <video> --mesh $WORK/<clip>_animer.npz \
#     --out $WORK/<clip>_dlc.npz --dest $WORK/<clip>_dlc_raw
#
# FOCAL LENGTH is the one input no stage can measure. depth_calib reads it from
# a seed json; if none exists this script writes one using FOCAL_RATIO (focal /
# frame width). The three clips with a real four-point calibration give 0.791 /
# 0.825 / 0.858, so the default is their median. Override with
#   FOCAL_RATIO=0.86 capture/run_default.sh ...
# or drop a hand-made $CALIB/<clip>_seed.json in place.
#
# Every stage is skipped if its output already exists, so a re-run resumes. To
# genuinely redo a clip, delete its artifacts first -- otherwise you are
# measuring the old run.
set -euo pipefail

VIDEO="${1:?usage: run_default.sh <video> <clip> [trim_start,trim_end]}"
CLIP="${2:?usage: run_default.sh <video> <clip> [trim_start,trim_end]}"
TRIM="${3:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- configuration; all overridable from the environment (see capture/paths.py)
SSD="${A2G2_SSD:-/media/SHARED_DATA/postcapitalistrobots/a2g2}"
WORK="${A2G2_WORK:-$SSD/work/capture}"
CALIB="${A2G2_CALIB:-$WORK/calib_depth}"
PY_CAPTURE="${PY_CAPTURE:-$HOME/anaconda3/envs/animal/bin/python}"
PY_DLC="${PY_DLC:-$SSD/venvs/dlcenv/bin/python}"   # optional QA only
DEPTH_MODEL="${DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf}"
export HF_HOME="${HF_HOME:-$SSD/models/hf}"

[ -x "$PY_CAPTURE" ] || {
  echo "PY_CAPTURE=$PY_CAPTURE is not executable." >&2
  echo "Point it at the perception env's python (see README, 'the capture environment')." >&2
  exit 1; }

VIDEO="$(cd "$(dirname "$VIDEO")" && pwd)/$(basename "$VIDEO")"
mkdir -p "$WORK/processed" "$CALIB" "$REPO/media"
cd "$REPO"

TRIMARG=""
[ -n "$TRIM" ] && TRIMARG="--trim $TRIM"

# PYTHONNOUSERSITE is NOT a blanket setting -- the stages disagree about it,
# and both directions have been observed to fail:
#   * stage 2 REQUIRES it. ~/.local holds a broken soundfile which transformers
#     imports; without the guard you get ModuleNotFoundError: _cffi_backend,
#     surfacing as "Could not import module 'AutoImageProcessor'".
#   * stages 1 and 3 REQUIRE THE OPPOSITE. Both reach amr/models/__init__.py,
#     whose import chain hits einops -- and einops is installed ONLY in
#     ~/.local, not in the conda env. With the guard on they die at import.
#   * stage 8 (viz_world) also needs user site-packages.
# Hence two runners, not one.
run() {      PYTHONPATH="$REPO" "$PY_CAPTURE" -m "$@"; }   # amr stages + viz
run_clean() { PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" "$PY_CAPTURE" -m "$@"; }  # transformers

echo "=== 1/8 AniMer pose ==="
[ -f "$WORK/${CLIP}_animer.npz" ] || \
  PYOPENGL_PLATFORM=egl FVCORE_CACHE="$SSD/caches/fvcore" \
  run capture.animer_infer --video "$VIDEO" --out "$WORK/${CLIP}_animer.npz"

echo "=== 2/8 ground plane from metric depth (no clicking) ==="
SEED="$CALIB/${CLIP}_seed.json"
if [ ! -f "$CALIB/${CLIP}_depth.json" ] && [ ! -f "$SEED" ]; then
  # depth_calib reads ONLY img_size / focal_px / max_side from the seed; the
  # plane, H and camera height all come from the depth model.
  IFS=, read -r VW VH < <(ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height -of csv=p=0 "$VIDEO")
  # The seed must live in the INFERENCE pixel frame, not the source video's:
  # animer_infer downscales to max_side 1280 (rounded, floored to even), and
  # contacts_kine rejects a calibration whose img_size disagrees with it.
  # LC_ALL=C or a comma-decimal locale (es_ES) emits "686,4" and breaks the JSON
  IFS=, read -r IW IH < <(LC_ALL=C awk -v w="$VW" -v h="$VH" 'BEGIN{
    m=1280; mx=(w>h?w:h); s=(mx>m ? m/mx : 1.0);
    W=int(w*s+0.5); W-=W%2; H=int(h*s+0.5); H-=H%2; printf "%d,%d\n", W, H}')
  FOCAL=$(LC_ALL=C awk -v w="$IW" -v r="${FOCAL_RATIO:-0.825}" 'BEGIN{printf "%.1f", w*r}')
  echo "  no seed for $CLIP -- assuming focal $FOCAL px (${FOCAL_RATIO:-0.825} x ${IW} px inference width; source ${VW}x${VH})."
  echo "  THIS IS AN ASSUMPTION, not a measurement. See the header."
  printf '{"camera":"%s_seed","img_size":[%d,%d],"max_side":1280,"focal_px":%s}\n' \
    "$CLIP" "$IW" "$IH" "$FOCAL" > "$SEED"
fi
if [ -f "$CALIB/${CLIP}_depth.json" ]; then
  # A calibration from a different depth model would be silently reused and
  # would quietly change every metre downstream. Say so.
  HAVE=$("$PY_CAPTURE" -c "import json,sys;print(json.load(open(sys.argv[1])).get('depth_model','(pre-DepthAnything/ZoeDepth era)'))" \
         "$CALIB/${CLIP}_depth.json")
  echo "  reusing existing calibration, produced by: $HAVE"
  case "$HAVE" in
    "$DEPTH_MODEL") ;;
    *) echo "  WARNING: current depth model is $DEPTH_MODEL."
       echo "  WARNING: delete $CALIB/${CLIP}_depth.json to regenerate." ;;
  esac
else
  run_clean capture.depth_calib --video "$VIDEO" --ref-calib "$SEED" \
    --model "$DEPTH_MODEL" --out "$CALIB/${CLIP}_depth.json"
fi

INFER="$WORK/${CLIP}_animer.npz"

echo "=== 3/8 contacts ==="
[ -f "$WORK/${CLIP}_contacts.npz" ] || \
  run capture.contacts_kine --infer "$INFER" \
    --calib "$CALIB/${CLIP}_depth.json" --out "$WORK/${CLIP}_contacts.npz"

echo "=== 4/8 world placement (BA, no size prior) ==="
[ -f "$WORK/${CLIP}_world.npz" ] || \
  run capture.world_place_ba --infer "$INFER" \
    --contacts "$WORK/${CLIP}_contacts.npz" --calib "$CALIB/${CLIP}_depth.json" \
    --out "$WORK/${CLIP}_world.npz" --size-prior 0 $TRIMARG

echo "=== 5/8 npz contract ==="
# --source names the file the retargeter writes, NOT the npz filename. Keep it
# equal to $CLIP or stage 6 will overwrite a different clip's motion.
[ -f "$WORK/processed/${CLIP}.npz" ] || \
  run capture.parse_video --world "$WORK/${CLIP}_world.npz" \
    --source "$CLIP" --out "$WORK/processed/${CLIP}.npz"

echo "=== 6/8 retarget to Go2 ==="
[ -f "$REPO/motions/${CLIP}.pkl" ] || \
  uv run python retarget/retarget.py "$WORK/processed/${CLIP}.npz"

echo "=== 7/8 render the Go2 (from the source video's solved camera) ==="
[ -f "$REPO/media/go2_${CLIP}.mp4" ] || \
  MUJOCO_GL=egl uv run python viz/playback.py "$REPO/motions/${CLIP}.pkl" \
    --camera-world "$WORK/${CLIP}_world.npz" \
    --camera-calib "$CALIB/${CLIP}_depth.json" \
    --out "$REPO/media/go2_${CLIP}.mp4"

echo "=== 8/8 side-by-side ==="
# viz_world is the one stage that NEEDS user site-packages; see the header.
[ -f "$REPO/media/sbs_${CLIP}.mp4" ] || \
  run capture.viz_world \
    --world "$WORK/${CLIP}_world.npz" --video "$VIDEO" \
    --right-video "$REPO/media/go2_${CLIP}.mp4" \
    --out "$REPO/media/sbs_${CLIP}.mp4"

echo
echo "done"
echo "  motion       $REPO/motions/${CLIP}.pkl"
echo "  side-by-side $REPO/media/sbs_${CLIP}.mp4"
