"""Monocular video -> metric world-frame animal motion.

The front half of the pipeline: it produces exactly the npz that
`retarget/retarget.py` already consumes, so a video and a BVH mocap clip are
interchangeable from the retargeter's point of view.

    video ─1─ AniMer SMAL pose ─2─ ground plane (metric depth)
                    │                      │
                    └──3── contacts ───4── world placement (BA)
                                                  │
                                        5── processed/<clip>.npz
                                                  │
                                        retarget/retarget.py (unchanged)

  1 animer_infer     SMAL pose/shape per frame, shape frozen to the clip median
  2 depth_calib      plane normal + camera height, from metric depth
  3 contacts_kine    which paws are planted, per frame
  4 world_place_ba   clip-wide trajectory + metric scale (bundle adjustment)
  5 parse_video      the npz contract

Run the whole thing with `capture/run_default.sh`. Every module is also a CLI
in its own right: `python -m capture.<stage> --help`.

Docstrings here cite `brief_claude.md`, `PLAN.md` and `STATUS.md`. Those are
AniMer-side design documents and did not migrate; the findings that still bind
this code are restated in the README's "capture" section.

ENVIRONMENT: everything in this package runs under the perception interpreter
($PY_CAPTURE), never under this repo's uv env — stages 1-2 need torch,
detectron2 and transformers, which the mujoco env deliberately does not have.
The dividing line is the package boundary: `capture/` is torch, everything
else in the repo is uv. The two exchange plain npz files and nothing else.
"""
