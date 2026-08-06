# Milestone 2 — Phase 3 training runs

**STATUS (2026-08-05): VIDEO-TO-POLICY LOOP CLOSED (stage13) — one policy
tracks all three video-captured clips (dog_1/2/3, including the Veo-generated
dog_2) with joint err 0.030–0.041 rad, survival 96–100%, from-zero 100%.
Reference checkpoint `stage13_video3/model_5600`.**

Previous status (2026-07-19): CURRICULUM COMPLETE (stage7, all 3 clips, one
policy, 4000 iters) — walk 0.031 / 95.8% ✓✓, trot 0.089 / 99.5% ✓✓,
canter 0.090 joint (✓ under bar!) / 47.9% survival (✗, per the brief's
pre-authorized allowance: the clip contains a 4.1 m/s fully-airborne
gallop burst demanding 66 rad/s — 2.2× the actuator limit — plus a 167°
pirouette; physically infeasible segments, the article's "physics
disagrees" moment). Next: Phase 4/5 (assets, robustness, export) or
canter salvage options (see Observations).**

One row per run; one change per run (brief §4). Eval numbers from `eval.py`
(512 envs, deterministic, randomization off) on the run's final checkpoint.

| run | change vs. previous | iters | train reward | train ep len | eval mean joint err (rad) | eval ep len %max | outcome |
|---|---|---|---|---|---|---|---|
| `2026-07-17_14-09-42_stage1_trot` | — (baseline: single-clip trot `D1_009_KAN01_002`, brief §4 defaults) | 3000 | 65.7 | 218/500 | 0.310 | 29.4% | ✗ acceptance failed — see below |
| `2026-07-17_15-38-22_stage1b_trot_rootpos` | root height term → full 3D root-pos term (DeepMimic com: w 0.05→0.1, k 100→10, xy included) | 3000 | 80.4* | 226/500 | 0.267 | 30.3% | ✗ tracking −14%, but still 100% `root_pos` terminations (*reward not comparable: weights changed) |
| `2026-07-17_17-30-44_stage1c_trot_term2m` | `term_root_pos_err` 0.5 → 2.0 m (termination only; run-2 reward kept) | 3000 | 79.6* | 232/500 | 0.347 | 29.8% | ✗✗ **backfired** — drift 0.17→0.67 m/s, all metrics worse; REVERTED (*not comparable) |
| `2026-07-17_19-17-15_stage1d_trot_velx2` | bound back at 0.5 m; `rew_root_vel_w` 0.15 → 0.30 (velocity error is the drift source and is actor-observable via stance-leg dof_vel) | 3000 | 107.3* | 231/500 | 0.277 | 30.3% | ✗ xy err −20% (0.057), ori/height better, but ep len identical — led to finding the wrap bug (*not comparable) |
| `2026-07-18_14-22-09_stage1e_trot_wrapfix` | wrap fix only (+ 8192 envs, new speed default) | 1800 (stopped: plateau from ~1000) | ~100 | 225/500 | 0.271 | 31.6% | ✗ but diagnostic gold: `root_pos` wall GONE, terminations now 100% `root_ori` — unobservable heading drift hits the 45° bound at ~3 s. Joint plateau unchanged (expected: same actions/gains). Eval at model_1800, stage1-era cfg pinned via overrides |
| `2026-07-18_15-03-39_stage2_peng_trot` | Peng-alignment batch: ref-centered actions, kp=100/kd=1.0, vel-limit 30, ee reward (w.2 k40), merged pose kernel (20/10, w.15), split vel kernel (2/0.2, w.1) | 2200 (stopped: plateau from ~1300) | ~90* | 205/500 | **0.0746** | 30.7% | ✓✗ **joint acceptance SMASHED** (0.075 ≪ 0.15 bar, was 0.271; max 0.34); survival still 0% — 511/512 `root_ori` terminations (*reward not comparable) |
| `2026-07-18_16-03-12_stage3_heading_trot` | + heading-error obs (sin/cos ref-relative yaw; contract 80→82) | 3000 | ~98 | 210/500 | 0.0737 | 30.0% | ✗ **no change vs stage2** (len 150 vs 154, 512/512 `root_ori`) — and the yaw/tilt decomposition (added to eval.py) shows why: **yaw was never the problem** (end-of-episode yaw err 4.5°); end-of-episode TILT is 17° and accelerating ⇒ deaths are acute trip/fall events, not slow heading drift. The 45° ori bound is acting as a fall detector |
| `2026-07-18_17-45-50_stage4_acyclic_trot` | clips acyclic (episode = clip, end = truncation; root cause #2 fix) + injective acyclic phase encoding | 2000 (stopped: plateau from ~1300) | ~92 | 210 (cap ≈228) | 0.0905 | **survival 98.2%** (504/513 reach clip end; 9 `root_pos` deaths, all in the fast-trot final phase) | ✓✓ **STAGE-1 ACCEPTANCE MET**: joint err 0.09 < 0.15 AND survival ≫ 90%. `root_ori` deaths: 512→0. Video (one full episode, frame 0 → truncation): `videos/play/rl-video-step-0.mp4` |
| `2026-07-18_19-36-23_stage5_phasefree_trot` | phase-free tracker: phase obs removed, preview extended to steps [1,2,15,50] (20 ms–1 s, Peng-style); contract 82→116 actor / 124 critic | 2000 (stopped: plateau from ~1400) | ~93 | 212 | **0.0802** | survival 92.6% (474/512; 38 `root_pos` deaths, phase 0.8–1.0) | ✓ **parity: acceptance still met** (0.08 < 0.15, 92.6% > 90%) with *better* tracking; survival −5.6 pts vs stage4 — drift deaths in the fast section 9→38 (xy err 0.113→0.131). The phase-free contract is validated; the fast-section drift leak is the open item |
| `2026-07-18_20-36-50_stage6_multiclip_walktrot` | + walk clip (2-clip RSI, one policy, no clip ID — preview-only conditioning) | 3000 (still improving at 2200, ran full) | ~140* | ~230 | walk **0.0286** / trot 0.1085 | walk **95.4%** / trot **99.4%** | ✓✓ **STAGE-2 ACCEPTANCE MET** — both clips pass both bars from one policy. Interference trade: trot tracking −35% vs single-clip best (0.080→0.109) yet trot survival UP (92.6→99.4%; walk co-training regularizes). Walk deaths: 22 root_pos in its fast lead-out (phase 0.8–1.0). Per-clip videos in `videos/play/` (*2-clip reward, not comparable) |
| `2026-07-19_stage7_multiclip_all` (run dir dated 2026-07-18) | + canter clip (3-clip RSI, 4000 iters per user) | 4000 | — | — | walk 0.0306 / trot **0.0892** / canter 0.0900 | walk 95.8% / trot **99.5%** / **canter 47.9%** | ✓✓✗ walk+trot HOLD (trot tracking improved 0.109→0.089 with more data); canter joint err under the bar but survival fails — deaths cluster at phase 0.1–0.2 (the 4.1 m/s, 100%-flight gallop burst, ref dof_vel 66 rad/s = 2.2× actuator limit — infeasible) and 0.8–1.0 (sustained 2–2.5 m/s return canter). Per the brief: acceptable failure, article content. 3 per-clip videos in `videos/play/`. **2026-07-20 audit revised this diagnosis — see "Canter feasibility audit" below** |
| `2026-07-22_11-41-22_stage9_canter_timewarp` | canter **speed-aware time warp only** (`retarget/timewarp.py`, feasibility projection v1): planar root speed capped at 3.2 m/s by local playback slowdown — burst plays at ~0.75×, 73% of the clip bit-identical, 300→319 frames (5.98→6.36 s), smoothed peak 4.32→3.29 m/s, dof_vel peak 40→38.6 rad/s. Return canter (2–2.5 m/s) untouched by design. stage8 pkl → `motions_quarantine/D1_010_KAN01_004.stage8.pkl`, feet caches regenerated | 4000 | 112.5 | 199 | walk 0.0303 / trot 0.0823 / canter 0.0931 | walk 95.5% / trot 98.8% / **canter 61.2%** | ✓✓✗ walk/trot hold; **canter +5.8 pts from the warp** (55.4→61.2). Deaths 73: burst region (warped phase 0.1–0.4) still 43 — root_pos exits, i.e. the robot *survives* the slowed burst but can't hold xy tracking through it (front-support defect: front feet held high, 13/20% contact); return canter 21. From-zero: exit still deterministic in the burst but ~2× deeper (step 42 vs stage8's 22); trot from-zero 50%→98.9% (was knife-edge). Verdict: speed was real but secondary — **front-support repair is the next data lever** |
| `2026-07-22_14-25-32_stage10_5clip_jumpsit` | +2 clips: jump `D1_ex04_KAN02_003` (8.3 s) + sit/foot-up `D1_ex03_KAN02_013` (76.3 s), both sanitized (relabel+despike; jump contacts 0.25→0.42, dof_vel 84→40; originals in `motions_quarantine/*.orig.pkl`; timewarp no-op — planar 2.1/1.1 m/s). Infra batch: timewarp = postprocess step 8; **MotionLib `equal_clip_steps`** (clip ∼ 1/duration → equal env-step share; else sit eats ~93%); `episode_length_s` 10→80; feet caches ×5; 6000 iters | 6000 | 112.6 | 199 | walk 0.0344 / trot 0.0939 / canter 0.0677 / jump 0.0563 / sit 0.0602* | walk 98.9% / trot 97.9% / **canter 69.3%** / **jump 99.3%** / sit 56.2%* | ✓✓✓✓~ **everything improved**: canter +8.1 pts over stage9 (and tracking 0.093→0.068 best-ever), jump passes both bars outright. From-zero: walk 0→**92.5%**, trot **100%**, jump 49.6%, **sit 100%** (512/512 track all 76 s: sit-down→hold→foot-up→stand-up), canter 0% but exit deepened step 42→~198 (through the burst, now dies in the return section). *sit numbers from dedicated `eval_results_sit_only.md` — the shared eval's 512-episode cap starves long clips (4 eps only, all cold-spawn deaths); its 56.2% phase-averaged deficit = RSI cold-spawns INTO the deep sit (48–58° ref tilt, z 0.08 m; root_ori + 9 base_contact at phase 0.5–0.9), not the behavior. Residual levers: canter front-support repair; jump from-zero knife-edge; spawn-into-sit fragility |
| `2026-07-23_16-16-08_stage11b_revert_confirm` | none — confirmation rerun of the stage11 recipe on the reverted (byte-identical stage11) data after the stage12 revert | 6000 | 132.7 | 234 | identical to stage11 per checkpoint | identical to stage11 per checkpoint | ✓✓ **revert verified bit-exact**: training curve (132.66/233.73) and every sweep checkpoint reproduce stage11 digit-for-digit (m5600: walk 100/95.3fz, trot 82.5/30.6fz, canter 73.7/0fz, jump 98.4/94.8fz, sit 54.5/0fz; m5999 trot collapse to 10% included) — training is deterministic given identical data+seed, and the repo state is exactly stage11. Reference checkpoint: `model_5600`. 15 videos: `videos/play/*_stage11b_m5600_*` |
| `2026-07-23_13-21-46_stage12_clearance` | kinematic clearance projection (knees-below-floor poses lifted/re-IK'd; sit 39% of frames were 22 cm deep) — **REVERTED 2026-07-23 per user decision**: jump not improved (98% both ways), canter regressed 66→43% (the projection put an 8 cm root ramp through the fast pirouette; 108/144 deaths at its ramp-down, phase 0.5–0.6), sit look gain not worth it. One genuine positive: **checkpoint churn vanished** (all 4 sweep checkpoints agree ±3 pts) — evidence the infeasible poses were feeding stage11's oscillation. A pose-gated v2 (project only runs ≥0.5 s AND ≤0.6 m/s — separates sit 28 s@0.03 m/s from canter turn 1.2 s@1.85 m/s cleanly) was built+tested but not trained. Code archived in `motions_quarantine/reverted_code_stage12/`; data restored byte-identical to stage11 | 6000 | 101.1 | 172 | walk 0.0356 / trot 0.0920 / canter 0.0677 / jump 0.0507 / sit 0.0372 (m5200) | walk 100 / trot 88.4 / canter **41.5** / jump 98.1 / sit 55.1* (m5200) | ✗ reverted — **repo state after revert = stage11 code + stage11 data; stage11 `model_5600` remains the reference checkpoint** |
| `2026-07-22_19-51-38_stage11_jump_reground` | jump clip **re-ground only** (`retarget/reground.py`, new postprocess step 8: support-aware re-grounding — the 1.74 s "hover" was the dog standing on a hidden ~0.16 m object; root z projected down, contacts relabeled 0.42→0.46, real 240 ms leap untouched; stage10 pkl → `motions_quarantine/D1_ex04_KAN02_003.stage10.pkl`; verified no-op on the other 4 clips; 4 unit tests) | 6000 | 132.7 | 234 | (per-ckpt — see below) | (per-ckpt — see below) | ✓✗ **the jump fix worked** (from-zero 49.6→90.1% at m5999, 94.8% at m5600 — robust across checkpoints) **but training exhibits late multi-behavior churn**: different checkpoints own different behaviors and no single one passes all 5 (map in "Stage11 checkpoint churn" below). Best compromise m5600: walk 100/95.3, trot 82.5/30.6, canter **73.7**/0, jump 98.4/94.8, sit 54.5/0 (std/from-zero). Final m5999: sit 68.8/**91.2** + jump 96.7/90.1 but trot **10/1** (root_ori, end tilt 39°). Reward kept climbing through the churn (132.7 final) — total reward masks per-clip collapse |
| `2026-08-05_18-41-47_stage13_video3` | **first video-captured clip set** — walk-pace clips `dog_1` (11.8 s, iPhone), `dog_2` (8.0 s, Veo-generated), `dog_3` (5.0 s) from `capture/run_default.sh`, replacing the 5 mocap clips; from scratch (no warm start), stage11 recipe unchanged (6000 iters, 8192 envs, `env.motion_files` override). Capture-side fixes en route: seed json now written in the inference pixel frame (max_side 1280 — full-res seeds made contacts_kine reject the calibration); dog_1 seeded with its measured focal (1012 px, ratio 0.791) and trimmed 0–710 to cut a 9-frame paw-below-plane glitch | 6000 | 149.0 | 183 | dog_1 0.0409 / dog_2 0.0334 / dog_3 0.0300 (m5600) | dog_1 95.8% / dog_2 **100%** / dog_3 **100%** (m5600); **from-zero 100% on all three** | ✓✓✓ **all three behaviors pass both bars from one policy, and no checkpoint churn** (m4800–m5999 agree within ~7 pts; from-zero 100% at every sweep point — the video walk-pace clips are a far easier curriculum than the mocap set: peak dof_vel 22 rad/s vs canter's 40). Deaths (dog_1 std): 3/72, phase 0.3–0.4 + 0.9–1.0. Reference checkpoint: `model_5600`. Videos: `media/policy_dog_{1,2,3}.mp4` |
| `2026-07-20_14-50-16_stage8_canter_datafix` | canter **data fix only** (no env/reward change): despiked `dof_pos` (BVLS velocity clamp at 40 rad/s — trot's raw peak; 3 joints, ≤5 frames each, max 0.29 rad) + contacts relabeled from retargeted foot height (world z < 0.030 m + 1-frame morph cleanup; fraction 0.13→0.30), feet cache regenerated. Original pkl: `motions_quarantine/D1_010_KAN01_004.orig.pkl` | 4000 | 109.5 | 189 | walk 0.0291 / trot 0.0870 / canter 0.0993 | walk **96.8%** / trot 98.9% / **canter 55.4%** | ✓✓✗ walk best-ever tracking+survival, trot holds; **canter +7.5 pts survival from the data fix alone** (joint err 0.090→0.099, still under bar). Deaths now cleanly bimodal: 34/79 at phase 0.1–0.2 (the 4.5 m/s hind-leg-only gallop burst — the audit's genuine-physics residue) and 31/79 at 0.8–1.0 (fast return canter). Verdict: ~⅓ of the stage7 deficit was the data defect; the rest is the infeasible burst → trim (frames ~90–300) remains the honest lever if canter must pass |

## Peng 2020 comparison (2026-07-18)

Full numbers in memory note `peng2020-reference-numbers` (extracted from
arXiv:2004.00784 v3 + the released `motion_imitation` code). Deviations
found, ranked by suspected impact on the 0.27 rad plateau:

1. **Action parameterization** (confirmed): paper = absolute PD targets,
   bounds ±2π, low-pass filtered, fixed σ; ours was `default + 0.25·a`
   (needs |a| ≈ 5). Live stage1e logging confirms: `Action/abs_mean 1.25`,
   `abs_max 5.5`, 6% soft-limit clamp.
2. **PD gains**: paper kp=220 (Laikago); Go2 cfg kp=25 → open-loop floor
   τ_RMS/kp ≈ 0.15 rad. **pd_floor.py sweep** (512 envs, 600 steps, trot):

   | kp | kd | mean joint err floor (rad) |
   |---|---|---|
   | 25 | 0.50 | 0.168 |
   | 60 | 0.77 | 0.107 |
   | 100 | 1.00 | 0.091 |
   | 220 | 1.48 | 0.084 |

   Front-calf error ~0.26 rad persists at ALL kp → not stiffness-limited;
   suspect DCMotor `velocity_limit = 21 rad/s` derating torque during swing
   (ref dof_vel peaks 38; real Go2 spec ~30). **Open decision: raise
   velocity_limit to the Go2 datasheet 30 rad/s?** (contract-facing).
3. **Missing end-effector reward**: paper w=0.2, exp(−40·Σ‖Δx_ee‖²),
   root-relative — its 2nd-largest tracking term; we had none.
4. **Root vel kernel**: paper exp(−2‖Δv‖² − 0.2‖Δω‖²) w=0.1; ours had a
   shared k=2 (10× too stiff on ang vel → saturated kernel) at w=0.30.
5. **Termination inversion**: paper terminates on falls only and fights
   drift with a strong global root-pos reward (k=20); we had weak pos
   reward (k=10, w=0.1) + hard 0.5 m termination on an actor-unobservable
   quantity. stage2 adopts the paper reward (k=20 merged pose term); bound
   kept at 0.5 m for now.
6. Preview horizon: paper goals at t+1,2,10,30 (~1 s); ours t+1,2 (40 ms).
   Deferred to multi-clip (phase obs makes single-clip memorizable).
7. Minor: γ 0.95 vs ours 0.99; fixed vs learned σ; 512-256 nets (ours
   512-256-128); ~200M samples (ours ≈295M+). Same source mocap dataset
   (Zhang et al. [68] = KAN clips); paper's own sim trot return only 0.75.

**stage2 config** (implemented, smoke-tested; one paper-alignment batch):
`action_center = "ref"` (targets = ref_dof_pos(t+1) + 0.25·a; zero action
holds the reference, also at RSI), kp=100/kd=1.0, velocity_limit 21→30
rad/s (Go2 datasheet; user-approved), ee reward w=0.2 k=40
(root-frame foot pos vs sim-generated `motions/<clip>_feet.npz` caches —
`gen_feet_cache.py`, replay FK, root-relative → wrap-safe), merged root
pose term w=0.15 exp(−20·pos²−10·ori²), root vel w=0.1 exp(−2·lin−0.2·ang).
Kept from before: joint pos/vel terms, contact_match 0.1, action_rate,
torque penalties, all terminations, obs contract UNCHANGED. Deliberately
NOT in the batch: γ, preview horizon, termination redesign, velocity_limit.

## Open items (2026-07-18)

- **Action reach analysis** (offline, trot clip): max |q_ref − default| =
  1.28 rad (thigh), mean 0.42 rad ⇒ with `action_scale = 0.25` the policy
  needs |a| ≈ 5 at the extremes (≈ 5σ at init noise 1.0). Peng 2020
  parameterizes targets around the *reference* pose, not a static default —
  prime suspect for the 0.27 rad joint-error plateau. Torque is NOT the
  binding constraint: RMS ≈ 3.7 N·m/joint vs 23.7 N·m limit.
- Reference dof_vel peaks 38.4 rad/s > Go2 ~30 rad/s (peaks only).
- ~~Action-saturation logging~~ DONE (2026-07-18): `Action/abs_mean`,
  `abs_max`, `clamp_frac` logged from the env; stage1e live values 1.25 /
  5.5 / 6% confirm the reach analysis above.
- ~~Speed~~ DONE (2026-07-18): governor was already `performance`; 8192
  envs ≈ 156k steps/s (~2× of 4096), only ~7.3 GB VRAM — new default for
  all runs. NOTE: 8192 × 24 doubles samples/iter; the 1500-iter checkpoint
  is the sample-parity point vs the 4096-env stage1a–d runs.
- ~~velocity_limit~~ DECIDED (user, 2026-07-18): raised 21 → 30 rad/s (Go2
  datasheet) and included in stage2. Rationale: kp-independent front-calf
  PD-floor error (~0.26 rad) at the stock 21; ref peaks 38.4.
- Episode-length note: the trot clip is 9.12 s and max episode 10 s ⇒ under
  the wrap bug NO episode could survive (must cross a wrap); explains 0%
  survival everywhere.

## Observations

- **ROOT CAUSE #2 FOUND (2026-07-18, termination-phase diagnostic): the
  clips are not loops.** eval.py now logs death-phase + episode-length
  histograms: **511/512 stage3 deaths land in phase bin 0.9–1.0** — at the
  wrap seam, not spread over the gait. Clip forensics (offline, pkl):
  `D1_009` "trot" = 2.7 s motionless crouch (36% of the clip!) → stand-up
  transition → 5.5 s trot ending mid-flight. The cyclic wrap therefore
  demands trot-flight → crouch in one 20 ms frame: Δdof 1.34 rad, Δz
  −246 mm, Δyaw 20°, contacts 0→4. The *reference* teleports in
  orientation/pose space at every loop — ori error crosses 45° within
  steps regardless of policy quality (deaths are reference teleports, NOT
  physical falls; the "end tilt 17°" was tilt vs a reference slerping into
  the crouch). All three curriculum clips are diseased: walk seam 1.24 rad
  + 234 mm z (9% still), trot 1.34 rad + 246 mm (36% still), canter 1.03
  rad + **167° yaw** (the dog turns around mid-clip). Every episode-length
  / survival number in this table is seam-bounded (max achievable ep len ≈
  time-to-first-seam; ≈ 228 mean under uniform RSI). Same *class* of bug
  as the xy wrap teleport but in the clip CONTENT — M1 exported raw
  captures with lead-ins/outs; the "clean cyclic gait" premise was never
  true. FIX: trim clips to stride-cycle-aligned gait loops (offline tool),
  regen feet caches, retrain. The stage2/3 joint tracking (0.074) survived
  all this — the policy itself is likely fine.
- **stage3 eval + ori decomposition (2026-07-18, the "drift" narrative was
  wrong)**: eval.py now reports yaw and tilt separately (mean and
  end-of-episode). stage3: yaw err 3.7° mean / 4.5° at end — flat, benign;
  tilt 4.7° mean / **17° at end and accelerating** (45° crossed during the
  terminal step). Episodes end in acute tip-over events (~every 3 s), not
  accumulated heading drift — the ori bound is effectively a fall detector,
  so our termination is already ≈ the paper's falls-only in practice. The
  heading obs (stage3) was aimed at a non-problem; kept in the contract for
  now (2 dims, harmless, plausibly useful for turning clips; revisit at the
  Phase 5 freeze). Joint acceptance stands at 0.074. NEXT DIAGNOSTIC
  (cheap, before any new training): log clip phase at termination + episode
  length distribution — lethal-clip-segment hypothesis (a specific stride
  event trips the robot) vs uniform stochastic stumbling; then inspect
  swing-foot clearance in that segment (the M1 foot-skate removal may have
  left low clearance, and the k=40 ee kernel pulls feet onto those paths).
- **stage2 eval** (512 eps, model_2200): joint err 0.271 → **0.0746** in
  one batch — the paper-alignment diagnosis was right (parameterization +
  gains + missing ee term, compounding). Action noise std collapsed to
  0.16 (policy confident in small ref-relative corrections; abs_mean
  0.53). Height err 0.0175, xy 0.101. Remaining failure is singular:
  yaw drifts ~2°/s until the 45° `root_ori` bound at ~3 s (mean ori err
  6.56°, was 2.53° — the merged kernel trades ori for pos+joints once yaw
  is past its gradient range). Ghost video:
  `logs/.../stage2_peng_trot/videos/play/rl-video-step-0.mp4`.
  Next decision (single change): (A) add heading-error obs — sin/cos of
  ref-relative yaw, contract 80→82, what Peng does (observes IMU yaw);
  vs (B) tilt-only termination letting yaw drift free — but that breaks
  the global pos reward/termination as the path curves (they'd need a
  heading-aligned reformulation). A is the smaller coherent change.
- **stage1e eval** (513 eps, model_1800): joint err 0.271 (= stage1d, as
  expected — no action/gain change), xy err 0.070, ori err 2.53° mean, yet
  **100% `root_ori` terminations**: with the position teleport fixed, the
  next binding constraint is heading drift — yaw error is unobservable to
  the actor (no yaw/heading obs, by contract; Peng 2020 observes IMU
  roll-pitch-yaw) and random-walks to the 45° bound in ~3 s. Partial
  mitigation exists (ref root vel is expressed in the current base frame,
  so a yawed robot sees a rotated velocity command); if `root_ori`
  terminations still dominate after stage2, adding a heading-error obs is
  the paper-sanctioned contract change to consider.
- **Eval gotcha (added 2026-07-18)**: `eval.py` builds the env from the
  CURRENT registered cfg defaults (`@hydra_task_config`), not the run's
  dumped `env.yaml` — after stage2 changed the defaults (kp, action
  centering, velocity_limit), evaluating older runs requires pinning their
  era's values via hydra overrides (see stage1e eval command) or the
  numbers are invalid.

- **stage1_trot**: learned to trot without falling within ~300 iters
  (`base_contact` terminations → ~0), then plateaued from ~iter 700
  (ep len ~210–230/500, reward ~61–66). Dominant termination: `root_pos`
  drift > 0.5 m (~20/24 episode ends) — small velocity-tracking errors
  integrate into xy drift that nothing in the reward corrects (no xy term
  by design, brief §3). Eval + ghost video to discriminate: good gait that
  drifts vs. subtly wrong gait.
- **stage1_trot eval** (model_2999, 514 eps): mean ep len 147 (29.4% of max),
  survival 0% — every episode ends by `root_pos` drift. Root tracking is
  good (ori err 2.5°, height err 1.7 cm, mean xy err 0.09 m) but **mean
  joint err 0.310 rad** (bar: < 0.15) — the gait shape itself is off, not
  just drifting. Deterministic ep len (147) < stochastic training ep len
  (218): exploration noise was masking a systematic velocity deficit —
  drift to 0.5 m in ~2.9 s ⇒ ~0.17 m/s mean velocity error. Ghost video
  (`logs/.../stage1_trot/videos/play/rl-video-step-0.mp4`, side-by-side
  ghost at +1 m y): qualitatively a real trot, upright and rhythmic —
  verdict is "good-looking gait that drifts", not a broken gait.
  Candidate single changes for run 2 (pick ONE): add a soft root xy
  position term (DeepMimic/Peng 2020 have a com-position term; w≈0.1,
  fold with height into full root-pos tracking — reward-only, does not
  touch the frozen obs contract), relax `term_root_pos_err` 0.5→1.0 m,
  or raise `rew_joint_pos_k`/weight.
- **stage1b eval** (516 eps): joint err 0.310→0.267, xy err 0.090→0.071,
  ori 2.52°→2.24°, height 0.017→0.032 (z-kernel softening side effect) —
  every tracking metric improved except height, yet survival still 0% with
  100% `root_pos` terminations and ep len ~151. Structural conclusion: the
  actor cannot observe accumulated drift (no world pos / base lin vel in
  obs, by design), so reward alone cannot cancel it; and the brief's own
  philosophy says xy drift should be free — the 0.5 m `root_pos`
  termination bound contradicts it and truncates every episode. Next
  lever: the termination rule, not the reward.
- **stage1c eval** (512 eps): joint err 0.347, ori 3.55°, xy 0.114 — all
  worse than stage1b; ep len 149 *at a 4× looser bound* ⇒ deterministic
  drift ~0.67 m/s (4× stage1b). Lesson (Peng 2018/2020, confirmed here):
  **the termination bound is itself the anti-drift learning signal** — the
  exp position kernel has no gradient beyond ~0.6 m, so with the bound at
  2.0 m nothing opposes drift and the policy stops caring, degrading all
  root motion. Bound reverted to 0.5 m. Note: eval episode lengths are not
  comparable across runs with different bounds (eval uses the run's cfg).
- **ROOT CAUSE FOUND (2026-07-18): cyclic wrap teleport.** Four runs with
  different rewards and bounds all died at ~150 steps (3.0 s) — episode
  length was insensitive to every change, including a 4× looser bound
  (stage1c died at 2.0 m in the same time 0.5 m runs died — impossible for
  a steady drift process). Cause: `MotionLib.get_frame` interpolated raw
  stored `root_pos` at `t % duration`, so at every loop wrap the reference
  position teleported back to the loop start (meters, instantly) — the
  `root_pos` termination then fired at the first wrap regardless of policy
  quality or bound. ~150 steps ≈ mean time-to-first-wrap under uniform RSI
  phase. All stage1a–d episode-length/survival numbers (and the "drift"
  narrative) are artifacts of this; per-frame tracking errors (joint, ori)
  remain valid. Fixed: per-loop xy displacement accumulated across wraps
  (z periodic), wrap-seam lerp continues into the next loop; regression
  test `test_cyclic_root_pos_accumulates_across_loops` (14/14 pass).
- **Tooling gotcha (cost ~2 render cycles)**: the Go2 USD is instanceable →
  the ghost's `collision_enabled=False` and transparent material silently
  fail; an overlaid ghost ejects the robot at every RSI reset. Ghost now
  renders side-by-side (`ghost_y_offset = 1.0`). First two stage-1 videos
  were invalid for this reason.
- **Same gotcha, second bite (2026-07-19)**: for clips whose path doubles
  back (canter's 167° pirouette), a 1 m lateral offset is not enough — the
  return leg drives the robot through the solid ghost, and the first
  stage7 canter video showed constant robot–ghost collisions. Videos only
  (train/eval spawn no ghost — metrics unaffected). Refilmed with
  `env.ghost_y_offset=4.0`; rule of thumb: offset > the clip's lateral
  path spread.
- **Gotcha KILLED (2026-07-20)**: the ghost is now truly collision-free.
  `_setup_scene` de-instances the ghost subtree (`SetInstanceable(False)`,
  looped for nesting) so the prims become authorable, then sets
  `collisionEnabled=False` on every `CollisionAPI` prim. Verified: 0/27
  ghost colliders enabled (robot 27/27 intact), and a ghost parked
  *directly on* the robot produces 2.7 mm displacement in 0.1 s (the old
  bug ejected instantly at reset). The lateral offset is now cosmetic
  (camera framing), not a physics requirement; no-reset filming can't be
  destabilized by ghost contact regardless of where the robot wanders.
  Third bite of the same gotcha, hiding inside the fix: de-instancing
  made the spawn-time transparent material (opacity 0.35) live for the
  first time — the ghost body stopped rendering and only its drop shadow
  showed on video. Resolution: ghost material is now OPAQUE solid blue
  (opacity 1.0) and explicitly rebound to every mesh with
  `strongerThanDescendants` after de-instancing (prototype-local
  bindings otherwise win and leave meshes dark). Ghost = blue, policy
  robot = white. In doubling-back clips the white robot may pass
  *through* the blue ghost on video — harmless now, by construction.

## Stage11 checkpoint churn (2026-07-23)

Trot's 97.9→10% collapse at the final checkpoint triggered a bisect, which
found not one collapse but **oscillation**: per-checkpoint standard-eval
survival (%), plus from-zero where measured:

| ckpt | walk | trot | canter | jump | sit* |
|---|---|---|---|---|---|
| m3000 | — | 62.5 | — | — | — |
| m4800 | 82.2 | **95.0** (fz 73.7) | 66.7 | 96.5 | fz 0 |
| m5200 | 86.6 | 73.9 | 66.8 | 97.7 | — |
| m5600 | **100** (fz 95.3) | 82.5 (fz 30.6) | **73.7** | **98.4** (fz 94.8) | 54.5 (fz 0) |
| m5999 | 100 (fz 50) | 10.0 (fz 1) | 66.2 | 96.7 (fz 90.1) | **68.8** (fz **91.2**) |

(*sit numbers from dedicated sit-only evals; shared eval starves it.)

Facts: behaviors trade places between checkpoints 200–400 iters apart while
total training reward rises monotonically; the failure signatures differ
per checkpoint era (trot dies by root_ori/tilt at m5999; walk died by
root_ori at m4800-from-zero; sit-from-zero dies by xy drift at m5600 vs
surviving 91% at m5999). Hypotheses (not verified): gradient conflict
between sit's required ~50° reference tilt and the gait clips' tilt
regulation, amplified once the re-grounded jump became learnable (richer
reward → larger jump/sit gradient share late in training); no clip ID in
the obs contract means all disambiguation flows through the preview.
Candidate mitigations for a stage12: late-training LR/plasticity decay,
periodic multi-clip eval + best-checkpoint selection as infrastructure,
checkpoint weight averaging (same-run soup), or accepting per-behavior
checkpoint selection at deployment (breaks the one-policy milestone goal).

## Stage10 video review + jump forensics (2026-07-22)

User review of the 15 from-zero takes: canter visually much closer to the
reference, drift negligible on camera; sit/walk/trot/jump all look good —
except the jump never fully leaves the ground. Forensics on the (sanitized)
jump pkl explain it. **Measured facts** (finite-diff on the pkl):

- The clip has two unlike "flight" segments. Frames 38–50 (phase 0.09,
  240 ms): apex +0.15 m, mean root az **−7.1 m/s²** — near-ballistic, a
  genuine and feasible jump (takeoff vz 1.43 m/s ≤ what 40 rad/s legs can
  produce). Frames 98–185 (phase 0.24–0.45, **1740 ms**): root z holds
  ≈ 0.43 m with all four feet labeled airborne and mean az **−0.65 m/s²**
  — physically impossible sustained hover (same artifact class as the
  canter "levitation", 7× longer). A third 40 ms micro-flight at phase
  0.95. `contact_match` (w = 0.1) therefore rewards being airborne for the
  full 1.7 s hover. Despike touched frames 45–85 and 402–410 (takeoff/
  landing IK spikes; max 0.46 rad).
- Policy behavior (videos + eval): grounded approximation of the whole
  sequence, 99.3% phase-averaged survival, joint err 0.056.

**Hypotheses** (plausible, not verified against the source): the source
dog jumped **onto a raised object** (~0.10–0.15 m), stood on it, and
hopped off — the flat-ground retarget puts its feet above the 3 cm contact
threshold, producing the hover. The policy's refusal to commit to flight
is then rational: the hover is unsatisfiable, and the real 240 ms leap
risks landing terminations for marginal reward. Verification would need
the source video/BVH (foot heights during the segment).

**Canter improvement attribution (stage9 → stage10, +8.1 pts)**. Fact:
the sampler change barely moved canter's training share — duration-
weighted (stage9, 3 clips) already gave it ≈ 18% of env-steps vs 20%
under equal_clip_steps. So "better RSI sampling" is NOT the explanation.
Hypotheses for the actual drivers: (a) 6000 vs 4000 iters (≈ 1.6× canter
samples in absolute terms); (b) **jump co-training** — the burst is a
hind-leg bound and the jump clip trains exactly hind-driven explosive
extension + landing recovery; (c) sit co-training (balance at extreme
tilt). Same cross-behavior regularization pattern as walk→trot in stage6;
not isolated per-factor.

Pipeline lesson (for the video→robot feasibility layer): add a
**ballistic consistency check** — sustained all-airborne segments with
az ≉ −g indicate hidden terrain or retarget artifacts; flag or re-ground
before training. The timewarp (root-speed cap) cannot catch these:
vertical/support infeasibility is invisible to a planar speed test.

## Canter feasibility audit (2026-07-20) — revises the stage7 diagnosis

Policy-independent audit of `D1_010_KAN01_004` (finite-diff kinematics of
the raw pkl vs. Go2 limits, walk/trot as feasible controls). Three findings:

1. **The "66 rad/s = 2.2× actuator limit" was a retarget artifact**, not a
   physical demand: a single-frame 1.33 rad discontinuity (FR thigh,
   1.63→0.30 rad in 20 ms, frame ~63). 3-frame-smoothed true demand is
   36.6 rad/s = 1.2× limit — marginal, not absurd (trot's raw peak is
   40.8 rad/s and tracks at 99.5% survival).
2. **Contact labels were broken and fed the reward.** Labels come from the
   *source dog's toes* with a horizontal-speed threshold (retarget.py),
   which fails at gallop speed: 13% contact fraction vs 52/55% for
   walk/trot (real canter duty ≈ 30–40%). The burst was labeled 100%
   airborne for 1.2 s while retargeted feet sat at ground height (70
   foot-frames < 2.8 cm labeled airborne); root az ≈ −0.3 m/s² during this
   "flight" (levitation; trot's real flights show a proper ballistic
   −7…−8). `contact_match` (w=0.1) therefore *rewarded avoiding stance*
   through the burst and much of the return canter — plausibly part of the
   47.9% survival, including the 2–2.5 m/s return-canter deaths.
3. **What is genuinely infeasible: the speed, and front support.** 4.5 m/s
   peak (30 rad/s joints + ~0.35 m legs → practical tracking ceiling
   ~3–3.5 m/s), and the retarget leaves front feet high through the burst
   (FR/FL contact 13/20% vs RR/RL 43/43%) — the reference effectively asks
   for a hind-leg bound at 4.5 m/s.

**From-zero eval (2026-07-20, stage8 checkpoint,
`eval_results_fromzero.md`)**: same clean protocol but
`env.rsi_start_at_zero=true` — every episode must track its whole clip
from frame 0. Results: **walk 0%** (deterministic `root_pos` exit at
phase 0.65–0.8, accumulated xy drift 0.26 m), **trot 50%** (knife-edge:
51 `root_ori` / 51 time_out — per-env PhysX nondeterminism splits an
otherwise deterministic trajectory), **canter 0%** (deterministic
`root_pos` exit at ~step 22 ≈ phase 0.07, the burst's onset). Lesson:
the headline survival numbers are **phase-averaged** (uniform-RSI starts,
the protocol of every row above) and are much more forgiving than
continuous from-zero tracking — most eval episodes start mid-clip and
face only a suffix of the hazards, and short horizons accumulate less xy
drift. Both metrics are honest; they answer different questions
("recover-and-track from anywhere" vs "reproduce the whole clip"). A
from-zero video therefore shows the envelope exit essentially every
time, at the same spot, regardless of the 55%/97% numbers.

Fix applied for stage8 (data only, minimal deformation): BVLS velocity
clamp at 40 rad/s on `dof_pos` (only 3 joints changed, ≤5 frames each,
max 0.29 rad, mean 0.00035 rad); contacts relabeled from retargeted foot
world height (z < 0.030 m; stance rest is 0.0239, old labels' z p95 was
0.028) + 1-frame morphological cleanup; feet cache regenerated via
kinematic replay. After fix: airborne fraction 59.7%→21.7% (trot-like),
longest flight 1220→240 ms, peak dof_vel 40.0 rad/s. Audit/fix scripts:
session scratchpad (`feasibility.py`, `fix_canter_despike.py`,
`relabel_contacts.py`); original pkl backed up to
`motions_quarantine/D1_010_KAN01_004.orig.pkl`. Walk/trot labels look
sane and were left untouched.
