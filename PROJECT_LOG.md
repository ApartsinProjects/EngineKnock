# EngineKnock — Project Log

**Goal:** A novel, publishable method for **unsupervised** anomaly detection of engine
knock / NVH faults using **public data only**. Detection-only (localization dropped for now).

**Core thesis (to be tested):** Existing unsupervised machine-sound/vibration anomaly
methods (DCASE/MIMII autoencoders, embedding+Mahalanobis) treat signals as generic
spectrograms and ignore that machinery signals are periodic and phase-locked to shaft
rotation. An **order-synchronous** inductive bias (resample to the angular/order domain
using the tacho, model normal-only cycle-consistent structure, score deviation), optionally
with **cross-modal** (accel<->mic) agreement, should categorically improve unsupervised
detection at equal backbone. Grounded in classical angle-domain (time-to-angle) NVH analysis.

**Primary testbed:** MaFaulDa (accel + mic + tacho). Generalization: MIMII valve + DCASE
Task2. Real-engine sanity: DEFault. See scout/scout_datasets_2026-08-29.md.

---

## Chronological log

| # | Date | Experiment | Result | Sanity | Finding |
|---|------|-----------|--------|--------|---------|
| 0 | 2026-08-29 | Scout public datasets | 4 tiers mapped; MaFaulDa chosen primary | n/a | No public raw-knock set; MaFaulDa unique (accel+mic+tacho) |
| 1 | 2026-08-29 | EXP-001 MaFaulDa ingest + order-resampling smoke | Layout confirmed; tacho freq within ~2% of nominal; order overlay aligns at angle-locked events | PASSED | Order-sync pipeline works on real data; premise visible even on healthy bearing |
| 2 | 2026-08-29 | EXP-002/003 TIME vs ORDER, matched speed (PCA, misalignment) | TIME 0.856, ORDER 0.872, degenerate 0.851, shuffle 0.850 | Controls PASS (invariant, shuffle, null) | Mechanism validated but gain only +1.6pp: NOT categorical. Reframe to speed-shift regime |
| 3 | 2026-08-29 | EXP-003b/c/d cross-speed (confound-free) + hybrid | cross-speed mean AUC TIME 0.721 > ORDER 0.620; hybrid-time diff -0.059 CI[-0.096,-0.021] | Confound in 003b caught; 003c/d clean | THESIS REFUTED: order-sync loses cross-speed; order channel hurts. Root cause: order smears fixed-Hz resonances |
| 4 | 2026-08-29 | EXP-004 cross-modal accel<->mic | matched: xmodal_m2a 0.945 (best) but ~concat 0.926; cross-speed accel_pca 0.721 best | Gate failed; shuffle control OK | Cross-modal doesn't beat accel-PCA cross-speed; marginal matched. 2nd novelty fails clean bar |
| 5 | 2026-08-29 | EXP-005 SPEED-CONDITIONAL (kNN-in-speed normal model) | matched 0.856->0.957+-0.008 (+0.101); held-out-speed 0.689->0.794 | 10-seed stable; corrected guard: shuffle normal speed -> AUC 0.419 (mechanism confirmed) | POSITIVE. Tacho-conditioned normal modeling categorically beats accel-PCA. Paper core claim |
| 6 | 2026-08-29 | EXP-006 multi-fault validity (5 faults, base/inv/cond) | cond wins ALL 5 faults both regimes; POOLED matched 0.955->0.989, LOBO 0.897->0.937; inv never beats base cross-speed | float64 OOM fixed via float32 rewrite | POSITIVE, externally valid across fault types. Core result solid |
| 7 | 2026-08-29 | EXP-007 backbone-agnostic (PCA vs AE, pooled) | cond effect PCA +0.033/+0.040, AE +0.030/+0.044; AE alone !> PCA alone | 3-seed stable | POSITIVE. Conditioning is backbone-agnostic + dominant over backbone choice. Kills "PCA too weak" |
| 8 | 2026-08-29 | EXP-008 multi-channel (accel×2+mic)+fusion | cond wins all 3 channels; FUSED cond 0.992/0.959; fusion hurts w/o cond, best w/ cond | MemoryError fixed via streaming rewrite | POSITIVE. Holds across channels incl mic; conditioning enables fusion. Kills "single channel" |

## Key findings
- MaFaulDa: 50 kHz, 250k x 8 (tacho, 2x tri-axial accel, mic); normal=49 files by speed.
- Order-domain resampling via tacho pulses is robust to actual-vs-nominal speed drift.
- Order-synchronous representation does NOT help unsupervised detection (refuted, EXP-003).
- Cross-modal accel<->mic residual is best matched-speed detector but marginal vs concat and
  fails cross-speed (EXP-004).
- **SPEED-CONDITIONAL normal modeling (condition on tacho speed) is the winning idea**:
  +0.10 AUC matched-range AND held-out-speed vs strong accel-PCA baseline, mechanism-confirmed
  (EXP-005). Candidate paper core.

## Current standing + next
- PAPER DRAFTED + BUILT. paper/index.html (house style) + paper/EngineKnock_speed_conditional.docx
  (camera-ready, render-verified: 6 pages, 1/1 figure, 3/3 tables). Title+abstract via Fable agent;
  citations verified by web-researcher (fixed 3 errors: refs [2] title/year, [3] lead author, [8]
  hallucinated author). Figure paper/figures/fig_summary.png.
- CORE RESULT (validated + hardened): speed-conditional unsupervised AD wins across 5 fault types
  (EXP-006), PCA+AE (EXP-007), 3 channels incl mic + fusion (EXP-008), mechanism-confirmed
  (EXP-005 guard), novelty confirmed (conditioning vs invariance).
- Optional next: OASSL reproduction as a named competitor; MIMII-DG discrete-attribute-conditioning
  breadth experiment; paper-reviewer polish pass; anonymize/deanonymize author block for submission.
