# EXP-005: SPEED-CONDITIONAL unsupervised anomaly detection (POSITIVE)

**Status:** completed (positive, mechanism-confirmed)
**Date:** 2026-08-29

## Hypothesis
Unconditional detectors let fault energy hide in the speed-induced spread of the normal
class. Modeling the normal manifold CONDITIONED on the measured tacho speed (score =
deviation from the speed-local normal) removes that spread and improves fault detection.
Exploits MaFaulDa's continuous physical speed covariate, which DCASE domain-shift AD lacks.

## Setup
- MaFaulDa normal + horizontal-misalignment; accel col1; 16-rev windows; log-spectrum (dim 2049).
- Conditional model: subtract kNN-in-speed normal mean (k=15 nearest normal-train points in
  speed), then PCA(k=32) recon error on residuals. Baseline: unconditional PCA. Also poly and
  poly-clamp trend variants. Reuses cache_ch1_r16_s256.npz.

## Headline numbers
| Protocol | Unconditional | poly-clamp | kNN-speed |
|---|---|---|---|
| Matched-range (10-seed) | 0.856 +-0.009 | 0.900 | 0.957 +-0.008 |
| Held-out-speed (LOBO mean) | 0.689 | 0.734 | 0.794 |
- Matched-range paired diff kNN-uncond = +0.101 +-0.006 (min per-seed +0.087); bootstrap CI
  of kNN AUC [0.943, 0.978] (vs uncond 0.856). k-robust (k=5..60 all 0.91-0.98).

## Sanity / controls
- 10-seed stability: yes (std 0.008).
- Corrected leakage guard: detrend normal-TEST with TRUE speed -> AUC 0.961; with SHUFFLED
  speed -> AUC 0.419 (collapses), mean normal score 1.04 -> 1.22. Confirms the win is causally
  from correct speed-conditioning of the NORMAL manifold, not a confound.
- Note: shuffling FAULT speed does NOT drop AUC (0.989) -- expected; wrong trend just adds
  mismatch to faults. Mechanism is on the normal side. No train/test leakage (kNN uses train only).

## Conclusion
First solid, well-controlled positive. Speed-conditional (tacho-conditioned) normal modeling
categorically improves unsupervised fault detection: +0.10 AUC matched-range and +0.10 on
held-out speeds, over the strong accel-PCA baseline that beat the two earlier novelties
(order-sync, cross-modal). Mechanism confirmed. This is the paper's candidate core claim.

## Next (hardening for paper)
- External validity: confirm across fault types (bearing underhang/overhang, imbalance,
  vertical-misalignment; needs downloads ~9GB) and channels (mic, other accels).
- Scout published domain-shift AD baselines (DCASE 2021-2023 / MIMII-DG) before claiming SOTA.
- Optional: apply the conditioning principle to MIMII-DG (attribute-conditioned) for breadth.
- Optional stronger backbone (AE); PCA already shows the effect cleanly.

## Status: completed (positive)
