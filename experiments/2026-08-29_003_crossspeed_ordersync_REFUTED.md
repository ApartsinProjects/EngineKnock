# EXP-003b/c/d: cross-speed order-sync — THESIS REFUTED (diagnostic, not paper material)

**Status:** completed (negative)
**Date:** 2026-08-29

## Hypothesis (tested)
Order-domain (tacho-resampled) features are speed-invariant, so a narrow-speed-trained
normal-only detector should generalize across speeds and keep faults separable, where the
TIME-spectrum baseline (bins move with RPM) should collapse. Order-sync therefore wins
STRUCTURALLY under speed shift.

## Key experiments
- EXP-003b (speed bands, all-speed fault test): appeared to favor ORDER (0.688 vs TIME 0.508
  trained low). **CONFOUNDED**: AUC compared out-of-band normal vs all-speed fault, so it
  partly measured speed-distance, not fault. High-band training INVERTED ORDER to 0.270,
  exposing the confound.
- EXP-003c (confound-free: train band A, test normal AND fault both on disjoint band B):
  off-diagonal cross-speed mean AUC = TIME 0.721, ORDER 0.620, ORDER-shuffled 0.584.
  Headline (train 20-28, test others): TIME 0.609 > ORDER 0.588 ~ shuffle 0.562.
- EXP-003d (salvage: hybrid = time|order concat): hybrid 0.662 < time 0.721; paired
  hybrid-time diff -0.059, 95% CI [-0.096, -0.021] -> order channel SIGNIFICANTLY HURTS.

## Sanity / bug checks
- Not a bug: order features work at MATCHED speed (0.870 >= 0.853), so extraction is correct.
- Null (N-vs-N) ~0.5 in EXP-002 for all modes.
- Controls consistent: order ~ order_shuffled cross-speed => angle-locking carries little info here.

## Root cause (mechanism, not bug)
Order resampling is invariant only for ORDER-LOCKED components (shaft harmonics). Fixed-Hz
structural resonances (which carry misalignment/bearing fault energy) map to speed-dependent
order bins, so the order domain SMEARS them across speed while the time domain holds them
fixed. Hence TIME is more speed-robust, the opposite of the thesis.

## Conclusion
The order-synchronous unsupervised anomaly-detection thesis is refuted on MaFaulDa
misalignment under clean controls. Does NOT go in a paper. Open caveats: only misalignment
tested (bearing faults untested, but low prior given order~shuffle); PCA detector only;
single channel (accel col1); mic channel + cross-modal fusion untested.

## Status: completed (negative; thesis dropped)
