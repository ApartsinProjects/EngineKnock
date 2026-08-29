# EXP-002/003: TIME-spectrum baseline vs ORDER-synchronous, equal backbone (PCA)

**Status:** in_progress
**Date:** 2026-08-29

## Hypothesis
An order-synchronous representation (tacho-resampled to fixed samples/rev, then spectrum)
gives a normal-only PCA detector higher ROC-AUC at separating FAULT windows from held-out
NORMAL windows than a construct-matched TIME-spectrum representation, because fault
signatures are phase-locked to shaft rotation and the order domain concentrates that energy
into stable bins while speed variation smears it in the frequency domain.

## Setup
- Data: MaFaulDa normal.zip (49 files, inliers) + horizontal-misalignment.zip (fault).
- Channel: underhang axial accel (col1). Window: 16 revolutions, 256 samples/rev -> dim 2049.
- Detector: PCA (k=32) recon error, standardized bins. Same detector for all modes (one pass).
- Modes: TIME, ORDER, ORDER-degenerate (uniform angle map; invariant), ORDER-shuffled (control).
- Metric: ROC-AUC(normal-test vs fault). Plus normal-vs-normal null AUC (~0.5 expected).

## Sanity checks (stated in advance)
- null(N-vs-N) AUC ~ 0.5 for every mode. [PASS: 0.512/0.505/0.479/0.511 on normal-only run]
- degenerate invariant: ORDER-degenerate AUC ~= TIME AUC (uniform map removes the tacho warp).
- tacho-shuffle: ORDER-shuffled AUC < ORDER AUC (destroying the true angle map removes the edge).
- If ORDER does NOT beat TIME on misalignment, note it: misalignment is a 1x/2x-order
  phenomenon; the strongest case for order-sync is bearing faults (underhang/overhang) --
  escalate to those before concluding.

## Headline numbers (ch1, 16 revs, 256 samp/rev, PCA k=32, nfault=2107)
| Mode | Fault AUC | Null(N-vs-N) |
|---|---|---|
| TIME | 0.856 | 0.512 |
| ORDER | 0.872 | 0.505 |
| ORDER-degenerate (invariant) | 0.851 | 0.479 |
| ORDER-shuffled (control) | 0.850 | 0.511 |

## Sanity checks — result
- null(N-vs-N) ~0.5 all modes: PASS.
- degenerate invariant 0.851 ~= TIME 0.856: PASS (uniform angle map reproduces time mode).
- tacho-shuffle 0.850 < ORDER 0.872: PASS (destroying the true angle map removes the edge).
  Three independent non-order numbers cluster 0.850-0.856; only true order reaches 0.872.

## Conclusion
Mechanism validated (controls clean), but the gain is only +1.6 AUC pts on misalignment --
NOT categorical. This matches the pre-registered caveat: misalignment is a 1x/2x-order
phenomenon, a weak test. Two escalations queued: (a) bearing faults (angle-locked impulses,
where order-sync should have a bigger edge); (b) the SPEED-SHIFT regime (train normal on a
narrow speed band, test faults across all speeds) where order-domain features are
speed-invariant by construction and time-frequency bins move with speed -- the regime where
order-sync could be structurally (not incrementally) superior. (b) is the higher-leverage
reframe and is tested next (EXP-003b) on data in hand before downloading bearing sets.
No bootstrap CI yet; add per-window score saving + bootstrap in EXP-003b.

## Status: completed (weak-positive; reframed, not accepted as final)
