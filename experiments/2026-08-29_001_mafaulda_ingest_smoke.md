# EXP-001: MaFaulDa ingest + order-resampling smoke test

**Status:** in_progress
**Date:** 2026-08-29

## Hypothesis
MaFaulDa provides synchronized tachometer + accelerometer + microphone channels at a
constant sample rate, from which we can (a) estimate instantaneous shaft rotation from the
tacho pulse train and (b) resample a vibration/acoustic window into a fixed number of
samples-per-revolution (the order/angle domain). If this pipeline runs end-to-end on one
file and the recovered rotation speed matches the file's labeled RPM, the order-synchronous
representation is feasible and the project can proceed to baselines (EXP-002).

## Setup
- Data: MaFaulDa (UFRJ). Download a few files: >=2 normal (different speeds) + 1 fault.
- Expected channel layout to CONFIRM: col0 tacho, col1-3 underhang accel (axial/radial/tang),
  col4-6 overhang accel, col7 microphone; fs = 50 kHz; 5 s per file (~250k rows).
- Tools: numpy 2.2.6, scipy 1.17.1, local CPU.
- Code commit: (to fill)

## Procedure
1. Resolve MaFaulDa download mechanics (per-file vs tarball).
2. Download a minimal subset into data/mafaulda_smoke/.
3. Load one file; confirm shape, fs, channel semantics (which col is tacho, which is mic).
4. Estimate rotation frequency from the tacho pulse spacing; compare to the labeled RPM in
   the file path/name.
5. Order-resample one accel channel to fixed samples/rev; plot a few overlaid revolutions
   to confirm cycle alignment.

## Sanity checks (stated in advance)
- Tacho-derived rotation frequency must match labeled RPM within a few %.
- Overlaid revolutions after order-resampling should visually align (periodic structure
  preserved), unlike overlaid fixed-time windows.

## Headline numbers
- normal.zip: 49 CSV files, 250000x8 each, fs=50 kHz. Confirmed layout: col0 tacho,
  col1-3 underhang accel, col4-6 overhang accel, col7 mic.
- Tacho-estimated rotation vs filename nominal (Hz): 12.08/12.29, 35.56/36.45, 60.02/61.44.
  Ratio est/nominal ~0.977-0.983 across all three speeds (consistent ~2% actual-below-nominal;
  irrelevant to method since order-resampling keys off actual pulses).
- Order-resampling: 36.4544.csv -> 177 revolutions x 720 samples/rev, clean.
- Overlay plot results/exp001_order_overlay.png: order-domain revolutions align at sharp
  recurring events (~25 deg, ~115 deg dips) where fixed-time windows smear. Premise holds.

## Sanity checks — PASSED
- Tacho freq matches nominal within ~2% (invariant met); consistent bias explained (actual<nominal).
- Order-domain overlay visibly tighter than fixed-time control at angle-locked features.

## Conclusion
Ingest + tacho rotation estimation + order-domain resampling all work end-to-end on real
MaFaulDa data. The order-synchronous representation is feasible and its premise (angular
coherence lost by generic windowing) is visible even on a healthy bearing. Proceed to
EXP-002 (unsupervised baseline: normal-only train, normal+fault test) then EXP-003
(order-sync vs plain-spectrogram at equal backbone).

## Status: completed
