# EXP-006: multi-fault-type external validity (baseline vs invariance vs conditioning)

**Status:** in_progress
**Date:** 2026-08-29

## Hypothesis
The speed-conditional win (EXP-005) generalizes across MaFaulDa fault types, and beats both
the unconditional baseline and the invariance (order-tracking) competitor on each fault type
and pooled.

## Setup
- Normal + all 5 fault categories (horizontal/vertical misalignment, imbalance, underhang,
  overhang). Accel col1, 16-rev windows.
- Three normal-only detectors: baseline (unconditional PCA / time-spectrum), invariance
  (PCA / order-tracked spectrum), conditioning (kNN-in-speed conditional PCA / time-spectrum).
- Protocols: matched-range (5-seed mean+-std) + leave-one-band-out (interpolation).
- Per fault type + POOLED. cache mf_<stem>.npz per source.

## Pre-registered expectations / gates
- Conditioning > baseline on matched-range for EVERY fault type (per-seed, small std).
- Conditioning >= baseline on LOBO for most fault types (boundary bands may tie).
- Invariance (order) ~<= baseline (consistent with EXP-003 refutation) -> supports the
  "conditioning beats invariance" narrative. If invariance unexpectedly wins on bearing
  faults, report honestly and investigate (bearing defect freqs are order-locked).

## Headline numbers (accel col1; matched-range 5-seed mean±std; LOBO = leave-one-band-out mean)
| Fault | base | invariance(order) | cond(ours) | LOBO base/inv/ours |
|---|---|---|---|---|
| horizontal-misalignment | 0.854±.01 | 0.873±.01 | 0.957±.01 | 0.689/0.627/0.794 |
| vertical-misalignment | 0.977±.00 | 0.953±.01 | 0.996±.00 | 0.970/0.872/0.984 |
| imbalance | 0.858±.01 | 0.891±.01 | 0.969±.01 | 0.741/0.669/0.852 |
| underhang (bearing) | 0.998±.00 | 0.992±.00 | 1.000±.00 | 0.997/0.960/0.999 |
| overhang (bearing) | 0.981±.00 | 0.974±.00 | 0.995±.00 | 0.939/0.873/0.955 |
| POOLED | 0.955±.00 | 0.955±.01 | 0.989±.00 | 0.897/0.835/0.937 |

Results saved: results/exp006_results.json. (First run OOM'd on underhang float64; rewrote
memory-light exp006b with float32 + pooled-via-score-accumulation. 4.9 min.)

## Conclusion (COMPLETE, POSITIVE)
Conditioning (speed-conditional) wins EVERY fault type in BOTH regimes. Pooled: matched
0.955->0.989, cross-speed (LOBO) 0.897->0.937. Invariance (order) NEVER beats baseline under
speed shift on any fault type (crux of "conditioning beats invariance"). Bearing faults near-
saturate at matched speed (ceiling), so headline matched gains are on misalignment/imbalance
(0.85->0.96-0.97); cross-speed gains are universal. External validity across fault types
established on MaFaulDa (accel col1).

## Remaining to harden for paper
- Multi-channel (used accel col1 only; add mic + other accels / fuse).
- Compare vs named competitor OASSL (order-adaptive unsupervised) if reproducible.
- Optional AE backbone (PCA already shows effect); breadth via MIMII-DG attribute-conditioning.

## Status: completed (positive, multi-fault validated)
