# EXP-017: Engine Journal Bearings (Mendeley 3fcrrdjjvk) — real-hardware decision-rule check

**Status:** completed (inconclusive / diagnostic only — NOT for paper)
**Date:** 2026-08-31

## Dataset reality (corrects the scout's "3.2 kHz csv waveform" claim)
Random-vibration SHAKER test of engine journal bearings, not a running engine.
- 254 MB main zip = 547 Word/RTF test-report documents (summary RMS + plot images only; no signal).
- 12 MB companion = 134 CSV waveforms, 44 healthy / 88 usable faulty, 4 accel channels,
  ~296 Hz, ~2 s each. Covariates: RPM {1000,1500,2000}, Humidity {0,50,100%}, Temp {-10..45C}.
- Shaker-response, low-rate (Nyquist ~148 Hz), short, small.

## Method
Per-record concat of 4-channel log-mag spectra (512-pt) -> PCA recon-error detector on healthy.
IGNORE vs CONDITION-on-RPM (per-RPM healthy mean subtracted) vs CONDITION-on-humidity.

## Result
- Pooled (in-distribution ref): AUC 1.0 for all arms -> saturates/trivially separable, not leakage-free.
- Leave-one-RPM-out (leakage-free): ignore 0.705, condition-RPM 0.729 (+0.024, 3 folds).

## Verdict
Weakly consistent with condition >= ignore, but small and noisy; dataset too small/low-rate/
shaker-driven to be credible engine evidence. Stays in registry as diagnostic; NOT in the paper.
Real running-engine data (Kaggle BMW ai-mechanic / Car Diagnostics .wav) is the proper path but
needs a Kaggle API token (none on this machine).
