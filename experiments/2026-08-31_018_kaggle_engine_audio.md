# EXP-018: Kaggle car-diagnostics engine audio — CONFOUNDED (diagnostic only)

**Status:** completed (confounded; NOT for paper)
**Date:** 2026-08-31

## Data
Kaggle "Car Diagnostics" (malakragaie), real running-engine audio, mono 44.1 kHz, ~1.5 s clips.
- idle state: normal_engine_idle (264) vs low_oil (107), power_steering (129), serpentine_belt (116),
  + combined-fault folders.
- startup state: normal_engine_startup (61) vs bad_ignition (62), dead_battery (57).
Also downloaded Kaggle "ai mechanic" (eoinedge, BMW M54) — tiny (4 normal clips), not used.

## Pipeline (user's proposed recipe)
Avg log power spectrum (2048/1024) per clip -> PCA and dense AE trained on NORMAL only ->
reconstruction residual -> threshold at 99th pct of training-normal.

## Superficial result
idle: PCA pooled AUC 0.985, AE 0.987 (per-fault 0.95-1.0). startup: PCA 0.882, AE 0.892.

## Controls (verify-before-reporting) -> RESULT IS CONFOUNDED
- Normal-vs-normal NULL: train on half of normal-idle, score other half -> AUC **0.821** (want ~0.5).
  The normal class is heterogeneous; held-out normal already scores "anomalous". FATAL for the AUCs.
- Healthy FPR at the 99th-pct threshold = 0.237 (PCA) / 0.15 (AE): threshold does NOT hold on normal.
- RMS-only AUC: normal-vs-low_oil 0.70 (low_oil louder), normal-vs-power_steering 0.44; duration-only 0.5.
  So NOT a pure level/duration confound, but recording/session heterogeneity dominates.

## Verdict
Found-data engine audio carries uncontrolled recording heterogeneity; no session metadata to split on.
The strong AUCs reflect session separation, not fault signatures. Diagnostic only; NOT in paper.
Scientific takeaway (reinforces thesis): fixed-threshold plain AE fails its own null (0.82) and throws
24% false alarms on real non-stationary engine audio -> motivates condition-aware modeling + leakage-free
evaluation, which we validate on MaFaulDa (controlled rig with a measured speed covariate).
