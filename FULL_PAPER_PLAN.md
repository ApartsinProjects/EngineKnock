# Full-paper expansion plan + review (Fable agent, 2026-08-29)

Target venue: **Mechanical Systems and Signal Processing (MSSP)** (OASSL's venue; invariance-vs-
conditioning speaks natively there). Secondary: IEEE T-IM. ~30 manuscript pp, ~45 refs.

## Must-have NEW experiments (A)
- A1 [MUST] Consolidated ONE-PASS eval: one script/config/artifact -> per-fault + pooled AUC+pAUC
  for base/inv/cond, both regimes, per-band LOBO breakdown, 10 seeds matched, paired bootstrap/
  DeLong CIs, Wilcoxon cond-vs-base and cond-vs-inv. Fixes the 0.955/0.957/0.961 inconsistency.
- A2 [MUST] k-sensitivity sweep (k=1..all), both regimes, with degenerate-endpoint invariants.
- A3 [MUST] Discrete-bin conditioning baseline (4/8/16 bins) vs continuous kNN -> converts the
  "continuous covariate" novelty from assertion to measurement. HIGHEST-VALUE CHEAP ITEM.
- A4 [MUST] Investigate overhang matched<LOBO inversion (0.889<0.952) + mic LOBO gap. Bug until
  explained (verify-before-reporting). Dump per-band/per-file scores.
- A5 [MUST, highest risk] Strengthen invariance arm: sweep order resolution/order-count + AE
  capacity with equal budget; attempt OASSL from [9]; fallback = tuned sweep + hyperparam table.
- A6 [MUST] MIMII-DG expansion: all 7 machine types + official DCASE 2022 AE baseline + source/
  target/harmonic-mean.
- A7 [SHOULD] Local-Mahalanobis conditional (per-neighbourhood covariance) vs conditional-mean.
- A8 [SHOULD] Tacho-noise sensitivity (jitter s 1-10%).
- A9 [NICE] mel vs linear spectrum + window ablation; signal-estimated speed (no tacho); DEFault row.

## Must-have WRITING (B)
- B1 [MUST] Formal problem + method section (equations for m(s)=mean kNN, r=x-m(s), score;
  algorithm box; hyperparameter table; complexity note).
- B2 [MUST] Rebuild all tables from A1 artifact with CIs + significance; fix footer; reconcile
  every in-text number to the artifact.
- B3 [MUST] Rewrite novelty claim (narrower scope; contextual/conditional AD prior art exists:
  Song et al. 2007, CVAE detectors, contextual TS anomaly); expand related work to ~45 refs
  (web-researcher + bibtest). Fix flagged prose.
- B4 [MUST] New figures: motivating spectra (fault hiding in speed spread), method schematic,
  speed-vs-score scatter, per-band degradation curves, ROC panel, k-sweep plot.
- B5 [SHOULD] Scope/failure-analysis section (from A4 + extrapolation-band analysis).
- B6 [SHOULD] Reproducibility statement + Zenodo deposit; journal-title front-matter variant.
- B7 [NICE] paper-reviewer polish + paper-build regenerate-all.

## Critical review — key issues to fix
1. OVERCLAIM: "conditioning on a measured continuous covariate has not been studied" -> narrow to
   "not for tachometer speed in normal-only machine-fault detection with a head-to-head vs
   order-tracking invariance". Contextual/conditional AD exists (Song 2007, CVAE, contextual TS).
2. Invariance = possible strawman; OASSL stand-in invites "under-tuned" attack. Tune it equally
   or implement OASSL; cite classical order-tracking, not just [9].
3. NUMBER INCONSISTENCIES: base pooled 0.955 (T1) vs 0.957 (T2) vs 0.961 (mechanism para, no
   provenance); AE gain text +0.032/+0.043 vs exp007 +0.0305/+0.0442 vs LOG +0.030/+0.044;
   overhang cond LOBO 0.952 > matched 0.889 (unexplained ordering); footer says EXP-005-008 but
   uses 009/010. Fix via A1 single-artifact rebuild.
4. LOBO under-specified: no band edges/count, no per-band values, no variance; thesis lives here.
5. MIMII-DG: 2/7 machines, no DCASE baseline, tiny bearing gains w/o variance -> reads selective.

Prose fixes: journal title without contraction/imperative; drop "honest formulation", "the wrong
move/right one", "these are directions... not caveats" (argues with imagined critic); "confirms
causation" -> "requires the true speed labels"; soften "almost always instrumented with a tacho".

## Execution order
Must-have core = A1-A6 + B1-B4. Highest risk A5; highest-value-cheap A3. Start: A4 (correctness
gate) + A1 (foundational artifact), then A3, A2, then writing B1/B2, then A5/A6, then figures B4.
