# EXP-013: invariance-arm tuning (defeats the under-tuned-strawman objection)
Status: completed. Order features (MaFaulDa accel col1), sweep detector on the INVARIANCE arm.
PCA rank {16,32,64,128,256}: matched .941-.967, LOBO .824-.837. AE lat {16,32,64}: matched ~.96,
LOBO .777-.806. BEST invariance LOBO = 0.837 (PCA-256), still BELOW unconditional baseline (0.897)
and far below conditioning (0.937). Matched improves with capacity but cross-speed plateaus: the
limitation is representational (order tracking discards fixed-Hz fault energy), not tuning.
results/exp013_invariance_tuning.json. Folded into paper (backbone/invariance paragraph).
