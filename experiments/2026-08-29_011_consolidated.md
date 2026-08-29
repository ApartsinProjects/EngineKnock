# EXP-011: consolidated one-pass evaluation (single source of truth)
Status: completed. MaFaulDa accel col1, base/inv/cond, one config -> results/exp011_consolidated.json.
Matched (10 seeds): base 0.956±0.004 (pAUC 0.866, CI[.941,.965]); inv 0.954±0.007 (pAUC 0.832);
cond 0.989±0.002 (pAUC 0.949, CI[.981,.993]). LOBO per-band mean±std: base 0.897±0.033,
inv 0.835±0.095, cond 0.937±0.041. Wilcoxon: matched cond-vs-base/inv p=0.005; LOBO p=0.046/0.028.
Resolves the 0.955/0.957/0.961 cross-table inconsistency (canonical: base 0.956/0.897, cond 0.989/0.937).
