# EXP-016 (RunPod): leakage-free leave-one-speed-out, cross-fitted (review W3/W4/W5)
Ran on RunPod (local host commit-limit failures). Leave-one-RECORDING-out over 44 healthy
recordings; conditional mean cross-fitted (held-out recording excluded); sklearn AUC; cluster
bootstrap over recordings; scipy paired Wilcoxon.
matched_leaky (window-level, reproduces exp011): base 0.956, inv 0.954, cond 0.989.
LOSO leakage-free: base 0.901±0.089 CI[.874,.928]; inv 0.820±0.150 CI[.777,.865];
cond 0.956±0.050 CI[.941,.971]. Wilcoxon cond-vs-base p=1e-11, cond-vs-inv p=1e-13.
interp/extrap (cond) 0.955/0.993. => core result SURVIVES + STRENGTHENS leakage-free
(advantage over base +0.055, over inv +0.136). results/exp016_loso_leakagefree.json.
