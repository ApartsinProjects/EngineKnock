# EXP-009: strong-invariance rebuttal (deep backbone; OASSL stand-in)
Status: completed (positive). Pooled MaFaulDa, {PCA,AE} x {uncond, order-invariance, speed-cond}.
Result (matched/held-out AUC): PCA 0.957/0.897 | 0.959/0.835 | 0.990/0.937 ;
AE 0.952/0.890 | 0.960/0.809 | 0.984/0.933.
Conclusion: invariance (order tracking) loses cross-speed at BOTH backbones (0.835, 0.809 < uncond),
so a deep backbone does not rescue it; conditioning wins regardless. OASSL (MSSP 2026) is paywalled
with no code, so a deep AE on order-tracked features is our reproducible in-class representative.
Results: results/exp009_results.json. Folded into paper Table 2.
