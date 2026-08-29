# A4: overhang matched<LOBO 'inversion' -> NOT a bug (macro vs micro aggregation)
overhang-axial cond matched 0.889 (train 60%) / 0.901 (train 83%); per-band LOBO cond
{12-20:.894,20-28:.999,28-36:.972,36-44:.983,44-52:.866,52-61:.997} mean 0.952.
Cause: LOBO AUC = per-band MACRO-average (each band scored fault-vs-normal independently, removing
cross-speed ranking confusion); matched AUC = single POOLED micro-average across all speeds. For a
channel with high but variable within-band separability (overhang), the macro-mean can exceed the
pooled score. Document as a scope note; the two aggregations are not directly comparable.
