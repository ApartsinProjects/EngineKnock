# EXP-010: MIMII-DG cross-dataset breadth (discrete + continuous attribute)
Status: completed (positive). MIMII-DG (DCASE 2022 DG), bearing (rotation speed=continuous) +
valve (operation pattern=discrete), section_00. Log-mel(128) mean+std -> PCA recon error;
unconditional vs attribute-conditional (kNN-in-attr). Metric AUC per source/target domain.
Result: bearing source 0.609->0.649, target 0.623->0.645; valve source 0.690->0.734,
target 0.376->0.632 (unconditional below chance on shifted target; conditioning recovers it).
Conclusion: the conditioning principle transfers to a different (acoustic) dataset and to a discrete
covariate; sharpest gain on the domain-shifted target. Framed as breadth, not a leaderboard head-to-head.
Results: results/exp010_mimii_results.json. Folded into paper Table 4.
