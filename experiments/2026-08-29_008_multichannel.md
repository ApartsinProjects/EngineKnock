# EXP-008: multi-channel (accel×2 + microphone) + fusion

**Status:** completed (positive)
**Date:** 2026-08-29

## Hypothesis
Speed-conditioning wins across channels (not just accel col1), including the microphone, and
fusion improves.

## Setup
- Channels: 1 underhang-axial, 4 overhang-axial, 7 microphone. Pooled all-fault MaFaulDa.
- Uncond vs cond (kNN-in-speed) PCA; per channel + FUSED (z-scored recon-error average).
- matched (3 seeds) + LOBO. Memory-light (streaming parse, one fault at a time).
  Results: results/exp008_mc_results.json. (First attempt exp008 hit MemoryError; rewrote
  exp008b streaming + score-accumulation.)

## Headline numbers (matched / cross-speed LOBO)
| Channel | uncond | cond |
|---|---|---|
| underhang-axial | 0.957 / 0.897 | 0.990 / 0.937 |
| overhang-axial  | 0.726 / 0.591 | 0.889 / 0.952 |
| microphone      | 0.873 / 0.658 | 0.936 / 0.716 |
| FUSED           | 0.940 / 0.830 | 0.992 / 0.959 |

## Conclusion
Conditioning wins on EVERY channel (accel×2 + mic) in both regimes; overhang gains are large
(cross-speed 0.591->0.952). Secondary finding: fusion HURTS without conditioning (0.940 < best
single 0.957, weak channels drag the average) but is BEST with conditioning (0.992/0.959) --
conditioning enables effective multi-channel fusion. Answers the "single channel" concern.

## Status: completed (positive)
