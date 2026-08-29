# Paper outline — Speed-Conditional Unsupervised Machine Anomaly Detection

**Working title:** "Condition, Don't Invariance: Operating-Speed-Conditional Unsupervised
Anomaly Detection for Rotating Machinery"

**One-line contribution:** For unsupervised (normal-only) machine anomaly detection under
operating-condition shift, conditioning the normal model on a *measured continuous covariate*
(tachometer speed) beats both unconditional detectors and the standard *speed-invariance*
(order-tracking) approach — across all fault types, both matched- and cross-speed, and
independent of backbone.

## Claims (each backed by a co-computed experiment)
1. **Conditioning > baseline**, all 5 MaFaulDa fault types, matched + cross-speed (EXP-006).
2. **Conditioning > invariance (order-tracking)**; invariance never beats baseline under
   speed shift (EXP-006; order-sync analysis EXP-003).
3. **Backbone-agnostic**: same gain under PCA and AE; conditioning dominates backbone choice
   (EXP-007).
4. **Multi-channel / mic**: holds across accelerometer channels and the microphone, and fusion
   improves (EXP-008). [pending]
5. **Mechanism confirmed**: shuffling the normal-test speed covariate collapses AUC
   0.961->0.419 (EXP-005 guard) — the gain is causally the speed-conditioning.

## Section plan
- **1 Intro**: field uses NVH/vibration for machine health; unsupervised (normal-only) is the
  realistic setting (faults rare/unlabeled); operating-condition shift (speed/load) is the
  core difficulty (cite DCASE-DG framing). Gap: prior work removes speed (invariance /
  order-tracking) or uses discrete domain labels; none condition on the measured continuous
  covariate. Contribution = speed-conditional normal modeling.
- **2 Related work**: (a) unsupervised machine AD (DCASE Task 2, MIMII, autoencoder/embedding
  +Mahalanobis); (b) domain shift / domain generalization AD (MIMII-DG, discrete domains);
  (c) speed-invariant fault diagnosis / order tracking (OASSL, order-tracking memory-AE) =
  invariance; (d) conditional/contextual AD (CFLOW-AD position, ID-conditioned AE) — condition
  on non-physical variables. Position: continuous physical-covariate conditioning is the gap.
- **3 Method**: window by revolutions; log-spectrum features; conditional normal model =
  kNN-in-speed mean subtraction + PCA/AE recon error; contrast with invariance (order-track).
- **4 Experiments**: MaFaulDa (accel+mic+tacho, 5 fault types, continuous speed 12-60 Hz).
  Protocols: matched-range, leave-one-band-out (cross-speed). Baselines: unconditional,
  invariance(order). Metrics: ROC-AUC + bootstrap/seed variance.
- **5 Results**: Table (5 faults x {base,inv,cond} x {matched,LOBO}) [EXP-006, fig done];
  backbone table [EXP-007]; multi-channel [EXP-008]; mechanism guard [EXP-005].
- **6 Discussion/limitations**: validated on MaFaulDa test-rig with a clean tacho; extension
  to load/temperature covariates and to discrete-attribute conditioning (MIMII-DG) is future
  work; bearing faults near-saturate at matched speed (gains concentrate on harder faults and
  cross-speed).

## Baselines to cite (from scout)
- Unsupervised MaFaulDa AE AUC 0.86 (verify source). DCASE-DG methods. OASSL (order-adaptive,
  invariance) = nearest competitor; cite + differentiate (conditioning vs invariance).
  CFLOW-AD, ID-conditioned AE (conditioning on non-physical vars).

## Assets
- Results JSON: exp006_results.json, exp007_ae_results.json, exp008_mc_results.json.
- Figure: fig_exp006_multifault.png. Registry: experiments/INDEX.md.
- Skills for writeup: paper-build (house style HTML/DOCX), paper-reviewer (tone/abstract),
  bibtest (citation validation).
