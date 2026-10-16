# NEXT PROJECT: Which frame is blind to which fault

Working title: *A healthy-data frame map predicts, and a two-frame normal-only detector exploits, competing
speed-equivariances in variable-speed machinery.*

Authors: Alexander Apartsin (Holon Institute of Technology), Yehudit Aperstein (Afeka).
Status: proposal after paper-discovery Cycle 3 (deepen + break), 2026-10-15. Supersedes the
ignore/condition/remove paper as the next build. Decision inputs: `scratchpad/discovery_brief.md`,
`discovery_cycle1.md`, `discovery_cycle2.md`; smoke numbers in `results/twoframe.json` and
`results/twoframe_perclass.json` (script `exp_twoframe.py`, RunPod wrapper `runpod_twoframe.py`).

## 1. Framing

Every speed-invariant representation of a rotating-machine vibration signal is blind to one half of the
fault physics. Order tracking (ORD) holds shaft-synchronous content still and smears structural
resonances; the physical-frequency spectrum (HZ) holds resonances still and smears synchronous content.
The shipped paper already showed, on the one leakage-free public benchmark (MaFaulDa, leave-one-speed-out),
that the field-default "remove speed by order tracking" is the worst treatment for normal-only anomaly
detection (0.820 vs 0.956 for speed-conditioning, p~1e-11) and built the asset this project reuses: three
co-registered frames (HZ, REV, ORD) of every window with its instantaneous speed, a speed-conditioned
normal-only detector (kNN-in-speed healthy-mean subtraction plus PCA/AE reconstruction error) with
leakage-free LOSO, bootstrap CIs and Wilcoxon, and the validated physics split (structural/alignment faults
stationary in HZ, synchronous/bearing faults stationary in order). This project turns that split into two
things: a **frame map** (C2), computed from healthy speed sweeps only, that scores per spectral band how
speed-stable the healthy background is in each frame and therefore how much detectability each frame has
in that band; and a **two-frame detector** (C1) that scores every window in HZ and ORD and fuses the two
with a parameter-free, validity-preserving merge of held-out-healthy percentiles. The map predicts which
fault families each frame is blind to and by how much; the detector's per-fault winner must match the map,
its fusion must beat both single frames, and an HZ+REV control (two frames of the same invariance family)
must not gain. The motivating puzzle is Domingues et al. (PHM Society 2024): supervised, on Ottawa
variable-speed bearings, the order domain came out worst on outer-race and ball faults and the authors
offered no explanation. Competing equivariance explains it, and this paper demonstrates it normal-only.

## 2. Why this, not the others

- **C5 Campbell factorization: dead.** The healthy speed-frequency grid with per-bin noise-floor statistics
  is already patented (Rolls-Royce US9874472B2, https://patents.google.com/patent/US9874472B2/en) and
  the healthy-TFR-baseline distance is Klein PHM 2013
  (https://papers.phmsociety.org/index.php/phmconf/article/view/2250); the attribution half is circular and
  a low-rank factorization will not beat the non-parametric kNN conditioning at 0.956.
- **C3 invariance-tax theory: folded** into the discussion of this paper (methods-venue reframing only).
- **C4 protocol inversion (within-speed vs LOSO flips the ranking): second fallback.** Crowded by Vieira et
  al. MSSP 258 (2026), arXiv 2509.22267.
- **C6 conformal frame selection and C7 MIMII-DG replication: instrumental**, absorbed as the fusion rule
  (validity-preserving p-value merge) and the cross-setting replication (section 8, phase 5).
- **Three confounded datasets: confound demonstrations only.** Kaggle audio (null 0.82), Mendeley
  journal-bearing (within noise), Ottawa variable-speed (energy plus session confound). They appear in the
  paper as evidence that benchmark confounds, not frames, drive naive results.
- **Engine / gas-genset application framing: dropped.** No public engine-combustion waveforms exist.
- **Healthy-placebo benchmark-admissibility pivot: killed** (PlaceboAUC 0.59 gave no separation).

## 3. Load-bearing assumption, verified

**Assumption.** A frame map computed from HEALTHY speed sweeps only (per band, the speed-variance of the
healthy spectrum in HZ vs in ORD) adds predictive value beyond the textbook physics "resonances sit at fixed
Hz, synchronous content at integer orders". If the per-fault winning frame is fully fixed by the fault
signature's a-priori slope, the map is decorative and C2 collapses to C1.

**Break attempt against primary sources (what the field already states).**
- Siemens Simcenter KB, *Order tracking: fixed sampling versus synchronous sampling*
  (https://community.sw.siemens.com/articles/en_US/Knowledge/order-tracking-fixed-sampling-versus-synchronous-sampling):
  "In a fixed sampled map, resonances appear as vertical lines at fixed frequencies"; in the order-tracked
  map "some resonance information is truncated". Qualitative only; its only numbers are order-smear widths
  (1 Hz at 1st order, 4 Hz at 4th order over 1000 to 1060 rpm). No per-band rule, no healthy-variance
  criterion, no detectability statement.
- Abboud, Antoni, Eltabach, Sieg-Zieba, *The speed-dependent spectral correlation* (CETIM copy,
  https://www.cetim.fr/mecatheque/Publications-scientifiques-et-techniques/The-speed-dependent-spectral-correlation)
  and *Order-frequency analysis of machine signals* (HAL hal-01714332, MSSP; HAL is bot-blocked, CETIM
  abstract read): kinematic effects are angle-periodic, the transfer function "imposes time-invariant
  resonance frequencies"; the OFSC is a joint order-frequency tool. The dichotomy is stated as physics; no
  detectability rule, no representation choice per fault, no healthy-only probe.
- Antoni, *Cyclostationarity by examples*, MSSP 23 (2009) 987-1036, DOI 10.1016/j.ymssp.2008.10.010:
  tutorial on cyclic modulation spectrum and spectral coherence; angle/time cyclostationarity is the later
  Abboud et al. Measurement 2015 line (HAL hal-01714350). Nothing on per-frame detectability.
- Klein, *A method for anomaly detection for non-stationary vibration signatures*, PHM Society 2013
  (https://papers.phmsociety.org/index.php/phmconf/article/view/2250): healthy-machine TFR baseline in
  the RPM-order domain, distance TFR, region characterization. Single frame; no HZ-vs-ORD comparison; no
  per-bin variance as a detectability quantity.
- Rolls-Royce US9874472B2 (https://patents.google.com/patent/US9874472B2/en): per speed-frequency bin,
  fit a Gamma distribution to normal-run data and set a Gumbel 0.99 noise-floor threshold (20 x 100 bins).
  This IS a per-bin healthy "denominator", but in one frame (speed-conditioned HZ) and never used to
  compare frames or to predict per-fault blindness.
- Schmidt and Gryllias, Measurement 169 (2021) 108517 (https://repository.up.ac.za/handle/2263/86233):
  healthy historical data plus optimisation-based band selection under time-varying speed. Single frame,
  gearbox, band chosen by a fault-sensitivity objective, not a cross-frame speed-variance map.

**Does the a-priori slope rule already fix the per-fault winner?** Partly, and that is exactly where the
map earns or loses its place. For bearings and imbalance the textbook slope (order-locked) predicts ORD and
the smoke agrees (underhang 0.910 vs 0.849, overhang 0.847 vs 0.819, imbalance 0.627 vs 0.528). For
misalignment the textbook says 1x/2x shaft orders, i.e. order-locked, so the slope rule predicts ORD; the
smoke says the opposite (horizontal misalignment HZ 0.801 / ORD 0.500 = chance; vertical 0.788 / 0.691).
So the a-priori signature location does NOT fix the winner for this fault family: on this rig misalignment
is detected through speed-fixed structural content, not through its nominal orders. That is the strongest
evidence that something machine-specific decides the winner, and the strongest risk, because the healthy
map on its own supplies no numerator (it does not know where misalignment energy lands either).

**Does the "detectability denominator" defense hold?** Partly. Per-bin healthy statistics in one frame are
prior art (US9874472B2). What is not in any source is the cross-frame comparison: for each band, the
ratio of healthy speed-variance in HZ to healthy speed-variance in ORD, which says where each frame has a
clean background and where it has a smeared one. That ratio is machine-specific (it depends on the rig's
resonances and speed range), not derivable from the textbook, and it is estimable from healthy sweeps
alone. But a denominator alone does not predict a per-fault winner; a winner needs numerator and
denominator. The defense therefore holds at BAND level and holds for a fault only when combined with
where that fault's energy lands.

**VERDICT: MUST BE SCOPED.** The map is retained as a *per-band frame-blindness map*, not a law and not a
stand-alone per-fault oracle. It makes two claims, each falsifiable without fault labels at map-building
time:
1. Band-level (label-free, primary): the map predicts, per band, the detection AUC of each frame for an
   anomaly injected in that band, including the magnitude of the ORD-vs-HZ gap, across bands where both
   frames work, where only one works, and where neither works. Tested by controlled injection (Hz-fixed and
   order-locked tones at matched SNR) into held-out healthy windows and by band-restricted detectors on real
   faults. The a-priori rule predicts only the sign; the map predicts the number. A shuffled-band map must
   predict at chance.
2. Fault-level (secondary, one piece of fault knowledge allowed): per-fault winner = argmax over frames of
   map-weighted energy of the fault's empirical signature footprint (its band-energy profile, estimated once
   from a handful of fault windows or, for the textbook variant, from the nominal fault orders). The
   textbook-numerator variant is reported as a prediction that may fail (misalignment is where it is
   expected to fail); the empirical-footprint variant must succeed out-of-sample (fit the footprint on one
   speed, predict winners on the other speeds, and transfer to a second rig if available).
If claim 1 fails (map no better than shuffled on injected anomalies), C2 is dropped and the paper is C1
alone: the normal-only HZ+ORD detector with validity-preserving fusion, per-fault winner flip, HZ+REV
control, Domingues explanation, and the confounded datasets as cautionary demonstrations.

## 4. Novelty claim and strongest surviving prior-art threats

**Claim.** First normal-only variable-speed anomaly detector that (a) scores each window in two frames with
opposite speed-equivariance (HZ and ORD) and fuses them without tuned weights by a validity-preserving
merge of held-out-healthy percentiles, (b) reports a pre-registered per-fault winner that flips with fault
physics and a same-family (HZ+REV) control that does not gain, and (c) explains and quantifies per-frame
blindness with a healthy-data frame map that holds out-of-sample at band level. It also explains an
unexplained published result (Domingues 2024).

**Threats and why the contribution stands.**
- OASSL, Order-Adaptive Subspace Scale Learning, MSSP 248 (2026) 114001
  (https://scholars.cityu.edu.hk/en/publications/order-adaptive-subspace-scale-learning-for-unsupervised-anomaly-d/):
  unsupervised, angular resampling to order-frequency features, subspace scale learning against false
  alarms under speed change. Single invariance family (order); no HZ frame, no fusion, no per-fault
  blindness analysis. It is the strongest baseline to beat, and its blind spot (Hz-stationary faults) is
  our prediction P3.
- Wilkinghoff, *Design choices for learning embeddings from auxiliary tasks for domain generalization in
  anomalous sound detection*, ICASSP 2023, DOI 10.1109/ICASSP49357.2023.10097176
  (https://publica.fraunhofer.de/handle/publica/488154): "combining multiple input representations"
  improves ASD. Auxiliary-label supervised embeddings, both representations Hz-based (no angular frame),
  no speed covariate, acoustic DCASE data, no physics-predicted winner. Our HZ+REV control is designed to
  separate "any extra representation helps" from "opposite equivariance helps".
- Domingues et al., PHM Society Annual Conference 2024
  (https://papers.phmsociety.org/index.php/phmconf/article/view/4146): supervised deep learning on
  time/frequency/angular/order inputs under varying speed; frequency best overall, order no advantage
  (abstract); per-class Table 7 order-worst on outer-race and ball as read in Cycle 2 (re-verify against the
  PDF before citing the numbers). No mechanism given. We supply the mechanism, normal-only.
- Rolls-Royce US9874472B2 and Klein PHM 2013 (URLs above): per-bin healthy baselines in ONE frame. The
  cross-frame map and its per-band blindness prediction are not there.
- Vieira et al., MSSP 258 (2026), arXiv 2509.22267 (https://arxiv.org/abs/2509.22267): leakage-aware
  evaluation, supervised. We adopt the leakage statement; we do not compete on protocol.
- Fabricated items removed: "CMAE 2025" and "SiAE 2026" do not exist and must not be cited.

## 4b. Venue fit

Siblings and venues (from Cycle 2, URLs where read): OASSL MSSP 248:114001 (above); Vieira MSSP 258
(arXiv 2509.22267); Brito et al. MSSP 181 (unsupervised AD plus SHAP on MaFaulDa-type data); Domingues
PHM 2024 (above); Klein PHM 2013 (above); Wilkinghoff ICASSP 2023 (above); Bruguiere et al., Forum
Acusticum 2026 order-CAE; Michau and Fink, Knowledge-Based Systems 2021 (one-class across operating
conditions); Liang et al., IJPHM 2026 key-order weighting (https://papers.phmsociety.org/index.php/ijphm/article/download/4626/2818).

- **Two-benchmark outcome (Mendeley KFUPM y8mv28hjwb, DOI 10.17632/y8mv28hjwb.1, passes the energy-only and
  healthy-placebo confound checks):** target **MSSP** (Q1). Requirements: OASSL-class baselines (angular
  resampling plus DeepSVDD / AE / OCSVM / IF, HZ and ORD inputs each), map prediction holding out-of-sample
  on both rigs, Vieira-style leakage statement. Accept read: plausible if the map transfers and fusion
  beats order-only on Hz-stationary faults by a CI-backed margin; desk-reject risk if reviewers read the
  map as the textbook dichotomy restated, which the band-level injection test is there to prevent.
- **One-benchmark outcome (Mendeley fails):** target **PHM Society Annual Conference** (first), then
  **IJPHM** or **ISA Transactions** / **Measurement** for the extended version, with MIMII-DG (acoustic,
  discrete domain shift) as the cross-setting replication and Ottawa/Kaggle as confound demonstrations.
  Accept read: good at PHM Society (mechanism plus explanation of their own 2024 paper); MSSP would
  desk-reject a single-rig "law", so the word law is not used anywhere.
- **Baseline to beat:** OASSL-style order-only unsupervised detector (and our own ORD-frame
  speed-conditioned detector as its strongest in-house proxy).

## 5. Headline results as falsifiable predictions (smoke numbers DIRECTIONAL, 1 seed, subsampled)

- **P1 (fusion gain).** HZ+ORD fused macro-AUC exceeds the best single frame by >= 0.05 with a 95%
  bootstrap CI excluding 0 and Wilcoxon p < 0.01 across 44 LOSO folds. Smoke: 0.848 vs 0.757 (+0.091).
- **P2 (winner flips with physics).** Misalignment (horizontal and vertical) is won by HZ; bearings and
  imbalance by ORD; each per-fault gap has a CI excluding 0 except where both frames are near chance.
  Smoke: horiz-mis HZ 0.801 / ORD 0.500; vertical 0.788 / 0.691; imbalance 0.528 / 0.627; underhang
  0.849 / 0.910; overhang 0.819 / 0.847.
- **P3 (order-only blind spot).** An OASSL-style order-only detector scores <= 0.60 AUC on horizontal
  misalignment while the fused detector scores >= 0.80.
- **P4 (same-family control).** HZ+REV fusion gains < 0.02 macro-AUC over the best of HZ and REV, and
  HZ+HZ self-fusion gains exactly 0.
- **P5 (band-level map, label-free).** On injected anomalies (Hz-fixed and order-locked tones at matched SNR
  in held-out healthy windows), the map's per-band frame-gap prediction correlates with the measured
  per-band AUC gap at Spearman rho >= 0.6; a shuffled-band map gives |rho| < 0.2.
- **P6 (fault-level map, out-of-sample).** With the fault footprint estimated on one speed and the map
  from healthy sweeps, the predicted winner matches the measured winner on >= 4 of 5 MaFaulDa fault
  classes on the other speeds. Textbook-numerator variant: predicted to FAIL on misalignment (may go
  either way; a success would mean the map alone carries it, a failure confirms the scoping in section 3).
- **P7 (fusion vs conditioning, may go either way).** Two-frame fusion of speed-conditioned detectors
  beats single-frame speed-conditioning (0.956 macro-AUC in the shipped paper) by >= 0.01; if it does not,
  the paper reports fusion as the gain on the UN-conditioned frames and conditioning as the stronger
  treatment, which is itself a result.
- **P8 (transfer, may go either way, conditional on Mendeley passing confound checks).** The healthy map
  built on Mendeley predicts misalignment = HZ-won and imbalance = ORD-won on that rig; fusion beats
  order-only on misalignment by a CI-backed margin.
- **P9 (acoustic replication).** On MIMII-DG, where no tacho exists and ORD is estimated from the signal,
  two-frame fusion does not lose to the best single frame on any machine type (gain >= 0).

## 6. Experiments

| # | Experiment | Measures | Pre-registered invariant (must reproduce) | Kill condition |
|---|---|---|---|---|
| E1 | Confirmatory two-frame run, full MaFaulDa, 44 LOSO folds, 5 seeds, bootstrap CI, Wilcoxon | P1, P2 | HZ+HZ self-fusion gain = 0.000 exactly; covariate (speed) shuffle collapses conditioned AUC to the ignore baseline | Fused gain CI includes 0 |
| E2 | HZ+REV same-family control, same folds | P4 | REV+REV gain = 0 | HZ+REV gains >= HZ+ORD gain (then the result is a generic-ensemble effect; drop the physics claim) |
| E3 | Frame map from healthy sweeps (per band, speed-variance in HZ vs ORD; also REV) | map itself | Map built from speed-shuffled healthy windows is flat (ratio ~1 everywhere) | Map is flat on real data (no band separates the frames) |
| E4 | Injection test: Hz-fixed and order-locked tones at matched SNR into held-out healthy windows, per band, per frame AUC | P5 | Injected order-locked tone at an order band the map calls ORD-clean is detected in ORD at AUC >= 0.9; shuffled-band map rho ~ 0 | rho < 0.3 (map does not predict band-level detectability; C2 dropped) |
| E5 | Fault-level winner prediction: map x footprint, footprint fit on one speed, predicted on others; textbook-numerator variant alongside | P6 | Footprint fit and tested on the same speed reproduces the measured winner 5/5 | Out-of-sample < 3/5 |
| E6 | OASSL-style baselines: angular resampling + DeepSVDD / AE / OCSVM / IF, HZ and ORD inputs | P3 | Our ORD-frame detector and the ORD-input baselines agree on the misalignment blind spot (all <= 0.65) | Order-only baseline >= 0.75 on horizontal misalignment |
| E7 | Fusion on top of speed-conditioned detectors | P7 | Fusing identical conditioned scores gains 0 | None (either outcome is reported) |
| E8 | Mendeley KFUPM (conditional): energy-only and healthy-placebo confound checks, then E1/E3/E5 replicated | P8 | PlaceboAUC (healthy-vs-healthy across sessions) <= 0.6 before any fault result is used | Placebo > 0.6 or energy-only >= full-feature AUC (dataset demoted to confound demo) |
| E9 | MIMII-DG replication with signal-estimated ORD | P9 | Fusion with a random-permutation second frame gains 0 | Fusion loses > 0.02 on any machine type |

## 7. Deliverables

- Paper (HTML + DOCX via `paper-build`, two-column via `TwoColPaper`), venue per section 4b.
- Repo gains: `frame_map.py` (healthy-only map, HZ/ORD/REV), `fuse.py` (validity-preserving percentile
  merge: max-of-percentiles, Bonferroni, harmonic-mean p-merge), `inject.py` (controlled Hz-fixed /
  order-locked anomaly injection), OASSL-style baseline suite, Mendeley loader with confound checks,
  MIMII-DG loader with tacho-less order estimation; all results appended per fold as JSONL, resumable.
- Measurable results: the P1..P9 table with CIs; the per-band map figure (HZ vs ORD healthy speed-variance
  with injected-anomaly AUC overlaid); the per-fault winner table with map predictions beside measurements;
  the Domingues explanation figure (Ottawa, confound-aware).
- Experiment registry entries for every negative path (shuffled map, HZ+REV, textbook-numerator failure).

## 8. Phased plan (agent wall-clock and compute cost; not human-developer time)

| Phase | Content | Where | Agent wall-clock | Compute $ |
|---|---|---|---|---|
| 0 | Smoke tests from section 9 (three scripts on cached `feats.npz`) | local CPU | 30-45 min | 0 |
| 1 | E1 + E2 confirmatory (full MaFaulDa, 5 seeds, 44 folds, CIs, Wilcoxon, HZ+REV, self-fusion invariants) | RunPod (kNN + AE over ~50k windows x 3 frames), via `gpu2runpod` | 60-90 min agent (job ~40 min wall-clock) | ~$3-5 |
| 2 | E3 + E4 + E5 frame map, injection test, out-of-sample winner prediction | local CPU (map, injection) + one RunPod job for the injected-anomaly detector sweep | 90-120 min agent (job ~30 min) | ~$2-3 |
| 3 | E6 OASSL-style baselines (4 detectors x 2 inputs x 44 folds) + E7 fusion-on-conditioned | RunPod | 60-90 min agent (job ~60 min) | ~$5-8 |
| 4 | E8 Mendeley: download, confound checks (energy-only, healthy placebo across acquisitions), then replication if it passes | local CPU for checks; RunPod for replication | 45 min agent for checks; +60 min if it passes | 0 / ~$3 |
| 5 | E9 MIMII-DG replication (tacho-less order estimation + two-frame) | RunPod | 60-90 min agent (job ~45 min) | ~$4-6 |
| 6 | Write-up: figures, P-table, paper-reviewer + bibtest passes, paper-build | local | 2-3 h agent | 0 |
| Total | | | ~8-10 h agent across sessions | ~$20-30 |

External gates: RunPod queue/availability (minutes), Mendeley download (~GB, throttled, give it its own
wall-clock), none human-approval bound.

## 9. Top-3 risks and the cheapest smoke test for each (run BEFORE phase 1)

1. **Risk: the fusion gain is a generic-ensemble effect, not competing equivariance.** Smoke: on the
   existing 8000-fault subsample and current folds, compute HZ+REV and HZ+HZ fusion with the same
   max-of-percentile rule (10 min local). Pass: HZ+REV gain < 0.02 and HZ+HZ = 0. Fail: the paper is a
   fusion-tool paper at best, and the physics framing is withdrawn.
2. **Risk: the healthy map is flat or does not predict band-level detectability (C2 is decorative).**
   Smoke: build the HZ/ORD map from healthy windows of all 44 speeds (per band variance of the healthy
   log-spectrum across speed, 5 min local); inject one Hz-fixed tone and one order-locked tone at three
   bands each (map-HZ-clean, map-ORD-clean, map-both-smeared) at fixed SNR, score with the cached
   detector (15 min local). Pass: detection AUC pattern follows the map in all six cells and the
   shuffled-band map pattern does not. Fail: drop C2, proceed with C1 alone (fallback in section 3).
3. **Risk: misalignment's HZ win is a MaFaulDa artifact (e.g. a session or mounting effect confounded with
   the misalignment recordings), not structural physics.** Smoke: healthy-placebo check restricted to the
   misalignment recording sessions (healthy windows from those sessions vs other healthy sessions, 5 min
   local) and an energy-only HZ detector on misalignment. Pass: placebo AUC <= 0.6 and energy-only AUC
   well below the spectral-shape AUC (0.801). Fail: misalignment is removed from the P2 flip claim and the
   flip rests on imbalance vs bearings vs the remaining structural evidence, which weakens the paper to
   the C1 fallback tier.

First smoke test to run: risk 1 (HZ+REV / HZ+HZ control on the existing subsample), because it costs
10 minutes, needs no new code beyond a frame swap in `exp_twoframe.py`, and its failure invalidates the
framing of everything downstream.

---
## DE-RISK OUTCOME (2026-10-15, run before any confirmatory spend) - CORRECTS THE PROPOSAL

The pre-registered P4 invariant PASSED (physics framing real), but the P1/P7 method test FAILED:
- Single-frame macro-AUC (leakage-free LOSO, speed-conditioned per frame): REV 0.885, HZ 0.757, ORD 0.714.
- Fusions (max of held-out-healthy percentiles): HZ+ORD 0.849 (+0.092 over its best MEMBER), HZ+REV 0.884
  (+0.000), HZ+HZ 0.757 (0.000 sanity), REV+ORD 0.841 (-0.044 vs REV alone).
- VERDICT: the two-frame fusion does NOT beat the strongest single representation (REV = the paper's
  existing conditioning). C1 (fusion as a better detector) is KILLED. The earlier "+0.091 lead" only beat
  fusion's own weak members (HZ, ORD), never REV.

WHAT SURVIVES (the keeper): the MECHANISM / frame-map (C2). Validated, striking, physics-predicted:
order tracking is BLIND to structural faults (horizontal misalignment: ORD 0.50 = chance vs HZ 0.80);
bearing/imbalance favor order. No single invariant representation sees all fault families. This EXPLAINS
why REMOVE (order tracking) is the worst treatment and explains Domingues PHM 2024's unexplained result.

REVISED DIRECTION: fold the per-fault frame-blindness map into the EXISTING covariate-treatment paper as
the mechanism (upgrades "order tracking is worst empirically" -> "...because it is provably blind to
structural faults, physics-predicted, winner-flip validated"). Pragmatic recommendation stays REV
(revolution-normalized conditioned). Do NOT ship fusion as a method. Venue read unchanged (single clean
benchmark -> PHM Society / IJPHM / ISA Transactions / Measurement; MSSP only with a vetted 2nd rig).
