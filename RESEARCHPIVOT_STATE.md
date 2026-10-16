# EngineKnock — ResearchPivot State Brief

## 1. Problem + current claim (why it is only incremental)
Unsupervised (normal-only) anomaly detection for rotating machinery must cope with a machine's
operating speed changing over time. Speed is measured (every drive reports rpm). We frame the choice of
what to do with that measured speed as a three-way decision and show, on one public benchmark, that
**conditioning** the healthy model on speed beats **ignoring** speed, which beats **removing** speed by
order tracking (the field-default fix, which is the WORST). The result is correct, rigorously verified,
and mechanistically explained — but it rests on a **single clean benchmark**, and the taxonomy itself
(ignore/condition/remove; operating-condition compensation) is not new to the condition-monitoring / SHM
community. Fable's read: mid-venue accept, MSSP major-revision. The ceiling is the single benchmark.

## 2. Evidence so far (headline numbers)
MaFaulDa (public: 50 kHz vibration + microphone + tachometer, rotor rig, 1 healthy recording per speed
across 12-60 Hz, 5 fault types). Leakage-free leave-one-speed-out ROC-AUC, cross-fitted conditional mean,
cluster-bootstrap CI over recordings, paired Wilcoxon:
- Conditioning 0.956 [0.941, 0.970]  >  Ignore 0.901 [0.874, 0.927]  >  Remove/order-tracking 0.820 [0.778, 0.864].
- adjust>ignore p=1e-11; adjust>remove p=1e-13.
- Energy-normalized (pure spectral SHAPE, level removed): condition 0.916 > ignore 0.787 > remove 0.647 (ordering sharpens).
- Score-normalization control (rescale the ignore score per speed, feature space untouched): 0.895, = ignore (p=0.70); unstable operating point (TPR across-band CoV 0.20) vs conditioning uniform (CoV 0.05).
- Backbone-agnostic (PCA and autoencoder), works on accel and mic; MIMII-DG discrete-covariate breadth.

## 3. Key insights (non-obvious)
- The field-default remedy for speed (order tracking) is ACTIVELY HARMFUL for anomaly detection under
  speed shift, not merely unhelpful. Mechanism: structural resonances are fixed in Hz; order-domain
  resampling makes them migrate across order bins as speed changes, smearing exactly the energy that
  separates healthy from fault.
- The gain must come from the FEATURE model (conditioning), not from rescaling the score margin: a
  per-speed score-normalization shortcut fails.
- Whether to condition is itself decidable from healthy data alone (variance the covariate explains).

## 4. Tried and rejected (DO NOT recycle)
- Order-synchronous / speed-invariance thesis: REFUTED (order tracking is worst).
- Cross-modal accel<->mic residual detector: dropped (marginal vs simpler baselines).
- A clean SECOND real dataset to replicate the ordering: THREE attempts, ALL CONFOUNDED / unusable:
  (a) Kaggle car-engine audio: healthy-vs-healthy null 0.82; (b) Mendeley engine journal-bearing (shaker
  test, not running engine): within-noise; (c) University of Ottawa variable-speed bearing set: raw AUC
  ~0.95 but = energy confound (Inner-race faults 6x healthy energy) + recording-session fingerprints
  (energy-normalized null ~1.0). No public dataset exists where healthy and fault share recording sessions.
- Engine / gas-genset APPLICATION framing: dropped (no public engine-combustion waveform data anywhere;
  motivation-evidence gap = reject pattern).
- Detection-latency / early-warning experiment: had an in-sample-threshold bug and served the dropped
  engine framing.
- PIVOT CYCLE 1 (ChatGPT): "Healthy-only benchmark admissibility / placebo test" on Ottawa. KILLED
  (validated). Ottawa healthy placebo (train PCA on trial A of a speed-trajectory, score trials B vs C,
  same trajectory so speed matched, L2-normalized + RPM-conditioned, max(AUC,1-AUC), 12 tests):
  PlaceboAUC median = 0.59, 0/12 >= 0.70, ROBUST across PCA rank k=4..32 (not a degeneracy artifact).
  KEY REFINEMENT: Ottawa's earlier energy-norm "null ~1.0" was CROSS-TRAJECTORY (different speed ranges)
  = a speed/trajectory effect + energy, NOT a pure acquisition-session fingerprint. Within a matched
  trajectory + RPM conditioning, healthy trials are NOT separable. So Ottawa's 3 same-setup trials are
  too similar to expose a session confound; the placebo admissibility idea has no independent
  same-condition healthy sessions to bite on in our assets.

## 5. Assets available (what a cheap experiment can touch TODAY, no new data)
- MaFaulDa fully extracted + cached: results/feats.npz (healthy 521 windows/49 speeds; ~22k fault windows
  across 5 faults) with 3 representations per window: HZ (physical-Hz spectrum), REV (revolution-normalized),
  ORD (order-tracked), plus per-window speed and recording id. Also multi-channel (accel x2 + mic) caches.
- MIMII-DG features (5 machines) cached.
- The full ignore/condition/remove/score-norm/support-aware pipeline, vectorized, verified.
- Confounded external datasets are ON DISK (Ottawa 60 .mat with encoder speed; Kaggle audio; journal-bearing)
  — usable as NEGATIVE / confound-demonstration material, not as clean benchmarks.
- Compute: local CPU + RunPod GPU on demand. Cheap smoke = minutes.

## 6. Target
Novel AND impactful AND valuable AND highly publishable, WITHOUT requiring a new clean dataset (the one
thing unattainable). Venue class: strong ML-for-monitoring or a top mechanical-systems journal (MSSP), or
a methods/eval contribution at an ML venue. Any positive lead must pass correctness validation (null
control ~0.5, energy-normalization, covariate-shuffle collapse) before it is believed — three datasets
already fooled the raw numbers.
