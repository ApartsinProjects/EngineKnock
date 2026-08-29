# Scout: public datasets for unsupervised engine knock / NVH anomaly detection

Date: 2026-08-29. Method: web-researcher agent, verified primary sources.

## Headline
No public dataset of RAW combustion-knock signals (in-cylinder pressure / knock-sensor
waveform) with unsupervised framing exists. Real knock/NVH studies keep test-bench data
private. Usable public data is either (B) a diesel fault *feature* set, (C) DCASE/MIMII/
ToyADMOS machine-sound anomaly benchmarks (the de-facto unsupervised acoustic baselines),
or (D) rotating-machinery vibration proxies (CWRU, Paderborn, MFPT, MaFaulDa).

## A. True combustion / knock -- nothing downloadable
- Fine-Grained Engine Fault Sound Event Detection Using Multimodal Signals (mic + tri-axial
  accel in engine bay, 10 fine-grained faults incl. engine knock/rattle/belt-squeal),
  arXiv:2403.11037. Closest conceptual match; NO confirmed public download. -> email authors.
- Dual-fuel block-vibration knock (Ismail et al., Results in Engineering 2021, PMC8720894): private.
- SI heavy-duty knock-sensor ML (Measurement 2023, S0263224123014240): private.

## B. IC-engine mechanical fault (real engine, feature-level)
- DEFault -- Diesel Engine Faults Features. IEEE DataPort DOI 10.21227/3q67-j268
  (paywalled) + free Kaggle mirror sujaykapadnis/diesel-engine-faults-features-dataset-3500.
  In-cylinder pressure + torsional vibration -> 84-dim FEATURE vectors (NOT raw waveforms).
  MWM Acteon 6.12 diesel @ 2500 rpm. Labeled: normal(250) + 3 fault types = 3500 samples.
  Limitation for us: no raw signal, no cam/crank, so our own cycle-separation can't be re-run.

## C. Machine-sound acoustic anomaly benchmarks (unsupervised standard)
All 16 kHz, 10 s clips, normal-only training (true unsupervised), anomalies by physical damage.
- MIMII -- Zenodo 3384388. Machine types: valve, pump, fan, slide rail; 8-mic array; CC BY-SA 4.0.
  Variants: MIMII-DUE (4740355), MIMII-DG (6529888).
- ToyADMOS / ToyADMOS2 -- github.com/YumaKoizumi/ToyADMOS-dataset (ToyCar/ToyTrain/ToyConveyor).
- DCASE Task 2 (unsupervised ASD): 2020 valve/pump/fan/slider+Toy (Zenodo 3678171 ...);
  2023 first-shot adds gearbox+bearing (arXiv:2305.07828); 2024/2025 add motors/robotic arm.

## D. Rotating-machinery vibration proxies (NOT combustion engines)
- CWRU bearing -- engineering.case.edu/bearingdatacenter. Normal + inner/ball/outer, 3 sizes.
- Paderborn (PU) -- 64 kHz, real+artificial bearing damage.
- MFPT -- motor test-rig bearing vibration.
- MaFaulDa (UFRJ Machinery Fault Database) -- 50 kHz*, 1951 samples, 6 conditions
  (normal, imbalance, horiz/vert misalignment, underhang/overhang bearing faults);
  8 channels incl. tachometer + microphone + 2 tri-axial accelerometers.
  http://www02.smt.ufrj.br/~offshore/mfs/page_01.html
  UNIQUE VALUE for us: only public set with accel + mic + tacho together -> enables both
  order-tracking (tacho) and cross-modal (accel<->mic) variants of the proposed method.

## Decision for this project
- Primary dev/testbed: MaFaulDa (tacho+accel+mic).
- Acoustic-only generalization: MIMII valve + one DCASE Task2 machine type.
- Real-engine sanity: DEFault (features only, confirmatory row).
- Aspirational: arXiv:2403.11037 (request from authors).

*Confirm MaFaulDa exact fs and channel order at ingest.
