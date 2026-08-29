# EngineKnock — Backlog (re-ranked on every finding)

## P0 — make-or-break
- [ ] EXP-001: MaFaulDa ingest smoke test. Download a handful of files (normal + 1 fault),
      confirm fs / channel order (tacho, 2x tri-axial accel, mic), estimate rotation from
      tacho, do order-domain resampling on ONE cycle end-to-end. Gate for everything.
- [ ] EXP-002: Establish the plain baseline on MaFaulDa in an unsupervised split
      (train normal-only, test normal+faults): log-mel spectrogram autoencoder recon-error
      AUC + embedding+Mahalanobis. This is the number order-sync must beat.

## P1 — the novel hook
- [ ] EXP-003: Order-synchronous representation. Resample each window to fixed samples-per-
      revolution via tacho; compare order-spectrum AE vs plain-spectrogram AE at equal backbone.
      DEGENERATE INVARIANT: order-resampling off must reproduce EXP-002 exactly.
      TACHO-SHUFFLE control: scrambled angle map must drop AUC toward chance.
- [ ] EXP-004: Cross-modal agreement score (predict accel order-spectrum from mic and vice
      versa; disagreement = anomaly). MaFaulDa only (needs both modalities).

## P2 — generalization / rigor
- [ ] EXP-005: Acoustic-only transfer to MIMII valve + one DCASE Task2 type; estimate the
      fundamental from audio when no tacho.
- [ ] EXP-006: DEFault real-engine confirmatory row (feature-level, limited).
- [ ] Scout published MaFaulDa/MIMII unsupervised AUC baselines before running our own.

## P3 — reach
- [ ] Email arXiv:2403.11037 authors for the multimodal engine-fault set.
- [ ] Revisit localization (angular/spatial) if a suitable multi-location set appears.

## P1 — venue-strengthening (added 2026-08-29, user-requested)
- [ ] EXP-009: Reproduce OASSL (order-adaptive subspace scale learning) as a NAMED
      speed-invariance competitor on MaFaulDa; add to the base/inv/cond table. If OASSL not
      faithfully reproducible from public info, implement the strongest reproducible in-class
      order-tracking-invariance method and cite OASSL. (scouting method + code now.)
- [ ] EXP-010: MIMII-DG breadth. Show the CONDITIONING principle extends to a DISCRETE
      operating attribute: on MIMII-DG (DCASE 2022 DG), condition the normal-only detector on
      the provided attribute/domain label vs unconditional + DCASE baseline. Download 2-3
      machine types; audio log-mel/embedding + AE. Cite official DCASE baselines. (scouting
      download + attribute structure now.)

## Re-rank log
- 2026-08-29: initial ranking.
- 2026-08-29: added EXP-009 (OASSL competitor) + EXP-010 (MIMII-DG breadth) as P1 after paper build.
