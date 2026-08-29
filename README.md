# Condition, Don't Remove: Speed-Conditioned Normal Modeling for Unsupervised Fault Detection in Rotating Machinery

Unsupervised (normal-only) anomaly detection for rotating machinery under operating-speed
shift. Instead of *removing* speed (order-tracking / invariance) or treating it as a *discrete*
domain label, we **condition** the normal-only model on the continuous tachometer speed
covariate. On MaFaulDa (accelerometers + microphone + tachometer, five fault types, 12&ndash;60 Hz)
conditioning raises pooled ROC-AUC from 0.955 to 0.989 at matched speed and from 0.897 to 0.937
on held-out speeds, wins on every fault type, is backbone-agnostic (PCA and autoencoder), holds
across accelerometer and microphone channels, and generalizes to the MIMII-DG acoustic benchmark.

## Read the paper

- **Web (GitHub Pages):** https://apartsinprojects.github.io/EngineKnock/
- **Word:** [`docs/EngineKnock_speed_conditional.docx`](docs/EngineKnock_speed_conditional.docx)
- **Source HTML:** [`paper/index.html`](paper/index.html)

## Repository

- `paper/`, `docs/` &mdash; the paper (HTML source + figures + DOCX; `docs/` serves the Pages site).
- `scripts/` &mdash; all experiments (`exp001`&ndash;`exp010`): ingest, baselines, the speed-conditional
  method, controls, multi-fault / backbone / multi-channel validation, and the MIMII-DG breadth test.
- `experiments/` &mdash; per-experiment registry (hypothesis, setup, numbers, conclusion) and `INDEX.md`.
- `results/` &mdash; result JSONs and figures.
- `PROJECT_LOG.md`, `BACKLOG.md`, `PAPER_OUTLINE.md` &mdash; project log and plan.

## Data

The datasets are public but not redistributed here. MaFaulDa:
`https://www02.smt.ufrj.br/~offshore/mfs/`. MIMII-DG (DCASE 2022 Task 2):
Zenodo `10.5281/zenodo.6529888`. Scripts read the dataset zips from `data/` (git-ignored).

## Status

Preprint draft. Author block anonymized.
