# Post-Kaggle Regeneration Specification

## Purpose

Once the user re-runs the 6 DL kernel families on Kaggle with frozen configs and downloads fresh per-sample residuals, the local pipeline must regenerate every derived artifact from those residuals and rewrite the results document with paired metrics as the canonical comparison. Scenarios below assume downloaded Kaggle outputs already exist locally.

## Requirements

### Requirement: Regeneration Gated On Downloaded Kaggle Outputs

The full regeneration procedure (residuals-derived significance, degradation, volatility retro+ex-ante, paired-audit CSVs, figures, and `documento-resultados.md`) MUST run only after fresh per-sample residual CSVs from the frozen-config Kaggle re-run are present under `docs/resultados/residuos-multihorizon/`, and MUST NOT run against stale pre-recertification residuals.

#### Scenario: Missing residuals fail fast

- GIVEN residual CSVs for the frozen-config re-run are absent
- WHEN the regeneration procedure is invoked
- THEN it fails fast with a clear "missing residuals" error instead of silently using stale CSVs

#### Scenario: Full regeneration on fresh residuals

- GIVEN fresh downloaded residual CSVs for all 6 families
- WHEN the regeneration procedure runs
- THEN significance, volatility, ex-ante, and paired-audit CSVs, and figures, are all recomputed from those residuals

### Requirement: Paired Metrics Canonical In Results Document

`docs/resultados/documento-resultados.md` MUST report paired-audit metrics (paired MAE/RMSE deltas and significance) as the canonical DL-vs-persistence comparison; any DL-vs-XGBoost superiority claim MUST be stated conditionally, not as unconditional fact.

#### Scenario: Headline table cites paired metrics

- GIVEN the regenerated `paired_dl_persistence_metrics.csv`
- WHEN the results document's headline comparison table is rewritten
- THEN it cites values from that paired-metrics CSV

#### Scenario: DL-vs-XGBoost claims are conditional

- GIVEN paired-audit results show a reduced or reversed DL advantage vs. a prior report
- WHEN the results document states any DL-vs-XGBoost superiority claim
- THEN the claim is qualified (e.g. by horizon/corridor/condition), not stated unconditionally

### Requirement: Paired Audit Error-Path Coverage

The paired-audit test suite MUST cover the missing-residual-columns failure mode and the duplicate-residual-file guard in `src/evaluation/paired_audit.py`.

#### Scenario: Missing residual columns raise

- GIVEN a residual frame missing a required column (e.g. `y_pred_persist`)
- WHEN `paired_metrics_table` or `build_paired_metrics` processes it
- THEN it raises `ValueError` naming the missing column(s)

#### Scenario: Duplicate residual groups raise

- GIVEN two residual files that both cover the same `(model, corridor, horizon)` group
- WHEN `build_paired_metrics` runs
- THEN it raises `ValueError` identifying the duplicate group
