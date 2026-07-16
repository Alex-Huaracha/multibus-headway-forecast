# Tasks: Paper Recertification

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~900-1100 hand-written; notebooks/CSVs generated, counted separately |
| 400-line budget risk | High |
| Delivery strategy | direct-commits-to-main (user decision, 2026-07-11) |
| Chain strategy | none — no PRs; each work unit lands as 1-3 conventional commits on main |

Decision needed before apply: none pending — delivery resolved as direct commits to main.
Delivery: direct commits to main per work unit below; keep tests and implementation in the same commit; conventional commits, no AI attribution.
400-line budget risk: mitigated by per-work-unit commits (each unit stays reviewable in isolation).

Scope amendment (2026-07-14): the NB11 residual-provenance audit layer (Phases 11-13 of a prior revision: keyed residual exports, schema-v2 sidecars, receipts, quarantine/promotion gates) was removed as over-engineering. Retained essentials: the frozen input-hash gate + required atypical-day feature (work unit 11 below), the `02-eda-corridors` kernel source, seed-before-construction tests, thread-pins, and the notebook integrity guard.

### Suggested Work Units

| Work unit | Goal | Depends | Commit/review boundary |
|-----------|------|---------|------------------------|
| 1 | `exante_terciles.py` + unit tests | none | 240 lines |
| 2 | Wire thresholds into both builders + integration test | work unit 1 | 190 lines |
| 3 | Thread-pin 4 remaining report builders | none | 70 lines |
| 4 | Notebook integrity guard test | none | 100 lines |
| 5 | Local calibration CSV regeneration (slow) | work units 2, 3 | data |
| 6 | Regenerate 6 notebook families | work units 2-4, 11 | generated |
| 7 | Paired-audit error-path tests | none | 70 lines |
| 8 | Reproducibility docs (README, manifest) | work unit 5 | 100 lines |
| 9 | Kaggle re-run gate (external) | work units 1-8 | 0 lines |
| 10 | Post-Kaggle regeneration + results rewrite | work unit 9 | 250+ lines |
| 11 | Frozen input-hash gate + required atypical feature, 6 families | none | ~300 lines |

Legend: EVC=exante-volatility-calibration, NGI=notebook-generation-integrity, RM=reproducibility-manifest, PKR=post-kaggle-regeneration.

## Phase 1: Frozen Threshold Module (PR1)

- [x] 1.1 RED/GREEN `tests/evaluation/test_exante_terciles.py` + `src/evaluation/exante_terciles.py`: thresholds from train+val only, test extremes ignored [EVC1].
- [x] 1.2 RED/GREEN: NaN-contaminated input yields finite p33/p66, `calib_n` = post-filter count [EVC2].
- [x] 1.3 RED/GREEN: empty/too-small filtered array raises explicit error [EVC2].
- [x] 1.4 RED/GREEN: `assign_terciles` classifies against frozen thresholds; non-finite rows excluded [EVC3].

## Phase 2: Wire Calibration Into Builders (PR2, needs PR1)

- [x] 2.1 RED/GREEN `src/build_exante_volatility.py`: `materialize_corridor` gets `splits` arg (default `("test",)`) for train+val materialization (D2), with real-path regression coverage for the public default `("test",)`.
- [x] 2.2 GREEN: `compute_stratification` uses `assign_terciles`; emits `p33_threshold`, `p66_threshold`, `calib_split`, `calib_n`.
- [x] 2.3 GREEN: `src/build_exante_correlation.py` `compute_exante_terciles` uses shared `assign_terciles`; drop in-cell recompute.
- [x] 2.4 RED/GREEN: integration test — both entrypoints independently materialize and calibrate non-overlapping train+val fixture windows; extreme test-only perturbations leave both public threshold pairs invariant [EVC4].
- [x] 2.5 Pin `POLARS_MAX_THREADS=1` before first polars import in both files (before line-40 import in correlation builder, D4).

## Phase 3: Determinism Thread-Pin, Other Builders (PR3)

- [x] 3.1 RED/GREEN: byte-identical-rerun test; subprocess import probe asserts `POLARS_MAX_THREADS` at the first direct or transitive Polars import for `build_paired_audit.py`, `build_exante_curve.py`, `build_volatility_table.py`, and `build_volatility_curve.py`; pin each builder to `1` [RM4].

## Phase 4: Notebook Integrity Guard (PR4)

- [x] 4.1 RED/GREEN `tests/test_notebook_integrity_guard.py`: nbformat-scan `notebooks/{11,12,13,17,18,19}_*/` for `non_train`, `winsorize_train_p99(train_df)`, and `pl.concat([df_winsor, non_train])`; fail with pattern + path + zero-based cell index; skip absent families [NGI2, NGI3; D3].

## Phase 5: Local Calibration Regeneration (PR5, needs PR2+3)

- [x] 5.1 **Runtime warning (slow, ~6x test-row unvectorized pass)**: `uv run python src/build_exante_volatility.py` then `src/build_exante_correlation.py`; regenerates both ex-ante CSVs. Done 2026-07-16 on fresh recertified residuals. Both builders redirected to `docs/resultados/recertificado/` (RESID_DIR + OUT_DIR); `exante_volatility` 9/9 ALIGNMENT PASS (max|Δ| < 1e-2), DL beats persistence in the high-volatility tercile for all 9 corridor×horizon; `exante_correlation` confirms anti-circularity (Pearson r ≈ 0.22–0.27, r² ≈ 0.05–0.07, lift 1.11–1.29).
- [x] 5.2 `uv run pytest tests/evaluation/test_exante_terciles.py tests/evaluation/test_volatility.py -q`. 16 passed.

## Phase 6: Notebook Regeneration (PR6, needs PR2-4)

- [x] 6.1 `uv run python src/build_notebook_{11,12,13,17,18,19}_*.py` for all 6 families [NGI1].
- [x] 6.2 `uv run pytest tests/test_notebook_integrity_guard.py -q` passes on regenerated set.

## Phase 7: Paired-Audit Error-Path Coverage (PR7)

- [x] 7.1 RED/GREEN `tests/evaluation/test_paired_audit.py`: missing-residual-column and duplicate-group `ValueError` cases against existing `src/evaluation/paired_audit.py` [PKR3]. Done 2026-07-16: `test_rejects_frame_missing_residual_column` (drops `y_pred_persist` → `paired_metrics_table` raises "missing residual columns") + `test_rejects_duplicate_model_corridor_horizon_groups` (two slug-only `lstm_residuals_h3.csv` → `build_paired_metrics` raises "duplicate model/corridor/horizon groups"). 9 passed.

## Phase 8: Reproducibility Docs (PR8, needs PR5)

- [ ] 8.1 `README.md`: reproduction section (env, Kaggle download, local rebuild commands) [RM1].
- [ ] 8.2 `docs/dataset-manifest.md`: recertification section — pins per family, non-versioned artifact inventory [RM2, RM3].
- [ ] 8.3 Document test-clipping sensitivity note + no-clipping plan for Kaggle re-run.

## Phase 9: Kaggle Re-run Gate (PR9, blocking, user-owned)

- [x] 9.1 **BLOCKING**: re-run 6 kernel families on Kaggle, frozen configs, per `docs/dataset-manifest.md`; download fresh residuals. Done 2026-07-15 (24/24 kernels: 6 families x h1/h3/h5/h10). Deviations from original plan, all validated per-log (`Atypical days loaded: 17 dates`, winsorize E2 28.4679 / E59 27.9969 / E4 29.0984, no traceback): (a) fresh residuals landed in `docs/resultados/recertificado/residuos-multihorizon/` (heavy, gitignored) + results CSVs in `docs/resultados/recertificado/csv-multihorizon/` (tracked), NOT the original `docs/resultados/residuos-multihorizon/` path — Phase 10 builders must point at `recertificado/`; (b) atypical CSV mounted via the `alexhuaracha/atypical-days-frozen` dataset (hash-pinned) instead of `02-eda-corridors`, because CLI push does not reliably attach new kernel_sources; (c) loader fix: `load_atypical_days` now reads the frozen CSV's `day` column (was `date`); (d) E4 baselines-name fix: NB16 writes `baselines_E4_results_multih.csv` — the E4 model notebooks now search that name first. All 24 committed to main.

## Phase 10: Post-Kaggle Regeneration (PR10, needs PR9)

- [ ] 10.1 Fail-fast if fresh residuals absent [PKR1].
- [ ] 10.2 Regenerate significance, degradation, volatility, paired-audit CSVs and figures from fresh residuals [PKR1].
- [ ] 10.3 Rewrite `documento-resultados.md`: headline cites `paired_dl_persistence_metrics.csv`; DL-vs-XGBoost claims conditional [PKR2].

## Phase 11: Frozen Input-Hash Gate + Required Atypical Feature (work unit 11, done 2026-07-14)

- [x] 11.1 RED/GREEN `tests/test_notebook_input_gate.py`: every generated DL notebook (6 families x 4 horizons) pins frozen input SHA-256s, resolves inputs via `_resolve_input`, requires `atypical_days.csv` (no silent fallback, empty-set hard fail), and declares `alexhuaracha/02-eda-corridors` in kernel-metadata; NB12 h10 keeps the `h10b` replacement slug (original h10 kernel corrupt on Kaggle, commit e0757b6).
- [x] 11.2 GREEN: all 6 builders replace name-only `_find_parquet`/atypical fallback with the hash-verifying resolver (search by filename under `/kaggle/input` then `.`, accept any copy whose bytes match the frozen hash, fail closed otherwise); NB12 builder emits the `h10b` slug for h=10.
- [x] 11.3 Hashes independently verified: `headways_E2/E59` against local `data/processed` copies; `headways_E4` and `atypical_days.csv` against freshly downloaded pinned kernel outputs (`16-e4-data-baselines`, `02-eda-corridors`).
- [x] 11.4 Removed over-engineered NB11 provenance layer (keyed residual exports, schema-v2 sidecars, receipts, quarantine/promotion gates, `tests/test_nb11_residual_provenance.py`, `specs/nb11-residual-provenance/`): residual CSV schema stays `corridor,direction,horizon,y_true,y_pred_dl,y_pred_persist` for all horizons. Failed-run diagnostics under `diagnostics/` are preserved as audit trail.
