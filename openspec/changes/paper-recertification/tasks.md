# Tasks: Paper Recertification

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~900-1100 hand-written; notebooks/CSVs generated, counted separately |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | Stacked PRs to main, PR1-PR10 |
| Delivery strategy | direct-commits-to-main (user decision, 2026-07-11) |
| Chain strategy | none — no PRs; each work unit lands as 1-3 conventional commits on main |

Decision needed before apply: No (resolved)
Delivery: direct commits to main per work unit below; keep tests and implementation in the same commit; conventional commits, no AI attribution.
400-line budget risk: mitigated by per-work-unit commits (each unit stays reviewable in isolation)

### Suggested Work Units

| PR | Goal | Depends | ~Lines |
|----|------|---------|--------|
| 1 | `exante_terciles.py` + unit tests | none | 240 |
| 2 | Wire thresholds into both builders + integration test | PR1 | 190 |
| 3 | Thread-pin 4 remaining report builders | none | 70 |
| 4 | Notebook integrity guard test | none | 100 |
| 5 | Local calibration CSV regeneration (slow) | PR2,3 | data |
| 6 | Regenerate 6 notebook families | PR2-4 | generated |
| 7 | Paired-audit error-path tests | none | 70 |
| 8 | Reproducibility docs (README, manifest) | PR5 | 100 |
| 9 | Kaggle re-run gate (external) | PR1-8 | 0 |
| 10 | Post-Kaggle regeneration + results rewrite | PR9 | 250+ |

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

- [ ] 3.1 RED/GREEN: byte-identical-rerun test; pin `POLARS_MAX_THREADS=1` in `build_paired_audit.py`, `build_exante_curve.py`, `build_volatility_table.py`, `build_volatility_curve.py` [RM4].

## Phase 4: Notebook Integrity Guard (PR4)

- [ ] 4.1 RED/GREEN `tests/test_notebook_integrity_guard.py`: nbformat-scan `notebooks/{11,12,13,17,18,19}_*/` for `non_train` pattern; fails with path+cell index; skips absent families [NGI2, NGI3].

## Phase 5: Local Calibration Regeneration (PR5, needs PR2+3)

- [ ] 5.1 **Runtime warning (slow, ~6x test-row unvectorized pass)**: `uv run python src/build_exante_volatility.py` then `src/build_exante_correlation.py`; regenerates both ex-ante CSVs.
- [ ] 5.2 `uv run pytest tests/evaluation/test_exante_terciles.py tests/evaluation/test_volatility.py -q`.

## Phase 6: Notebook Regeneration (PR6, needs PR2-4)

- [ ] 6.1 `uv run python src/build_notebook_{11,12,13,17,18,19}_*.py` for all 6 families [NGI1].
- [ ] 6.2 `uv run pytest tests/test_notebook_integrity_guard.py -q` passes on regenerated set.

## Phase 7: Paired-Audit Error-Path Coverage (PR7)

- [ ] 7.1 RED/GREEN `tests/evaluation/test_paired_audit.py`: missing-residual-column and duplicate-group `ValueError` cases against existing `src/evaluation/paired_audit.py` [PKR3].

## Phase 8: Reproducibility Docs (PR8, needs PR5)

- [ ] 8.1 `README.md`: reproduction section (env, Kaggle download, local rebuild commands) [RM1].
- [ ] 8.2 `docs/dataset-manifest.md`: recertification section — pins per family, non-versioned artifact inventory [RM2, RM3].
- [ ] 8.3 Document test-clipping sensitivity note + no-clipping plan for Kaggle re-run.

## Phase 9: Kaggle Re-run Gate (PR9, blocking, user-owned)

- [ ] 9.1 **BLOCKING**: re-run 6 kernel families on Kaggle, frozen configs, per `docs/dataset-manifest.md`; download fresh residuals into `docs/resultados/residuos-multihorizon/`. All Phase 10 tasks depend on this.

## Phase 10: Post-Kaggle Regeneration (PR10, needs PR9)

- [ ] 10.1 Fail-fast if fresh residuals absent [PKR1].
- [ ] 10.2 Regenerate significance, degradation, volatility, paired-audit CSVs and figures from fresh residuals [PKR1].
- [ ] 10.3 Rewrite `documento-resultados.md`: headline cites `paired_dl_persistence_metrics.csv`; DL-vs-XGBoost claims conditional [PKR2].
