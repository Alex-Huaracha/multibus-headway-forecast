# Proposal: Paper Recertification — Ex-ante Calibration, Notebook Integrity, Reproducibility

## Intent

**Research goal**: Close the remaining methodological gaps flagged in `docs/mejoras-resultados.md` (Phases 2-5) so the multi-horizon headway-forecast results are fair, paired, and reproducible before the IJACSA submission. Paired-audit and winsorization fixes already landed in the working tree (baseline, not this change). This change recertifies the experiment; it does not re-tune models.

## Scope

### In Scope
- Fix ex-ante volatility tercile calibration: derive p33/p66 thresholds from train (or train+val) volatility, freeze them, apply to test in `compute_stratification` (`src/build_exante_volatility.py`) and the mirrored `compute_exante_terciles` (`src/build_exante_correlation.py`). TDD: tests first.
- Regenerate the 6 corrected notebook families (11/12/13 multihorizon, 17/18/19 E4) from builders; add an automated guard asserting no train-only winsorization (`non_train`) pattern in generated `.ipynb`.
- Frozen input-hash gate across the 6 families: resolve `headways_E{2,59,4}.parquet` and `atypical_days.csv` by filename + frozen SHA-256 (fail closed before training on missing or altered bytes); declare `alexhuaracha/02-eda-corridors` as kernel source. Discovery (2026-07-14): the atypical-day feature (DL-2) was silently inert (all zeros) in every prior Kaggle run because the CSV never mounted and notebooks fell back gracefully — the user decided to activate it properly in all 6 families for the re-run, keeping frozen winning configurations (no re-tuning).
- Reproducibility docs: README reproduction section + `docs/dataset-manifest.md` (env, Kaggle download/rebuild commands, non-versioned artifacts, dataset/kernel versions).
- Documented test-clipping sensitivity note (winsorizing test ground truth) + no-clipping sensitivity plan for the Kaggle re-run.
- Post-Kaggle regeneration (BLOCKED on external re-run): residuals, significance, degradation, volatility, paired-audit CSVs, figures; rewrite `docs/resultados/documento-resultados.md` with PAIRED metrics as canonical, honestly.

### Out of Scope
- Grid search, architecture changes, re-tuning hyperparameters.
- Older notebooks 05-09, 14, 15 (exploratory/superseded, no corrected builders) — documented as out of scope.
- Local training/evaluation — all heavy compute runs on Kaggle.

## Capabilities

### New Capabilities
- `exante-volatility-calibration`: frozen train+val-derived tercile thresholds applied to test for operational ("live-executable") volatility regimes.
- `notebook-generation-integrity`: generated Kaggle notebooks must match corrected builders; guard blocks stale preprocessing patterns.
- `reproducibility-manifest`: documented path to rebuild tables/figures from pinned Kaggle datasets/kernels.
- `post-kaggle-regeneration`: Kaggle-gated regeneration of residuals, result CSVs, figures, and the results document with paired metrics as the canonical comparison.

### Modified Capabilities
- None (no existing specs).

## Approach

Strict TDD per phase (`uv run pytest`). Isolate threshold computation so both ex-ante modules share one frozen-threshold path. Add a nbformat-based guard test. Local work only edits source, regenerates notebook text, and runs targeted tests; Kaggle re-run is a user-owned external step gating Phase 5.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/build_exante_volatility.py` | Modified | Train-derived frozen tercile thresholds |
| `src/build_exante_correlation.py` | Modified | Same frozen regime assignment |
| `notebooks/{11,12,13,17,18,19}_*/` | Modified | Regenerated from corrected builders |
| `tests/` | New | Threshold + notebook-integrity guards |
| `README.md`, `docs/dataset-manifest.md` | New/Modified | Reproduction path |
| `docs/resultados/documento-resultados.md`, `*_multihorizon.csv`, figures | Modified | Post-Kaggle recertified outputs |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Frozen thresholds shift tercile shares / weaken high-vol finding | Med | Report honestly; thresholds are the correct operational claim |
| Paired deltas shrink, altering conclusions | High | Truthful rewrite; keep DL-vs-XGBoost claim conditional |
| Notebook regeneration leaks unintended diffs | Med | Guard test + scoped diff review |
| Phase 5 blocked on external Kaggle re-run | High | Sequence Phases 2-4 locally; gate Phase 5 |

## Rollback Plan

Each phase is an independent slice under 400 changed lines. Revert per commit: source fixes, notebook regeneration, and docs are separable. If recertified numbers regress the thesis, retain prior `documento-resultados.md` via git history and re-evaluate before publishing.

## Dependencies

- User re-runs the 6 DL kernel families on Kaggle with frozen configs and downloads outputs (blocks Phase 5).
- Existing per-sample residuals under `docs/resultados/residuos-multihorizon/`.

## Success Criteria

- [ ] Ex-ante terciles computed from train/val, frozen, applied to test — verified by tests.
- [ ] 6 notebook families regenerated; guard test blocks train-only winsorization.
- [ ] README + dataset-manifest let a reviewer rebuild tables/figures.
- [ ] Test-clipping sensitivity documented.
- [ ] `documento-resultados.md` uses paired metrics as canonical; claims truthful and conditional.

## Proposal question round

Resolved with the user (2026-07-11):
1. Tercile thresholds: **train+val** volatility distribution (more samples, still no test leakage) — CONFIRMED.
2. Older notebooks 05-09/14/15: **out of scope** (exploratory/superseded) — CONFIRMED.
3. No-clipping sensitivity: **documented as a planned check** in the Kaggle re-run design, not implemented locally this change — CONFIRMED.

Resolved with the user (2026-07-14):
4. Atypical-day feature (silently inert in all prior runs): **activate properly in all 6 families** with required, hash-verified `atypical_days.csv`; re-run keeps frozen winning configurations, no re-tuning — CONFIRMED.
5. NB11 residual-provenance audit layer (keyed exports, sidecars, receipts, promotion gates): **removed as over-engineering**; only the simple input-hash gate is retained — CONFIRMED.
