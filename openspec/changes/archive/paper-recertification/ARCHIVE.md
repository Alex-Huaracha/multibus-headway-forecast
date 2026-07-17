# Archive Report — paper-recertification

**Archived:** 2026-07-17 · **Status:** complete (all tasks checked)

## What this change did

Recertified the paper's deep-learning results by re-running every DL Kaggle kernel with a
corrected pipeline (p99 winsorization applied to **all** splits, active atypical-day feature,
loader `day`/`date` fix, frozen input-hash gate), then regenerated the full local analysis and
rewrote `docs/resultados/documento-resultados.md` from the fresh residuals.

## Capabilities promoted to `openspec/specs/`

- `exante-volatility-calibration`
- `notebook-generation-integrity`
- `post-kaggle-regeneration`
- `reproducibility-manifest`

## Outcome

- Core thesis intact: DL beats persistence at h≥3 (53/54 cells significant), LSTM beats XGBoost
  in all 8 E2+E59 cells, no sign flips. The advantage **narrowed** once the test was correctly
  winsorized (E2 h10 −1.87→−1.57, E59 −1.37→−1.09, E4 −1.74→−1.42) — reported as-is.
- Baselines (kernels 10/16, incl. XGBoost) were **not** re-run: their harness winsorization was
  already correct (full-split, unchanged since May) over the same unregenerated headways, so a
  re-run is a no-op; the ~0.3 gap in `paired_vs_reported_audit.csv` is a sample-set difference
  (DL drops cold-start window rows), not staleness.
- Document independently verified: **182/182** numeric claims re-derived from the CSVs match
  (3 minor rounding typos found and fixed).

## Completing commits

- `10774b1` fix winsorization + hash gate in NB14/NB15 (scope extension)
- `6a9bb4a` fresh NB14/NB15 CSVs
- `170f4aa` consolidate fresh outputs to canonical paths
- `d3df5c6` regenerate 10.2 tables/figures + 10.1 fail-fast test
- `7e69bd7` rewrite documento-resultados.md (10.3)
