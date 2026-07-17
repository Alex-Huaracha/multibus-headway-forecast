# Exploration: Phase 5 E2/h=3 Alignment Blocker

## Current State

Phase 5 stopped fail-closed before writing either Phase 5 CSV. The E2/h=3 LSTM
residual artifact has 599,117 E2 rows, the same count as reconstruction, but
the reconstructed target and persistence values disagree. This is an evidence
investigation only: no result, source, notebook, task, data, or configuration
artifact was changed.

### Provenance traced

1. `src/build_exante_volatility.py:353-395` loads the E2 processed Parquet,
   assigns temporal splits, computes the train-p99 ceiling, applies that ceiling
   to the full split frame, derives train-only z-score statistics, materializes
   test windows, and fail-closes before loading DL predictions for stratification.
2. Window identity is implicit: `make_window_index` orders by
   `(empresaid, direction, pair_rank, start_idx)` and each slot's timestamps are
   sorted. Materialization emits direction `-1` then `+1`, flattens each window
   in C order, and retains `target_mask & persistence_mask`.
3. The residual CSV contains only `corridor,direction,horizon,y_true,y_pred_dl,
   y_pred_persist`; it has no timestamp, `pair_rank`, window start, source hash,
   split fingerprint, preprocessing version, or kernel version. Thus it cannot
   independently establish row identity.
4. The generated NB11 residual cell uses the same direction order, paired mask,
   direct horizon semantics (`target` at `T_IN + HORIZON - 1`), and persistence
   as the final input timestep. The local reconstruction replicates those rules.
5. The current transformation contract computes train p99 from train rows but
   clips train, validation, and test values. Z-score statistics remain train-only;
   reconstruction denormalizes with the same `mean` and `std + Z_EPS`. No
   rounding or ex-ante tercile calculation occurs before the alignment gate.

## Evidence

| Evidence | Observation | Interpretation |
|---|---|---|
| Builder gate | `n_csv=n_rec=599117`; max target and persistence delta = `1.532078` | A row-count gate cannot detect provenance mismatch; both value vectors fail. |
| Independent read-only reconstruction | Target mismatches >=0.01: 6,043; persistence: 5,246; delta vectors are not equal (`corr=0.02157`) | Equal maxima do **not** indicate a shared global row/key shift. |
| First divergent values | CSV target `28.967319` reconstructed as `28.467922`; later CSV persistence `28.967319` reconstructed as `28.467922` | Exact train-p99 clipping signature; the same temporal observation can appear as a target or a final input in different windows. |
| Max divergent values | CSV target/persistence `30.0`, reconstructed `28.467922`; delta `-1.532078` | The reported equal maxima are explained by the common clipping ceiling, not equal per-row deltas. |
| Current Parquet transformation | train p99 = `28.4679230336`; raw E2 test max = `30.0`; clipped test max = `28.4679230336` | Directly matches reconstruction's cap and the mismatch magnitude. |
| Residual artifact | CSV max target/persistence = `30.0`; E2 rows above current threshold: target 6,080, persistence 5,288; SHA-256 `29edd7e4...ecc4c6b4`; local mtime 2026-07-05 | Residuals retain test values that the documented current pipeline clips. |
| Git history | `8ca6f9e` (2026-07-11) changed DL builders and reconstruction from train-only clipping to full-split clipping; its message explicitly says Kaggle kernels must be rebuilt/rerun before DL metrics reflect the fix. The residual export predates it (`a9c3155`, 2026-06-16). | Strong artifact-version mismatch evidence. |
| Tests | 24 focused tests passed: calibration, frozen thresholds, and full-split winsorization. The PyTorch/NumPy ABI warning was emitted but tests completed. | Code contract is covered on fixtures; it does not certify provenance of an existing external residual CSV. |

## Affected Areas

- `src/build_exante_volatility.py` — reconstructs E2/h=3 targets and persistence and implements the hard gate.
- `src/build_notebook_11.py` / `notebooks/11_lstm_multihorizon/h3/11_lstm_h3.ipynb` — source/kernel path that produces the residual artifact.
- `src/evaluation/splits.py` — train-derived/full-frame winsorization contract responsible for the observed cap.
- `src/data/windowing.py` — implicit row ordering and direct-horizon semantics.
- `docs/resultados/residuos-multihorizon/11-lstm/h3/lstm_residuals_h3.csv` — stale/unprovenanced residual artifact; not modified.
- `tests/test_exante_builder_calibration.py`, `tests/test_preprocessing_winsorization_contract.py` — fixture coverage; no real-artifact provenance check exists.

## Hypotheses and Falsification

1. **Artifact-version mismatch caused by pre-fix train-only clipping** — **high confidence (0.97)**.
   - Supported by the exact ceiling signature, mismatch magnitude, dated artifact,
     and `8ca6f9e`'s explicit rerun requirement.
   - Falsified if a versioned execution using the full-frame clipping contract and
     the same E2 source Parquet emits these un-clipped residual values, or if
     keyed comparisons show divergent windows where neither target nor persistence
     crosses the ceiling.

2. **Shared temporal/key shift** — **low confidence (0.05)**.
   - Equal maxima alone suggested this possibility, but the target/persistence
     delta vectors are not equal and their correlation is only 0.02157.
   - Falsified conclusively by keyed residual output `(corridor, direction,
     pair_rank, input_end_t, target_t)` matching the current reconstruction after
     clipping. It remains untestable from the present six-column CSV.

3. **Reconstruction defect in split, horizon, corridor, index, or ordering** —
   **low confidence (0.10)**.
   - The reconstruction mirrors the notebook's split, direction, direct-horizon,
     window, mask, and flattening contracts; row counts match exactly.
   - Falsified by a corrected, keyed kernel export that matches the CSV but not a
     reconstruction using identical keys and values. Confirmed only if a keyed,
     same-version comparison isolates a non-clipping discrepancy.

4. **PyTorch/NumPy ABI warning caused the values** — **very low confidence
   (0.01)**.
   - The builder reached a NumPy/Polars reconstruction gate; test-only
     reconstruction reproduced the numerical signature despite the warning.
   - Falsified by running the same read-only reconstruction in a compatible
     environment and obtaining a materially different clipping signature. It is
     still an environment risk for future kernel execution.

## Approaches

1. **Preserve the failed gate and establish a versioned corrected residual lineage**
   - Obtain a Kaggle execution from the regenerated NB11 h=3 notebook using the
     full-frame train-p99 contract, frozen dataset identity, and recorded commit/
     kernel version. It must export residual rows with immutable window keys and
     retain the unmodified old artifact for comparison.
   - Pros: restores the documented experimental provenance; independently
     reproducible; tests the actual scientific outcome rather than hiding it.
   - Cons: external compute; model outputs and conclusions may change.
   - Effort: High.

2. **Read-only keyed provenance audit before any rerun**
   - Produce a diagnostic-only keyed reconstruction and compare it to a keyed
     residual export from the original kernel/version, if recoverable.
   - Pros: definitively separates clipping from row identity/version causes.
   - Cons: impossible with the present six-column residual CSV; requires original
     kernel artifacts or a diagnostic rerun.
   - Effort: Medium/High.

3. **Force the present gate to pass by clipping, filtering, reordering, joining,
   tolerance expansion, or changing the comparison** — **REJECTED**.
   - It would manipulate or conceal the observed mismatch and would not restore
     the provenance of the reported DL residuals. It is scientifically invalid.

## Recommendation

Do not modify the gate or consume the current residual CSV for Phase 5. The
evidence supports an artifact-version mismatch caused by the known
full-split-winsorization correction, rather than a shared temporal/key shift.
The correction prerequisite is a versioned, independently reproducible Kaggle
rerun (or recoverable keyed export from that corrected lineage) before any CSV
regeneration decision. The rerun must preserve the failed artifact, capture the
dataset checksum, notebook/source commit, kernel version, preprocessing ceiling,
and per-row window keys; then the same fail-closed comparison must pass without
any data manipulation.

## Risks

- The current six-column residual schema cannot prove row identity, so a purely
  local value comparison cannot rule out every key-level defect.
- A corrected rerun may change DL predictions and paper conclusions; those values
  must be reported as obtained, not reconciled to the old CSV.
- The PyTorch/NumPy ABI warning is non-causal for this gate but must be resolved
  or version-pinned for a reliable external rerun.
- Existing Phase 5 output remains absent; proceeding to the correlation builder
  or checking Phase 5 tasks would violate the stop-on-failure protocol.

## Ready for Proposal

No. Evidence supports the provenance diagnosis, but no implementation should be
proposed until the versioned corrected residual lineage (including keyed audit
evidence) is available.
