# Design: Paper Recertification

## Technical Approach

Recertify (not redesign) the multi-horizon results by fixing ex-ante tercile
calibration to a frozen train+val basis, adding an on-disk notebook integrity
guard, pinning report-builder determinism, and documenting a local→Kaggle→local
reproduction chain. Model configs, tensor shapes, and processed Parquet schemas
are untouched. Work splits into local phases (source, notebook text, tests, docs)
and one external Kaggle-gated phase (heavy compute). Strict TDD: tests first.

## Architecture Decisions

### D1 — Shared frozen-threshold module

| Option | Tradeoff | Decision |
|--------|----------|----------|
| Duplicate threshold logic in each builder | Two drift-prone p33/p66 sites (current bug) | Reject |
| Helper in `build_exante_volatility` | correlation already imports it, but couples analysis to that entrypoint | Reject |
| New `src/evaluation/exante_terciles.py`, both builders import | Single source of truth, unit-testable, no parquet | **Choose** |

Module exposes `TercileThresholds(p33, p66, calib_split, calib_n)`,
`compute_frozen_thresholds(calib_values) -> TercileThresholds`, and
`assign_terciles(values, thr) -> codes{0,1,2}`. Thresholds are computed per
`(corridor, horizon)` from the train+val ex-ante sigma distribution (ex-ante
sigma is horizon-invariant but the valid window set is not, so calibrate per
cell), frozen, then applied to test. `compute_stratification`
(build_exante_volatility.py:288) and `compute_exante_terciles`
(build_exante_correlation.py:78) both consume `assign_terciles`; neither
recomputes percentiles from test.

**NaN calibration contract (mandatory)**: ex-ante sigma arrays contain NaN by
construction — `materialize_direction` initializes `ex_ante_std_2d` to `np.nan`
and only overwrites cells with `len(valid) >= 2`
(build_exante_volatility.py:178-189); the existing test-path guard at line 298
confirms NaNs are present. `np.percentile` propagates NaN, so
`compute_frozen_thresholds` MUST drop NaN entries (`values[~np.isnan(values)]`)
before computing p33/p66. `calib_n` records the post-filter count. Skipping this
silently yields NaN thresholds that collapse every test sample into one tercile
bucket.

### D2 — Materialize the calibration split

`materialize_corridor` hardcodes the test split. Add a `splits` argument
(default `("test",)`) so the same code path materializes train+val ex-ante sigma
for calibration. **Rationale**: reuses the validated materialization; avoids a
parallel implementation that could diverge. No leakage — thresholds see only
train+val. **Interface contract**: the calibration sigma array passed to
`compute_frozen_thresholds` is NaN-contaminated by construction (see D1); the
helper drops NaN before percentile computation, so callers pass the raw
materialized array without pre-filtering.

### D3 — On-disk notebook integrity guard

| Option | Tradeoff | Decision |
|--------|----------|----------|
| Keep only in-memory cell test | Guards builder source, not the shipped `.ipynb` | Insufficient |
| Extend guard to scan generated `.ipynb` on disk (all 6 families, nbformat) | Catches stale/unregenerated artifacts uploaded to Kaggle | **Choose** |

New `tests/test_notebook_integrity_guard.py` reads every `.ipynb` under
`notebooks/{11,12,13,17,18,19}_*/**` and asserts no cell contains `non_train`,
`winsorize_train_p99(train_df)`, or `pl.concat([df_winsor, non_train])`. Skips
gracefully (pytest skip) when a family's notebooks are absent so fresh checkouts
pass; the regeneration workflow runs builders first, then this guard fails-closed
on any surviving pattern. The existing in-memory contract test is retained
(guards the builder); this adds the artifact-level guard.

### D4 — Deterministic report builders

| Option | Tradeoff | Decision |
|--------|----------|----------|
| Documented tolerance (round on write) | Not byte-identical; defeats clean git diffs | Reject |
| Sorted reductions everywhere | Rewrite every agg; still leaves numpy thread order | Reject |
| Pin `POLARS_MAX_THREADS=1` in report scripts | Byte-identical reruns; negligible cost on aggregation-only scripts | **Choose** |

Float noise (12th–16th digit) comes from thread-order summation in
`group_by().agg(mean/sqrt)` (paired_audit.py:209-218). Set
`os.environ.setdefault("POLARS_MAX_THREADS", "1")` at the top of each report
script **before the first transitive polars import** (ordering gotcha — a
post-import call has no effect; Polars reads the var once at import).
For `build_exante_correlation.py` the pin MUST precede the module-scope
`from src.build_exante_volatility import ...` at line 40, because that import
transitively pulls in polars (build_exante_volatility.py:18) **before** the
script's own redundant local `import polars` inside `main()`. Pinning only
against that local import would silently no-op determinism for
`exante_correlation_multihorizon.csv`. Applies to `build_paired_audit`,
`build_exante_volatility`, `build_exante_correlation`, and the curve/table
builders. These are aggregation scripts (single-thread feasible); training stays
on Kaggle, so the thread-pin runtime impact is immaterial.

**Calibration compute cost (accepted)**: D2 adds a second full materialization
pass over train+val per corridor×horizon. train+val is ~6x test row volume and
`materialize_direction` is an unvectorized Python loop, so this is the dominant
new local wall-clock cost — materially larger than test-only materialization,
NOT covered by the "negligible" thread-pin claim above. Explicitly accepted: it
runs on the laptop as a one-off recertification step (source/CSV generation, not
training), remains bounded by the existing single-corridor loop, and is the
correct price for leakage-free frozen thresholds. If it proves too slow,
materialize train+val sigma once and reuse across horizons for the same corridor
(the sigma values are window-set dependent per horizon, so this is a follow-up
optimization, not a blocker).

### D5 — Kaggle re-run boundary + manifest

Local produces `.ipynb` text and passing tests. Operator uploads/re-runs the 6
kernel families with frozen configs, versions each kernel, downloads residual
and results CSVs into `docs/resultados/`. Local report builders then regenerate
CSVs/figures and rewrite `documento-resultados.md` (paired metrics canonical,
DL-vs-XGBoost claim conditional). Kernel version pins recorded in
`docs/dataset-manifest.md` (new recertification section) for traceability.

## Data Flow

    train+val sigma ─calibrate→ TercileThresholds(frozen)
                                      │ applied to
    test sigma ───────────────────────┴→ terciles → volatility CSV + correlation CSV
                                                          │
    Kaggle residuals ─→ report builders (1-thread) ─→ CSVs → figures → documento

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/evaluation/exante_terciles.py` | Create | Frozen-threshold helper (D1) |
| `src/build_exante_volatility.py` | Modify | Use frozen thresholds; `splits` arg (D2); pin threads (D4); emit threshold cols |
| `src/build_exante_correlation.py` | Modify | Consume shared assign; pin threads before line-40 transitive import; plumb `TercileThresholds` through `build_csv_row` (replace in-cell p33/p66 with `assign_terciles`) |
| `src/build_paired_audit.py`, `build_exante_curve.py`, `build_volatility_table.py`, `build_volatility_curve.py` | Modify | Pin `POLARS_MAX_THREADS=1` |
| `tests/evaluation/test_exante_terciles.py` | Create | Unit tests for frozen thresholds |
| `tests/test_notebook_integrity_guard.py` | Create | On-disk `.ipynb` scan (D3) |
| `notebooks/{11,12,13,17,18,19}_*/**` | Regenerate | From corrected builders |
| `README.md`, `docs/dataset-manifest.md` | Modify | Reproduction section + kernel pins (D5) |
| `docs/resultados/documento-resultados.md` | Modify | Post-Kaggle recertified rewrite |

## Interfaces / Contracts

**Data-contract changes** (rules.design):
- `exante_volatility_multihorizon.csv`: ADD `p33_threshold`, `p66_threshold`,
  `calib_split`, `calib_n`; tercile `share`/`delta_mae` values shift. Consumer
  `build_exante_curve.py` (→ `volatilidad-exante.png`) reads existing columns
  and ignores the additions — compatible.
- `exante_correlation_multihorizon.csv`: `frac_highexante_*` and `lift_high`
  values shift under frozen thresholds; column set unchanged.
- Residual CSVs, Parquet schemas, tensor shapes, model configs: **unchanged**.

## Testing Strategy

| Layer | What | Approach |
|-------|------|----------|
| Unit | Frozen thresholds, tercile assignment, no test leakage | pytest on `exante_terciles` with synthetic arrays |
| Unit | NaN-contaminated calibration input yields finite thresholds (required) | Pass an array with NaN entries to `compute_frozen_thresholds`; assert p33/p66 finite and `calib_n` == post-filter count |
| Integration | Both builders' threshold paths agree | Assert `build_exante_volatility` and `build_exante_correlation` produce identical `(p33, p66)` for the same fixture corridor×horizon (guards cross-process, separate-CSV divergence) |
| Contract | Builders pass full split frame (existing) | Retain in-memory cell test |
| Integrity | No `non_train` in generated `.ipynb` | nbformat scan, 6 families |
| Determinism | Byte-identical CSV across reruns | Run builder twice on fixture; assert equal bytes |

## Migration / Rollout

No data migration. Phased per proposal: Phases 2-4 local and independently
revertible (<400 lines each); Phase 5 gated on the external Kaggle re-run.

## Open Questions

- [ ] NaN calibration filtering is a hard correctness contract (D1), not
  optional — enforced by the required NaN unit test in the Testing Strategy.
  Skipping it silently produces NaN thresholds. Threshold columns added to the
  volatility CSV are additive and backward-compatible for the figure builder.
