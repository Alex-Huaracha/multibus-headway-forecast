# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a paper (target: IJACSA) on forecasting the full headway vector
of urban bus corridors from real SIT Arequipa GPS data. The published line compares
LSTM against XGBoost and persistence; SpatialConvLSTM and SpatialTransformer belong
to the frozen generation and are out of scope for it. The deliverable is the paper's
results, not a product.

Paper prose is governed by the `redaccion-paper` skill
(`.claude/skills/redaccion-paper/`), not by this file.

## Environment & commands

Python and deps are managed with `uv` (Python pinned to 3.12 in `.python-version`).
**Everything runs through `uv run` — do not call `python`, `pytest`, `jupyter`, or
`kaggle` directly.**

```bash
uv sync                                        # reproduce env from lockfile
uv run pytest -q                               # full test suite (~880 tests)
uv run pytest tests/test_paired_audit.py       # one file
uv run pytest tests/test_train.py::test_name   # one test
uv run python src/build_notebook_11.py         # regenerate a notebook from its builder
uv run --env-file .env kaggle kernels push -p notebooks/11_lstm_multihorizon/h3/   # launch a Kaggle run
```

### The Kaggle CLI

It is a project dependency, so **never a global `kaggle`** — and the credentials
in `.env` must be loaded, or authentication fails. `uv run` does **not** read
`.env` on its own (verified; the `--no-env-file` flag implies otherwise, but the
default is off). Two equivalent ways:

```bash
uv run --env-file .env kaggle ...     # per call
$env:UV_ENV_FILE = ".env"             # once per shell (PowerShell); then plain `uv run kaggle ...`
```

Bare `uv run kaggle ...` appears throughout `README.md` and
`docs/correr-kaggle.md`; those examples assume `UV_ENV_FILE` is exported.

Credentials are **project-local, not machine-local**. `.env` (gitignored, see
`.env.example`) carries two variables:

- `KAGGLE_API_TOKEN` — from kaggle.com/settings → API → Create New Token. The
  token authenticates on its own; **there is no username to configure**, the
  client derives it by introspecting the token. `KAGGLE_USERNAME`/`KAGGLE_KEY` is
  the superseded API-key route Kaggle no longer issues.
- `KAGGLE_CONFIG_DIR` → `<repo>/.kaggle-local`. Required even though nothing is
  read from it: importing the kaggle module runs `os.makedirs()` on this path and
  would otherwise create `~/.kaggle`.

So deleting `.env` and `.kaggle-local/` removes every credential trace from the
machine — that is the point of the setup, and it must stay that way.

⚠️ **Never run `kaggle auth login`.** Its OAuth flow writes
`~/.kaggle/credentials.json` through a hardcoded `expanduser()` that
`KAGGLE_CONFIG_DIR` does not redirect, which breaks the guarantee above.

⚠️ **Running the full suite rewrites generated notebooks in place.** The
`tests/test_build_notebook_*.py` tests execute the builders, which write the
`.ipynb` files under `notebooks/`. After `uv run pytest`, run `git status` and
revert unintended notebook changes with `git checkout -- notebooks/`.

## The core architecture: builders → Kaggle → analysis

The single most important thing to understand: **the notebooks in `notebooks/` are
generated artifacts, never hand-edited.** Each `notebooks/<family>/.../*.ipynb` is
emitted by a `src/build_notebook_*.py` builder (shared helpers in
`src/notebook_utils.py`). To change notebook logic, edit the builder and regenerate —
editing an `.ipynb` directly will be silently overwritten on the next build, and the
Kaggle kernel keeps running stale logic until the notebook is regenerated AND pushed.

### Kaggle run gotchas (some fixes are web-only, user-owned)

- **New `kernel_sources` must be attached once via the web "Add Input".** A CLI
  `push` does NOT reliably attach a notebook-output source that was never attached
  before — Kaggle lists it transiently then drops it (verify with
  `kaggle kernels pull -m`). Sources already attached historically (e.g.
  `04-preprocessing`, `10-baselines-multi-horizonte`) persist across pushes; a
  brand-new one like `02-eda-corridors` (which provides the hash-pinned
  `atypical_days.csv`) needs a one-time web "Add Input" on the kernel, after which
  CLI pushes preserve it. When mounted, it lands at
  `/kaggle/input/notebooks/<owner>/<slug>/`.
- **CUDA `no kernel image is available for execution on the device`**
  (`cudaErrorNoKernelImageForDevice`) is a Kaggle GPU/environment mismatch, not a
  code bug — the notebook uses Kaggle's stock torch and requests `GPU_T4X2`. **The
  user resolves this from the web** (environment/GPU settings); do not try to fix it
  via CLI or by editing the builder.

The end-to-end loop:

1. **Build** — `src/build_notebook_*.py` emits the `.ipynb` + `kernel-metadata.json`.
2. **Run on Kaggle** — `uv run kaggle kernels push` uploads a new version; training
   runs on Kaggle GPU (source data lives in Kaggle Datasets, not Git).
3. **Download** — outputs (per-sample residuals, `*_results_*.csv`, logs) are pulled
   back to `docs/resultados/residuos-multihorizon/` and `docs/resultados/csv-multihorizon/`.
   Exact paths and the full runbook are in `docs/correr-kaggle.md`.
4. **Analyze locally** — `src/evaluation/` and the `src/build_*.py` report scripts
   consume the downloaded residuals to produce tables/figures and
   `docs/resultados/documento-resultados.md`.

## File map

Phase-by-phase walkthrough with `file:line` anchors: `docs/proceso/fases.md`.

### `src/preprocessing/` — raw GPS → headway parquet (pure transforms, no I/O)

| File | What it holds |
|---|---|
| `config.py` | Every threshold as frozen dataclass fields + per-company overrides. `centerline_strategy_for` picks `two-pass` (E2/E59) vs `single` (E4). |
| `projection.py` | `attach_observed_speed` (speed derived from displacement, provider's `velocidad` unused); `project_to_centerline` / `project_per_direction` → `s` (arc length) + `lateral_m`, drops pings > 300 m off-axis. |
| `corridor.py` | Fits the centerline **from the data**: PCA on moving pings (≥ 10 km/h), 50 bins, per-bin median, tail trim, smoothing. |
| `direction.py` | `infer_direction` = `sign(rolling_mean(ds, 5))` → `direction ∈ {+1,−1,0}`. |
| `trips.py` | `assign_trip_ids` (cuts on > 30 min gap, direction reversal, terminal dwell); `build_snapshots` = the **60 s** grid. |
| `headways.py` | `compute_pairs` (consecutive bus pairs) and `compute_headways_c2` (**the headway**, by trailing crossing). |
| `pipeline.py` | ⚠️ **Not the orchestrator.** Partial re-implementation; nothing in `src/` calls it outside tests. The code that runs is inlined in `build_notebook_04.py:210-236`. |

### `src/data/` — headway parquet → tensors

| File | What it holds |
|---|---|
| `normalization.py` | `compute_normalization_stats` + `apply_zscore` → creates `delta_t_min_z`. Stats are per `(empresaid, direction)`; **train-only is the caller's contract, not enforced here**. |
| `windowing.py` | `DEFAULT_T_IN = 12`; `compute_max_N` = train p99 of `(n_buses − 1)`. |
| `context_features.py` | Legacy 5-feature context encoder (includes `atypical_flag`). |
| `sample_index.py` | Canonical sample population. Identity anchored on a **timestamp**; enforces strict minute contiguity. |
| `contiguous_dataset.py` | Live loader. 4 causal context features; **raises on `atypical_flag`**. `materialize_arrays` is the path the Kaggle notebooks take. |
| `dataset.py` | Legacy loader, anchored on row positions. Retained only for notebooks 12/13/18/19. |

### `src/models/` and `src/train.py`

| File | What it holds |
|---|---|
| `models/lstm.py` | `HeadwayLSTM` (flat `[headways ‖ context]`) + `masked_mse_loss`, the only loss used. |
| `models/spatial_conv_lstm.py` | `SpatialConvLSTM` — `Conv1d` kernel 3 across the bus axis, then LSTM. |
| `models/spatial_transformer.py` | `SpatialTransformer` — multi-head self-attention across vector positions, then LSTM. |
| `train.py` | Training loop, early stopping, `set_seed`, checkpointing. Forward-signature dispatch is duck-typed on `model.spatial`; class dispatch in `grid_search` is nhead-first. |

### `src/baselines/`

| File | What it holds |
|---|---|
| `statistical.py` | B0 train mean, **B1 persistence** (`forward_fill().shift(horizon)` per slot), B2 causal moving average, B3 SES α=0.3, B4 hourly historical average. |
| `fitted.py` | B5 XGBoost — 12 lags + calendar, 24-config seeded search selected on validation. |
| `contiguous_features.py` | B5 features read off the contiguous grid: no forward-fill, no positional shift, no atypical flag. |
| `harness.py` | `run_corridor` = split → winsorize → all baselines → test-only metrics. |
| `paired_export.py` | Per-sample XGBoost export with `pair_rank` in the key. |

### `src/evaluation/` — analysis layer

| File | What it answers |
|---|---|
| `splits.py` | Temporal split by fixed calendar dates + winsorization contract. Defines the 3 expanding-window origins (`r1`, `r2`, `main`). |
| `metrics.py` | `mae` / `rmse` primitives. Raises rather than returning NaN on all-null input. |
| `paired_audit.py` | Do aggregate metrics survive re-derivation over identical samples — and does the verdict's sign flip? |
| `significance.py` | Diebold–Mariano with HAC, Wilcoxon, cross-cell sign test. |
| `significance_clustered.py` | DM with HLN correction and cluster-robust variance on service day (effective n = days, not rows). |
| `vector_metrics.py` | What a scalar metric cannot see: positional error profile, regularity (CV), bunching detection. Bunching = headway < 0.5 × its own vector's mean; min vector length 3; `trivial_f1` is the floor. |
| `volatility.py` | Retrospective volatility strata — descriptive only, **no p-values by design** (the regime is persistence's own error). |
| `exante_volatility.py`, `exante_terciles.py` | Testable ex-ante strata: input-window dispersion, terciles frozen on train+val. |
| `degradation.py` | Metric decay across horizons. |
| `multiseed.py` | Mean ± Student-t CI across the 5 seeds. |
| `residual_export.py` | Canonical residual key `(corridor, direction, horizon, split, start_ts, target_ts, pair_rank)` + `assert_key_is_unique`. |
| `xgb_paired.py` | XGBoost-vs-DL joins, population framing (`multiplicity_matched` vs `distinct_target`). |

### Analysis builders worth knowing

| Builder | What it produces |
|---|---|
| `build_sample_index.py` | The sample-population manifest and its SHA-256 digests — the gate every training kernel validates against. |
| `build_contiguous_significance.py` | The published paired verdicts (LSTM / XGBoost / persistence). Refuses to report over a partial corpus. |
| `build_contiguous_vector_metrics.py` | Positional profile + regularity + bunching detection. |
| `build_detection_calibrated.py` | Out-of-sample operating point: threshold fitted on origin `r2`, scored on `main`, objective MCC (F1 degenerates). |
| `build_router*.py` | A regime-conditional switching **policy**, not a model. `build_router_temporal.py` is the honest one — the uniform split put near-twin overlapping windows on both sides. |
| `build_rolling_origin_*.py`, `build_multiseed_table.py` | External validity: does the finding hold at other origins / other seeds? |
| `build_contiguity_audit.py`, `build_contiguous_winsorization_sensitivity.py`, `build_mi_recheck.py`, `build_centerline_sweep.py` | Threats-to-validity probes. |
| `build_*_figures.py`, `build_*_curve.py` | Paper figures. Built from committed CSVs, never raw residuals, so a figure cannot contradict its table. |

## Experiment map

- **Two generations of experiments coexist. Do not mix them.**
  - *Frozen*: `11/12/13` = LSTM / SpatialConvLSTM / SpatialTransformer on corridors
    E2+E59; `17/18/19` = the same three architectures on E4; `20` = paired XGBoost.
    Residual schema has **no** `pair_rank` and no timestamp.
  - *Recertified ("contiguous")*: `21` = LSTM over all three rolling origins,
    `22` = XGBoost. This is the published line, and its only competitors are
    LSTM / XGBoost / persistence. Full residual key, causal 4-feature context.
- `10`/`16` build the frozen baseline/data inputs (`16` also **builds** E4's headway
  parquet). `01`–`09`, `14`, `15` are legacy/single-horizon and **out of scope** for
  the current recertification — but note `03` is what selected the headway
  formulation (four competing definitions, C.2 won), so it is history, not dead code.
- **Horizons**: each family has `h1/h3/h5/h10` subfolders (direct multi-horizon).
- Kaggle special case: `12-spatialconvlstm-multihorizon-h10` uses the id `…-h10b`
  (original `-h10` is corrupt on Kaggle). The local metadata already points to `h10b`;
  do not "fix" it back.

## Methodological contracts (do not regress these)

These are enforced by tests — breaking them silently invalidates the paper's results:

- **Winsorization**: the p99 threshold of `delta_t_min` is computed on **train only**
  and applied to **all splits** (train/val/test). Builders must pass the full
  split-tagged frame to `winsorize_train_p99` (`src/evaluation/splits.py`). The old
  bug applied clipping to train only, leaving val/test raw — guarded by
  `tests/test_preprocessing_winsorization_contract.py` and
  `tests/test_notebook_integrity_guard.py`.
- **Frozen input-hash gate**: notebooks pin the SHA-256 of every training input and
  fail closed before training if bytes don't match. See `tests/test_notebook_input_gate.py`.
  - In the *frozen* generation `atypical_days.csv` is a **required** input and the
    atypical-day feature must be active (a set that parses empty raises, rather than
    silently zeroing the flag). Kaggle `kernel_sources` must include
    `alexhuaracha/02-eda-corridors` so the CSV mounts.
  - In the *recertified* generation `atypical_days.csv` is **deliberately excluded**
    from the gate and `atypical_flag` is **forbidden as a feature** — the loader
    raises (`src/data/contiguous_dataset.py:142`, `:236`). It is a whole-day
    aggregate whose threshold was fitted across all 152 days including test, so
    classifying a day needs that day's full record count: leakage by design, removed
    rather than recalibrated. Do not "restore" it.
- **Shared-population gate** (recertified only): the kernel recomputes the sample
  index and its SHA-256 and aborts with `SHARED-POPULATION GATE FAILED` before
  touching the GPU if it disagrees with `sample_index_manifest.csv`.
- **Ex-ante tercile calibration**: volatility terciles are frozen on train+val and
  applied to test — never calibrated on test (`src/evaluation/exante_terciles.py`).
- **Deterministic report builds**: report/analysis builders set
  `POLARS_MAX_THREADS=1` (via `os.environ.setdefault`) before importing polars, for
  byte-identical outputs (`tests/test_report_builder_determinism.py`).
- **Canonical comparison**: any strong "A beats B" claim (DL vs persistence) must trace
  to the paired audit over identical samples, not aggregate metrics.

## Data conventions

- **Composite key is always `(empresaid, unidadid)`** — `unidadid` is reused across
  companies (34 of 150 appear in 3+); never key on `unidadid` alone.
- Corridors in scope: companies **2, 4, 59** (referred to as E2/E4/E59). Company 58
  *is* referenced — `build_notebook_01.py:385-401` has
  `SELECTED_EMPRESAS = [2, 4, 58, 59]`, so E58 survives the corridor filter — but it
  **never gets processed**: preprocessing and modelling run with `[2, 59]` and `[4]`,
  and there is no E58 parquet, no residual, no result. Treat any mention of four
  corridors as stale.
- Processed data is **Parquet only**; no CSV in the internal pipeline. Raw/processed
  data is gitignored and lives in Kaggle Datasets (pinned in `docs/dataset-manifest.md`).
