# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a paper (target: IJACSA) on forecasting the full headway vector
of urban bus corridors from real SIT Arequipa GPS data, comparing deep learning
(LSTM / SpatialConvLSTM / SpatialTransformer) against classical and persistence
baselines. The deliverable is the paper's results, not a product.

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
uv run kaggle kernels push -p notebooks/11_lstm_multihorizon/h3/   # launch a Kaggle run
```

The Kaggle CLI is a project dependency and **must be invoked as `uv run kaggle ...`**,
not a global `kaggle`. Credentials live in `~/.kaggle/access_token` (chmod 600) —
not `kaggle.json`, which is what this machine actually has.

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

`src/evaluation/` is the analysis layer: `splits.py` (temporal split +
winsorization contract), `metrics.py`, `significance.py` (paired DM/Wilcoxon),
`paired_audit.py` (canonical DL-vs-persistence over identical samples),
`degradation.py`, `volatility.py`, `exante_terciles.py` (frozen tercile calibration).
`src/baselines/` holds persistence/statistical/fitted (XGBoost) baselines.

## Experiment map

- **Notebook families**: `11/12/13` = LSTM / SpatialConvLSTM / SpatialTransformer on
  corridors E2+E59; `17/18/19` = the same three architectures on E4. `10`/`16` build
  the frozen baseline/data inputs. `01`–`09`, `14`, `15` are legacy/single-horizon and
  **out of scope** for the current recertification.
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
  fail closed before training if bytes don't match; `atypical_days.csv` is a **required**
  input and the atypical-day feature must be active (a set that parses empty raises,
  rather than silently zeroing the flag). See `tests/test_notebook_input_gate.py`.
  Kaggle `kernel_sources` must include `alexhuaracha/02-eda-corridors` so the CSV mounts.
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
- Corridors in scope: companies **2, 4, 59** (referred to as E2/E4/E59). The
  proposal also names company 58, but E58 never entered the pipeline — no
  processed parquet, no results, no builder references it. Treat any mention of
  four corridors as stale.
- Processed data is **Parquet only**; no CSV in the internal pipeline. Raw/processed
  data is gitignored and lives in Kaggle Datasets (pinned in `docs/dataset-manifest.md`).
