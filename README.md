# multibus-headway-forecast

Code for a study on forecasting the headway vector of urban bus corridors from
SIT Arequipa GPS data. This repository holds the full pipeline: preprocessing,
training, evaluation, and the generation of every table and figure in the paper.

- Manuscript: [`docs/paper/paper.md`](docs/paper/paper.md)
- Data: [`…-raw`](https://www.kaggle.com/datasets/alexhuaracha/multibus-headway-forecast-raw)
  (raw GPS) and
  [`…-clean`](https://www.kaggle.com/datasets/alexhuaracha/multibus-headway-forecast-clean)
  (cleaned GPS + headways), on Kaggle under CC BY 4.0

---

## Requirements

- **[`uv`](https://docs.astral.sh/uv/) installed.** It manages Python and the
  dependencies. Everything runs through `uv run`; you do not need to install
  Python or create a virtual environment yourself.

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh                              # macOS / Linux
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
  ```

- **A Kaggle account with an API token**, only for the steps that download
  results or launch training runs. The local analysis steps do not need one.
- **A GPU**, only if you retrain. Reproducing the tables and figures does not
  require one.

Python 3.12 is pinned in `.python-version` and `uv` installs it for you.

---

## Installation

```bash
uv sync          # creates .venv with Python 3.12 and the locked dependencies
```

For the Kaggle steps, copy `.env.example` to `.env` and paste your token
(kaggle.com → Settings → API → Create New Token):

```bash
cp .env.example .env
```

> **`uv run` does not read `.env` on its own.** Export it once per shell, or pass
> it on every call. Without this, every `kaggle` command fails to authenticate.
>
> ```bash
> export UV_ENV_FILE=.env                  # bash / zsh
> $env:UV_ENV_FILE = ".env"                # PowerShell
>
> uv run --env-file .env kaggle ...        # alternative, per call
> ```

---

## Reproducing the results

### 1. Download the residuals from Kaggle

They are the ground truth of the analysis: one row per sample, carrying each
method's error.

```bash
uv run kaggle kernels output alexhuaracha/21-lstm-contiguous-h3 \
  -p docs/resultados/residuos-multihorizon/21-lstm-contiguous/
uv run kaggle kernels output alexhuaracha/22-xgb-contiguous \
  -p docs/resultados/residuos-multihorizon/22-xgb-contiguous/
```

Repeat per horizon (`h1`, `h3`, `h5`, `h10`) and per group (`-e4-` for the third
corridor). The full runbook is in
[`docs/correr-kaggle.md`](docs/correr-kaggle.md).

### 2. Rebuild the tables

No GPU and no retraining. These read the residuals and write to
`docs/resultados/csv-multihorizon/`.

```bash
uv run python -m src.build_sample_index                          # freezes the shared sample population
uv run python -m src.build_contiguous_significance               # paired tests clustered by service day
uv run python -m src.build_contiguous_paired_audit               # framing-bias audit
uv run python -m src.build_contiguous_vector_metrics             # metrics over the vector
uv run python -m src.build_detection_calibrated                  # out-of-sample recalibrated threshold
uv run python -m src.build_contiguous_volatility                 # volatility strata (slow)
uv run python -m src.build_contiguous_winsorization_sensitivity  # robustness to the p99 cap
```

Several **fail closed** when the sample population does not match the frozen one.
That is intentional: it prevents comparing methods over different sets.

### 3. Rebuild the paper's figures and tables

These are built from the CSVs of the previous step, never from the residuals, so
a figure cannot disagree with the table next to it.

```bash
uv run python -m src.build_contiguous_figures    # result figures
uv run python -m src.build_schematic_figures     # method schematics
uv run python -m src.build_paper_tables          # tables as Markdown
```

---

## Optional

### Rebuilding `raw_gps.parquet` from the original CSVs

The published starting point is `raw_gps.parquet`, already available on Kaggle.
These scripts are only needed to rebuild it from the original export, and they
take the paths as arguments because the CSVs live outside the repository:

```bash
uv run python src/inspect_raw.py <csv> [<csv> ...]   # per-file summary
uv run python src/merge_raw.py  <csv> [<csv> ...]    # → data/raw/raw_gps.parquet
```

### Relaunching a training run on Kaggle

Requires a token and a GPU. The notebooks under `notebooks/` are emitted by the
builders, so regenerate them first and push afterwards.

```bash
uv run python src/build_notebook_21_lstm_contiguous.py            # regenerates the 8 notebooks
uv run kaggle kernels push -p notebooks/21_lstm_contiguous/h3/    # uploads a version and runs it
```

Kaggle allows **2 concurrent GPU sessions**: launch them two at a time. A kernel
failing with `no kernel image is available for execution on the device` is a
Kaggle environment mismatch (P100 vs T4×2), fixed from the web UI rather than the
CLI.

---

## Layout

```
src/preprocessing/   raw GPS → headways
src/data/            headways → tensors
src/models/          LSTM and the two spatial architectures it was contrasted against
src/baselines/       persistence, hourly historical average, XGBoost
src/evaluation/      metrics, significance tests, audits
src/build_*.py       notebook, table and figure generators
notebooks/           artifacts emitted by the builders
docs/                manuscript, results and process notes
data/                local data (not versioned)
```

A phase-by-phase walkthrough with `file:line` anchors is in
[`docs/proceso/fases.md`](docs/proceso/fases.md).
