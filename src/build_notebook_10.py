"""Generate notebooks/10_baselines_multihorizon/10_baselines_multihorizon.ipynb for Kaggle.

10 — Baselines multi-horizonte (auto-generado por build_notebook_10.py)

Evalúa B0–B4 a horizontes 1/3/5/10 min sobre los corredores E2 y E59.
B1/B2/B3 son horizon-aware (implementados en Ola 3 de Fase 6.5).
B0/B4 son horizon-agnósticos (su predicción no depende del horizonte).

Inline-embed pattern (mirror of build_notebook_06.py):
  - Read each src/evaluation/*.py and src/baselines/*.py source file via
    Path.read_text(), strip relative imports, and inject as a code cell.
  - This ensures the notebook is always a faithful flat copy of the modules
    at generation time.  Tests run against the modules directly so the
    notebook never diverges.
  - Stable cell IDs (cell-10-N) prevent git flutter on re-runs (AC-NB10-2).

Output: notebooks/10_baselines_multihorizon/10_baselines_multihorizon.ipynb
Kaggle kernel: alexhuaracha/10-baselines-multi-horizonte
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
# Add repo root to sys.path so that src.notebook_utils is importable
# regardless of invocation directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_utils import _strip_relative_imports  # noqa: E402

SRC_EVAL = ROOT / "src" / "evaluation"
SRC_BASE = ROOT / "src" / "baselines"
OUT = (
    ROOT
    / "notebooks"
    / "10_baselines_multihorizon"
    / "10_baselines_multihorizon.ipynb"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Kaggle kernel metadata — written alongside the notebook.
# ---------------------------------------------------------------------------

_KERNEL_META_BASE = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": ["alexhuaracha/04-preprocessing"],
    "competition_sources": [],
}

# ---------------------------------------------------------------------------
# Cell builder helpers — every cell gets an explicit stable ID.
# ---------------------------------------------------------------------------

_cell_counter = 0
nb = nbf.v4.new_notebook()
cells: list = []


def _next_id() -> str:
    global _cell_counter
    _cell_counter += 1
    return f"cell-10-{_cell_counter}"


def md(text: str, cell_id: str | None = None) -> None:
    cell = nbf.v4.new_markdown_cell(text.strip())
    cell["id"] = cell_id or _next_id()
    cells.append(cell)


def code(src: str, cell_id: str | None = None) -> None:
    cell = nbf.v4.new_code_cell(src.rstrip())
    cell["id"] = cell_id or _next_id()
    cells.append(cell)


def embed_module(rel_path: str, header_md: str, cell_id_md: str, cell_id_code: str) -> None:
    """Embed a source file (relative to ROOT/src/) as a markdown + code cell pair."""
    md(header_md, cell_id=cell_id_md)
    raw = (ROOT / "src" / rel_path).read_text(encoding="utf-8")
    code(_strip_relative_imports(raw), cell_id=cell_id_code)


# ---------------------------------------------------------------------------
# Cell 1: Title (cell-10-title)
# ---------------------------------------------------------------------------

md(
    """
# 10 — Baselines multi-horizonte  (auto-generado por build_notebook_10.py)

Evalúa baselines clásicos (B0 media global, B1 persistencia naive,
B2 media móvil w∈{5,10,15}, B3 suavizado exponencial simple α=0.3,
B4 promedio histórico por hora del día) **y un baseline ajustado**
(B5_XGB, gradient boosting sobre 12 lags + features de calendario/slot)
sobre los corredores **E2** y **E59** a horizontes **h ∈ {1, 3, 5, 10} minutos**.

B1/B2/B3/B5_XGB son horizon-aware. B0/B4 son horizon-agnósticos: su predicción
no depende del horizonte. El CSV de salida incluye columna `horizon`.
""",
    cell_id="cell-10-title",
)

# ---------------------------------------------------------------------------
# Cell 2: Setup — locate parquets and output dir (cell-10-setup)
# ---------------------------------------------------------------------------

code(
    """
import polars as pl
import numpy as np
from pathlib import Path

# Locate headways parquets for E2 and E59 under /kaggle/input or local dir.
def _find_parquet(empresa_id: int) -> Path:
    name = f"headways_E{empresa_id}.parquet"
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(name))
        if candidates:
            return candidates[0]
    candidates = list(Path(".").rglob(name))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"{name} not found. Expected at /kaggle/input/**/{name}"
    )

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_OUT = OUTPUT_DIR / "baselines_results_multih.csv"

print(f"Output dir: {OUTPUT_DIR}")
""",
    cell_id="cell-10-setup",
)

# ---------------------------------------------------------------------------
# Module embed cells — dependency order: splits → metrics → statistical → harness
# ---------------------------------------------------------------------------

embed_module(
    "evaluation/splits.py",
    """## Module: evaluation/splits

Temporal split helper (`split_temporal`) and train-only p99 winsorization
(`winsorize_train_p99`).  Split date ranges are locked in spec §3.
""",
    cell_id_md="cell-10-embed-splits-md",
    cell_id_code="cell-10-embed-splits",
)

embed_module(
    "evaluation/metrics.py",
    """## Module: evaluation/metrics

`mae` and `rmse` in minutes.  Both accept polars Series or numpy arrays.
Null/NaN rows are dropped before aggregation.  MAPE is explicitly excluded
(spec B3-NO-MAPE).
""",
    cell_id_md="cell-10-embed-metrics-md",
    cell_id_code="cell-10-embed-metrics",
)

embed_module(
    "baselines/statistical.py",
    """## Module: baselines/statistical

B0 global mean, B1 naive persistence (horizon-aware), B2 moving average
(w∈{5,10,15}, horizon-aware), B3 simple exponential smoothing (α=0.3,
horizon-aware), B4 historical average per hour.
All operate per slot `(empresaid, direction, pair_rank)`.
""",
    cell_id_md="cell-10-embed-statistical-md",
    cell_id_code="cell-10-embed-statistical",
)

embed_module(
    "baselines/fitted.py",
    """## Module: baselines/fitted

`predict_b5_xgb` — fitted gradient-boosted baseline (B5_XGB). Sees the same
12-lag window as the DL models (lag_1 == B1 persistence) plus calendar/slot
features. Answers the "where is a fitted/ML baseline?" reviewer reflex.
Must be embedded BEFORE harness, which calls it.
""",
    cell_id_md="cell-10-embed-fitted-md",
    cell_id_code="cell-10-embed-fitted",
)

embed_module(
    "baselines/harness.py",
    """## Module: baselines/harness

`evaluate_corridor` composes split → winsorize → all baselines (B0-B4 plus the
fitted B5_XGB when include_fitted=True, the default) → metrics per
(direction × baseline), returning a tidy 48-row long DataFrame.
Accepts `horizon` to thread h to B1/B2/B3/B5_XGB.
""",
    cell_id_md="cell-10-embed-harness-md",
    cell_id_code="cell-10-embed-harness",
)

# ---------------------------------------------------------------------------
# Cell: Load data — E2 and E59 ONLY (cell-10-load)
# ---------------------------------------------------------------------------

md(
    """## Cargar datos — E2 y E59

Lee los parquets generados por el notebook 04 (two-pass pipeline).
""",
    cell_id="cell-10-load-md",
)

code(
    """
# `empresaid` is implicit in the filename — inject it as a literal column
# so it matches the baselines contract (slot key requires it).
hw_e2  = pl.read_parquet(_find_parquet(2)).with_columns(pl.lit(2,  dtype=pl.Int64).alias("empresaid"))
hw_e59 = pl.read_parquet(_find_parquet(59)).with_columns(pl.lit(59, dtype=pl.Int64).alias("empresaid"))

print(f"E2:  {hw_e2.height:,} rows, {hw_e2.width} cols")
print(f"E59: {hw_e59.height:,} rows, {hw_e59.width} cols")
""",
    cell_id="cell-10-load",
)

# ---------------------------------------------------------------------------
# Cell: Sanity check (cell-10-sanity)
# ---------------------------------------------------------------------------

md(
    """## Sanidad — cobertura por (empresa, dirección)

Fracción de `delta_t_min` no-nulo y conteo de filas por dirección.
Permite detectar baja cobertura en dir=+1 (riesgo R-DIR1-COVERAGE).
""",
    cell_id="cell-10-sanity-md",
)

code(
    """
for label, hw in [("E2", hw_e2), ("E59", hw_e59)]:
    print(f"\\n=== {label} ===")
    summary = (
        hw.with_columns(pl.col("t").dt.date().alias("day"))
          .group_by(["empresaid", "direction"])
          .agg([
              pl.len().alias("n_rows"),
              (pl.col("delta_t_min").is_not_null().sum() / pl.len()).alias("non_null_frac"),
          ])
          .sort(["empresaid", "direction"])
    )
    print(summary)
""",
    cell_id="cell-10-sanity",
)

# ---------------------------------------------------------------------------
# Cell: Run harness — multi-horizon loop (cell-10-run-harness)
# ---------------------------------------------------------------------------

md(
    """## Ejecutar harness — loop multi-horizonte

Llama a `evaluate_corridor` para E2 y E59 a cada horizonte h ∈ {1, 3, 5, 10}
y concatena todos los resultados agregando la columna `horizon`.
""",
    cell_id="cell-10-run-harness-md",
)

code(
    """
HORIZONS = [1, 3, 5, 10]
frames = []
for h in HORIZONS:
    r_e2  = evaluate_corridor(hw_e2,  "E2",  horizon=h)
    r_e59 = evaluate_corridor(hw_e59, "E59", horizon=h)
    frames.append(
        pl.concat([r_e2, r_e59]).with_columns(pl.lit(h, dtype=pl.Int64).alias("horizon"))
    )
results = pl.concat(frames)
print(f"Total rows: {results.height}  (expected 384 = 4 horizons x 2 corridors x 48)")
print(results.head(10))
""",
    cell_id="cell-10-run-harness",
)

# ---------------------------------------------------------------------------
# Cell: Write CSV (cell-10-write-csv)
# ---------------------------------------------------------------------------

md(
    """## Escribir CSV — baselines_results_multih.csv

Escribe la tabla long-form a `/kaggle/working/baselines_results_multih.csv`.
Columnas: `corridor, direction, baseline, metric, value, horizon`.
""",
    cell_id="cell-10-write-csv-md",
)

code(
    """
results.write_csv(CSV_OUT)
print(f"CSV written to: {CSV_OUT}")
print(f"Rows: {results.height}  Columns: {results.columns}")
""",
    cell_id="cell-10-write-csv",
)

# ---------------------------------------------------------------------------
# Cell: Summary table (cell-10-summary)
# ---------------------------------------------------------------------------

md(
    """## Tabla resumen (wide format)

Pivote ancho para lectura humana: filas = (corridor, horizon, direction, metric),
columnas = baseline.
""",
    cell_id="cell-10-summary-md",
)

code(
    """
wide = results.pivot(
    on="baseline",
    index=["corridor", "horizon", "direction", "metric"],
    values="value",
)
print(wide.sort(["corridor", "horizon", "direction", "metric"]))
""",
    cell_id="cell-10-summary",
)

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Notebook written: {OUT}  ({len(cells)} cells)")

# Write kernel-metadata.json.
kernel_meta = {
    "id": "alexhuaracha/10-baselines-multi-horizonte",
    "title": "10 — Baselines multi-horizonte",
    "code_file": "10_baselines_multihorizon.ipynb",
    **_KERNEL_META_BASE,
}
meta_path = OUT.parent / "kernel-metadata.json"
meta_path.write_text(json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8")
print(f"Kernel metadata written: {meta_path}")
