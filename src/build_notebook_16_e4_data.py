"""Generate notebooks/16_e4_data/16_e4_data.ipynb for Kaggle.

16 — E4 data + baselines  (auto-generado por build_notebook_16_e4_data.py)

E4 (empresaid=4) is the THIRD corridor — a VALIDATION corridor for external
validity. This notebook is a single CPU kernel that does BOTH:

  1. E4 preprocessing  — mirror of build_notebook_04.py, but scoped to
     EMPRESAS = [4]. Reads clean_gps.parquet (which already contains empresa 4)
     and writes headways_E4.parquet (+ cleaned_gps_E4, null_buckets).
  2. E4 baselines      — mirror of build_notebook_10.py, but scoped to E4.
     Loads the headways computed in-notebook and runs evaluate_corridor for
     h ∈ {1, 3, 5, 10}, writing baselines_E4_results_multih.csv.

CRITICAL: the existing notebooks NB04–NB13 are IMMUTABLE experiment artifacts.
E4 is a NEW experiment that REUSES the library code via this new builder — it
does NOT touch any frozen builder or notebook. The single-pass vs two-pass
branch is gated on `centerline_strategy_for(4)` (returns "single") — never
hardcoded here.

Inline-embed pattern (mirror of build_notebook_04.py + build_notebook_10.py):
  - Read each src/preprocessing/*.py, src/evaluation/*.py and src/baselines/*.py
    source file via Path.read_text(), strip relative imports, and inject as a
    code cell. The notebook stays a faithful flat copy of the modules at
    generation time; tests run against the modules directly so it never diverges.
  - Stable cell IDs (cell-16-N) prevent git flutter on re-runs (AC-NB16-2/3).

Output: notebooks/16_e4_data/16_e4_data.ipynb
Kaggle kernel: alexhuaracha/16-e4-data
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

SRC_PREP = ROOT / "src" / "preprocessing"
OUT = ROOT / "notebooks" / "16_e4_data" / "16_e4_data.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Kaggle kernel metadata — written alongside the notebook.
#
# E4 reads the SAME clean dataset NB04 reads (clean_gps.parquet already contains
# empresa 4), so it is a dataset source — NOT a kernel source. This makes NB16
# self-contained: it does not depend on NB04's output kernel.
# ---------------------------------------------------------------------------

_KERNEL_META_BASE = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": ["alexhuaracha/multibus-headway-forecast-clean"],
    "kernel_sources": [],
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
    return f"cell-16-{_cell_counter}"


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
# Cell: Title (cell-16-title)
# ---------------------------------------------------------------------------

md(
    """
# 16 — E4 data + baselines  (auto-generado por build_notebook_16_e4_data.py)

**Corredor de validación E4** (`empresaid=4`) — tercer corredor para validez
externa. Este kernel CPU hace TODO en un solo notebook:

1. **Preprocessing E4** — pipeline de Fase 2 (mirror de NB04) sobre
   `clean_gps.parquet` (que ya contiene la empresa 4), produciendo
   `headways_E4.parquet` (+ `cleaned_gps_E4`, `headway_null_buckets_E4`).
2. **Baselines E4** — B0–B4 + B5_XGB (mirror de NB10) a horizontes
   h ∈ {1, 3, 5, 10}, escribiendo `baselines_E4_results_multih.csv`.

E4 usa la estrategia de centerline `"single"` (gated por
`centerline_strategy_for(4)`, NO hardcodeado) y `has_heading=True`. Los
notebooks NB04–NB13 son artefactos congelados: E4 REUSA la librería vía este
builder nuevo sin tocarlos.
""",
    cell_id="cell-16-title",
)

# ---------------------------------------------------------------------------
# Cell: Setup — locate clean_gps.parquet, output dir, EMPRESAS = [4]
# (cell-16-setup)
# ---------------------------------------------------------------------------

code(
    """
import polars as pl
import numpy as np
from pathlib import Path
import os

# Locate clean_gps.parquet under /kaggle/input (or local working directory).
candidates = list(Path("/kaggle/input").rglob("clean_gps.parquet")) if Path("/kaggle/input").exists() else []
if not candidates:
    candidates = list(Path(".").rglob("clean_gps.parquet"))
if not candidates:
    raise FileNotFoundError("clean_gps.parquet not found. Expected at /kaggle/input/**/clean_gps.parquet")
INPUT = candidates[0]
print(f"Input: {INPUT}")

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_OUT = OUTPUT_DIR / "baselines_E4_results_multih.csv"
print(f"Output dir: {OUTPUT_DIR}")

# E4 ONLY — frozen corridors E2/E59 are NOT touched by this notebook.
EMPRESAS = [4]
""",
    cell_id="cell-16-setup",
)

# ---------------------------------------------------------------------------
# Preprocessing module cells (mirror of build_notebook_04.py, dependency order)
# ---------------------------------------------------------------------------

embed_module(
    "preprocessing/config.py",
    """## Module: preprocessing/config

Parámetros productivos congelados desde `docs/decisiones-headway-fase2.md §3`.
E4 está en `EMPRESA_CONFIG` (empresaid=4, has_heading=True, estrategia "single").
""",
    cell_id_md="cell-16-embed-config-md",
    cell_id_code="cell-16-embed-config",
)

embed_module(
    "preprocessing/corridor.py",
    """## Module: preprocessing/corridor

Construcción del trazado del corredor via PCA + binned median.
""",
    cell_id_md="cell-16-embed-corridor-md",
    cell_id_code="cell-16-embed-corridor",
)

embed_module(
    "preprocessing/projection.py",
    """## Module: preprocessing/projection

Speed observado (`step_m / dt_s`) y proyección arc-length `s`.
Filtra pings off-route con `lateral_m > LATERAL_OFFSET_THRESHOLD_M`.
""",
    cell_id_md="cell-16-embed-projection-md",
    cell_id_code="cell-16-embed-projection",
)

embed_module(
    "preprocessing/direction.py",
    """## Module: preprocessing/direction

Inferencia de sentido ida/vuelta desde `sign(rolling_mean(ds, win=5))`.
E4 reporta `direccion` (has_heading=True), usado como verificación cruzada.
""",
    cell_id_md="cell-16-embed-direction-md",
    cell_id_code="cell-16-embed-direction",
)

embed_module(
    "preprocessing/trips.py",
    """## Module: preprocessing/trips

Segmentación de viajes (gap 30 min / reversal / terminal dwell 5 min)
y grilla de snapshots con alineación minuto-exacta (INV-6).
""",
    cell_id_md="cell-16-embed-trips-md",
    cell_id_code="cell-16-embed-trips",
)

embed_module(
    "preprocessing/headways.py",
    """## Module: preprocessing/headways

C.2 trailing crossing — pure polars+numpy. Para pares sin historial previo
se emite `delta_t_min = null` (NO se descarta). Winsorización en Fase 5.
""",
    cell_id_md="cell-16-embed-headways-md",
    cell_id_code="cell-16-embed-headways",
)

embed_module(
    "preprocessing/pipeline.py",
    """## Module: preprocessing/pipeline

Orquestación gated por `centerline_strategy_for(empresaid)` (R-CFG1).
E4 → "single" (single-pass) PROVISIONALMENTE: sin evidencia multi-filar aún.
""",
    cell_id_md="cell-16-embed-pipeline-md",
    cell_id_code="cell-16-embed-pipeline",
)

# ---------------------------------------------------------------------------
# Cell: Run preprocessing pipeline for E4 (cell-16-run-prep)
# ---------------------------------------------------------------------------

md(
    """## Ejecutar pipeline de preprocessing — E4

Carga `clean_gps.parquet`, aplica todos los módulos en orden de dependencia
para `empresaid=4`, y escribe `headways_E4.parquet` (+ cleaned_gps, null_buckets).

La estrategia (single/two-pass) la decide `centerline_strategy_for(4)` — NO se
hardcodea. E4 devuelve "single" (single-pass).
""",
    cell_id="cell-16-run-prep-md",
)

code(
    """
lf = (
    pl.scan_parquet(INPUT)
    .filter(
        pl.col("empresaid").is_in(EMPRESAS)
        & pl.col("time").is_not_null()
        & pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
        & (pl.col("lat") != 0) & (pl.col("lon") != 0)
    )
    .with_columns(pl.col("time").dt.date().alias("day"))
    .sort(["empresaid", "unidadid", "time"])
)
gps_all = lf.collect(engine="streaming")
print(f"Rows loaded: {gps_all.height:,}")

for empresaid in EMPRESAS:
    print(f"\\n--- Empresa {empresaid} ---")
    sub = gps_all.filter(pl.col("empresaid") == empresaid)

    sub = attach_observed_speed(sub)
    strategy = centerline_strategy_for(empresaid)

    if strategy == "two-pass":
        print(f"  Strategy: two-pass (multi-filar)")
        # Pass-1: single centerline → crude direction labels
        centerline = build_centerline(sub, empresaid=empresaid)
        sub = project_to_centerline(sub, centerline, empresaid=empresaid)
        sub = infer_direction(sub)
        pass1_s = sub["s"]  # snapshot for continuity assertion

        # Pass-2: per-direction centerlines → refined labels
        cls = build_centerline_per_direction(
            sub, empresaid=empresaid,
            min_pings_per_dir=PRODUCTIVE_PARAMS.centerline_min_pings_per_direction,
        )
        sub = project_per_direction(sub, cls, empresaid=empresaid)
        sub = infer_direction(sub)

        # s-continuity runtime assertion (spec Q-S-CONTINUITY)
        assert "s" in sub.columns, "s column missing after pass-2 projection"

        # R-PIPE2: assign_trip_ids MUST run AFTER second infer_direction
        sub = assign_trip_ids(sub)
    else:
        print(f"  Strategy: single-pass")
        centerline = build_centerline(sub, empresaid=empresaid)
        sub = project_to_centerline(sub, centerline, empresaid=empresaid)
        sub = infer_direction(sub)
        sub = assign_trip_ids(sub)

    snaps = build_snapshots(sub)
    heads, null_buckets = compute_headways_c2(snaps, sub)

    out_gps = OUTPUT_DIR / f"cleaned_gps_E{empresaid}.parquet"
    out_hw = OUTPUT_DIR / f"headways_E{empresaid}.parquet"
    out_buckets = OUTPUT_DIR / f"headway_null_buckets_E{empresaid}.parquet"
    sub.rename({"time": "t"}).select(
        ["unidadid", "t", "lat", "lon", "s", "direction", "speed_kmh", "lateral_m"]
    ).write_parquet(out_gps)
    heads.write_parquet(out_hw)
    null_buckets.write_parquet(out_buckets)

    print(f"  cleaned_gps:  {sub.height:,} rows → {out_gps}")
    print(f"  headways:     {heads.height:,} rows → {out_hw}")
    print(f"  non-null hw:  {heads.filter(pl.col('delta_t_min').is_not_null()).height:,}")
    print(f"  null_buckets: {null_buckets.height:,} rows → {out_buckets}")
""",
    cell_id="cell-16-run-prep",
)

# ---------------------------------------------------------------------------
# Cell: Sanity audit on the produced E4 parquets (cell-16-audit)
# ---------------------------------------------------------------------------

md(
    """## Auditoría de sanidad — E4

Verifica invariantes (INV-4/6/7/8) y la cobertura de `delta_t_min`. La tasa de
crossings stale aquí es la evidencia para decidir si E4 debe pasar a "two-pass"
(flip de 1 línea en `config.py`).
""",
    cell_id="cell-16-audit-md",
)

code(
    """
for empresaid in EMPRESAS:
    out_gps = OUTPUT_DIR / f"cleaned_gps_E{empresaid}.parquet"
    out_hw = OUTPUT_DIR / f"headways_E{empresaid}.parquet"
    if not out_gps.exists() or not out_hw.exists():
        print(f"E{empresaid}: output files not found, skip audit")
        continue

    gps_e = pl.read_parquet(out_gps)
    hw_e = pl.read_parquet(out_hw)

    print(f"\\n=== E{empresaid} audit ===")
    print(f"  cleaned_gps: {gps_e.height:,} rows, {gps_e.width} cols")
    print(f"  headways:    {hw_e.height:,} rows, {hw_e.width} cols")

    if hw_e.height > 0:
        bad_seconds = hw_e.filter(pl.col("t").dt.second() != 0).height
        print(f"  INV-6 violations (t.second != 0): {bad_seconds}")
        bad_n = hw_e.filter(pl.col("n_buses") < 2).height
        print(f"  INV-4 violations (n_buses < 2): {bad_n}")
        bad_pair = hw_e.filter(pl.col("bus_front") == pl.col("bus_back")).height
        print(f"  INV-7 violations (bus_front == bus_back): {bad_pair}")

    if gps_e.height > 0:
        bad_lat = gps_e.filter(pl.col("lateral_m") > 300.0).height
        print(f"  INV-8 violations (lateral_m > 300): {bad_lat}")

    if hw_e.height > 0:
        null_frac = hw_e.filter(pl.col("delta_t_min").is_null()).height / hw_e.height
        print(f"  delta_t_min null fraction: {null_frac:.1%}")
        print(f"  delta_t_min stats: {hw_e['delta_t_min'].drop_nulls().describe()}")

    if hw_e.height > 0:
        pairs_per_day = (
            hw_e.filter(pl.col("delta_t_min").is_not_null())
            .with_columns(pl.col("t").dt.date().alias("day"))
            .group_by("day").len().sort("day")
        )
        print(f"  pairs_efectivo/day: min={pairs_per_day['len'].min():,} "
              f"max={pairs_per_day['len'].max():,} mean={int(pairs_per_day['len'].mean()):,}")
""",
    cell_id="cell-16-audit",
)

# ---------------------------------------------------------------------------
# Baselines module cells (mirror of build_notebook_10.py, dependency order:
# splits → metrics → statistical → fitted → harness)
# ---------------------------------------------------------------------------

md(
    """## Baselines — embed de la librería de evaluación

A partir de aquí se reusa la librería de NB10 (sin tocarla) para evaluar los
baselines clásicos sobre las headways E4 recién calculadas.
""",
    cell_id="cell-16-baselines-section-md",
)

embed_module(
    "evaluation/splits.py",
    """## Module: evaluation/splits

Temporal split (`split_temporal`) y winsorización train-only p99
(`winsorize_train_p99`). Los rangos de fecha están fijados en spec §3.
""",
    cell_id_md="cell-16-embed-splits-md",
    cell_id_code="cell-16-embed-splits",
)

embed_module(
    "evaluation/metrics.py",
    """## Module: evaluation/metrics

`mae` y `rmse` en minutos. Aceptan polars Series o numpy. Filas null/NaN se
descartan antes de agregar. MAPE excluido (spec B3-NO-MAPE).
""",
    cell_id_md="cell-16-embed-metrics-md",
    cell_id_code="cell-16-embed-metrics",
)

embed_module(
    "baselines/statistical.py",
    """## Module: baselines/statistical

B0 media global, B1 persistencia naive (horizon-aware), B2 media móvil
(w∈{5,10,15}, horizon-aware), B3 suavizado exponencial (α=0.3, horizon-aware),
B4 promedio histórico por hora. Operan por slot `(empresaid, direction, pair_rank)`.
""",
    cell_id_md="cell-16-embed-statistical-md",
    cell_id_code="cell-16-embed-statistical",
)

embed_module(
    "baselines/fitted.py",
    """## Module: baselines/fitted

`predict_b5_xgb` — baseline ajustado (B5_XGB) sobre 12 lags + features de
calendario/slot. Debe embeberse ANTES de harness, que lo llama.
""",
    cell_id_md="cell-16-embed-fitted-md",
    cell_id_code="cell-16-embed-fitted",
)

embed_module(
    "baselines/harness.py",
    """## Module: baselines/harness

`evaluate_corridor` compone split → winsorize → B0-B4 + B5_XGB
(include_fitted=True por defecto) → métricas por (direction × baseline),
devolviendo un DataFrame long de 48 filas. Acepta `horizon`.
""",
    cell_id_md="cell-16-embed-harness-md",
    cell_id_code="cell-16-embed-harness",
)

# ---------------------------------------------------------------------------
# Cell: Load E4 headways back from the in-notebook output (cell-16-load)
# ---------------------------------------------------------------------------

md(
    """## Cargar headways E4 (calculadas en este mismo kernel)

Lee `headways_E4.parquet` recién escrito por la sección de preprocessing.
Inyecta `empresaid=4` como columna literal (el contrato del slot lo requiere).
""",
    cell_id="cell-16-load-md",
)

code(
    """
hw_e4 = pl.read_parquet(OUTPUT_DIR / "headways_E4.parquet").with_columns(
    pl.lit(4, dtype=pl.Int64).alias("empresaid")
)
print(f"E4: {hw_e4.height:,} rows, {hw_e4.width} cols")

# Cobertura por dirección (riesgo R-DIR1-COVERAGE).
summary = (
    hw_e4.group_by(["empresaid", "direction"])
    .agg([
        pl.len().alias("n_rows"),
        (pl.col("delta_t_min").is_not_null().sum() / pl.len()).alias("non_null_frac"),
    ])
    .sort(["empresaid", "direction"])
)
print(summary)
""",
    cell_id="cell-16-load",
)

# ---------------------------------------------------------------------------
# Cell: Run harness — multi-horizon loop for E4 (cell-16-run-harness)
# ---------------------------------------------------------------------------

md(
    """## Ejecutar harness — loop multi-horizonte (E4)

Llama a `evaluate_corridor(hw_e4, "E4", horizon=h)` para h ∈ {1, 3, 5, 10}
y concatena agregando la columna `horizon`. Salida esperada:
**192 filas = 4 horizontes × 1 corredor × 48** (3 dir × 8 baselines × 2 métricas).
""",
    cell_id="cell-16-run-harness-md",
)

code(
    """
HORIZONS = [1, 3, 5, 10]
frames = []
for h in HORIZONS:
    r_e4 = evaluate_corridor(hw_e4, "E4", horizon=h)
    frames.append(r_e4.with_columns(pl.lit(h, dtype=pl.Int64).alias("horizon")))
results = pl.concat(frames)
print(f"Total rows: {results.height}  (expected 192 = 4 horizons x 1 corridor x 48)")
print(results.head(10))
""",
    cell_id="cell-16-run-harness",
)

# ---------------------------------------------------------------------------
# Cell: Write CSV (cell-16-write-csv)
# ---------------------------------------------------------------------------

md(
    """## Escribir CSV — baselines_E4_results_multih.csv

Escribe la tabla long-form a `/kaggle/working/baselines_E4_results_multih.csv`.
Columnas: `corridor, direction, baseline, metric, value, horizon`.
""",
    cell_id="cell-16-write-csv-md",
)

code(
    """
results.write_csv(CSV_OUT)
print(f"CSV written to: {CSV_OUT}")
print(f"Rows: {results.height}  Columns: {results.columns}")
""",
    cell_id="cell-16-write-csv",
)

# ---------------------------------------------------------------------------
# Cell: Summary table (cell-16-summary)
# ---------------------------------------------------------------------------

md(
    """## Tabla resumen (wide format)

Pivote ancho para lectura humana: filas = (corridor, horizon, direction, metric),
columnas = baseline.
""",
    cell_id="cell-16-summary-md",
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
    cell_id="cell-16-summary",
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
    # Title must slugify to the id, else Kaggle creates the kernel under a
    # title-derived slug (observed: "16 — E4 data + baselines" → 16-e4-data-baselines).
    "id": "alexhuaracha/16-e4-data-baselines",
    "title": "16 E4 data baselines",
    "code_file": "16_e4_data.ipynb",
    **_KERNEL_META_BASE,
}
meta_path = OUT.parent / "kernel-metadata.json"
meta_path.write_text(json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8")
print(f"Kernel metadata written: {meta_path}")
