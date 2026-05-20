"""Generate the 04_preprocessing.ipynb file for Kaggle.

Inline-embed pattern: read each src/preprocessing/*.py source file via
Path.read_text() and inject the content as a code cell. This ensures the
notebook is always a faithful flat copy of the modules at generation time;
tests run against the modules directly so the notebook never diverges.

Output: notebooks/04_preprocessing/04_preprocessing.ipynb
Kaggle kernel: alexhuaracha/04-preprocessing
"""
import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "preprocessing"
OUT = ROOT / "notebooks" / "04_preprocessing" / "04_preprocessing.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _strip_relative_imports(src: str) -> str:
    """Remove all relative 'from .xxx import ...' statements from source.

    Uses ast.get_source_segment to locate and remove the EXACT text for every
    relative ImportFrom node (level > 0). This handles both single-line and
    parenthesized multi-line import blocks without regex fragility.

    Single-line:  from .config import X
    Multi-line:   from .config import (
                      X,
                      Y,
                  )

    Inside the notebook, all modules are inlined into the same flat namespace
    so relative imports would raise ImportError at Kaggle runtime. This is the
    only transformation applied to module source code before embedding.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # If the source itself is broken, return as-is so the compile test
        # catches it with a clear error.
        return src

    # Collect source segments for all relative ImportFrom nodes.
    segments_to_remove: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            segment = ast.get_source_segment(src, node)
            if segment:
                segments_to_remove.append(segment)

    result = src
    for segment in segments_to_remove:
        # Remove the segment plus its trailing newline (if any).
        result = result.replace(segment + "\n", "")
        result = result.replace(segment, "")

    return result


nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.rstrip()))


def embed_module(name: str, header_md: str) -> None:
    md(header_md)
    raw = (SRC / name).read_text(encoding="utf-8")
    code(_strip_relative_imports(raw))


# ---------------------------------------------------------------------------
# Cell 1: Title
# ---------------------------------------------------------------------------

md("""
# 04 — Preprocessing y headways  (auto-generado por build_notebook_04.py)

Este notebook aplica el pipeline de Fase 2 al dataset `clean_gps.parquet` para
producir `cleaned_gps_E{empresa}.parquet` y `headways_E{empresa}.parquet` por
corredor. Formulación adoptada: **Opción C.2 — trailing crossing** (ver
`docs/decisiones-headway-fase2.md §2`).

Parámetros productivos congelados en `config.py` desde
`docs/decisiones-headway-fase2.md §3`.
""")

# ---------------------------------------------------------------------------
# Cell 2: Setup (Kaggle locate input, output dirs)
# ---------------------------------------------------------------------------

code("""
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
print(f"Output dir: {OUTPUT_DIR}")

EMPRESAS = [2, 59]
""")

# ---------------------------------------------------------------------------
# Module cells in dependency order
# ---------------------------------------------------------------------------

embed_module(
    "config.py",
    """## Module: config

Parámetros productivos congelados desde `docs/decisiones-headway-fase2.md §3`.
Cualquier cambio requiere actualizar ese documento primero.
""",
)

embed_module(
    "corridor.py",
    """## Module: corridor

Construcción del trazado del corredor via PCA + binned median.
Fuente: `build_notebook_03.py` líneas 279-361.
""",
)

embed_module(
    "projection.py",
    """## Module: projection

Speed observado (`step_m / dt_s`, no `velocidad`) y proyección arc-length `s`.
Filtra pings off-route con `lateral_m > LATERAL_OFFSET_THRESHOLD_M`.
""",
)

embed_module(
    "direction.py",
    """## Module: direction

Inferencia de sentido ida/vuelta desde `sign(rolling_mean(ds, win=5))`.
El campo `direccion` se usa solo como verificación cruzada en E2.
""",
)

embed_module(
    "trips.py",
    """## Module: trips

Segmentación de viajes (gap 30 min / reversal / terminal dwell 5 min)
y grilla de snapshots con alineación minuto-exacta (INV-6).
""",
)

embed_module(
    "headways.py",
    """## Module: headways

C.2 trailing crossing — pure polars+numpy. Para pares sin historial previo
se emite `delta_t_min = null` (NO se descarta — clarification #17 rule 2).
Winsorización aplica en Fase 5, NO aquí (Caveat 2).
""",
)

# ---------------------------------------------------------------------------
# Pipeline runner cell
# ---------------------------------------------------------------------------

md("""## Ejecutar pipeline por empresa

Carga `clean_gps.parquet`, aplica todos los módulos en orden de dependencia,
y escribe los artefactos intermedios por corredor.
""")

code("""
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
    centerline = build_centerline(sub, empresaid=empresaid)
    sub = project_to_centerline(sub, centerline, empresaid=empresaid)
    sub = infer_direction(sub)
    sub = assign_trip_ids(sub)
    snaps = build_snapshots(sub)
    heads = compute_headways_c2(snaps, sub)

    out_gps = OUTPUT_DIR / f"cleaned_gps_E{empresaid}.parquet"
    out_hw = OUTPUT_DIR / f"headways_E{empresaid}.parquet"
    sub.rename({"time": "t"}).select(
        ["unidadid", "t", "lat", "lon", "s", "direction", "speed_kmh", "lateral_m"]
    ).write_parquet(out_gps)
    heads.write_parquet(out_hw)

    print(f"  cleaned_gps:  {sub.height:,} rows → {out_gps}")
    print(f"  headways:     {heads.height:,} rows → {out_hw}")
    print(f"  non-null hw:  {heads.filter(pl.col('delta_t_min').is_not_null()).height:,}")
""")

# ---------------------------------------------------------------------------
# Sanity audit cell
# ---------------------------------------------------------------------------

md("""## Auditoría de sanidad

Verifica los invariantes del spec (INV-1..INV-8) sobre los parquets producidos.
""")

code("""
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

    # INV-6: all t values have second == 0
    if hw_e.height > 0:
        bad_seconds = hw_e.filter(pl.col("t").dt.second() != 0).height
        print(f"  INV-6 violations (t.second != 0): {bad_seconds}")

    # INV-4: n_buses >= 2
    if hw_e.height > 0:
        bad_n = hw_e.filter(pl.col("n_buses") < 2).height
        print(f"  INV-4 violations (n_buses < 2): {bad_n}")

    # INV-7: bus_front != bus_back
    if hw_e.height > 0:
        bad_pair = hw_e.filter(pl.col("bus_front") == pl.col("bus_back")).height
        print(f"  INV-7 violations (bus_front == bus_back): {bad_pair}")

    # INV-8: lateral_m <= 300
    if gps_e.height > 0:
        bad_lat = gps_e.filter(pl.col("lateral_m") > 300.0).height
        print(f"  INV-8 violations (lateral_m > 300): {bad_lat}")

    # NULL rate in delta_t_min
    if hw_e.height > 0:
        null_frac = hw_e.filter(pl.col("delta_t_min").is_null()).height / hw_e.height
        print(f"  delta_t_min null fraction: {null_frac:.1%}")
        print(f"  delta_t_min stats: {hw_e['delta_t_min'].drop_nulls().describe()}")

    # n_pairs_efectivo per day
    if hw_e.height > 0:
        pairs_per_day = (
            hw_e.filter(pl.col("delta_t_min").is_not_null())
            .group_by("day").len().sort("day")
        )
        print(f"  pairs_efectivo/day: min={pairs_per_day['len'].min():,} "
              f"max={pairs_per_day['len'].max():,} mean={int(pairs_per_day['len'].mean()):,}")
""")

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
