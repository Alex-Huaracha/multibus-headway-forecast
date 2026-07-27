"""Generate notebooks/20_xgb_paired/20_xgb_paired.ipynb for Kaggle.

20 — XGB paired export  (auto-generado por build_notebook_20_xgb_paired.py)

WHY THIS NOTEBOOK EXISTS
------------------------
The paper's "LSTM beats the leveled XGBoost baseline in 8/8 cells" claim compares
MAEs computed over DIFFERENT sample populations (every non-null test row for XGB
vs. the DL window population for the deep models). Re-scoring XGB over exactly
the DL's rows needs per-sample XGB TEST predictions carrying the FULL unique key
of the headways frame, ``(t, direction, pair_rank)``. The existing export
``xgb_residuals_multih.csv`` drops ``pair_rank``, so it cannot be joined
row-for-row against anything (~4.2 headway rows share a ``(t, direction)``).

This kernel produces that keyed export for ALL THREE corridors — E2, E59 and E4 —
at h ∈ {1, 3, 5, 10}, reusing ``src/baselines/paired_export.py`` on top of the
unmodified ``harness.run_corridor``.

WHAT THIS NOTEBOOK MUST NEVER DO
--------------------------------
  * It must NOT write any parquet. In particular it must never write
    ``headways_E4.parquet``: that file's SHA-256 is frozen in the INPUT_HASHES of
    NB17/NB18/NB19, so a non-byte-identical regeneration would make the three E4
    DL notebooks fail closed and force a full E4 GPU retrain. E4 headways are
    MOUNTED read-only here (kernel source ``16-e4-data-baselines``), never rebuilt.
  * Its output filenames must not contain the substring ``_results_``:
    ``build_degradation_curve.py`` globs ``*_results_*.csv`` and
    ``evaluation/degradation.py`` hard-requires the tidy metrics schema, so such a
    name would crash the degradation build or contaminate
    ``consolidated_multihorizon.csv`` and Figure 1. The names below also avoid
    ``*_residuals_h*.csv`` (globbed by ``evaluation/paired_audit.py``) and
    ``*_multiseed_*.csv`` (globbed by ``evaluation/multiseed.py``).

OUTPUT SHAPE — one combined CSV
-------------------------------
``xgb_paired_persample_test.csv`` holds all 3 corridors × 4 horizons in long
form, with ``corridor`` and ``horizon`` as columns (both are part of the join
key, so splitting by them would encode key material in filenames). One file =
one join for the consumer and no discover-and-concat step that could silently
miss a corridor. See ``src/baselines/paired_export.py`` for the full rationale.
A second, small CSV (``xgb_paired_search_provenance.csv``) records the winning
hyperparameter configuration per (corridor, horizon) so this refit can be diffed
against the frozen ``xgb_search_config_multih.csv`` /
``xgb_search_config_E4_multih.csv`` — a mismatch there invalidates the export.

HASH GATE
---------
All four inputs are hash-pinned and resolved through ``_resolve_input``, using
the SAME frozen digests as the DL notebooks: the three headways parquets plus
``atypical_days.csv``. Pinning the parquets (which NB10 did not do) is the point
of this kernel: the export is only comparable to the DL residuals if XGB was
fitted on byte-identical inputs. ``atypical_days.csv`` is a REQUIRED input and an
empty parsed date set raises, so the atypical-day feature can never be silently
inert.

RUNTIME NOTE
------------
12 fits (3 corridors × 4 horizons), each running the seeded 24-configuration
random search — roughly NB10 (8 fits) plus NB16 (4 fits) in a single CPU kernel.
The combined CSV is rewritten after each corridor completes so an interrupted
session still leaves the finished corridors on disk.

Inline-embed pattern (mirror of build_notebook_10.py):
  - Read each src/**.py dependency via Path.read_text(), strip relative imports,
    and inject as a code cell, so the notebook is a faithful flat copy of the
    modules at generation time. Tests run against the modules directly.
  - Stable cell IDs (cell-20-*) prevent git flutter on re-runs.

Output: notebooks/20_xgb_paired/20_xgb_paired.ipynb
Kaggle kernel: alexhuaracha/20-xgb-paired-export
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

OUT = ROOT / "notebooks" / "20_xgb_paired" / "20_xgb_paired.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Output CSV names. Deliberately free of `_results_`, `_residuals_h` and
# `_multiseed_` so no analysis-layer glob can pick them up by accident.
CSV_PAIRED_NAME = "xgb_paired_persample_test.csv"
CSV_PROVENANCE_NAME = "xgb_paired_search_provenance.csv"

# ---------------------------------------------------------------------------
# Kaggle kernel metadata — written alongside the notebook.
#
# kernel_sources:
#   04-preprocessing        → headways_E2.parquet, headways_E59.parquet
#   02-eda-corridors        → atypical_days.csv (hash-pinned, required)
#   16-e4-data-baselines    → headways_E4.parquet (MOUNTED, never rebuilt here)
# ---------------------------------------------------------------------------

_KERNEL_META_BASE = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [
        "alexhuaracha/04-preprocessing",
        "alexhuaracha/02-eda-corridors",
        "alexhuaracha/16-e4-data-baselines",
    ],
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
    return f"cell-20-{_cell_counter}"


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
# Cell: Title (cell-20-title)
# ---------------------------------------------------------------------------

md(
    """
# 20 — XGB paired export  (auto-generado por build_notebook_20_xgb_paired.py)

Exporta las predicciones **por muestra** de B5_XGB sobre el split de **test**
para los tres corredores (**E2**, **E59**, **E4**) a horizontes
**h ∈ {1, 3, 5, 10}**, con la clave completa
`(corridor, direction, horizon, t, pair_rank)`.

**Por qué**: las métricas agregadas de XGB
(`baselines_results_multih.csv`) se calculan sobre TODAS las filas de test con
predicción no nula, mientras que las métricas DL se calculan sobre la población
de ventanas (sin cold-start, con el target replicado por slot de anclaje). El
sesgo de encuadre medido para la persistencia (0.28-0.53 min) es MAYOR que 7 de
los 8 márgenes reclamados frente a XGB, así que la comparación agregada no es
defendible. Para re-puntuar XGB sobre exactamente las mismas filas que ve el DL
hace falta la clave única de la tabla de headways, y `pair_rank` es
imprescindible: `t` sola NO es única (~4.2 filas por `(t, direction)`).

**Este kernel NO reescribe ningún artefacto congelado.** `headways_E4.parquet`
se **monta** desde `16-e4-data-baselines` y nunca se regenera: su SHA-256 está
congelado en los INPUT_HASHES de NB17/NB18/NB19. Este notebook no escribe
parquet alguno.
""",
    cell_id="cell-20-title",
)

# ---------------------------------------------------------------------------
# Cell: Setup — hash gate, output paths (cell-20-setup)
# ---------------------------------------------------------------------------

md(
    """## Setup — gate de hashes congelados y rutas de salida

Los CUATRO inputs pasan por el gate: los tres parquets de headways y
`atypical_days.csv`. Los digests son los MISMOS que congelan los notebooks DL
(NB11/NB13 para E2+E59, NB17/NB19 para E4), porque este export sólo es
comparable con los residuos DL si XGB se ajustó sobre bytes idénticos.
""",
    cell_id="cell-20-setup-md",
)

code(
    """
import hashlib
import time

import polars as pl
import numpy as np
from pathlib import Path

# Frozen SHA-256 of every input. Identical digests to the DL notebooks:
#   headways_E2/E59  → NB11/NB12/NB13 INPUT_HASHES
#   headways_E4      → NB17/NB18/NB19 INPUT_HASHES
#   atypical_days    → all of the above + NB10/NB16
# The parquets are hash-gated here (NB10 did not gate them) because a paired
# re-scoring is only valid if XGB saw byte-identical inputs to the DL models.
INPUT_HASHES = {
    "headways_E2.parquet": "82a34eaffc79cd82346d4595a2e72f5d3ffb751ed37fa0fc0cde3a8f8fb345d4",
    "headways_E59.parquet": "0b5f5593caaa94e4e6af7da672bc2cad7b49b69b7cbd0a22092f15700a89a448",
    "headways_E4.parquet": "1dde7f38eea9bc7d9941c17cbc3d326cb864e70be815a1a7e3d0ae2691f19273",
    "atypical_days.csv": "2054245cc830e58b9397b75ea3b55d034581046b64e73b1630ca7d464e3ecb86",
}

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# Locate a required input by filename and verify its frozen SHA-256.
def _resolve_input(name: str) -> Path:
    roots = [Path("/kaggle/input"), Path(".")]
    candidates = [p for root in roots if root.exists() for p in sorted(root.rglob(name))]
    if not candidates:
        raise FileNotFoundError(f"Required input not found anywhere: {name}")
    for path in candidates:
        if _sha256_file(path) == INPUT_HASHES[name]:
            return path
    raise ValueError(
        f"No copy of {name} matches its frozen SHA-256 — "
        f"candidates: {[str(p) for p in candidates]}"
    )

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)

# Output names avoid `_results_` (globbed by build_degradation_curve.py),
# `_residuals_h` (globbed by evaluation/paired_audit.py) and `_multiseed_`
# (globbed by evaluation/multiseed.py) so no analysis-layer discovery can pick
# them up and crash on the foreign schema.
PAIRED_OUT = OUTPUT_DIR / "xgb_paired_persample_test.csv"
PROVENANCE_OUT = OUTPUT_DIR / "xgb_paired_search_provenance.csv"

HORIZONS = [1, 3, 5, 10]
CORRIDORS = [("E2", 2), ("E59", 59), ("E4", 4)]

print(f"Output dir: {OUTPUT_DIR}")
print(f"Paired export: {PAIRED_OUT}")
print(f"Search provenance: {PROVENANCE_OUT}")
""",
    cell_id="cell-20-setup",
)

# ---------------------------------------------------------------------------
# Module embed cells — dependency order:
#   splits → metrics → statistical → context_features → fitted → harness
#   → paired_export
# ---------------------------------------------------------------------------

embed_module(
    "evaluation/splits.py",
    """## Module: evaluation/splits

Split temporal (`split_temporal`) y winsorización train-only p99
(`winsorize_train_p99`). El umbral p99 se calcula SOLO en train y se aplica a
TODOS los splits — `run_corridor` le pasa el frame completo con la etiqueta de
split, así que el contrato viaja intacto hasta este export.
""",
    cell_id_md="cell-20-embed-splits-md",
    cell_id_code="cell-20-embed-splits",
)

embed_module(
    "evaluation/metrics.py",
    """## Module: evaluation/metrics

`mae` y `rmse` en minutos (los consume `harness`; este notebook exporta residuos
por muestra, no métricas agregadas).
""",
    cell_id_md="cell-20-embed-metrics-md",
    cell_id_code="cell-20-embed-metrics",
)

embed_module(
    "baselines/statistical.py",
    """## Module: baselines/statistical

B0-B4. Aquí importa sobre todo **B1** (persistencia naive, horizon-aware): es la
columna `y_pred_persist` del export emparejado.
""",
    cell_id_md="cell-20-embed-statistical-md",
    cell_id_code="cell-20-embed-statistical",
)

embed_module(
    "data/context_features.py",
    """## Module: data/context_features

`load_atypical_days` + `encode_context` — los MISMOS helpers que usan los
notebooks DL. B5_XGB recibe `atypical_flag` a través de ellos. Debe embeberse
ANTES de `fitted`, que importa `encode_context`.
""",
    cell_id_md="cell-20-embed-context-md",
    cell_id_code="cell-20-embed-context",
)

embed_module(
    "baselines/fitted.py",
    """## Module: baselines/fitted

`fit_predict_b5_xgb` — baseline ajustado B5_XGB (12 lags + calendario/slot +
flag de día atípico) con random search sembrado de 24 configuraciones elegidas
**solo** sobre validación. `B5FitResult.predictions` CONSERVA `pair_rank`: es la
puerta de entrada del export emparejado.
""",
    cell_id_md="cell-20-embed-fitted-md",
    cell_id_code="cell-20-embed-fitted",
)

embed_module(
    "baselines/harness.py",
    """## Module: baselines/harness

`run_corridor` compone split → winsorize → B0-B4 + B5_XGB → métricas. Se embebe
**verbatim y sin modificar**: es el mismo código inlineado en NB10/NB16, y
tocarlo rompería la identidad byte a byte de esos notebooks congelados.
""",
    cell_id_md="cell-20-embed-harness-md",
    cell_id_code="cell-20-embed-harness",
)

embed_module(
    "baselines/paired_export.py",
    """## Module: baselines/paired_export

`export_paired_xgb` / `paired_xgb_test_frame` — reencuadran las filas de test de
`B5FitResult.predictions` al schema
`[corridor, empresaid, direction, horizon, t, pair_rank, y_true, y_pred_xgb,
y_pred_persist]`, con la MISMA semántica de filtrado que
`harness._build_xgb_residuals` (target y ambas predicciones no nulas) y la MISMA
convención de etiqueta de dirección (`"-1"` / `"+1"`) que los exports de
residuos DL. Debe embeberse DESPUÉS de `harness`, que es de quien importa.
""",
    cell_id_md="cell-20-embed-paired-export-md",
    cell_id_code="cell-20-embed-paired-export",
)

# ---------------------------------------------------------------------------
# Cell: atypical days (cell-20-atypical)
# ---------------------------------------------------------------------------

md(
    """## Días atípicos — input requerido y verificado por hash

`atypical_days.csv` (salida de NB02, kernel_source `alexhuaracha/02-eda-corridors`)
alimenta el flag `atypical_flag` que recibe B5_XGB. Es un input **requerido**: si
falta, si sus bytes no coinciden con el snapshot congelado, o si el set parsea
vacío, el kernel falla ANTES de ajustar nada — nunca se degrada en silencio a un
flag todo-ceros.
""",
    cell_id="cell-20-atypical-md",
)

code(
    """
atypical_path = _resolve_input("atypical_days.csv")
atypical_dates = load_atypical_days(atypical_path)
if not atypical_dates:
    raise ValueError(f"atypical_days.csv parsed to an empty date set: {atypical_path}")
print(f"Atypical days loaded: {len(atypical_dates)} dates (path={atypical_path})")
""",
    cell_id="cell-20-atypical",
)

# ---------------------------------------------------------------------------
# Cell: Load the three headways parquets (cell-20-load)
# ---------------------------------------------------------------------------

md(
    """## Cargar headways — E2, E59 y E4 (sólo lectura)

Los tres parquets se resuelven por el gate de hashes. `empresaid` es implícito en
el nombre del archivo, así que se inyecta como columna literal (el contrato del
slot `(empresaid, direction, pair_rank)` lo requiere).

E4 se **monta** desde `16-e4-data-baselines`: este notebook no ejecuta el
preprocessing de E4 ni escribe `headways_E4.parquet`.
""",
    cell_id="cell-20-load-md",
)

code(
    """
headways = {}
for label, empresa_id in CORRIDORS:
    path = _resolve_input(f"headways_E{empresa_id}.parquet")
    frame = pl.read_parquet(path).with_columns(
        pl.lit(empresa_id, dtype=pl.Int64).alias("empresaid")
    )
    headways[label] = frame
    non_null = frame.filter(pl.col("delta_t_min").is_not_null()).height
    print(f"{label}: {frame.height:,} rows, {frame.width} cols, "
          f"non-null delta_t_min={non_null:,}  ({path})")
""",
    cell_id="cell-20-load",
)

# ---------------------------------------------------------------------------
# Cell: Run the paired export loop (cell-20-run-export)
# ---------------------------------------------------------------------------

md(
    """## Loop de export — 3 corredores × 4 horizontes

`export_paired_xgb` llama a `run_corridor` (split → winsorize train-only p99 →
B0-B4 → B5_XGB con random search sobre validación) y reencuadra las filas de
test al schema emparejado. Nada del pipeline se reimplementa aquí.

Son 12 ajustes en un solo kernel CPU (≈ NB10 + NB16). El CSV combinado se
reescribe al terminar cada corredor, para que una sesión interrumpida deje en
disco los corredores ya completados.
""",
    cell_id="cell-20-run-export-md",
)

code(
    """
paired_frames = []
provenance_rows = []

for label, empresa_id in CORRIDORS:
    for h in HORIZONS:
        started = time.time()
        paired, run = export_paired_xgb(
            headways[label], label, horizon=h, atypical_dates=atypical_dates
        )
        paired_frames.append(paired)
        provenance_rows.append(search_provenance_row(run, label, horizon=h))
        fit = run.fit_result
        print(f"{label} h={h}: {paired.height:,} paired test samples  "
              f"best_val_rmse={fit.best_val_rmse:.5f}  "
              f"({fit.n_configs_evaluated} configs, {time.time() - started:.1f}s)")

    # Flush after each corridor so an interrupted session keeps finished work.
    pl.concat(paired_frames).write_csv(PAIRED_OUT)
    print(f"  [flush] {label} done → {PAIRED_OUT}")

paired_export = pl.concat(paired_frames)
provenance = pl.DataFrame(provenance_rows)
print(f"\\nTotal paired samples: {paired_export.height:,} "
      f"over {len(CORRIDORS)} corridors x {len(HORIZONS)} horizons")
""",
    cell_id="cell-20-run-export",
)

# ---------------------------------------------------------------------------
# Cell: Key-uniqueness verification (cell-20-verify)
# ---------------------------------------------------------------------------

md(
    """## Verificación — la clave exportada ES única

El defecto que este notebook corrige es exactamente una clave no única: el export
histórico (`xgb_residuals_multih.csv`) usa `t` como si fuera clave, pero hay ~4.2
filas por `(t, direction)`. Aquí se afirma en runtime que
`(corridor, direction, horizon, t, pair_rank)` no tiene duplicados, y se
contrasta contra el conteo de la clave sin `pair_rank` para dejar el factor de
colapso en el log.
""",
    cell_id="cell-20-verify-md",
)

code(
    """
n_rows = paired_export.height
n_full_key = paired_export.select(XGB_PAIRED_KEY).n_unique()
if n_full_key != n_rows:
    raise ValueError(
        f"paired export key is NOT unique: {n_rows:,} rows but {n_full_key:,} "
        f"distinct {XGB_PAIRED_KEY} tuples"
    )

key_without_pair_rank = [c for c in XGB_PAIRED_KEY if c != "pair_rank"]
n_collapsed = paired_export.select(key_without_pair_rank).n_unique()
print(f"Rows: {n_rows:,}")
print(f"Unique {XGB_PAIRED_KEY}: {n_full_key:,}  (must equal rows)")
print(f"Unique {key_without_pair_rank}: {n_collapsed:,} "
      f"→ {n_rows / max(n_collapsed, 1):.2f} rows per (t, direction) — this is why "
      f"pair_rank is required in the key")
print(paired_export.head(10))
print(paired_export.group_by(["corridor", "horizon"]).len().sort(["corridor", "horizon"]))
""",
    cell_id="cell-20-verify",
)

# ---------------------------------------------------------------------------
# Cell: Write CSVs (cell-20-write-csv)
# ---------------------------------------------------------------------------

md(
    """## Escribir CSVs (sólo CSV — este notebook NO escribe parquet)

1. `xgb_paired_persample_test.csv` — export emparejado por muestra
   (`corridor, empresaid, direction, horizon, t, pair_rank, y_true, y_pred_xgb,
   y_pred_persist`). Un único archivo combinado: `corridor` y `horizon` son parte
   de la clave, así que van como columnas y no como nombres de archivo.
2. `xgb_paired_search_provenance.csv` — configuración ganadora por
   (corredor, horizonte). Sirve para diferenciar este re-ajuste contra
   `xgb_search_config_multih.csv` / `xgb_search_config_E4_multih.csv`: si no
   coinciden, el export no es comparable y no debe usarse.

Ninguno de los dos nombres contiene `_results_`, `_residuals_h` ni `_multiseed_`,
para que ningún glob de la capa de análisis los descubra por accidente.
""",
    cell_id="cell-20-write-csv-md",
)

code(
    """
paired_export.write_csv(PAIRED_OUT)
print(f"Paired export written to: {PAIRED_OUT}")
print(f"Rows: {paired_export.height:,}  Columns: {paired_export.columns}")

provenance.write_csv(PROVENANCE_OUT)
print(f"Search provenance written to: {PROVENANCE_OUT}")
print(provenance)
""",
    cell_id="cell-20-write-csv",
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
# Title must slugify to the id, else Kaggle creates the kernel under a
# title-derived slug (see the NB16 note).
kernel_meta = {
    "id": "alexhuaracha/20-xgb-paired-export",
    "title": "20 XGB Paired Export",
    "code_file": "20_xgb_paired.ipynb",
    **_KERNEL_META_BASE,
}
meta_path = OUT.parent / "kernel-metadata.json"
meta_path.write_text(json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8")
print(f"Kernel metadata written: {meta_path}")
