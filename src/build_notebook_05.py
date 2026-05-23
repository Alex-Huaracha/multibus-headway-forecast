"""Generate notebooks/05_dataset/05_dataset.ipynb for Kaggle.

Inline-embed pattern (mirror of build_notebook_06.py):
  - Read each src/evaluation/*.py and src/data/*.py source file via
    Path.read_text(), strip relative imports, and inject as a code cell.
  - This ensures the notebook is always a faithful flat copy of the modules
    at generation time.  Tests run against the modules directly so the
    notebook never diverges.
  - Stable cell IDs (cell-05-*) prevent git flutter on re-runs (AC-NB-3).

Output: notebooks/05_dataset/05_dataset.ipynb
Kaggle kernel: alexhuaracha/05-dataset
kernel_sources: ["alexhuaracha/04-preprocessing"]  (DL-7)
"""
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
SRC_DATA = ROOT / "src" / "data"
OUT = ROOT / "notebooks" / "05_dataset" / "05_dataset.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cell builder helpers — every cell gets an explicit stable ID.
# ---------------------------------------------------------------------------

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str, cell_id: str) -> None:
    cell = nbf.v4.new_markdown_cell(text.strip())
    cell["id"] = cell_id
    cells.append(cell)


def code(src: str, cell_id: str) -> None:
    cell = nbf.v4.new_code_cell(src.rstrip())
    cell["id"] = cell_id
    cells.append(cell)


def embed_module(rel_path: str, header_md: str, cell_id_md: str, cell_id_code: str) -> None:
    """Embed a source file (relative to ROOT/src/) as a markdown + code cell pair."""
    md(header_md, cell_id=cell_id_md)
    raw = (ROOT / "src" / rel_path).read_text(encoding="utf-8")
    code(_strip_relative_imports(raw), cell_id=cell_id_code)


# ---------------------------------------------------------------------------
# Cell 1: Title (cell-05-title)
# ---------------------------------------------------------------------------

md(
    """
# 05 — Dataset supervisado  (auto-generado por build_notebook_05.py)

Construye el dataset supervisado para los corredores **E2** y **E59**:
ventanas deslizantes, normalización z-score por dirección, máscaras de
cardinalidad variable y un `HeadwayDataset` compatible con PyTorch DataLoader.

Referencia: `docs/plan-de-desarrollo.md §3 Fase 3 — Dataset supervisado`.
""",
    cell_id="cell-05-title",
)

# ---------------------------------------------------------------------------
# Cell 2: Setup — locate parquets and output dir (cell-05-setup)
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
STATS_CSV = OUTPUT_DIR / "dataset_stats.csv"

print(f"Output dir: {OUTPUT_DIR}")
""",
    cell_id="cell-05-setup",
)

# ---------------------------------------------------------------------------
# Module embed cells — dependency order:
#   splits → windowing → normalization → context_features → dataset
# ---------------------------------------------------------------------------

embed_module(
    "evaluation/splits.py",
    """## Module: evaluation/splits

Temporal split helper (`split_temporal`) and train-only p99 winsorization
(`winsorize_train_p99`).  Split date ranges are locked in spec §3.
""",
    cell_id_md="cell-05-embed-splits-md",
    cell_id_code="cell-05-embed-splits",
)

embed_module(
    "evaluation/metrics.py",
    """## Module: evaluation/metrics

`mae` and `rmse` in minutes.  Included for downstream sanity checks.
""",
    cell_id_md="cell-05-embed-metrics-md",
    cell_id_code="cell-05-embed-metrics",
)

embed_module(
    "data/windowing.py",
    """## Module: data/windowing

`make_window_index` — per-slot deterministic window index.
`compute_max_N` — train-p99 of (n_buses - 1) per (empresaid, direction).
Constants: `DEFAULT_T_IN=12`, `DEFAULT_T_OUT=1`, `DEFAULT_STRIDE=1`.
""",
    cell_id_md="cell-05-embed-windowing-md",
    cell_id_code="cell-05-embed-windowing",
)

embed_module(
    "data/normalization.py",
    """## Module: data/normalization

`compute_normalization_stats` — per-direction z-score stats from TRAIN ONLY.
`apply_zscore` — add `delta_t_min_z` column; no clipping (DL-8).
""",
    cell_id_md="cell-05-embed-normalization-md",
    cell_id_code="cell-05-embed-normalization",
)

embed_module(
    "data/context_features.py",
    """## Module: data/context_features

`encode_context` — add 5 cyclical + atypical-flag columns.
`load_atypical_days` — graceful fallback to empty set when CSV absent (DL-2).
""",
    cell_id_md="cell-05-embed-context-md",
    cell_id_code="cell-05-embed-context",
)

embed_module(
    "data/dataset.py",
    """## Module: data/dataset  (first torch import)

`HeadwayDataset` — on-the-fly window materialization with masks (DL-11).
`collate_fn` — batch stacking for variable-N edge cases (REQ-6).
""",
    cell_id_md="cell-05-embed-dataset-md",
    cell_id_code="cell-05-embed-dataset",
)

# ---------------------------------------------------------------------------
# Cell: Load data — E2 and E59 (cell-05-load)
# ---------------------------------------------------------------------------

md(
    """## Cargar datos — E2 y E59

Lee los parquets v8 generados por el notebook 04 (NB04b, kernel_sources:
`alexhuaracha/04-preprocessing`).  Inyecta `empresaid` como columna literal.

Row-count assertion vs dataset-manifest.md v8 pins (AC-NB-5, DL-7):
  - E2:  1,009,284 rows
  - E59: 2,069,193 rows
""",
    cell_id="cell-05-load-md",
)

code(
    """
# empresaid is implicit in the filename in the v8 parquets — inject it as a
# literal column so it matches the supervised-dataset contract (slot key requires it).
hw_e2  = pl.read_parquet(_find_parquet(2)).with_columns(pl.lit(2,  dtype=pl.Int64).alias("empresaid"))
hw_e59 = pl.read_parquet(_find_parquet(59)).with_columns(pl.lit(59, dtype=pl.Int64).alias("empresaid"))

# Row-count assertion against dataset-manifest.md v8 values (AC-NB-5, R-KERNEL-SOURCES-PIN).
E2_EXPECTED_ROWS  = 1_009_284
E59_EXPECTED_ROWS = 2_069_193
assert hw_e2.height == E2_EXPECTED_ROWS, (
    f"E2 row count mismatch: expected {E2_EXPECTED_ROWS:,}, got {hw_e2.height:,}. "
    "Is kernel_sources pointing to the correct NB04 run?"
)
assert hw_e59.height == E59_EXPECTED_ROWS, (
    f"E59 row count mismatch: expected {E59_EXPECTED_ROWS:,}, got {hw_e59.height:,}. "
    "Is kernel_sources pointing to the correct NB04 run?"
)

print(f"E2:  {hw_e2.height:,} rows, {hw_e2.width} cols — OK")
print(f"E59: {hw_e59.height:,} rows, {hw_e59.width} cols — OK")
""",
    cell_id="cell-05-load",
)

# ---------------------------------------------------------------------------
# Cell: Split + winsorize (cell-05-split-winsor)
# ---------------------------------------------------------------------------

md(
    """## Split temporal + winsorización

Aplica `split_temporal` y luego `winsorize_train_p99` (INV-1, INV-6).
Ambas funciones provienen de `src/evaluation/splits.py`.
""",
    cell_id="cell-05-split-winsor-md",
)

code(
    """
# Pipeline INV-1: split → winsorize → norm stats → z-score → max_N → windows → Dataset
results_e2 = {}
results_e59 = {}

def prepare_corridor(hw: pl.DataFrame, label: str) -> pl.DataFrame:
    df_split = split_temporal(hw)
    train_df  = df_split.filter(pl.col("split") == "train")
    df_winsor, threshold = winsorize_train_p99(train_df)
    # Re-attach the winsorized train to the full frame
    non_train = df_split.filter(pl.col("split") != "train")
    df_full = pl.concat([df_winsor, non_train])
    print(f"{label}: split counts = {df_split.group_by('split').agg(pl.len()).sort('split')}")
    print(f"{label}: winsorize threshold = {threshold:.4f} min")
    return df_full

df_e2  = prepare_corridor(hw_e2,  "E2")
df_e59 = prepare_corridor(hw_e59, "E59")
""",
    cell_id="cell-05-split-winsor",
)

# ---------------------------------------------------------------------------
# Cell: Normalization stats (cell-05-norm-stats)
# ---------------------------------------------------------------------------

md(
    """## Estadísticas de normalización (train only)

Computa media y desviación estándar de `delta_t_min` por `(empresaid, direction)`
usando SOLO filas de entrenamiento (INV-2, AC-NORM-1).
""",
    cell_id="cell-05-norm-stats-md",
)

code(
    """
def compute_stats_for(df: pl.DataFrame, label: str) -> "NormalizationStats":
    train_only = df.filter(pl.col("split") == "train")
    stats = compute_normalization_stats(train_only)
    print(f"\\n{label} normalization stats:")
    for key in sorted(stats.means.keys()):
        print(f"  (empresa={key[0]}, dir={key[1]}): mean={stats.means[key]:.4f}, std={stats.stds[key]:.4f}")
    return stats

stats_e2  = compute_stats_for(df_e2,  "E2")
stats_e59 = compute_stats_for(df_e59, "E59")
""",
    cell_id="cell-05-norm-stats",
)

# ---------------------------------------------------------------------------
# Cell: Z-score application (cell-05-zscore)
# ---------------------------------------------------------------------------

md(
    """## Aplicar z-score

Añade columna `delta_t_min_z` a todos los splits (train/val/test) usando
las estadísticas derivadas exclusivamente del tren (INV-2, DL-8 — sin clipping).
""",
    cell_id="cell-05-zscore-md",
)

code(
    """
df_e2  = apply_zscore(df_e2,  stats_e2)
df_e59 = apply_zscore(df_e59, stats_e59)

# Sanity: train z-score should have mean ≈ 0 and std ≈ 1 per direction
for label, df, stats in [("E2", df_e2, stats_e2), ("E59", df_e59, stats_e59)]:
    train_z = df.filter(pl.col("split") == "train")
    print(f"\\n{label} train z-score sanity:")
    for (emp, dirn) in sorted(stats.means.keys()):
        subset = train_z.filter(
            (pl.col("empresaid") == emp) & (pl.col("direction") == dirn)
        )["delta_t_min_z"].drop_nulls()
        if subset.len() > 0:
            print(f"  (empresa={emp}, dir={dirn}): mean_z={subset.mean():.4f}, std_z={subset.std():.4f}")
""",
    cell_id="cell-05-zscore",
)

# ---------------------------------------------------------------------------
# Cell: Context features (cell-05-context)
# ---------------------------------------------------------------------------

md(
    """## Features de contexto

Codificación cíclica de hora y día de semana + flag de día atípico (DL-2).
Fallback gracioso a `atypical_flag=0` si el CSV está ausente.
""",
    cell_id="cell-05-context-md",
)

code(
    """
# Try to locate atypical_days.csv (graceful fallback per DL-2)
atypical_path = None
if Path("/kaggle/input").exists():
    candidates = list(Path("/kaggle/input").rglob("atypical_days.csv"))
    if candidates:
        atypical_path = candidates[0]
if atypical_path is None:
    local_candidates = list(Path(".").rglob("atypical_days.csv"))
    if local_candidates:
        atypical_path = local_candidates[0]

atypical_dates = load_atypical_days(atypical_path)
print(f"Atypical days loaded: {len(atypical_dates)} dates (path={atypical_path})")

df_e2  = encode_context(df_e2,  atypical_dates=atypical_dates)
df_e59 = encode_context(df_e59, atypical_dates=atypical_dates)

print(f"E2 context columns:  {[c for c in df_e2.columns  if c in CONTEXT_FEATURE_NAMES]}")
print(f"E59 context columns: {[c for c in df_e59.columns if c in CONTEXT_FEATURE_NAMES]}")
""",
    cell_id="cell-05-context",
)

# ---------------------------------------------------------------------------
# Cell: max_N computation (cell-05-maxn)
# ---------------------------------------------------------------------------

md(
    """## Cómputo de max_N

`max_N = train-p99(n_buses - 1)` por `(empresaid, direction)` (DL-5, AC-MAXN-1).
Las snapshots de val/test que excedan `max_N` serán truncadas en el Dataset.
""",
    cell_id="cell-05-maxn-md",
)

code(
    """
def compute_maxn_for(df: pl.DataFrame, label: str) -> dict:
    train_only = df.filter(pl.col("split") == "train")
    max_n = compute_max_N(train_only, quantile=0.99)
    print(f"\\n{label} max_N per (empresa, direction):")
    for key in sorted(max_n.keys()):
        print(f"  (empresa={key[0]}, dir={key[1]}): max_N={max_n[key]}")

    # Truncation rate on val and test (R-MAXN-TRUNCATION guard, AC-MAXN-3)
    for split_name in ["val", "test"]:
        split_df = df.filter(pl.col("split") == split_name)
        if split_df.is_empty():
            continue
        # Count snapshots where n_buses - 1 > max_N
        n_total = split_df.height
        n_truncated = 0
        for (emp, dirn), cap in max_n.items():
            subset = split_df.filter(
                (pl.col("empresaid") == emp) & (pl.col("direction") == dirn)
            )
            if "n_buses" in subset.columns:
                n_truncated += subset.filter(pl.col("n_buses") - 1 > cap).height
        rate = n_truncated / n_total if n_total > 0 else 0.0
        print(f"  {label} {split_name} truncation rate: {rate:.4%} ({n_truncated}/{n_total})")
    return max_n

max_n_e2  = compute_maxn_for(df_e2,  "E2")
max_n_e59 = compute_maxn_for(df_e59, "E59")
""",
    cell_id="cell-05-maxn",
)

# ---------------------------------------------------------------------------
# Cell: Window indexes (cell-05-windows)
# ---------------------------------------------------------------------------

md(
    """## Índices de ventanas deslizantes

`make_window_index` per split per corredor (T_in=12, T_out=1, stride=1).
No materializa las ventanas — solo crea el índice de `(slot, start_idx)`.
""",
    cell_id="cell-05-windows-md",
)

code(
    """
T_IN  = DEFAULT_T_IN   # 12
T_OUT = DEFAULT_T_OUT  # 1

window_idx_e2  = {}
window_idx_e59 = {}

for split_name in ["train", "val", "test"]:
    idx_e2  = make_window_index(df_e2.filter(pl.col("split")  == split_name), T_in=T_IN, T_out=T_OUT)
    idx_e59 = make_window_index(df_e59.filter(pl.col("split") == split_name), T_in=T_IN, T_out=T_OUT)
    window_idx_e2[split_name]  = idx_e2
    window_idx_e59[split_name] = idx_e59
    print(f"E2  {split_name:5s}: {len(idx_e2):,} windows")
    print(f"E59 {split_name:5s}: {len(idx_e59):,} windows")
""",
    cell_id="cell-05-windows",
)

# ---------------------------------------------------------------------------
# Cell: HeadwayDataset smoke test (cell-05-dataset)
# ---------------------------------------------------------------------------

md(
    """## HeadwayDataset — smoke test

Instancia el dataset para el split de entrenamiento de E2 y verifica que
`__getitem__` retorna 5 tensores con las formas correctas (INV-4, AC-DS-1..4).
""",
    cell_id="cell-05-dataset-md",
)

code(
    """
import torch
from torch.utils.data import DataLoader

# Instantiate HeadwayDataset for E2 train split
ds_e2_train = HeadwayDataset(
    df=df_e2.filter(pl.col("split") == "train"),
    window_index=window_idx_e2["train"],
    max_N_by_direction=max_n_e2,
    T_in=T_IN,
    T_out=T_OUT,
)
print(f"E2 train dataset: {len(ds_e2_train):,} windows")

# Smoke test: __getitem__(0) must return 5 tensors with expected shapes
sample = ds_e2_train[0]
for key, tensor in sample.items():
    print(f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

# Verify shape invariants (INV-4)
assert sample["input"].shape[0] == T_IN,  f"input T_IN mismatch: {sample['input'].shape}"
assert sample["target"].shape[0] == T_OUT, f"target T_OUT mismatch: {sample['target'].shape}"
assert sample["context"].shape == (T_IN, 5), f"context shape mismatch: {sample['context'].shape}"
print("Shape invariants: OK")
""",
    cell_id="cell-05-dataset",
)

# ---------------------------------------------------------------------------
# Cell: DataLoader smoke test (cell-05-dataloader)
# ---------------------------------------------------------------------------

md(
    """## DataLoader — smoke test

Itera un batch del DataLoader de entrenamiento con `collate_fn` (AC-DS-5, AC-DS-6).
""",
    cell_id="cell-05-dataloader-md",
)

code(
    """
loader_e2_train = DataLoader(
    ds_e2_train,
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=False,
)

batch = next(iter(loader_e2_train))
print("One batch shapes:")
for key, tensor in batch.items():
    print(f"  {key}: {tuple(tensor.shape)}, dtype={tensor.dtype}")
print(f"Batch size: {batch['input'].shape[0]}")
""",
    cell_id="cell-05-dataloader",
)

# ---------------------------------------------------------------------------
# Cell: Stats CSV (cell-05-stats)
# ---------------------------------------------------------------------------

md(
    """## Estadísticas del dataset (dataset_stats.csv)

Escribe métricas por `(corridor, direction, split)` a `/kaggle/working/dataset_stats.csv`.
Columnas: corridor, direction, split, n_rows, n_windows, max_N,
          mean_delta_t_min, std_delta_t_min, truncation_rate.
""",
    cell_id="cell-05-stats-md",
)

code(
    """
stats_rows = []

for label, df, stats, max_n, win_idx in [
    ("E2",  df_e2,  stats_e2,  max_n_e2,  window_idx_e2),
    ("E59", df_e59, stats_e59, max_n_e59, window_idx_e59),
]:
    for split_name in ["train", "val", "test"]:
        split_df = df.filter(pl.col("split") == split_name)
        n_rows   = split_df.height
        n_windows = len(win_idx[split_name])

        for (emp, dirn), cap in max_n.items():
            mean_val = stats.means.get((emp, dirn), float("nan"))
            std_val  = stats.stds.get((emp, dirn), float("nan"))

            # Truncation rate (0 for train by definition)
            if split_name == "train" or "n_buses" not in split_df.columns:
                trunc_rate = 0.0
            else:
                subset = split_df.filter(
                    (pl.col("empresaid") == emp) & (pl.col("direction") == dirn)
                )
                n_total_sub = subset.height
                n_trunc_sub = subset.filter(pl.col("n_buses") - 1 > cap).height if n_total_sub > 0 else 0
                trunc_rate  = n_trunc_sub / n_total_sub if n_total_sub > 0 else 0.0

            stats_rows.append({
                "corridor":         label,
                "direction":        dirn,
                "split":            split_name,
                "n_rows":           n_rows,
                "n_windows":        n_windows,
                "max_N":            cap,
                "mean_delta_t_min": mean_val,
                "std_delta_t_min":  std_val,
                "truncation_rate":  trunc_rate,
            })

stats_df = pl.DataFrame(stats_rows)
stats_df.write_csv(STATS_CSV)
print(f"CSV written to: {STATS_CSV}")
print(stats_df)
""",
    cell_id="cell-05-stats",
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
