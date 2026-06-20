"""Generate notebooks/14_lstm_minigrid_h10/14_lstm_minigrid_h10.ipynb for Kaggle.

Produces ONE notebook running a hyperparameter sensitivity mini-grid for the
LSTM model at horizon h=10 only.  Trains 8 configs total (4 per corridor):
  - 1 winner control (the frozen Fase-5 best config re-run at h=10 for internal
    comparison — same seed, same environment as the neighbors)
  - 3 neighbors (each changes exactly ONE hyperparameter from the winner)

Winning configs (from Fase 5 grid search, h=1):
  E2:  hidden=32, layers=1, dropout=0.0, lr=0.0005
  E59: hidden=32, layers=2, dropout=0.2, lr=0.0005

8 configs (all LSTM, h=10):
  E2-ctrl:  hidden=32, layers=1, dropout=0.0, lr=0.0005  (winner/control)
  E2-1:     hidden=64, layers=1, dropout=0.0, lr=0.0005  (vary hidden)
  E2-2:     hidden=32, layers=1, dropout=0.2, lr=0.0005  (vary dropout)
  E2-3:     hidden=32, layers=1, dropout=0.0, lr=0.001   (vary lr)
  E59-ctrl: hidden=32, layers=2, dropout=0.2, lr=0.0005  (winner/control)
  E59-4:    hidden=64, layers=2, dropout=0.2, lr=0.0005  (vary hidden)
  E59-5:    hidden=32, layers=2, dropout=0.0, lr=0.0005  (vary dropout)
  E59-6:    hidden=32, layers=2, dropout=0.2, lr=0.001   (vary lr)

Output CSV: lstm_minigrid_h10.csv
Columns: corridor, hidden_size, num_layers, dropout, lr, mae, rmse, role
  role: 'winner' for control rows, 'neighbor' for the 3 neighbors per corridor.

The notebook reuses the same data loading, splits, normalization, context
features, and dataset construction as build_notebook_11.py (h=10) so all 8
configs are trained under identical conditions (same seed, same pipeline/splits).

Kernel:
  Folder/slug: notebooks/14_lstm_minigrid_h10/
  Title:       "14 LSTM Minigrid h10"
  id:          alexhuaracha/14-lstm-minigrid-h10
  GPU:         T4x2, enable_gpu=True
  kernel_sources: ["alexhuaracha/04-preprocessing",
                   "alexhuaracha/10-baselines-multi-horizonte"]
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_utils import _strip_relative_imports  # noqa: E402

# ---------------------------------------------------------------------------
# Kaggle kernel metadata
# ---------------------------------------------------------------------------

_KERNEL_META = {
    "id": "alexhuaracha/14-lstm-minigrid-h10",
    "title": "14 LSTM Minigrid h10",
    "code_file": "14_lstm_minigrid_h10.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "accelerator": "GPU_T4X2",
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [
        "alexhuaracha/04-preprocessing",
        "alexhuaracha/10-baselines-multi-horizonte",
    ],
    "competition_sources": [],
}

# ---------------------------------------------------------------------------
# Cell accumulator (module-level, reset each build)
# ---------------------------------------------------------------------------

_cells: list = []
_nb = None


def _reset() -> None:
    global _cells, _nb
    _cells = []
    _nb = nbf.v4.new_notebook()


def md(text: str, cell_id: str) -> None:
    cell = nbf.v4.new_markdown_cell(text.strip())
    cell["id"] = cell_id
    _cells.append(cell)


def code(src: str, cell_id: str) -> None:
    cell = nbf.v4.new_code_cell(src.rstrip())
    cell["id"] = cell_id
    _cells.append(cell)


def embed_module(rel_path: str, header_md: str, cell_id_md: str, cell_id_code: str) -> None:
    """Embed a source file (relative to ROOT/src/) as a markdown + code cell pair."""
    md(header_md, cell_id=cell_id_md)
    raw = (ROOT / "src" / rel_path).read_text(encoding="utf-8")
    code(_strip_relative_imports(raw), cell_id=cell_id_code)


# ---------------------------------------------------------------------------
# Module embed cells — identical dependency order to NB11
# ---------------------------------------------------------------------------

def _add_embed_cells() -> None:
    embed_module(
        "evaluation/splits.py",
        """## Module: evaluation/splits

Temporal split helper (`split_temporal`) and train-only p99 winsorization
(`winsorize_train_p99`).  Split date ranges are locked in spec §3.
""",
        cell_id_md="cell-14-embed-splits-md",
        cell_id_code="cell-14-embed-splits",
    )

    embed_module(
        "evaluation/metrics.py",
        """## Module: evaluation/metrics

`mae` and `rmse` in minutes.  Both accept polars Series or numpy arrays.
Null/NaN rows are dropped before aggregation.
""",
        cell_id_md="cell-14-embed-metrics-md",
        cell_id_code="cell-14-embed-metrics",
    )

    embed_module(
        "data/windowing.py",
        """## Module: data/windowing

`make_window_index` — per-slot deterministic window index.
`compute_max_N` — train-p99 of (n_buses - 1) per (empresaid, direction).
Constants: `DEFAULT_T_IN=12`, `DEFAULT_T_OUT=1`, `DEFAULT_STRIDE=1`.
""",
        cell_id_md="cell-14-embed-windowing-md",
        cell_id_code="cell-14-embed-windowing",
    )

    embed_module(
        "data/normalization.py",
        """## Module: data/normalization

`compute_normalization_stats` — per-direction z-score stats from TRAIN ONLY.
`apply_zscore` — add `delta_t_min_z` column; no clipping (DL-8).
""",
        cell_id_md="cell-14-embed-normalization-md",
        cell_id_code="cell-14-embed-normalization",
    )

    embed_module(
        "data/context_features.py",
        """## Module: data/context_features

`encode_context` — add 5 cyclical + atypical-flag columns.
`load_atypical_days` — graceful fallback to empty set when CSV absent (DL-2).
""",
        cell_id_md="cell-14-embed-context-md",
        cell_id_code="cell-14-embed-context",
    )

    embed_module(
        "data/dataset.py",
        """## Module: data/dataset  (first torch import)

`HeadwayDataset` — on-the-fly window materialization with masks (DL-11).
`collate_fn` — batch stacking for variable-N edge cases (REQ-6).
""",
        cell_id_md="cell-14-embed-dataset-md",
        cell_id_code="cell-14-embed-dataset",
    )

    embed_module(
        "models/lstm.py",
        """## Module: models/lstm

`HeadwayLSTM` — flat LSTM encoder (batch_first, last hidden state → Linear head).
`masked_mse_loss` — MSE over valid (mask==True) positions; clamp(min=1) prevents
zero-division on all-False masks.
""",
        cell_id_md="cell-14-embed-lstm-md",
        cell_id_code="cell-14-embed-lstm",
    )

    embed_module(
        "train.py",
        """## Module: train

`TrainConfig`, `TrainResult` — hyperparameter and result dataclasses.
`set_seed` — reproducibility seeds (torch + cuda + numpy).
`train_one_epoch`, `evaluate_epoch` — single-epoch train/eval loops.
`EarlyStopping` — patience-based early stopping with best-state copy.
`train_model` — full training loop with early stopping.
`grid_search` — run configs; return sorted by best_val_loss.
`denormalize_predictions` — z-score → minutes conversion.
""",
        cell_id_md="cell-14-embed-train-md",
        cell_id_code="cell-14-embed-train",
    )


# ---------------------------------------------------------------------------
# Notebook cells
# ---------------------------------------------------------------------------

def _add_title_cell() -> None:
    md(
        """
# 14 — LSTM Mini-Grid Sensibilidad h=10  (auto-generado por build_notebook_14.py)

Entrena **8 configs** (4 por corredor) para **E2** y **E59** a horizonte directo
**h=10 minutos**.  Para cada corredor: 1 config de control (ganador congelado de
Fase 5, re-entrenado aquí con el mismo seed y pipeline para comparación interna)
+ 3 vecinos que cambian exactamente UN hiperparámetro.

| Config | Corredor | hidden | layers | dropout | lr | role | Cambio |
|--------|----------|--------|--------|---------|-----|------|--------|
| E2-ctrl  | E2  | 32 | 1 | 0.0 | 5e-4 | winner | control (ganador Fase 5) |
| E2-1     | E2  | 64 | 1 | 0.0 | 5e-4 | neighbor | hidden ↑ |
| E2-2     | E2  | 32 | 1 | 0.2 | 5e-4 | neighbor | dropout ↑ |
| E2-3     | E2  | 32 | 1 | 0.0 | 1e-3 | neighbor | lr ↑ |
| E59-ctrl | E59 | 32 | 2 | 0.2 | 5e-4 | winner | control (ganador Fase 5) |
| E59-4    | E59 | 64 | 2 | 0.2 | 5e-4 | neighbor | hidden ↑ |
| E59-5    | E59 | 32 | 2 | 0.0 | 5e-4 | neighbor | dropout ↓ |
| E59-6    | E59 | 32 | 2 | 0.2 | 1e-3 | neighbor | lr ↑ |

Exporta `lstm_minigrid_h10.csv` con columnas:
`corridor, hidden_size, num_layers, dropout, lr, mae, rmse, role`

La columna `role` es `winner` para la config de control y `neighbor` para los vecinos.
La comparación es **interna**: mismo seed, mismo entorno, mismo pipeline/splits.
""",
        cell_id="cell-14-title",
    )


def _add_setup_cell() -> None:
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
HORIZON = 10
MINIGRID_CSV_OUT = OUTPUT_DIR / "lstm_minigrid_h10.csv"

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"Output dir: {OUTPUT_DIR}")
print(f"Horizon:    {HORIZON}")
print(f"Device:     {DEVICE}")
""",
        cell_id="cell-14-setup",
    )


def _add_load_cell() -> None:
    md(
        """## Cargar datos — E2 y E59

Lee los parquets generados por NB04 (kernel_source: `alexhuaracha/04-preprocessing`).
Inyecta `empresaid` como columna literal para cumplir el contrato de slot key.
""",
        cell_id="cell-14-load-md",
    )
    code(
        """
hw_e2  = pl.read_parquet(_find_parquet(2)).with_columns(pl.lit(2,  dtype=pl.Int64).alias("empresaid"))
hw_e59 = pl.read_parquet(_find_parquet(59)).with_columns(pl.lit(59, dtype=pl.Int64).alias("empresaid"))

print(f"E2:  {hw_e2.height:,} rows, {hw_e2.width} cols")
print(f"E59: {hw_e59.height:,} rows, {hw_e59.width} cols")
""",
        cell_id="cell-14-load",
    )


def _add_split_cell() -> None:
    md(
        """## Split temporal + winsorización

Aplica `split_temporal` y `winsorize_train_p99` (INV-1, INV-6).
El umbral de winsorización se computa exclusivamente sobre el split `train`
(AC-WINSOR-1, AC-WINSOR-2 — leakage guard).
""",
        cell_id="cell-14-split-md",
    )
    code(
        """
def prepare_corridor(hw: pl.DataFrame, label: str) -> pl.DataFrame:
    df_split = split_temporal(hw)
    train_df = df_split.filter(pl.col("split") == "train")
    df_winsor, threshold = winsorize_train_p99(train_df)
    non_train = df_split.filter(pl.col("split") != "train")
    df_full = pl.concat([df_winsor, non_train])
    print(f"{label}: split counts = {df_split.group_by('split').agg(pl.len()).sort('split')}")
    print(f"{label}: winsorize threshold = {threshold:.4f} min")
    return df_full

df_e2  = prepare_corridor(hw_e2,  "E2")
df_e59 = prepare_corridor(hw_e59, "E59")
""",
        cell_id="cell-14-split",
    )


def _add_norm_cell() -> None:
    md(
        """## Normalización z-score (train only)

Computa estadísticas de normalización exclusivamente sobre filas de entrenamiento
(INV-2, AC-NORM-1) y aplica z-score a todos los splits.
""",
        cell_id="cell-14-norm-md",
    )
    code(
        """
def normalize_corridor(df: pl.DataFrame, label: str):
    train_only = df.filter(pl.col("split") == "train")
    stats = compute_normalization_stats(train_only)
    df_z = apply_zscore(df, stats)
    print(f"\\n{label} normalization stats:")
    for key in sorted(stats.means.keys()):
        print(f"  (empresa={key[0]}, dir={key[1]}): "
              f"mean={stats.means[key]:.4f}, std={stats.stds[key]:.4f}")
    return df_z, stats

df_e2,  stats_e2  = normalize_corridor(df_e2,  "E2")
df_e59, stats_e59 = normalize_corridor(df_e59, "E59")
""",
        cell_id="cell-14-norm",
    )


def _add_context_cell() -> None:
    md(
        """## Features de contexto

Codificación cíclica de hora y día de semana + flag de día atípico (DL-2).
Fallback gracioso a `atypical_flag=0` si el CSV está ausente.
""",
        cell_id="cell-14-context-md",
    )
    code(
        """
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
print(f"E2  context columns: {[c for c in df_e2.columns  if c in CONTEXT_FEATURE_NAMES]}")
print(f"E59 context columns: {[c for c in df_e59.columns if c in CONTEXT_FEATURE_NAMES]}")
""",
        cell_id="cell-14-context",
    )


def _add_dataset_cell() -> None:
    md(
        """## Construcción del Dataset — horizonte h=10

Calcula `max_N` (train-p99 de n_buses-1 por dirección), toma el máximo global
por corredor para dimensionar el modelo.

**Adaptación multi-horizonte (DIRECT)**: `window_size = T_IN + HORIZON`.
El loop en `fast_materialize` sólo asigna target en el último paso de la ventana
(`t_idx == window_size - 1`); los pasos intermedios entre input y target se saltan.
`make_window_index` recibe `horizon=HORIZON` para que el guard de slot vacío
rechace ventanas cuyo target +h no existe.

Nota: la construcción del dataset es IDÉNTICA a NB11-h10, así los resultados
del mini-grid son directamente comparables con los del ganador congelado.
""",
        cell_id="cell-14-dataset-md",
    )
    code(
        """
import torch
import time as _time

T_IN  = DEFAULT_T_IN   # 12
T_OUT = 1              # always 1 output row (DIRECT horizon — single target step)
BATCH_SIZE = 128

def _build_snapshot_lookup(df: pl.DataFrame, max_N: int, context_cols: list[str]):
    \"""Build a dict: (empresaid, direction, timestamp) -> (values, mask, context).

    One pass through a Polars group_by — O(n_unique_snapshots).
    \"""
    agg_exprs = [pl.col("pair_rank"), pl.col("delta_t_min_z")]
    for c in context_cols:
        agg_exprs.append(pl.col(c).first())

    grouped = df.group_by(["empresaid", "direction", "t"]).agg(agg_exprs)

    lookup = {}
    for row in grouped.iter_rows(named=True):
        key = (row["empresaid"], row["direction"], row["t"])
        vals = np.zeros(max_N, dtype=np.float32)
        mask = np.zeros(max_N, dtype=np.bool_)
        for pr, v in zip(row["pair_rank"], row["delta_t_min_z"]):
            if 0 <= pr < max_N and v is not None:
                vals[pr] = v
                mask[pr] = True
        ctx = np.array([row[c] if row[c] is not None else 0.0 for c in context_cols],
                       dtype=np.float32)
        lookup[key] = (vals, mask, ctx)
    return lookup

def _build_slot_timestamps(df: pl.DataFrame):
    \"""Build dict: (empresaid, direction, pair_rank) -> sorted list of timestamps.

    One Polars group_by — O(n_rows).
    \"""
    grouped = (
        df.group_by(["empresaid", "direction", "pair_rank"])
        .agg(pl.col("t").sort())
    )
    slots = {}
    for row in grouped.iter_rows(named=True):
        key = (row["empresaid"], row["direction"], row["pair_rank"])
        slots[key] = row["t"]
    return slots

def fast_materialize(df: pl.DataFrame, window_index, max_N: int,
                     context_cols: list[str], label: str):
    \"""Materialize all windows into batched tensor lists using dict lookups.

    DIRECT multi-horizon: window_size = T_IN + HORIZON.
    The loop fills:
      - t_idx < T_IN           -> input row
      - t_idx == window_size-1 -> the single target row at +HORIZON
      - else (intermediate)    -> skipped (not input, not target)

    Returns a list of batch dicts (same format as DataLoader output).
    \"""
    t0 = _time.time()
    window_size = T_IN + HORIZON
    n_windows = len(window_index)

    lookup = _build_snapshot_lookup(df, max_N, context_cols)
    slots = _build_slot_timestamps(df)
    t_lookup = _time.time()
    print(f"  {label}: lookup built in {t_lookup - t0:.1f}s "
          f"({len(lookup):,} snapshots, {len(slots):,} slots)")

    all_input      = np.zeros((n_windows, T_IN,  max_N), dtype=np.float32)
    all_target     = np.zeros((n_windows, T_OUT, max_N), dtype=np.float32)
    all_input_mask = np.zeros((n_windows, T_IN,  max_N), dtype=np.bool_)
    all_target_mask= np.zeros((n_windows, T_OUT, max_N), dtype=np.bool_)
    all_context    = np.zeros((n_windows, T_IN,  len(context_cols)), dtype=np.float32)

    for i, entry in enumerate(window_index):
        slot_key = (entry["empresaid"], entry["direction"], entry["pair_rank"])
        emp, dirn = entry["empresaid"], entry["direction"]
        ts_list = slots[slot_key]
        start = entry["start_idx"]

        for t_idx in range(window_size):
            ts = ts_list[start + t_idx]
            snap = lookup.get((emp, dirn, ts))
            if snap is None:
                continue
            vals, mask, ctx = snap
            if t_idx < T_IN:
                all_input[i, t_idx] = vals
                all_input_mask[i, t_idx] = mask
                all_context[i, t_idx] = ctx
            elif t_idx == window_size - 1:
                # Single target row at +HORIZON (DIRECT forecast)
                all_target[i, 0] = vals
                all_target_mask[i, 0] = mask
            # else: intermediate row — skip (not input, not target)

    # Convert to tensors and split into batches.
    tensors = {
        "input":       torch.from_numpy(all_input),
        "target":      torch.from_numpy(all_target),
        "input_mask":  torch.from_numpy(all_input_mask),
        "target_mask": torch.from_numpy(all_target_mask),
        "context":     torch.from_numpy(all_context),
    }

    batches = []
    for start in range(0, n_windows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_windows)
        batch = {k: v[start:end] for k, v in tensors.items()}
        batches.append(batch)

    elapsed = _time.time() - t0
    print(f"  {label}: {n_windows:,} windows -> {len(batches):,} batches in {elapsed:.1f}s")
    return batches

CTX_COLS = list(CONTEXT_FEATURE_NAMES)

def build_corridor_data(df: pl.DataFrame, label: str):
    \"""Build cached batches for one corridor (both directions combined).\"""
    train_df = df.filter(pl.col("split") == "train")
    max_n = compute_max_N(train_df, quantile=0.99)
    global_max_N = max(max_n.values())
    print(f"\\n{label} max_N per direction: {max_n}")
    print(f"{label} global max_N (for model): {global_max_N}")

    cached = {}
    for split_name in ["train", "val"]:
        split_df = df.filter(pl.col("split") == split_name)
        idx = make_window_index(split_df, T_in=T_IN, horizon=HORIZON)
        cached[split_name] = fast_materialize(
            split_df, idx, global_max_N, CTX_COLS, f"{label} {split_name}")

    cached_test = {}
    for direction in [-1, 1]:
        test_df = df.filter(
            (pl.col("split") == "test") & (pl.col("direction") == direction)
        )
        idx = make_window_index(test_df, T_in=T_IN, horizon=HORIZON)
        cached_test[direction] = fast_materialize(
            test_df, idx, global_max_N, CTX_COLS, f"{label} test dir={direction:+d}")

    return global_max_N, cached, cached_test

max_N_e2,  cached_e2,  cached_test_e2  = build_corridor_data(df_e2,  "E2")
max_N_e59, cached_e59, cached_test_e59 = build_corridor_data(df_e59, "E59")
print("\\nDataset construction complete.")
""",
        cell_id="cell-14-dataset",
    )


def _add_train_cell() -> None:
    md(
        """## Entrenamiento LSTM — mini-grid (8 configs: 1 winner + 3 neighbors por corredor)

Entrena 4 configs por corredor con `grid_search`:
- `winner`: config ganadora congelada de Fase 5, re-entrenada aquí con el mismo
  seed/pipeline para comparación interna (no depende de los números de NB11).
- `neighbor` (×3): cada uno cambia exactamente UN hiperparámetro del ganador.

El resultado de cada config (MAE y RMSE en test) se reporta en la tabla final
con la columna `role` para distinguir control de vecinos.
""",
        cell_id="cell-14-train-md",
    )
    code(
        """
# 8 configs: 4 per corridor (1 winner control + 3 neighbors).
# Each entry is a tuple (TrainConfig, role) where role is 'winner' or 'neighbor'.
#
# Winning configs (frozen, from Fase-5 grid search):
#   E2:  hidden=32, layers=1, dropout=0.0, lr=5e-4
#   E59: hidden=32, layers=2, dropout=0.2, lr=5e-4
#
# Controls are re-trained here (same seed, same environment) so the comparison
# is internal — no reliance on NB11's historical numbers.
MINIGRID_CONFIGS = {
    "E2": [
        (TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=0.0005), "winner"),    # control
        (TrainConfig(hidden_size=64, num_layers=1, dropout=0.0, lr=0.0005), "neighbor"),  # vary hidden
        (TrainConfig(hidden_size=32, num_layers=1, dropout=0.2, lr=0.0005), "neighbor"),  # vary dropout
        (TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=0.001),  "neighbor"),  # vary lr
    ],
    "E59": [
        (TrainConfig(hidden_size=32, num_layers=2, dropout=0.2, lr=0.0005), "winner"),    # control
        (TrainConfig(hidden_size=64, num_layers=2, dropout=0.2, lr=0.0005), "neighbor"),  # vary hidden
        (TrainConfig(hidden_size=32, num_layers=2, dropout=0.0, lr=0.0005), "neighbor"),  # vary dropout
        (TrainConfig(hidden_size=32, num_layers=2, dropout=0.2, lr=0.001),  "neighbor"),  # vary lr
    ],
}

def run_corridor_minigrid(loaders: dict, max_N: int, label: str):
    \"""Run grid_search over 4 configs (winner + 3 neighbors) for one corridor.

    Returns list of (TrainResult, role) tuples in grid_search order (ascending by
    val_loss). The role is matched to each result by config identity, so it stays
    correct regardless of the val_loss ordering.
    \"""
    configs_with_roles = MINIGRID_CONFIGS[label]
    configs = [cfg for cfg, _role in configs_with_roles]
    # grid_search returns results sorted ascending by val_loss, so the result
    # order does NOT match MINIGRID_CONFIGS input order. Map role by config
    # identity (hyperparameters) so the 'winner'/'neighbor' label stays attached
    # to the right config instead of leaking onto whichever config won val_loss.
    role_by_key = {
        (cfg.hidden_size, cfg.num_layers, cfg.dropout, cfg.lr): role
        for cfg, role in configs_with_roles
    }

    print(f"\\n{label} LSTM mini-grid: max_N={max_N}, device={DEVICE}, "
          f"n_configs={len(configs)} (1 winner + 3 neighbors)")
    results = grid_search(
        train_dl=loaders["train"],
        val_dl=loaders["val"],
        max_N=max_N,
        configs=configs,
        device=DEVICE,
    )
    results_with_roles = []
    for r in results:
        key = (r.config.hidden_size, r.config.num_layers, r.config.dropout, r.config.lr)
        role = role_by_key[key]
        print(f"  {label} [{role}] hidden={r.config.hidden_size} layers={r.config.num_layers} "
              f"dropout={r.config.dropout} lr={r.config.lr}: "
              f"val_loss={r.best_val_loss:.6f} (epoch {r.best_epoch})")
        results_with_roles.append((r, role))
    return results_with_roles

results_e2  = run_corridor_minigrid(cached_e2,  max_N_e2,  "E2")
results_e59 = run_corridor_minigrid(cached_e59, max_N_e59, "E59")
""",
        cell_id="cell-14-train",
    )


def _add_evaluate_cell() -> None:
    md(
        """## Evaluación en test — MAE y RMSE por config

Evalúa cada modelo entrenado en test, separando por dirección
para desnormalizar con la media/std específica de cada una.
`target.squeeze(1)` colapsa `(B, 1, max_N)` → `(B, max_N)` (invariante REQ-2).

Produce `corridor_config_metrics`: dict[(corridor, config_idx)] -> (config, mae, rmse, role).
""",
        cell_id="cell-14-evaluate-md",
    )
    code(
        """
def evaluate_single_config(train_result, test_loaders, max_N, stats, label, config_idx, role):
    \"""Evaluate one trained model on per-direction test splits.

    Returns aggregate (mae, rmse) across both directions.
    \"""
    empresa_id = int(list(stats.means.keys())[0][0])

    model = HeadwayLSTM(
        input_size=max_N + CONTEXT_DIM,
        hidden_size=train_result.config.hidden_size,
        output_size=max_N,
        num_layers=train_result.config.num_layers,
        dropout=train_result.config.dropout,
    )
    model.load_state_dict(train_result.state_dict)
    model.eval()
    model.to(torch.device(DEVICE))

    all_preds_global   = []
    all_targets_global = []

    for direction in [-1, 1]:
        mean_val = stats.means[(empresa_id, direction)]
        std_val  = stats.stds[(empresa_id, direction)]

        all_preds   = []
        all_targets = []
        all_masks   = []

        with torch.no_grad():
            for batch in test_loaders[direction]:
                inp    = batch["input"].to(DEVICE)
                ctx    = batch["context"].to(DEVICE)
                target = batch["target"].to(DEVICE)
                mask   = batch["target_mask"].to(DEVICE)

                x    = torch.cat([inp, ctx], dim=-1)
                pred = model(x)

                target_sq = target.squeeze(1)
                mask_sq   = mask.squeeze(1)

                pred_min   = denormalize_predictions(pred,      mean_val, std_val)
                target_min = denormalize_predictions(target_sq, mean_val, std_val)

                all_preds.append(pred_min.cpu().numpy())
                all_targets.append(target_min.cpu().numpy())
                all_masks.append(mask_sq.cpu().numpy())

        preds_flat   = np.concatenate([p.ravel() for p in all_preds])
        targets_flat = np.concatenate([t.ravel() for t in all_targets])
        masks_flat   = np.concatenate([m.ravel() for m in all_masks]).astype(bool)

        all_preds_global.append(preds_flat[masks_flat])
        all_targets_global.append(targets_flat[masks_flat])

    # Aggregate across both directions
    agg_preds   = np.concatenate(all_preds_global)
    agg_targets = np.concatenate(all_targets_global)
    agg_mae  = mae(agg_targets,  agg_preds)
    agg_rmse = rmse(agg_targets, agg_preds)

    cfg = train_result.config
    print(f"{label}[{config_idx}][{role}] hidden={cfg.hidden_size} layers={cfg.num_layers} "
          f"dropout={cfg.dropout} lr={cfg.lr}: "
          f"TEST MAE={agg_mae:.4f} min, RMSE={agg_rmse:.4f} min (n={len(agg_preds):,})")
    return agg_mae, agg_rmse

# Evaluate all 8 configs (results_e2 / results_e59 are lists of (TrainResult, role))
corridor_config_metrics = {}

for i, (r, role) in enumerate(results_e2):
    m, s = evaluate_single_config(r, cached_test_e2, max_N_e2, stats_e2, "E2", i, role)
    corridor_config_metrics[("E2", i)] = (r.config, m, s, role)

for i, (r, role) in enumerate(results_e59):
    m, s = evaluate_single_config(r, cached_test_e59, max_N_e59, stats_e59, "E59", i, role)
    corridor_config_metrics[("E59", i)] = (r.config, m, s, role)

print("\\nAll 8 configs evaluated.")
""",
        cell_id="cell-14-evaluate",
    )


def _add_results_cell() -> None:
    md(
        """## Exportar resultados — lstm_minigrid_h10.csv

Guarda los resultados del mini-grid en un CSV de una fila por config (8 total).
Schema: corridor, hidden_size, num_layers, dropout, lr, mae, rmse, role

La columna `role` es `winner` para la config de control (ganador Fase 5 re-entrenado
con el mismo seed/entorno) y `neighbor` para los 3 vecinos por corredor.
""",
        cell_id="cell-14-results-md",
    )
    code(
        """
# role values: 'winner' for the control config (frozen Fase-5 winner re-run here),
#              'neighbor' for the 3 sensitivity neighbors per corridor.
rows = []
for (corridor, idx), (cfg, mae_val, rmse_val, role) in sorted(corridor_config_metrics.items()):
    rows.append({
        "corridor":    corridor,
        "hidden_size": cfg.hidden_size,
        "num_layers":  cfg.num_layers,
        "dropout":     cfg.dropout,
        "lr":          cfg.lr,
        "mae":         float(mae_val),
        "rmse":        float(rmse_val),
        "role":        role,  # 'winner' | 'neighbor'
    })

minigrid_results = pl.DataFrame(rows)
# MINIGRID_CSV_OUT = OUTPUT_DIR / "lstm_minigrid_h10.csv"  (set in setup cell)
minigrid_results.write_csv(MINIGRID_CSV_OUT)
print(f"\\nMini-grid results written to: lstm_minigrid_h10.csv  ({minigrid_results.height} rows)")
print(minigrid_results)
""",
        cell_id="cell-14-results",
    )


# ---------------------------------------------------------------------------
# Top-level build function
# ---------------------------------------------------------------------------

def build_notebook() -> None:
    """Build and write the mini-grid notebook for h=10."""
    _reset()

    out_dir = ROOT / "notebooks" / "14_lstm_minigrid_h10"
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_filename = "14_lstm_minigrid_h10.ipynb"
    nb_path = out_dir / notebook_filename

    _add_title_cell()
    _add_setup_cell()
    _add_embed_cells()
    _add_load_cell()
    _add_split_cell()
    _add_norm_cell()
    _add_context_cell()
    _add_dataset_cell()
    _add_train_cell()
    _add_evaluate_cell()
    _add_results_cell()

    _nb["cells"] = _cells
    _nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    nb_path.write_text(nbf.writes(_nb), encoding="utf-8")
    print(f"Notebook written: {nb_path}  ({len(_cells)} cells)")

    meta_path = out_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(_KERNEL_META, indent=2) + "\n", encoding="utf-8")
    print(f"Kernel metadata written: {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_notebook()
