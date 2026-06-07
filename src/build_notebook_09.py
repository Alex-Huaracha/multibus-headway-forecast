"""Generate two per-corridor notebooks for Kaggle from the SpatialTransformer experiment.

NB09a — E2:  notebooks/09_spatial_transformer/09a_e2/notebook.ipynb
NB09b — E59: notebooks/09_spatial_transformer/09b_e59/notebook.ipynb

Splitting reason: running both corridors sequentially in one Kaggle session
may hit the 12 h GPU limit. Each per-corridor kernel fits comfortably under 12 h.
(MHA cost ≈ 6a conv_channels=8; 24 configs × ~15 min/config ≈ 6h — safe margin.)

Inline-embed pattern (mirror of build_notebook_08.py):
  - Read each src/evaluation/*.py, src/data/*.py, src/models/*.py, and src/train.py
    source file via Path.read_text(), strip relative imports, and inject as a code cell.
  - This ensures the notebook is always a faithful flat copy of the modules
    at generation time.  Tests run against the modules directly so the
    notebook never diverges.
  - Stable cell IDs (cell-09-*) prevent git flutter on re-runs (AC-NB09-2).
    Both notebooks share the same cell-09-* prefix — identical IDs across
    separate files are valid (and required by AC-NB09-3).

Kaggle kernels:
  09a:  alexhuaracha/09a-spatialtransformer-e2    (E2, full 32-config grid)
  09b1: alexhuaracha/09b1-spatialtransformer-e59  (E59, tanda 1 = grid[0:16])
  09b2: alexhuaracha/09b2-spatialtransformer-e59  (E59, tanda 2 = grid[16:32])

kernel_sources: ["alexhuaracha/04-preprocessing", "alexhuaracha/06-baselines-stat",
                 "alexhuaracha/07-lstm-baseline",
                 "alexhuaracha/08a-spatialconvlstm-e2",
                 "alexhuaracha/08b-spatialconvlstm-e59"]

Architecture decisions applied (from design doc AD-10):
  - Mirrors build_notebook_08.py structure exactly.
  - Embeds models/spatial_transformer.py in addition to models/lstm.py and
    models/spatial_conv_lstm.py.
  - Uses TRANSFORMER_GRID instead of SPATIAL_GRID (32 configs, obs #417).
  - Cell IDs use cell-09-* prefix.
  - kernel_sources adds 08a/08b for SpatialConvLSTM results CSV comparison.
  - Model forward call passes (inp, ctx, input_mask) — spatial 3-arg dispatch (AD-1).
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

# ---------------------------------------------------------------------------
# Per-build state — reset at the start of each corridor build.
# ---------------------------------------------------------------------------

_cells: list = []
_nb = None


def _reset() -> None:
    """Reset global cell accumulator and notebook object for a fresh build."""
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
# Shared kernel metadata fields (identical for both corridors).
# ---------------------------------------------------------------------------

_KERNEL_SOURCES = [
    "alexhuaracha/04-preprocessing",
    "alexhuaracha/06-baselines-stat",
    "alexhuaracha/07-lstm-baseline",
    "alexhuaracha/08a-spatialconvlstm-e2",
    "alexhuaracha/08b-spatialconvlstm-e59",
]

_KERNEL_META_BASE = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "accelerator": "GPU_T4X2",
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": _KERNEL_SOURCES,
    "competition_sources": [],
}


# ---------------------------------------------------------------------------
# Module embed cells — dependency order (identical for every corridor).
# ---------------------------------------------------------------------------

def _add_embed_cells() -> None:
    """Add all module-embed cell pairs (same for both corridors)."""

    embed_module(
        "evaluation/splits.py",
        """## Module: evaluation/splits

Temporal split helper (`split_temporal`) and train-only p99 winsorization
(`winsorize_train_p99`).  Split date ranges are locked in spec §3.
""",
        cell_id_md="cell-09-embed-splits-md",
        cell_id_code="cell-09-embed-splits",
    )

    embed_module(
        "evaluation/metrics.py",
        """## Module: evaluation/metrics

`mae` and `rmse` in minutes.  Both accept polars Series or numpy arrays.
Null/NaN rows are dropped before aggregation.
""",
        cell_id_md="cell-09-embed-metrics-md",
        cell_id_code="cell-09-embed-metrics",
    )

    embed_module(
        "data/windowing.py",
        """## Module: data/windowing

`make_window_index` — per-slot deterministic window index.
`compute_max_N` — train-p99 of (n_buses - 1) per (empresaid, direction).
Constants: `DEFAULT_T_IN=12`, `DEFAULT_T_OUT=1`, `DEFAULT_STRIDE=1`.
""",
        cell_id_md="cell-09-embed-windowing-md",
        cell_id_code="cell-09-embed-windowing",
    )

    embed_module(
        "data/normalization.py",
        """## Module: data/normalization

`compute_normalization_stats` — per-direction z-score stats from TRAIN ONLY.
`apply_zscore` — add `delta_t_min_z` column; no clipping (DL-8).
""",
        cell_id_md="cell-09-embed-normalization-md",
        cell_id_code="cell-09-embed-normalization",
    )

    embed_module(
        "data/context_features.py",
        """## Module: data/context_features

`encode_context` — add 5 cyclical + atypical-flag columns.
`load_atypical_days` — graceful fallback to empty set when CSV absent (DL-2).
""",
        cell_id_md="cell-09-embed-context-md",
        cell_id_code="cell-09-embed-context",
    )

    embed_module(
        "data/dataset.py",
        """## Module: data/dataset  (first torch import)

`HeadwayDataset` — on-the-fly window materialization with masks (DL-11).
`collate_fn` — batch stacking for variable-N edge cases (REQ-6).
""",
        cell_id_md="cell-09-embed-dataset-md",
        cell_id_code="cell-09-embed-dataset",
    )

    embed_module(
        "models/lstm.py",
        """## Module: models/lstm

`HeadwayLSTM` — flat LSTM encoder (batch_first, last hidden state → Linear head).
`masked_mse_loss` — MSE over valid (mask==True) positions; clamp(min=1) prevents
zero-division on all-False masks.
""",
        cell_id_md="cell-09-embed-lstm-md",
        cell_id_code="cell-09-embed-lstm",
    )

    embed_module(
        "models/spatial_conv_lstm.py",
        """## Module: models/spatial_conv_lstm

`SpatialConvLSTM` — 1D spatial conv encoder + LSTM temporal encoder.
Conv1d(1, conv_channels, kernel_size=3, padding=1) applied per-timestep;
LSTM input_size = conv_channels * max_N + context_size.
Duck-type flag: `spatial = True` (AD-4 dispatch in train.py).
""",
        cell_id_md="cell-09-embed-spatial-conv-md",
        cell_id_code="cell-09-embed-spatial-conv",
    )

    embed_module(
        "models/spatial_transformer.py",
        """## Module: models/spatial_transformer

`SpatialTransformer` — multi-head self-attention spatial encoder + LSTM temporal encoder.
MHA(d_model, nhead, batch_first=True) applied per-timestep via B*T_in batched reshape.
proj_out: Linear(d_model, 1) collapses d_model→1 per position so LSTM input_size = max_N + 5.
Key mask polarity inversion: kpm = ~input_mask (AD-4).
NaN guard for fully-masked snapshots (AD-5).
Duck-type flag: `spatial = True` (AD-4 dispatch in train.py).
""",
        cell_id_md="cell-09-embed-spatial-tr-md",
        cell_id_code="cell-09-embed-spatial-tr",
    )

    embed_module(
        "train.py",
        """## Module: train

`TrainConfig`, `TrainResult` — hyperparameter and result dataclasses.
`nhead` and `d_model` fields added to TrainConfig for SpatialTransformer configs (AD-6).
`set_seed` — reproducibility seeds (torch + cuda + numpy).
`train_one_epoch`, `evaluate_epoch` — single-epoch train/eval loops with
duck-type dispatch: if `model.spatial`, calls `model(inp, ctx, input_mask)` (AD-4).
`EarlyStopping` — patience-based early stopping with best-state copy.
`TRANSFORMER_GRID` — 32 TrainConfig entries for the SpatialTransformer grid search (obs #417).
`train_model` — full training loop with early stopping.
`grid_search` — run all grid configs; return sorted by best_val_loss.
`save_checkpoint`, `load_checkpoint` — model persistence with transformer support (AD-8).
`denormalize_predictions` — z-score → minutes conversion.
""",
        cell_id_md="cell-09-embed-train-md",
        cell_id_code="cell-09-embed-train",
    )


# ---------------------------------------------------------------------------
# Per-corridor cell builders.
# ---------------------------------------------------------------------------

def _add_title_cell(corridor_label: str, empresa_id: int) -> None:
    md(
        f"""
# 09 — SpatialTransformer — Corredor {corridor_label}  (auto-generado por build_notebook_09.py)

Entrena un modelo `SpatialTransformer` (MHA espacial + LSTM temporal) sobre
el corredor **{corridor_label}** (empresa {empresa_id}) usando las ventanas deslizantes
construidas en Fase 3 (NB05).

Grid search sobre 32 configuraciones (`TRANSFORMER_GRID`): nhead ∈ {{1,2}}
× d_model ∈ {{16,32}} × hidden ∈ {{32,64}} × dropout ∈ {{0.0,0.2}} × lr ∈ {{1e-3,5e-4}})
con early stopping (num_layers=1 fijo).
Evalúa MAE/RMSE en test y compara con los resultados LSTM de NB07 y
SpatialConvLSTM de NB08.

Referencia: `docs/plan-de-desarrollo.md §7 Fase 6b — SpatialTransformer`.
""",
        cell_id="cell-09-title",
    )


def _add_setup_cell() -> None:
    code(
        """
import polars as pl
import numpy as np
from pathlib import Path

# Locate headways parquets under /kaggle/input or local dir.
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

def _find_baselines_csv() -> Path | None:
    name = "baselines_results.csv"
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(name))
        if candidates:
            return candidates[0]
    candidates = list(Path(".").rglob(name))
    if candidates:
        return candidates[0]
    return None

def _find_lstm_csv() -> Path | None:
    \"\"\"Locate lstm_results.csv from NB07 kernel output (kernel_source: alexhuaracha/07-lstm-baseline).\"\"\"
    name = "lstm_results.csv"
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(name))
        if candidates:
            return candidates[0]
    candidates = list(Path(".").rglob(name))
    if candidates:
        return candidates[0]
    return None

def _find_spatial_conv_lstm_csv() -> Path | None:
    \"\"\"Locate spatial_conv_lstm_results.csv from NB08 (kernel_source: 08a/08b).\"\"\"
    name = "spatial_conv_lstm_results.csv"
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(name))
        if candidates:
            return candidates[0]
    candidates = list(Path(".").rglob(name))
    if candidates:
        return candidates[0]
    return None

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)
TRANSFORMER_CSV_OUT = OUTPUT_DIR / "spatial_transformer_results.csv"

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"Output dir: {OUTPUT_DIR}")
print(f"Device:     {DEVICE}")
""",
        cell_id="cell-09-setup",
    )


def _add_load_cell(corridor_label: str, empresa_id: int) -> None:
    md(
        f"""## Cargar datos — {corridor_label}

Lee el parquet generado por NB04 (kernel_source: `alexhuaracha/04-preprocessing`).
Inyecta `empresaid` como columna literal para cumplir el contrato de slot key.
""",
        cell_id="cell-09-load-md",
    )
    code(
        f"""
df = pl.read_parquet(_find_parquet({empresa_id})).with_columns(
    pl.lit({empresa_id}, dtype=pl.Int64).alias("empresaid")
)
print(f"{corridor_label}: {{df.height:,}} rows, {{df.width}} cols")
""",
        cell_id="cell-09-load",
    )


def _add_split_cell(corridor_label: str) -> None:
    md(
        f"""## Split temporal + winsorización

Aplica `split_temporal` y `winsorize_train_p99` (INV-1, INV-6).
El umbral de winsorización se computa exclusivamente sobre el split `train`
(AC-WINSOR-1, AC-WINSOR-2 — leakage guard).
""",
        cell_id="cell-09-split-md",
    )
    code(
        f"""
train_df = df.filter(pl.col("split") == "train") if "split" in df.columns else df
df_split = split_temporal(df)
train_split = df_split.filter(pl.col("split") == "train")
df_winsor, threshold = winsorize_train_p99(train_split)
non_train = df_split.filter(pl.col("split") != "train")
df = pl.concat([df_winsor, non_train])
print(f"{corridor_label}: split counts = {{df_split.group_by('split').agg(pl.len()).sort('split')}}")
print(f"{corridor_label}: winsorize threshold = {{threshold:.4f}} min")
""",
        cell_id="cell-09-split",
    )


def _add_norm_cell(corridor_label: str) -> None:
    md(
        """## Normalización z-score (train only)

Computa estadísticas de normalización exclusivamente sobre filas de entrenamiento
(INV-2, AC-NORM-1) y aplica z-score a todos los splits.
""",
        cell_id="cell-09-norm-md",
    )
    code(
        f"""
train_only = df.filter(pl.col("split") == "train")
stats = compute_normalization_stats(train_only)
df = apply_zscore(df, stats)
print(f"\\n{corridor_label} normalization stats:")
for key in sorted(stats.means.keys()):
    print(f"  (empresa={{key[0]}}, dir={{key[1]}}): "
          f"mean={{stats.means[key]:.4f}}, std={{stats.stds[key]:.4f}}")
""",
        cell_id="cell-09-norm",
    )


def _add_context_cell(corridor_label: str) -> None:
    md(
        """## Features de contexto

Codificación cíclica de hora y día de semana + flag de día atípico (DL-2).
Fallback gracioso a `atypical_flag=0` si el CSV está ausente.
""",
        cell_id="cell-09-context-md",
    )
    code(
        f"""
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
print(f"Atypical days loaded: {{len(atypical_dates)}} dates (path={{atypical_path}})")

df = encode_context(df, atypical_dates=atypical_dates)
print(f"{corridor_label} context columns: {{[c for c in df.columns if c in CONTEXT_FEATURE_NAMES]}}")
""",
        cell_id="cell-09-context",
    )


def _add_dataset_cell(corridor_label: str) -> None:
    md(
        """## Construcción del Dataset

Calcula `max_N` (train-p99 de n_buses-1 por dirección), toma el máximo global
para dimensionar el modelo.  Materializa TODAS las ventanas en
tensores usando un lookup por timestamp (O(1) por acceso) en vez de filtros
Polars por ventana.  Train/val combinan ambas direcciones; test se separa
por dirección para desnormalizar con stats específicas.
""",
        cell_id="cell-09-dataset-md",
    )
    code(
        f"""
import torch
import time as _time

T_IN  = DEFAULT_T_IN   # 12
T_OUT = DEFAULT_T_OUT  # 1
BATCH_SIZE = 128

def _build_snapshot_lookup(df: pl.DataFrame, max_N: int, context_cols: list[str]):
    \"\"\"Build a dict: (empresaid, direction, timestamp) -> (values, mask, context).

    One pass through a Polars group_by — O(n_unique_snapshots).
    \"\"\"
    agg_exprs = [pl.col("pair_rank"), pl.col("delta_t_min_z")]
    for c in context_cols:
        agg_exprs.append(pl.col(c).first())

    grouped = df.group_by(["empresaid", "direction", "t"]).agg(agg_exprs)

    lookup = {{}}
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
    \"\"\"Build dict: (empresaid, direction, pair_rank) -> sorted list of timestamps.

    One Polars group_by — O(n_rows).
    \"\"\"
    grouped = (
        df.group_by(["empresaid", "direction", "pair_rank"])
        .agg(pl.col("t").sort())
    )
    slots = {{}}
    for row in grouped.iter_rows(named=True):
        key = (row["empresaid"], row["direction"], row["pair_rank"])
        slots[key] = row["t"]
    return slots

def fast_materialize(df: pl.DataFrame, window_index, max_N: int,
                     context_cols: list[str], label: str):
    \"\"\"Materialize all windows into batched tensor lists using dict lookups.

    Returns a list of batch dicts (same format as DataLoader output).
    \"\"\"
    t0 = _time.time()
    window_size = T_IN + T_OUT
    n_windows = len(window_index)

    lookup = _build_snapshot_lookup(df, max_N, context_cols)
    slots = _build_slot_timestamps(df)
    t_lookup = _time.time()
    print(f"  {{label}}: lookup built in {{t_lookup - t0:.1f}}s "
          f"({{len(lookup):,}} snapshots, {{len(slots):,}} slots)")

    all_input      = np.zeros((n_windows, T_IN, max_N), dtype=np.float32)
    all_target     = np.zeros((n_windows, T_OUT, max_N), dtype=np.float32)
    all_input_mask = np.zeros((n_windows, T_IN, max_N), dtype=np.bool_)
    all_target_mask= np.zeros((n_windows, T_OUT, max_N), dtype=np.bool_)
    all_context    = np.zeros((n_windows, T_IN, len(context_cols)), dtype=np.float32)

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
            else:
                out_idx = t_idx - T_IN
                all_target[i, out_idx] = vals
                all_target_mask[i, out_idx] = mask

    # Convert to tensors and split into batches.
    tensors = {{
        "input":       torch.from_numpy(all_input),
        "target":      torch.from_numpy(all_target),
        "input_mask":  torch.from_numpy(all_input_mask),
        "target_mask": torch.from_numpy(all_target_mask),
        "context":     torch.from_numpy(all_context),
    }}

    batches = []
    for start in range(0, n_windows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_windows)
        batch = {{k: v[start:end] for k, v in tensors.items()}}
        batches.append(batch)

    elapsed = _time.time() - t0
    print(f"  {{label}}: {{n_windows:,}} windows -> {{len(batches):,}} batches in {{elapsed:.1f}}s")
    return batches

CTX_COLS = list(CONTEXT_FEATURE_NAMES)

# Combined train/val (both directions).
train_split_df = df.filter(pl.col("split") == "train")
max_n_per_dir = compute_max_N(train_split_df, quantile=0.99)
max_N = max(max_n_per_dir.values())
print(f"\\n{corridor_label} max_N per direction: {{max_n_per_dir}}")
print(f"{corridor_label} global max_N (for model): {{max_N}}")

cached = {{}}
for split_name in ["train", "val"]:
    split_df = df.filter(pl.col("split") == split_name)
    idx = make_window_index(split_df, T_in=T_IN, T_out=T_OUT)
    cached[split_name] = fast_materialize(
        split_df, idx, max_N, CTX_COLS, f"{corridor_label} {{split_name}}")

# Per-direction test (for direction-specific denormalization).
cached_test = {{}}
for direction in [-1, 1]:
    test_df = df.filter(
        (pl.col("split") == "test") & (pl.col("direction") == direction)
    )
    idx = make_window_index(test_df, T_in=T_IN, T_out=T_OUT)
    cached_test[direction] = fast_materialize(
        test_df, idx, max_N, CTX_COLS, f"{corridor_label} test dir={{direction:+d}}")

print("\\nDataset construction complete.")
""",
        cell_id="cell-09-dataset",
    )


def _add_train_cell(
    corridor_label: str, grid_slice: tuple[int, int] | None = None
) -> None:
    if grid_slice is None:
        configs_expr = "TRANSFORMER_GRID"
        grid_desc = "× 32 configs del `TRANSFORMER_GRID`"
        tanda_note = ""
    else:
        start, end = grid_slice
        configs_expr = f"TRANSFORMER_GRID[{start}:{end}]"
        grid_desc = (
            f"× {end - start} configs del `TRANSFORMER_GRID[{start}:{end}]` (tanda)"
        )
        tanda_note = (
            f"\n\n**Tanda `[{start}:{end}]`** — esta corrida evalúa solo ese "
            "subconjunto del grid para no exceder el límite de 12h de sesión de "
            "Kaggle en el corredor grande (E59). El ganador global es el de menor "
            "`best val loss` entre las dos tandas, idéntico a `argmin` sobre el "
            "grid completo de 32 configs."
        )
    md(
        f"""## Grid search — entrenamiento SpatialTransformer

Entrena UN `SpatialTransformer` para {corridor_label} (ambas direcciones combinadas) {grid_desc}
(nhead ∈ {{1,2}} × d_model ∈ {{16,32}} × hidden ∈ {{32,64}}
× dropout ∈ {{0.0,0.2}} × lr ∈ {{1e-3,5e-4}}; num_layers=1 fijo).
Usa `train_model` con `EarlyStopping` (patience=10, max_epochs=50).

El dispatcher en `train_one_epoch`/`evaluate_epoch` detecta `model.spatial == True`
y pasa `(inp, ctx, input_mask)` en lugar de `cat([inp, ctx])` (AD-4).{tanda_note}
""",
        cell_id="cell-09-train-md",
    )
    code(
        f"""
print(f"\\n{corridor_label} transformer grid search: max_N={{max_N}}, "
      f"{{len({configs_expr})}} configs, device={{DEVICE}}")
results = grid_search(
    train_dl=cached["train"],
    val_dl=cached["val"],
    max_N=max_N,
    configs={configs_expr},
    device=DEVICE,
)
best = results[0]
print(f"  Best config:    nhead={{best.config.nhead}}, "
      f"d_model={{best.config.d_model}}, "
      f"hidden={{best.config.hidden_size}}, "
      f"dropout={{best.config.dropout}}, lr={{best.config.lr}}")
print(f"  Best val loss:  {{best.best_val_loss:.6f}} (epoch {{best.best_epoch}})")
print(f"  Epochs trained: {{best.epochs_trained}}")
""",
        cell_id="cell-09-train",
    )


def _add_evaluate_cell(corridor_label: str, empresa_id: int) -> None:
    md(
        """## Evaluación en test — MAE y RMSE por dirección

Evalúa el modelo sobre el split test, separando por dirección para desnormalizar
con la media/std específica de cada una.
La inferencia usa la firma espacial: `model(inp, ctx, input_mask)` (AD-1).
Computa MAE/RMSE por dirección; el agregado se obtiene juntando predicciones de ambas.
""",
        cell_id="cell-09-evaluate-md",
    )
    code(
        f"""
model = SpatialTransformer(
    max_N=max_N,
    nhead=best.config.nhead,
    d_model=best.config.d_model,
    hidden_size=best.config.hidden_size,
    output_size=max_N,
    num_layers=best.config.num_layers,
    dropout=best.config.dropout,
)
model.load_state_dict(best.state_dict)
model.eval()
model.to(torch.device(DEVICE))

dir_metrics = {{}}
dir_arrays  = {{}}

for direction in [-1, 1]:
    mean_val = stats.means[({empresa_id}, direction)]
    std_val  = stats.stds[({empresa_id}, direction)]

    all_preds   = []
    all_targets = []
    all_masks   = []

    with torch.no_grad():
        for batch in cached_test[direction]:
            inp        = batch["input"].to(DEVICE)
            ctx        = batch["context"].to(DEVICE)
            input_mask = batch["input_mask"].to(DEVICE)
            target     = batch["target"].to(DEVICE)
            mask       = batch["target_mask"].to(DEVICE)

            # SpatialTransformer forward: 3-arg spatial dispatch (AD-1, AD-4)
            pred = model(inp, ctx, input_mask)

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

    valid_preds   = preds_flat[masks_flat]
    valid_targets = targets_flat[masks_flat]

    mae_val  = mae(valid_targets,  valid_preds)
    rmse_val = rmse(valid_targets, valid_preds)
    print(f"{corridor_label} dir={{direction:+d}} SpatialTransformer: MAE={{mae_val:.4f}} min, "
          f"RMSE={{rmse_val:.4f}} min (n_valid={{masks_flat.sum():,}})")

    dir_metrics[direction] = (mae_val, rmse_val)
    dir_arrays[direction]  = (preds_flat, targets_flat, masks_flat)
""",
        cell_id="cell-09-evaluate",
    )


def _add_results_cell(corridor_label: str) -> None:
    md(
        f"""## Tabla de resultados — spatial_transformer_results.csv

Guarda los resultados del SpatialTransformer en formato long-form idéntico al schema
del harness de NB06/NB07/NB08 (columnas: corridor, direction, baseline, metric, value).
direction ∈ {{"-1", "+1", "aggregate"}} — 6 filas totales (3 dir × 2 métricas) para {corridor_label}.
""",
        cell_id="cell-09-results-md",
    )
    code(
        f"""
rows = []
for direction_int in [-1, 1]:
    direction_str = f"+{{direction_int}}" if direction_int > 0 else str(direction_int)
    mae_val, rmse_val = dir_metrics[direction_int]
    for metric_name, metric_val in [("MAE", mae_val), ("RMSE", rmse_val)]:
        rows.append({{
            "corridor":  "{corridor_label}",
            "direction": direction_str,
            "baseline":  "SpatialTransformer",
            "metric":    metric_name,
            "value":     float(metric_val),
        }})

# Aggregate: pool valid predictions from both directions.
all_preds   = np.concatenate([dir_arrays[d][0][dir_arrays[d][2]] for d in [-1, 1]])
all_targets = np.concatenate([dir_arrays[d][1][dir_arrays[d][2]] for d in [-1, 1]])
agg_mae  = mae(all_targets,  all_preds)
agg_rmse = rmse(all_targets, all_preds)
print(f"{corridor_label} aggregate SpatialTransformer: MAE={{agg_mae:.4f}} min, "
      f"RMSE={{agg_rmse:.4f}} min (n_valid={{len(all_preds):,}})")
for metric_name, metric_val in [("MAE", agg_mae), ("RMSE", agg_rmse)]:
    rows.append({{
        "corridor":  "{corridor_label}",
        "direction": "aggregate",
        "baseline":  "SpatialTransformer",
        "metric":    metric_name,
        "value":     float(metric_val),
    }})

transformer_results = pl.DataFrame(rows)
transformer_results.write_csv(TRANSFORMER_CSV_OUT)
print(f"\\nSpatialTransformer results written to: {{TRANSFORMER_CSV_OUT}}  ({{transformer_results.height}} rows)")
print(transformer_results)
""",
        cell_id="cell-09-results",
    )


def _add_compare_cell(corridor_label: str) -> None:
    md(
        f"""## Comparación con SpatialConvLSTM (NB08), LSTM (NB07) y baselines estadísticos (NB06)

Carga `spatial_conv_lstm_results.csv` generado por NB08
(kernel_sources: `alexhuaracha/08a-spatialconvlstm-e2` / `08b-spatialconvlstm-e59`),
`lstm_results.csv` de NB07 (kernel_source: `alexhuaracha/07-lstm-baseline`) y
`baselines_results.csv` de NB06 (kernel_source: `alexhuaracha/06-baselines-stat`)
para comparar con el SpatialTransformer en el corredor {corridor_label}.
Una tabla pivote facilita la lectura.
""",
        cell_id="cell-09-compare-md",
    )
    code(
        f"""
spatial_conv_csv = _find_spatial_conv_lstm_csv()
lstm_csv = _find_lstm_csv()
baselines_csv = _find_baselines_csv()

all_results = [transformer_results]

if spatial_conv_csv is not None:
    sc_df = pl.read_csv(spatial_conv_csv).filter(pl.col("corridor") == "{corridor_label}")
    print(f"SpatialConvLSTM results loaded from: {{spatial_conv_csv}} ({{sc_df.height}} rows for {corridor_label})")
    all_results.append(sc_df)
else:
    print("spatial_conv_lstm_results.csv not found — skipping SpatialConvLSTM comparison.")
    print("Ensure kernel_sources includes alexhuaracha/08a-spatialconvlstm-e2 and 08b-spatialconvlstm-e59.")

if lstm_csv is not None:
    lstm_df = pl.read_csv(lstm_csv).filter(pl.col("corridor") == "{corridor_label}")
    print(f"LSTM results loaded from: {{lstm_csv}} ({{lstm_df.height}} rows for {corridor_label})")
    all_results.append(lstm_df)
else:
    print("lstm_results.csv not found — skipping LSTM comparison.")
    print("Ensure kernel_sources includes alexhuaracha/07-lstm-baseline.")

if baselines_csv is not None:
    baselines_df = pl.read_csv(baselines_csv).filter(pl.col("corridor") == "{corridor_label}")
    print(f"Baselines loaded from: {{baselines_csv}} ({{baselines_df.height}} rows for {corridor_label})")
    all_results.append(baselines_df)
else:
    print("baselines_results.csv not found — skipping statistical baseline comparison.")
    print("Ensure kernel_sources includes alexhuaracha/06-baselines-stat.")

if len(all_results) > 1:
    comparison = pl.concat(all_results)
    # Filter to MAE for a clean comparison
    mae_comparison = comparison.filter(pl.col("metric") == "MAE")

    print("\\nMAE comparison (minutes) — lower is better:")
    print(mae_comparison.sort(["corridor", "direction", "baseline"]))

    # Wide pivot for quick human inspection
    wide = mae_comparison.pivot(
        on="baseline",
        index=["corridor", "direction", "metric"],
        values="value",
    )
    print("\\nPivot table (MAE, minutes):")
    print(wide.sort(["corridor", "direction", "metric"]))
else:
    print("\\nSpatialTransformer results summary (MAE):")
    for row in transformer_results.filter(pl.col("metric") == "MAE").iter_rows(named=True):
        print(f"  {corridor_label} dir={{row['direction']}}: MAE={{row['value']:.4f}} min")
""",
        cell_id="cell-09-compare",
    )


# ---------------------------------------------------------------------------
# Top-level build function — one call per corridor.
# ---------------------------------------------------------------------------

def build_corridor_notebook(
    *,
    corridor_label: str,
    empresa_id: int,
    out_dir: Path,
    notebook_filename: str,
    kernel_id: str,
    kernel_title: str,
    grid_slice: tuple[int, int] | None = None,
) -> None:
    """Build and write a single per-corridor notebook + kernel-metadata.json.

    grid_slice: optional (start, end) to run only TRANSFORMER_GRID[start:end] in this
    notebook. Used to split the larger corridor (E59) into two tandas so each Kaggle
    session stays under the 12h timeout. None (default) runs the full 32-config grid.
    """
    _reset()

    _add_title_cell(corridor_label, empresa_id)
    _add_setup_cell()
    _add_embed_cells()
    _add_load_cell(corridor_label, empresa_id)
    _add_split_cell(corridor_label)
    _add_norm_cell(corridor_label)
    _add_context_cell(corridor_label)
    _add_dataset_cell(corridor_label)
    _add_train_cell(corridor_label, grid_slice)
    _add_evaluate_cell(corridor_label, empresa_id)
    _add_results_cell(corridor_label)
    _add_compare_cell(corridor_label)

    # Write notebook.
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / notebook_filename
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

    # Write kernel-metadata.json.
    kernel_meta = {
        "id": kernel_id,
        "title": kernel_title,
        "code_file": notebook_filename,
        **_KERNEL_META_BASE,
    }
    meta_path = out_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8")
    print(f"Kernel metadata written: {meta_path}")


# ---------------------------------------------------------------------------
# Entry point — build per-corridor notebooks.
#   09a  : E2,  full 32-config grid (fits in one 12h Kaggle session).
#   09b1 : E59, tanda 1 = TRANSFORMER_GRID[0:16].
#   09b2 : E59, tanda 2 = TRANSFORMER_GRID[16:32].
# E59 is split into two tandas: the full 32-config grid exceeds Kaggle's 12h
# session limit on the larger corridor. The global best is the lower best_val_loss
# across both tandas (identical to argmin over the full grid).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_corridor_notebook(
        corridor_label="E2",
        empresa_id=2,
        out_dir=ROOT / "notebooks" / "09_spatial_transformer" / "09a_e2",
        notebook_filename="notebook.ipynb",
        kernel_id="alexhuaracha/09a-spatialtransformer-e2",
        kernel_title="09a — SpatialTransformer E2",
    )

    build_corridor_notebook(
        corridor_label="E59",
        empresa_id=59,
        out_dir=ROOT / "notebooks" / "09_spatial_transformer" / "09b1_e59",
        notebook_filename="notebook.ipynb",
        kernel_id="alexhuaracha/09b1-spatialtransformer-e59",
        kernel_title="09b1 — SpatialTransformer E59",
        grid_slice=(0, 16),
    )

    build_corridor_notebook(
        corridor_label="E59",
        empresa_id=59,
        out_dir=ROOT / "notebooks" / "09_spatial_transformer" / "09b2_e59",
        notebook_filename="notebook.ipynb",
        kernel_id="alexhuaracha/09b2-spatialtransformer-e59",
        kernel_title="09b2 — SpatialTransformer E59",
        grid_slice=(16, 32),
    )
