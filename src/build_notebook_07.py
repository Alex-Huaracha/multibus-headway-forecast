"""Generate notebooks/07_lstm/07_lstm.ipynb for Kaggle.

Inline-embed pattern (mirror of build_notebook_05.py and build_notebook_06.py):
  - Read each src/evaluation/*.py, src/data/*.py, src/models/*.py, and src/train.py
    source file via Path.read_text(), strip relative imports, and inject as a code cell.
  - This ensures the notebook is always a faithful flat copy of the modules
    at generation time.  Tests run against the modules directly so the
    notebook never diverges.
  - Stable cell IDs (cell-07-*) prevent git flutter on re-runs (AC-NB-2).

Output: notebooks/07_lstm/07_lstm.ipynb
Kaggle kernel: alexhuaracha/07-lstm
kernel_sources: ["alexhuaracha/04-preprocessing", "alexhuaracha/06-baselines-stat"]
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

OUT = ROOT / "notebooks" / "07_lstm" / "07_lstm.ipynb"
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
# Cell 1: Title (cell-07-title)
# ---------------------------------------------------------------------------

md(
    """
# 07 — LSTM Baseline  (auto-generado por build_notebook_07.py)

Entrena un modelo LSTM (`HeadwayLSTM`) sobre los corredores **E2** y **E59**
usando las ventanas deslizantes construidas en Fase 3 (NB05).

Grid search sobre 24 configuraciones (hidden ∈ {32,64,128} × layers ∈ {1,2}
× dropout ∈ {0.0,0.2} × lr ∈ {1e-3,5e-4}) con early stopping.
Evalúa MAE/RMSE en test y compara con los baselines estadísticos de NB06.

Referencia: `docs/plan-de-desarrollo.md §5 Fase 5 — LSTM Baseline`.
""",
    cell_id="cell-07-title",
)

# ---------------------------------------------------------------------------
# Cell 2: Setup — imports and helpers (cell-07-setup)
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

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)
LSTM_CSV_OUT = OUTPUT_DIR / "lstm_results.csv"

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"Output dir: {OUTPUT_DIR}")
print(f"Device:     {DEVICE}")
""",
    cell_id="cell-07-setup",
)

# ---------------------------------------------------------------------------
# Module embed cells — dependency order:
#   splits → metrics → windowing → normalization → context_features → dataset
#   → models/lstm → train
# ---------------------------------------------------------------------------

embed_module(
    "evaluation/splits.py",
    """## Module: evaluation/splits

Temporal split helper (`split_temporal`) and train-only p99 winsorization
(`winsorize_train_p99`).  Split date ranges are locked in spec §3.
""",
    cell_id_md="cell-07-embed-splits-md",
    cell_id_code="cell-07-embed-splits",
)

embed_module(
    "evaluation/metrics.py",
    """## Module: evaluation/metrics

`mae` and `rmse` in minutes.  Both accept polars Series or numpy arrays.
Null/NaN rows are dropped before aggregation.
""",
    cell_id_md="cell-07-embed-metrics-md",
    cell_id_code="cell-07-embed-metrics",
)

embed_module(
    "data/windowing.py",
    """## Module: data/windowing

`make_window_index` — per-slot deterministic window index.
`compute_max_N` — train-p99 of (n_buses - 1) per (empresaid, direction).
Constants: `DEFAULT_T_IN=12`, `DEFAULT_T_OUT=1`, `DEFAULT_STRIDE=1`.
""",
    cell_id_md="cell-07-embed-windowing-md",
    cell_id_code="cell-07-embed-windowing",
)

embed_module(
    "data/normalization.py",
    """## Module: data/normalization

`compute_normalization_stats` — per-direction z-score stats from TRAIN ONLY.
`apply_zscore` — add `delta_t_min_z` column; no clipping (DL-8).
""",
    cell_id_md="cell-07-embed-normalization-md",
    cell_id_code="cell-07-embed-normalization",
)

embed_module(
    "data/context_features.py",
    """## Module: data/context_features

`encode_context` — add 5 cyclical + atypical-flag columns.
`load_atypical_days` — graceful fallback to empty set when CSV absent (DL-2).
""",
    cell_id_md="cell-07-embed-context-md",
    cell_id_code="cell-07-embed-context",
)

embed_module(
    "data/dataset.py",
    """## Module: data/dataset  (first torch import)

`HeadwayDataset` — on-the-fly window materialization with masks (DL-11).
`collate_fn` — batch stacking for variable-N edge cases (REQ-6).
""",
    cell_id_md="cell-07-embed-dataset-md",
    cell_id_code="cell-07-embed-dataset",
)

embed_module(
    "models/lstm.py",
    """## Module: models/lstm

`HeadwayLSTM` — flat LSTM encoder (batch_first, last hidden state → Linear head).
`masked_mse_loss` — MSE over valid (mask==True) positions; clamp(min=1) prevents
zero-division on all-False masks.
""",
    cell_id_md="cell-07-embed-lstm-md",
    cell_id_code="cell-07-embed-lstm",
)

embed_module(
    "train.py",
    """## Module: train

`TrainConfig`, `TrainResult` — hyperparameter and result dataclasses.
`set_seed` — reproducibility seeds (torch + cuda + numpy).
`train_one_epoch`, `evaluate_epoch` — single-epoch train/eval loops.
`EarlyStopping` — patience-based early stopping with best-state copy.
`GRID` — 24 TrainConfig entries for the hyperparameter grid search.
`train_model` — full training loop with early stopping.
`grid_search` — run all GRID configs; return sorted by best_val_loss.
`save_checkpoint`, `load_checkpoint` — model persistence.
`denormalize_predictions` — z-score → minutes conversion.
""",
    cell_id_md="cell-07-embed-train-md",
    cell_id_code="cell-07-embed-train",
)

# ---------------------------------------------------------------------------
# Cell: Load data — E2 and E59 (cell-07-load)
# ---------------------------------------------------------------------------

md(
    """## Cargar datos — E2 y E59

Lee los parquets generados por NB04 (kernel_source: `alexhuaracha/04-preprocessing`).
Inyecta `empresaid` como columna literal para cumplir el contrato de slot key.
""",
    cell_id="cell-07-load-md",
)

code(
    """
hw_e2  = pl.read_parquet(_find_parquet(2)).with_columns(pl.lit(2,  dtype=pl.Int64).alias("empresaid"))
hw_e59 = pl.read_parquet(_find_parquet(59)).with_columns(pl.lit(59, dtype=pl.Int64).alias("empresaid"))

print(f"E2:  {hw_e2.height:,} rows, {hw_e2.width} cols")
print(f"E59: {hw_e59.height:,} rows, {hw_e59.width} cols")
""",
    cell_id="cell-07-load",
)

# ---------------------------------------------------------------------------
# Cell: Temporal split + winsorize (cell-07-split)
# ---------------------------------------------------------------------------

md(
    """## Split temporal + winsorización

Aplica `split_temporal` y `winsorize_train_p99` (INV-1, INV-6).
El umbral de winsorización se computa exclusivamente sobre el split `train`
(AC-WINSOR-1, AC-WINSOR-2 — leakage guard).
""",
    cell_id="cell-07-split-md",
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
    cell_id="cell-07-split",
)

# ---------------------------------------------------------------------------
# Cell: Normalization stats + z-score (cell-07-norm)
# ---------------------------------------------------------------------------

md(
    """## Normalización z-score (train only)

Computa estadísticas de normalización exclusivamente sobre filas de entrenamiento
(INV-2, AC-NORM-1) y aplica z-score a todos los splits.
""",
    cell_id="cell-07-norm-md",
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
    cell_id="cell-07-norm",
)

# ---------------------------------------------------------------------------
# Cell: Context features (cell-07-context)
# ---------------------------------------------------------------------------

md(
    """## Features de contexto

Codificación cíclica de hora y día de semana + flag de día atípico (DL-2).
Fallback gracioso a `atypical_flag=0` si el CSV está ausente.
""",
    cell_id="cell-07-context-md",
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
    cell_id="cell-07-context",
)

# ---------------------------------------------------------------------------
# Cell: Dataset construction — max_N + window index + HeadwayDataset (cell-07-dataset)
# ---------------------------------------------------------------------------

md(
    """## Construcción del Dataset — por corredor

Calcula `max_N` (train-p99 de n_buses-1 por dirección), toma el máximo global
por corredor para dimensionar el modelo.  Materializa TODAS las ventanas en
tensores usando un lookup por timestamp (O(1) por acceso) en vez de filtros
Polars por ventana.  Train/val combinan ambas direcciones; test se separa
por dirección para desnormalizar con stats específicas.
""",
    cell_id="cell-07-dataset-md",
)

code(
    """
import torch
import time as _time

T_IN  = DEFAULT_T_IN   # 12
T_OUT = DEFAULT_T_OUT  # 1
BATCH_SIZE = 32

def _build_snapshot_lookup(df: pl.DataFrame, max_N: int, context_cols: list[str]):
    \"\"\"Build a dict: (empresaid, direction, timestamp) -> (values, mask, context).

    One pass through a Polars group_by — O(n_unique_snapshots).
    \"\"\"
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
    \"\"\"Build dict: (empresaid, direction, pair_rank) -> sorted list of timestamps.

    One Polars group_by — O(n_rows).
    \"\"\"
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
    \"\"\"Materialize all windows into batched tensor lists using dict lookups.

    Returns a list of batch dicts (same format as DataLoader output).
    \"\"\"
    t0 = _time.time()
    window_size = T_IN + T_OUT
    n_windows = len(window_index)

    lookup = _build_snapshot_lookup(df, max_N, context_cols)
    slots = _build_slot_timestamps(df)
    t_lookup = _time.time()
    print(f"  {label}: lookup built in {t_lookup - t0:.1f}s "
          f"({len(lookup):,} snapshots, {len(slots):,} slots)")

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
    \"\"\"Build cached batches for one corridor (both directions combined).\"\"\"
    train_df = df.filter(pl.col("split") == "train")
    max_n = compute_max_N(train_df, quantile=0.99)
    global_max_N = max(max_n.values())
    print(f"\\n{label} max_N per direction: {max_n}")
    print(f"{label} global max_N (for model): {global_max_N}")

    # Combined train/val (both directions).
    cached = {}
    for split_name in ["train", "val"]:
        split_df = df.filter(pl.col("split") == split_name)
        idx = make_window_index(split_df, T_in=T_IN, T_out=T_OUT)
        cached[split_name] = fast_materialize(
            split_df, idx, global_max_N, CTX_COLS, f"{label} {split_name}")

    # Per-direction test (for direction-specific denormalization).
    cached_test = {}
    for direction in [-1, 1]:
        test_df = df.filter(
            (pl.col("split") == "test") & (pl.col("direction") == direction)
        )
        idx = make_window_index(test_df, T_in=T_IN, T_out=T_OUT)
        cached_test[direction] = fast_materialize(
            test_df, idx, global_max_N, CTX_COLS, f"{label} test dir={direction:+d}")

    return global_max_N, cached, cached_test

max_N_e2,  cached_e2,  cached_test_e2  = build_corridor_data(df_e2,  "E2")
max_N_e59, cached_e59, cached_test_e59 = build_corridor_data(df_e59, "E59")
print("\\nDataset construction complete.")
""",
    cell_id="cell-07-dataset",
)

# ---------------------------------------------------------------------------
# Cell: Grid search training (cell-07-train)
# ---------------------------------------------------------------------------

md(
    """## Grid search — entrenamiento LSTM por corredor

Entrena UN `HeadwayLSTM` por corredor (ambas direcciones combinadas) × 24 configs
del `GRID` (hidden ∈ {32,64,128} × layers ∈ {1,2} × dropout ∈ {0.0,0.2} × lr ∈ {1e-3,5e-4}).
Usa `train_model` con `EarlyStopping` (patience=10, max_epochs=50).
Produce 2 modelos: uno para E2, uno para E59.
""",
    cell_id="cell-07-train-md",
)

code(
    """
def run_corridor_grid_search(loaders: dict, max_N: int, label: str):
    \"\"\"Grid search for one corridor (both directions combined).\"\"\"
    print(f"\\n{label} grid search: max_N={max_N}, "
          f"{len(GRID)} configs, device={DEVICE}")
    results = grid_search(
        train_dl=loaders["train"],
        val_dl=loaders["val"],
        max_N=max_N,
        configs=GRID,
        device=DEVICE,
    )
    best = results[0]
    print(f"  Best config:    hidden={best.config.hidden_size}, "
          f"layers={best.config.num_layers}, "
          f"dropout={best.config.dropout}, lr={best.config.lr}")
    print(f"  Best val loss:  {best.best_val_loss:.6f} (epoch {best.best_epoch})")
    print(f"  Epochs trained: {best.epochs_trained}")
    return results

results_e2  = run_corridor_grid_search(cached_e2,  max_N_e2,  "E2")
results_e59 = run_corridor_grid_search(cached_e59, max_N_e59, "E59")
""",
    cell_id="cell-07-train",
)

# ---------------------------------------------------------------------------
# Cell: Evaluation — denormalize and compute MAE/RMSE (cell-07-evaluate)
# ---------------------------------------------------------------------------

md(
    """## Evaluación en test — MAE y RMSE por dirección

Evalúa el modelo de cada corredor sobre el split test, separando por dirección
para desnormalizar con la media/std específica de cada una.
Computa MAE/RMSE por dirección; el agregado se obtiene juntando predicciones de ambas.
""",
    cell_id="cell-07-evaluate-md",
)

code(
    """
def evaluate_corridor_model(best_result, test_loaders, max_N, stats, label):
    \"\"\"Evaluate one corridor model on per-direction test splits.

    Uses direction-specific mean/std from NormalizationStats for denormalization.
    Returns (dir_metrics, dir_arrays) for downstream result building.
    \"\"\"
    empresa_id = int(list(stats.means.keys())[0][0])

    model = HeadwayLSTM(
        input_size=max_N + CONTEXT_DIM,
        hidden_size=best_result.config.hidden_size,
        output_size=max_N,
        num_layers=best_result.config.num_layers,
        dropout=best_result.config.dropout,
    )
    model.load_state_dict(best_result.state_dict)
    model.eval()
    model.to(torch.device(DEVICE))

    dir_metrics = {}
    dir_arrays  = {}

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

        valid_preds   = preds_flat[masks_flat]
        valid_targets = targets_flat[masks_flat]

        mae_val  = mae(valid_targets,  valid_preds)
        rmse_val = rmse(valid_targets, valid_preds)
        print(f"{label} dir={direction:+d} LSTM: MAE={mae_val:.4f} min, RMSE={rmse_val:.4f} min "
              f"(n_valid={masks_flat.sum():,})")

        dir_metrics[direction] = (mae_val, rmse_val)
        dir_arrays[direction]  = (preds_flat, targets_flat, masks_flat)

    return dir_metrics, dir_arrays

dir_metrics_e2,  dir_arrays_e2  = evaluate_corridor_model(
    results_e2[0], cached_test_e2, max_N_e2, stats_e2, "E2")
dir_metrics_e59, dir_arrays_e59 = evaluate_corridor_model(
    results_e59[0], cached_test_e59, max_N_e59, stats_e59, "E59")
""",
    cell_id="cell-07-evaluate",
)

# ---------------------------------------------------------------------------
# Cell: Results table — save as CSV (cell-07-results)
# ---------------------------------------------------------------------------

md(
    """## Tabla de resultados — lstm_results.csv

Guarda los resultados del LSTM en formato long-form idéntico al schema del harness de NB06
(columnas: corridor, direction, baseline, metric, value).
direction ∈ {\"-1\", \"+1\", \"aggregate\"} — 12 filas totales (3 dir × 2 métricas × 2 corredores).
""",
    cell_id="cell-07-results-md",
)

code(
    """
def build_lstm_rows(corridor, dir_metrics, dir_arrays):
    \"\"\"Build long-form rows matching harness schema.

    direction values: \"-1\", \"+1\", \"aggregate\"
    baseline: \"LSTM\"
    metric:   \"MAE\", \"RMSE\"

    Produces 6 rows per corridor (3 directions × 2 metrics).
    \"\"\"
    rows = []
    for direction_int in [-1, 1]:
        direction_str = f"+{direction_int}" if direction_int > 0 else str(direction_int)
        mae_val, rmse_val = dir_metrics[direction_int]
        for metric_name, metric_val in [("MAE", mae_val), ("RMSE", rmse_val)]:
            rows.append({
                "corridor":  corridor,
                "direction": direction_str,
                "baseline":  "LSTM",
                "metric":    metric_name,
                "value":     float(metric_val),
            })

    # Aggregate: pool valid predictions from both directions.
    all_preds   = np.concatenate([dir_arrays[d][0][dir_arrays[d][2]] for d in [-1, 1]])
    all_targets = np.concatenate([dir_arrays[d][1][dir_arrays[d][2]] for d in [-1, 1]])
    agg_mae  = mae(all_targets,  all_preds)
    agg_rmse = rmse(all_targets, all_preds)
    print(f"{corridor} aggregate LSTM: MAE={agg_mae:.4f} min, RMSE={agg_rmse:.4f} min "
          f"(n_valid={len(all_preds):,})")
    for metric_name, metric_val in [("MAE", agg_mae), ("RMSE", agg_rmse)]:
        rows.append({
            "corridor":  corridor,
            "direction": "aggregate",
            "baseline":  "LSTM",
            "metric":    metric_name,
            "value":     float(metric_val),
        })
    return rows

lstm_rows = (
    build_lstm_rows("E2",  dir_metrics_e2,  dir_arrays_e2)
    + build_lstm_rows("E59", dir_metrics_e59, dir_arrays_e59)
)

lstm_results = pl.DataFrame(lstm_rows)
lstm_results.write_csv(LSTM_CSV_OUT)
print(f"\\nLSTM results written to: {LSTM_CSV_OUT}  ({lstm_results.height} rows)")
print(lstm_results)
""",
    cell_id="cell-07-results",
)

# ---------------------------------------------------------------------------
# Cell: Comparison with baselines (cell-07-compare)
# ---------------------------------------------------------------------------

md(
    """## Comparación con baselines estadísticos (NB06)

Carga `baselines_results.csv` generado por NB06
(kernel_source: `alexhuaracha/06-baselines-stat`) y compara con los resultados
del LSTM. Una tabla pivote facilita la lectura.
""",
    cell_id="cell-07-compare-md",
)

code(
    """
baselines_csv = _find_baselines_csv()
if baselines_csv is not None:
    baselines = pl.read_csv(baselines_csv)
    print(f"Baselines loaded from: {baselines_csv} ({baselines.height} rows)")

    # Filter to MAE for a clean comparison
    comparison = pl.concat([
        baselines.filter(pl.col("metric") == "MAE"),
        lstm_results.filter(pl.col("metric") == "MAE"),
    ])

    print("\\nMAE comparison (minutes) — lower is better:")
    print(comparison.sort(["corridor", "direction", "baseline"]))

    # Wide pivot for quick human inspection
    wide = comparison.pivot(
        on="baseline",
        index=["corridor", "direction", "metric"],
        values="value",
    )
    print("\\nPivot table:")
    print(wide.sort(["corridor", "direction", "metric"]))
else:
    print("baselines_results.csv not found — skipping comparison.")
    print("Ensure kernel_sources includes alexhuaracha/06-baselines-stat.")
    print("LSTM results summary:")
    for row in lstm_results.filter(pl.col("metric") == "MAE").iter_rows(named=True):
        print(f"  {row['corridor']} dir={row['direction']}: MAE={row['value']:.4f} min")
""",
    cell_id="cell-07-compare",
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
