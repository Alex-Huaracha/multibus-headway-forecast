"""Generate notebooks/19_e4_transformer/h{H}/19_e4_transformer_h{H}.ipynb for Kaggle.

Produces 4 notebooks — one per horizon h ∈ {1, 3, 5, 10}:
  19_e4_transformer_h1.ipynb   → alexhuaracha/19-e4-transformer-h1
  19_e4_transformer_h3.ipynb   → alexhuaracha/19-e4-transformer-h3
  19_e4_transformer_h5.ipynb   → alexhuaracha/19-e4-transformer-h5
  19_e4_transformer_h10.ipynb  → alexhuaracha/19-e4-transformer-h10

E4-ONLY external-validity replication of NB13 (SpatialTransformer multi-horizonte).
Each notebook trains a `SpatialTransformer` (MHA espacial + LSTM temporal) on the
THIRD corridor **E4** only, using sliding windows with the DIRECT multi-horizon
target alignment (single timestep at +h after the last input; T_OUT=1).

Key differences from build_notebook_13.py (E2+E59 SpatialTransformer notebook):
  - E4-ONLY: loads only headways_E4 via _find_parquet(4); prepares, normalizes,
    encodes, builds, trains and evaluates a single corridor (E4). No E2/E59.
  - REUSED config: a single TrainConfig (NB13's E2 winning config) — no grid.
  - Output CSVs are E4-scoped: spatial_transformer_E4_results_h{H}.csv,
    spatial_transformer_E4_residuals_h{H}.csv.
  - Cell IDs use cell-19-* prefix.

Reuses every library function unchanged via the same inline-embed mechanism.

kernel_sources: ["alexhuaracha/16-e4-data-baselines"]
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
# Kaggle kernel metadata — written alongside each per-horizon notebook.
# ---------------------------------------------------------------------------

_KERNEL_META_BASE = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "accelerator": "GPU_T4X2",
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [
        "alexhuaracha/16-e4-data-baselines",
    ],
    "competition_sources": [],
}

# ---------------------------------------------------------------------------
# Per-build state — reset at the start of each horizon build.
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
# Module embed cells — dependency order (identical for every horizon).
# splits → metrics → windowing → normalization → context_features → dataset
# → models/lstm → models/spatial_conv_lstm → models/spatial_transformer → train
# ---------------------------------------------------------------------------

def _add_embed_cells() -> None:
    """Add all module-embed cell pairs (same for every horizon)."""

    embed_module(
        "evaluation/splits.py",
        """## Module: evaluation/splits

Temporal split helper (`split_temporal`) and train-only p99 winsorization
(`winsorize_train_p99`).  Split date ranges are locked in spec §3.
""",
        cell_id_md="cell-19-embed-splits-md",
        cell_id_code="cell-19-embed-splits",
    )

    embed_module(
        "evaluation/metrics.py",
        """## Module: evaluation/metrics

`mae` and `rmse` in minutes.  Both accept polars Series or numpy arrays.
Null/NaN rows are dropped before aggregation.
""",
        cell_id_md="cell-19-embed-metrics-md",
        cell_id_code="cell-19-embed-metrics",
    )

    embed_module(
        "data/windowing.py",
        """## Module: data/windowing

`make_window_index` — per-slot deterministic window index.
`compute_max_N` — train-p99 of (n_buses - 1) per (empresaid, direction).
Constants: `DEFAULT_T_IN=12`, `DEFAULT_T_OUT=1`, `DEFAULT_STRIDE=1`.
""",
        cell_id_md="cell-19-embed-windowing-md",
        cell_id_code="cell-19-embed-windowing",
    )

    embed_module(
        "data/normalization.py",
        """## Module: data/normalization

`compute_normalization_stats` — per-direction z-score stats from TRAIN ONLY.
`apply_zscore` — add `delta_t_min_z` column; no clipping (DL-8).
""",
        cell_id_md="cell-19-embed-normalization-md",
        cell_id_code="cell-19-embed-normalization",
    )

    embed_module(
        "data/context_features.py",
        """## Module: data/context_features

`encode_context` — add 5 cyclical + atypical-flag columns.
`load_atypical_days` — graceful fallback to empty set when CSV absent (DL-2).
""",
        cell_id_md="cell-19-embed-context-md",
        cell_id_code="cell-19-embed-context",
    )

    embed_module(
        "data/dataset.py",
        """## Module: data/dataset  (first torch import)

`HeadwayDataset` — on-the-fly window materialization with masks (DL-11).
`collate_fn` — batch stacking for variable-N edge cases (REQ-6).
""",
        cell_id_md="cell-19-embed-dataset-md",
        cell_id_code="cell-19-embed-dataset",
    )

    embed_module(
        "models/lstm.py",
        """## Module: models/lstm

`HeadwayLSTM` — flat LSTM encoder (batch_first, last hidden state → Linear head).
`masked_mse_loss` — MSE over valid (mask==True) positions; clamp(min=1) prevents
zero-division on all-False masks.
""",
        cell_id_md="cell-19-embed-lstm-md",
        cell_id_code="cell-19-embed-lstm",
    )

    embed_module(
        "models/spatial_conv_lstm.py",
        """## Module: models/spatial_conv_lstm

`SpatialConvLSTM` — 1D spatial conv encoder + LSTM temporal encoder.
Conv1d(1, conv_channels, kernel_size=3, padding=1) applied per-timestep;
LSTM input_size = conv_channels * max_N + context_size.
Duck-type flag: `spatial = True` (AD-4 dispatch in train.py).
""",
        cell_id_md="cell-19-embed-spatial-conv-md",
        cell_id_code="cell-19-embed-spatial-conv",
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
        cell_id_md="cell-19-embed-spatial-tr-md",
        cell_id_code="cell-19-embed-spatial-tr",
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
`TRANSFORMER_GRID` — 32 TrainConfig entries (kept for reference; NB19 uses one config only).
`train_model` — full training loop with early stopping.
`grid_search` — run configs; return sorted by best_val_loss.
`save_checkpoint`, `load_checkpoint` — model persistence.
`denormalize_predictions` — z-score → minutes conversion.
""",
        cell_id_md="cell-19-embed-train-md",
        cell_id_code="cell-19-embed-train",
    )


# ---------------------------------------------------------------------------
# Per-horizon cell builders.
# ---------------------------------------------------------------------------

def _add_title_cell(horizon: int) -> None:
    md(
        f"""
# 19 — SpatialTransformer E4 (validez externa) h={horizon}  (auto-generado por build_notebook_19_e4_transformer.py)

Entrena un modelo `SpatialTransformer` (MHA espacial + LSTM temporal) sobre el
TERCER corredor **E4** usando ventanas deslizantes con horizonte directo
**h={horizon} minuto(s)**. Valida E4 como tercer corredor (validez externa).

Reusa la configuración ganadora del primer corredor (de NB13) como config única
para E4 — una sola corrida, sin grid. Evalúa MAE/RMSE en test y compara con los
baselines de E4 (NB16).

Referencia: `docs/plan-de-desarrollo.md §6.5 Fase 6.5 — Multi-Horizonte`.
""",
        cell_id="cell-19-title",
    )


def _add_setup_cell(horizon: int) -> None:
    code(
        f"""
import polars as pl
import numpy as np
from pathlib import Path

# Locate headways parquet for E4 under /kaggle/input or local dir.
def _find_parquet(empresa_id: int) -> Path:
    name = f"headways_E{{empresa_id}}.parquet"
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(name))
        if candidates:
            return candidates[0]
    candidates = list(Path(".").rglob(name))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"{{name}} not found. Expected at /kaggle/input/**/{{name}}"
    )

def _find_baselines_csv() -> Path | None:
    name = "baselines_results_multih.csv"
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
HORIZON = {horizon}
TRANSFORMER_CSV_OUT = OUTPUT_DIR / f"spatial_transformer_E4_results_h{{HORIZON}}.csv"

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"Output dir: {{OUTPUT_DIR}}")
print(f"Horizon:    {{HORIZON}}")
print(f"Device:     {{DEVICE}}")
""",
        cell_id="cell-19-setup",
    )


def _add_load_cell() -> None:
    md(
        """## Cargar datos — E4 (tercer corredor)

Lee el parquet de E4 generado por NB16 (kernel_source: `alexhuaracha/16-e4-data-baselines`).
Inyecta `empresaid` como columna literal para cumplir el contrato de slot key.
""",
        cell_id="cell-19-load-md",
    )
    code(
        """
hw_e4 = pl.read_parquet(_find_parquet(4)).with_columns(pl.lit(4, dtype=pl.Int64).alias("empresaid"))

print(f"E4: {hw_e4.height:,} rows, {hw_e4.width} cols")
""",
        cell_id="cell-19-load",
    )


def _add_split_cell() -> None:
    md(
        """## Temporal split + winsorization

Applies `split_temporal` and `winsorize_train_p99` (INV-1, INV-6).
The winsorization threshold is computed only from the `train` split and then
applied to the full split frame (AC-WINSOR-1, AC-WINSOR-2 — leakage guard).
""",
        cell_id="cell-19-split-md",
    )
    code(
        """
def prepare_corridor(hw: pl.DataFrame, label: str) -> pl.DataFrame:
    df_split = split_temporal(hw)
    df_winsor, threshold = winsorize_train_p99(df_split)
    print(f"{label}: split counts = {df_split.group_by('split').agg(pl.len()).sort('split')}")
    print(f"{label}: winsorize threshold = {threshold:.4f} min")
    return df_winsor

df_e4 = prepare_corridor(hw_e4, "E4")
""",
        cell_id="cell-19-split",
    )


def _add_norm_cell() -> None:
    md(
        """## Normalización z-score (train only)

Computa estadísticas de normalización exclusivamente sobre filas de entrenamiento
(INV-2, AC-NORM-1) y aplica z-score a todos los splits.
""",
        cell_id="cell-19-norm-md",
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

df_e4, stats_e4 = normalize_corridor(df_e4, "E4")
""",
        cell_id="cell-19-norm",
    )


def _add_context_cell() -> None:
    md(
        """## Features de contexto

Codificación cíclica de hora y día de semana + flag de día atípico (DL-2).
Fallback gracioso a `atypical_flag=0` si el CSV está ausente.
""",
        cell_id="cell-19-context-md",
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

df_e4 = encode_context(df_e4, atypical_dates=atypical_dates)
print(f"E4 context columns: {[c for c in df_e4.columns if c in CONTEXT_FEATURE_NAMES]}")
""",
        cell_id="cell-19-context",
    )


def _add_dataset_cell(horizon: int) -> None:
    md(
        f"""## Construcción del Dataset — horizonte h={horizon}

Calcula `max_N` (train-p99 de n_buses-1 por dirección), toma el máximo global
por corredor para dimensionar el modelo.

**Adaptación multi-horizonte (DIRECT)**: `window_size = T_IN + HORIZON`.
El loop en `fast_materialize` sólo asigna target en el último paso de la ventana
(`t_idx == window_size - 1`); los pasos intermedios entre input y target se saltan.
`make_window_index` recibe `horizon=HORIZON` para que el guard de slot vacío
rechace ventanas cuyo target +h no existe.
""",
        cell_id="cell-19-dataset-md",
    )
    code(
        f"""
import torch
import time as _time

T_IN  = DEFAULT_T_IN   # 12
T_OUT = 1              # always 1 output row (DIRECT horizon — single target step)
HORIZON = {horizon}
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

    DIRECT multi-horizon: window_size = T_IN + HORIZON.
    The loop fills:
      - t_idx < T_IN           → input row
      - t_idx == window_size-1 → the single target row at +HORIZON
      - else (intermediate)    → skipped (not input, not target)

    Returns a list of batch dicts (same format as DataLoader output).
    \"\"\"
    t0 = _time.time()
    window_size = T_IN + HORIZON
    n_windows = len(window_index)

    lookup = _build_snapshot_lookup(df, max_N, context_cols)
    slots = _build_slot_timestamps(df)
    t_lookup = _time.time()
    print(f"  {{label}}: lookup built in {{t_lookup - t0:.1f}}s "
          f"({{len(lookup):,}} snapshots, {{len(slots):,}} slots)")

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

def build_corridor_data(df: pl.DataFrame, label: str):
    \"\"\"Build cached batches for one corridor (both directions combined).\"\"\"
    train_df = df.filter(pl.col("split") == "train")
    max_n = compute_max_N(train_df, quantile=0.99)
    global_max_N = max(max_n.values())
    print(f"\\n{{label}} max_N per direction: {{max_n}}")
    print(f"{{label}} global max_N (for model): {{global_max_N}}")

    # Combined train/val (both directions).
    cached = {{}}
    for split_name in ["train", "val"]:
        split_df = df.filter(pl.col("split") == split_name)
        idx = make_window_index(split_df, T_in=T_IN, horizon=HORIZON)
        cached[split_name] = fast_materialize(
            split_df, idx, global_max_N, CTX_COLS, f"{{label}} {{split_name}}")

    # Per-direction test (for direction-specific denormalization).
    cached_test = {{}}
    for direction in [-1, 1]:
        test_df = df.filter(
            (pl.col("split") == "test") & (pl.col("direction") == direction)
        )
        idx = make_window_index(test_df, T_in=T_IN, horizon=HORIZON)
        cached_test[direction] = fast_materialize(
            test_df, idx, global_max_N, CTX_COLS, f"{{label}} test dir={{direction:+d}}")

    return global_max_N, cached, cached_test

max_N_e4, cached_e4, cached_test_e4 = build_corridor_data(df_e4, "E4")
print("\\nDataset construction complete.")
""",
        cell_id="cell-19-dataset",
    )


def _add_train_cell() -> None:
    md(
        """## Entrenamiento SpatialTransformer E4 — config reusada

Reusa la configuración ganadora del primer corredor (de NB13) como config única
para E4. Una sola corrida, sin grid. El dispatcher en `train_one_epoch`/`evaluate_epoch`
detecta `model.spatial == True` y pasa `(inp, ctx, input_mask)` (AD-4).
""",
        cell_id="cell-19-train-md",
    )
    code(
        """
# Reused config: NB13 first-corridor winning SpatialTransformer config applied to E4.
E4_CONFIG = TrainConfig(nhead=1, d_model=16, hidden_size=64, num_layers=1, dropout=0.0, lr=5e-4)

def run_corridor_single(loaders: dict, max_N: int, label: str):
    \"\"\"Train with the reused winning config for E4.\"\"\"
    print(f"\\n{label} SpatialTransformer training: max_N={max_N}, device={DEVICE}")
    results = grid_search(
        train_dl=loaders["train"],
        val_dl=loaders["val"],
        max_N=max_N,
        configs=[E4_CONFIG],
        device=DEVICE,
    )
    best = results[0]
    print(f"  {label} config: nhead={best.config.nhead}, d_model={best.config.d_model}, "
          f"hidden={best.config.hidden_size}, "
          f"layers={best.config.num_layers}, "
          f"dropout={best.config.dropout}, lr={best.config.lr}")
    print(f"  {label} val loss: {best.best_val_loss:.6f} (epoch {best.best_epoch})")
    return results

results_e4 = run_corridor_single(cached_e4, max_N_e4, "E4")
""",
        cell_id="cell-19-train",
    )


def _add_evaluate_cell() -> None:
    md(
        """## Evaluación en test — MAE y RMSE por dirección

Evalúa el modelo de E4 sobre el split test, separando por dirección
para desnormalizar con la media/std específica de cada una.
La inferencia usa la firma espacial: `model(inp, ctx, input_mask)` (AD-1, AD-4).
`target.squeeze(1)` colapsa `(B, 1, max_N)` → `(B, max_N)` (invariante REQ-2).
""",
        cell_id="cell-19-evaluate-md",
    )
    code(
        """
def evaluate_corridor_model(best_result, test_loaders, max_N, stats, label):
    \"\"\"Evaluate one corridor SpatialTransformer model on per-direction test splits.

    Uses direction-specific mean/std from NormalizationStats for denormalization.
    Returns (dir_metrics, dir_arrays) for downstream result building.
    \"\"\"
    empresa_id = int(list(stats.means.keys())[0][0])

    model = SpatialTransformer(
        max_N=max_N,
        nhead=best_result.config.nhead,
        d_model=best_result.config.d_model,
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
        all_persist = []   # persistence (B1) baseline = last observed input
        all_pmask   = []   # validity of that last input step

        with torch.no_grad():
            for batch in test_loaders[direction]:
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

                # Persistence (B1): predict y(t+HORIZON) = last observed y(t),
                # the final input step (t_idx == T_IN-1) of each window.
                # Same samples as the DL model -> exact paired comparison, no join.
                persist_z   = inp[:, T_IN - 1, :]
                persist_min = denormalize_predictions(persist_z, mean_val, std_val)
                pmask_step  = input_mask[:, T_IN - 1, :]

                all_preds.append(pred_min.cpu().numpy())
                all_targets.append(target_min.cpu().numpy())
                all_masks.append(mask_sq.cpu().numpy())
                all_persist.append(persist_min.cpu().numpy())
                all_pmask.append(pmask_step.cpu().numpy())

        preds_flat   = np.concatenate([p.ravel() for p in all_preds])
        targets_flat = np.concatenate([t.ravel() for t in all_targets])
        masks_flat   = np.concatenate([m.ravel() for m in all_masks]).astype(bool)
        persist_flat = np.concatenate([p.ravel() for p in all_persist])
        pmask_flat   = np.concatenate([m.ravel() for m in all_pmask]).astype(bool)

        valid_preds   = preds_flat[masks_flat]
        valid_targets = targets_flat[masks_flat]

        mae_val  = mae(valid_targets,  valid_preds)
        rmse_val = rmse(valid_targets, valid_preds)
        print(f"{label} dir={direction:+d} SpatialTransformer: MAE={mae_val:.4f} min, "
              f"RMSE={rmse_val:.4f} min (n_valid={masks_flat.sum():,})")

        dir_metrics[direction] = (mae_val, rmse_val)
        dir_arrays[direction]  = (preds_flat, targets_flat, masks_flat,
                                  persist_flat, pmask_flat)

    return dir_metrics, dir_arrays

dir_metrics_e4, dir_arrays_e4 = evaluate_corridor_model(
    results_e4[0], cached_test_e4, max_N_e4, stats_e4, "E4")
""",
        cell_id="cell-19-evaluate",
    )


def _add_results_cell() -> None:
    md(
        """## Tabla de resultados — spatial_transformer_E4_results_h{H}.csv

Guarda los resultados del SpatialTransformer en E4 en formato long-form con columna
`horizon` adicional (schema: corridor, direction, baseline, metric, value, horizon).
direction ∈ {"-1", "+1", "aggregate"} — 6 filas para E4.
""",
        cell_id="cell-19-results-md",
    )
    code(
        """
def build_transformer_rows(corridor, dir_metrics, dir_arrays):
    \"\"\"Build long-form rows with horizon column.

    direction values: \"-1\", \"+1\", \"aggregate\"
    baseline: \"SpatialTransformer\"
    metric:   \"MAE\", \"RMSE\"
    horizon:  HORIZON (constant injected by builder)

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
                "baseline":  "SpatialTransformer",
                "metric":    metric_name,
                "value":     float(metric_val),
                "horizon": HORIZON,
            })

    # Aggregate: pool valid predictions from both directions.
    all_preds   = np.concatenate([dir_arrays[d][0][dir_arrays[d][2]] for d in [-1, 1]])
    all_targets = np.concatenate([dir_arrays[d][1][dir_arrays[d][2]] for d in [-1, 1]])
    agg_mae  = mae(all_targets,  all_preds)
    agg_rmse = rmse(all_targets, all_preds)
    print(f"{corridor} aggregate SpatialTransformer: MAE={agg_mae:.4f} min, RMSE={agg_rmse:.4f} min "
          f"(n_valid={len(all_preds):,})")
    for metric_name, metric_val in [("MAE", agg_mae), ("RMSE", agg_rmse)]:
        rows.append({
            "corridor":  corridor,
            "direction": "aggregate",
            "baseline":  "SpatialTransformer",
            "metric":    metric_name,
            "value":     float(metric_val),
            "horizon": HORIZON,
        })
    return rows

transformer_rows = build_transformer_rows("E4", dir_metrics_e4, dir_arrays_e4)

transformer_results = pl.DataFrame(transformer_rows)
# Output file: spatial_transformer_E4_results_h{HORIZON}.csv (corridor + horizon discriminated)
transformer_results.write_csv(TRANSFORMER_CSV_OUT)
print(f"\\nSpatialTransformer E4 results written to: {TRANSFORMER_CSV_OUT}  ({transformer_results.height} rows)")
print(transformer_results)
""",
        cell_id="cell-19-results",
    )


def _add_residuals_cell() -> None:
    md(
        """## Residuos por muestra — spatial_transformer_E4_residuals_h{H}.csv (significancia)

Exporta el error por muestra del SpatialTransformer y de la persistencia (B1) sobre
EXACTAMENTE las mismas muestras (mismas ventanas), para los tests de significancia
(Diebold-Mariano / Wilcoxon) que se corren en local.

La persistencia se recomputa dentro del kernel como el último input observado
(`t_idx == T_IN-1`), que está HORIZON pasos antes del target → es la predicción
de B1 a este horizonte. Se conserva la muestra solo si el target Y el último input
son válidos. Schema: corridor, direction, horizon, y_true, y_pred_dl, y_pred_persist.
""",
        cell_id="cell-19-residuals-md",
    )
    code(
        """
TRANSFORMER_RESID_OUT = OUTPUT_DIR / f"spatial_transformer_E4_residuals_h{HORIZON}.csv"

def build_residuals_df(corridor, dir_arrays):
    \"\"\"Per-sample paired errors (DL vs persistence) for one corridor.

    dir_arrays[d] = (preds, targets, target_mask, persist, persist_mask).
    A sample is kept only where BOTH the target and the persistence (last input)
    are valid — the paired set the significance tests need.
    \"\"\"
    frames = []
    for direction_int in [-1, 1]:
        direction_str = f"+{direction_int}" if direction_int > 0 else str(direction_int)
        preds, targets, tmask, persist, pmask = dir_arrays[direction_int]
        keep = tmask & pmask
        frames.append(
            pl.DataFrame({
                "y_true":         targets[keep].astype("float64"),
                "y_pred_dl":      preds[keep].astype("float64"),
                "y_pred_persist": persist[keep].astype("float64"),
            }).with_columns(
                corridor=pl.lit(corridor),
                direction=pl.lit(direction_str),
                horizon=pl.lit(HORIZON),
            ).select(
                ["corridor", "direction", "horizon",
                 "y_true", "y_pred_dl", "y_pred_persist"]
            )
        )
    return pl.concat(frames)

residuals = build_residuals_df("E4", dir_arrays_e4)
residuals.write_csv(TRANSFORMER_RESID_OUT)
print(f"Residuals written to: {TRANSFORMER_RESID_OUT}  ({residuals.height:,} rows)")

# Sanity check: recomputed persistence MAE should match B1 from NB16 (E4).
for corr in ["E4"]:
    sub = residuals.filter(pl.col("corridor") == corr)
    dl_mae = (sub["y_true"] - sub["y_pred_dl"]).abs().mean()
    p_mae  = (sub["y_true"] - sub["y_pred_persist"]).abs().mean()
    print(f"  {corr} h={HORIZON}: DL MAE={dl_mae:.4f}  persist(B1) MAE={p_mae:.4f}  "
          f"n={sub.height:,}")
""",
        cell_id="cell-19-residuals",
    )


def _add_compare_cell() -> None:
    md(
        """## Comparación con baselines de E4 (NB16)

Carga `baselines_results_multih.csv` generado por NB16
(kernel_source: `alexhuaracha/16-e4-data-baselines`) y filtra al horizonte
actual antes de comparar con los resultados del SpatialTransformer en E4.
""",
        cell_id="cell-19-compare-md",
    )
    code(
        """
baselines_csv = _find_baselines_csv()
if baselines_csv is not None:
    baselines = pl.read_csv(baselines_csv)
    print(f"Baselines loaded from: {baselines_csv} ({baselines.height} rows total)")

    # Filter to the current horizon before comparison
    baselines = baselines.filter(pl.col("horizon") == HORIZON)
    print(f"Baselines filtered to horizon={HORIZON}: {baselines.height} rows")

    # Filter to MAE for a clean comparison
    comparison = pl.concat([
        baselines.filter(pl.col("metric") == "MAE"),
        transformer_results.filter(pl.col("metric") == "MAE"),
    ])

    print(f"\\nMAE comparison (minutes) at horizon={HORIZON} — lower is better:")
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
    print("baselines_results_multih.csv not found — skipping comparison.")
    print("Ensure kernel_sources includes alexhuaracha/16-e4-data-baselines.")
    print("SpatialTransformer E4 results summary:")
    for row in transformer_results.filter(pl.col("metric") == "MAE").iter_rows(named=True):
        print(f"  {row['corridor']} dir={row['direction']}: MAE={row['value']:.4f} min")
""",
        cell_id="cell-19-compare",
    )


# ---------------------------------------------------------------------------
# Top-level build function — one call per horizon.
# ---------------------------------------------------------------------------

def build_horizon_notebook(horizon: int) -> None:
    """Build and write a single per-horizon notebook (E4 only).

    The HORIZON constant is injected into setup and dataset cells.
    """
    _reset()

    out_dir = ROOT / "notebooks" / "19_e4_transformer" / f"h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_filename = f"19_e4_transformer_h{horizon}.ipynb"
    nb_path = out_dir / notebook_filename

    _add_title_cell(horizon)
    _add_setup_cell(horizon)
    _add_embed_cells()
    _add_load_cell()
    _add_split_cell()
    _add_norm_cell()
    _add_context_cell()
    _add_dataset_cell(horizon)
    _add_train_cell()
    _add_evaluate_cell()
    _add_results_cell()
    _add_residuals_cell()
    _add_compare_cell()

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
    # NOTE: title slugifies to id (lowercase words, no punctuation) so Kaggle
    # resolves the kernel under the same slug as the id.
    kernel_meta = {
        "id": f"alexhuaracha/19-e4-transformer-h{horizon}",
        "title": f"19 e4 transformer h{horizon}",
        "code_file": notebook_filename,
        **_KERNEL_META_BASE,
    }
    meta_path = out_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8")
    print(f"Kernel metadata written: {meta_path}")


# ---------------------------------------------------------------------------
# Entry point — build one notebook per horizon.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for h in [1, 3, 5, 10]:
        build_horizon_notebook(h)
