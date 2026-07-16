"""Generate notebooks/15_lstm_multiseed/h{H}/15_lstm_multiseed_h{H}.ipynb for Kaggle.

Produces 4 notebooks — one per horizon h ∈ {1, 3, 5, 10}:
  15_lstm_multiseed_h1.ipynb   → alexhuaracha/15-lstm-multiseed-h1
  15_lstm_multiseed_h3.ipynb   → alexhuaracha/15-lstm-multiseed-h3
  15_lstm_multiseed_h5.ipynb   → alexhuaracha/15-lstm-multiseed-h5
  15_lstm_multiseed_h10.ipynb  → alexhuaracha/15-lstm-multiseed-h10

NB15 is a MULTI-SEED ROBUSTNESS study of the frozen winning LSTM config
(paper-audit gap C2: the DL results are single-seed; reviewers want seed
variance / confidence intervals on the degradation curve).

It re-trains the SAME frozen winning config (E2 and E59) N times with N
different seeds and exports per-seed MAE/RMSE, so confidence intervals can be
drawn per horizon.  Everything else is a faithful clone of build_notebook_11.py:
same inline-embed pattern, same data pipeline cells, same per-horizon build loop,
same kernel metadata structure.

Surgical differences from build_notebook_11.py:
  - Output dir/files: notebooks/15_lstm_multiseed/h{H}/15_lstm_multiseed_h{H}.ipynb
  - Cell IDs: cell-15-* prefix.
  - Kernel metadata: id alexhuaracha/15-lstm-multiseed-h{H}, title
    "15 LSTM Multiseed h{H}", code_file 15_lstm_multiseed_h{H}.ipynb.
  - Train cell: injects SEEDS = [42, 123, 456, 789, 999] and loops over them,
    building a per-seed config via dataclasses.replace(WINNING_CONFIGS[label],
    seed=s).  TrainConfig already has a `seed` field and train_model calls
    set_seed(config.seed), so each seed yields a genuinely different model.
    Each corridor yields a list of (seed, TrainResult).
  - Evaluate cell: evaluates EACH per-seed model on test (verbatim NB11 logic).
  - Results CSV: lstm_multiseed_h{H}.csv with an extra `seed` column
    (schema: corridor, direction, baseline, metric, value, horizon, seed).
    Rows per kernel = 3 directions × 2 metrics × 2 corridors × 5 seeds = 60.
  - DROPS NB11's residuals + compare cells (per-seed significance out of scope).

kernel_sources: ["alexhuaracha/04-preprocessing", "alexhuaracha/10-baselines-multi-horizonte", "alexhuaracha/02-eda-corridors"]
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
        "alexhuaracha/04-preprocessing",
        "alexhuaracha/10-baselines-multi-horizonte",
        "alexhuaracha/02-eda-corridors",
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
# → models/lstm → train
# ---------------------------------------------------------------------------

def _add_embed_cells() -> None:
    """Add all module-embed cell pairs (same for every horizon)."""

    embed_module(
        "evaluation/splits.py",
        """## Module: evaluation/splits

Temporal split helper (`split_temporal`) and train-only p99 winsorization
(`winsorize_train_p99`).  Split date ranges are locked in spec §3.
""",
        cell_id_md="cell-15-embed-splits-md",
        cell_id_code="cell-15-embed-splits",
    )

    embed_module(
        "evaluation/metrics.py",
        """## Module: evaluation/metrics

`mae` and `rmse` in minutes.  Both accept polars Series or numpy arrays.
Null/NaN rows are dropped before aggregation.
""",
        cell_id_md="cell-15-embed-metrics-md",
        cell_id_code="cell-15-embed-metrics",
    )

    embed_module(
        "data/windowing.py",
        """## Module: data/windowing

`make_window_index` — per-slot deterministic window index.
`compute_max_N` — train-p99 of (n_buses - 1) per (empresaid, direction).
Constants: `DEFAULT_T_IN=12`, `DEFAULT_T_OUT=1`, `DEFAULT_STRIDE=1`.
""",
        cell_id_md="cell-15-embed-windowing-md",
        cell_id_code="cell-15-embed-windowing",
    )

    embed_module(
        "data/normalization.py",
        """## Module: data/normalization

`compute_normalization_stats` — per-direction z-score stats from TRAIN ONLY.
`apply_zscore` — add `delta_t_min_z` column; no clipping (DL-8).
""",
        cell_id_md="cell-15-embed-normalization-md",
        cell_id_code="cell-15-embed-normalization",
    )

    embed_module(
        "data/context_features.py",
        """## Module: data/context_features

`encode_context` — add 5 cyclical + atypical-flag columns.
`load_atypical_days` — in this notebook the CSV is a required, hash-verified
input (DL-2); the run stops before training if it is absent or altered.
""",
        cell_id_md="cell-15-embed-context-md",
        cell_id_code="cell-15-embed-context",
    )

    embed_module(
        "data/dataset.py",
        """## Module: data/dataset  (first torch import)

`HeadwayDataset` — on-the-fly window materialization with masks (DL-11).
`collate_fn` — batch stacking for variable-N edge cases (REQ-6).
""",
        cell_id_md="cell-15-embed-dataset-md",
        cell_id_code="cell-15-embed-dataset",
    )

    embed_module(
        "models/lstm.py",
        """## Module: models/lstm

`HeadwayLSTM` — flat LSTM encoder (batch_first, last hidden state → Linear head).
`masked_mse_loss` — MSE over valid (mask==True) positions; clamp(min=1) prevents
zero-division on all-False masks.
""",
        cell_id_md="cell-15-embed-lstm-md",
        cell_id_code="cell-15-embed-lstm",
    )

    embed_module(
        "train.py",
        """## Module: train

`TrainConfig`, `TrainResult` — hyperparameter and result dataclasses.
`set_seed` — reproducibility seeds (torch + cuda + numpy); called by `train_model`
with `config.seed`, so each seed produces a genuinely different model.
`train_one_epoch`, `evaluate_epoch` — single-epoch train/eval loops.
`EarlyStopping` — patience-based early stopping with best-state copy.
`GRID` — 24 TrainConfig entries (kept for reference; NB15 uses WINNING_CONFIGS only).
`train_model` — full training loop with early stopping.
`grid_search` — run configs; return sorted by best_val_loss.
`save_checkpoint`, `load_checkpoint` — model persistence.
`denormalize_predictions` — z-score → minutes conversion.
""",
        cell_id_md="cell-15-embed-train-md",
        cell_id_code="cell-15-embed-train",
    )


# ---------------------------------------------------------------------------
# Per-horizon cell builders.
# ---------------------------------------------------------------------------

def _add_title_cell(horizon: int) -> None:
    md(
        f"""
# 15 — LSTM Multi-Seed Robustez h={horizon}  (auto-generado por build_notebook_15.py)

Estudio de **robustez multi-seed** del LSTM ganador (`HeadwayLSTM`) sobre los
corredores **E2** y **E59** con horizonte directo **h={horizon} minuto(s)**.

Re-entrena la MISMA configuración ganadora congelada de Fase 5
(documentada en `docs/resultados/configuraciones-ganadoras.md`) con **5 seeds**
distintos por corredor, para cuantificar la varianza por seed y poder dibujar
intervalos de confianza sobre la curva de degradación (gap C2 de la auditoría).

`TrainConfig` ya tiene un campo `seed` y `train_model` llama `set_seed(config.seed)`,
así que cada seed produce un modelo genuinamente distinto. Exporta MAE/RMSE
por seed en test (sin tests de significancia: fuera de alcance).

Referencia: `docs/plan-de-desarrollo.md §6.5 Fase 6.5 — Multi-Horizonte`.
""",
        cell_id="cell-15-title",
    )


def _add_setup_cell(horizon: int) -> None:
    code(
        f"""
import hashlib

import polars as pl
import numpy as np
from pathlib import Path

# Frozen SHA-256 of every required training input (recertification contract).
# The run stops BEFORE training when a required file is missing or its bytes
# differ from the pinned Kaggle snapshot; extra mounted copies are fine as
# long as one matches.
INPUT_HASHES = {{
    "headways_E2.parquet": "82a34eaffc79cd82346d4595a2e72f5d3ffb751ed37fa0fc0cde3a8f8fb345d4",
    "headways_E59.parquet": "0b5f5593caaa94e4e6af7da672bc2cad7b49b69b7cbd0a22092f15700a89a448",
    "atypical_days.csv": "2054245cc830e58b9397b75ea3b55d034581046b64e73b1630ca7d464e3ecb86",
}}

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
        raise FileNotFoundError(f"Required input not found anywhere: {{name}}")
    for path in candidates:
        if _sha256_file(path) == INPUT_HASHES[name]:
            return path
    raise ValueError(
        f"No copy of {{name}} matches its frozen SHA-256 — "
        f"candidates: {{[str(p) for p in candidates]}}"
    )

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)
HORIZON = {horizon}
LSTM_CSV_OUT = OUTPUT_DIR / f"lstm_multiseed_h{{HORIZON}}.csv"

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"Output dir: {{OUTPUT_DIR}}")
print(f"Horizon:    {{HORIZON}}")
print(f"Device:     {{DEVICE}}")
""",
        cell_id="cell-15-setup",
    )


def _add_load_cell() -> None:
    md(
        """## Cargar datos — E2 y E59

Lee los parquets generados por NB04 (kernel_source: `alexhuaracha/04-preprocessing`).
Inyecta `empresaid` como columna literal para cumplir el contrato de slot key.
""",
        cell_id="cell-15-load-md",
    )
    code(
        """
hw_e2  = pl.read_parquet(_resolve_input("headways_E2.parquet")).with_columns(pl.lit(2,  dtype=pl.Int64).alias("empresaid"))
hw_e59 = pl.read_parquet(_resolve_input("headways_E59.parquet")).with_columns(pl.lit(59, dtype=pl.Int64).alias("empresaid"))

print(f"E2:  {hw_e2.height:,} rows, {hw_e2.width} cols")
print(f"E59: {hw_e59.height:,} rows, {hw_e59.width} cols")
""",
        cell_id="cell-15-load",
    )


def _add_split_cell() -> None:
    md(
        """## Split temporal + winsorización

Aplica `split_temporal` y `winsorize_train_p99` (INV-1, INV-6).
El umbral de winsorización se computa exclusivamente sobre el split `train`
y luego se aplica al frame completo de splits (AC-WINSOR-1, AC-WINSOR-2 — leakage guard).
""",
        cell_id="cell-15-split-md",
    )
    code(
        """
def prepare_corridor(hw: pl.DataFrame, label: str) -> pl.DataFrame:
    df_split = split_temporal(hw)
    df_winsor, threshold = winsorize_train_p99(df_split)
    print(f"{label}: split counts = {df_split.group_by('split').agg(pl.len()).sort('split')}")
    print(f"{label}: winsorize threshold = {threshold:.4f} min")
    return df_winsor

df_e2  = prepare_corridor(hw_e2,  "E2")
df_e59 = prepare_corridor(hw_e59, "E59")
""",
        cell_id="cell-15-split",
    )


def _add_norm_cell() -> None:
    md(
        """## Normalización z-score (train only)

Computa estadísticas de normalización exclusivamente sobre filas de entrenamiento
(INV-2, AC-NORM-1) y aplica z-score a todos los splits.
""",
        cell_id="cell-15-norm-md",
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
        cell_id="cell-15-norm",
    )


def _add_context_cell() -> None:
    md(
        """## Features de contexto

Codificación cíclica de hora y día de semana + flag de día atípico (DL-2).
`atypical_days.csv` es un input requerido y verificado por hash: la corrida
se detiene antes de entrenar si falta o si sus bytes difieren del snapshot.
""",
        cell_id="cell-15-context-md",
    )
    code(
        """
atypical_path = _resolve_input("atypical_days.csv")
atypical_dates = load_atypical_days(atypical_path)
if not atypical_dates:
    raise ValueError(f"atypical_days.csv parsed to an empty date set: {atypical_path}")
print(f"Atypical days loaded: {len(atypical_dates)} dates (path={atypical_path})")

df_e2  = encode_context(df_e2,  atypical_dates=atypical_dates)
df_e59 = encode_context(df_e59, atypical_dates=atypical_dates)
print(f"E2  context columns: {[c for c in df_e2.columns  if c in CONTEXT_FEATURE_NAMES]}")
print(f"E59 context columns: {[c for c in df_e59.columns if c in CONTEXT_FEATURE_NAMES]}")
""",
        cell_id="cell-15-context",
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

Nota: la construcción del dataset es IDÉNTICA a NB11-h{horizon}, así los
resultados por seed son directamente comparables con el ganador congelado.
""",
        cell_id="cell-15-dataset-md",
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

max_N_e2,  cached_e2,  cached_test_e2  = build_corridor_data(df_e2,  "E2")
max_N_e59, cached_e59, cached_test_e59 = build_corridor_data(df_e59, "E59")
print("\\nDataset construction complete.")
""",
        cell_id="cell-15-dataset",
    )


def _add_train_cell() -> None:
    md(
        """## Entrenamiento LSTM — multi-seed (config ganadora × 5 seeds por corredor)

Reusa la configuración ganadora de Fase 5 (grid search original con h=1)
documentada en `docs/resultados/configuraciones-ganadoras.md` y la re-entrena con
**5 seeds** por corredor.

Para cada corredor se itera sobre `SEEDS`, se construye una config por-seed con
`dataclasses.replace(WINNING_CONFIGS[label], seed=s)`, y se llama `grid_search`
con esa única config (`configs=[cfg]`).  `train_model` llama `set_seed(config.seed)`
al inicio, así cada seed entrena un modelo genuinamente distinto.

Cada corredor produce una lista de tuplas `(seed, TrainResult)`.
""",
        cell_id="cell-15-train-md",
    )
    code(
        """
import dataclasses

# Frozen winning configs (from Fase-5 grid search, h=1).
#   E2:  hidden=32, layers=1, dropout=0.0, lr=5e-4
#   E59: hidden=32, layers=2, dropout=0.2, lr=5e-4
WINNING_CONFIGS = {
    "E2":  TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=5e-4),
    "E59": TrainConfig(hidden_size=32, num_layers=2, dropout=0.2, lr=5e-4),
}

# Multi-seed robustness: re-train the SAME winning config under N seeds.
# TrainConfig has a `seed` field and train_model calls set_seed(config.seed),
# so each seed yields a genuinely different model — no changes to train.py.
SEEDS = [42, 123, 456, 789, 999]

def run_corridor_multiseed(loaders: dict, max_N: int, label: str):
    \"\"\"Re-train the winning config once per seed for one corridor.

    For each seed, build a per-seed config via dataclasses.replace(..., seed=s)
    and run grid_search with that single config (configs=[cfg]). grid_search
    returns a 1-element list; take results[0] as that seed's TrainResult.

    Returns a list of (seed, TrainResult), one entry per seed (input order).
    \"\"\"
    base_cfg = WINNING_CONFIGS[label]
    print(f"\\n{label} LSTM multi-seed: max_N={max_N}, device={DEVICE}, "
          f"n_seeds={len(SEEDS)} (hidden={base_cfg.hidden_size}, "
          f"layers={base_cfg.num_layers}, dropout={base_cfg.dropout}, lr={base_cfg.lr})")

    seed_results = []
    for s in SEEDS:
        cfg = dataclasses.replace(base_cfg, seed=s)
        results = grid_search(
            train_dl=loaders["train"],
            val_dl=loaders["val"],
            max_N=max_N,
            configs=[cfg],
            device=DEVICE,
        )
        best = results[0]
        print(f"  {label} seed={s}: val_loss={best.best_val_loss:.6f} "
              f"(epoch {best.best_epoch})")
        seed_results.append((s, best))
    return seed_results

results_e2  = run_corridor_multiseed(cached_e2,  max_N_e2,  "E2")
results_e59 = run_corridor_multiseed(cached_e59, max_N_e59, "E59")
""",
        cell_id="cell-15-train",
    )


def _add_evaluate_cell() -> None:
    md(
        """## Evaluación en test — MAE y RMSE por seed y dirección

Evalúa CADA modelo por-seed de cada corredor sobre el split test, separando por
dirección para desnormalizar con la media/std específica de cada una.
`target.squeeze(1)` colapsa `(B, 1, max_N)` → `(B, max_N)` (invariante REQ-2).

Produce `seed_metrics`: dict[(corridor, seed)] -> (dir_metrics, dir_arrays).
""",
        cell_id="cell-15-evaluate-md",
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

# Evaluate every per-seed model on test.
# seed_metrics[(corridor, seed)] = (dir_metrics, dir_arrays)
seed_metrics = {}

for seed, r in results_e2:
    print(f"\\n=== E2 seed={seed} ===")
    dm, da = evaluate_corridor_model(r, cached_test_e2, max_N_e2, stats_e2, f"E2 s{seed}")
    seed_metrics[("E2", seed)] = (dm, da)

for seed, r in results_e59:
    print(f"\\n=== E59 seed={seed} ===")
    dm, da = evaluate_corridor_model(r, cached_test_e59, max_N_e59, stats_e59, f"E59 s{seed}")
    seed_metrics[("E59", seed)] = (dm, da)

print(f"\\nAll {len(seed_metrics)} (corridor, seed) models evaluated.")
""",
        cell_id="cell-15-evaluate",
    )


def _add_results_cell() -> None:
    md(
        """## Tabla de resultados — lstm_multiseed_h{H}.csv

Guarda los resultados del LSTM en formato long-form con columnas `horizon` y `seed`
adicionales (schema: corridor, direction, baseline, metric, value, horizon, seed).
direction ∈ {"-1", "+1", "aggregate"}.

Filas por kernel = 3 direcciones × 2 métricas × 2 corredores × 5 seeds = **60 filas**.
""",
        cell_id="cell-15-results-md",
    )
    code(
        """
def build_lstm_rows(corridor, seed, dir_metrics, dir_arrays):
    \"\"\"Build long-form rows for one (corridor, seed) with horizon + seed columns.

    direction values: \"-1\", \"+1\", \"aggregate\"
    baseline: \"LSTM\"
    metric:   \"MAE\", \"RMSE\"
    horizon:  HORIZON (constant injected by builder)
    seed:     the training seed for this model

    Produces 6 rows per (corridor, seed) (3 directions × 2 metrics).
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
                "horizon":   HORIZON,
                "seed":      int(seed),
            })

    # Aggregate: pool valid predictions from both directions.
    all_preds   = np.concatenate([dir_arrays[d][0][dir_arrays[d][2]] for d in [-1, 1]])
    all_targets = np.concatenate([dir_arrays[d][1][dir_arrays[d][2]] for d in [-1, 1]])
    agg_mae  = mae(all_targets,  all_preds)
    agg_rmse = rmse(all_targets, all_preds)
    print(f"{corridor} seed={seed} aggregate LSTM: MAE={agg_mae:.4f} min, "
          f"RMSE={agg_rmse:.4f} min (n_valid={len(all_preds):,})")
    for metric_name, metric_val in [("MAE", agg_mae), ("RMSE", agg_rmse)]:
        rows.append({
            "corridor":  corridor,
            "direction": "aggregate",
            "baseline":  "LSTM",
            "metric":    metric_name,
            "value":     float(metric_val),
            "horizon":   HORIZON,
            "seed":      int(seed),
        })
    return rows

lstm_rows = []
for (corridor, seed), (dir_metrics, dir_arrays) in sorted(seed_metrics.items()):
    lstm_rows.extend(build_lstm_rows(corridor, seed, dir_metrics, dir_arrays))

lstm_results = pl.DataFrame(lstm_rows)
# Output file: lstm_multiseed_h{HORIZON}.csv (horizon-discriminated filename)
lstm_results.write_csv(LSTM_CSV_OUT)
print(f"\\nLSTM multi-seed results written to: {LSTM_CSV_OUT}  ({lstm_results.height} rows)")
print(lstm_results)
""",
        cell_id="cell-15-results",
    )


# ---------------------------------------------------------------------------
# Top-level build function — one call per horizon.
# ---------------------------------------------------------------------------

def build_horizon_notebook(horizon: int) -> None:
    """Build and write a single per-horizon multi-seed notebook.

    Both corridors (E2 + E59) are re-trained under 5 seeds in the same notebook.
    The HORIZON constant is injected into setup and dataset cells.
    """
    _reset()

    out_dir = ROOT / "notebooks" / "15_lstm_multiseed" / f"h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_filename = f"15_lstm_multiseed_h{horizon}.ipynb"
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
        "id": f"alexhuaracha/15-lstm-multiseed-h{horizon}",
        "title": f"15 LSTM Multiseed h{horizon}",
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
