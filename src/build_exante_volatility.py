"""Ex-ante volatility stratification — reviewer response artifact.

Computes std of the 12 raw input-window headway values (ex-ante feature)
for each test-set sample, stratifies into terciles, and checks whether
the DL model beats persistence in the high-volatility tercile.

Covers corridors: E2, E59, E4 at horizons h=3, h=5 and h=10.

Usage:
    uv run python src/build_exante_volatility.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.normalization import (
    NormalizationStats,
    Z_EPS,
    apply_zscore,
    compute_normalization_stats,
)
from src.data.windowing import compute_max_N, make_window_index
from src.evaluation.splits import split_temporal, winsorize_train_p99
from src.evaluation.exante_terciles import (
    TercileThresholds,
    assign_terciles,
    compute_frozen_thresholds,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = ROOT / "data" / "processed"
RESID_DIR = ROOT / "docs" / "resultados" / "recertificado" / "residuos-multihorizon" / "11-lstm"
OUT_DIR = ROOT / "docs" / "resultados" / "recertificado" / "csv-multihorizon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_IN = 12  # DEFAULT_T_IN


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_parquet(empresaid: int) -> pl.DataFrame:
    """Load headway parquet and add empresaid literal column."""
    name = f"headways_E{empresaid}.parquet"
    path = DATA_DIR / name
    df = pl.read_parquet(path)
    return df.with_columns(pl.lit(empresaid, dtype=pl.Int64).alias("empresaid"))


def prepare_df(empresaid: int) -> tuple[pl.DataFrame, NormalizationStats]:
    """Full preprocessing pipeline: split → train-p99 winsorize full frame → zscore."""
    df = load_parquet(empresaid)
    df = split_temporal(df)

    # Threshold is computed from train rows and applied to every split row.
    df, _threshold = winsorize_train_p99(df)

    # Normalization stats from winsorized train only
    train_only = df.filter(pl.col("split") == "train")
    stats = compute_normalization_stats(train_only)

    # Apply z-score to full df
    df = apply_zscore(df, stats)
    return df, stats


# ---------------------------------------------------------------------------
# Core materialization logic (replicates fast_materialize without torch)
# ---------------------------------------------------------------------------

def _build_snapshot_lookup(df: pl.DataFrame, max_N: int) -> dict:
    """Build dict: (empresaid, direction, timestamp) → (z_vals, mask)."""
    grouped = (
        df.group_by(["empresaid", "direction", "t"])
        .agg([pl.col("pair_rank"), pl.col("delta_t_min_z"), pl.col("delta_t_min")])
    )
    lookup = {}
    for row in grouped.iter_rows(named=True):
        key = (row["empresaid"], row["direction"], row["t"])
        z_vals = np.zeros(max_N, dtype=np.float32)
        raw_vals = np.zeros(max_N, dtype=np.float32)
        mask = np.zeros(max_N, dtype=np.bool_)
        for pr, z, raw in zip(row["pair_rank"], row["delta_t_min_z"], row["delta_t_min"]):
            if 0 <= pr < max_N and z is not None and raw is not None:
                z_vals[pr] = z
                raw_vals[pr] = raw
                mask[pr] = True
        lookup[key] = (z_vals, raw_vals, mask)
    return lookup


def _build_slot_timestamps(df: pl.DataFrame) -> dict:
    """Build dict: (empresaid, direction, pair_rank) → sorted list of timestamps."""
    grouped = (
        df.group_by(["empresaid", "direction", "pair_rank"])
        .agg(pl.col("t").sort())
    )
    slots = {}
    for row in grouped.iter_rows(named=True):
        key = (row["empresaid"], row["direction"], row["pair_rank"])
        slots[key] = row["t"]
    return slots


def materialize_direction(
    test_df: pl.DataFrame,
    max_N: int,
    horizon: int,
    stats: NormalizationStats,
    empresaid: int,
    direction: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize one direction's test windows.

    Returns (targets_raw, persist_raw, target_mask, persist_mask, ex_ante_std)
    all of shape (n_windows * max_N,), with keep = target_mask & persist_mask.
    """
    mean_val = stats.means[(empresaid, direction)]
    std_val = stats.stds[(empresaid, direction)]

    # Filter to this direction only for window index construction
    dir_df = test_df.filter(pl.col("direction") == direction)

    window_index = make_window_index(dir_df, T_in=T_IN, horizon=horizon)
    n_windows = len(window_index)

    if n_windows == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, np.array([], dtype=bool), np.array([], dtype=bool), empty

    window_size = T_IN + horizon
    lookup = _build_snapshot_lookup(dir_df, max_N)
    slots = _build_slot_timestamps(dir_df)

    # Allocate arrays
    all_input_z = np.zeros((n_windows, T_IN, max_N), dtype=np.float32)
    all_input_raw = np.zeros((n_windows, T_IN, max_N), dtype=np.float32)
    all_input_mask = np.zeros((n_windows, T_IN, max_N), dtype=np.bool_)
    all_target_z = np.zeros((n_windows, max_N), dtype=np.float32)
    all_target_mask = np.zeros((n_windows, max_N), dtype=np.bool_)

    for i, entry in enumerate(window_index):
        slot_key = (entry["empresaid"], entry["direction"], entry["pair_rank"])
        emp = entry["empresaid"]
        dirn = entry["direction"]
        ts_list = slots[slot_key]
        start = entry["start_idx"]

        for t_idx in range(window_size):
            ts = ts_list[start + t_idx]
            snap = lookup.get((emp, dirn, ts))
            if snap is None:
                continue
            z_vals, raw_vals, mask = snap
            if t_idx < T_IN:
                all_input_z[i, t_idx] = z_vals
                all_input_raw[i, t_idx] = raw_vals
                all_input_mask[i, t_idx] = mask
            elif t_idx == window_size - 1:
                all_target_z[i] = z_vals
                all_target_mask[i] = mask

    # Denormalize: raw = z * (std + Z_EPS) + mean
    # target raw
    all_target_raw = all_target_z.astype(np.float64) * (std_val + Z_EPS) + mean_val

    # Persistence = last input timestep (T_IN-1), denormalized
    persist_z = all_input_z[:, T_IN - 1, :]  # (n_windows, max_N)
    persist_raw = persist_z.astype(np.float64) * (std_val + Z_EPS) + mean_val
    persist_mask = all_input_mask[:, T_IN - 1, :]  # (n_windows, max_N)

    # Ex-ante std: std of the 12 raw input values per (window, pair_rank)
    # all_input_raw: (n_windows, T_IN, max_N); all_input_mask: same shape
    ex_ante_std_2d = np.full((n_windows, max_N), np.nan, dtype=np.float64)
    for pr in range(max_N):
        # shape (n_windows, T_IN)
        vals = all_input_raw[:, :, pr].astype(np.float64)
        masks = all_input_mask[:, :, pr]  # (n_windows, T_IN)
        # For each window, compute std of valid values
        for i in range(n_windows):
            valid = vals[i, masks[i]]
            if len(valid) >= 2:
                ex_ante_std_2d[i, pr] = float(np.std(valid, ddof=1))

    # Ravel everything in C order (window-major)
    targets_flat = all_target_raw.ravel()
    persist_flat = persist_raw.ravel()
    tmask_flat = all_target_mask.ravel()
    pmask_flat = persist_mask.ravel()
    ex_ante_flat = ex_ante_std_2d.ravel()

    return targets_flat, persist_flat, tmask_flat, pmask_flat, ex_ante_flat


def materialize_corridor(
    df: pl.DataFrame,
    stats: NormalizationStats,
    empresaid: int,
    horizon: int,
    splits: tuple[str, ...] = ("test",),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize selected splits for one corridor; return kept samples.

    Order: dir=-1 then dir=+1 (matching residual CSV order).
    Returns (targets, persist, ex_ante_std), each already filtered to the kept
    (valid target & persistence) samples.
    """
    train_df = df.filter(pl.col("split") == "train")
    selected_df = df.filter(pl.col("split").is_in(splits))

    max_n = compute_max_N(train_df, quantile=0.99)
    global_max_N = max(max_n.values())

    all_targets = []
    all_persist = []
    all_ex_ante = []

    for direction in [-1, 1]:
        targets_flat, persist_flat, tmask_flat, pmask_flat, ex_ante_flat = \
            materialize_direction(selected_df, global_max_N, horizon, stats,
                                   empresaid, direction)
        keep = tmask_flat & pmask_flat
        all_targets.append(targets_flat[keep])
        all_persist.append(persist_flat[keep])
        all_ex_ante.append(ex_ante_flat[keep])

    targets = np.concatenate(all_targets)
    persist = np.concatenate(all_persist)
    ex_ante = np.concatenate(all_ex_ante)

    return targets, persist, ex_ante


# ---------------------------------------------------------------------------
# Alignment verification
# ---------------------------------------------------------------------------

def verify_alignment(
    corridor: str,
    horizon: int,
    recon_targets: np.ndarray,
    recon_persist: np.ndarray,
    csv_path: Path,
    csv_corridor_filter: str | None = None,
) -> bool:
    """Check that reconstructed targets and persistence match the residual CSV.

    Returns True if alignment passes (max abs diff < 1e-2), False otherwise.
    """
    csv_df = pl.read_csv(csv_path)
    if csv_corridor_filter is not None:
        csv_df = csv_df.filter(pl.col("corridor") == csv_corridor_filter)

    n_csv = csv_df.height
    n_rec = len(recon_targets)

    max_diff_target = float("nan")
    max_diff_persist = float("nan")
    passed = False

    if n_csv != n_rec:
        print(f"  ALIGNMENT FAIL [{corridor} h={horizon}]: "
              f"row count mismatch — CSV={n_csv}, reconstructed={n_rec}")
    else:
        csv_targets = csv_df["y_true"].to_numpy()
        csv_persist = csv_df["y_pred_persist"].to_numpy()
        max_diff_target = float(np.max(np.abs(csv_targets - recon_targets)))
        max_diff_persist = float(np.max(np.abs(csv_persist - recon_persist)))
        passed = max_diff_target < 1e-2 and max_diff_persist < 1e-2
        status = "PASS" if passed else "FAIL"
        print(f"  ALIGNMENT {status} [{corridor} h={horizon}]: "
              f"n_csv={n_csv}, n_rec={n_rec}, "
              f"max|Δtarget|={max_diff_target:.6f}, "
              f"max|Δpersist|={max_diff_persist:.6f}")

    return passed


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def compute_stratification(
    corridor: str,
    horizon: int,
    y_true: np.ndarray,
    y_pred_dl: np.ndarray,
    y_pred_persist: np.ndarray,
    ex_ante_std: np.ndarray,
    thresholds: TercileThresholds,
) -> list[dict]:
    """Apply frozen ex-ante terciles and compute MAE per tercile."""
    valid_mask = np.isfinite(ex_ante_std)
    y_true_v = y_true[valid_mask]
    y_dl_v = y_pred_dl[valid_mask]
    y_persist_v = y_pred_persist[valid_mask]
    ex_v = ex_ante_std[valid_mask]

    if len(ex_v) == 0:
        print(f"  WARNING [{corridor} h={horizon}]: no valid ex_ante_std samples")
        return []

    tercile_codes = assign_terciles(ex_v, thresholds)
    tercile_masks = [tercile_codes == code for code in range(3)]
    tercile_names = ["low", "mid", "high"]

    rows = []
    total = len(ex_v)
    for name, mask in zip(tercile_names, tercile_masks):
        n = int(mask.sum())
        if n == 0:
            continue
        share = n / total
        mae_persist = float(np.mean(np.abs(y_true_v[mask] - y_persist_v[mask])))
        mae_dl = float(np.mean(np.abs(y_true_v[mask] - y_dl_v[mask])))
        delta_mae = mae_dl - mae_persist  # negative = DL wins
        rows.append({
            "corridor": corridor,
            "horizon": horizon,
            "tercile": name,
            "n": n,
            "share": share,
            "mae_persist": mae_persist,
            "mae_dl": mae_dl,
            "delta_mae": delta_mae,
            "p33_threshold": thresholds.p33,
            "p66_threshold": thresholds.p66,
            "calib_split": thresholds.calib_split,
            "calib_n": thresholds.calib_n,
        })

    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_corridor(
    corridor: str,
    empresaid: int,
    horizons: list[int],
    resid_path_fn,
    csv_corridor_filter: str | None = None,
) -> list[dict]:
    """Full pipeline for one corridor across multiple horizons."""
    print(f"\n=== {corridor} (empresaid={empresaid}) ===")
    df, stats = prepare_df(empresaid)

    all_rows = []
    for horizon in horizons:
        print(f"\n  Horizon h={horizon}")
        _, _, calibration_ex_ante = materialize_corridor(
            df, stats, empresaid, horizon, splits=("train", "val")
        )
        thresholds = compute_frozen_thresholds(calibration_ex_ante)
        targets, persist, ex_ante = materialize_corridor(df, stats, empresaid, horizon)

        csv_path = resid_path_fn(horizon)

        # Alignment check
        passed = verify_alignment(
            corridor, horizon, targets, persist, csv_path,
            csv_corridor_filter=csv_corridor_filter,
        )
        if not passed:
            print(f"  HARD GATE FAILED for {corridor} h={horizon} — stopping.")
            sys.exit(1)

        # Load DL predictions from CSV
        csv_df = pl.read_csv(csv_path)
        if csv_corridor_filter is not None:
            csv_df = csv_df.filter(pl.col("corridor") == csv_corridor_filter)
        y_pred_dl = csv_df["y_pred_dl"].to_numpy()

        rows = compute_stratification(
            corridor, horizon, targets, y_pred_dl, persist, ex_ante, thresholds
        )
        all_rows.extend(rows)

    return all_rows


def main() -> None:
    print("=" * 70)
    print("Ex-ante Volatility Stratification")
    print("=" * 70)

    horizons = [3, 5, 10]
    all_rows: list[dict] = []

    # E2
    all_rows.extend(run_corridor(
        corridor="E2",
        empresaid=2,
        horizons=horizons,
        resid_path_fn=lambda h: RESID_DIR / f"h{h}" / f"lstm_residuals_h{h}.csv",
        csv_corridor_filter="E2",
    ))

    # E59
    all_rows.extend(run_corridor(
        corridor="E59",
        empresaid=59,
        horizons=horizons,
        resid_path_fn=lambda h: RESID_DIR / f"h{h}" / f"lstm_residuals_h{h}.csv",
        csv_corridor_filter="E59",
    ))

    # E4
    all_rows.extend(run_corridor(
        corridor="E4",
        empresaid=4,
        horizons=horizons,
        resid_path_fn=lambda h: RESID_DIR / f"h{h}" / f"lstm_E4_residuals_h{h}.csv",
        csv_corridor_filter=None,
    ))

    # Save output
    result_df = pl.DataFrame(all_rows)
    out_path = OUT_DIR / "exante_volatility_multihorizon.csv"
    result_df.write_csv(out_path)
    print(f"\nResults written to: {out_path}")

    # Print full stratification table
    print("\n" + "=" * 70)
    print("STRATIFICATION TABLE")
    print("=" * 70)
    with pl.Config(tbl_rows=50, tbl_width_chars=120, float_precision=4):
        print(result_df.sort(["corridor", "horizon", "tercile"]))

    # Headline finding
    print("\n" + "=" * 70)
    print("HEADLINE FINDING: DL vs Persistence in HIGH-volatility tercile")
    print("=" * 70)
    high_tercile = result_df.filter(pl.col("tercile") == "high").sort(
        ["corridor", "horizon"]
    )
    dl_wins_all = True
    for row in high_tercile.iter_rows(named=True):
        wins = row["delta_mae"] < 0
        status = "DL WINS" if wins else "DL LOSES"
        if not wins:
            dl_wins_all = False
        print(
            f"  {row['corridor']} h={row['horizon']:2d}  "
            f"tercile=high  n={row['n']:,}  share={row['share']:.2%}  "
            f"MAE_persist={row['mae_persist']:.4f}  "
            f"MAE_DL={row['mae_dl']:.4f}  "
            f"delta_MAE={row['delta_mae']:+.4f}  [{status}]"
        )

    print()
    if dl_wins_all:
        print("DL beats persistence in the high-volatility tercile for ALL "
              "corridor/horizon combinations.")
    else:
        print("DL does NOT beat persistence in high-volatility tercile for:")
        for row in high_tercile.filter(pl.col("delta_mae") >= 0).iter_rows(named=True):
            print(f"  -> {row['corridor']} h={row['horizon']}")


if __name__ == "__main__":
    main()
