"""XGBoost features built on the canonical sample index — same population, same window.

``fitted._build_features`` derives its lags with
``pl.col("delta_t_min").forward_fill().shift(horizon + k - 1).over(_SLOT_COLS)``.
That shift is **positional**: it steps back `k` rows inside a
``(empresaid, direction, pair_rank)`` slot without checking that consecutive rows
are consecutive minutes. It is the very defect audited in §3 for the DL windows,
reaching the fitted baseline through a different mechanism — so "levelled
competitor" was never quite true: the two families were mis-specified in
different ways over different populations.

This module rebuilds the feature matrix from the shared sample index instead.
Because contract C2 guarantees the ``T_in + horizon`` timestamps of a sample are
consecutive minutes, ``lag_k`` can be read directly off the grid at
``start_ts + (T_in - k)`` — no forward-fill, no positional shift, no silent
bridging of a day boundary.

Consequence, and the point of the exercise: with ``N_LAGS == T_in == 12`` the
XGBoost sees **exactly the twelve observations the LSTM sees**, for exactly the
same set of samples. Levelling stops being an argument and becomes a property.

The atypical flag is absent by design (plan-reentrenamiento.md C3): it is a
whole-day aggregate and therefore not knowable at prediction time.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# Number of lag features. Kept equal to T_in so the fitted baseline and the
# network consume the same window; changing one without the other breaks the
# levelling claim.
N_LAGS: int = 12

FEATURE_NAMES: list[str] = (
    [f"lag_{k}" for k in range(1, N_LAGS + 1)]
    + ["hour", "weekday", "direction", "pair_rank"]
)


def _dense_grid(
    frame: pl.DataFrame, *, max_N: int, value_col: str
) -> tuple[dict, np.ndarray, np.ndarray]:
    """(timestamp -> row) map plus dense value / validity grids for one series."""
    timestamps = frame.select("t").unique().sort("t").get_column("t").to_numpy()
    ts_index = {ts: i for i, ts in enumerate(timestamps)}

    values = np.full((timestamps.size, max_N), np.nan, dtype=np.float64)

    rows = frame.select(["t", "pair_rank", value_col])
    row_of = np.array([ts_index[ts] for ts in rows.get_column("t").to_numpy()])
    pr = rows.get_column("pair_rank").to_numpy()
    val = np.asarray(rows.get_column(value_col).to_numpy(), dtype=np.float64)

    # np.isnan, not polars' is_null: converting a column to numpy turns nulls
    # into NaN and drops the null flag, so is_null answers False for all of them.
    # Harmless here only because the grid defaults to NaN and callers re-check
    # with np.isnan — but relying on that would be luck, not design.
    ok = (pr >= 0) & (pr < max_N) & ~np.isnan(val)
    values[row_of[ok], pr[ok]] = val[ok]

    return ts_index, timestamps, values


def build_contiguous_features(
    frame: pl.DataFrame,
    sample_index: pl.DataFrame,
    *,
    horizon: int,
    T_in: int,
    max_N_by_direction: dict[tuple[int, int], int],
    value_col: str = "delta_t_min",
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Feature matrix, target vector and key frame for one corridor.

    One row per ``(sample, pair_rank)`` whose target and whose ``lag_1``
    (persistence) are both present — the paired set, matching what the DL export
    keeps.

    Returns
    -------
    X : (n_rows, len(FEATURE_NAMES)) float64
    y : (n_rows,) float64 — the target headway in minutes
    keys : DataFrame with empresaid, direction, start_ts, target_ts, horizon,
           pair_rank and ``y_pred_persist`` (== lag_1), in row order of ``X``.
    """
    if N_LAGS > T_in:
        raise ValueError(f"N_LAGS ({N_LAGS}) exceeds T_in ({T_in})")

    blocks_X: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    blocks_key: list[pl.DataFrame] = []

    for (empresaid, direction), idx_part in _partition_index(sample_index):
        max_N = max_N_by_direction[(empresaid, direction)]
        series = frame.filter(
            (pl.col("empresaid") == empresaid) & (pl.col("direction") == direction)
        )
        ts_index, _timestamps, values = _dense_grid(
            series, max_N=max_N, value_col=value_col
        )

        starts = idx_part.get_column("start_ts").to_numpy()
        targets = idx_part.get_column("target_ts").to_numpy()
        start_rows = np.array([ts_index[ts] for ts in starts], dtype=np.int64)
        # C2 makes the run contiguous, so the target sits a fixed offset away.
        target_rows = start_rows + (T_in - 1 + horizon)

        # lag_k is the observation k-1 minutes before the end of the window.
        # k=1 is the last input snapshot, i.e. exactly B1 persistence.
        lag_rows = np.stack(
            [start_rows + (T_in - k) for k in range(1, N_LAGS + 1)], axis=1
        )

        # (n_samples, max_N) -> flattened per pair_rank below.
        y_grid = values[target_rows]                      # (n_samples, max_N)
        lag_grid = values[lag_rows]                       # (n_samples, N_LAGS, max_N)

        keep = ~np.isnan(y_grid) & ~np.isnan(lag_grid[:, 0, :])
        if not keep.any():
            continue
        s_i, pr_i = np.nonzero(keep)

        lags = lag_grid[s_i, :, pr_i]                     # (n_kept, N_LAGS)
        # Remaining lags may still be missing; XGBoost handles NaN natively, so
        # they are passed through rather than imputed (no forward-fill here).

        target_ts = targets[s_i]
        hours = target_ts.astype("datetime64[h]").astype(np.int64) % 24
        # Monday=1 .. Sunday=7, matching polars' dt.weekday().
        days = (
            (target_ts.astype("datetime64[D]").astype(np.int64) + 3) % 7
        ) + 1

        X = np.column_stack(
            [
                lags,
                hours.astype(np.float64),
                days.astype(np.float64),
                np.full(s_i.size, float(direction)),
                pr_i.astype(np.float64),
            ]
        )
        blocks_X.append(X)
        blocks_y.append(y_grid[s_i, pr_i])
        blocks_key.append(
            pl.DataFrame(
                {
                    "empresaid": np.full(s_i.size, empresaid, dtype=np.int64),
                    "direction": np.full(s_i.size, direction, dtype=np.int64),
                    "start_ts": starts[s_i],
                    "target_ts": target_ts,
                    "horizon": np.full(s_i.size, horizon, dtype=np.int64),
                    "pair_rank": pr_i.astype(np.int64),
                    "y_pred_persist": lags[:, 0],
                }
            )
        )

    if not blocks_X:
        return (
            np.zeros((0, len(FEATURE_NAMES))),
            np.zeros(0),
            pl.DataFrame(),
        )

    return (
        np.concatenate(blocks_X),
        np.concatenate(blocks_y),
        pl.concat(blocks_key),
    )


def _partition_index(sample_index: pl.DataFrame):
    """Yield ((empresaid, direction), sub-frame) in deterministic order."""
    parts = sample_index.sort(["empresaid", "direction", "start_ts"]).partition_by(
        ["empresaid", "direction"], maintain_order=True
    )
    for part in parts:
        first = part.row(0, named=True)
        yield (int(first["empresaid"]), int(first["direction"])), part
