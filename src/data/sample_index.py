"""Canonical sample index — the shared population contract (C1 + C2).

This module exists because the project never wrote down what a sample *is*.
``windowing.make_window_index`` anchors windows on a **row index** inside a
``(empresaid, direction, pair_rank)`` slot, which produces two defects:

  C1 violation — the target of a snapshot is emitted once per anchoring slot,
  so every target is counted 2.4-5.4 times and the reported MAE is a
  fleet-density-weighted mean.

  C2 violation — consecutive row positions are not checked to be consecutive
  minutes, so the nominal horizon is a row offset, not a time offset. A window
  crossing a day boundary or a trip cut yields a "10-minute" target that is
  hours away.

``windowing.make_window_index`` is deliberately left untouched: notebooks 12/13
(and the E4 twins 18/19) must keep reproducing the frozen architecture
comparison, whose validity rests on all three architectures sharing the same
flaw. This module is the population for the *retrained* pipeline only.

Contract enforced here
----------------------
C1  A sample is ``(empresaid, direction, start_ts, horizon)``. The anchor is an
    instant, not a row position, and each one is emitted exactly once.
C2  A sample is valid only when the ``T_in + horizon`` snapshot timestamps
    starting at ``start_ts`` are strictly consecutive minutes, so the target
    lands exactly ``horizon`` minutes after the end of the input window.

``pair_rank`` survives as the position *within* the predicted vector, never as
an anchoring axis.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import polars as pl

# One snapshot per minute: the grid the headway series is resampled onto.
GRID_STEP_MINUTES: int = 1

# Columns that identify one snapshot series. `pair_rank` is intentionally absent.
_SERIES_COLS: list[str] = ["empresaid", "direction"]


class SampleIndexEntry(TypedDict):
    """One canonical sample.

    empresaid: int — corridor identifier.
    direction: int — bus direction (-1 or +1).
    start_ts: the timestamp of the FIRST input snapshot.
    target_ts: the timestamp of the predicted snapshot. Always exactly
               ``horizon`` minutes after the last input snapshot.
    horizon: int — prediction offset in minutes (now genuinely minutes).
    """

    empresaid: int
    direction: int
    start_ts: object
    target_ts: object
    horizon: int


def _contiguous_run_mask(ts: np.ndarray, span: int) -> np.ndarray:
    """Mask of window starts whose ``span`` timestamps are consecutive minutes.

    Parameters
    ----------
    ts:
        Sorted, unique timestamps for one series, as numpy datetime64.
    span:
        Number of consecutive timestamps a valid window must cover
        (``T_in + horizon``).

    Returns
    -------
    Boolean array of length ``len(ts)``. ``mask[i]`` is True when
    ``ts[i:i+span]`` are consecutive minutes. Positions with fewer than
    ``span`` timestamps remaining are False.
    """
    n = ts.size
    mask = np.zeros(n, dtype=bool)
    if n < span:
        return mask

    step = np.timedelta64(GRID_STEP_MINUTES, "m")
    # gap[i] is True when ts[i+1] follows ts[i] by exactly one grid step.
    gap_ok = np.diff(ts) == step

    # A window at i needs the span-1 gaps starting at i to all be contiguous.
    # Cumulative-sum trick: count of good gaps in [i, i+span-2] must equal span-1.
    need = span - 1
    if need == 0:
        mask[:] = True
        return mask

    cum = np.concatenate(([0], np.cumsum(gap_ok)))
    starts = np.arange(n - span + 1)
    good = cum[starts + need] - cum[starts]
    mask[: n - span + 1] = good == need
    return mask


def make_sample_index(
    df: pl.DataFrame,
    *,
    horizon: int,
    T_in: int,
) -> pl.DataFrame:
    """Canonical sample index for one frame. Enforces C1 and C2.

    Parameters
    ----------
    df:
        Headway frame. Required columns: ``empresaid``, ``direction``, ``t``.
        ``pair_rank`` may be present; it is ignored for anchoring.
    horizon:
        Prediction offset in minutes.
    T_in:
        Input window length in snapshots.

    Returns
    -------
    DataFrame with one row per canonical sample, columns
    ``empresaid, direction, start_ts, target_ts, horizon``, sorted
    deterministically. Empty (with the right schema) when no window qualifies.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if T_in < 1:
        raise ValueError(f"T_in must be >= 1, got {T_in}")

    span = T_in + horizon
    rows: list[pl.DataFrame] = []

    # One series per (empresaid, direction) — NOT per pair_rank. That collapse is
    # the whole point of C1: the target is the snapshot, not the slot.
    series = (
        df.select(_SERIES_COLS + ["t"])
        .unique()
        .sort(_SERIES_COLS + ["t"])
        .partition_by(_SERIES_COLS, maintain_order=True)
    )

    for s in series:
        if s.is_empty():
            continue
        ts = s.get_column("t").to_numpy()
        mask = _contiguous_run_mask(ts, span)
        if not mask.any():
            continue

        first = s.row(0, named=True)
        starts = ts[mask]
        # The target sits `horizon` minutes after the LAST input snapshot, which
        # is start + (T_in - 1) steps. Contiguity is already guaranteed by mask.
        targets = starts + np.timedelta64((T_in - 1 + horizon) * GRID_STEP_MINUTES, "m")

        rows.append(
            pl.DataFrame(
                {
                    "empresaid": np.full(starts.size, int(first["empresaid"]), dtype=np.int64),
                    "direction": np.full(starts.size, int(first["direction"]), dtype=np.int64),
                    "start_ts": starts,
                    "target_ts": targets,
                    "horizon": np.full(starts.size, int(horizon), dtype=np.int64),
                }
            )
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "empresaid": pl.Int64,
                "direction": pl.Int64,
                "start_ts": df.schema["t"],
                "target_ts": df.schema["t"],
                "horizon": pl.Int64,
            }
        )

    return pl.concat(rows).sort(["empresaid", "direction", "start_ts"])


def effective_horizon_minutes(
    index: pl.DataFrame,
    *,
    T_in: int,
) -> pl.Series:
    """Realized gap in minutes between end-of-window and target, per sample.

    Under C2 this is constant and equal to ``horizon`` for every row. It is
    exposed so a test can assert that rather than trust the construction.
    """
    window_end = pl.col("start_ts") + pl.duration(minutes=(T_in - 1) * GRID_STEP_MINUTES)
    return (
        index.select(
            ((pl.col("target_ts") - window_end).dt.total_minutes()).alias("eff")
        )
        .get_column("eff")
    )
