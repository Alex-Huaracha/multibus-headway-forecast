"""Deterministic synthetic headways DataFrame factory for Fase 3 tests.

Returns a pl.DataFrame with the minimal columns that baselines consume:
    empresaid (Int64), t (Datetime us), direction (Int64),
    pair_rank (Int32), delta_t_min (Float64, nullable).

No `split` column is included — `split_temporal` (from src.evaluation.splits)
adds it downstream.  All delta_t_min values are hand-supplied so tests are
deterministic without RNG involvement in the values themselves.

Design: mirrors tests/fixtures/synthetic.py style — module-level helpers,
no classes, all hard-coded when possible.

Usage:
    from tests.fixtures.headways_factory import make_headways_fixture
    from datetime import date, datetime

    df = make_headways_fixture(
        empresaid=2,
        train_dates=[date(2023, 11, 1)],
        test_dates=[date(2024, 2, 10)],
        delta_values_per_slot={(−1, 1): [3.0, 5.0, None, 7.0]},
    )
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

# Fixed anchor time (well inside train range 2023-10-01 → 2024-01-15).
# Used when callers do not supply explicit datetime offsets.
_T_ANCHOR = datetime(2023, 12, 1, 0, 0, 0)

# One snapshot step = 1 minute (the paper's forecast horizon h=1 min).
_STEP = timedelta(minutes=1)


def make_headways_fixture(
    *,
    empresaid: int = 2,
    n_slots: int = 2,
    train_dates: list[date],
    test_dates: list[date],
    val_dates: list[date] | None = None,
    delta_values_per_slot: dict[tuple[int, int], list[float | None]] | None = None,
    rng_seed: int = 42,
) -> pl.DataFrame:
    """Build a deterministic headways DataFrame with the R7 v4 schema subset.

    Parameters
    ----------
    empresaid:
        empresa identifier; same for all rows.
    n_slots:
        Number of (direction, pair_rank) slots to generate when
        delta_values_per_slot is not provided.  Ignored when
        delta_values_per_slot is supplied.
    train_dates:
        Dates on which to place train-split rows (one row per slot per date).
        Timestamps are set to midnight + per-slot minute offset so rows are
        unique and sortable.
    test_dates:
        Same for test-split rows.
    val_dates:
        Same for val-split rows.  Defaults to [] (no val rows).
    delta_values_per_slot:
        Mapping from (direction, pair_rank) → list[float | None].
        Values are assigned in chronological order across all dates
        (train_dates first, then val_dates, then test_dates).
        If None, all delta_t_min values are set to 1.0 for simplicity.
    rng_seed:
        Kept for API compatibility; unused (no RNG in value assignment).

    Returns
    -------
    pl.DataFrame with columns:
        empresaid (Int64), t (Datetime[μs]), direction (Int64),
        pair_rank (Int32), delta_t_min (Float64 nullable).
    """
    if val_dates is None:
        val_dates = []

    # Build the canonical list of (date, split_label) pairs in chronological order
    # (for value assignment only; we don't attach a split column here).
    all_dates_ordered: list[date] = (
        sorted(train_dates) + sorted(val_dates) + sorted(test_dates)
    )

    # Determine slot list.
    if delta_values_per_slot is not None:
        slots: list[tuple[int, int]] = sorted(delta_values_per_slot.keys())
    else:
        # Default: direction=-1 and +1 with pair_rank=1..ceil(n_slots/2)
        slots = []
        for pr in range(1, n_slots // 2 + 1):
            slots.append((-1, pr))
            slots.append((1, pr))
        if n_slots % 2 == 1:
            slots.append((-1, n_slots // 2 + 1))
        slots = slots[:n_slots]

    rows: list[dict[str, Any]] = []
    for slot_idx, (direction, pair_rank) in enumerate(slots):
        if delta_values_per_slot is not None:
            values = list(delta_values_per_slot[(direction, pair_rank)])
        else:
            values = [1.0] * len(all_dates_ordered)

        for date_idx, d in enumerate(all_dates_ordered):
            # Use minute offset to make timestamps unique across slots.
            t = datetime(d.year, d.month, d.day, 0, 0, 0) + timedelta(
                minutes=slot_idx
            )
            delta = values[date_idx] if date_idx < len(values) else None
            rows.append(
                {
                    "empresaid": empresaid,
                    "t": t,
                    "direction": direction,
                    "pair_rank": pair_rank,
                    "delta_t_min": delta,
                }
            )

    if not rows:
        # Return empty frame with correct schema.
        return pl.DataFrame(
            {
                "empresaid": pl.Series([], dtype=pl.Int64),
                "t": pl.Series([], dtype=pl.Datetime("us")),
                "direction": pl.Series([], dtype=pl.Int64),
                "pair_rank": pl.Series([], dtype=pl.Int32),
                "delta_t_min": pl.Series([], dtype=pl.Float64),
            }
        )

    df = pl.DataFrame(rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("direction").cast(pl.Int64),
        pl.col("pair_rank").cast(pl.Int32),
        pl.col("delta_t_min").cast(pl.Float64),
    )
    return df
