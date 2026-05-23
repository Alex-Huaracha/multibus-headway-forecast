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

Extensions for Fase 3 DL (supervised-dataset-construction):
    make_multi_corridor_fixture — multi-empresa train/val/test frame for windowing + max_N
    make_split_fixture          — per-direction train/test frame with differing means
    make_dataset_fixture        — (df, window_index, max_N_by_direction) ready for HeadwayDataset
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from src.evaluation.splits import SPLIT_TRAIN_END, SPLIT_VAL_END, SPLIT_TEST_END

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


# ---------------------------------------------------------------------------
# Fase 3 DL extensions — supervised-dataset-construction
# ---------------------------------------------------------------------------

def make_multi_corridor_fixture(
    *,
    n_days: int = 14,
    empresaids: tuple[int, ...] = (2, 59),
    n_directions: int = 2,
    n_pair_ranks: int = 3,
    add_n_buses_col: bool = True,
    train_end: date = SPLIT_TRAIN_END,
    val_end: date = SPLIT_VAL_END,
    test_end: date = SPLIT_TEST_END,
) -> pl.DataFrame:
    """Build a multi-corridor headways frame spanning train/val/test ranges.

    AC-MAXN-1, AC-MAXN-2: used by test_windowing.py::TestMaxN which needs
    an n_buses column to derive max_N per (empresaid, direction).

    Returns the R7 v4 schema subset PLUS an `n_buses` Int32 column and a
    `split` Utf8 column.  When add_n_buses_col=False, omits n_buses for
    tests that do not need it.
    """
    from src.evaluation.splits import (
        SPLIT_TRAIN_START,
        SPLIT_VAL_START,
        SPLIT_TEST_START,
    )

    directions: list[int] = [-1, 1] if n_directions == 2 else [-1]

    # Map each date to a split label deterministically.
    train_start = SPLIT_TRAIN_START
    val_start = SPLIT_VAL_START
    test_start = SPLIT_TEST_START

    def _split_label(d: date) -> str:
        if train_start <= d <= train_end:
            return "train"
        if val_start <= d <= val_end:
            return "val"
        return "test"

    # Generate n_days days: first 7 in train, next 4 in val, last 3 in test.
    n_train = max(1, n_days * 7 // 14)
    n_val = max(1, n_days * 4 // 14)
    n_test = n_days - n_train - n_val

    def _date_range(start: date, count: int) -> list[date]:
        return [start + timedelta(days=i) for i in range(count)]

    all_dates = (
        _date_range(train_start, n_train)
        + _date_range(val_start, n_val)
        + _date_range(test_start, n_test)
    )

    rows: list[dict[str, Any]] = []
    for emp in empresaids:
        for direction in directions:
            # n_buses varies per snapshot to give compute_max_N signal.
            # train rows: n_buses in [2, 3, 4]; val/test: some rows have 5 buses.
            for day_idx, d in enumerate(all_dates):
                split = _split_label(d)
                n_buses = (day_idx % n_pair_ranks) + 2
                for pr in range(n_pair_ranks):
                    t = datetime(d.year, d.month, d.day, 8, pr, 0)
                    delta = float(pr + 1) + 0.5 * day_idx
                    row: dict[str, Any] = {
                        "empresaid": emp,
                        "t": t,
                        "direction": direction,
                        "pair_rank": pr,
                        "delta_t_min": delta,
                        "split": split,
                    }
                    if add_n_buses_col:
                        row["n_buses"] = n_buses
                    rows.append(row)

    schema_extras: dict[str, Any] = {}
    if add_n_buses_col:
        schema_extras["n_buses"] = pl.Int32

    df = pl.DataFrame(rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("direction").cast(pl.Int64),
        pl.col("pair_rank").cast(pl.Int32),
        pl.col("delta_t_min").cast(pl.Float64),
        pl.col("split").cast(pl.Utf8),
        *([pl.col("n_buses").cast(pl.Int32)] if add_n_buses_col else []),
    )
    return df


def make_split_fixture(
    *,
    empresaid: int = 2,
    train_means: dict[int, float] | None = None,
    test_means: dict[int, float] | None = None,
) -> pl.DataFrame:
    """Build a frame where train mean differs from test mean per direction.

    AC-NORM-LEAK-1: train mean must be measurably different from full-frame
    mean so the leakage guard has clear signal (mirrors AC-WINSOR-2 setup).

    Each direction gets 5 train rows and 5 test rows; values are set from
    train_means / test_means (default: direction=-1 train=2.0 test=10.0,
    direction=+1 train=3.0 test=12.0 so the means are clearly distinct).
    """
    from src.evaluation.splits import SPLIT_TRAIN_START, SPLIT_TEST_START

    if train_means is None:
        train_means = {-1: 2.0, 1: 3.0}
    if test_means is None:
        test_means = {-1: 10.0, 1: 12.0}

    n_per_split = 5
    rows: list[dict[str, Any]] = []
    for direction in (-1, 1):
        for i in range(n_per_split):
            train_date = SPLIT_TRAIN_START + timedelta(days=i)
            t_train = datetime(train_date.year, train_date.month, train_date.day, 8, direction + 2, 0)
            rows.append({
                "empresaid": empresaid,
                "t": t_train,
                "direction": direction,
                "pair_rank": 0,
                "delta_t_min": train_means[direction],
                "split": "train",
            })
        for i in range(n_per_split):
            test_date = SPLIT_TEST_START + timedelta(days=i)
            t_test = datetime(test_date.year, test_date.month, test_date.day, 8, direction + 2, 0)
            rows.append({
                "empresaid": empresaid,
                "t": t_test,
                "direction": direction,
                "pair_rank": 0,
                "delta_t_min": test_means[direction],
                "split": "test",
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("direction").cast(pl.Int64),
        pl.col("pair_rank").cast(pl.Int32),
        pl.col("delta_t_min").cast(pl.Float64),
        pl.col("split").cast(pl.Utf8),
    )


def make_dataset_fixture(
    *,
    T_in: int = 3,
    T_out: int = 1,
    max_N: int = 2,
    n_snapshots: int = 6,
    include_context: bool = True,
) -> "tuple[Any, list[dict], dict[tuple[int, int], int]]":
    """Build (df, window_index, max_N_by_direction) ready for HeadwayDataset.

    Constructs a deterministic fixture for a single (empresaid=2, direction=-1) slot
    with ``n_snapshots`` timesteps. Each timestep has ``max_N`` pair_ranks so that
    all positions are filled (no padding), making tensor shape assertions exact.

    Returns
    -------
    df:
        pl.DataFrame with columns: empresaid, t, direction, pair_rank,
        delta_t_min, delta_t_min_z, and 5 context feature columns when
        ``include_context=True``.
    window_index:
        list[WindowIndexEntry] from make_window_index(df, T_in=T_in, T_out=T_out).
    max_N_by_direction:
        {(2, -1): max_N} — the single direction present in this fixture.

    Used by tests/data/test_dataset.py (torch-dependent callers only).
    """
    from src.data.normalization import NormalizationStats, apply_zscore
    from src.data.context_features import encode_context
    from src.data.windowing import make_window_index

    empresaid = 2
    direction = -1
    anchor = datetime(2023, 11, 1, 8, 0, 0)
    step = timedelta(minutes=1)

    rows: list[dict[str, Any]] = []
    for snap_idx in range(n_snapshots):
        t = anchor + snap_idx * step
        for pr in range(max_N):
            rows.append({
                "empresaid": empresaid,
                "t": t,
                "direction": direction,
                "pair_rank": pr,
                "delta_t_min": float(snap_idx + pr + 1),
            })

    df = pl.DataFrame(rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("direction").cast(pl.Int64),
        pl.col("pair_rank").cast(pl.Int32),
        pl.col("delta_t_min").cast(pl.Float64),
    )

    # Apply z-score with constant stats (mean=0, std=1 for simplicity).
    stats = NormalizationStats(
        means={(empresaid, direction): 0.0},
        stds={(empresaid, direction): 1.0},
    )
    df = apply_zscore(df, stats)

    if include_context:
        df = encode_context(df, atypical_dates=None)

    window_index = make_window_index(df, T_in=T_in, T_out=T_out)
    max_N_by_direction: dict[tuple[int, int], int] = {(empresaid, direction): max_N}

    return df, window_index, max_N_by_direction
