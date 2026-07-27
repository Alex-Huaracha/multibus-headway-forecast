"""Tests for the contiguous XGBoost feature builder.

The module reads its lags with advanced numpy indexing over a
``(n_samples, N_LAGS, max_N)`` block. That is exactly the shape of expression
that silently transposes, so the ordering of ``lag_k`` is asserted against the
timestamps it is supposed to come from rather than trusted.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.baselines.contiguous_features import (
    FEATURE_NAMES,
    N_LAGS,
    build_contiguous_features,
)
from src.data.sample_index import make_sample_index

T_IN = 12
MAX_N = 3
BASE = datetime(2024, 2, 8, 8, 0)
MAX_N_BY_DIR = {(2, 1): MAX_N, (2, -1): MAX_N}


def _frame(offsets, *, empresaid=2, direction=1, pair_ranks=(0, 1, 2)):
    """delta_t_min encodes (minute offset, pair_rank) so misreads are visible."""
    rows = []
    for off in offsets:
        for pr in pair_ranks:
            rows.append(
                {
                    "empresaid": empresaid,
                    "direction": direction,
                    "pair_rank": pr,
                    "t": BASE + timedelta(minutes=int(off)),
                    "delta_t_min": float(off) * 100.0 + pr,
                }
            )
    return pl.DataFrame(rows)


def _build(df, horizon=3):
    idx = make_sample_index(df, horizon=horizon, T_in=T_IN)
    X, y, keys = build_contiguous_features(
        df, idx, horizon=horizon, T_in=T_IN, max_N_by_direction=MAX_N_BY_DIR
    )
    return idx, X, y, keys


class TestLagOrdering:
    def test_lag_k_comes_from_the_right_timestamp(self):
        """lag_k must be the observation at start_ts + (T_in - k)."""
        df = _frame(range(60))
        idx, X, y, keys = _build(df)

        starts = keys.get_column("start_ts").to_numpy()
        prs = keys.get_column("pair_rank").to_numpy()
        for k in range(1, N_LAGS + 1):
            offsets = (
                (starts - np.datetime64(BASE)) / np.timedelta64(1, "m")
            ) + (T_IN - k)
            expected = offsets * 100.0 + prs
            assert np.allclose(X[:, k - 1], expected), f"lag_{k} misaligned"

    def test_lag_1_is_persistence(self):
        df = _frame(range(60))
        _, X, _, keys = _build(df)
        assert np.allclose(X[:, 0], keys.get_column("y_pred_persist").to_numpy())

    def test_lags_are_strictly_ordered_backwards_in_time(self):
        """lag_1 is the most recent; lag_12 the oldest. Encoded values decrease."""
        df = _frame(range(60))
        _, X, _, _ = _build(df)
        lags = X[:, :N_LAGS]
        assert np.all(np.diff(lags, axis=1) < 0), "lag order is not backwards in time"

    def test_target_is_horizon_minutes_after_lag_1(self):
        for horizon in (1, 3, 5, 10):
            df = _frame(range(60))
            _, X, y, keys = _build(df, horizon=horizon)
            # Encoded value carries the minute offset in its hundreds digit.
            lag1_offset = (X[:, 0] - keys.get_column("pair_rank").to_numpy()) / 100.0
            y_offset = (y - keys.get_column("pair_rank").to_numpy()) / 100.0
            assert np.allclose(y_offset - lag1_offset, horizon)


class TestShapeAndSchema:
    def test_feature_count_matches_names(self):
        _, X, _, _ = _build(_frame(range(40)))
        assert X.shape[1] == len(FEATURE_NAMES) == N_LAGS + 4

    def test_rows_align_across_X_y_and_keys(self):
        _, X, y, keys = _build(_frame(range(40)))
        assert X.shape[0] == y.size == keys.height

    def test_direction_and_pair_rank_columns(self):
        df = pl.concat([_frame(range(40), direction=1), _frame(range(40), direction=-1)])
        _, X, _, keys = _build(df)
        assert np.array_equal(X[:, N_LAGS + 2], keys.get_column("direction").to_numpy())
        assert np.array_equal(X[:, N_LAGS + 3], keys.get_column("pair_rank").to_numpy())


class TestContiguity:
    def test_no_row_bridges_a_gap(self):
        offsets = list(range(40)) + [40 + 23 * 60 + k for k in range(40)]
        _, X, _, keys = _build(_frame(offsets))
        starts = keys.get_column("start_ts").to_numpy()
        targets = keys.get_column("target_ts").to_numpy()
        gaps = (targets - starts) / np.timedelta64(1, "m")
        assert np.all(gaps == T_IN - 1 + 3)

    def test_empty_when_no_contiguous_run_is_long_enough(self):
        _, X, y, keys = _build(_frame(range(5)))
        assert X.shape[0] == 0
        assert y.size == 0


class TestPairing:
    def test_rows_without_persistence_are_dropped(self):
        """A sample whose lag_1 is null cannot be paired against B1."""
        df = _frame(range(40))
        # Null out the lag_1 source for the very first window: start 0 -> minute 11.
        df = df.with_columns(
            pl.when(
                (pl.col("t") == BASE + timedelta(minutes=T_IN - 1))
                & (pl.col("pair_rank") == 0)
            )
            .then(None)
            .otherwise(pl.col("delta_t_min"))
            .alias("delta_t_min")
        )
        _, X, _, keys = _build(df)
        first = keys.filter(
            (pl.col("start_ts") == BASE) & (pl.col("pair_rank") == 0)
        )
        assert first.height == 0

    def test_guard_rejects_n_lags_over_t_in(self):
        df = _frame(range(40))
        idx = make_sample_index(df, horizon=3, T_in=T_IN)
        with pytest.raises(ValueError, match="exceeds T_in"):
            build_contiguous_features(
                df, idx, horizon=3, T_in=N_LAGS - 1, max_N_by_direction=MAX_N_BY_DIR
            )


class TestMissingValues:
    """Same NaN-vs-null defect as the DL loader, different blast radius.

    `_dense_grid` had the identical `pl.Series(arr).is_null()` mistake. Here it
    was harmless: the grid already defaults to NaN, so writing NaN over NaN
    changes nothing, and callers re-check with `np.isnan`. That is luck, not
    design — these tests pin the behaviour so the XGBoost path cannot start
    depending on it.
    """

    def _frame_with_nulls(self):
        df = _frame(range(60))
        return df.with_columns(
            pl.when(
                (pl.col("t") == BASE + timedelta(minutes=20)) & (pl.col("pair_rank") == 1)
            )
            .then(None)
            .otherwise(pl.col("delta_t_min"))
            .alias("delta_t_min")
        )

    def test_rows_whose_target_is_missing_are_dropped(self):
        _, X, y, keys = _build(self._frame_with_nulls())
        assert not np.isnan(y).any(), "a missing target reached the label vector"

    def test_rows_whose_persistence_is_missing_are_dropped(self):
        _, X, _, keys = _build(self._frame_with_nulls())
        assert not np.isnan(X[:, 0]).any(), "lag_1 must be observed for a paired row"
        assert not np.isnan(keys.get_column("y_pred_persist").to_numpy()).any()

    def test_deeper_lags_may_still_be_missing(self):
        """XGBoost handles NaN natively, so lags beyond lag_1 pass through."""
        _, X, _, _ = _build(self._frame_with_nulls())
        assert X.shape[0] > 0  # rows survive despite an interior gap

    def test_result_is_identical_with_and_without_the_null_rows_present(self):
        """The blanked cell must not shift any surviving row's values."""
        clean = _frame(range(60)).filter(
            ~((pl.col("t") == BASE + timedelta(minutes=20)) & (pl.col("pair_rank") == 1))
        )
        _, Xa, ya, _ = _build(clean)
        _, Xb, yb, _ = _build(self._frame_with_nulls())
        assert Xa.shape == Xb.shape
        assert np.allclose(ya, yb)
        assert np.allclose(Xa, Xb, equal_nan=True)
