"""Tests for the full-key residual export contract.

The defect being guarded: an export that drops part of the key looks fine until
someone needs to pair it with another model's residuals, at which point the only
remedy is another training run. These tests make the key's uniqueness and its
alignment with the population structural rather than assumed.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.evaluation.residual_export import (
    RESIDUAL_COLUMNS,
    RESIDUAL_KEY_COLUMNS,
    assert_key_is_unique,
    build_keyed_residuals,
    direction_label,
)

BASE = datetime(2024, 2, 8, 8, 0)
MAX_N = 3


def _index(n_samples: int, *, horizon: int = 5, direction: int = 1) -> pl.DataFrame:
    starts = [BASE + timedelta(minutes=i) for i in range(n_samples)]
    return pl.DataFrame(
        {
            "empresaid": [2] * n_samples,
            "direction": [direction] * n_samples,
            "start_ts": starts,
            "target_ts": [s + timedelta(minutes=11 + horizon) for s in starts],
            "horizon": [horizon] * n_samples,
        }
    )


def _arrays(n_samples: int, *, all_valid: bool = True):
    shape = (n_samples, MAX_N)
    y_true = np.arange(n_samples * MAX_N, dtype=np.float64).reshape(shape)
    y_pred = y_true + 0.5
    persist = y_true - 0.25
    mask = np.ones(shape, dtype=bool)
    if not all_valid:
        mask[0, 0] = False
        mask[2, 1] = False
    return y_true, y_pred, persist, mask, mask.copy()


def _build(n_samples=6, *, all_valid=True, **kw):
    idx = _index(n_samples, **kw)
    y_true, y_pred, persist, tmask, pmask = _arrays(n_samples, all_valid=all_valid)
    return idx, build_keyed_residuals(
        idx,
        corridor="E2",
        split="test",
        y_true=y_true,
        y_pred_model=y_pred,
        y_pred_persist=persist,
        target_mask=tmask,
        persist_mask=pmask,
    )


class TestSchema:
    def test_columns_and_order(self):
        _, res = _build()
        assert res.columns == RESIDUAL_COLUMNS

    def test_key_precedes_values(self):
        assert RESIDUAL_COLUMNS[: len(RESIDUAL_KEY_COLUMNS)] == RESIDUAL_KEY_COLUMNS

    def test_carries_what_the_legacy_export_dropped(self):
        """`t` and `pair_rank` are exactly what blocked pendings #5 and #6."""
        _, res = _build()
        for col in ("start_ts", "target_ts", "pair_rank", "split"):
            assert col in res.columns


class TestKeyUniqueness:
    def test_key_is_unique(self):
        _, res = _build(n_samples=20)
        assert_key_is_unique(res)
        keys = res.select(RESIDUAL_KEY_COLUMNS)
        assert keys.height == keys.unique().height

    def test_target_timestamp_alone_is_not_unique(self):
        """The precise reason `t` was an invalid join key: pair_rank multiplies it."""
        _, res = _build(n_samples=6)
        by_ts = res.select(["target_ts", "direction"])
        assert by_ts.height > by_ts.unique().height

    def test_duplicated_key_is_detected(self):
        _, res = _build(n_samples=4)
        doubled = pl.concat([res, res])
        with pytest.raises(ValueError, match="residual key is not unique"):
            assert_key_is_unique(doubled)


class TestAlignment:
    def test_row_count_matches_valid_cells(self):
        _, res = _build(n_samples=6)
        assert res.height == 6 * MAX_N

    def test_masked_cells_are_dropped(self):
        _, res = _build(n_samples=6, all_valid=False)
        assert res.height == 6 * MAX_N - 2

    def test_values_land_on_the_right_key(self):
        """A transposed reshape would scramble this; the encoding catches it."""
        idx, res = _build(n_samples=6)
        row = res.filter(
            (pl.col("start_ts") == BASE + timedelta(minutes=2))
            & (pl.col("pair_rank") == 1)
        )
        assert row.height == 1
        # y_true was arange over (n_samples, MAX_N): sample 2, position 1.
        assert row.get_column("y_true")[0] == pytest.approx(2 * MAX_N + 1)

    def test_misaligned_arrays_are_rejected(self):
        idx = _index(6)
        y_true, y_pred, persist, tmask, pmask = _arrays(5)
        with pytest.raises(ValueError, match="misaligned with its population"):
            build_keyed_residuals(
                idx,
                corridor="E2",
                split="test",
                y_true=y_true,
                y_pred_model=y_pred,
                y_pred_persist=persist,
                target_mask=tmask,
                persist_mask=pmask,
            )

    def test_shape_disagreement_is_rejected(self):
        idx = _index(6)
        y_true, y_pred, persist, tmask, pmask = _arrays(6)
        with pytest.raises(ValueError, match="disagree in shape"):
            build_keyed_residuals(
                idx,
                corridor="E2",
                split="test",
                y_true=y_true,
                y_pred_model=y_pred[:, :2],
                y_pred_persist=persist,
                target_mask=tmask,
                persist_mask=pmask,
            )


class TestDirectionLabel:
    @pytest.mark.parametrize("value,expected", [(1, "+1"), (-1, "-1")])
    def test_matches_the_legacy_convention(self, value, expected):
        assert direction_label(value) == expected

    def test_both_directions_survive_the_export(self):
        _, plus = _build(direction=1)
        _, minus = _build(direction=-1)
        assert plus.get_column("direction").unique().to_list() == ["+1"]
        assert minus.get_column("direction").unique().to_list() == ["-1"]
