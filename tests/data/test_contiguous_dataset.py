"""Tests for the sample-index-backed loader.

Covers the tensor contract (so ``train.py`` and ``collate_fn`` keep working
unchanged), the contiguity guarantee at materialization time, and the removal of
the leaking atypical flag.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from src.data.contiguous_dataset import (
    CAUSAL_CONTEXT_FEATURE_NAMES,
    ContiguousHeadwayDataset,
)
from src.data.sample_index import make_sample_index

T_IN = 12
MAX_N = 4
BASE = datetime(2024, 1, 8, 8, 0)
MAX_N_BY_DIR = {(2, 1): MAX_N, (2, -1): MAX_N}


def _frame(offsets_minutes, *, empresaid=2, direction=1, pair_ranks=(0, 1, 2)):
    """Frame with the z-scored value column and the causal context columns."""
    rows = []
    for off in offsets_minutes:
        ts = BASE + timedelta(minutes=int(off))
        hour_angle = 2 * math.pi * ts.hour / 24
        dow_angle = 2 * math.pi * ts.weekday() / 7
        for pr in pair_ranks:
            rows.append(
                {
                    "empresaid": empresaid,
                    "direction": direction,
                    "pair_rank": pr,
                    "t": ts,
                    # Encodes (offset, pair_rank) so a misaligned slice is visible.
                    "delta_t_min_z": float(off) + pr / 10.0,
                    "hour_sin": math.sin(hour_angle),
                    "hour_cos": math.cos(hour_angle),
                    "dow_sin": math.sin(dow_angle),
                    "dow_cos": math.cos(dow_angle),
                }
            )
    return pl.DataFrame(rows)


def _dataset(df, horizon):
    idx = make_sample_index(df, horizon=horizon, T_in=T_IN)
    return idx, ContiguousHeadwayDataset(
        df,
        idx,
        max_N_by_direction=MAX_N_BY_DIR,
        T_in=T_IN,
        horizon=horizon,
    )


class TestTensorContract:
    def test_shapes_match_the_legacy_contract(self):
        idx, ds = _dataset(_frame(range(60)), 5)
        item = ds[0]
        assert item["input"].shape == (T_IN, MAX_N)
        assert item["target"].shape == (1, MAX_N)
        assert item["input_mask"].shape == (T_IN, MAX_N)
        assert item["target_mask"].shape == (1, MAX_N)
        assert item["context"].shape == (T_IN, len(CAUSAL_CONTEXT_FEATURE_NAMES))
        assert item["input"].dtype == torch.float32
        assert item["input_mask"].dtype == torch.bool

    def test_len_equals_index_height(self):
        idx, ds = _dataset(_frame(range(60)), 3)
        assert len(ds) == idx.height

    def test_absent_pair_ranks_are_masked_not_zero_valued(self):
        """pair_rank 3 is never present, so its column must be all-False."""
        _, ds = _dataset(_frame(range(40), pair_ranks=(0, 1, 2)), 3)
        item = ds[0]
        assert not item["input_mask"][:, 3].any()
        assert torch.all(item["input"][:, 3] == 0.0)
        assert item["input_mask"][:, :3].all()


class TestContiguityAtMaterialization:
    def test_target_is_exactly_horizon_minutes_after_window_end(self):
        for horizon in (1, 3, 5, 10):
            df = _frame(range(60))
            idx, ds = _dataset(df, horizon)
            for i in (0, len(ds) // 2, len(ds) - 1):
                start = idx.get_column("start_ts").to_numpy()[i]
                target = idx.get_column("target_ts").to_numpy()[i]
                gap = (target - start) / np.timedelta64(1, "m")
                assert gap == T_IN - 1 + horizon

    def test_values_align_with_the_declared_window(self):
        """The slice must hold the encoded offsets, not a shifted run."""
        df = _frame(range(60))
        idx, ds = _dataset(df, 5)
        item = ds[3]
        start_off = 3  # window index 3 starts at BASE + 3 minutes
        expected_input = np.array(
            [[float(start_off + k) + pr / 10.0 for pr in range(3)] for k in range(T_IN)],
            dtype=np.float32,
        )
        assert np.allclose(item["input"][:, :3].numpy(), expected_input)
        expected_target = np.array(
            [[float(start_off + T_IN - 1 + 5) + pr / 10.0 for pr in range(3)]],
            dtype=np.float32,
        )
        assert np.allclose(item["target"][:, :3].numpy(), expected_target)

    def test_no_sample_bridges_a_gap(self):
        """A frame split by a 23-hour jump yields only within-run samples."""
        offsets = list(range(40)) + [40 + 23 * 60 + k for k in range(40)]
        df = _frame(offsets)
        idx, ds = _dataset(df, 3)

        span_minutes = T_IN - 1 + 3
        starts = idx.get_column("start_ts").to_numpy()
        targets = idx.get_column("target_ts").to_numpy()
        gaps = (targets - starts) / np.timedelta64(1, "m")
        assert np.all(gaps == span_minutes)

        # Every item materializes without tripping the defense-in-depth check.
        for i in range(len(ds)):
            ds[i]


class TestAtypicalFlagRemoved:
    def test_default_context_has_four_causal_columns(self):
        assert CAUSAL_CONTEXT_FEATURE_NAMES == (
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        )
        assert "atypical_flag" not in CAUSAL_CONTEXT_FEATURE_NAMES

    def test_passing_the_flag_is_rejected(self):
        df = _frame(range(40)).with_columns(pl.lit(0.0).alias("atypical_flag"))
        idx = make_sample_index(df, horizon=3, T_in=T_IN)
        with pytest.raises(ValueError, match="atypical_flag is a leaking feature"):
            ContiguousHeadwayDataset(
                df,
                idx,
                max_N_by_direction=MAX_N_BY_DIR,
                T_in=T_IN,
                horizon=3,
                context_cols=CAUSAL_CONTEXT_FEATURE_NAMES + ("atypical_flag",),
            )

    def test_missing_context_column_is_rejected(self):
        df = _frame(range(40)).drop("dow_cos")
        idx = make_sample_index(df, horizon=3, T_in=T_IN)
        with pytest.raises(ValueError, match="missing context columns"):
            ContiguousHeadwayDataset(
                df, idx, max_N_by_direction=MAX_N_BY_DIR, T_in=T_IN, horizon=3
            )


class TestNoFleetDensityWeighting:
    def test_item_count_is_independent_of_pair_rank_count(self):
        thin_idx, thin = _dataset(_frame(range(50), pair_ranks=(0,)), 5)
        wide_idx, wide = _dataset(_frame(range(50), pair_ranks=(0, 1, 2)), 5)
        assert len(thin) == len(wide) == thin_idx.height == wide_idx.height


class TestMaterializeArrays:
    """`materialize_arrays` is the path the Kaggle notebooks take.

    It must agree item-for-item with the Dataset, or the trained model and the
    tested contract describe different things.
    """

    def _materialize(self, df, horizon):
        from src.data.contiguous_dataset import materialize_arrays

        idx = make_sample_index(df, horizon=horizon, T_in=T_IN)
        arrays = materialize_arrays(
            df, idx, max_N=MAX_N, T_in=T_IN, horizon=horizon
        )
        return idx, arrays

    def test_shapes(self):
        idx, arrays = self._materialize(_frame(range(60)), 5)
        n = idx.height
        assert arrays["input"].shape == (n, T_IN, MAX_N)
        assert arrays["target"].shape == (n, 1, MAX_N)
        assert arrays["input_mask"].shape == (n, T_IN, MAX_N)
        assert arrays["context"].shape == (n, T_IN, len(CAUSAL_CONTEXT_FEATURE_NAMES))

    def test_agrees_with_the_dataset_item_for_item(self):
        df = _frame(range(50))
        idx, ds = _dataset(df, 3)
        _, arrays = self._materialize(df, 3)
        for i in (0, len(ds) // 3, len(ds) - 1):
            item = ds[i]
            assert np.allclose(item["input"].numpy(), arrays["input"][i])
            assert np.allclose(item["target"].numpy(), arrays["target"][i])
            assert np.array_equal(item["input_mask"].numpy(), arrays["input_mask"][i])
            assert np.array_equal(item["target_mask"].numpy(), arrays["target_mask"][i])
            assert np.allclose(item["context"].numpy(), arrays["context"][i])

    def test_both_directions_are_filled(self):
        df = pl.concat(
            [_frame(range(40), direction=1), _frame(range(40), direction=-1)]
        )
        idx, arrays = self._materialize(df, 3)
        assert idx.height > 0
        # Every row must have been written by some series, not left at zero.
        assert arrays["input_mask"].any(axis=(1, 2)).all()

    def test_rejects_the_atypical_flag(self):
        from src.data.contiguous_dataset import materialize_arrays

        df = _frame(range(40)).with_columns(pl.lit(0.0).alias("atypical_flag"))
        idx = make_sample_index(df, horizon=3, T_in=T_IN)
        with pytest.raises(ValueError, match="atypical_flag is a leaking feature"):
            materialize_arrays(
                df,
                idx,
                max_N=MAX_N,
                T_in=T_IN,
                horizon=3,
                context_cols=CAUSAL_CONTEXT_FEATURE_NAMES + ("atypical_flag",),
            )

    def test_empty_index_yields_empty_arrays(self):
        idx, arrays = self._materialize(_frame(range(5)), 10)
        assert idx.height == 0
        assert arrays["input"].shape[0] == 0


class TestMissingValuesNeverReachTheTensors:
    """Regression guard for the NaN-vs-null defect.

    Converting a polars column to numpy turns nulls into NaN and discards the
    null flag, so `pl.Series(arr).is_null()` reports False for every one of
    them. The first version of `_SeriesGrid` checked missingness that way, so
    NaN entered the tensor with mask=True. Training then produced a NaN loss,
    no epoch ever improved, and `state_dict` came back empty — surfacing far
    away as "Missing key(s) in state_dict".

    Every fixture in this file was fully populated, which is exactly why the
    suite stayed green while the notebook failed on Kaggle.
    """

    def _frame_with_nulls(self):
        df = _frame(range(60))
        # Blank a handful of observations across different timestamps and slots.
        return df.with_columns(
            pl.when(
                ((pl.col("t") == BASE + timedelta(minutes=5)) & (pl.col("pair_rank") == 1))
                | ((pl.col("t") == BASE + timedelta(minutes=20)) & (pl.col("pair_rank") == 0))
                | (pl.col("t") == BASE + timedelta(minutes=33))
            )
            .then(None)
            .otherwise(pl.col("delta_t_min_z"))
            .alias("delta_t_min_z")
        )

    def test_dataset_items_contain_no_nan(self):
        _, ds = _dataset(self._frame_with_nulls(), 3)
        for i in range(len(ds)):
            item = ds[i]
            assert not torch.isnan(item["input"]).any(), f"NaN in input at {i}"
            assert not torch.isnan(item["target"]).any(), f"NaN in target at {i}"

    def test_materialized_arrays_contain_no_nan(self):
        from src.data.contiguous_dataset import materialize_arrays

        df = self._frame_with_nulls()
        idx = make_sample_index(df, horizon=3, T_in=T_IN)
        arrays = materialize_arrays(df, idx, max_N=MAX_N, T_in=T_IN, horizon=3)
        for key in ("input", "target", "context"):
            assert not np.isnan(arrays[key]).any(), f"NaN in {key}"

    def test_missing_cells_are_masked_out(self):
        """A blanked observation must read as value 0.0 with mask False."""
        from src.data.contiguous_dataset import materialize_arrays

        df = self._frame_with_nulls()
        idx = make_sample_index(df, horizon=3, T_in=T_IN)
        arrays = materialize_arrays(df, idx, max_N=MAX_N, T_in=T_IN, horizon=3)

        starts = idx.get_column("start_ts").to_numpy()
        row = int(np.nonzero(starts == np.datetime64(BASE))[0][0])
        # Minute 5 is offset 5 inside the window that starts at BASE.
        assert arrays["input_mask"][row, 5, 1] == False  # noqa: E712
        assert arrays["input"][row, 5, 1] == 0.0
        # Its neighbours stay observed.
        assert arrays["input_mask"][row, 5, 0] == True  # noqa: E712

    def test_a_masked_loss_over_these_tensors_is_finite(self):
        """The end-to-end symptom: the loss must not be NaN."""
        from src.data.contiguous_dataset import materialize_arrays

        df = self._frame_with_nulls()
        idx = make_sample_index(df, horizon=3, T_in=T_IN)
        a = materialize_arrays(df, idx, max_N=MAX_N, T_in=T_IN, horizon=3)

        target = torch.from_numpy(a["target"][:, 0])
        mask = torch.from_numpy(a["target_mask"][:, 0])
        pred = torch.zeros_like(target)
        loss = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        assert torch.isfinite(loss), "masked loss went non-finite — the Kaggle failure"
