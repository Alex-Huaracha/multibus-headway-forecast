"""Tests for HeadwayDataset and collate_fn — W4 RED.

All tests in this module require torch. The importorskip at module level skips
the entire file when torch is not installed (AC-DEP-3 torch isolation).

ACs covered: AC-DS-1..6, AC-MASK-1..4, AC-MAXN-1,4 (padding zeroes and masks).
Design refs: spec §4 (AC-DS-*, AC-MASK-*), design §2.5, §4, §5, INV-4, INV-5, INV-7.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from unittest.mock import patch

from tests.fixtures.headways_factory import make_dataset_fixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ds(T_in: int = 3, T_out: int = 1, max_N: int = 2, n_snapshots: int = 6):
    """Return a HeadwayDataset built from the deterministic fixture."""
    from src.data.dataset import HeadwayDataset

    df, window_index, max_N_by_direction = make_dataset_fixture(
        T_in=T_in, T_out=T_out, max_N=max_N, n_snapshots=n_snapshots
    )
    return HeadwayDataset(
        df=df,
        window_index=window_index,
        max_N_by_direction=max_N_by_direction,
        T_in=T_in,
        T_out=T_out,
    )


# ---------------------------------------------------------------------------
# TestDatasetConstruction
# ---------------------------------------------------------------------------

class TestDatasetConstruction:

    def test_init_does_not_call_getitem(self):
        """AC-DS-NOMAT-1 / INV-7: __init__ MUST NOT iterate windows.

        Uses mock.patch.object to guarantee __getitem__ is never invoked
        during construction.
        """
        from src.data.dataset import HeadwayDataset

        df, window_index, max_N_by_direction = make_dataset_fixture()

        with patch.object(
            HeadwayDataset,
            "__getitem__",
            side_effect=AssertionError("__init__ must not call __getitem__"),
        ):
            # Instantiation must succeed without triggering the mock.
            ds = HeadwayDataset(
                df=df,
                window_index=window_index,
                max_N_by_direction=max_N_by_direction,
                T_in=3,
                T_out=1,
            )
        assert ds is not None

    def test_len_equals_window_index_length(self):
        """AC-DS-4: len(dataset) == len(window_index)."""
        df, window_index, max_N_by_direction = make_dataset_fixture(
            T_in=3, T_out=1, n_snapshots=6
        )
        from src.data.dataset import HeadwayDataset

        ds = HeadwayDataset(
            df=df,
            window_index=window_index,
            max_N_by_direction=max_N_by_direction,
            T_in=3,
            T_out=1,
        )
        assert len(ds) == len(window_index)
        assert len(ds) > 0


# ---------------------------------------------------------------------------
# TestGetItem
# ---------------------------------------------------------------------------

class TestGetItem:

    def test_input_shape_T_in_by_max_N(self):
        """AC-DS-2 / INV-4: input.shape == (T_in, max_N)."""
        ds = _make_ds(T_in=3, T_out=1, max_N=2)
        item = ds[0]
        assert item["input"].shape == (3, 2)

    def test_target_shape_T_out_by_max_N(self):
        """AC-DS-2 / INV-4: target.shape == (T_out, max_N)."""
        ds = _make_ds(T_in=3, T_out=1, max_N=2)
        item = ds[0]
        assert item["target"].shape == (1, 2)

    def test_mask_shapes_match_data(self):
        """AC-DS-2 / INV-4: input_mask.shape == input.shape, target_mask.shape == target.shape."""
        ds = _make_ds(T_in=3, T_out=1, max_N=2)
        item = ds[0]
        assert item["input_mask"].shape == item["input"].shape
        assert item["target_mask"].shape == item["target"].shape

    def test_context_shape_T_in_by_5(self):
        """AC-DS-2 / INV-4: context.shape == (T_in, 5)."""
        ds = _make_ds(T_in=3, T_out=1, max_N=2)
        item = ds[0]
        assert item["context"].shape == (3, 5)

    def test_dtype_float32_for_data_bool_for_masks(self):
        """AC-DS-3 / INV-4: float32 for input/target/context; bool for masks."""
        ds = _make_ds()
        item = ds[0]
        assert item["input"].dtype == torch.float32
        assert item["target"].dtype == torch.float32
        assert item["context"].dtype == torch.float32
        assert item["input_mask"].dtype == torch.bool
        assert item["target_mask"].dtype == torch.bool

    def test_present_slot_mask_is_true(self):
        """AC-MASK-1 / INV-5: True = VALID. Present non-null slot → mask True."""
        ds = _make_ds(T_in=3, T_out=1, max_N=2)
        item = ds[0]
        # All positions in this fixture are filled (max_N pair_ranks all present,
        # delta_t_min is non-null for all rows), so every mask entry must be True.
        assert item["input_mask"].all().item()

    def test_absent_slot_mask_is_false_and_value_zero(self):
        """AC-MASK-2 / INV-5: Absent slot → mask False, value 0.0.

        Build a fixture where max_N=3 but only 2 pair_ranks exist in the data.
        The third column (pair_rank index 2) must be padded with zeros and False mask.
        """
        from src.data.dataset import HeadwayDataset
        from src.data.normalization import NormalizationStats, apply_zscore
        from src.data.context_features import encode_context
        from src.data.windowing import make_window_index
        import polars as pl
        from datetime import datetime, timedelta

        empresaid = 2
        direction = -1
        anchor = datetime(2023, 11, 1, 8, 0, 0)
        n_snapshots = 5
        actual_pair_ranks = 2  # only 0 and 1 exist

        rows = []
        for snap_idx in range(n_snapshots):
            t = anchor + snap_idx * timedelta(minutes=1)
            for pr in range(actual_pair_ranks):
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
        stats = NormalizationStats(
            means={(empresaid, direction): 0.0},
            stds={(empresaid, direction): 1.0},
        )
        df = apply_zscore(df, stats)
        df = encode_context(df, atypical_dates=None)

        max_N = 3  # one more slot than exists in data
        window_index = make_window_index(df, T_in=3, T_out=1)
        max_N_by_direction = {(empresaid, direction): max_N}

        ds = HeadwayDataset(
            df=df,
            window_index=window_index,
            max_N_by_direction=max_N_by_direction,
            T_in=3,
            T_out=1,
        )
        item = ds[0]

        # Column index 2 (pair_rank=2) does not exist in the data.
        assert item["input"][:, 2].eq(0.0).all().item(), "Padded column must be 0.0"
        assert (~item["input_mask"][:, 2]).all().item(), "Padded column mask must be False"


# ---------------------------------------------------------------------------
# TestCollate
# ---------------------------------------------------------------------------

class TestCollate:

    def test_collate_stacks_along_batch_axis(self):
        """AC-DS-5 / AC-DS-6: collate_fn stacks dicts into batched tensors on dim 0."""
        from src.data.dataset import collate_fn

        ds = _make_ds(T_in=3, T_out=1, max_N=2)
        batch = [ds[i] for i in range(min(4, len(ds)))]
        result = collate_fn(batch)

        B = len(batch)
        assert result["input"].shape == (B, 3, 2)
        assert result["target"].shape == (B, 1, 2)
        assert result["input_mask"].shape == (B, 3, 2)
        assert result["target_mask"].shape == (B, 1, 2)
        assert result["context"].shape == (B, 3, 5)

    def test_collate_preserves_per_key_dtype(self):
        """AC-DS-5: collate_fn preserves float32 for data tensors and bool for masks."""
        from src.data.dataset import collate_fn

        ds = _make_ds()
        batch = [ds[0], ds[1]]
        result = collate_fn(batch)

        assert result["input"].dtype == torch.float32
        assert result["target"].dtype == torch.float32
        assert result["context"].dtype == torch.float32
        assert result["input_mask"].dtype == torch.bool
        assert result["target_mask"].dtype == torch.bool

    def test_dataloader_iterates_without_error(self):
        """AC-DS-6: DataLoader with collate_fn iterates over a small dataset."""
        from src.data.dataset import collate_fn
        from torch.utils.data import DataLoader

        ds = _make_ds(T_in=3, T_out=1, max_N=2, n_snapshots=8)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, shuffle=False)
        batches = list(loader)
        assert len(batches) > 0
        first = batches[0]
        assert "input" in first
        assert "target" in first
        assert "input_mask" in first
        assert "target_mask" in first
        assert "context" in first
