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


# ---------------------------------------------------------------------------
# Fase 6.5: horizon parameter (AC-DS-H1..H5)
# ---------------------------------------------------------------------------

def _make_ds_horizon(T_in: int, horizon: int, max_N: int = 2, n_snapshots: int = 20):
    """Build a HeadwayDataset with the given horizon param.

    The fixture assigns delta_t_min_z = float(snap_idx + pr + 1) so values are
    known and distinct per (snap_idx, pair_rank). With stats mean=0/std=1 the
    z-scored values equal the raw values.
    """
    from src.data.dataset import HeadwayDataset
    from src.data.windowing import make_window_index

    df, _, max_N_by_direction = make_dataset_fixture(
        T_in=T_in, T_out=1, max_N=max_N, n_snapshots=n_snapshots
    )
    window_index = make_window_index(df, T_in=T_in, horizon=horizon)
    return HeadwayDataset(
        df=df,
        window_index=window_index,
        max_N_by_direction=max_N_by_direction,
        T_in=T_in,
        T_out=1,
        horizon=horizon,
    )


class TestDatasetHorizon:
    """Strict TDD RED tests for HeadwayDataset with horizon parameter (Fase 6.5).

    All tests must fail before implementation (TypeError or wrong value).
    """

    def test_dataset_target_is_plus_h_row(self):
        """AC-DS-H1 (LOAD-BEARING): T_in=4, horizon=3 → target row index = 6, NOT 4.

        Fixture row values for pair_rank=0: snap_idx + 0 + 1 = snap_idx + 1.
        Window 0 spans snap_idx [0..6] (T_in+h-1=6).
        target_row = T_in + horizon - 1 = 6 → value for pr=0 = 6 + 1 = 7.0.
        The h=1 row (snap_idx=4) would be 5.0 — different, so off-by-h is detectable.
        """
        ds = _make_ds_horizon(T_in=4, horizon=3, max_N=2, n_snapshots=20)
        item = ds[0]
        target = item["target"]  # shape (1, max_N)
        # pair_rank=0 at snap_idx=6: value = 6 + 0 + 1 = 7.0
        assert abs(float(target[0, 0]) - 7.0) < 1e-4, (
            f"LOAD-BEARING: target[0,0] should be 7.0 (row T_in+h-1=6), got {float(target[0, 0])}"
        )

    def test_dataset_target_is_NOT_plus_1_row_when_h_gt_1(self):
        """AC-DS-H2 (anti-leakage): with horizon=3, target != value at row T_in (snap_idx=4).

        At snap_idx=4, pair_rank=0 has value 5.0. Target must NOT be 5.0.
        """
        ds = _make_ds_horizon(T_in=4, horizon=3, max_N=2, n_snapshots=20)
        item = ds[0]
        target_value = float(item["target"][0, 0])
        plus_1_value = 5.0  # snap_idx=4, pr=0: 4+0+1=5
        assert abs(target_value - plus_1_value) > 1e-4, (
            f"Anti-leakage: target should NOT equal the h=1 row (5.0), got {target_value}"
        )

    def test_dataset_target_shape_h1(self):
        """AC-DS-H3: target.shape == (1, max_N) for horizon=1."""
        ds = _make_ds_horizon(T_in=4, horizon=1, max_N=3, n_snapshots=20)
        item = ds[0]
        assert item["target"].shape == (1, 3), (
            f"Expected (1, 3), got {item['target'].shape}"
        )

    def test_dataset_target_shape_h3(self):
        """AC-DS-H3: target.shape == (1, max_N) for horizon=3."""
        ds = _make_ds_horizon(T_in=4, horizon=3, max_N=3, n_snapshots=20)
        item = ds[0]
        assert item["target"].shape == (1, 3), (
            f"Expected (1, 3), got {item['target'].shape}"
        )

    def test_dataset_target_shape_h5(self):
        """AC-DS-H3: target.shape == (1, max_N) for horizon=5."""
        ds = _make_ds_horizon(T_in=4, horizon=5, max_N=3, n_snapshots=20)
        item = ds[0]
        assert item["target"].shape == (1, 3), (
            f"Expected (1, 3), got {item['target'].shape}"
        )

    def test_dataset_horizon_1_regression(self):
        """AC-DS-H4 / h=1 regression: horizon=1 target == current behavior (values[T_in:T_in+1])."""
        from src.data.dataset import HeadwayDataset

        T_in, max_N = 3, 2
        df, window_index_T_out, max_N_by_direction = make_dataset_fixture(
            T_in=T_in, T_out=1, max_N=max_N, n_snapshots=10
        )
        from src.data.windowing import make_window_index
        window_index_h1 = make_window_index(df, T_in=T_in, horizon=1)

        ds_old = HeadwayDataset(
            df=df,
            window_index=window_index_T_out,
            max_N_by_direction=max_N_by_direction,
            T_in=T_in,
            T_out=1,
        )
        ds_new = HeadwayDataset(
            df=df,
            window_index=window_index_h1,
            max_N_by_direction=max_N_by_direction,
            T_in=T_in,
            T_out=1,
            horizon=1,
        )
        # Both datasets should produce the same target for corresponding windows.
        for i in range(min(len(ds_old), len(ds_new))):
            old_target = ds_old[i]["target"]
            new_target = ds_new[i]["target"]
            assert torch.allclose(old_target, new_target), (
                f"Window {i}: horizon=1 target differs from T_out=1 target.\n"
                f"old={old_target.tolist()}, new={new_target.tolist()}"
            )

    def test_dataset_squeeze1_compat(self):
        """AC-DS-H4 (squeeze compat): target.squeeze(1) → (B, max_N) for h∈{1,3,5,10}."""
        from src.data.dataset import collate_fn
        from torch.utils.data import DataLoader

        for h in [1, 3, 5, 10]:
            ds = _make_ds_horizon(T_in=4, horizon=h, max_N=2, n_snapshots=30)
            loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=False)
            batch = next(iter(loader))
            squeezed = batch["target"].squeeze(1)
            assert squeezed.shape == (4, 2), (
                f"horizon={h}: squeeze(1) should give (B=4, max_N=2), got {squeezed.shape}"
            )

    def test_dataset_mask_correctness_at_plus_h(self):
        """AC-DS-H5: row T_in+horizon-1 has a null for pair_rank 0 → target_mask[0,0] is False.

        We inject a null by creating a custom fixture where snap_idx=T_in+h-1, pr=0 is null.
        """
        import polars as pl
        from datetime import datetime, timedelta
        from src.data.dataset import HeadwayDataset
        from src.data.windowing import make_window_index
        from src.data.normalization import NormalizationStats, apply_zscore
        from src.data.context_features import encode_context

        T_in, h, max_N = 3, 2, 2
        # target_row = T_in + h - 1 = 4 (0-indexed). We want n_snapshots > T_in+h.
        n_snaps = 10
        anchor = datetime(2023, 11, 1, 8, 0, 0)

        rows: list[dict] = []
        for snap_idx in range(n_snaps):
            t = anchor + timedelta(minutes=snap_idx)
            for pr in range(max_N):
                # Make snap_idx=4, pr=0 null to trigger mask=False at target row.
                val = None if (snap_idx == T_in + h - 1 and pr == 0) else float(snap_idx + pr + 1)
                rows.append({
                    "empresaid": 2,
                    "t": t,
                    "direction": -1,
                    "pair_rank": pr,
                    "delta_t_min": val,
                })

        df = pl.DataFrame(rows).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        stats = NormalizationStats(
            means={(2, -1): 0.0},
            stds={(2, -1): 1.0},
        )
        df = apply_zscore(df, stats)
        df = encode_context(df, atypical_dates=None)

        window_index = make_window_index(df, T_in=T_in, horizon=h)
        ds = HeadwayDataset(
            df=df,
            window_index=window_index,
            max_N_by_direction={(2, -1): max_N},
            T_in=T_in,
            T_out=1,
            horizon=h,
        )
        item = ds[0]
        # target row is snap_idx=4, pr=0 has null → mask should be False.
        assert item["target_mask"][0, 0] == False, (
            f"target_mask[0,0] should be False (null at target row), got {item['target_mask'][0, 0]}"
        )
