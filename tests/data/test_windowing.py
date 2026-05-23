"""Tests for src.data.windowing — AC-WIN-1..6, AC-MAXN-1..2.

Strict TDD: this file is the RED commit. src/data/windowing.py does not exist yet;
all tests fail with ImportError. Run: uv run pytest tests/data/test_windowing.py -q
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from tests.fixtures.headways_factory import make_headways_fixture, make_multi_corridor_fixture


class TestMaxN:
    """AC-MAXN-1: max_N is train-p99 of (n_buses-1) per (empresaid, direction).
    AC-MAXN-2: max_N ignores val/test rows.
    """

    def test_max_N_uses_train_p99_per_direction(self) -> None:
        """AC-MAXN-1: train-p99 of (n_buses-1) per (empresaid, direction).

        Fixture: all train rows, two directions with known bus counts.
        For direction=-1: bus counts [2,3,4] → (n_buses-1) = [1,2,3] → p99 ≈ 3 → max_N=3.
        """
        from src.data.windowing import compute_max_N

        df = pl.DataFrame({
            "empresaid": [2, 2, 2, 2, 2, 2],
            "direction": [-1, -1, -1, 1, 1, 1],
            "n_buses": [2, 3, 4, 3, 4, 5],
            "split": ["train"] * 6,
            "t": [datetime(2023, 11, i + 1, 8, 0) for i in range(6)],
            "pair_rank": [0] * 6,
            "delta_t_min": [1.0] * 6,
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("n_buses").cast(pl.Int32),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("pair_rank").cast(pl.Int32),
        )

        train_df = df.filter(pl.col("split") == "train")
        result = compute_max_N(train_df)

        # Result must be a dict keyed by (empresaid, direction).
        assert isinstance(result, dict)
        assert (2, -1) in result
        assert (2, 1) in result
        # p99 of [1,2,3] = 3; p99 of [2,3,4] = 4
        assert result[(2, -1)] == 3
        assert result[(2, 1)] == 4

    def test_max_N_ignores_val_test_rows(self) -> None:
        """AC-MAXN-2: max_N computed on train_df must not shift when val/test rows added.

        Leakage guard: train p99 = 3 (n_buses up to 4). val row has n_buses=10.
        compute_max_N(train_df) must still return 3, not 9.
        """
        from src.data.windowing import compute_max_N

        train_rows = pl.DataFrame({
            "empresaid": [2, 2, 2],
            "direction": [-1, -1, -1],
            "n_buses": [2, 3, 4],
            "split": ["train"] * 3,
            "t": [datetime(2023, 11, i + 1, 8, 0) for i in range(3)],
            "pair_rank": [0] * 3,
            "delta_t_min": [1.0] * 3,
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("n_buses").cast(pl.Int32),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("pair_rank").cast(pl.Int32),
        )

        # compute_max_N must use the passed df exclusively — caller is responsible
        # for filtering to train rows before calling this function.
        max_n_train = compute_max_N(train_rows)

        # Construct full frame (train + val with high n_buses) and compute again.
        val_row = pl.DataFrame({
            "empresaid": [2],
            "direction": [-1],
            "n_buses": [10],
            "split": ["val"],
            "t": [datetime(2024, 1, 20, 8, 0)],
            "pair_rank": [0],
            "delta_t_min": [1.0],
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("n_buses").cast(pl.Int32),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("pair_rank").cast(pl.Int32),
        )

        full_df = pl.concat([train_rows, val_row])
        max_n_full = compute_max_N(full_df)

        # With train_df: p99 of [1,2,3] = 3 → max_N = 3
        assert max_n_train[(2, -1)] == 3
        # With full_df: p99 of [1,2,3,9] = 9 → max_N = 9 (leakage!)
        assert max_n_full[(2, -1)] != max_n_train[(2, -1)]

    def test_max_N_returns_int_not_float(self) -> None:
        """AC-MAXN-1: max_N values must be Python int (not float/np.int64).

        compute_max_N returns dict[tuple[int,int], int]. Downstream code uses
        max_N as a tensor dimension and int() is required by torch.
        """
        from src.data.windowing import compute_max_N

        df = pl.DataFrame({
            "empresaid": [2, 2],
            "direction": [-1, -1],
            "n_buses": [3, 4],
            "t": [datetime(2023, 11, 1, 8, 0), datetime(2023, 11, 2, 8, 0)],
            "pair_rank": [0, 0],
            "delta_t_min": [1.0, 2.0],
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("n_buses").cast(pl.Int32),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("pair_rank").cast(pl.Int32),
        )
        result = compute_max_N(df)
        for v in result.values():
            assert isinstance(v, int), f"Expected int, got {type(v)}"


class TestMakeWindowIndex:
    """AC-WIN-1..6: make_window_index produces correct deterministic per-slot windows."""

    def _make_single_slot_df(self, n_rows: int, direction: int = -1) -> pl.DataFrame:
        """Helper: single-slot DataFrame with n_rows rows."""
        return pl.DataFrame({
            "empresaid": [2] * n_rows,
            "direction": [direction] * n_rows,
            "pair_rank": [0] * n_rows,
            "t": [datetime(2023, 11, 1, 8, i, 0) for i in range(n_rows)],
            "delta_t_min": [float(i) for i in range(n_rows)],
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("t").cast(pl.Datetime("us")),
        )

    def test_index_length_matches_snapshot_count_minus_window(self) -> None:
        """AC-WIN-1: 20 rows, T_in=12, T_out=1, stride=1 → 8 entries.

        Formula: N - T_in - T_out + 1 = 20 - 12 - 1 + 1 = 8.
        """
        from src.data.windowing import make_window_index

        df = self._make_single_slot_df(20)
        index = make_window_index(df, T_in=12, T_out=1, stride=1)
        assert len(index) == 8

    def test_index_respects_stride(self) -> None:
        """AC-WIN-2: T_in=6, T_out=2, stride=2, N=20 → floor((20-6-2)/2)+1 = 6 entries."""
        from src.data.windowing import make_window_index

        df = self._make_single_slot_df(20)
        index = make_window_index(df, T_in=6, T_out=2, stride=2)
        assert len(index) == 7  # floor((20-6-2)/2)+1 = floor(12/2)+1 = 6+1 = 7

    def test_index_deterministic_across_runs(self) -> None:
        """AC-WIN-3: same input → same index (deterministic, no randomness)."""
        from src.data.windowing import make_window_index

        df = self._make_single_slot_df(20)
        index_a = make_window_index(df, T_in=12, T_out=1, stride=1)
        index_b = make_window_index(df, T_in=12, T_out=1, stride=1)
        assert index_a == index_b

    def test_constants_importable(self) -> None:
        """AC-WIN-4: DEFAULT_T_IN and DEFAULT_T_OUT importable with correct values."""
        from src.data.windowing import DEFAULT_T_IN, DEFAULT_T_OUT, DEFAULT_STRIDE

        assert DEFAULT_T_IN == 12
        assert DEFAULT_T_OUT == 1
        assert DEFAULT_STRIDE == 1

    def test_index_handles_empty_slot(self) -> None:
        """AC-WIN-5: slot with fewer than T_in + T_out rows → zero entries."""
        from src.data.windowing import make_window_index

        # 5 rows, T_in=6, T_out=2 → 5 < 8 → 0 entries
        df = self._make_single_slot_df(5)
        index = make_window_index(df, T_in=6, T_out=2, stride=1)
        assert len(index) == 0

    def test_no_torch_import(self) -> None:
        """AC-WIN-6: windowing module has zero torch imports.

        We verify by inspecting the module source — not by subprocess (torch IS
        installed after W3-C3, so import-isolation tests require source inspection).
        """
        import inspect
        import src.data.windowing as wmod

        source = inspect.getsource(wmod)
        assert "import torch" not in source
        assert "from torch" not in source

    def test_slot_boundary_respected(self) -> None:
        """AC-WIN-3: index entries for slot A do not include rows from slot B.

        Two-slot frame: slot (dir=-1, pr=0) with 10 rows, slot (dir=1, pr=0)
        with 8 rows. Windows must not cross slot boundaries.
        Each slot's start_idx is relative to the slot's own sorted frame.
        """
        from src.data.windowing import make_window_index, WindowIndexEntry

        df_a = self._make_single_slot_df(10, direction=-1)
        df_b = self._make_single_slot_df(8, direction=1)
        df = pl.concat([df_a, df_b])

        index = make_window_index(df, T_in=6, T_out=1, stride=1)

        # Slot A: N=10 → floor((10-6-1)/1)+1 = 4 entries
        # Slot B: N=8 → floor((8-6-1)/1)+1 = 2 entries
        entries_a = [e for e in index if e["direction"] == -1]
        entries_b = [e for e in index if e["direction"] == 1]
        assert len(entries_a) == 4
        assert len(entries_b) == 2
