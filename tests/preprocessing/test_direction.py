"""Tests for src/preprocessing/direction.py.

Covers:
  T1.5a — monotonically increasing s sequence → direction = +1 after window stabilises.
  T1.5b — monotonically decreasing s sequence → direction = -1 after window stabilises.
  Additional: direction dtype is Int8; cross_check_heading is a no-op for E59.
"""
from __future__ import annotations
from datetime import datetime, timedelta

import polars as pl
import pytest

from src.preprocessing.direction import cross_check_heading, infer_direction
from src.preprocessing.config import PRODUCTIVE_PARAMS


def _make_monotonic_gps(
    empresaid: int,
    unidadid: int,
    s_values: list[float],
    t0: datetime | None = None,
) -> pl.DataFrame:
    """Construct a minimal GPS frame with (empresaid, unidadid, time, s) columns."""
    if t0 is None:
        t0 = datetime(2024, 1, 23, 7, 0, 0)
    n = len(s_values)
    return pl.DataFrame({
        "empresaid": [empresaid] * n,
        "unidadid": [unidadid] * n,
        "time": [t0 + timedelta(seconds=i * 20) for i in range(n)],
        "s": s_values,
    })


WIN = PRODUCTIVE_PARAMS.direction_smooth_win   # = 5


class TestDirectionInference:
    """T1.5 — direction inference from sign of rolling ds/dt."""

    def test_ascending_s_gives_positive_direction(self):
        """T1.5a — monotonically increasing s → direction = +1 after window stabilises.

        Failure mode: sign(rolling_mean) computes wrong polarity or the
        over() partition groups incorrectly.

        The first (WIN - 1) pings may be 0 (undetermined, not enough history).
        From ping WIN onward, all should be +1.
        """
        s_vals = [float(i * 100) for i in range(40)]
        gps = _make_monotonic_gps(2, 201, s_vals)
        result = infer_direction(gps)

        dirs = result["direction"].to_list()
        stable_slice = dirs[WIN:]  # after window fills
        assert all(d == 1 for d in stable_slice), (
            f"Expected all +1 for ascending s after win={WIN}; got {set(stable_slice)}"
        )

    def test_descending_s_gives_negative_direction(self):
        """T1.5b — monotonically decreasing s → direction = -1 after window stabilises.

        Failure mode: sign convention reversed (- sign dropped somewhere in
        ds_raw or ds_smooth computation).
        """
        s_vals = [float((40 - i) * 100) for i in range(40)]
        gps = _make_monotonic_gps(2, 201, s_vals)
        result = infer_direction(gps)

        dirs = result["direction"].to_list()
        stable_slice = dirs[WIN:]
        assert all(d == -1 for d in stable_slice), (
            f"Expected all -1 for descending s after win={WIN}; got {set(stable_slice)}"
        )

    def test_direction_dtype_is_int8(self):
        """direction column must be Int8 (not Int32 or larger) per R6 schema."""
        s_vals = [float(i * 100) for i in range(20)]
        gps = _make_monotonic_gps(2, 201, s_vals)
        result = infer_direction(gps)
        assert result["direction"].dtype == pl.Int8, (
            f"Expected Int8; got {result['direction'].dtype}"
        )

    def test_direction_values_only_in_valid_set(self):
        """direction values must be in {-1, 0, +1} only."""
        s_vals = [float(i * 100) for i in range(10)] + [float((20 - i) * 100) for i in range(10)]
        gps = _make_monotonic_gps(2, 201, s_vals)
        result = infer_direction(gps)
        unique_dirs = set(result["direction"].unique().to_list())
        assert unique_dirs.issubset({-1, 0, 1}), (
            f"Unexpected direction values: {unique_dirs}"
        )

    def test_per_bus_direction_is_independent(self):
        """Two buses in opposite directions must infer independently (no cross-bus contamination)."""
        gps_a = _make_monotonic_gps(2, 201, [float(i * 100) for i in range(20)])
        gps_b = _make_monotonic_gps(2, 202, [float((20 - i) * 100) for i in range(20)])
        gps = pl.concat([gps_a, gps_b])
        result = infer_direction(gps)

        bus_a_dirs = set(result.filter(pl.col("unidadid") == 201)["direction"].to_list()[WIN:])
        bus_b_dirs = set(result.filter(pl.col("unidadid") == 202)["direction"].to_list()[WIN:])
        assert bus_a_dirs == {1}, f"Bus 201 (ascending) should be all +1; got {bus_a_dirs}"
        assert bus_b_dirs == {-1}, f"Bus 202 (descending) should be all -1; got {bus_b_dirs}"


class TestCrossCheckHeading:
    """cross_check_heading must be a no-op for E59 (no direccion column, has_heading=False)."""

    def test_noop_for_e59_no_column(self):
        """E59 has no direccion column and has_heading=False. cross_check_heading
        must return the frame unchanged (no heading_agrees column added).

        Failure mode: the function ignores has_heading and crashes on missing column.
        """
        s_vals = [float(i * 100) for i in range(10)]
        gps = _make_monotonic_gps(59, 501, s_vals)
        gps = infer_direction(gps)
        result = cross_check_heading(gps, empresaid=59)
        assert "heading_agrees" not in result.columns, (
            "cross_check_heading should NOT add heading_agrees for E59"
        )
        # Frame should be identical (same columns and rows).
        assert set(result.columns) == set(gps.columns)
