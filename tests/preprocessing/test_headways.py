"""Tests for src/preprocessing/headways.py.

Covers:
  T1.8 — C.2 deterministic crossing: delta_t_min within 0.5s of expected.
  T1.8 sub-scenario — NULL emission: when bus_back has zero history, row is
      emitted with delta_t_min IS NULL (NOT dropped). pair_rank remains dense.
  Additional: pair count = N-1 for N buses per snapshot group.
"""
from __future__ import annotations
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.preprocessing.headways import compute_headways_c2, compute_pairs


T0 = datetime(2024, 1, 23, 8, 0, 0)


def _make_snapshot_row(
    empresaid: int,
    unidadid: int,
    t: datetime,
    s: float,
    speed_kmh: float = 20.0,
    direction: int = 1,
    day=None,
) -> dict:
    if day is None:
        day = t.date()
    return {
        "empresaid": empresaid,
        "unidadid": unidadid,
        "t": t,
        "s": s,
        "speed_kmh": speed_kmh,
        "direction": direction,
        "day": day,
    }


def _make_gps_pings(
    empresaid: int,
    unidadid: int,
    times: list[datetime],
    s_values: list[float],
    direction: int = 1,
) -> list[dict]:
    """Build GPS ping rows for a bus's historical trajectory."""
    rows = []
    for t, s in zip(times, s_values):
        rows.append({
            "empresaid": empresaid,
            "unidadid": unidadid,
            "time": t,
            "s": float(s),
            "speed_kmh": 20.0,
            "direction": direction,
            "lat": -16.4,
            "lon": -71.52,
        })
    return rows


def _build_snapshots_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns([
        pl.col("empresaid").cast(pl.Int64),
        pl.col("unidadid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("s").cast(pl.Float64),
        pl.col("speed_kmh").cast(pl.Float64),
        pl.col("direction").cast(pl.Int8),
    ])


def _build_gps_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns([
        pl.col("empresaid").cast(pl.Int64),
        pl.col("unidadid").cast(pl.Int64),
        pl.col("time").cast(pl.Datetime("us")),
        pl.col("s").cast(pl.Float64),
        pl.col("speed_kmh").cast(pl.Float64),
        pl.col("direction").cast(pl.Int8),
    ])


class TestPairStructure:
    """N buses per snapshot → exactly N-1 pair rows (pair count test)."""

    def test_n_minus_one_pairs(self):
        """Failure mode: shift is applied on wrong axis or sort order is wrong,
        producing too few or too many pairs.

        3 buses at T0: pair count must be exactly 2 (pairs (front=2, back=1)
        and (front=3, back=2) when sorted by s ascending).
        """
        rows = [
            _make_snapshot_row(2, 201, T0, s=100.0),
            _make_snapshot_row(2, 202, T0, s=200.0),
            _make_snapshot_row(2, 203, T0, s=300.0),
        ]
        snaps = _build_snapshots_df(rows)
        pairs = compute_pairs(snaps)
        assert len(pairs) == 2, f"Expected 2 pairs for 3 buses; got {len(pairs)}"

    def test_pair_rank_is_dense(self):
        """pair_rank must be 1, 2 for 3 buses (N-1 = 2 pairs, ranks 1 and 2)."""
        rows = [
            _make_snapshot_row(2, 201, T0, s=100.0),
            _make_snapshot_row(2, 202, T0, s=200.0),
            _make_snapshot_row(2, 203, T0, s=300.0),
        ]
        snaps = _build_snapshots_df(rows)
        pairs = compute_pairs(snaps)
        ranks = sorted(pairs["pair_rank"].to_list())
        assert ranks == [1, 2], f"Expected ranks [1, 2]; got {ranks}"

    def test_bus_front_greater_s_than_bus_back(self):
        """After sorting by s ascending, bus_front must have s > bus_back."""
        rows = [
            _make_snapshot_row(2, 201, T0, s=100.0),
            _make_snapshot_row(2, 202, T0, s=200.0),
        ]
        snaps = _build_snapshots_df(rows)
        pairs = compute_pairs(snaps)
        assert len(pairs) == 1
        row = pairs.row(0, named=True)
        assert row["s_front"] > row["s_back"], (
            f"s_front ({row['s_front']}) must > s_back ({row['s_back']})"
        )


class TestC2KnownCrossing:
    """T1.8 — deterministic crossing: delta_t_min within 0.5s of expected."""

    def test_c2_known_crossing_delta_t(self):
        """Failure mode: regression of the trailing-crossing algorithm — if the
        pandas-converted loop is accidentally used or the sign-change scan is
        wrong, delta_t_min deviates from the analytically known value.

        Setup: bus_front is at s=500 at T=T0. bus_back crossed s=500 exactly
        5 minutes before T0 (i.e. t_cross = T0 - 5min). Expected delta_t_min = 5.0.

        bus_back trajectory: 10 pings from s=0 to s=1000 over 10 minutes,
        crossing s=500 at t_cross = T0 - 5min exactly.
        """
        T = T0
        t_cross_expected = T0 - timedelta(minutes=5)

        # bus_back trajectory: s linearly from 0 to 1000 over 10 min (20s pings).
        # At t_cross = T0-5min, bus_back is at s=500.
        n_back = 30
        dt_back = timedelta(seconds=20)
        t_start_back = T0 - timedelta(minutes=10)
        times_back = [t_start_back + i * dt_back for i in range(n_back)]
        # s goes from 0 at T0-10min to 1000 at T0 (linear)
        s_back_arr = np.linspace(0, 1000, n_back)
        # Ensure s=500 is at exactly t_cross = T0-5min.
        # times_back: index at T0-5min = 5min / 20s = 15 pings → times_back[15] = T0-5min
        # s_back_arr[15] should be 500.
        # With linspace(0,1000,30): s_back_arr[15] = 15/29 * 1000 ≈ 517.2 (not exactly 500).
        # Manually fix the trajectory: place a point exactly at (T0-5min, 500).
        times_back = [t_start_back + i * dt_back for i in range(n_back)]
        # Insert exact crossing point at index 15.
        times_back[15] = t_cross_expected
        s_back_arr[15] = 500.0
        # Fix adjacent points to ensure monotonicity.
        s_back_arr[14] = 480.0
        s_back_arr[16] = 520.0

        gps_rows = _make_gps_pings(2, 202, times_back, s_back_arr.tolist(), direction=1)

        # Snapshots: bus_front at s=500, bus_back at s=400 (behind front) at T.
        snap_rows = [
            _make_snapshot_row(2, 201, T, s=500.0),   # bus_front
            _make_snapshot_row(2, 202, T, s=400.0),   # bus_back
        ]

        snaps = _build_snapshots_df(snap_rows)
        gps = _build_gps_df(gps_rows)

        result = compute_headways_c2(snaps, gps, min_buses=2)
        assert len(result) == 1, f"Expected 1 pair row; got {len(result)}"

        row = result.row(0, named=True)
        assert row["delta_t_min"] is not None, "delta_t_min must not be null for known crossing"
        assert abs(row["delta_t_min"] - 5.0) < 0.5 / 60.0, (
            f"Expected delta_t_min ≈ 5.0 min; got {row['delta_t_min']:.6f} min "
            f"(error = {abs(row['delta_t_min'] - 5.0) * 60:.3f} s)"
        )


class TestNullEmission:
    """T1.8 sub-scenario (clarification #17 rule 2): NULL emission for no-history pairs."""

    def test_null_emitted_not_dropped_when_no_crossing(self):
        """Failure mode: C.2 drops the row instead of emitting null, breaking
        INV-3 (pair_rank density) and INV-4 (n_buses consistency).

        Scenario: bus_back has NO historical pings before T. The pair row must
        be emitted with delta_t_min IS NULL; bus_front != bus_back (INV-7);
        pair_rank remains dense (1 for the only pair).
        """
        T = T0
        # Only one snapshot pair: front at s=500, back at s=400.
        snap_rows = [
            _make_snapshot_row(2, 201, T, s=500.0),   # bus_front
            _make_snapshot_row(2, 202, T, s=400.0),   # bus_back
        ]
        snaps = _build_snapshots_df(snap_rows)

        # GPS: only bus_front has history; bus_back has NO pings at all.
        gps_rows = _make_gps_pings(
            2, 201,
            [T0 - timedelta(minutes=5), T0 - timedelta(minutes=3), T0],
            [200.0, 350.0, 500.0],
            direction=1,
        )
        gps = _build_gps_df(gps_rows)

        result = compute_headways_c2(snaps, gps, min_buses=2)

        # Row must be present (not dropped).
        assert len(result) == 1, (
            f"Row must be EMITTED with null delta_t_min (not dropped); got {len(result)} rows"
        )

        row = result.row(0, named=True)
        # delta_t_min must be null.
        assert row["delta_t_min"] is None, (
            f"Expected null delta_t_min for no-history back-bus; got {row['delta_t_min']}"
        )
        # INV-7: bus_front != bus_back.
        assert row["bus_front"] != row["bus_back"], "bus_front must != bus_back (INV-7)"
        # INV-3: pair_rank = 1 (dense, even with null delta_t_min).
        assert row["pair_rank"] == 1, f"pair_rank must be 1; got {row['pair_rank']}"
        # INV-4: n_buses >= 2.
        assert row["n_buses"] >= 2, f"n_buses must be >= 2; got {row['n_buses']}"

    def test_pair_rank_dense_with_mixed_null_and_nonnull(self):
        """INV-3: pair_rank must be dense 1..N-1 even when some rows have null delta_t_min.

        3 buses: pair (2,1) has crossing history → non-null; pair (3,2) has no
        history → null. Both must be emitted; pair_rank must be 1 and 2.
        """
        T = T0
        snap_rows = [
            _make_snapshot_row(2, 201, T, s=100.0),
            _make_snapshot_row(2, 202, T, s=200.0),
            _make_snapshot_row(2, 203, T, s=300.0),
        ]
        snaps = _build_snapshots_df(snap_rows)

        # bus 201 history: crossed s=100 5 min ago.
        # bus 202 history: does NOT cross s=200 (never reaches above s=200).
        # bus 203: has no history at all.
        gps_rows = (
            _make_gps_pings(2, 201,
                [T0 - timedelta(minutes=8), T0 - timedelta(minutes=5), T0 - timedelta(minutes=2)],
                [80.0, 100.0, 120.0],
                direction=1)
            + _make_gps_pings(2, 202,
                [T0 - timedelta(minutes=8), T0 - timedelta(minutes=5)],
                [50.0, 150.0],   # never crosses 200
                direction=1)
        )
        gps = _build_gps_df(gps_rows)

        result = compute_headways_c2(snaps, gps, min_buses=2)

        # Must have 2 rows (N-1 = 2 pairs for 3 buses).
        assert len(result) == 2, f"Expected 2 pair rows; got {len(result)}"

        ranks = sorted(result["pair_rank"].to_list())
        assert ranks == [1, 2], f"pair_rank must be dense [1, 2]; got {ranks}"


class TestLookbackBound:
    """Bound: stale historical crossings → NULL emission (proposal c2-lookback-fix)."""

    def test_headways_lookback_bound(self):
        """Failure mode: a bus_back whose only historical crossing of s_front is
        older than max_interpolation_lookback_minutes (default 30 min) is emitted
        with delta_t_min IS NULL. Without the bound, the kernel would emit
        delta_t_min ≈ 45 min (absurd for an urban headway).

        Scenario: at T=T0, bus_front is at s=500, bus_back is at s=400. bus_back
        crossed s=500 at T0 - 45 min (older than the 30-min default lookback) and
        has NOT crossed s=500 since. Expected: pair emitted with delta_t_min IS NULL
        (not 45.0).
        """
        T = T0
        t_cross_stale = T0 - timedelta(minutes=45)

        # bus_back trajectory: crosses s=500 at T-45min, then drifts backwards
        # (does not approach s_front again). Simulates a parallel-route bus that
        # was last near s_front 45 min ago.
        # Pings: 6 points spanning T-50min .. T-1min, s monotonically decreasing
        # after the early crossing.
        times_back = [
            T0 - timedelta(minutes=50),  # s = 600
            T0 - timedelta(minutes=46),  # s = 520
            t_cross_stale,               # s = 500 (the crossing, 45 min before T)
            T0 - timedelta(minutes=44),  # s = 480
            T0 - timedelta(minutes=30),  # s = 450
            T0 - timedelta(minutes=1),   # s = 400 (where it ends up at T)
        ]
        s_back_arr = [600.0, 520.0, 500.0, 480.0, 450.0, 400.0]

        gps_rows = _make_gps_pings(2, 202, times_back, s_back_arr, direction=1)

        snap_rows = [
            _make_snapshot_row(2, 201, T, s=500.0),   # bus_front
            _make_snapshot_row(2, 202, T, s=400.0),   # bus_back
        ]
        snaps = _build_snapshots_df(snap_rows)
        gps = _build_gps_df(gps_rows)

        # Default params (max_interpolation_lookback_minutes = 30.0).
        result = compute_headways_c2(snaps, gps, min_buses=2)

        assert len(result) == 1, (
            f"Pair must be EMITTED (not dropped); got {len(result)} rows"
        )
        row = result.row(0, named=True)
        # Critical assertion: stale crossing → NULL, not ~45 min.
        assert row["delta_t_min"] is None, (
            f"Expected NULL for stale historical crossing (>30 min old); "
            f"got {row['delta_t_min']} min — bound is not enforced"
        )
        # INV-3: pair_rank still dense (1 for the only pair).
        assert row["pair_rank"] == 1, f"pair_rank must be 1; got {row['pair_rank']}"
