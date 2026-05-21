"""Tests for src/preprocessing/headways.py.

Covers:
  T1.8 — C.2 deterministic crossing: delta_t_min within 0.5s of expected.
  T1.8 sub-scenario — NULL emission: when bus_back has zero history, row is
      emitted with delta_t_min IS NULL (NOT dropped). pair_rank remains dense.
  Additional: pair count = N-1 for N buses per snapshot group.
  AC-C5..C9: lateral pair filter and R7 schema extension.
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
    lateral_m: float | None = None,
) -> dict:
    if day is None:
        day = t.date()
    row: dict = {
        "empresaid": empresaid,
        "unidadid": unidadid,
        "t": t,
        "s": s,
        "speed_kmh": speed_kmh,
        "direction": direction,
        "day": day,
    }
    if lateral_m is not None:
        row["lateral_m"] = lateral_m
    return row


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
    df = pl.DataFrame(rows).with_columns([
        pl.col("empresaid").cast(pl.Int64),
        pl.col("unidadid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("s").cast(pl.Float64),
        pl.col("speed_kmh").cast(pl.Float64),
        pl.col("direction").cast(pl.Int8),
    ])
    if "lateral_m" in df.columns:
        df = df.with_columns(pl.col("lateral_m").cast(pl.Float64))
    return df


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


class TestPairLateralFilter:
    """AC-C5, AC-C6, AC-C7: lateral pair filter in compute_pairs."""

    def test_lateral_filter_drops_cross_track(self):
        """AC-C5: cross-track pair is dropped when an explicit override activates the filter.

        Default is float('inf') (filter OFF). This test verifies the filter mechanism
        works when opted-in via EmpresaConfig.lateral_pair_threshold_m_override=50 m:
        |lateral_delta|=120 m > 50 m override → pair dropped.

        Failure mode: if compute_pairs does not respect the per-empresa override, the
        cross-street pair is emitted when it should be suppressed.
        """
        import src.preprocessing.config as config_module
        from src.preprocessing.config import EmpresaConfig
        original_config = config_module.EMPRESA_CONFIG
        try:
            config_module.EMPRESA_CONFIG = {
                2: EmpresaConfig(
                    empresaid=2,
                    has_heading=True,
                    lateral_pair_threshold_m_override=50.0,
                ),
                59: EmpresaConfig(empresaid=59, has_heading=False),
            }
            rows = [
                _make_snapshot_row(2, 201, T0, s=100.0, lateral_m=10.0),
                _make_snapshot_row(2, 202, T0, s=200.0, lateral_m=130.0),  # delta = 120 m
            ]
            snaps = _build_snapshots_df(rows)
            pairs = compute_pairs(snaps)
            assert len(pairs) == 0, (
                f"Expected 0 pairs (cross-street delta=120 m > override 50 m); got {len(pairs)}"
            )
        finally:
            config_module.EMPRESA_CONFIG = original_config

    def test_lateral_filter_keeps_same_track(self):
        """AC-C5 (retain case): |lateral_delta|=20 m with default float('inf') → pair retained (filter OFF)."""
        rows = [
            _make_snapshot_row(2, 201, T0, s=100.0, lateral_m=10.0),
            _make_snapshot_row(2, 202, T0, s=200.0, lateral_m=30.0),  # delta = 20 m
        ]
        snaps = _build_snapshots_df(rows)
        pairs = compute_pairs(snaps)
        assert len(pairs) == 1, (
            f"Expected 1 pair (same-track delta=20 m <= threshold 50 m); got {len(pairs)}"
        )

    def test_lateral_filter_uses_empresa_override(self):
        """AC-C4 integration + AC-C5: empresa override = 10 m drops 20 m delta pair.

        Uses monkeypatching on EMPRESA_CONFIG to set override for empresa 2.
        """
        import src.preprocessing.config as config_module
        from src.preprocessing.config import EmpresaConfig
        original_config = config_module.EMPRESA_CONFIG
        try:
            config_module.EMPRESA_CONFIG = {
                2: EmpresaConfig(
                    empresaid=2,
                    has_heading=True,
                    lateral_pair_threshold_m_override=10.0,
                ),
                59: EmpresaConfig(empresaid=59, has_heading=False),
            }
            rows = [
                _make_snapshot_row(2, 201, T0, s=100.0, lateral_m=10.0),
                _make_snapshot_row(2, 202, T0, s=200.0, lateral_m=30.0),  # delta = 20 m
            ]
            snaps = _build_snapshots_df(rows)
            pairs = compute_pairs(snaps)
            assert len(pairs) == 0, (
                f"Expected 0 pairs (delta=20 m > empresa override 10 m); got {len(pairs)}"
            )
        finally:
            config_module.EMPRESA_CONFIG = original_config

    def test_lateral_filter_boundary_retained(self):
        """AC-C7: |delta| == 50.0 with default float('inf') → pair retained (filter is no-op by default).

        Consistent with R-LB1 boundary semantics (>= for retain, > for drop).
        With default=inf, any finite delta is always retained.
        """
        rows = [
            _make_snapshot_row(2, 201, T0, s=100.0, lateral_m=0.0),
            _make_snapshot_row(2, 202, T0, s=200.0, lateral_m=50.0),  # delta = 50.0 exactly
        ]
        snaps = _build_snapshots_df(rows)
        pairs = compute_pairs(snaps)
        assert len(pairs) == 1, (
            f"Expected 1 pair (boundary |delta|=50.0 == threshold 50.0 → retained); got {len(pairs)}"
        )

    def test_lateral_filter_retains_null_lateral(self):
        """AC-C6: when lateral_m is null for a bus, the pair is retained conservatively."""
        rows = [
            {"empresaid": 2, "unidadid": 201, "t": T0, "s": 100.0,
             "speed_kmh": 20.0, "direction": 1, "day": T0.date(), "lateral_m": None},
            {"empresaid": 2, "unidadid": 202, "t": T0, "s": 200.0,
             "speed_kmh": 20.0, "direction": 1, "day": T0.date(), "lateral_m": 200.0},
        ]
        snaps = pl.DataFrame(rows).with_columns([
            pl.col("empresaid").cast(pl.Int64),
            pl.col("unidadid").cast(pl.Int64),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("s").cast(pl.Float64),
            pl.col("speed_kmh").cast(pl.Float64),
            pl.col("direction").cast(pl.Int8),
            pl.col("lateral_m").cast(pl.Float64),
        ])
        pairs = compute_pairs(snaps)
        assert len(pairs) == 1, (
            f"Expected 1 pair (null lateral_m → conservative retain); got {len(pairs)}"
        )


class TestR7Schema:
    """AC-C8, AC-C9: R7 schema extension with lateral_m_front, lateral_m_back."""

    def test_lateral_columns_emitted(self):
        """AC-C8: compute_pairs output must contain lateral_m_front and lateral_m_back (Float64)."""
        rows = [
            _make_snapshot_row(2, 201, T0, s=100.0, lateral_m=10.0),
            _make_snapshot_row(2, 202, T0, s=200.0, lateral_m=20.0),
        ]
        snaps = _build_snapshots_df(rows)
        pairs = compute_pairs(snaps)
        assert "lateral_m_front" in pairs.columns, (
            "lateral_m_front column missing from compute_pairs output (AC-C8)"
        )
        assert "lateral_m_back" in pairs.columns, (
            "lateral_m_back column missing from compute_pairs output (AC-C8)"
        )
        assert pairs.schema["lateral_m_front"] == pl.Float64, (
            f"lateral_m_front must be Float64; got {pairs.schema['lateral_m_front']}"
        )
        assert pairs.schema["lateral_m_back"] == pl.Float64, (
            f"lateral_m_back must be Float64; got {pairs.schema['lateral_m_back']}"
        )

    def test_n_minus_one_pairs_unchanged_when_all_same_track(self):
        """AC-C9 regression guard: 3 same-track buses → exactly 2 pairs (N-1).

        Verifies the lateral filter does not accidentally drop valid same-track pairs.
        All buses within 5 m lateral → well within 50 m threshold.
        """
        rows = [
            _make_snapshot_row(2, 201, T0, s=100.0, lateral_m=5.0),
            _make_snapshot_row(2, 202, T0, s=200.0, lateral_m=8.0),
            _make_snapshot_row(2, 203, T0, s=300.0, lateral_m=10.0),
        ]
        snaps = _build_snapshots_df(rows)
        pairs = compute_pairs(snaps)
        assert len(pairs) == 2, (
            f"Expected 2 pairs for 3 same-track buses; got {len(pairs)} "
            "(filter may be incorrectly dropping valid pairs)"
        )


    def test_lateral_columns_in_compute_headways_c2_output(self):
        """AC-S1/AC-S2: compute_headways_c2 must forward lateral_m_front and
        lateral_m_back as the last two columns of the returned DataFrame.

        Failure mode (CRITICAL #1 from verify): the final .select in
        compute_headways_c2 enumerates only 11 columns explicitly and drops the
        two lateral columns emitted by compute_pairs.

        Setup: two buses with distinct lateral_m values form a same-track pair.
        bus_back has GPS history that crosses s_front 3 minutes before T0, so
        delta_t_min is non-null (ensures the pair row is not trivially filtered).
        """
        T = T0
        t_cross = T0 - timedelta(minutes=3)

        # bus_back trajectory: crosses s_front at t_cross.
        gps_rows = _make_gps_pings(
            2, 202,
            [T0 - timedelta(minutes=6), t_cross, T0 - timedelta(minutes=1)],
            [80.0, 200.0, 150.0],  # crosses s=200 (s_front) at t_cross
            direction=1,
        )
        gps = _build_gps_df(gps_rows)

        snap_rows = [
            _make_snapshot_row(2, 201, T, s=200.0, lateral_m=5.0),   # bus_front
            _make_snapshot_row(2, 202, T, s=100.0, lateral_m=8.0),   # bus_back
        ]
        snaps = _build_snapshots_df(snap_rows)

        result = compute_headways_c2(snaps, gps, min_buses=2)
        assert len(result) == 1, f"Expected 1 pair row; got {len(result)}"

        # AC-S1: lateral_m_front must be present.
        assert "lateral_m_front" in result.columns, (
            "lateral_m_front missing from compute_headways_c2 output (AC-S1)"
        )
        # AC-S2: lateral_m_back must be present.
        assert "lateral_m_back" in result.columns, (
            "lateral_m_back missing from compute_headways_c2 output (AC-S2)"
        )
        # Dtype must be Float64.
        assert result.schema["lateral_m_front"] == pl.Float64, (
            f"lateral_m_front dtype must be Float64; got {result.schema['lateral_m_front']}"
        )
        assert result.schema["lateral_m_back"] == pl.Float64, (
            f"lateral_m_back dtype must be Float64; got {result.schema['lateral_m_back']}"
        )
        # Columns must be the last two.
        assert result.columns[-2] == "lateral_m_front", (
            f"lateral_m_front must be second-to-last column; got {result.columns}"
        )
        assert result.columns[-1] == "lateral_m_back", (
            f"lateral_m_back must be last column; got {result.columns}"
        )
        # Values must match the snapshot lateral_m values (front=5.0, back=8.0).
        row = result.row(0, named=True)
        assert row["lateral_m_front"] == pytest.approx(5.0), (
            f"lateral_m_front value must be 5.0; got {row['lateral_m_front']}"
        )
        assert row["lateral_m_back"] == pytest.approx(8.0), (
            f"lateral_m_back value must be 8.0; got {row['lateral_m_back']}"
        )


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
