"""Tests for src/preprocessing/trips.py.

Covers:
  T1.6 — trip segmentation by gap: 31-min gap creates exactly 2 trip segments.
  T1.7 — trip segmentation by reversal: mid-route reversal creates 2 segments.
  Additional: terminal cut on E59 bus 502 (dwells 6 min near s_max);
              build_snapshots grid timestamps satisfy t.dt.second() == 0 (INV-6).
"""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.preprocessing.trips import assign_trip_ids, build_snapshots
from src.preprocessing.config import PRODUCTIVE_PARAMS

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _make_bus_gps(
    empresaid: int,
    unidadid: int,
    times: list[datetime],
    s_values: list[float],
    speed_kmh: list[float] | None = None,
    direction: list[int] | None = None,
) -> pl.DataFrame:
    """Build a minimal GPS frame suitable for assign_trip_ids."""
    n = len(times)
    if speed_kmh is None:
        speed_kmh = [30.0] * n
    if direction is None:
        direction = [1] * n

    df = pl.DataFrame({
        "empresaid": [empresaid] * n,
        "unidadid": [unidadid] * n,
        "time": times,
        "s": s_values,
        "speed_kmh": speed_kmh,
        "direction": direction,
        "lat": [-16.4] * n,
        "lon": [-71.52] * n,
    })
    # Compute dt_s (required by assign_trip_ids).
    df = df.with_columns([
        pl.col("time").shift(1).over(["empresaid", "unidadid"]).alias("time_prev"),
    ]).with_columns([
        (pl.col("time") - pl.col("time_prev")).dt.total_seconds().alias("dt_s"),
    ]).drop("time_prev")
    return df


class TestGapCut:
    """T1.6 — a 31-min gap produces exactly 2 trip segments."""

    def test_31min_gap_creates_two_trips(self):
        """Failure mode: gap-cut threshold wrong (e.g. uses > instead of >=),
        or dt_s computation is off-by-one at the gap boundary.

        40 pings at 20s interval, then a 31-min gap, then 20 more pings.
        Expected: exactly 2 distinct trip_id values (0 and 1).
        """
        t0 = datetime(2024, 1, 23, 7, 0, 0)
        ping_interval = timedelta(seconds=20)
        gap = timedelta(minutes=31)

        # First segment: 40 pings
        times_a = [t0 + i * ping_interval for i in range(40)]
        s_a = [float(i * 100) for i in range(40)]

        # After 31-min gap
        t_after_gap = times_a[-1] + gap + ping_interval
        times_b = [t_after_gap + i * ping_interval for i in range(20)]
        s_b = [float((40 + i) * 100) for i in range(20)]

        all_times = times_a + times_b
        all_s = s_a + s_b
        gps = _make_bus_gps(2, 201, all_times, all_s)

        result = assign_trip_ids(gps)
        trip_ids = result.sort("time")["trip_id"].to_list()

        unique_trips = set(trip_ids)
        assert len(unique_trips) == 2, (
            f"Expected 2 unique trip_ids (one for each side of the gap); got {unique_trips}"
        )

        # The gap must coincide with the trip_id increment boundary.
        # Pings before the gap must all have the same trip_id.
        trips_before = result.filter(pl.col("time") < t_after_gap)["trip_id"].unique().to_list()
        trips_after = result.filter(pl.col("time") >= t_after_gap)["trip_id"].unique().to_list()
        assert len(trips_before) == 1, f"Before-gap pings span {len(trips_before)} trip_ids"
        assert len(trips_after) == 1, f"After-gap pings span {len(trips_after)} trip_ids"
        assert trips_before[0] < trips_after[0], "trip_id must be monotonically increasing"


class TestReversalCut:
    """T1.7 — mid-route reversal produces exactly 2 trip segments."""

    def test_reversal_creates_two_trips(self):
        """Failure mode: direction==0 transient pings (at the smoothing window
        boundary during the reversal) are incorrectly treated as cut events,
        producing 3+ trips instead of 2.

        20 pings ascending s (+1 direction) then 20 pings descending (-1 direction).
        Expected: exactly 2 distinct trip_ids.
        """
        t0 = datetime(2024, 1, 23, 7, 0, 0)
        ping_interval = timedelta(seconds=20)

        n = 20
        times_ida = [t0 + i * ping_interval for i in range(n)]
        s_ida = [float(i * 100) for i in range(n)]
        dir_ida = [1] * n

        t_vuelta_start = times_ida[-1] + ping_interval
        times_vuelta = [t_vuelta_start + i * ping_interval for i in range(n)]
        s_vuelta = [float((n - 1 - i) * 100) for i in range(n)]
        dir_vuelta = [-1] * n

        all_times = times_ida + times_vuelta
        all_s = s_ida + s_vuelta
        all_dir = dir_ida + dir_vuelta

        gps = _make_bus_gps(2, 201, all_times, all_s, direction=all_dir)
        result = assign_trip_ids(gps)

        unique_trips = set(result["trip_id"].to_list())
        assert len(unique_trips) == 2, (
            f"Expected 2 unique trip_ids for ida+vuelta; got {unique_trips} "
            "(possible transient direction==0 cut or reversal cut on 0-direction pings)"
        )

        # Validate that ida pings are in one trip and vuelta in another.
        trips_ida = result.filter(pl.col("time") <= times_ida[-1])["trip_id"].unique().to_list()
        trips_vuelta = result.filter(pl.col("time") >= t_vuelta_start)["trip_id"].unique().to_list()
        assert len(trips_ida) == 1, f"Ida pings have {len(trips_ida)} trip_ids"
        assert len(trips_vuelta) == 1, f"Vuelta pings have {len(trips_vuelta)} trip_ids"
        assert trips_ida[0] < trips_vuelta[0], "trip_id must increment at reversal"


class TestTerminalCut:
    """Terminal cut: bus stops near s_max for 6 min → new trip on EXIT."""

    def test_terminal_cut_creates_new_trip(self):
        """Failure mode: terminal-cut semantics flipped (cut on ENTRY instead of EXIT),
        causing the dwell pings to appear in trip_id N+1 instead of trip_id N.
        Or: dwell duration not computed correctly → no cut at all.

        Scenario: bus travels from s=0 to s=5000 (well past TERMINAL_BAND_M=200 m),
        then stops (speed=0) near s_max=5000 for 6 min (18 pings @ 20s), then exits.
        Expected: at least 2 distinct trip_ids (new trip starts on the exit ping).
        """
        from src.preprocessing.config import TERMINAL_DWELL_SECONDS, TERMINAL_BAND_M

        t0 = datetime(2024, 1, 23, 7, 0, 0)
        ping_interval = timedelta(seconds=20)

        # ida: 80 pings, s goes from 0 to 5000 (well > 2 * TERMINAL_BAND_M)
        n_ida = 80
        times_ida = [t0 + i * ping_interval for i in range(n_ida)]
        s_ida = [float(i * 5000 / (n_ida - 1)) for i in range(n_ida)]
        speed_ida = [30.0] * n_ida
        dir_ida = [1] * n_ida

        # Dwell: 18 pings at s≈5000, speed=0 → total 360 s = 6 min
        n_dwell = 18
        s_max = 5000.0
        dwell_s = s_max - 50.0   # within TERMINAL_BAND_M (200 m) of s_max
        dwell_start = times_ida[-1] + ping_interval
        times_dwell = [dwell_start + i * ping_interval for i in range(n_dwell)]
        s_dwell = [dwell_s] * n_dwell
        speed_dwell = [0.0] * n_dwell
        dir_dwell = [1] * n_dwell

        # Exit: 10 more pings, speed > threshold (resuming movement)
        exit_start = times_dwell[-1] + ping_interval
        times_exit = [exit_start + i * ping_interval for i in range(10)]
        s_exit = [dwell_s + float(i * 100) for i in range(10)]
        speed_exit = [10.0] * 10
        dir_exit = [1] * 10

        all_times = times_ida + times_dwell + times_exit
        all_s = s_ida + s_dwell + s_exit
        all_speed = speed_ida + speed_dwell + speed_exit
        all_dir = dir_ida + dir_dwell + dir_exit

        gps = _make_bus_gps(
            empresaid=59,
            unidadid=502,
            times=all_times,
            s_values=all_s,
            speed_kmh=all_speed,
            direction=all_dir,
        )

        result = assign_trip_ids(gps, s_min=0.0, s_max=s_max)
        unique_trips = set(result["trip_id"].to_list())
        assert len(unique_trips) >= 2, (
            f"Expected >= 2 trip_ids (terminal cut after 6 min dwell); got {unique_trips}"
        )

        # The exit pings must be in a different trip than the dwell pings.
        trips_before_exit = result.filter(
            pl.col("time") <= times_dwell[-1]
        )["trip_id"].unique().to_list()
        trips_exit = result.filter(
            pl.col("time") > times_dwell[-1]
        )["trip_id"].unique().to_list()

        # Exit pings should be in a LATER trip than dwell pings.
        assert max(trips_before_exit) < min(trips_exit), (
            f"Cut should be on EXIT ping; dwell trips={trips_before_exit}, "
            f"exit trips={trips_exit}"
        )


class TestSnapshotGridAlignment:
    """build_snapshots must produce minute-aligned timestamps (INV-6, clarification #17 rule 1)."""

    def test_snapshot_timestamps_are_minute_aligned(self):
        """Failure mode: if build_snapshots does not apply the epoch-floor pattern,
        grid timestamps can be at arbitrary seconds within a minute.

        INV-6: all t values must satisfy t.dt.second() == 0 when grid_seconds=60.
        """
        t0 = datetime(2024, 1, 23, 7, 3, 17)   # deliberately not on a minute boundary
        ping_interval = timedelta(seconds=20)
        n = 60
        times = [t0 + i * ping_interval for i in range(n)]
        s = [float(i * 50) for i in range(n)]
        direction = [1] * n
        speed = [30.0] * n

        gps = pl.DataFrame({
            "empresaid": [2] * n,
            "unidadid": [201] * n,
            "time": times,
            "s": s,
            "speed_kmh": speed,
            "direction": direction,
            "lat": [-16.4] * n,
            "lon": [-71.52] * n,
        }).with_columns([
            pl.col("time").dt.date().alias("day"),
            pl.col("time").shift(1).over(["empresaid", "unidadid"]).alias("time_prev"),
        ]).with_columns([
            (pl.col("time") - pl.col("time_prev")).dt.total_seconds().alias("dt_s"),
        ]).drop("time_prev")

        snaps = build_snapshots(gps, grid_seconds=60)
        assert not snaps.is_empty(), "build_snapshots returned empty frame"

        # All t values must have second == 0.
        seconds_col = snaps["t"].dt.second()
        non_zero_seconds = seconds_col.filter(seconds_col != 0)
        assert len(non_zero_seconds) == 0, (
            f"Found {len(non_zero_seconds)} snapshot timestamps with non-zero seconds; "
            "grid is not minute-aligned (INV-6 violation)"
        )
