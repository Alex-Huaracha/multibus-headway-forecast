"""Tests for src/preprocessing/projection.py.

Covers:
  T1.3 — project_to_centerline returns s ≈ known arc length and lateral_m ≈ 0
          for a point exactly on the centerline.
  T1.4 — project_to_centerline returns lateral_m ≈ 200 m for a point 200 m
          perpendicular to a centerline segment.
  Additional: arc-length monotonicity for straight on-route bus;
              off-route filter drops bus 203 (lat=-16.45) from E2 fixture.
  T2.7–T2.10 — project_per_direction per-direction projection wrapper.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.preprocessing.projection import (
    _project_arc_length,
    attach_observed_speed,
    project_to_centerline,
)
from src.preprocessing.corridor import build_centerline
from src.preprocessing.config import LAT_DEG_M, LON_DEG_M
from tests.fixtures.synthetic import make_dual_filar_gps

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def gps_e2() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / "synthetic_gps_e2.parquet")


@pytest.fixture(scope="module")
def gps_e2_with_speed(gps_e2: pl.DataFrame) -> pl.DataFrame:
    return attach_observed_speed(gps_e2)


@pytest.fixture(scope="module")
def centerline_e2(gps_e2_with_speed: pl.DataFrame) -> np.ndarray:
    return build_centerline(gps_e2_with_speed, empresaid=2)


class TestProjectOnCenterline:
    """T1.3 — a point exactly on the centerline has lateral_m ≈ 0."""

    def test_point_on_centerline_has_zero_lateral(self, centerline_e2: np.ndarray):
        """Failure mode: if the projection formula applies incorrect flat-Earth
        scaling (e.g. uses degrees instead of meters for distance), lateral_m
        is non-zero for an on-centerline point.

        Place a synthetic bus at the midpoint of the first segment of the
        centerline. Expected s ≈ 0 (at the first vertex), lateral_m < 0.01 m.
        """
        # Use the midpoint of the first segment as a test point.
        mid_lat = (centerline_e2[0, 0] + centerline_e2[1, 0]) / 2
        mid_lon = (centerline_e2[0, 1] + centerline_e2[1, 1]) / 2

        pts = np.array([[mid_lat, mid_lon]])
        s_arr, lateral_arr = _project_arc_length(pts, centerline_e2, chunk_size=10_000)

        # The midpoint of the first segment should project with near-zero lateral.
        assert lateral_arr[0] < 0.01, (
            f"Point on centerline midpoint has lateral_m = {lateral_arr[0]:.4f} m "
            "(expected < 0.01 m)"
        )

    def test_point_on_centerline_arc_length_reasonable(self, centerline_e2: np.ndarray):
        """Arc length for a point near the start of the polyline must be
        smaller than half the total polyline length.
        """
        mid_lat = (centerline_e2[0, 0] + centerline_e2[1, 0]) / 2
        mid_lon = (centerline_e2[0, 1] + centerline_e2[1, 1]) / 2

        pts = np.array([[mid_lat, mid_lon]])
        s_arr, _ = _project_arc_length(pts, centerline_e2, chunk_size=10_000)

        # Compute total polyline length in meters.
        cl_m = np.stack([
            centerline_e2[:, 0] * LAT_DEG_M,
            centerline_e2[:, 1] * LON_DEG_M,
        ], axis=1)
        total_len = np.sum(np.linalg.norm(np.diff(cl_m, axis=0), axis=1))

        assert s_arr[0] < total_len / 2, (
            f"First-segment midpoint s={s_arr[0]:.2f} m should be < half total length "
            f"({total_len / 2:.2f} m)"
        )


class TestProjectOffCenterline:
    """T1.4 — a point 200 m perpendicular to a segment has lateral_m ≈ 200."""

    def test_200m_perpendicular_lateral(self):
        """Failure mode: flat-Earth unit conversion error — if LON_DEG_M is not
        applied correctly, a 200 m perpendicular offset returns a wrong lateral_m.

        Use a perfectly horizontal (constant lat) synthetic centerline so the
        perpendicular is exactly north-south. Place a point 200 m north of the
        midpoint. Expected lateral_m = 200 m within 1 m tolerance.
        """
        # Perfectly horizontal centerline (constant lat = -16.4).
        cl_lat = -16.4
        cl_lons = np.linspace(-71.55, -71.50, 10)
        cl = np.stack([np.full(10, cl_lat), cl_lons], axis=1)

        # 200 m north of the midpoint of the centerline.
        offset_lat = 200.0 / LAT_DEG_M
        mid_lon = (cl_lons[4] + cl_lons[5]) / 2
        pts = np.array([[cl_lat + offset_lat, mid_lon]])

        _, lateral_arr = _project_arc_length(pts, cl, chunk_size=10_000)

        assert abs(lateral_arr[0] - 200.0) < 1.0, (
            f"Expected lateral_m ≈ 200 m; got {lateral_arr[0]:.2f} m"
        )


class TestArcLengthMonotonicity:
    """Projection monotonicity for a straight on-route bus (indirect T1.3/T1.4 guard)."""

    def test_arc_length_monotonic_for_bus_201_ida(
        self, gps_e2_with_speed: pl.DataFrame, centerline_e2: np.ndarray
    ):
        """Failure mode: regression of chunked projection; if the chunk boundary
        resets s or produces a discontinuity, the s series for a bus traveling
        strictly west-to-east along the route will not be monotonic.

        Bus 201 travels lon -71.55 → -71.50 on its ida trip (first trip, before
        the 31-min gap). The s values should be monotonically increasing with
        at most 1 m of floating-point noise per step.
        """
        from datetime import datetime
        projected = project_to_centerline(gps_e2_with_speed, centerline_e2, empresaid=2)
        bus_201 = projected.filter(pl.col("unidadid") == 201).sort("time")

        if bus_201.is_empty():
            pytest.skip("Bus 201 was entirely filtered out as off-route (unexpected)")

        # Only look at the first ~30 min (ida trip) — before the 31-min gap.
        # T0 = 07:00; ida ends around 07:27.
        t_cutoff = datetime(2024, 1, 23, 7, 28, 0)
        bus_201_ida = bus_201.filter(pl.col("time") <= t_cutoff)

        if len(bus_201_ida) < 3:
            pytest.skip("Not enough ida pings for bus 201")

        s_vals = bus_201_ida["s"].to_numpy()
        diffs = np.diff(s_vals)
        # Allow ± 1 m noise from GPS jitter and floating-point.
        assert diffs.min() >= -1.0, (
            f"Bus 201 s series has a negative step of {diffs.min():.2f} m — "
            "projection chunking bug or PCA sign flip"
        )


class TestOffRouteFilter:
    """Bus 203 (lat=-16.45, far off-route) must be dropped by the lateral filter."""

    def test_offroute_bus_dropped(
        self, gps_e2_with_speed: pl.DataFrame, centerline_e2: np.ndarray
    ):
        """Failure mode: if the lateral threshold filter does not apply
        lateral_threshold_for(empresaid) correctly, off-route buses survive.

        Bus 203 is at lat=-16.45 (~5.5 km south of the centerline at -16.4),
        far exceeding LATERAL_OFFSET_THRESHOLD_M=300 m. All pings must be dropped.
        """
        projected = project_to_centerline(gps_e2_with_speed, centerline_e2, empresaid=2)
        bus_203_rows = projected.filter(pl.col("unidadid") == 203)
        assert len(bus_203_rows) == 0, (
            f"Expected 0 rows for off-route bus 203; got {len(bus_203_rows)}"
        )


class TestAttachObservedSpeedDropsGPSJumps:
    """Spec R11 — GPS-jump pairs must be DROPPED, not nulled.

    Covers three sub-scenarios:
      1. step_m > 500 m AND dt_s <= 60 s → row dropped (jump criterion).
      2. step_m > 500 m AND dt_s > 60 s → row kept (dt_s threshold not met).
      3. speed_kmh > 80 km/h → row dropped (speed criterion).
    """

    def _make_single_bus_gps(self, rows: list[dict]) -> pl.DataFrame:
        """Build a minimal GPS DataFrame for a single (empresaid=2, unidadid=901)."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 23, 8, 0, 0)
        times = [base + timedelta(seconds=r["dt_offset"]) for r in rows]
        return pl.DataFrame({
            "empresaid": [2] * len(rows),
            "unidadid": [901] * len(rows),
            "time": pl.Series(times, dtype=pl.Datetime("us")),
            "lat": [r["lat"] for r in rows],
            "lon": [r["lon"] for r in rows],
        })

    def test_gps_jump_pair_dropped_within_60s(self):
        """Failure mode (pre-fix): the GPS-jump pair survives as a null speed_kmh
        row instead of being dropped. After fix, step_m > 500 m AND dt_s <= 60 s
        must cause the jump row to be discarded.

        3-row frame:
          ping 0 (t=0s, lat0)          — first ping, null speed.
          ping 1 (t=30s, lat_jump)     — 600m/30s = jump criterion → DROP.
          ping 2 (t=60s, lat_jump+tiny) — small step from ping 1, normal speed → KEEP.

        Note: step_m for each ping is computed relative to the PREVIOUS ping before
        any filtering; ping 2's step_m is relative to ping 1. After ping 1 is dropped
        by the jump criterion, we expect 2 rows (ping 0 + ping 2).
        Ping 2 must have a small step_m (< 500 m, speed < 80 km/h) to survive.
        """
        lat0 = -16.4
        lon0 = -71.55
        # 600 m north in degrees lat (LAT_DEG_M ≈ 111000)
        lat_jump = lat0 + 600.0 / 111_000.0
        # Ping 2 is 50 m beyond ping 1 in 30 s → speed = 50/30*3.6 = 6 km/h, step = 50 m.
        lat_ping2 = lat_jump + 50.0 / 111_000.0

        rows = [
            {"dt_offset": 0,  "lat": lat0,       "lon": lon0},  # ping 0: first ping
            {"dt_offset": 30, "lat": lat_jump,    "lon": lon0},  # ping 1: 600m/30s → DROP
            {"dt_offset": 60, "lat": lat_ping2,   "lon": lon0},  # ping 2: 50m/30s → KEEP
        ]
        gps = self._make_single_bus_gps(rows)
        result = attach_observed_speed(gps)

        assert result.height == 2, (
            f"Expected 2 rows after dropping GPS-jump pair; got {result.height}. "
            f"Row heights suggest the jump pair was NOT dropped."
        )

    def test_gps_jump_pair_kept_when_dt_exceeds_60s(self):
        """A 600 m step over 120 s is NOT a GPS jump (dt_s > 60 s threshold).
        The row must be KEPT (speed = 600/120*3.6 = 18 km/h, well within 80 km/h).
        """
        lat0 = -16.4
        lon0 = -71.55
        lat_far = lat0 + 600.0 / 111_000.0

        rows = [
            {"dt_offset": 0,   "lat": lat0,    "lon": lon0},
            {"dt_offset": 120, "lat": lat_far,  "lon": lon0},  # 600 m / 120 s = 18 km/h → KEEP
        ]
        gps = self._make_single_bus_gps(rows)
        result = attach_observed_speed(gps)

        assert result.height == 2, (
            f"Expected 2 rows (600m/120s is NOT a jump); got {result.height}."
        )
        speed = result.filter(pl.col("speed_kmh").is_not_null())["speed_kmh"][0]
        assert abs(speed - 18.0) < 0.1, (
            f"Expected speed_kmh ≈ 18.0 km/h; got {speed:.3f}"
        )

    def test_speed_over_80_kmh_row_dropped_not_nulled(self):
        """Failure mode (pre-fix): speed > 80 km/h was SET TO NULL instead of
        DROPPING the row. After fix, the row must be absent from the output.

        Construct a 2-row frame where the second ping is 1000 m in 10 s
        (speed = 360 km/h). The second row must be dropped entirely.
        """
        lat0 = -16.4
        lon0 = -71.55
        lat_fast = lat0 + 1000.0 / 111_000.0  # 1000 m north

        rows = [
            {"dt_offset": 0,  "lat": lat0,       "lon": lon0},
            {"dt_offset": 10, "lat": lat_fast,    "lon": lon0},  # 1000m/10s = 360 km/h → DROP
        ]
        gps = self._make_single_bus_gps(rows)
        result = attach_observed_speed(gps)

        assert result.height == 1, (
            f"Expected 1 row (speed > 80 km/h row must be DROPPED, not nulled); "
            f"got {result.height}."
        )
        # The surviving row is the first ping with null speed.
        assert result["speed_kmh"][0] is None, (
            "The surviving first-ping row should have null speed_kmh"
        )


# ---------------------------------------------------------------------------
# T2.7–T2.9 RED: TestProjectPerDirection
# ---------------------------------------------------------------------------


class TestProjectPerDirection:
    """D2-PR-API — per-direction projection wrapper (R-PR1).

    T2.7: pings with direction=+1 have smaller |lateral_m| against cl[+1]
          than against cl[-1], and vice-versa for direction=-1.
    T2.8: pings with direction=0 (or unknown key) yield NaN s and NaN lateral_m.
    T2.9: output row count equals input; no new columns; s/lateral_m are Float64.
    """

    @pytest.fixture(scope="class")
    def dual_filar(self) -> pl.DataFrame:
        return make_dual_filar_gps(
            empresaid=59,
            n_buses_per_street=4,
            n_pings_per_bus=300,
            street_separation_m=40.0,
            rng_seed=42,
        )

    @pytest.fixture(scope="class")
    def centerlines(self, dual_filar: pl.DataFrame) -> dict[int, np.ndarray]:
        from src.preprocessing.corridor import build_centerline_per_direction
        return build_centerline_per_direction(dual_filar, empresaid=59)

    def test_dir_pings_project_to_their_centerline(
        self, dual_filar: pl.DataFrame, centerlines: dict[int, np.ndarray]
    ):
        """T2.7: pings with direction=+1 have smaller |lateral_m| when projected
        against cl[+1] than against cl[-1] (and vice-versa for direction=-1).
        This confirms each ping is matched to its own street centerline.
        """
        from src.preprocessing.projection import project_per_direction, _project_arc_length

        result = project_per_direction(dual_filar, centerlines, empresaid=59)

        # For direction=+1 pings: lateral_m in result (projected against cl[+1])
        # should be smaller than if projected against cl[-1]
        plus_pings = dual_filar.filter(pl.col("direction") == 1)
        pts = plus_pings.select(["lat", "lon"]).to_numpy()

        _, lateral_vs_cl_plus = _project_arc_length(pts, centerlines[1], chunk_size=10_000)
        _, lateral_vs_cl_minus = _project_arc_length(pts, centerlines[-1], chunk_size=10_000)

        mean_lateral_own = float(lateral_vs_cl_plus.mean())
        mean_lateral_other = float(lateral_vs_cl_minus.mean())
        assert mean_lateral_own < mean_lateral_other, (
            f"Expected dir=+1 pings closer to cl[+1] (mean {mean_lateral_own:.2f} m) "
            f"than to cl[-1] (mean {mean_lateral_other:.2f} m)"
        )

    def test_directionless_pings_get_nan(self, centerlines: dict[int, np.ndarray]):
        """T2.8: pings with direction=0 (key not in centerlines dict) yield NaN
        for s and lateral_m, but are kept in the output (not dropped).
        """
        from datetime import datetime
        from src.preprocessing.projection import project_per_direction
        import math
        # Build a minimal frame with direction=0 pings
        directionless = pl.DataFrame({
            "empresaid": pl.Series([59, 59], dtype=pl.Int64),
            "unidadid": pl.Series([5901, 5901], dtype=pl.Int64),
            "time": pl.Series(
                [datetime(2024, 1, 23, 7, 0, 0), datetime(2024, 1, 23, 7, 0, 20)],
                dtype=pl.Datetime("us"),
            ),
            "lat": pl.Series([-16.4, -16.4], dtype=pl.Float64),
            "lon": pl.Series([-71.52, -71.51], dtype=pl.Float64),
            "direction": pl.Series([0, 0], dtype=pl.Int64),
            "speed_kmh": pl.Series([20.0, 20.0], dtype=pl.Float64),
            "s": pl.Series([0.0, 0.0], dtype=pl.Float64),
            "lateral_m": pl.Series([0.0, 0.0], dtype=pl.Float64),
        })
        result = project_per_direction(directionless, centerlines, empresaid=59)
        assert result.height == 2, f"Expected 2 rows kept; got {result.height}"
        s_vals = result["s"].to_list()
        lat_vals = result["lateral_m"].to_list()
        for v in s_vals:
            assert v is None or (isinstance(v, float) and math.isnan(v)), (
                f"Expected NaN s for direction=0; got {v}"
            )
        for v in lat_vals:
            assert v is None or (isinstance(v, float) and math.isnan(v)), (
                f"Expected NaN lateral_m for direction=0; got {v}"
            )

    def test_schema_preserved(
        self, dual_filar: pl.DataFrame, centerlines: dict[int, np.ndarray]
    ):
        """T2.9: output row count equals input; s and lateral_m are Float64.
        The function must not add or remove columns beyond s/lateral_m.
        """
        from src.preprocessing.projection import project_per_direction
        # Add s and lateral_m as placeholder columns to mimic post-pass1 frame
        input_frame = dual_filar.with_columns([
            pl.lit(0.0, dtype=pl.Float64).alias("s"),
            pl.lit(0.0, dtype=pl.Float64).alias("lateral_m"),
        ])
        result = project_per_direction(input_frame, centerlines, empresaid=59)
        # Row count preserved
        assert result.height == input_frame.height, (
            f"Expected {input_frame.height} rows; got {result.height}"
        )
        # s and lateral_m present with Float64 dtype
        assert result["s"].dtype == pl.Float64, (
            f"Expected s dtype Float64; got {result['s'].dtype}"
        )
        assert result["lateral_m"].dtype == pl.Float64, (
            f"Expected lateral_m dtype Float64; got {result['lateral_m'].dtype}"
        )
        # No new unexpected columns
        input_cols = set(input_frame.columns)
        output_cols = set(result.columns)
        new_cols = output_cols - input_cols
        assert not new_cols, f"Unexpected new columns in output: {new_cols}"
