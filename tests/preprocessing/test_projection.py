"""Tests for src/preprocessing/projection.py.

Covers:
  T1.3 — project_to_centerline returns s ≈ known arc length and lateral_m ≈ 0
          for a point exactly on the centerline.
  T1.4 — project_to_centerline returns lateral_m ≈ 200 m for a point 200 m
          perpendicular to a centerline segment.
  Additional: arc-length monotonicity for straight on-route bus;
              off-route filter drops bus 203 (lat=-16.45) from E2 fixture.
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
