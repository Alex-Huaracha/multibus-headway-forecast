"""Generate deterministic synthetic GPS parquets for the preprocessing test suite.

Run with:
    python -m tests.fixtures.synthetic

Outputs:
    tests/fixtures/synthetic_gps_e2.parquet   (~30 KB, 3 buses)
    tests/fixtures/synthetic_gps_e59.parquet  (~15 KB, 2 buses)

Design:
  E2 — empresaid=2, unidadid in {201, 202, 203}
    - Bus 201: ida lon=-71.55 → -71.50 then vuelta back; includes a 31-min gap.
    - Bus 202: same route, 5 min later (provides headway pairs with bus 201).
    - Bus 203: 10 pings off-route (lat=-16.45, lateral > 300 m) to test off-route drop.
    - 2 GPS jumps (lat-lon spike) on bus 201 to test speed-cap logic.
    - Total coverage: ~1 hour at 20 s ping interval.

  E59 — empresaid=59, unidadid in {501, 502}; NO direccion column.
    - Bus 501: normal straight route (no anomalies).
    - Bus 502: stops within 150 m of s_max for 6 min at 0 km/h → triggers terminal-cut.
    - Route is long enough that s_max and dwell position are >= 2 * TERMINAL_BAND_M apart.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

FIXTURES_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Shared geometry — straight east-west line through Arequipa (-16.4 lat)
# ---------------------------------------------------------------------------

BASE_LAT: float = -16.4
LON_START: float = -71.55
LON_END: float = -71.50
PING_INTERVAL_S: int = 20
T0 = datetime(2024, 1, 23, 7, 0, 0)  # 07:00 local


def _straight_pings(
    empresaid: int,
    unidadid: int,
    n_pings: int,
    lon_from: float,
    lon_to: float,
    t_start: datetime,
    lat_fixed: float = BASE_LAT,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Return a list of ping dicts for one bus traveling a straight east-west route."""
    if rng is None:
        rng = np.random.default_rng(42)
    lons = np.linspace(lon_from, lon_to, n_pings)
    # Tiny lat jitter (± 0.0001 deg ≈ ±11 m) to simulate real GPS noise
    lat_jitter = rng.normal(0, 0.0001, n_pings)
    rows = []
    for i in range(n_pings):
        rows.append({
            "empresaid": empresaid,
            "unidadid": unidadid,
            "time": t_start + timedelta(seconds=i * PING_INTERVAL_S),
            "lat": lat_fixed + lat_jitter[i],
            "lon": lons[i],
        })
    return rows


def _build_e2() -> pl.DataFrame:
    rng = np.random.default_rng(42)

    # --- Bus 201: ida then vuelta, with a 31-min gap after ida ---
    # ida: 80 pings from LON_START to LON_END (≈ 1600 s = 26.7 min)
    ida_201 = _straight_pings(2, 201, 80, LON_START, LON_END, T0, rng=rng)

    # 31-min gap: last ida ping + 31 min = gap_start
    gap_end_time = ida_201[-1]["time"] + timedelta(minutes=31)

    # vuelta: 80 pings from LON_END back to LON_START, starting after gap
    vuelta_201 = _straight_pings(2, 201, 80, LON_END, LON_START, gap_end_time, rng=rng)

    # Inject 2 GPS spikes into bus 201 (pings 10 and 11 of the ida — large jump, fast)
    ida_201[10]["lat"] = BASE_LAT + 0.5   # ~55 km north — impossible speed
    ida_201[11]["lat"] = BASE_LAT         # back on route

    pings_201 = ida_201 + vuelta_201

    # --- Bus 202: same route as 201's ida, starting 5 min later ---
    t_202 = T0 + timedelta(minutes=5)
    pings_202 = _straight_pings(2, 202, 80, LON_START, LON_END, t_202, rng=rng)

    # --- Bus 203: 10 pings off-route (lat=-16.45, ~5.5 km south of centerline) ---
    off_route_lat = -16.45   # lateral offset ≈ 5.5 km >> LATERAL_OFFSET_THRESHOLD_M=300 m
    pings_203 = _straight_pings(
        2, 203, 10, LON_START + 0.01, LON_START + 0.03, T0, lat_fixed=off_route_lat, rng=rng,
    )

    all_rows = pings_201 + pings_202 + pings_203
    return pl.DataFrame(all_rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("unidadid").cast(pl.Int64),
        pl.col("time").cast(pl.Datetime("us")),
        pl.col("lat").cast(pl.Float64),
        pl.col("lon").cast(pl.Float64),
    )


def _build_e59() -> pl.DataFrame:
    """E59 fixture — NO direccion column; bus 502 dwells near s_max for 6 min."""
    rng = np.random.default_rng(99)

    # Use a longer route so s_max >> TERMINAL_BAND_M (200 m).
    # lon range -71.60 → -71.50 is ~900 m per degree at this lat,
    # actual ~860 m * 10 / 100 ≈ 860 m total — need a wider range.
    # Use -71.65 → -71.50 ≈ 1290 m, comfortably > 2 * 200 m = 400 m.
    lon_start_59 = -71.65
    lon_end_59 = -71.50

    # Bus 501: straight ida, no anomalies (180 pings = 60 min at 20s interval)
    pings_501 = _straight_pings(59, 501, 180, lon_start_59, lon_end_59, T0, rng=rng)

    # Bus 502: ida, then stops near lon_end_59 for 6 min (18 pings @ 20s) with speed ~0
    t_502 = T0 + timedelta(minutes=3)
    ida_502 = _straight_pings(59, 502, 160, lon_start_59, lon_end_59, t_502, rng=rng)

    # Dwell pings: position within the terminal band of the route (near s_max but on-route).
    # Use lon_end_59 - 0.0008 ≈ 85 m west of the route endpoint, staying on the centerline
    # (lat=BASE_LAT, very small jitter so lateral_m << 300 m).
    dwell_start_time = ida_502[-1]["time"] + timedelta(seconds=PING_INTERVAL_S)
    dwell_lon = lon_end_59 - 0.0008   # ~85 m west of terminus, within TERMINAL_BAND_M
    dwell_pings = []
    for i in range(18):  # 18 * 20 s = 360 s = 6 min
        dwell_pings.append({
            "empresaid": 59,
            "unidadid": 502,
            "time": dwell_start_time + timedelta(seconds=i * PING_INTERVAL_S),
            "lat": BASE_LAT + rng.normal(0, 0.000005),  # nearly stationary, on-route
            "lon": dwell_lon + rng.normal(0, 0.000002),  # very small lon jitter
        })

    # Resume: bus continues forward a bit more after dwell
    resume_start = dwell_pings[-1]["time"] + timedelta(seconds=PING_INTERVAL_S)
    resume_502 = _straight_pings(59, 502, 10, dwell_lon, lon_end_59, resume_start, rng=rng)

    pings_502 = ida_502 + dwell_pings + resume_502

    all_rows = pings_501 + pings_502
    return pl.DataFrame(all_rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("unidadid").cast(pl.Int64),
        pl.col("time").cast(pl.Datetime("us")),
        pl.col("lat").cast(pl.Float64),
        pl.col("lon").cast(pl.Float64),
    )


def generate() -> None:
    """Write both synthetic parquets to FIXTURES_DIR."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    e2 = _build_e2()
    e59 = _build_e59()

    path_e2 = FIXTURES_DIR / "synthetic_gps_e2.parquet"
    path_e59 = FIXTURES_DIR / "synthetic_gps_e59.parquet"

    e2.write_parquet(path_e2)
    e59.write_parquet(path_e59)

    print(f"E2:  {len(e2)} rows → {path_e2} ({path_e2.stat().st_size // 1024} KB)")
    print(f"E59: {len(e59)} rows → {path_e59} ({path_e59.stat().st_size // 1024} KB)")

    # Sanity assertions
    assert set(e2["empresaid"].unique().to_list()) == {2}
    assert set(e2["unidadid"].unique().to_list()) == {201, 202, 203}
    off_route_rows = e2.filter(pl.col("unidadid") == 203)
    assert len(off_route_rows) >= 10, "Bus 203 must have >= 10 off-route pings"
    assert off_route_rows["lat"].min() <= -16.44, "Bus 203 pings must be at lat <= -16.44"

    assert set(e59["empresaid"].unique().to_list()) == {59}
    assert "direccion" not in e59.columns, "E59 must NOT have a direccion column"
    assert len(e59.filter(pl.col("unidadid") == 502)) >= 160 + 18, (
        "Bus 502 must have at least 160 ida pings + 18 dwell pings"
    )

    print("Assertions passed.")


if __name__ == "__main__":
    generate()
