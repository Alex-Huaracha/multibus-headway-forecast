"""Trip segmentation and snapshot grid construction.

assign_trip_ids — split each bus trajectory into trips on three cut conditions:
  1. GAP cut: dt_s > GAP_CUT_SECONDS (30 min) between consecutive pings.
  2. DIRECTION REVERSAL cut: primary direction flips +1↔-1 (transient 0s skipped).
  3. TERMINAL cut: bus stops near s_min/s_max for >= TERMINAL_DWELL_SECONDS (5 min).

build_snapshots — resample each bus to a minute-aligned uniform time grid, with
  linear interpolation of s and speed_kmh, and nearest-known direction by
  left-search.

Source: derived from build_notebook_03.py lines 606-660 (build_snapshots).
Trip segmentation is net-new for production — the probe deferred it.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import (
    GAP_CUT_SECONDS,
    PRODUCTIVE_PARAMS,
    TERMINAL_BAND_M,
    TERMINAL_DWELL_SECONDS,
    TERMINAL_MAX_SPEED_KMH,
)


def _compute_trip_ids_for_bus(
    s_arr: np.ndarray,
    dt_s_arr: np.ndarray,
    speed_arr: np.ndarray,
    dir_arr: np.ndarray,
    time_arr: np.ndarray,
    s_min: float,
    s_max: float,
) -> np.ndarray:
    """Compute trip_id per ping for a single (empresaid, unidadid).

    Returns a uint32 array of the same length as the input arrays, where each
    element is the trip_id for that ping (monotonically non-decreasing).

    Cut conditions:
      1. GAP: dt_s > GAP_CUT_SECONDS
      2. REVERSAL: last-known direction flips +1 ↔ -1 (direction==0 skipped)
      3. TERMINAL EXIT: bus leaves a near-terminal stopped zone that lasted >= DWELL_SECONDS

    The terminal cut is placed on the EXIT ping (the first ping after leaving
    the dwell zone that exceeded the duration threshold).
    """
    n = len(s_arr)
    trip_ids = np.zeros(n, dtype=np.uint32)
    current_trip = np.uint32(0)

    # --- Pre-compute per-ping flags ---
    is_gap = np.zeros(n, dtype=bool)
    is_gap[1:] = dt_s_arr[1:] > GAP_CUT_SECONDS

    # REVERSAL: forward-fill direction ignoring 0s; detect sign flip.
    last_dir = 0
    prev_last_dir = 0
    is_reversal = np.zeros(n, dtype=bool)
    for i in range(n):
        d = int(dir_arr[i])
        if d != 0:
            if prev_last_dir != 0 and d != last_dir:
                # But we only cut on the non-zero flip, not on the first transition.
                # We set is_reversal at position i (the new direction starts here).
                is_reversal[i] = True
            prev_last_dir = last_dir
            last_dir = d

    # TERMINAL DWELL: track cumulative time near terminal while stopped.
    near = (s_arr < (s_min + TERMINAL_BAND_M)) | (s_arr > (s_max - TERMINAL_BAND_M))
    stopped = speed_arr < TERMINAL_MAX_SPEED_KMH
    in_dwell = near & stopped

    is_terminal_exit = np.zeros(n, dtype=bool)
    dwell_start_time = None
    dwell_block_exceeded = False

    for i in range(n):
        if in_dwell[i]:
            if dwell_start_time is None:
                dwell_start_time = time_arr[i]
                dwell_block_exceeded = False
            elapsed = float(time_arr[i] - dwell_start_time) / 1e9  # ns → s
            if elapsed >= TERMINAL_DWELL_SECONDS:
                dwell_block_exceeded = True
        else:
            if dwell_block_exceeded:
                # This is the EXIT ping.
                is_terminal_exit[i] = True
            dwell_start_time = None
            dwell_block_exceeded = False

    # --- Assemble trip_ids from cut flags ---
    for i in range(n):
        if i > 0 and (is_gap[i] or is_reversal[i] or is_terminal_exit[i]):
            current_trip += np.uint32(1)
        trip_ids[i] = current_trip

    return trip_ids


def assign_trip_ids(
    gps: pl.DataFrame,
    s_min: float | None = None,
    s_max: float | None = None,
) -> pl.DataFrame:
    """Assign a monotonic trip_id per (empresaid, unidadid) from three cut criteria.

    Cut conditions (any one triggers a new trip_id):
      1. GAP: dt_s > GAP_CUT_SECONDS between consecutive pings.
      2. REVERSAL: last-known direction flips +1 ↔ -1 (transient direction==0
         pings are skipped using forward-fill of the last non-zero direction).
      3. TERMINAL: bus is within TERMINAL_BAND_M of s_min or s_max AND stopped
         (speed_kmh < TERMINAL_MAX_SPEED_KMH) for >= TERMINAL_DWELL_SECONDS.
         The cut is placed on the EXIT ping of the dwell run (i.e. the first
         ping where the bus resumes movement or leaves the terminal band).

    trip_id is UInt32, monotonically increasing per bus, starting from 0 at
    the first ping of each (empresaid, unidadid). Trips of length < 2 pings
    are KEPT (downstream filters may drop them; we do not silently merge).

    Args:
        gps: must have (empresaid, unidadid, time, s, speed_kmh, direction, dt_s)
             sorted by (empresaid, unidadid, time).
        s_min: corridor start arc-length (meters). Computed from data if None.
        s_max: corridor end arc-length (meters). Computed from data if None.

    Returns:
        gps + column (trip_id: UInt32).

    Failure modes:
    - If terminal-cut boundary semantics flip (cut on ENTRY instead of EXIT),
      test_trips.py::test_terminal_cut_creates_new_trip_on_e59 catches it.
    - If reversal cut is placed on the direction==0 pings (short stops),
      trip count inflates; test_trips.py::test_gap_cut_creates_new_trip
      provides a stable baseline count.
    """
    if s_min is None:
        s_min = float(gps["s"].min() or 0.0)
    if s_max is None:
        s_max = float(gps["s"].max() or 0.0)

    gps = gps.sort(["empresaid", "unidadid", "time"])

    # Use row_index to guarantee correct positional mapping back to the sorted frame
    # after group_by (maintain_order=True guarantees group iteration order but not
    # row order within the full frame after re-join).
    gps_indexed = gps.with_row_index("_row_idx")
    trip_id_parts: list[pl.DataFrame] = []

    for keys, sub in gps_indexed.group_by(["empresaid", "unidadid"], maintain_order=True):
        sub_sorted = sub.sort("time")
        dt_s = sub_sorted["dt_s"].fill_null(0.0).to_numpy().astype(np.float64)
        s_arr = sub_sorted["s"].to_numpy().astype(np.float64)
        speed_arr = sub_sorted["speed_kmh"].fill_null(0.0).to_numpy().astype(np.float64)
        dir_arr = sub_sorted["direction"].to_numpy().astype(np.int64)
        time_arr = sub_sorted["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)

        trip_ids = _compute_trip_ids_for_bus(
            s_arr, dt_s, speed_arr, dir_arr, time_arr, s_min, s_max
        )
        trip_id_parts.append(pl.DataFrame({
            "_row_idx": sub_sorted["_row_idx"],
            "trip_id": trip_ids,
        }))

    if not trip_id_parts:
        return gps.with_columns(pl.lit(0, dtype=pl.UInt32).alias("trip_id"))

    trip_id_df = pl.concat(trip_id_parts)
    result = gps_indexed.join(trip_id_df, on="_row_idx", how="left").drop("_row_idx")
    return result


def build_snapshots(
    gps: pl.DataFrame,
    grid_seconds: int = PRODUCTIVE_PARAMS.grid_seconds,
) -> pl.DataFrame:
    """Resample each bus to a minute-aligned uniform time grid per (empresaid, day).

    Grid alignment: uses epoch-floor pattern (t_min_s // grid_s) * grid_s to
    ensure all t_grid timestamps satisfy t.second == 0 (clarification #17 rule 1,
    INV-6).

    Interpolation:
      s         — linear interpolation (np.interp)
      speed_kmh — linear interpolation (null → 0.0 before interpolating)
      direction — nearest known by left-search (latest known state)
      trip_id   — nearest by left-search (when column is present)

    Only grid points within the bus's reported [t_min, t_max] window are emitted.
    Buses with < 2 pings are skipped.

    Source: build_notebook_03.py lines 606-660 with epoch-floor alignment added.
    """
    snaps_per_eday: list[pl.DataFrame] = []

    # Add day column if not present.
    if "day" not in gps.columns:
        gps = gps.with_columns(pl.col("time").dt.date().alias("day"))

    has_trip_id = "trip_id" in gps.columns
    has_lateral_m = "lateral_m" in gps.columns

    for keys, sub_eday in gps.group_by(["empresaid", "day"], maintain_order=True):
        e, day = keys[0], keys[1]

        # Compute minute-aligned epoch-floor grid (INV-6 / clarification #17 rule 1).
        # Use numpy int64 microseconds (matching polars Datetime["us"] storage) to
        # avoid Python datetime.timestamp() UTC/local ambiguity.
        t_min_us = int(sub_eday["time"].to_numpy().astype("datetime64[us]").astype(np.int64).min())
        t_max_us = int(sub_eday["time"].to_numpy().astype("datetime64[us]").astype(np.int64).max())
        grid_us = grid_seconds * 1_000_000   # grid in microseconds
        t_grid_us = np.arange(
            (t_min_us // grid_us) * grid_us,
            ((t_max_us // grid_us) + 1) * grid_us + 1,
            grid_us,
            dtype=np.int64,
        )
        # Also keep ns for interp (t_arr will be ns from the per-bus conversion below).
        t_grid_ns = t_grid_us * 1_000

        for bus_keys, sub in sub_eday.group_by(["unidadid"], maintain_order=True):
            bus = bus_keys[0]
            sub_sorted = sub.sort("time")
            t_arr = sub_sorted["time"].to_numpy().astype("datetime64[us]").astype(np.int64) * 1_000
            s_arr = sub_sorted["s"].to_numpy().astype(np.float64)
            v_arr = sub_sorted["speed_kmh"].fill_null(0.0).to_numpy().astype(np.float64)
            d_arr = sub_sorted["direction"].to_numpy().astype(np.int64)

            if len(t_arr) < 2:
                continue

            # Only interpolate within the bus's reported window.
            in_window = (t_grid_ns >= t_arr[0]) & (t_grid_ns <= t_arr[-1])
            if not in_window.any():
                continue

            tg = t_grid_ns[in_window]
            s_interp = np.interp(tg, t_arr, s_arr)
            v_interp = np.interp(tg, t_arr, v_arr)

            # Direction: nearest known (left-search), carrying the latest known state.
            idx_left = np.searchsorted(t_arr, tg, side="right") - 1
            idx_left = np.clip(idx_left, 0, len(d_arr) - 1)
            d_interp = d_arr[idx_left]

            row_data: dict = {
                "empresaid": np.full(len(tg), int(e), dtype=np.int64),
                "day": [day] * len(tg),
                "t": tg,
                "unidadid": np.full(len(tg), int(bus), dtype=np.int64),
                "s": s_interp.astype(np.float64),
                "speed_kmh": v_interp.astype(np.float64),
                "direction": d_interp.astype(np.int8),
            }

            if has_trip_id:
                tid_arr = sub_sorted["trip_id"].to_numpy().astype(np.uint32)
                idx_trip = np.searchsorted(t_arr, tg, side="right") - 1
                idx_trip = np.clip(idx_trip, 0, len(tid_arr) - 1)
                row_data["trip_id"] = tid_arr[idx_trip]

            if has_lateral_m:
                # Linear interpolation of lateral_m alongside s/speed_kmh.
                # lateral_m is a continuous geometric quantity (orthogonal distance
                # to centerline) — same regularity class as s. np.interp handles
                # null/NaN by propagating them; fill_null(0.0) is intentionally NOT
                # used here because a null lateral_m carries meaning (ping without
                # projection), and we want to propagate it faithfully.
                lat_arr = sub_sorted["lateral_m"].fill_null(float("nan")).to_numpy().astype(np.float64)
                lat_interp = np.interp(tg, t_arr, lat_arr)
                # Convert NaN back to null via a float64 series.
                lat_series = pl.Series("lateral_m", lat_interp, dtype=pl.Float64)
                row_data["lateral_m"] = lat_interp

            snaps_per_eday.append(pl.DataFrame(row_data))

    if not snaps_per_eday:
        return pl.DataFrame()

    snaps = pl.concat(snaps_per_eday)
    # t is stored as int64 nanoseconds (from t_grid_ns); convert to Datetime[us].
    snaps = snaps.with_columns(
        (pl.col("t") // 1_000).cast(pl.Datetime("us")).alias("t")
    )
    return snaps
