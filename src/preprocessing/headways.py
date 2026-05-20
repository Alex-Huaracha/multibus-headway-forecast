"""Headway computation via C.2 — trailing crossing (pure polars + numpy).

compute_pairs — build the pair structure: for each (empresaid, day, t, direction),
    sort buses by s and emit one (front, back) row per consecutive pair.

compute_headways_c2 — for each pair, find the most recent past time when bus_back
    crossed s_front in its trajectory, and compute delta_t_min = T - t_cross.

Clarification #17 rule 2: when no crossing is found, the row is EMITTED with
delta_t_min = null (NOT dropped). This preserves pair_rank density (INV-3) and
n_buses consistency.

Note on NULL rows: they appear mostly in the first GRID_SECONDS of a bus's
trajectory (before bus_back has driven through any front position). The NULL
fraction should be < 5% globally; if higher, investigate trip-segmentation edge
cases. (Caveat per clarification #17 §Frequency expectation.)

Source: rewrite of build_notebook_03.py lines 751-813. The probe used pandas
    conversion + row-level Python loop. This implementation uses a trajectory
    index built with polars group_by + numpy numpy-escape per back-bus group
    (O(K) per group, not per pair).

winsorization: delta_t_min is stored RAW. Winsorization is a Fase 5 transformation
    applied at training time, NOT here (decisiones-headway-fase2.md §4 Caveat 2).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import PRODUCTIVE_PARAMS


def compute_pairs(snapshots: pl.DataFrame) -> pl.DataFrame:
    """Build consecutive (front, back) bus pairs per (empresaid, day, t, direction).

    For each snapshot group sorted by s (ascending), bus at rank i is "front" and
    bus at rank i-1 is "back". Drops direction == 0 rows.

    Args:
        snapshots: output of trips.build_snapshots with columns
                   (empresaid, day, t, unidadid, s, speed_kmh, direction).

    Returns:
        pl.DataFrame with columns:
          empresaid, day, t, direction,
          pair_rank (Int32, 1-indexed, dense per group),
          bus_front (Int64), bus_back (Int64),
          s_front (Float64), s_back (Float64),
          speed_front_kmh (Float64), speed_back_kmh (Float64),
          n_buses (Int32).

    Failure mode: if shift(1) is applied before sort, pair assignment is wrong;
    test_headways.py::test_pair_structure_count catches this.
    """
    s = snapshots.filter(pl.col("direction") != 0)
    s = s.sort(["empresaid", "day", "t", "direction", "s"])

    group_cols = ["empresaid", "day", "t", "direction"]

    s = s.with_columns([
        pl.col("s").shift(1).over(group_cols).alias("s_back"),
        pl.col("unidadid").shift(1).over(group_cols).alias("bus_back"),
        pl.col("speed_kmh").shift(1).over(group_cols).alias("speed_back_kmh"),
        pl.col("unidadid").count().over(group_cols).cast(pl.Int32).alias("n_buses"),
        # cum_count starts at 1 for the first row; after dropping the first row
        # (the "back" reference is null) we get ranks 2..N. Subtract 1 to get 1..N-1.
        (pl.col("s").cum_count().over(group_cols).cast(pl.Int32) - 1).alias("pair_rank"),
    ])

    # Drop the first bus in each group (shift produces null for it).
    s = s.filter(pl.col("s_back").is_not_null())

    return s.select([
        "empresaid",
        "day",
        "t",
        "direction",
        "pair_rank",
        pl.col("unidadid").alias("bus_front"),
        pl.col("bus_back").cast(pl.Int64),
        pl.col("s").alias("s_front"),
        pl.col("s_back").cast(pl.Float64),
        pl.col("speed_kmh").alias("speed_front_kmh"),
        pl.col("speed_back_kmh").cast(pl.Float64),
        "n_buses",
    ])


def _find_last_crossing_ns(
    t_arr: np.ndarray,
    s_arr: np.ndarray,
    T_ns: int,
    s_front: float,
    max_lookback_ns: float | None = None,
) -> float | None:
    """Find the most recent time (nanoseconds) when bus_back's s crossed s_front.

    Uses the probe's sign-change scan (build_notebook_03.py lines 796-806) on the
    trajectory of bus_back restricted to t <= T. Linear interpolation over the
    bracket gives the exact crossing nanosecond.

    Args:
        t_arr: int64 nanosecond timestamps, sorted ascending.
        s_arr: float64 arc-length values at those timestamps.
        T_ns:  snapshot time in nanoseconds (restrict to t <= T).
        s_front: arc-length of the front bus at T.
        max_lookback_ns: when not None, crossings older than this many nanoseconds
            before T are treated as 'no crossing found' and return None. Prevents
            stale historical crossings in multi-filar corridors (e.g. E2 Arequipa)
            from being emitted as absurd delta_t_min values (decisiones-headway-fase2 §3).

    Returns:
        t_cross in nanoseconds (float), or None if no crossing exists or the
        crossing is older than max_lookback_ns.
    """
    cutoff = int(np.searchsorted(t_arr, T_ns, side="right"))
    if cutoff < 2:
        return None

    s_past = s_arr[:cutoff]
    t_past = t_arr[:cutoff]

    diff = s_past - s_front

    # Case 1: exact zero crossing — bus_back was exactly at s_front.
    zero_mask = diff == 0.0
    if zero_mask.any():
        i = int(np.where(zero_mask)[0][-1])
        t_cross = float(t_past[i])
        if max_lookback_ns is not None and (T_ns - t_cross) > max_lookback_ns:
            return None
        return t_cross

    # Case 2: sign-change crossing — bus_back's s straddled s_front.
    signs = np.sign(diff)
    cross_mask = (signs[:-1] * signs[1:]) < 0

    if not cross_mask.any():
        return None

    # Most recent crossing (last True in cross_mask).
    i = int(np.where(cross_mask)[0][-1])

    ds = s_past[i + 1] - s_past[i]
    if ds == 0.0:
        return None

    frac = float((s_front - s_past[i]) / ds)
    t_cross = float(t_past[i]) + frac * float(t_past[i + 1] - t_past[i])
    if max_lookback_ns is not None and (T_ns - t_cross) > max_lookback_ns:
        return None
    return t_cross


def compute_headways_c2(
    snapshots: pl.DataFrame,
    gps: pl.DataFrame,
    min_buses: int = PRODUCTIVE_PARAMS.min_buses_per_snapshot,
    max_lookback_minutes: float = PRODUCTIVE_PARAMS.max_interpolation_lookback_minutes,
) -> pl.DataFrame:
    """C.2 trailing-crossing headway (pure polars + numpy).

    For each pair (bus_front at s_front, bus_back) at snapshot time T, finds the
    most recent past time when bus_back's s-trajectory crossed s_front (in the
    same direction) and computes:

        delta_t_min = (T - t_cross).total_seconds() / 60

    When no crossing is found (e.g. bus_back just entered the corridor and has
    not yet crossed s_front): emits the row with delta_t_min = null (NOT dropped).
    This is clarification #17 rule 2 — preserves INV-3 (dense pair_rank) and
    INV-4 (n_buses consistent with active bus count).

    Crossings whose interpolated t_cross is older than max_lookback_minutes are
    treated as 'no crossing' and emitted with delta_t_min = NULL (same semantics
    as clarification §2). This bound exists because multi-filar corridors project
    unrelated buses to the same s axis; without it, np.searchsorted finds ancient
    crossings and emits absurd headways (e.g. ~112 days on E2 dir=1).

    Algorithm:
    1. Build a trajectory index: group gps by (empresaid, unidadid, direction)
       → (t_arr, s_arr) sorted by time. This is O(N) per group.
    2. Build the pair frame via compute_pairs.
    3. Iterate groups (empresaid, bus_back, direction): for all pairs in this
       group, run the numpy sign-change scan and record delta_t_min. This is
       O(P_k × K_k) per group where P_k = pairs for this back-bus and K_k = traj
       length. Reassemble via an explicit row-index join.

    Args:
        snapshots: output of trips.build_snapshots.
        gps: post-projection, post-direction frame (full trajectory for crossing
             lookup). Should be filtered to the relevant empresa and day range.
        min_buses: drop snapshot groups with fewer buses (INV-4).
        max_lookback_minutes: crossings older than this many minutes before T are
            emitted as NULL (same as no-crossing). Default from ProductiveParams.

    Returns:
        pl.DataFrame matching R7 schema:
          t, direction, pair_rank (Int32), bus_front (Int64), bus_back (Int64),
          s_front, s_back, speed_front_kmh, speed_back_kmh,
          delta_t_min (Float64, may be null per clarification #17 rule 2),
          n_buses (Int32).

    Failure mode: if the pandas-conversion path is accidentally reintroduced,
    performance collapses on 47M-row E2 data. test_headways.py guards the polars
    purity requirement.
    """
    pairs = compute_pairs(snapshots)

    # Drop pairs from too-small snapshots (INV-4: n_buses >= min_buses).
    pairs = pairs.filter(pl.col("n_buses") >= min_buses)
    if pairs.is_empty():
        return pairs.with_columns(pl.lit(None, dtype=pl.Float64).alias("delta_t_min"))

    # Convert minutes → nanoseconds ONCE (kernel works in nanoseconds throughout).
    max_lookback_ns = float(max_lookback_minutes) * 60.0 * 1e9

    # --- Build trajectory index ---
    # Group gps by (empresaid, unidadid, direction) → sorted (t_arr, s_arr).
    gps_dir = gps.filter(pl.col("direction") != 0)
    traj_index: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    for keys, sub in gps_dir.group_by(
        ["empresaid", "unidadid", "direction"], maintain_order=False
    ):
        e, bus, dirc = int(keys[0]), int(keys[1]), int(keys[2])
        # Use microsecond-based int64 (Datetime["us"]) × 1000 → nanoseconds.
        t_arr = sub["time"].to_numpy().astype("datetime64[us]").astype(np.int64) * 1_000
        s_arr = sub["s"].to_numpy().astype(np.float64)
        order = np.argsort(t_arr)
        traj_index[(e, bus, dirc)] = (t_arr[order], s_arr[order])

    # --- Compute delta_t_min per pair ---
    # Attach a row index to pairs for result reassembly.
    pairs_indexed = pairs.with_row_index("_row_idx")

    # t column: snapshots use Datetime["us"], convert to nanoseconds for the lookup.
    t_ns_all = (
        pairs_indexed["t"].to_numpy().astype("datetime64[us]").astype(np.int64) * 1_000
    )
    s_front_all = pairs_indexed["s_front"].to_numpy().astype(np.float64)
    e_all = pairs_indexed["empresaid"].to_numpy().astype(np.int64)
    bus_back_all = pairs_indexed["bus_back"].to_numpy().astype(np.int64)
    dir_all = pairs_indexed["direction"].to_numpy().astype(np.int64)
    row_idx_all = pairs_indexed["_row_idx"].to_numpy().astype(np.int64)

    n = len(pairs_indexed)
    delta_t_min = np.full(n, np.nan, dtype=np.float64)

    # Iterate per (empresaid, bus_back, direction) group — O(P_k × K_k) per group.
    for keys, sub_idx in pairs_indexed.group_by(
        ["empresaid", "bus_back", "direction"], maintain_order=False
    ):
        e, bus, dirc = int(keys[0]), int(keys[1]), int(keys[2])
        traj_key = (e, bus, dirc)
        if traj_key not in traj_index:
            continue

        t_arr, s_arr = traj_index[traj_key]

        row_indices = sub_idx["_row_idx"].to_numpy().astype(np.int64)
        T_ns_group = t_ns_all[row_indices]
        s_front_group = s_front_all[row_indices]

        for j, (T_ns, sf) in enumerate(zip(T_ns_group, s_front_group)):
            t_cross = _find_last_crossing_ns(
                t_arr, s_arr, int(T_ns), float(sf),
                max_lookback_ns=max_lookback_ns,
            )
            if t_cross is not None:
                dt_ns = float(T_ns) - t_cross
                delta_t_min[row_indices[j]] = dt_ns / 1e9 / 60.0

    # Reassemble: NaN → null (clarification #17 rule 2 — emit null NOT drop).
    delta_series = pl.Series("delta_t_min", delta_t_min, dtype=pl.Float64)
    delta_series = delta_series.set(delta_series.is_nan(), None)

    result = pairs_indexed.drop("_row_idx").with_columns(delta_series)

    # Final schema cleanup: select only R7 columns.
    return result.select([
        "t",
        "direction",
        "pair_rank",
        "bus_front",
        "bus_back",
        "s_front",
        "s_back",
        "speed_front_kmh",
        "speed_back_kmh",
        "delta_t_min",
        "n_buses",
    ])


def filter_snapshot_size(headways: pl.DataFrame, min_buses: int) -> pl.DataFrame:
    """Drop rows belonging to snapshots with fewer than min_buses active buses.

    INV-4: n_buses >= min_buses for all rows.

    Implementation: filter on the pre-computed n_buses column (set by compute_pairs).
    """
    return headways.filter(pl.col("n_buses") >= min_buses)
