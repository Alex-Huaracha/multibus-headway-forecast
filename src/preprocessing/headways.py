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

import logging
from collections import Counter

import numpy as np
import polars as pl

from .config import PRODUCTIVE_PARAMS, lateral_pair_threshold_for

logger = logging.getLogger(__name__)


def compute_pairs(snapshots: pl.DataFrame) -> pl.DataFrame:
    """Build consecutive (front, back) bus pairs per (empresaid, day, t, direction).

    For each snapshot group sorted by s (ascending), bus at rank i is "front" and
    bus at rank i-1 is "back". Drops direction == 0 rows.

    Lateral pair filter (R-LAT3): after pair formation, drops pairs where
    |lateral_m_front − lateral_m_back| > lateral_pair_threshold_for(empresaid).
    Rows where either lateral value is null are RETAINED (conservative).
    Filter is applied only when the input snapshot frame contains a `lateral_m`
    column. When the column is absent, all pairs are retained (backward-compatible).

    Args:
        snapshots: output of trips.build_snapshots with columns
                   (empresaid, day, t, unidadid, s, speed_kmh, direction[, lateral_m]).

    Returns:
        pl.DataFrame with columns:
          empresaid, day, t, direction,
          pair_rank (Int32, 1-indexed, dense per group),
          bus_front (Int64), bus_back (Int64),
          s_front (Float64), s_back (Float64),
          speed_front_kmh (Float64), speed_back_kmh (Float64),
          n_buses (Int32)[, lateral_m_front (Float64), lateral_m_back (Float64)].
          The lateral columns are present only when the input has lateral_m.

    Failure mode: if shift(1) is applied before sort, pair assignment is wrong;
    test_headways.py::test_pair_structure_count catches this.
    """
    has_lateral = "lateral_m" in snapshots.columns

    s = snapshots.filter(pl.col("direction") != 0)
    s = s.sort(["empresaid", "day", "t", "direction", "s"])

    group_cols = ["empresaid", "day", "t", "direction"]

    shift_exprs = [
        pl.col("s").shift(1).over(group_cols).alias("s_back"),
        pl.col("unidadid").shift(1).over(group_cols).alias("bus_back"),
        pl.col("speed_kmh").shift(1).over(group_cols).alias("speed_back_kmh"),
        pl.col("unidadid").count().over(group_cols).cast(pl.Int32).alias("n_buses"),
        # cum_count starts at 1 for the first row; after dropping the first row
        # (the "back" reference is null) we get ranks 2..N. Subtract 1 to get 1..N-1.
        (pl.col("s").cum_count().over(group_cols).cast(pl.Int32) - 1).alias("pair_rank"),
    ]
    if has_lateral:
        # Shift lateral_m to get the back-bus value after pairing.
        shift_exprs.append(
            pl.col("lateral_m").shift(1).over(group_cols).alias("lateral_m_back_raw")
        )
        # The front bus keeps its own lateral_m (renamed after select).
        shift_exprs.append(
            pl.col("lateral_m").alias("lateral_m_front_raw")
        )

    s = s.with_columns(shift_exprs)

    # Drop the first bus in each group (shift produces null for it).
    s = s.filter(pl.col("s_back").is_not_null())

    if has_lateral:
        # Step 6: filter cross-street pairs.
        # Build per-empresa threshold mapping via Python-side lookup (task note:
        # fallback from vectorised when/then if empresa list varies).
        empresa_ids = s["empresaid"].unique().to_list()
        threshold_map = {int(e): lateral_pair_threshold_for(int(e)) for e in empresa_ids}

        # Build a Polars expression: pl.col("empresaid").replace(mapping, default=global)
        # Conservative rule: retain if either lateral value is null.
        # retain when: lateral_m_front_raw IS NULL
        #           OR lateral_m_back_raw IS NULL
        #           OR abs(front - back) <= threshold
        global_threshold = PRODUCTIVE_PARAMS.lateral_pair_threshold_m
        keep_expr = (
            pl.col("lateral_m_front_raw").is_null()
            | pl.col("lateral_m_back_raw").is_null()
            | (
                (pl.col("lateral_m_front_raw") - pl.col("lateral_m_back_raw")).abs()
                <= pl.col("empresaid").replace_strict(
                    threshold_map,
                    default=global_threshold,
                    return_dtype=pl.Float64,
                )
            )
        )
        s = s.filter(keep_expr)

    select_exprs = [
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
    ]
    if has_lateral:
        select_exprs += [
            pl.col("lateral_m_front_raw").cast(pl.Float64).alias("lateral_m_front"),
            pl.col("lateral_m_back_raw").cast(pl.Float64).alias("lateral_m_back"),
        ]

    return s.select(select_exprs)


# Canonical bucket names reported by _find_last_crossing_ns (5 paths).
# The 6th bucket ("traj-miss") is reported by the outer compute_headways_c2 loop.
_CROSSING_BUCKETS = ("success", "cutoff-lt-2", "no-crossing", "ds-zero", "stale-crossing")


def _find_last_crossing_ns(
    t_arr: np.ndarray,
    s_arr: np.ndarray,
    T_ns: int,
    s_front: float,
    max_lookback_ns: float | None = None,
) -> tuple[float | None, str]:
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
        (t_cross, bucket) where t_cross is in nanoseconds (float) or None,
        and bucket is one of _CROSSING_BUCKETS identifying the outcome.
    """
    cutoff = int(np.searchsorted(t_arr, T_ns, side="right"))
    if cutoff < 2:
        return None, "cutoff-lt-2"

    s_past = s_arr[:cutoff]
    t_past = t_arr[:cutoff]

    diff = s_past - s_front

    # Case 1: exact zero crossing — bus_back was exactly at s_front.
    zero_mask = diff == 0.0
    if zero_mask.any():
        i = int(np.where(zero_mask)[0][-1])
        t_cross = float(t_past[i])
        if max_lookback_ns is not None and (T_ns - t_cross) > max_lookback_ns:
            return None, "stale-crossing"
        return t_cross, "success"

    # Case 2: sign-change crossing — bus_back's s straddled s_front.
    signs = np.sign(diff)
    cross_mask = (signs[:-1] * signs[1:]) < 0

    if not cross_mask.any():
        return None, "no-crossing"

    # Most recent crossing (last True in cross_mask).
    i = int(np.where(cross_mask)[0][-1])

    ds = s_past[i + 1] - s_past[i]
    if ds == 0.0:
        return None, "ds-zero"

    frac = float((s_front - s_past[i]) / ds)
    t_cross = float(t_past[i]) + frac * float(t_past[i + 1] - t_past[i])
    if max_lookback_ns is not None and (T_ns - t_cross) > max_lookback_ns:
        return None, "stale-crossing"
    return t_cross, "success"


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

    # Trajectory-miss counter: accumulate misses per (empresaid, direction) for diagnostics.
    miss_counter: Counter[tuple[int, int]] = Counter()
    total_counter: Counter[tuple[int, int]] = Counter()

    # Iterate per (empresaid, bus_back, direction) group — O(P_k × K_k) per group.
    for keys, sub_idx in pairs_indexed.group_by(
        ["empresaid", "bus_back", "direction"], maintain_order=False
    ):
        e, bus, dirc = int(keys[0]), int(keys[1]), int(keys[2])
        total_counter[(e, dirc)] += 1
        traj_key = (e, bus, dirc)
        if traj_key not in traj_index:
            miss_counter[(e, dirc)] += 1
            continue

        t_arr, s_arr = traj_index[traj_key]

        row_indices = sub_idx["_row_idx"].to_numpy().astype(np.int64)
        T_ns_group = t_ns_all[row_indices]
        s_front_group = s_front_all[row_indices]

        for j, (T_ns, sf) in enumerate(zip(T_ns_group, s_front_group)):
            t_cross, _reason = _find_last_crossing_ns(
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

    # Emit per-(empresa, direction) trajectory-miss diagnostics before returning.
    # Prefix [traj-miss] is machine-grepable; [traj-miss-warning] fires when > 30%.
    for (e, d), total in total_counter.items():
        miss = miss_counter.get((e, d), 0)
        pct = (miss / total * 100.0) if total else 0.0
        logger.info(
            "[traj-miss] empresa=%d dir=%d miss=%d/%d (%.1f%%)",
            e, d, miss, total, pct,
        )
        if pct > 30.0:
            logger.warning(
                "[traj-miss-warning] empresa=%d dir=%d miss_pct=%.1f%% exceeds 30%%",
                e, d, pct,
            )

    # Final schema cleanup: select R7 columns, preserving lateral diagnostic
    # columns when the upstream compute_pairs emitted them (R-LAT4 / AC-S1 / AC-S2).
    r7_cols = [
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
    ]
    if "lateral_m_front" in pairs_indexed.columns:
        r7_cols.append("lateral_m_front")
    if "lateral_m_back" in pairs_indexed.columns:
        r7_cols.append("lateral_m_back")
    return result.select(r7_cols)


def filter_snapshot_size(headways: pl.DataFrame, min_buses: int) -> pl.DataFrame:
    """Drop rows belonging to snapshots with fewer than min_buses active buses.

    INV-4: n_buses >= min_buses for all rows.

    Implementation: filter on the pre-computed n_buses column (set by compute_pairs).
    """
    return headways.filter(pl.col("n_buses") >= min_buses)
