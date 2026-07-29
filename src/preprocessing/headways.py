"""Headway computation via C.2 — trailing crossing (pure polars + numpy).

⚠️ READ THIS BEFORE TOUCHING THE PAIR SEMANTICS ⚠️
The names ``front`` and ``back`` in this module are the MIRROR IMAGE of physical
motion, and every column they produce carries that inversion downstream:

    bus_back / s_back   →  the bus physically AHEAD  (the leader)
    bus_front / s_front →  the bus physically BEHIND (the follower)

The arithmetic is correct; only the labels are backwards. Verified against
``data/processed/`` for all three corridors: the shipped computation yields a
median headway of 4.96 min and a median implied speed of 9.6 km/h (70% of rows
inside 5-40 km/h), while the reading the names suggest yields 11.65 min and
2.0 km/h with 29% coverage. See ``docs/decisiones-headway-fase2.md`` §2.1.

Why the labels invert, in both directions — this follows from a DEFINITION, so
it cannot drift. ``direction`` is ``sign(rolling_mean(ds))`` (see
``direction.infer_direction``), therefore within a direction group the sense in
which ``s`` moves is fixed by construction:

    direction -1  →  s DECREASES as the bus advances  →  leader has the LOWER s
    direction +1  →  s INCREASES as the bus advances  →  leader has the HIGHER s

The sort key is ``s`` for direction -1 and ``-s`` for direction +1
(== CALIBRATED_INVERTED_DIRECTION), so ascending order puts the LEADER FIRST in
both cases. ``shift(1)`` then hands the first row to the ``back`` columns. The
inversion is therefore uniform across directions, which is exactly what makes
the pipeline correct despite the naming.

Do NOT "fix" this by renaming the columns. They are baked into the processed
parquet schema, the notebook builders and the downloaded residuals; renaming
cascades through all of it for zero analytical gain.

compute_pairs — build the pair structure: for each (empresaid, day, t, direction),
    sort buses by the direction-corrected key and emit one row per consecutive
    pair. The leader goes to the ``back`` slot (see above).

compute_headways_c2 — for each pair, find the most recent past time when the
    LEADER (``bus_back``) crossed the FOLLOWER's current position (``s_front``),
    and compute delta_t_min = T - t_cross. That elapsed time is the headway: how
    long ago the leading bus stood where the following bus stands now.

Clarification #17 rule 2: when no crossing is found, the row is EMITTED with
delta_t_min = null (NOT dropped). This preserves pair_rank density (INV-3) and
n_buses consistency.

Note on NULL rows: the dominant bucket is ``stale-crossing``, not ``no-crossing``
(E2: 42% vs 0.1%) — multi-filar corridors project unrelated buses onto one axis,
so a crossing is almost always found and the lookback bound is what rejects the
old ones. Genuine ``no-crossing`` rows sit in the first GRID_SECONDS of a
trajectory, before the leader has driven through any follower position.

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

from .config import CALIBRATED_INVERTED_DIRECTION, PRODUCTIVE_PARAMS, lateral_pair_threshold_for

logger = logging.getLogger(__name__)


def compute_pairs(snapshots: pl.DataFrame) -> pl.DataFrame:
    """Build consecutive (front, back) bus pairs per (empresaid, day, t, direction).

    For each snapshot group sorted by the direction-corrected key (ascending),
    bus at rank i is "front" and bus at rank i-1 is "back". Drops direction == 0
    rows.

    ⚠️ ``back`` is the bus physically AHEAD and ``front`` the one BEHIND — the
    labels are the mirror image of motion. See the module docstring for the
    measurement that establishes this and why the names are not being changed.

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
    # Direction-conditional sort key (SDD dir1-pair-ordering-h7, Encoding A).
    #
    # Purpose: make the LEADER land in the same slot in both directions.
    # `direction` is sign(rolling_mean(ds)) by definition, so within a direction
    # group the sense of s is fixed: dir -1 => s decreases as the bus advances
    # (leader has the LOWER s); dir +1 => s increases (leader has the HIGHER s).
    # Negating s for direction +1 (== CALIBRATED_INVERTED_DIRECTION) makes
    # ascending sort put the leader FIRST in both cases, so shift(1) hands it to
    # the `back` columns uniformly.
    #
    # The `back` slot therefore holds the bus physically AHEAD, and `front` the
    # one behind — the labels are inverted, the pairing is not. An earlier
    # version of this comment claimed the opposite ("places the physically-front
    # bus last"), which is false in BOTH directions and has already cost one
    # false bug report. See the module docstring.
    #
    # The negation is sort-time only; s_front/s_back retain raw arc-length values.
    _s_sort_key = (
        pl.when(pl.col("direction") == CALIBRATED_INVERTED_DIRECTION)
        .then(-pl.col("s"))
        .otherwise(pl.col("s"))
    )
    s = s.sort(["empresaid", "day", "t", "direction", _s_sort_key])

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

    Read with the module docstring: the trajectory handed in belongs to the bus
    physically AHEAD (``bus_back``) and ``s_front`` is where the bus BEHIND
    currently sits. The leader has already driven through that position, which is
    what makes the search well posed — swap the two and coverage falls to 29% and
    the implied speed to 2 km/h, because you are asking a bus when it passed a
    place it has not reached yet.

    Args:
        t_arr: int64 nanosecond timestamps, sorted ascending.
        s_arr: float64 arc-length values at those timestamps.
        T_ns:  snapshot time in nanoseconds (restrict to t <= T).
        s_front: arc-length of the trailing bus at T — the position whose
            crossing time is being recovered.
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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """C.2 trailing-crossing headway (pure polars + numpy).

    For each pair at snapshot time T, finds the most recent past time when
    ``bus_back``'s s-trajectory crossed ``s_front`` (in the same direction) and
    computes:

        delta_t_min = (T - t_cross).total_seconds() / 60

    In physical terms — and the labels invert, see the module docstring —
    ``bus_back`` is the LEADER and ``s_front`` is where the FOLLOWER stands at T,
    so delta_t_min is how long ago the leading bus was where the following bus is
    now. That is the headway.

    When no crossing is found (the leader has not yet driven through any follower
    position on this trajectory): emits the row with delta_t_min = null (NOT
    dropped).
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
        (headways_df, null_buckets_df) where:
          headways_df matches R7 schema:
            t, direction, pair_rank (Int32), bus_front (Int64), bus_back (Int64),
            s_front, s_back, speed_front_kmh, speed_back_kmh,
            delta_t_min (Float64, may be null per clarification #17 rule 2),
            n_buses (Int32).
          null_buckets_df schema (INV-N1 through INV-N4):
            empresaid (Int64), direction (Int8), bucket (Utf8),
            count (Int64), total_pairs (Int64).
            One row per (empresaid, direction, bucket) — always 6 buckets per group,
            count=0 rows included. INV-N2: sum(count) == total_pairs per group.

    Failure mode: if the pandas-conversion path is accidentally reintroduced,
    performance collapses on 47M-row E2 data. test_headways.py guards the polars
    purity requirement.
    """
    # Canonical bucket name ordering for null_buckets_df construction.
    _all_buckets = ("traj-miss", "cutoff-lt-2", "no-crossing", "ds-zero", "stale-crossing", "success")

    # Schema for null_buckets_df (locked; changes require spec revision).
    _null_buckets_schema = {
        "empresaid": pl.Int64,
        "direction": pl.Int8,
        "bucket": pl.Utf8,
        "count": pl.Int64,
        "total_pairs": pl.Int64,
    }

    pairs = compute_pairs(snapshots)

    # Drop pairs from too-small snapshots (INV-4: n_buses >= min_buses).
    pairs = pairs.filter(pl.col("n_buses") >= min_buses)
    if pairs.is_empty():
        empty_headways = pairs.with_columns(pl.lit(None, dtype=pl.Float64).alias("delta_t_min"))
        empty_null_buckets = pl.DataFrame(schema=_null_buckets_schema)
        return empty_headways, empty_null_buckets

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

    n = len(pairs_indexed)
    delta_t_min = np.full(n, np.nan, dtype=np.float64)

    # Bucket counter: accumulate per (empresaid, direction, bucket) across all pairs.
    # CORRECTNESS NOTE: total_counter counts PAIRS (not groups). Each increment by
    # len(sub_idx) (the number of rows/pairs in the group) ensures the discrimination
    # invariant INV-N2: sum(count over all buckets) == total_pairs per (e, d).
    # The previous code used += 1 (counting groups, not pairs) — that was wrong.
    bucket_counter: Counter[tuple[int, int, str]] = Counter()
    total_counter: Counter[tuple[int, int]] = Counter()

    # Iterate per (empresaid, bus_back, direction) group — O(P_k × K_k) per group.
    for keys, sub_idx in pairs_indexed.group_by(
        ["empresaid", "bus_back", "direction"], maintain_order=False
    ):
        e, bus, dirc = int(keys[0]), int(keys[1]), int(keys[2])
        # Count pairs in this group (not the group itself — correctness fix).
        total_counter[(e, dirc)] += len(sub_idx)
        traj_key = (e, bus, dirc)
        if traj_key not in traj_index:
            # All pairs in this group are traj-miss.
            bucket_counter[(e, dirc, "traj-miss")] += len(sub_idx)
            continue

        t_arr, s_arr = traj_index[traj_key]

        row_indices = sub_idx["_row_idx"].to_numpy().astype(np.int64)
        T_ns_group = t_ns_all[row_indices]
        s_front_group = s_front_all[row_indices]

        for j, (T_ns, sf) in enumerate(zip(T_ns_group, s_front_group)):
            t_cross, reason = _find_last_crossing_ns(
                t_arr, s_arr, int(T_ns), float(sf),
                max_lookback_ns=max_lookback_ns,
            )
            # Count each pair outcome by its bucket (1 per pair call).
            bucket_counter[(e, dirc, reason)] += 1
            if t_cross is not None:
                dt_ns = float(T_ns) - t_cross
                delta_t_min[row_indices[j]] = dt_ns / 1e9 / 60.0

    # Reassemble: NaN → null (clarification #17 rule 2 — emit null NOT drop).
    delta_series = pl.Series("delta_t_min", delta_t_min, dtype=pl.Float64)
    delta_series = delta_series.set(delta_series.is_nan(), None)

    result = pairs_indexed.drop("_row_idx").with_columns(delta_series)

    # --- Build null_buckets_df from counters ---
    # All 6 buckets per (empresaid, direction) with count=0 when bucket didn't fire.
    # This satisfies INV-N2: sum(count) == total_pairs for every group.
    bucket_rows = []
    for (e, d), total in total_counter.items():
        for b in _all_buckets:
            bucket_rows.append({
                "empresaid": e,
                "direction": d,
                "bucket": b,
                "count": int(bucket_counter.get((e, d, b), 0)),
                "total_pairs": int(total),
            })
    null_buckets_df = pl.DataFrame(bucket_rows, schema=_null_buckets_schema)

    # Emit per-(empresa, direction) trajectory-miss diagnostics.
    # Prefix [traj-miss] is machine-grepable; [traj-miss-warning] fires when > 30%.
    for (e, d), total in total_counter.items():
        miss = bucket_counter.get((e, d, "traj-miss"), 0)
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
    return result.select(r7_cols), null_buckets_df


def filter_snapshot_size(headways: pl.DataFrame, min_buses: int) -> pl.DataFrame:
    """Drop rows belonging to snapshots with fewer than min_buses active buses.

    INV-4: n_buses >= min_buses for all rows.

    Implementation: filter on the pre-computed n_buses column (set by compute_pairs).
    """
    return headways.filter(pl.col("n_buses") >= min_buses)
