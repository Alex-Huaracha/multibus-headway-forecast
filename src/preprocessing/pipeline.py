"""Two-pass pipeline orchestration for multi-filar corridor empresas.

Provides:
  run_two_pass_pipeline — pass-1 single centerline → crude direction labels →
                          pass-2 per-direction centerlines → refined labels →
                          assign_trip_ids (R-PIPE1, R-PIPE2).
  run_single_pass_pipeline — the original single-pass path for backward
                             compatibility and as a discriminator for tests.

Both functions accept a pre-filtered empresa sub-frame (already empresa-filtered,
speed-attached with attach_observed_speed) and return a processed DataFrame.

Design: docs/decisiones-headway-fase2.md §8, SDD multi-filar-direction-balanced-
centerline design §4.
"""
from __future__ import annotations

import polars as pl

from .corridor import build_centerline, build_centerline_per_direction
from .direction import infer_direction
from .projection import project_per_direction, project_to_centerline
from .trips import assign_trip_ids
from .config import PRODUCTIVE_PARAMS


def run_two_pass_pipeline(
    gps: pl.DataFrame,
    *,
    empresaid: int,
    return_pass1_s: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.Series]:
    """Execute the two-pass centerline pipeline for a multi-filar empresa.

    Pass 1 (existing behavior):
        build_centerline → project_to_centerline → infer_direction

    Pass 2 (new, per-direction):
        build_centerline_per_direction → project_per_direction → infer_direction

    Then (R-PIPE2 invariant — ALWAYS after the second infer_direction):
        assign_trip_ids

    Args:
        gps: empresa-filtered GPS DataFrame with columns
             (empresaid, unidadid, time, lat, lon, speed_kmh).
        empresaid: which empresa is being processed.
        return_pass1_s: if True, return a tuple (result, pass1_s_snapshot) for
                        the s-continuity test (T3.2). Otherwise return result only.

    Returns:
        Processed pl.DataFrame with direction, s, lateral_m, trip_id columns.
        If return_pass1_s is True, returns (result, pass1_s_snapshot) instead.
    """
    # ---- Pass 1 ----
    cl_pass1 = build_centerline(gps, empresaid=empresaid)
    sub = project_to_centerline(gps, cl_pass1, empresaid=empresaid)
    sub = infer_direction(sub)

    # Snapshot pass-1 s for the continuity guard (spec Q-S-CONTINUITY)
    pass1_s_snapshot = sub["s"]

    # ---- Pass 2 ----
    cls = build_centerline_per_direction(
        sub,
        empresaid=empresaid,
        min_pings_per_dir=PRODUCTIVE_PARAMS.centerline_min_pings_per_direction,
    )
    sub = project_per_direction(sub, cls, empresaid=empresaid)
    sub = infer_direction(sub)

    # s-continuity runtime assertion (spec Q-S-CONTINUITY)
    assert "s" in sub.columns, (
        "s column missing after pass-2 projection (with_columns upsert failed)"
    )

    # ---- R-PIPE2: assign_trip_ids MUST run AFTER second infer_direction ----
    sub = assign_trip_ids(sub)

    if return_pass1_s:
        return sub, pass1_s_snapshot
    return sub


def run_single_pass_pipeline(
    gps: pl.DataFrame,
    *,
    empresaid: int,
) -> pl.DataFrame:
    """Execute the original single-pass centerline pipeline.

    Single pass:
        build_centerline → project_to_centerline → infer_direction → assign_trip_ids

    Used for empresas with centerline_strategy == "single" and as a discriminator
    baseline in the two-pass integration test (T3.3).

    Args:
        gps: empresa-filtered GPS DataFrame with speed_kmh already attached.
        empresaid: which empresa is being processed.

    Returns:
        Processed pl.DataFrame with direction, s, lateral_m, trip_id columns.
    """
    cl = build_centerline(gps, empresaid=empresaid)
    sub = project_to_centerline(gps, cl, empresaid=empresaid)
    sub = infer_direction(sub)
    sub = assign_trip_ids(sub)
    return sub
