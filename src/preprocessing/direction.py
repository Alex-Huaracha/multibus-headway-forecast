"""Direction inference from the sign of the smoothed arc-length derivative.

Primary method: sign(rolling_mean(ds, DIRECTION_SMOOTH_WIN)) per (empresaid, unidadid).
This is the SOLE primary source (decisiones-limpieza-fase2 §3.1). The `direccion`
heading field is used only as a cross-check diagnostic for empresas that have it
(E2/E4) and is never used to overwrite the primary signal.

Source: derived from build_notebook_03.py lines 471-504.
"""
from __future__ import annotations

import polars as pl

from .config import EMPRESA_CONFIG, PRODUCTIVE_PARAMS


def infer_direction(gps: pl.DataFrame) -> pl.DataFrame:
    """Infer per-ping direction from sign(rolling_mean(ds, DIRECTION_SMOOTH_WIN)).

    Direction values:
      +1  = ida  (increasing s)
      -1  = vuelta (decreasing s)
       0  = undetermined (insufficient or ambiguous data at window start/end)

    Args:
        gps: must have (empresaid, unidadid, s) columns and be sorted by
             (empresaid, unidadid, time).

    Returns:
        gps + columns (ds_raw: Float64, ds_smooth: Float64, direction: Int8).

    Source: build_notebook_03.py lines 475-487.
    """
    win = PRODUCTIVE_PARAMS.direction_smooth_win

    gps = gps.with_columns([
        (
            pl.col("s") - pl.col("s").shift(1).over(["empresaid", "unidadid"])
        ).alias("ds_raw"),
    ])
    gps = gps.with_columns([
        pl.col("ds_raw")
          .rolling_mean(window_size=win, min_samples=1)
          .over(["empresaid", "unidadid"])
          .alias("ds_smooth"),
    ])
    gps = gps.with_columns([
        pl.when(pl.col("ds_smooth") > 0).then(pl.lit(1, dtype=pl.Int8))
          .when(pl.col("ds_smooth") < 0).then(pl.lit(-1, dtype=pl.Int8))
          .otherwise(pl.lit(0, dtype=pl.Int8))
          .alias("direction"),
    ])
    return gps


def cross_check_heading(gps: pl.DataFrame, empresaid: int) -> pl.DataFrame:
    """DIAGNOSTIC ONLY — add a heading_agrees column for empresas with GPS heading.

    For empresas with EMPRESA_CONFIG[e].has_heading = True, computes agreement
    between the primary direction signal and a threshold-based heading
    classification. Does NOT alter the primary `direction` column.

    - `direccion == 0` is treated as NULL (sentinel value, not north heading).
    - heading classified: 45–135° → +1 (ida), 225–315° → -1 (vuelta), else 0.
    - heading_agrees = (primary direction == heading direction) when both non-zero.

    For empresas without heading (has_heading=False, e.g. E59) this is a no-op
    that returns gps unchanged.

    Args:
        gps: frame with (empresaid, direction) columns.
        empresaid: empresa identifier.

    Returns:
        gps + column (heading_agrees: Boolean) when applicable; unchanged otherwise.
    """
    cfg = EMPRESA_CONFIG.get(empresaid)
    if cfg is None or not cfg.has_heading:
        return gps
    if "direccion" not in gps.columns:
        return gps

    gps = gps.with_columns([
        # Treat direccion == 0 as null (sentinel).
        pl.when(pl.col("direccion") == 0)
          .then(None)
          .otherwise(pl.col("direccion"))
          .alias("_heading_clean"),
    ])
    gps = gps.with_columns([
        pl.when(
            (pl.col("_heading_clean") >= 45) & (pl.col("_heading_clean") <= 135)
        ).then(pl.lit(1, dtype=pl.Int8))
          .when(
            (pl.col("_heading_clean") >= 225) & (pl.col("_heading_clean") <= 315)
        ).then(pl.lit(-1, dtype=pl.Int8))
          .otherwise(pl.lit(0, dtype=pl.Int8))
          .alias("_heading_dir"),
    ])
    gps = gps.with_columns([
        pl.when(
            (pl.col("direction") != 0) & (pl.col("_heading_dir") != 0)
        ).then(pl.col("direction") == pl.col("_heading_dir"))
          .otherwise(None)
          .alias("heading_agrees"),
    ]).drop(["_heading_clean", "_heading_dir"])

    return gps
