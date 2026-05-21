"""Temporal split and winsorization helpers for headway evaluation — Fase 3.

Public API:
    split_temporal(df: pl.DataFrame) -> pl.DataFrame
    winsorize_train_p99(df: pl.DataFrame) -> tuple[pl.DataFrame, float]

Constants (split date ranges, locked in spec §3 and design §5):
    SPLIT_TRAIN_START, SPLIT_TRAIN_END
    SPLIT_VAL_START,   SPLIT_VAL_END
    SPLIT_TEST_START,  SPLIT_TEST_END
    WINSOR_QUANTILE

Design decisions (locked in design §5 and §9):
  - Split key is pl.col("t").dt.date() membership, NOT row index.
  - Three ranges are exhaustive and mutually exclusive.
  - Rows outside all three ranges receive None (split column = null).
  - Winsorization threshold is computed on train rows only (AC-WINSOR-1, AC-WINSOR-2).
  - Null delta_t_min rows are NOT clipped (AC-WINSOR-3).
  - Rows above threshold are clipped (not dropped) (AC-WINSOR-4).
  - Constants live here (not PRODUCTIVE_PARAMS) — evaluation protocol concern.
  - WINSOR_QUANTILE and split dates are not added to pyproject.toml.
"""
from __future__ import annotations

from datetime import date

import polars as pl

# ---------------------------------------------------------------------------
# Split date range constants (spec §3, inclusive on both ends)
# ---------------------------------------------------------------------------

SPLIT_TRAIN_START: date = date(2023, 10, 1)
SPLIT_TRAIN_END:   date = date(2024, 1, 15)

SPLIT_VAL_START:   date = date(2024, 1, 16)
SPLIT_VAL_END:     date = date(2024, 2, 7)

SPLIT_TEST_START:  date = date(2024, 2, 8)
SPLIT_TEST_END:    date = date(2024, 2, 29)

WINSOR_QUANTILE: float = 0.99


def split_temporal(df: pl.DataFrame) -> pl.DataFrame:
    """Add a `split` column (Utf8) with values {"train", "val", "test"}.

    Membership is determined by pl.col("t").dt.date() against the six
    module-level date constants.  Rows outside all three ranges receive
    null (should not exist in the R7 v4 dataset; harness raises if found).

    Parameters
    ----------
    df:
        headways DataFrame containing at least a `t` (Datetime) column.

    Returns
    -------
    pl.DataFrame — input frame with one added column `split: Utf8`.
    """
    day = pl.col("t").dt.date()
    return df.with_columns(
        pl.when((day >= SPLIT_TRAIN_START) & (day <= SPLIT_TRAIN_END))
          .then(pl.lit("train"))
          .when((day >= SPLIT_VAL_START) & (day <= SPLIT_VAL_END))
          .then(pl.lit("val"))
          .when((day >= SPLIT_TEST_START) & (day <= SPLIT_TEST_END))
          .then(pl.lit("test"))
          .otherwise(None)
          .alias("split")
    )


def winsorize_train_p99(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, float]:
    """Clip delta_t_min to the 99th-percentile threshold computed on train rows only.

    The threshold is computed once as a scalar from non-null train-split rows.
    It is then applied as a clip ceiling to ALL rows (train + val + test).
    Null delta_t_min values are never clipped — they remain null (AC-WINSOR-3).

    Parameters
    ----------
    df:
        headways DataFrame that already has a `split` column (added by
        split_temporal) and a `delta_t_min` (Float64 nullable) column.

    Returns
    -------
    (clipped_df, threshold)
        clipped_df: same schema as df, delta_t_min clipped.
        threshold: the scalar train-p99 value used as the clip ceiling.

    Design note (AC-WINSOR-2 leakage guard):
        The filter `split == "train"` is applied BEFORE computing the quantile,
        so extreme outliers in val or test rows cannot shift the threshold.
    """
    threshold = float(
        df.filter(
            (pl.col("split") == "train") & pl.col("delta_t_min").is_not_null()
        )["delta_t_min"]
        .quantile(WINSOR_QUANTILE)
    )

    # Clip: preserve null rows; clip non-null rows to threshold from above.
    # pl.min_horizontal(col, lit(threshold)) would coerce null → 0 in some
    # polars versions, so we use the explicit when/then pattern (design §5).
    clipped = df.with_columns(
        pl.when(pl.col("delta_t_min").is_null())
          .then(None)
          .otherwise(
              pl.min_horizontal(pl.col("delta_t_min"), pl.lit(threshold))
          )
          .alias("delta_t_min")
    )
    return clipped, threshold
