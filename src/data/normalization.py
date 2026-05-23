"""Normalization module for supervised dataset construction — Fase 3 DL.

AC-NORM-1: compute_normalization_stats uses TRAIN ROWS ONLY.
AC-NORM-2: apply_zscore = (x - mean) / (std + Z_EPS) per (empresaid, direction).
AC-NORM-3: null delta_t_min passes through as null in the output column.
AC-NORM-4: no clipping — values with |z| > 5 are passed through unmodified (DL-8).
AC-NORM-5: zero torch imports at module level (INV-10).
AC-LEAK-1: leakage guard — caller must pass train_df only; this module does not filter.

Pre-condition: input df must already be winsorized via winsorize_train_p99 (INV-6).
Design decisions locked in design §2.3 and §5.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Z_EPS: float = 1e-8  # numerical safety: (x - mean) / (std + Z_EPS)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizationStats:
    """Per-(empresaid, direction) z-score parameters. Pure data, no torch.

    Attributes
    ----------
    means:
        Mean of delta_t_min per (empresaid, direction) computed from train rows only.
    stds:
        Standard deviation of delta_t_min per (empresaid, direction) from train only.
    """

    means: dict[tuple[int, int], float]
    stds: dict[tuple[int, int], float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookup_expr(
    stats: NormalizationStats,
    kind: str,
) -> pl.Expr:
    """Build a polars conditional expression mapping (empresaid, direction) → mean|std.

    Instead of a Python row-by-row loop, we build a pl.when/then chain over all
    known (empresa, direction) keys. Unknown keys return 0.0 (should not happen
    in a correctly filtered input; callers are expected to pass frames that
    only contain corridors present in train).

    Parameters
    ----------
    stats:
        NormalizationStats holding all known keys.
    kind:
        Either "mean" or "std".
    """
    lookup = stats.means if kind == "mean" else stats.stds

    if not lookup:
        return pl.lit(0.0)

    items = list(lookup.items())
    (emp0, dir0), val0 = items[0]
    expr = pl.when(
        (pl.col("empresaid") == emp0) & (pl.col("direction") == dir0)
    ).then(pl.lit(val0))

    for (emp, direction), val in items[1:]:
        expr = expr.when(
            (pl.col("empresaid") == emp) & (pl.col("direction") == direction)
        ).then(pl.lit(val))

    return expr.otherwise(pl.lit(0.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_normalization_stats(
    train_df: pl.DataFrame,
) -> NormalizationStats:
    """Mean/std of delta_t_min per (empresaid, direction) from TRAIN ROWS ONLY.

    AC-NORM-1: caller must pass a train-only DataFrame (no leakage protection
    inside this function — leakage guard is the caller's responsibility per INV-2).

    Null delta_t_min rows are excluded from the computation (standard mean/std
    ignores nulls in polars by default).

    Parameters
    ----------
    train_df:
        DataFrame with train rows only. Required columns: empresaid (Int64),
        direction (Int64), delta_t_min (Float64 nullable).

    Returns
    -------
    NormalizationStats with means and stds dicts keyed by (empresaid, direction).
    """
    agg = (
        train_df
        .group_by(["empresaid", "direction"])
        .agg(
            pl.col("delta_t_min").mean().alias("mean"),
            pl.col("delta_t_min").std().alias("std"),
        )
    )

    means: dict[tuple[int, int], float] = {}
    stds: dict[tuple[int, int], float] = {}

    for row in agg.iter_rows(named=True):
        key = (int(row["empresaid"]), int(row["direction"]))
        means[key] = float(row["mean"]) if row["mean"] is not None else 0.0
        stds[key] = float(row["std"]) if row["std"] is not None else 0.0

    return NormalizationStats(means=means, stds=stds)


def apply_zscore(
    df: pl.DataFrame,
    stats: NormalizationStats,
    *,
    out_col: str = "delta_t_min_z",
) -> pl.DataFrame:
    """Add z-scored column: (delta_t_min - mean) / (std + Z_EPS) per (empresa, direction).

    AC-NORM-2: formula is (x - mean) / (std + Z_EPS).
    AC-NORM-3: null delta_t_min rows produce null in out_col (no imputation).
    AC-NORM-4 + DL-8: no clipping — values with |z| > 5 pass through unchanged.

    Parameters
    ----------
    df:
        DataFrame to z-score. May be train, val, or test split. Required columns:
        empresaid, direction, delta_t_min.
    stats:
        NormalizationStats from compute_normalization_stats (train only).
    out_col:
        Name for the output z-scored column (default: delta_t_min_z).

    Returns
    -------
    pl.DataFrame — input frame with out_col (Float64 nullable) added.
    """
    mean_expr = _lookup_expr(stats, "mean")
    std_expr = _lookup_expr(stats, "std")

    return df.with_columns(
        pl.when(pl.col("delta_t_min").is_null())
        .then(None)
        .otherwise(
            (pl.col("delta_t_min") - mean_expr) / (std_expr + Z_EPS)
        )
        .alias(out_col)
        .cast(pl.Float64)
    )
