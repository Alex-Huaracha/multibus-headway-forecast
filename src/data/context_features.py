"""Context features module for supervised dataset construction — Fase 3 DL.

AC-CTX-1: encode_context adds hour_sin, hour_cos at midnight → (0, 1).
AC-CTX-2: encode_context adds dow_sin, dow_cos with period 7; emits 5 named columns.
AC-CTX-3: load_atypical_days(None) returns empty set (graceful fallback, DL-2).
AC-CTX-4: load_atypical_days(path) returns set[date] from CSV when file exists.
AC-CTX-5: atypical_flag=1.0 when timestamp date in atypical_dates, else 0.0.
AC-CTX-6: zero torch imports at module level (INV-10, DL-10).

Design decisions locked in design §2.4 and §5:
  - encode_context operates on a DataFrame with a `t` (Datetime) column.
  - Cyclical encoding: sin(2π * value / period), cos(2π * value / period).
  - atypical_flag = 1.0 when t.date() in atypical_dates else 0.0.
  - DL-2: graceful fallback to atypical_flag=0 when path is None or missing.
  - No torch imports (INV-10).
"""
from __future__ import annotations

import logging
import math
import warnings
from datetime import date
from pathlib import Path

import polars as pl

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "atypical_flag",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cyclical_pair(col: pl.Expr, period: int, prefix: str) -> list[pl.Expr]:
    """Emit [sin_expr, cos_expr] aliased <prefix>_sin, <prefix>_cos.

    Encoding: sin(2π * col / period), cos(2π * col / period).

    Parameters
    ----------
    col:
        Polars expression that yields a numeric value (e.g. hour 0-23, dow 0-6).
    period:
        Full cycle length (24 for hour, 7 for day-of-week).
    prefix:
        Column name prefix ("hour" or "dow").
    """
    angle = col * (2.0 * math.pi / period)
    return [
        angle.sin().alias(f"{prefix}_sin"),
        angle.cos().alias(f"{prefix}_cos"),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_context(
    df: pl.DataFrame,
    *,
    atypical_dates: set[date] | None = None,
) -> pl.DataFrame:
    """Add 5 context columns derived from the `t` (Datetime) column.

    AC-CTX-1..5. DL-2 graceful fallback: atypical_flag=0 when atypical_dates
    is None or empty.

    Parameters
    ----------
    df:
        DataFrame with a `t` (Datetime[us]) column.
    atypical_dates:
        Set of dates that are atypical (e.g. holidays, strikes). When None
        or empty, atypical_flag is 0.0 for all rows.

    Returns
    -------
    pl.DataFrame — input frame with 5 additional columns appended in the order
    defined by CONTEXT_FEATURE_NAMES.
    """
    if atypical_dates is None:
        atypical_dates = set()

    # Cyclical hour and day-of-week encodings.
    # polars dt.weekday() returns ISO weekday: Monday=1 .. Sunday=7.
    # We convert to 0-indexed (Monday=0 .. Sunday=6) to align with Python convention
    # so that midnight Monday → dow=0 → dow_sin=sin(0)=0, dow_cos=cos(0)=1 (AC-CTX-1).
    hour_expr = pl.col("t").dt.hour().cast(pl.Float64)
    dow_expr = (pl.col("t").dt.weekday() - 1).cast(pl.Float64)

    sin_cos_exprs: list[pl.Expr] = [
        *_cyclical_pair(hour_expr, 24, "hour"),
        *_cyclical_pair(dow_expr, 7, "dow"),
    ]

    # Atypical flag: 1.0 if the date is in the atypical set, else 0.0.
    if atypical_dates:
        # Build a list of date literals to check membership against.
        atypical_list = sorted(atypical_dates)
        date_col = pl.col("t").dt.date()
        flag_expr = pl.lit(0.0)

        # Chain when/then for each atypical date.
        flag_chain = pl.when(
            date_col == pl.lit(atypical_list[0])
        ).then(pl.lit(1.0))
        for d in atypical_list[1:]:
            flag_chain = flag_chain.when(
                date_col == pl.lit(d)
            ).then(pl.lit(1.0))
        flag_expr = flag_chain.otherwise(pl.lit(0.0))
    else:
        flag_expr = pl.lit(0.0)

    return df.with_columns(
        *sin_cos_exprs,
        flag_expr.cast(pl.Float64).alias("atypical_flag"),
    )


def load_atypical_days(
    path: Path | str | None,
) -> set[date]:
    """Read CSV with at least a `date` column; return set[date].

    AC-CTX-3 + DL-2: returns empty set when path is None OR file does not exist.
    A warning is emitted when the path is non-None but missing (so callers know
    the fallback was triggered — not a silent failure).

    Parameters
    ----------
    path:
        Path to a CSV file with a `date` column (ISO-8601 format).
        May be None, a string, or a Path object.

    Returns
    -------
    set[date] — parsed dates, or empty set on fallback.
    """
    if path is None:
        return set()

    resolved = Path(path)
    if not resolved.exists():
        warnings.warn(
            f"load_atypical_days: file not found at '{resolved}'; "
            "falling back to empty atypical set (atypical_flag=0 for all rows). "
            "DL-2 graceful fallback.",
            stacklevel=2,
        )
        return set()

    df = pl.read_csv(resolved, try_parse_dates=True)
    if "date" not in df.columns:
        warnings.warn(
            f"load_atypical_days: CSV at '{resolved}' has no 'date' column; "
            "falling back to empty set.",
            stacklevel=2,
        )
        return set()

    dates: set[date] = set()
    for val in df["date"].to_list():
        if val is not None:
            if isinstance(val, date):
                dates.add(val)
            else:
                try:
                    from datetime import datetime as _dt
                    dates.add(_dt.fromisoformat(str(val)).date())
                except ValueError:
                    _log.warning("Skipping unparseable date value: %s", val)

    return dates
