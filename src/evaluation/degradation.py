"""Multi-horizon consolidation and degradation curve — Fase 6.5.

The multi-horizon experiment leaves one CSV per model/horizon plus a baselines
CSV in docs/resultados/csv-multihorizon/, all sharing the tidy schema
    corridor,direction,baseline,metric,value,horizon

Public API:
    load_results(results_dir) -> pl.DataFrame
        Concatenate every CSV in the directory into one validated long frame.
    degradation_table(df, metric, direction, corridor) -> pl.DataFrame
        Reshape into a model x horizon table (one row per model, h{H} columns)
        — the data source for the degradation curve figure.

Design decisions:
  - The error vs horizon comparison is built only from already-aggregated
    metrics (MAE/RMSE). Per-sample errors are NOT available in these CSVs, so
    statistical significance (Diebold-Mariano / Wilcoxon) is out of scope here.
  - No new pyproject.toml dependencies (polars + numpy + matplotlib present).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

REQUIRED_COLUMNS = ["corridor", "direction", "baseline", "metric", "value", "horizon"]


def load_results(results_dir: str | Path, pattern: str = "*.csv") -> pl.DataFrame:
    """Load and concatenate every matching CSV in ``results_dir`` into one frame.

    Every selected CSV must carry the tidy schema in ``REQUIRED_COLUMNS``. The
    ``horizon`` column is cast to ``Int64`` so horizons sort numerically (h2 < h10).

    Parameters
    ----------
    results_dir:
        Directory holding the multi-horizon result CSVs.
    pattern:
        Glob selecting which CSVs to load. Defaults to ``"*.csv"``. Pass
        ``"*_results_*.csv"`` to skip co-located foreign-schema files such as
        ``significance_multihorizon.csv`` that share the directory.

    Returns
    -------
    pl.DataFrame with columns ``REQUIRED_COLUMNS`` (horizon as Int64).

    Raises
    ------
    ValueError
        If no CSV matches ``pattern``, or any matching CSV lacks a required column.
    """
    results_dir = Path(results_dir)
    csv_paths = sorted(results_dir.glob(pattern))
    if not csv_paths:
        raise ValueError(
            f"load_results: no CSV matching {pattern!r} found in {results_dir}"
        )

    frames: list[pl.DataFrame] = []
    for path in csv_paths:
        frame = pl.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"load_results: {path.name} has wrong schema — missing {missing}"
            )
        frames.append(
            frame.select(REQUIRED_COLUMNS).with_columns(
                pl.col("horizon").cast(pl.Int64)
            )
        )

    return pl.concat(frames, how="vertical")


def degradation_table(
    df: pl.DataFrame,
    metric: str = "MAE",
    direction: str = "aggregate",
    corridor: str = "E2",
) -> pl.DataFrame:
    """Pivot the long frame into a model x horizon table for one slice.

    Filters to a single (metric, direction, corridor) slice, then pivots so each
    model/baseline is a row and each horizon an ``h{H}`` column (ascending).
    Cells with no measured value (e.g. a DL model not run at h=1) are null.

    Parameters
    ----------
    df:
        Long frame from :func:`load_results`.
    metric:
        ``"MAE"`` or ``"RMSE"``.
    direction:
        ``"aggregate"``, ``"+1"`` or ``"-1"``.
    corridor:
        ``"E2"`` or ``"E59"``.

    Returns
    -------
    pl.DataFrame with a ``baseline`` column followed by ``h{H}`` columns.

    Raises
    ------
    ValueError
        If the requested slice has no rows.
    """
    sliced = df.filter(
        (pl.col("metric") == metric)
        & (pl.col("direction") == direction)
        & (pl.col("corridor") == corridor)
    )
    if sliced.height == 0:
        raise ValueError(
            f"degradation_table: no rows for metric={metric!r} "
            f"direction={direction!r} corridor={corridor!r}"
        )

    table = sliced.pivot(
        on="horizon", index="baseline", values="value", aggregate_function="first"
    )

    horizon_cols = sorted((c for c in table.columns if c != "baseline"), key=int)
    rename = {c: f"h{c}" for c in horizon_cols}
    return table.select(["baseline", *horizon_cols]).rename(rename)
