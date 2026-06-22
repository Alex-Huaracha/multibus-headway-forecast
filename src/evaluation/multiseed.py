"""Multi-seed variance and confidence intervals — C2 (NB15).

All deep-learning results in the paper come from a single seed. The reviewer's
automatic objection is: *was that seed lucky?* NB15 answers it by re-training the
frozen winning LSTM config with 5 seeds [42, 123, 456, 789, 999] at every horizon
h ∈ {1, 3, 5, 10}, on both corridors. Each kernel emits one CSV per horizon with
the canonical results schema plus a ``seed`` column:
    corridor,direction,baseline,metric,value,horizon,seed

Public API:
    load_multiseed(results_dir) -> pl.DataFrame
        Concatenate every per-seed CSV into one validated long frame.
    multiseed_summary(df, confidence) -> pl.DataFrame
        Reduce the per-seed samples to mean ± a Student-t confidence interval
        per (corridor, direction, baseline, metric, horizon) group.

Design decisions:
  - The interval uses the Student-t critical value (not the normal 1.96): with
    only n=5 seeds the t-distribution is the honest choice (t_{0.975,4}=2.776).
  - std is the SAMPLE std (ddof=1) — we estimate the seed population's spread
    from a sample, so the unbiased estimator is correct.
  - A single-seed group cannot estimate variance; it degrades to std=ci_half=0
    rather than emitting NaN, so the table never carries holes.
  - No new pyproject.toml dependencies (polars + numpy + scipy present).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

# Canonical results schema (see degradation.py) plus the per-seed column.
REQUIRED_COLUMNS = [
    "corridor", "direction", "baseline", "metric", "value", "horizon", "seed"
]
# The keys that identify one experimental cell; the seeds inside it are the sample.
GROUP_KEYS = ["corridor", "direction", "baseline", "metric", "horizon"]


def load_multiseed(
    results_dir: str | Path, pattern: str = "*_multiseed_*.csv"
) -> pl.DataFrame:
    """Load and concatenate every matching per-seed CSV into one long frame.

    Every selected CSV must carry the schema in ``REQUIRED_COLUMNS``. ``horizon``
    and ``seed`` are cast to ``Int64`` so they sort and group numerically.

    Parameters
    ----------
    results_dir:
        Directory holding the multi-seed result CSVs.
    pattern:
        Glob selecting which CSVs to load. Defaults to ``"*_multiseed_*.csv"`` so
        co-located single-run files (``*_results_*.csv``, which lack the ``seed``
        column) are skipped.

    Returns
    -------
    pl.DataFrame with columns ``REQUIRED_COLUMNS`` (horizon and seed as Int64).

    Raises
    ------
    ValueError
        If no CSV matches ``pattern``, or any matching CSV lacks a required column.
    """
    results_dir = Path(results_dir)
    csv_paths = sorted(results_dir.glob(pattern))
    if not csv_paths:
        raise ValueError(
            f"load_multiseed: no CSV matching {pattern!r} found in {results_dir}"
        )

    frames: list[pl.DataFrame] = []
    for path in csv_paths:
        frame = pl.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"load_multiseed: {path.name} has wrong schema — missing {missing}"
            )
        frames.append(
            frame.select(REQUIRED_COLUMNS).with_columns(
                pl.col("horizon").cast(pl.Int64), pl.col("seed").cast(pl.Int64)
            )
        )

    return pl.concat(frames, how="vertical")


def multiseed_summary(df: pl.DataFrame, confidence: float = 0.95) -> pl.DataFrame:
    """Reduce per-seed samples to mean ± a Student-t confidence interval.

    For each (corridor, direction, baseline, metric, horizon) group the function
    treats the per-seed ``value``s as a sample and reports the mean, the sample
    standard deviation (ddof=1), the half-width of the two-sided ``confidence``
    interval, the interval bounds, and the coefficient of variation.

    The half-width is ``t_{(1+confidence)/2, n-1} * std / sqrt(n)``. A group with a
    single seed (n=1) has no variance estimate and reports std=ci_half=0.

    Parameters
    ----------
    df:
        Long frame from :func:`load_multiseed`.
    confidence:
        Two-sided confidence level, e.g. ``0.95``.

    Returns
    -------
    pl.DataFrame with one row per group: ``GROUP_KEYS`` followed by ``n_seeds``,
    ``mean``, ``std``, ``ci_half``, ``ci_low``, ``ci_high`` and ``cv_pct``,
    sorted by the group keys.
    """
    rows: list[dict] = []
    for keys, group in df.group_by(GROUP_KEYS, maintain_order=True):
        vals = group["value"].to_numpy()
        n = vals.size
        mean = float(vals.mean())
        if n >= 2:
            std = float(vals.std(ddof=1))
            t_crit = float(stats.t.ppf((1 + confidence) / 2, n - 1))
            ci_half = t_crit * std / np.sqrt(n)
        else:
            std = 0.0
            ci_half = 0.0
        rows.append(
            {
                **dict(zip(GROUP_KEYS, keys)),
                "n_seeds": n,
                "mean": mean,
                "std": std,
                "ci_half": ci_half,
                "ci_low": mean - ci_half,
                "ci_high": mean + ci_half,
                "cv_pct": (100.0 * std / mean) if mean != 0 else 0.0,
            }
        )

    return pl.DataFrame(rows).sort(GROUP_KEYS)


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "docs/resultados/csv-multihorizon"
    summary = multiseed_summary(load_multiseed(src))
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=4):
        print(summary)
