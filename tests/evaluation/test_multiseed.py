"""Tests for src/evaluation/multiseed.py — multi-seed variance + confidence intervals.

The C2 experiment (NB15) re-trains the frozen winning LSTM config with 5 seeds
[42, 123, 456, 789, 999] at every horizon h ∈ {1, 3, 5, 10}, on both corridors.
Each kernel emits one CSV per horizon sharing the schema
    corridor,direction,baseline,metric,value,horizon,seed
i.e. the canonical results schema plus a ``seed`` column.

This module concatenates them and reduces the per-seed samples to mean ± a
Student-t confidence interval per (corridor, direction, baseline, metric,
horizon) group — the artifact that answers the reviewer's "lucky seed?" claim.

Acceptance criteria:
    AC-LOAD-1  load_multiseed concatenates every matching CSV into one frame
    AC-LOAD-2  load_multiseed preserves the 7-column schema (seed as integer)
    AC-LOAD-3  load_multiseed raises if the dir has no matching CSV
    AC-LOAD-4  load_multiseed raises if a CSV lacks the seed column
    AC-LOAD-5  load_multiseed(pattern=...) skips co-located foreign CSVs
    AC-SUM-1   multiseed_summary groups by the 5 keys, n_seeds counts the seeds
    AC-SUM-2   mean is the per-group average across seeds
    AC-SUM-3   std is the SAMPLE std (ddof=1); ci_half = t * std / sqrt(n)
    AC-SUM-4   ci_low/ci_high bracket the mean by ci_half; cv_pct = 100*std/mean
    AC-SUM-5   a higher confidence widens the interval
    AC-SUM-6   a single-seed group degrades gracefully (std=0, ci_half=0)
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy import stats

from src.evaluation.multiseed import load_multiseed, multiseed_summary

SCHEMA = ["corridor", "direction", "baseline", "metric", "value", "horizon", "seed"]


def _write_csv(path, rows):
    pl.DataFrame(rows, schema=SCHEMA, orient="row").write_csv(path)


@pytest.fixture
def multiseed_dir(tmp_path):
    """One five-seed group (E2/aggregate/LSTM/MAE/h5) plus a second horizon file."""
    _write_csv(
        tmp_path / "lstm_multiseed_h5.csv",
        [
            ("E2", "aggregate", "LSTM", "MAE", v, 5, seed)
            for v, seed in zip(
                [4.0, 4.2, 4.4, 4.6, 4.8], [42, 123, 456, 789, 999]
            )
        ],
    )
    _write_csv(
        tmp_path / "lstm_multiseed_h10.csv",
        [
            ("E2", "aggregate", "LSTM", "MAE", v, 10, seed)
            for v, seed in zip([5.0, 5.5, 6.0], [42, 123, 456])
        ],
    )
    return tmp_path


# --- AC-LOAD ---------------------------------------------------------------
class TestLoadMultiseed:
    def test_concatenates_all_csvs(self, multiseed_dir):
        """AC-LOAD-1/2: every row present, 7-col schema, seed integer-typed."""
        df = load_multiseed(multiseed_dir)
        assert df.columns == SCHEMA
        assert df.height == 8  # 5 (h5) + 3 (h10)
        assert df["seed"].dtype in (pl.Int64, pl.Int32)
        assert df["horizon"].dtype in (pl.Int64, pl.Int32)

    def test_empty_dir_raises(self, tmp_path):
        """AC-LOAD-3: no matching CSV -> ValueError."""
        with pytest.raises(ValueError, match="no CSV"):
            load_multiseed(tmp_path)

    def test_missing_seed_column_raises(self, tmp_path):
        """AC-LOAD-4: a results CSV without the seed column -> ValueError."""
        pl.DataFrame(
            {
                "corridor": ["E2"], "direction": ["aggregate"], "baseline": ["LSTM"],
                "metric": ["MAE"], "value": [4.0], "horizon": [5],
            }
        ).write_csv(tmp_path / "lstm_multiseed_h5.csv")
        with pytest.raises(ValueError, match="schema"):
            load_multiseed(tmp_path)

    def test_pattern_skips_foreign_csv(self, tmp_path):
        """AC-LOAD-5: the default pattern selects only *_multiseed_* files."""
        _write_csv(
            tmp_path / "lstm_multiseed_h5.csv",
            [("E2", "aggregate", "LSTM", "MAE", 4.0, 5, 42)],
        )
        # A single-run results file shares the dir but lacks the seed column.
        pl.DataFrame(
            {
                "corridor": ["E2"], "direction": ["aggregate"], "baseline": ["LSTM"],
                "metric": ["MAE"], "value": [4.0], "horizon": [5],
            }
        ).write_csv(tmp_path / "lstm_results_h5.csv")
        df = load_multiseed(tmp_path)  # default pattern skips the results file
        assert df.height == 1
        assert df["seed"].to_list() == [42]


# --- AC-SUM ----------------------------------------------------------------
class TestMultiseedSummary:
    def test_groups_and_counts_seeds(self, multiseed_dir):
        """AC-SUM-1: one row per (corridor,direction,baseline,metric,horizon)."""
        summary = multiseed_summary(load_multiseed(multiseed_dir))
        assert summary.height == 2  # h5 group + h10 group
        h5 = summary.filter(pl.col("horizon") == 5)
        assert h5["n_seeds"].item() == 5

    def test_mean_std_ci_match_scipy(self, multiseed_dir):
        """AC-SUM-2/3/4: mean, sample std, and t-interval match a numpy/scipy ref."""
        summary = multiseed_summary(load_multiseed(multiseed_dir), confidence=0.95)
        h5 = summary.filter(pl.col("horizon") == 5)

        vals = np.array([4.0, 4.2, 4.4, 4.6, 4.8])
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1)  # sample std
        ci_half = stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)

        assert h5["mean"].item() == pytest.approx(mean)
        assert h5["std"].item() == pytest.approx(std)
        assert h5["ci_half"].item() == pytest.approx(ci_half)
        assert h5["ci_low"].item() == pytest.approx(mean - ci_half)
        assert h5["ci_high"].item() == pytest.approx(mean + ci_half)
        assert h5["cv_pct"].item() == pytest.approx(100 * std / mean)

    def test_higher_confidence_widens_interval(self, multiseed_dir):
        """AC-SUM-5: the 99% interval is strictly wider than the 95% one."""
        df = load_multiseed(multiseed_dir)
        ci95 = multiseed_summary(df, confidence=0.95).filter(pl.col("horizon") == 5)
        ci99 = multiseed_summary(df, confidence=0.99).filter(pl.col("horizon") == 5)
        assert ci99["ci_half"].item() > ci95["ci_half"].item()

    def test_single_seed_group_is_graceful(self, tmp_path):
        """AC-SUM-6: n=1 cannot estimate variance -> std and ci_half are 0, no NaN."""
        _write_csv(
            tmp_path / "lstm_multiseed_h1.csv",
            [("E2", "aggregate", "LSTM", "MAE", 4.0, 1, 42)],
        )
        summary = multiseed_summary(load_multiseed(tmp_path))
        assert summary["n_seeds"].item() == 1
        assert summary["std"].item() == 0.0
        assert summary["ci_half"].item() == 0.0
        assert summary["mean"].item() == pytest.approx(4.0)
