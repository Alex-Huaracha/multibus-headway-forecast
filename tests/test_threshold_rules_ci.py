"""Does each threshold rule's verdict survive its own interval?

Table 3 counts the cells where a thresholded rule names the same winner as the
comparison with no threshold. A winner is only a winner when the difference
excludes zero, so the count needs an interval for every rule, not only for the
two that ``detection_ranking_ci.csv`` already bounds.

The rule in minutes has no row in ``threshold_denominators.csv``, so this table
also carries the point values Table 3 prints for every rule, and the tests pin
them to the two CSVs that already score those rules.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_threshold_denominators import OUT_CSV as RULES_CSV  # noqa: E402
from src.build_threshold_robustness import OUT_ABSOLUTE  # noqa: E402
from src.build_threshold_rules_ci import (  # noqa: E402
    ABSOLUTE_RATIO,
    OUT_CSV,
    RULES_CI,
    build,
)

N_CELLS = len(CORRIDORS) * len(HORIZONS)


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV.name} not built yet")
    return pl.read_csv(OUT_CSV)


class TestShape:
    def test_the_rules_are_the_three_denominators_and_the_rule_in_minutes(self):
        assert RULES_CI == ("pred_mean", "obs_mean", "rank", "absolute")

    def test_one_row_per_rule_and_cell(self, table):
        assert table.height == len(RULES_CI) * N_CELLS
        assert table.unique(subset=["rule", "corridor", "horizon"]).height == (
            len(RULES_CI) * N_CELLS
        )

    def test_every_interval_brackets_its_point(self, table):
        for row in table.iter_rows(named=True):
            assert row["ci_low"] <= row["delta_mcc"] <= row["ci_high"], row

    def test_survives_means_the_interval_excludes_zero(self, table):
        for row in table.iter_rows(named=True):
            excludes = row["ci_low"] > 0 or row["ci_high"] < 0
            assert row["survives"] == excludes, row

    def test_delta_is_learner_minus_rival(self, table):
        for row in table.iter_rows(named=True):
            assert abs(row["delta_mcc"] - (row["mcc_LSTM"] - row["mcc_Persistence"])) < 1e-9


class TestThePointValuesAreTheScoredOnes:
    """Each rule's point values match the CSV that already scores it."""

    def test_the_denominator_rules_match_their_csv(self, table):
        rules = pl.read_csv(RULES_CSV)
        for rule in ("pred_mean", "obs_mean", "rank"):
            for model in ("LSTM", "Persistence"):
                theirs = rules.filter(
                    (pl.col("rule") == rule) & (pl.col("model") == model)
                ).select("corridor", "horizon", "mcc", "rate_ratio",
                         "jaccard_truth_vs_published")
                ours = table.filter(pl.col("rule") == rule).select(
                    "corridor", "horizon",
                    pl.col(f"mcc_{model}"), pl.col(f"ae_{model}"), "jaccard",
                )
                joined = ours.join(theirs, on=["corridor", "horizon"], how="inner")
                assert joined.height == N_CELLS
                for row in joined.iter_rows(named=True):
                    assert abs(row[f"mcc_{model}"] - row["mcc"]) < 1e-9
                    assert abs(row[f"ae_{model}"] - row["rate_ratio"]) < 1e-9
                    assert abs(row["jaccard"] - row["jaccard_truth_vs_published"]) < 1e-9

    def test_the_rule_in_minutes_matches_its_csv(self, table):
        theirs = pl.read_csv(OUT_ABSOLUTE).filter(
            pl.col("absolute_ratio") == ABSOLUTE_RATIO
        )
        ours = table.filter(pl.col("rule") == "absolute")
        for model in ("LSTM", "Persistence"):
            joined = ours.join(
                theirs.filter(pl.col("model") == model),
                on=["corridor", "horizon"], how="inner",
            )
            assert joined.height == N_CELLS
            for row in joined.iter_rows(named=True):
                assert abs(row[f"mcc_{model}"] - row["mcc_absolute"]) < 1e-9
                ratio = row["fire_rate_absolute"] / row["base_rate_absolute"]
                assert abs(row[f"ae_{model}"] - ratio) < 1e-9

    def test_the_rule_in_minutes_uses_the_papers_quarter(self):
        assert ABSOLUTE_RATIO == 0.25


class TestBuildIsDeterministic:
    def test_two_builds_agree(self):
        assert build().equals(build())
