"""Does each threshold rule's verdict survive its own interval?

Table 3 counts the cells where a thresholded rule names the same winner as the
comparison with no threshold. A winner is only a winner when the difference
excludes zero, so the count needs an interval for every rule, not only for the
two that ``detection_ranking_ci.csv`` already bounds.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_threshold_denominators import OUT_CSV as RULES_CSV  # noqa: E402
from src.build_threshold_denominators import RULES  # noqa: E402
from src.build_threshold_rules_ci import OUT_CSV, build  # noqa: E402

N_CELLS = len(CORRIDORS) * len(HORIZONS)


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV.name} not built yet")
    return pl.read_csv(OUT_CSV)


class TestShape:
    def test_one_row_per_rule_and_cell(self, table):
        assert table.height == len(RULES) * N_CELLS
        assert table.unique(subset=["rule", "corridor", "horizon"]).height == (
            len(RULES) * N_CELLS
        )

    def test_every_interval_brackets_its_point(self, table):
        for row in table.iter_rows(named=True):
            assert row["ci_low"] <= row["delta_mcc"] <= row["ci_high"], row

    def test_survives_means_the_interval_excludes_zero(self, table):
        for row in table.iter_rows(named=True):
            excludes = row["ci_low"] > 0 or row["ci_high"] < 0
            assert row["survives"] == excludes, row


class TestTheIntervalBoundsTheTablesOwnNumber:
    """The point difference is the one Table 3's MCC column is made of."""

    def test_delta_is_learner_minus_persistence_from_the_rules_csv(self, table):
        rules = pl.read_csv(RULES_CSV).pivot(
            on="model", index=["rule", "corridor", "horizon"], values="mcc"
        )
        joined = table.join(rules, on=["rule", "corridor", "horizon"], how="inner")
        assert joined.height == table.height
        for row in joined.iter_rows(named=True):
            assert abs(row["delta_mcc"] - (row["LSTM"] - row["Persistence"])) < 1e-9


class TestBuildIsDeterministic:
    def test_two_builds_agree(self):
        assert build().equals(build())
