"""Which property of an event rule the compression reaches, and which it cannot.

Section V-C reports that the rule of Section III-C, carried without change onto
the forecast, stops firing. Section V-F closes one escape route: an absolute cut
in minutes does not rescue it either. Neither answers the question underneath
both — WHICH property of a rule decides whether the compression reaches it.

The builder under test runs the same event under three denominators, each one
applied identically to the observed vector and to the forecast:

``pred_mean``
    Half the mean of the vector being thresholded — the published rule. The
    denominator moves with whatever is scored.

``obs_mean``
    Half the mean of the vector observed at the forecast origin. Recomputed at
    every instant, so it still follows the corridor; identical on both sides, so
    it does not follow the forecast. This arm isolates self-reference.

``rank``
    The shortest positions of each vector, as many as the published rule marks
    on average. No denominator: the cut is an order statistic.

The four tests that carry the finding:

``TestTheLevelRulesCollapse``
    Both rules that name a LEVEL collapse for the learner and neither collapses
    for persistence. Removing the self-reference roughly doubles the firing and
    leaves it collapsed, so self-reference was never the mechanism.

``TestTheRankRulePreservesTheRate``
    Stated and tested as algebra. This arm is the control that isolates level
    against rank; it is not offered as a measurement.

``TestTheRankRuleRecoversTheThresholdFreeVerdict``
    The payoff. The same residuals give a verdict with no threshold at all, and
    only the rank rule reproduces it.

``TestRatePreservationIsNotDetection``
    The limit. Firing at the right volume is not firing at the right cells, and
    the rule that cannot collapse still loses to persistence in most cells.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_threshold_denominators import (  # noqa: E402
    MODELS,
    OUT_CSV,
    RULES,
    SCORING_ORIGIN,
    build,
    rank_rule_flags,
    threshold_free_agreement,
)
from src.evaluation.vector_metrics import VECTOR_KEY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
    / "contiguous_detection_calibrated.csv"
)
N_CELLS = len(CORRIDORS) * len(HORIZONS)


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV.name} not built yet")
    return pl.read_csv(OUT_CSV)


def _lstm(frame: pl.DataFrame, rule: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col("rule") == rule) & (pl.col("model") == "LSTM")
    ).sort(["corridor", "horizon"])


def _persistence(frame: pl.DataFrame, rule: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col("rule") == rule) & (pl.col("model") == "Persistence")
    ).sort(["corridor", "horizon"])


def _toy(values: list[float], lengths: list[int]) -> pl.DataFrame:
    """One cell of three-position vectors, in the shape the builder consumes."""
    return pl.DataFrame(
        {
            "corridor": ["E2"] * len(values),
            "direction": [1] * len(values),
            "horizon": [1] * len(values),
            "start_ts": [i // 3 for i in range(len(values))],
            "v": values,
            "vector_len": lengths,
        }
    )


class TestRankRuleFlags:
    """The rank cut, checked on its own before it carries a finding."""

    def test_the_quota_comes_from_the_length_and_the_rate(self):
        frame = _toy([5.0, 1.0, 3.0], [3, 3, 3])
        # round(3 * 0.34) == 1 position per vector.
        assert rank_rule_flags(frame, "v", 0.34).to_list() == [False, True, False]

    def test_a_rate_too_small_for_one_position_fires_nowhere(self):
        frame = _toy([5.0, 1.0, 3.0], [3, 3, 3])
        assert rank_rule_flags(frame, "v", 0.1).to_list() == [False] * 3

    def test_it_is_invariant_to_compression_toward_the_mean(self):
        """The whole reason this arm is in the comparison.

        Shrinking every value toward its vector's mean is the transformation
        Section V-B measures. It preserves order, so a rank cut cannot notice
        it, while a cut named in minutes must.
        """
        values = [2.0, 6.0, 10.0]
        mean = sum(values) / len(values)
        compressed = [mean + 0.1 * (v - mean) for v in values]
        observed = rank_rule_flags(_toy(values, [3] * 3), "v", 0.34)
        forecast = rank_rule_flags(_toy(compressed, [3] * 3), "v", 0.34)
        assert observed.to_list() == forecast.to_list()

    def test_ties_still_fire_the_exact_quota(self):
        """A constant vector fires once, deterministically, not zero or three."""
        frame = _toy([4.0, 4.0, 4.0], [3, 3, 3])
        assert rank_rule_flags(frame, "v", 0.34).sum() == 1

    def test_each_vector_is_counted_on_its_own(self):
        frame = _toy([5.0, 1.0, 3.0, 9.0, 8.0, 7.0], [3] * 6)
        flags = rank_rule_flags(frame, "v", 0.34).to_list()
        assert flags == [False, True, False, False, False, True]

    @pytest.mark.parametrize("rate", [-0.01, 1.01])
    def test_a_rate_outside_the_unit_interval_is_refused(self, rate):
        with pytest.raises(ValueError):
            rank_rule_flags(_toy([1.0, 2.0, 3.0], [3] * 3), "v", rate)

    def test_it_groups_on_the_canonical_vector_key(self):
        """A different grouping would silently make it a cell-wide rank."""
        assert VECTOR_KEY == ["corridor", "direction", "horizon", "start_ts"]


class TestTableShape:
    def test_one_row_per_cell_rule_and_model(self, table):
        expected = N_CELLS * len(RULES) * len(MODELS)
        assert table.height == expected
        assert table.unique(
            subset=["corridor", "horizon", "rule", "model"]
        ).height == expected

    def test_it_is_scored_where_the_rest_of_the_verdicts_are(self):
        """A different origin would make these columns incomparable with §V-C."""
        assert SCORING_ORIGIN == "main"

    def test_the_published_rule_is_the_first_one(self):
        """The table reads as a ladder away from the published rule."""
        assert RULES[0] == "pred_mean"


class TestItReproducesThePublishedRule:
    """The ``pred_mean`` arm must be the table Section V-C already reports.

    Without this the comparison could be running on a differently filtered
    population and crediting the difference to the denominator.
    """

    @pytest.fixture(scope="class")
    def calibrated(self) -> pl.DataFrame:
        if not CALIBRATED_CSV.exists():
            pytest.skip(f"{CALIBRATED_CSV.name} not built yet")
        return pl.read_csv(CALIBRATED_CSV)

    @pytest.mark.parametrize(
        "here,published",
        [("base_rate", "base_rate"), ("fire_rate", "fire_rate_fixed"),
         ("f1", "f1_fixed"), ("mcc", "mcc_fixed"), ("auc_published", "auc")],
    )
    def test_the_column_matches(self, table, calibrated, here, published):
        joined = _lstm(table, "pred_mean").join(
            calibrated.filter(pl.col("model") == "LSTM"),
            on=["corridor", "horizon"],
            how="inner",
            suffix="_ref",
        )
        assert joined.height == N_CELLS
        reference = published if published != here else f"{published}_ref"
        for row in joined.iter_rows(named=True):
            assert abs(row[here] - row[reference]) < 1e-9, (here, row)


class TestTheLevelRulesCollapse:
    """Both denominators named in minutes break, and only for the learner."""

    LEVEL_RULES = ("pred_mean", "obs_mean")

    @pytest.mark.parametrize("rule", LEVEL_RULES)
    def test_the_learner_underfires_in_every_cell(self, table, rule):
        ratios = _lstm(table, rule).get_column("rate_ratio").to_numpy()
        assert ratios.size == N_CELLS
        assert (ratios < 0.6).all(), ratios

    @pytest.mark.parametrize("rule", LEVEL_RULES)
    def test_persistence_does_not(self, table, rule):
        """The control. Persistence carries the observed dispersion, so a cut
        calibrated in observation space lands where it was designed to land."""
        ratios = _persistence(table, rule).get_column("rate_ratio").to_numpy()
        assert (ratios > 0.85).all(), ratios

    def test_moving_the_denominator_off_the_forecast_does_not_rescue_it(self, table):
        """The result the section is built on.

        ``obs_mean`` removes the self-reference entirely: truth and alarm are
        compared against the same observed number. If self-reference were the
        mechanism this arm would recover, and it recovers a factor of two out of
        more than ten.
        """
        published = float(_lstm(table, "pred_mean").get_column("rate_ratio").median())
        observed = float(_lstm(table, "obs_mean").get_column("rate_ratio").median())
        assert observed > published
        assert observed < 0.25

    @pytest.mark.parametrize("rule", LEVEL_RULES)
    def test_the_collapse_is_deeper_at_ten_minutes_than_at_one(self, table, rule):
        """Section V-B's compression deepens with the horizon; so does this.

        The ends rather than every step: under ``obs_mean`` the E2 series turns
        back up at ten minutes, and pinning strict monotonicity would assert
        something the corpus does not support.
        """
        for corridor in CORRIDORS:
            cell = (
                _lstm(table, rule)
                .filter(pl.col("corridor") == corridor)
                .sort("horizon")
            )
            ratios = cell.get_column("rate_ratio").to_list()
            assert ratios[0] > ratios[-1], (rule, corridor, ratios)

    def test_the_level_rules_keep_their_cut_where_it_was(self, table):
        """The mechanism, in minutes: neither cut travels with the forecast."""
        for rule in self.LEVEL_RULES:
            for row in _lstm(table, rule).iter_rows(named=True):
                assert abs(row["cut_alarm_min"] - row["cut_truth_min"]) < 0.5, row


class TestTheRankRulePreservesTheRate:
    """Stated as algebra, tested as algebra. This arm is the control."""

    def test_it_fires_exactly_as_often_as_the_event_occurs(self, table):
        cells = table.filter(pl.col("rule") == "rank")
        assert cells.height == N_CELLS * len(MODELS)
        for row in cells.iter_rows(named=True):
            assert abs(row["fire_rate"] - row["base_rate"]) < 1e-12, row
            assert abs(row["rate_ratio"] - 1.0) < 1e-12, row

    def test_its_rate_is_the_published_one_up_to_the_quota_rounding(self, table):
        """Set from the published rule's own rate, so the arms are comparable.

        It cannot match it exactly: a quota of positions is an integer and these
        vectors carry three to six of them. In E4, where the vectors are
        shortest, the rounding lifts the rate by about seven points, and the
        table carries both rates so that is visible rather than assumed.
        """
        difference = np.abs(
            _lstm(table, "rank").get_column("base_rate").to_numpy()
            - _lstm(table, "pred_mean").get_column("base_rate").to_numpy()
        )
        assert difference.max() < 0.10

    def test_it_raises_its_cut_onto_the_forecast(self, table):
        """What a rank rule does instead of holding a level."""
        for row in _lstm(table, "rank").iter_rows(named=True):
            assert row["cut_alarm_min"] > row["cut_truth_min"], row

    def test_the_cut_travels_farther_as_the_compression_deepens(self, table):
        """Section V-B's compression deepens with the horizon, so the distance
        the rank rule has to climb must grow with it."""
        cells = _lstm(table, "rank").with_columns(
            (pl.col("cut_alarm_min") - pl.col("cut_truth_min")).alias("_gap")
        )
        for corridor in CORRIDORS:
            gaps = (
                cells.filter(pl.col("corridor") == corridor)
                .sort("horizon")
                .get_column("_gap")
                .to_list()
            )
            assert gaps == sorted(gaps), (corridor, gaps)


class TestTheThreeEventsOverlapOnObservedData:
    """What makes the divergence on forecasts worth reporting at all.

    The overlap is substantial and it is not identity. The table carries it so
    the paper can report the real figure instead of claiming the three rules
    mark the same event.
    """

    def test_every_rule_overlaps_the_published_event(self, table):
        for rule in RULES:
            overlap = (
                table.filter(pl.col("rule") == rule)
                .get_column("jaccard_truth_vs_published")
                .to_numpy()
            )
            assert (overlap > 0.45).all(), (rule, overlap)

    def test_the_published_rule_agrees_with_itself(self, table):
        overlap = _lstm(table, "pred_mean").get_column(
            "jaccard_truth_vs_published"
        ).to_numpy()
        assert np.allclose(overlap, 1.0)

    def test_the_overlap_does_not_explain_the_divergence(self, table):
        """``obs_mean`` overlaps the published event MORE than ``rank`` does and
        behaves like it anyway. Whatever separates the arms, it is not how far
        each event sits from the published one."""
        closer = float(_lstm(table, "obs_mean").get_column(
            "jaccard_truth_vs_published").median())
        farther = float(_lstm(table, "rank").get_column(
            "jaccard_truth_vs_published").median())
        assert closer > farther
        assert (
            float(_lstm(table, "obs_mean").get_column("rate_ratio").median())
            < float(_lstm(table, "rank").get_column("rate_ratio").median())
        )


class TestTheRankRuleRecoversTheThresholdFreeVerdict:
    """The payoff, and the only place the three arms are ranked against truth.

    The same residuals give a verdict with no threshold at all. An event rule
    that inverts it is reporting the rule and not the corridor.
    """

    @pytest.fixture(scope="class")
    def agreement(self, table) -> pl.DataFrame:
        return threshold_free_agreement(table)

    def test_one_row_per_rule_and_cell(self, agreement):
        assert agreement.height == len(RULES) * N_CELLS

    def test_the_rank_rule_agrees_almost_everywhere(self, agreement):
        agrees = int(
            agreement.filter(pl.col("rule") == "rank").get_column("agrees").sum()
        )
        assert agrees == 11

    def test_neither_level_rule_comes_close(self, agreement):
        for rule in ("pred_mean", "obs_mean"):
            agrees = int(
                agreement.filter(pl.col("rule") == rule).get_column("agrees").sum()
            )
            assert agrees <= 7, (rule, agrees)

    def test_the_published_rule_never_lets_the_learner_win(self, table):
        """Under the published rule persistence takes all twelve cells, which
        is what made the collapse read as blindness."""
        lstm = _lstm(table, "pred_mean").get_column("mcc").to_numpy()
        persist = _persistence(table, "pred_mean").get_column("mcc").to_numpy()
        assert (lstm < persist).all()


class TestRatePreservationIsNotDetection:
    """Getting the volume right is not the same as getting the cells right."""

    def test_the_table_carries_the_quality_of_every_arm(self, table):
        for column in ("precision", "recall", "f1", "mcc", "auc_published"):
            values = table.get_column(column).to_numpy()
            assert np.isfinite(values).all(), column

    def test_the_rank_rule_beats_the_published_one_in_every_cell(self, table):
        """Whatever it costs elsewhere, it recovers discrimination the
        collapsed detector threw away."""
        assert (
            _lstm(table, "rank").get_column("mcc").to_numpy()
            > _lstm(table, "pred_mean").get_column("mcc").to_numpy()
        ).all()

    def test_it_still_loses_to_persistence_in_most_cells(self, table):
        """The honest limit. Repairing the rule does not turn the forecast into
        a better detector than the vector it failed to beat."""
        lost = int(
            (
                _lstm(table, "rank").get_column("mcc").to_numpy()
                < _persistence(table, "rank").get_column("mcc").to_numpy()
            ).sum()
        )
        assert lost == 7


class TestTheDocumentReportsTheExperiment:
    PAPER = REPO_ROOT / "docs" / "paper" / "paper.md"

    @pytest.fixture(scope="class")
    def paper(self) -> str:
        # Whitespace collapsed: these guards assert what the paper states, not
        # where its lines happen to wrap.
        return re.sub(r"\s+", " ", self.PAPER.read_text(encoding="utf-8"))

    def test_it_names_the_three_denominators(self, paper):
        """``promedio``, not ``media``: Section III-C already named that object."""
        for phrase in ("promedio del vector predicho",
                       "promedio del último vector observado",
                       "posiciones más cortas"):
            assert phrase in paper, phrase

    def test_it_prints_the_three_firing_medians(self, table, paper):
        for rule in RULES:
            median = float(_lstm(table, rule).get_column("rate_ratio").median())
            printed = f"{median:.3f}"
            assert printed in paper, (rule, printed)

    def test_it_prints_the_mcc_the_rank_rule_recovers(self, table, paper):
        for rule in ("pred_mean", "rank"):
            median = float(_lstm(table, rule).get_column("mcc").median())
            printed = f"{median:.3f}"
            assert printed in paper, (rule, printed)

    def test_it_reports_the_quota_rounding(self, table, paper):
        """The quota is an integer, so the rank rule marks a slightly more
        frequent event. Publishing the 11 of 12 without that would be selective.
        """
        published = _lstm(table, "pred_mean").get_column("base_rate").to_numpy()
        rank = _lstm(table, "rank").get_column("base_rate").to_numpy()
        printed = f"{100.0 * (rank - published).max():.1f}"
        assert printed in paper, printed

    def test_it_reports_the_limit_and_not_only_the_recovery(self, paper):
        """A section that published the 11 of 12 and withheld the eight cells
        the learner still loses would be selective."""
        assert "sigue por debajo de la persistencia" in paper


class TestBuildIsDeterministic:
    def test_two_builds_agree(self):
        assert build().equals(build())
