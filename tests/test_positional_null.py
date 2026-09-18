"""The threshold-free verdict needs a floor, and this is it.

Section IV-D already carries a floor for F1: the trivial detector that flags
every position. The AUC had none, and that gap is exactly where a reviewer
attacks — because the event label is a headway compared against the mean of its
own vector, and the positions of a vector are not exchangeable. Positions near
the front carry systematically shorter headways, so some of them fall below half
the vector mean for reasons of POSITION rather than of anything that happened in
the input window.

The null model here exploits that and nothing else: it answers with the mean
observed headway of each position, fitted on an earlier disjoint origin, held
constant for every minute of the scored period. It cannot anticipate. If it
ranks bunching risk as well as a trained model, the trained model's ranking
advantage was positional bookkeeping.

``TestTheNullCannotAnticipate``
    The construction. If this fails the null is not a null and nothing below it
    means anything.

``TestWhereTheNullIsAtChance``
    The result that protects the paper: in E4 and E59 the null does not
    discriminate, so the learner's advantage there is not positional.

``TestTheExceptionInE2``
    The result that costs the paper its showcase cell. E2 carries real
    positional structure, and at h=10 the learner falls below it. Pinned so the
    exception cannot quietly disappear from the document.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_positional_null import (  # noqa: E402
    FIT_ORIGIN,
    OUT_CSV,
    PROFILE_KEY,
    SCORING_ORIGIN,
    positional_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV.name} not built yet")
    return pl.read_csv(OUT_CSV)


class TestTheNullCannotAnticipate:
    """Its whole evidential value rests on carrying no temporal information."""

    def test_the_profile_is_keyed_only_on_position(self):
        """No timestamp in the key means one constant per position, so the null
        answers identically at every minute of the scored period."""
        assert "start_ts" not in PROFILE_KEY
        assert "target_ts" not in PROFILE_KEY
        assert set(PROFILE_KEY) == {"corridor", "direction", "horizon", "pair_rank"}

    def test_the_profile_has_one_value_per_position(self):
        profile = positional_profile(FIT_ORIGIN)
        assert profile.unique(subset=PROFILE_KEY).height == profile.height

    def test_the_fit_window_is_earlier_and_disjoint(self):
        """Fitting on the scored period would let the floor read the answers."""
        assert FIT_ORIGIN != SCORING_ORIGIN
        assert FIT_ORIGIN in {"r1", "r2"}
        assert SCORING_ORIGIN == "main"


class TestTableShape:
    def test_one_row_per_cell(self, table):
        expected = len(CORRIDORS) * len(HORIZONS)
        assert table.height == expected
        assert table.unique(subset=["corridor", "horizon"]).height == expected

    def test_the_three_predictors_are_scored_on_the_same_events(self, table):
        """A per-predictor base rate would mean the columns are not comparable."""
        assert table.get_column("base_rate").min() > 0.10


class TestWhereTheNullIsAtChance:
    """Two of three corridors: the learner's ranking advantage is real."""

    def test_the_null_does_not_discriminate_in_e4_and_e59(self, table):
        flat = table.filter(pl.col("corridor").is_in(["E4", "E59"]))
        assert flat.height == 2 * len(HORIZONS)
        assert (flat.get_column("auc_null") < 0.55).all(), flat

    def test_the_learner_clears_the_floor_there_by_a_wide_margin(self, table):
        flat = table.filter(pl.col("corridor").is_in(["E4", "E59"]))
        gap = flat.get_column("auc_lstm") - flat.get_column("auc_null")
        assert gap.min() > 0.05, gap.to_list()


class TestTheExceptionInE2:
    """One of three corridors: position alone ranks better than the learner."""

    def test_e2_carries_positional_structure(self, table):
        e2 = table.filter(pl.col("corridor") == "E2")
        assert (e2.get_column("auc_null") > 0.55).all(), e2

    def test_the_positional_floor_does_not_move_with_the_horizon(self, table):
        """A floor that decayed like the models would be carrying information
        about the forecast task. This one does not: it is pure bookkeeping."""
        e2 = table.filter(pl.col("corridor") == "E2").get_column("auc_null")
        assert e2.max() - e2.min() < 0.02, e2.to_list()

    def test_the_learner_falls_below_the_floor_at_ten_minutes(self, table):
        """The finding the document must declare. If a rebuild reverses this,
        the paper's wording has to change with it rather than silently hold."""
        row = table.filter(
            (pl.col("corridor") == "E2") & (pl.col("horizon") == 10)
        ).row(0, named=True)
        assert row["auc_null"] > row["auc_lstm"], row

    def test_the_exception_is_confined_to_that_single_cell(self, table):
        """Scoped so an overclaim in either direction fails: the null wins once,
        not never and not often."""
        beaten = table.filter(pl.col("auc_null") > pl.col("auc_lstm"))
        assert beaten.height == 1, beaten
        assert beaten.row(0, named=True)["corridor"] == "E2"
        assert beaten.row(0, named=True)["horizon"] == 10


class TestTheFloorIsBoundedLikeEveryOtherVerdict:
    """Section IV-E defines a verdict as three parts, and the floor owed two.

    ``Un veredicto … consta de tres partes: cuál de los dos gana, por cuánto y
    si la diferencia sobrevive su prueba``. The floor was published as a bare
    point estimate, so the manuscript asserted a sign at a resolution finer than
    every interval it prints — 0,013 against a comparable band of ±0,011. These
    columns give the floor the same instrument the rest of the table uses.
    """

    BOUNDED = ("lstm", "persist")

    @pytest.mark.parametrize("model", BOUNDED)
    def test_each_comparison_against_the_floor_carries_its_interval(
        self, table, model
    ):
        for suffix in ("delta", "ci_low", "ci_high", "survives"):
            assert f"{model}_vs_null_{suffix}" in table.columns

    @pytest.mark.parametrize("model", BOUNDED)
    def test_the_interval_brackets_its_own_point_estimate(self, table, model):
        """A band that excludes the difference it bounds is not that difference."""
        for row in table.iter_rows(named=True):
            low, delta, high = (
                row[f"{model}_vs_null_ci_low"],
                row[f"{model}_vs_null_delta"],
                row[f"{model}_vs_null_ci_high"],
            )
            assert low <= delta <= high, (row["corridor"], row["horizon"], model)

    @pytest.mark.parametrize("model", BOUNDED)
    def test_the_point_estimate_is_the_difference_of_the_published_areas(
        self, table, model
    ):
        for row in table.iter_rows(named=True):
            expected = row[f"auc_{model}"] - row["auc_null"]
            assert abs(row[f"{model}_vs_null_delta"] - expected) < 1e-9, row

    @pytest.mark.parametrize("model", BOUNDED)
    def test_survival_is_the_interval_clearing_zero(self, table, model):
        for row in table.iter_rows(named=True):
            clears = not (
                row[f"{model}_vs_null_ci_low"]
                <= 0.0
                <= row[f"{model}_vs_null_ci_high"]
            )
            assert bool(row[f"{model}_vs_null_survives"]) is clears, row

    def test_the_loss_in_e2_at_ten_minutes_survives_its_interval(self, table):
        """The concession the manuscript has to make in full.

        A point estimate alone would let the cell be dismissed as noise. It is
        not: the learner sits below a rule that reads no input window, and the
        bound says so.
        """
        row = table.filter(
            (pl.col("corridor") == "E2") & (pl.col("horizon") == 10)
        ).row(0, named=True)
        assert row["lstm_vs_null_delta"] < 0, row
        assert row["lstm_vs_null_ci_high"] < 0, row
        assert row["lstm_vs_null_survives"], row

    def test_the_bootstrap_is_the_one_the_rest_of_the_paper_uses(self, table):
        """A second resampling convention would make the columns incomparable."""
        from src.build_detection_ranking_ci import LEVEL, N_BOOT, SEED

        assert (table.get_column("n_boot") == N_BOOT).all()
        assert (table.get_column("seed") == SEED).all()
        assert (table.get_column("level") == LEVEL).all()


class TestTheHeadOfTheRankingDisagreesWithTheArea:
    """Two threshold-free scores, and in the disputed cell they part ways.

    Section V-E claims the AUC and the lift agree in all twelve cells. That
    holds for the learner-persistence pair. Against the floor it fails in E2 at
    ten minutes, and the manuscript has to report the disagreement rather than
    the half of it that suits either side.
    """

    def test_the_learner_clears_the_floor_on_lift_where_it_loses_on_area(
        self, table
    ):
        row = table.filter(
            (pl.col("corridor") == "E2") & (pl.col("horizon") == 10)
        ).row(0, named=True)
        assert row["auc_lstm"] < row["auc_null"], row
        assert row["ap_lift_lstm"] > row["ap_lift_null"], row

    def test_the_floor_still_outranks_persistence_on_both_scores_there(self, table):
        """The floor indicts the winner the transplanted threshold crowned."""
        row = table.filter(
            (pl.col("corridor") == "E2") & (pl.col("horizon") == 10)
        ).row(0, named=True)
        assert row["auc_null"] > row["auc_persist"], row
        assert row["ap_lift_null"] > row["ap_lift_persist"], row


class TestTheDocumentDeclaresTheFloor:
    """A floor computed and not reported is worse than never computing it."""

    PAPER = REPO_ROOT / "docs" / "paper" / "paper.md"

    @pytest.fixture(scope="class")
    def paper(self) -> str:
        return self.PAPER.read_text(encoding="utf-8")

    def test_the_paper_names_the_positional_floor(self, paper):
        assert "perfil posicional" in paper

    def test_the_paper_declares_the_e2_exception(self, paper):
        """The sentence that costs the showcase cell. Its absence is the
        selective-reporting failure this whole file exists to prevent."""
        assert "0.579" in paper

    def test_the_paper_bounds_that_exception(self, table, paper):
        """Reporting the sign without the bound is the defect Section IV-E names."""
        row = table.filter(
            (pl.col("corridor") == "E2") & (pl.col("horizon") == 10)
        ).row(0, named=True)
        band = "{:+.3f} [{:+.3f}, {:+.3f}]".format(
            row["lstm_vs_null_delta"],
            row["lstm_vs_null_ci_low"],
            row["lstm_vs_null_ci_high"],
        )
        assert band in paper, band

    def test_the_paper_reports_the_lift_of_the_floor(self, table, paper):
        """Publishing the score that loses and withholding the one that wins is
        selective reporting, whichever direction it favours."""
        row = table.filter(
            (pl.col("corridor") == "E2") & (pl.col("horizon") == 10)
        ).row(0, named=True)
        assert f"{row['ap_lift_null']:.2f}" in paper
