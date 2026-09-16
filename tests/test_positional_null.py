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
        assert "0,579" in paper
