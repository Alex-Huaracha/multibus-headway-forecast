"""The detection verdict must survive the removal of its own threshold.

Section 5 previously reported that persistence out-detects the learner by up to
253x. That number is real and reproducible, and it is an artifact of scoring a
compressed forecast with a cut calibrated on observations. This file is the
regression guard for the corrected reading, and it is written so that the
correction cannot silently rot back into the original claim:

``TestTheFixedThresholdIsPersistencesOwn``
    The mechanism. If fitting the cut on held-out data did NOT recover ~0.5x for
    persistence, then the published threshold was not "persistence's units" and
    the whole explanation collapses. This is the test that would fail first.

``TestThresholdFreeVerdict``
    The correction. AUC and out-of-sample MCC must hand the learner the win at
    h=10 in all three corridors. A reversal here is a real result and belongs in
    the document, not in a passing test.

``TestTheOldVerdictWasBelowTrivial``
    Why F1 was the wrong summary: the previously declared winner loses to a
    constant "always bunched" rule at h=10 everywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_detection_calibrated import (  # noqa: E402
    CALIBRATION_ORIGIN,
    MODELS,
    OUT_CSV,
    SCORING_ORIGIN,
    verdicts,
)
from src.evaluation.vector_metrics import (  # noqa: E402
    BUNCHING_RATIO,
    best_threshold,
    matthews_corrcoef,
    ranking_scores,
    trivial_f1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV.name} not built yet")
    return pl.read_csv(OUT_CSV)


@pytest.fixture(scope="module")
def summary(table) -> pl.DataFrame:
    return verdicts(table)


class TestTheCalibrationIsOutOfSample:
    def test_the_fit_window_is_not_the_scoring_window(self):
        """The one property that makes ``f1_calibrated`` reportable at all."""
        assert CALIBRATION_ORIGIN != SCORING_ORIGIN

    def test_the_fit_window_is_earlier(self):
        """Calibrating on a LATER window is not a deployment a transit operator
        could run. ``main`` is February; ``r1``/``r2`` precede it."""
        assert CALIBRATION_ORIGIN in {"r1", "r2"}
        assert SCORING_ORIGIN == "main"


class TestTableShape:
    def test_one_row_per_cell_and_model(self, table):
        expected = len(CORRIDORS) * len(HORIZONS) * len(MODELS)
        assert table.height == expected
        assert table.unique(
            subset=["corridor", "horizon", "model"]
        ).height == expected

    def test_base_rate_is_shared_by_both_models_in_a_cell(self, table):
        """Both models are scored against the SAME realized events. A per-model
        base rate would mean the two rows are not comparable at all."""
        for (corridor, horizon), cell in table.group_by(
            ["corridor", "horizon"], maintain_order=True
        ):
            rates = cell.get_column("base_rate").unique()
            assert rates.len() == 1, f"{corridor} h={horizon}: {rates.to_list()}"

    def test_bunching_is_not_a_rare_event(self, table):
        """Rare-event framing would justify F1; a 17-30% base rate does not."""
        assert table.get_column("base_rate").min() > 0.10


class TestTheFixedThresholdIsPersistencesOwn:
    """The mechanism, and the test that fails first if it is wrong.

    The published rule fires below ``BUNCHING_RATIO`` of the vector mean. Fitted
    freely on held-out data, persistence lands back on that same cut — because
    persistence propagates an observed vector, so the rule was written in its
    units. The learner's fitted cut has to move, and it moves in one direction.
    """

    def test_persistence_recovers_the_published_cut(self, table):
        fitted = table.filter(pl.col("model") == "Persistence").get_column(
            "threshold_fitted"
        )
        near = ((fitted + BUNCHING_RATIO).abs() < 0.10).sum()
        assert near >= 11, (
            "the published threshold is not persistence's own optimum — the "
            f"units explanation does not hold: {fitted.to_list()}"
        )

    def test_the_learner_needs_a_stricter_relative_cut(self, table):
        """Its vector is compressed, so the same relative cut sits further into
        its tail; the fitted cut must be LOOSER in ratio terms (closer to the
        mean) than 0.5x, i.e. more negative in score space."""
        fitted = table.filter(pl.col("model") == "LSTM").get_column(
            "threshold_fitted"
        )
        assert (fitted < -BUNCHING_RATIO).all(), fitted.to_list()

    def test_the_fixed_cut_silences_the_learner_and_not_persistence(self, table):
        """At the published cut the learner essentially never fires while
        persistence fires at the base rate. That asymmetry IS the 253x."""
        for (corridor, horizon), cell in table.group_by(
            ["corridor", "horizon"], maintain_order=True
        ):
            rows = {r["model"]: r for r in cell.iter_rows(named=True)}
            lstm, persist = rows["LSTM"], rows["Persistence"]
            assert lstm["fire_rate_fixed"] < lstm["base_rate"], (
                f"{corridor} h={horizon}: learner does not under-fire"
            )
            assert abs(persist["fire_rate_fixed"] - persist["base_rate"]) < 0.02, (
                f"{corridor} h={horizon}: persistence does not fire at the base rate"
            )

    def test_calibration_restores_the_learner_to_the_base_rate(self, table):
        """A mis-set threshold, not missing information: give the learner its own
        units and it fires roughly as often as the event occurs."""
        lstm = table.filter(pl.col("model") == "LSTM")
        ratio = lstm.get_column("fire_rate_calibrated") / lstm.get_column("base_rate")
        assert (ratio > 0.7).all() and (ratio < 2.0).all(), ratio.to_list()


class TestThresholdFreeVerdict:
    """The correction. Written so a reversal fails rather than averages away."""

    def test_the_learner_carries_information_everywhere(self, table):
        """Blindness would mean chance-level ranking. Nothing here is at chance."""
        lstm = table.filter(pl.col("model") == "LSTM")
        assert (lstm.get_column("auc") > 0.55).all(), lstm.get_column("auc").to_list()
        assert (lstm.get_column("ap_lift") > 1.1).all()

    def test_the_learner_out_discriminates_persistence_at_h10(self, summary):
        """Threshold-free, in all three corridors. This is the sign flip against
        the previously published verdict."""
        at_10 = summary.filter(pl.col("horizon") == 10)
        assert at_10.height == len(CORRIDORS)
        assert (at_10.get_column("winner_auc") == "LSTM").all(), at_10
        assert (at_10.get_column("winner_mcc_cal") == "LSTM").all(), at_10

    def test_persistence_still_wins_the_short_horizon(self, summary):
        """The correction is not "the learner wins everything". At h=1
        persistence genuinely discriminates better, and it also wins MAE there —
        the two metrics AGREE once the threshold artifact is removed."""
        at_1 = summary.filter(pl.col("horizon") == 1)
        assert at_1.height == len(CORRIDORS)
        assert (at_1.get_column("winner_auc") == "PERSIST").all(), at_1

    def test_the_fixed_threshold_verdict_is_unanimous_and_wrong(self, summary):
        """All 12 cells go to persistence under the published cut, including the
        h=10 cells that reverse threshold-free. Unanimity under one operating
        point and reversal under another is the definition of an artifact."""
        assert (summary.get_column("winner_fixed") == "PERSIST").all()


class TestTheOldVerdictWasBelowTrivial:
    def test_persistence_loses_to_a_constant_at_h10(self, summary):
        """Flagging EVERY cell beats the declared winner at h=10 in all three
        corridors. A metric that ranks a constant above both models cannot be
        the metric that settles which model detects bunching."""
        at_10 = summary.filter(pl.col("horizon") == 10)
        assert not at_10.get_column("persist_beats_trivial").any(), at_10

    def test_the_constant_rule_has_no_discriminative_content(self, table):
        """The reason F1 was the wrong summary, stated as arithmetic: MCC of the
        always-fire rule is exactly 0 while its F1 is a respectable 0.30-0.46."""
        for row in table.iter_rows(named=True):
            b = row["base_rate"]
            assert row["trivial_f1"] == pytest.approx(2 * b / (1 + b))
            assert row["trivial_f1"] > 0.29

    def test_f1_fitting_degenerates_where_the_base_rate_is_high(self, table):
        """Recorded as evidence for choosing MCC as the fitting objective: at E2
        (30% base rate) the F1-optimal cut collapses to flagging nearly
        everything, which is why the reported calibration does not use it."""
        e2 = table.filter((pl.col("corridor") == "E2") & (pl.col("horizon") >= 3))
        assert e2.get_column("fire_rate_f1fit").max() > 0.95, (
            "F1 fitting no longer degenerates; the justification for fitting on "
            "MCC needs to be rewritten rather than left asserted"
        )


class TestTheHelpersAreCorrect:
    """Unit-level: the metrics carrying the correction must be right on inputs
    whose answers are known by hand, not only self-consistent on real data."""

    def test_auc_of_a_constant_score_is_exactly_one_half(self):
        truth = np.array([True, False, True, False])
        assert ranking_scores(truth, np.zeros(4))["auc"] == pytest.approx(0.5)

    def test_auc_of_a_perfect_ranking_is_one(self):
        truth = np.array([True, True, False, False])
        assert ranking_scores(truth, np.array([2.0, 1.0, 0.0, -1.0]))[
            "auc"
        ] == pytest.approx(1.0)

    def test_auc_is_invariant_to_monotone_rescaling(self):
        """The property that makes AUC the right instrument here: compressing a
        forecast toward its mean cannot change it, while a fixed cut breaks."""
        rng = np.random.default_rng(0)
        score = rng.normal(size=500)
        truth = score + rng.normal(scale=0.5, size=500) > 0
        wide = ranking_scores(truth, score)["auc"]
        compressed = ranking_scores(truth, score * 0.1 - 3.0)["auc"]
        assert wide == pytest.approx(compressed)

    def test_mcc_of_the_always_fire_rule_is_zero(self):
        truth = np.array([True, False, False, True, False])
        assert matthews_corrcoef(truth, np.ones(5, dtype=bool)) == 0.0

    def test_mcc_of_a_perfect_classifier_is_one(self):
        truth = np.array([True, False, True, False])
        assert matthews_corrcoef(truth, truth) == pytest.approx(1.0)

    def test_trivial_f1_matches_the_closed_form(self):
        assert trivial_f1(0.30) == pytest.approx(2 * 0.30 / 1.30)
        assert trivial_f1(0.0) == 0.0

    def test_ap_lift_of_a_useless_score_is_about_one(self):
        rng = np.random.default_rng(1)
        truth = rng.random(4000) < 0.25
        lift = ranking_scores(truth, rng.normal(size=4000))["ap_lift"]
        assert 0.9 < lift < 1.1

    def test_best_threshold_is_applicable_as_a_ge_cut(self):
        """A cut taken from inside a run of equal scores would not reproduce the
        confusion matrix that chose it. Ties are the normal case for a
        compressed forecast, so this is the failure mode that matters."""
        score = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        truth = np.array([True, True, True, False, False, False])
        cut = best_threshold(truth, score)
        assert matthews_corrcoef(truth, score >= cut) == pytest.approx(1.0)

    def test_best_threshold_rejects_an_unknown_objective(self):
        with pytest.raises(ValueError, match="unknown objective"):
            best_threshold(np.array([True, False]), np.array([1.0, 0.0]), "accuracy")

    def test_mcc_fitting_refuses_the_degenerate_cut_that_f1_accepts(self):
        """The concrete reason ``objective='mcc'`` is the default: with a 40%
        base rate and a weak score, maximising F1 buys recall at any price."""
        rng = np.random.default_rng(2)
        n = 4000
        latent = rng.normal(size=n)
        truth = latent + rng.normal(scale=1.5, size=n) > 0.25
        score = latent
        fired_f1 = (score >= best_threshold(truth, score, "f1")).mean()
        fired_mcc = (score >= best_threshold(truth, score, "mcc")).mean()
        assert fired_f1 > fired_mcc, (
            f"F1 fitting did not over-fire: {fired_f1:.3f} vs {fired_mcc:.3f}"
        )


class TestTheDocumentQuotesTheTable:
    """Every number narrated in Section 5.3 must come off this CSV."""

    DOC = REPO_ROOT / "docs" / "resultados" / "documento-resultados.md"

    @pytest.fixture(scope="class")
    def doc(self) -> str:
        return self.DOC.read_text(encoding="utf-8")

    def test_the_document_reports_the_threshold_free_verdict(self, doc, summary):
        n = int((summary.get_column("winner_auc") == "LSTM").sum())
        assert f"{n} de las 12" in doc, (
            f"the learner wins AUC in {n} of 12 cells; the document does not "
            "state that count"
        )

    def test_the_document_no_longer_claims_a_detection_sweep(self, doc):
        """The retracted sentence. It asserted persistence wins all 12 cells on
        both vector metrics, which is only true at one arbitrary cut."""
        assert "gana las 12 combinaciones de corredor y horizonte en las dos" \
            not in doc
        assert "La persistencia gana **las 12 celdas**." not in doc

    def test_the_document_keeps_the_artifact_number(self, doc):
        """253x is not deleted — it is the artifact being explained, and hiding
        it would be its own dishonesty."""
        assert "253" in doc
