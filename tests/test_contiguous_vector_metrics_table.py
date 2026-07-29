"""Contracts on the committed vector-metric tables — audit pending #5.

These tests pin what this table measures, and the distinction is load-bearing.
The learners win the scalar MAE at long horizons and lose every column here at
every horizon, by margins that widen with the horizon. That single sentence
covers two findings of very different status:

* ``cv_bias`` — the forecast is less dispersed than the corridor really is.
  Real, threshold-free, and confirmed in 36 of 36 cells across three windows.
* ``bunching_f1`` — scored at a FIXED 0.5x relative cut, which is persistence's
  own optimum. It measures where the cut landed, not what the model knows;
  judged without a cut the h=10 verdict reverses in all three corridors.

An earlier version of this file conflated the two and read the whole thing as
lost information. The tests below keep the numbers and separate the readings;
the correction itself lives in ``tests/test_detection_calibrated.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_contiguous_vector_metrics import (  # noqa: E402
    MODELS,
    PROFILE_CSV,
    VECTOR_CSV,
    profile_shape,
    vector_verdict,
)

pytestmark = pytest.mark.skipif(
    not (VECTOR_CSV.exists() and PROFILE_CSV.exists()),
    reason="vector metric tables not generated — run src.build_contiguous_vector_metrics",
)


@pytest.fixture(scope="module")
def vector() -> pl.DataFrame:
    return pl.read_csv(VECTOR_CSV)


@pytest.fixture(scope="module")
def profile() -> pl.DataFrame:
    return pl.read_csv(PROFILE_CSV)


@pytest.fixture(scope="module")
def lstm(vector) -> pl.DataFrame:
    return vector.filter(pl.col("model") == "LSTM").sort(["corridor", "horizon"])


class TestCoverage:
    def test_one_row_per_model_and_cell(self, vector):
        expected = {
            (name, c, h) for name, _ in MODELS for c in CORRIDORS for h in HORIZONS
        }
        got = set(
            zip(
                vector.get_column("model"),
                vector.get_column("corridor"),
                vector.get_column("horizon"),
            )
        )
        assert got == expected

    def test_profile_covers_every_model_and_cell(self, profile):
        cells = set(
            zip(
                profile.get_column("model"),
                profile.get_column("corridor"),
                profile.get_column("horizon"),
            )
        )
        assert cells == {
            (name, c, h) for name, _ in MODELS for c in CORRIDORS for h in HORIZONS
        }

    def test_all_models_are_scored_on_the_same_cells(self, vector):
        """The comparison is only meaningful over one population."""
        per_cell = vector.group_by(["corridor", "horizon"]).agg(
            pl.col("n_cells").n_unique().alias("distinct_n"),
            pl.col("n_vectors").n_unique().alias("distinct_v"),
            pl.col("bunching_rate_true").n_unique().alias("distinct_truth"),
        )
        assert per_cell.get_column("distinct_n").to_list() == [1] * per_cell.height
        assert per_cell.get_column("distinct_v").to_list() == [1] * per_cell.height
        assert per_cell.get_column("distinct_truth").to_list() == [1] * per_cell.height


class TestPositionMatters:
    """The one vector result that SUPPORTS the framing."""

    def test_the_error_profile_is_not_flat(self, profile):
        shape = profile_shape(profile).filter(pl.col("model") == "LSTM")
        assert (shape.get_column("relative_spread") > 0.1).all()

    def test_every_cell_has_several_positions(self, profile):
        shape = profile_shape(profile)
        assert shape.get_column("n_positions").min() >= 8


class TestRegularityIsLost:
    def test_the_learners_always_predict_a_smoother_corridor(self, vector):
        learners = vector.filter(pl.col("model") != "Persistence")
        assert (learners.get_column("cv_bias") < 0).all()

    def test_the_flattening_worsens_with_the_horizon(self, lstm):
        for corridor, sub in lstm.sort("horizon").group_by(
            "corridor", maintain_order=True
        ):
            bias = sub.get_column("cv_bias").to_numpy()
            assert np.all(np.diff(bias) < 0), f"{corridor}: {bias}"

    def test_persistence_reproduces_the_irregularity(self, vector):
        """It propagates the observed vector, so its CV bias is ~0 by nature.
        That is not a trick — it is the property the learners lack."""
        persistence = vector.filter(pl.col("model") == "Persistence")
        assert persistence.get_column("cv_bias").abs().max() < 0.02

    def test_persistence_wins_regularity_in_every_cell(self, vector):
        verdict = vector_verdict(vector)
        assert set(verdict.get_column("best_regularity")) == {"Persistence"}


class TestBunchingIsNotFlaggedAtTheFixedCut:
    def test_bunching_is_common_enough_to_matter(self, vector):
        """Between one in six and one in three cells. Not a rare-event problem."""
        rates = vector.get_column("bunching_rate_true")
        assert rates.min() > 0.15
        assert rates.max() < 0.35

    def test_the_learners_almost_never_fire(self, lstm):
        assert (
            lstm.get_column("bunching_rate_pred")
            < lstm.get_column("bunching_rate_true") / 2
        ).all()

    def test_recall_collapses_as_the_horizon_grows(self, lstm):
        for corridor, sub in lstm.sort("horizon").group_by(
            "corridor", maintain_order=True
        ):
            recall = sub.get_column("bunching_recall").to_numpy()
            assert np.all(np.diff(recall) < 0), f"{corridor}: {recall}"
        assert lstm.filter(pl.col("horizon") == 10).get_column(
            "bunching_recall"
        ).max() < 0.02

    def test_the_learner_is_conservative_rather_than_wrong(self, lstm):
        """Precision holds up; it is recall that dies. The distinction matters —
        this is mean-regression, not noise."""
        fires = lstm.filter(pl.col("bunching_tp") + pl.col("bunching_fp") > 100)
        assert (fires.get_column("bunching_precision") > 0.45).all()

    def test_persistence_wins_bunching_f1_in_every_cell_at_the_fixed_cut(
        self, vector
    ):
        """Reproducible, and NOT a statement about detection ability.

        ``bunching_f1`` in this table is scored at the fixed 0.5x cut, which is
        persistence's own optimum — refit freely on held-out data it returns to
        0.5x in 11 of 12 cells. Judged without a cut the verdict reverses at
        h=10 in all three corridors. See ``tests/test_detection_calibrated.py``;
        this assertion exists so the artifact stays reproducible, not so the
        original reading stays true.
        """
        verdict = vector_verdict(vector)
        assert set(verdict.get_column("best_bunching")) == {"Persistence"}


class TestTheThresholdArtifact:
    """What the fixed-cut columns of this table can and cannot support.

    The original version of this class was called ``TestTheDissociation`` and
    read these monotone ratios as scalar MAE and vector fidelity moving in
    opposite directions. Half of that survives: the FLATTENING is real and
    MAE-invisible (``cv_bias``, negative in 36 of 36 cells across three
    windows). What does not survive is reading the F1 collapse as lost
    information — it is the same compression seen through a cut calibrated in
    observation space, and the monotonicity below is the evidence FOR that
    reading, not against it.
    """

    def test_the_f1_gap_widens_monotonically_with_the_horizon(self, vector):
        """Predicted by the units explanation: a longer horizon compresses the
        forecast further, so a fixed relative cut sits deeper in its tail. The
        monotonicity is what makes the explanation testable rather than post hoc.
        """
        verdict = vector_verdict(vector)
        for corridor, sub in verdict.sort("horizon").group_by(
            "corridor", maintain_order=True
        ):
            ratio = sub.get_column("f1_ratio_over_lstm").to_numpy()
            assert np.all(np.diff(ratio) > 0), f"{corridor}: {ratio}"

    def test_at_the_longest_horizon_the_artifact_is_an_order_of_magnitude(
        self, vector
    ):
        verdict = vector_verdict(vector).filter(pl.col("horizon") == 10)
        assert verdict.get_column("f1_ratio_over_lstm").min() > 5.0

    def test_no_learner_wins_either_fixed_cut_vector_metric(self, vector):
        """Both fixed-cut columns go to persistence everywhere. Regularity
        fidelity (``cv_bias``) is a REAL loss and stands; bunching F1 at this cut
        is the artifact. They are pinned together here because they come off the
        same table, and the document must keep them apart."""
        verdict = vector_verdict(vector)
        assert not verdict.get_column("best_regularity").is_in(["LSTM", "XGBoost"]).any()
        assert not verdict.get_column("best_bunching").is_in(["LSTM", "XGBoost"]).any()
