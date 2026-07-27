"""Tests for the vector-level metrics — audit pending #5.

The audit's charge is that grouped scalar MAE cannot tell a vector forecast from
N independent scalar forecasts. So the central test here is not that the
functions compute something, but that they SEPARATE two predictions a scalar
metric cannot: identical MAE, different vector shape. If they failed that, they
would not answer the finding.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation.vector_metrics import (
    BUNCHING_RATIO,
    MIN_VECTOR_LEN,
    bunching_flags,
    detection_scores,
    error_profile,
    regularity_error,
    vector_frame,
)

BASE = "2024-02-08T08:00:00"


def _residuals(vectors: dict[str, list[float]], preds: dict[str, list[float]] | None = None):
    """One frame from ``{start_ts: [headways...]}``; predictions default to truth."""
    rows = []
    for start_ts, values in vectors.items():
        predicted = (preds or {}).get(start_ts, values)
        for rank, (true, pred) in enumerate(zip(values, predicted)):
            rows.append(
                {
                    "corridor": "E2",
                    "direction": 1,
                    "horizon": 3,
                    "start_ts": start_ts,
                    "pair_rank": rank,
                    "y_true": float(true),
                    "y_pred_model": float(pred),
                }
            )
    return pl.DataFrame(rows)


class TestErrorProfile:
    def test_mae_is_reported_per_position(self):
        df = _residuals(
            {BASE: [10.0, 10.0, 10.0]}, {BASE: [10.0, 12.0, 7.0]}
        )
        profile = error_profile(df, "y_pred_model").sort("pair_rank")
        assert profile.get_column("mae").to_list() == pytest.approx([0.0, 2.0, 3.0])

    def test_positions_are_not_pooled(self):
        """Two vectors, same positions: the profile averages within a position."""
        df = _residuals(
            {BASE: [10.0, 10.0, 10.0], "2024-02-08T08:01:00": [10.0, 10.0, 10.0]},
            {BASE: [10.0, 14.0, 10.0], "2024-02-08T08:01:00": [10.0, 10.0, 10.0]},
        )
        profile = error_profile(df, "y_pred_model").sort("pair_rank")
        assert profile.get_column("mae").to_list() == pytest.approx([0.0, 2.0, 0.0])
        assert profile.get_column("n").to_list() == [2, 2, 2]


class TestVectorFrame:
    def test_cv_is_std_over_mean(self):
        values = [4.0, 8.0, 12.0]
        frame = vector_frame(_residuals({BASE: values}), ["y_true"])
        expected = np.std(values, ddof=1) / np.mean(values)
        assert frame.get_column("y_true_cv").item() == pytest.approx(expected)

    def test_short_vectors_are_dropped(self):
        short = ["2024-02-08T08:0%d:00" % i for i in range(2)]
        df = _residuals({short[0]: [5.0, 6.0], short[1]: [5.0, 6.0, 7.0]})
        frame = vector_frame(df, ["y_true"])
        assert frame.height == 1
        assert frame.get_column("vector_len").item() >= MIN_VECTOR_LEN

    def test_a_perfectly_regular_vector_has_zero_cv(self):
        frame = vector_frame(_residuals({BASE: [6.0, 6.0, 6.0]}), ["y_true"])
        assert frame.get_column("y_true_cv").item() == pytest.approx(0.0)


class TestRegularityError:
    def _paired(self, truth, predicted):
        keys = ["2024-02-08T08:%02d:00" % i for i in range(len(truth))]
        df = _residuals(
            dict(zip(keys, truth)), dict(zip(keys, predicted))
        )
        return vector_frame(df, ["y_true", "y_pred_model"])

    def test_a_flattening_model_gets_a_negative_bias(self):
        """The failure mode worth naming: predicting a smoother corridor."""
        truth = [[2.0, 6.0, 16.0], [3.0, 5.0, 20.0], [1.0, 9.0, 14.0]]
        flat = [[8.0, 8.0, 8.0], [9.0, 9.0, 9.5], [8.0, 8.0, 8.2]]
        out = regularity_error(self._paired(truth, flat), "y_pred_model")
        assert out["cv_bias"] < -0.3
        assert out["mean_cv_pred"] < out["mean_cv_true"]

    def test_a_shape_preserving_model_has_near_zero_bias(self):
        truth = [[2.0, 6.0, 16.0], [3.0, 5.0, 20.0], [1.0, 9.0, 14.0]]
        shifted = [[v + 0.1 for v in vec] for vec in truth]
        out = regularity_error(self._paired(truth, shifted), "y_pred_model")
        assert abs(out["cv_bias"]) < 0.05

    def test_too_few_vectors_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            regularity_error(self._paired([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]),
                             "y_pred_model")


class TestBunchingFlags:
    def test_flags_cells_below_the_ratio_of_their_own_vector_mean(self):
        # mean 8; ratio 0.5 -> threshold 4.
        df = _residuals({BASE: [2.0, 8.0, 14.0]})
        assert bunching_flags(df, "y_true").to_list() == [True, False, False]

    def test_the_threshold_is_relative_not_absolute(self):
        """Same absolute headway, different corridor state, different verdict."""
        busy = _residuals({BASE: [3.0, 3.0, 3.0, 3.0]})
        sparse = _residuals({BASE: [3.0, 20.0, 20.0, 20.0]})
        assert bunching_flags(busy, "y_true").to_list() == [False] * 4
        assert bunching_flags(sparse, "y_true").to_list()[0] is True

    def test_a_prediction_is_flagged_against_its_own_mean(self):
        """An operator has no access to the true mean, so neither does the flag."""
        df = _residuals({BASE: [2.0, 8.0, 14.0]}, {BASE: [7.0, 8.0, 9.0]})
        # Predicted mean 8, threshold 4: nothing predicted below it.
        assert bunching_flags(df, "y_pred_model").to_list() == [False] * 3
        assert bunching_flags(df, "y_true").to_list() == [True, False, False]

    def test_ratio_constant_is_the_one_being_applied(self):
        mean = 10.0
        just_under = BUNCHING_RATIO * mean - 0.01
        just_over = BUNCHING_RATIO * mean + 0.01
        df = _residuals({BASE: [just_under, just_over, 3 * mean - just_under - just_over]})
        assert bunching_flags(df, "y_true").to_list()[:2] == [True, False]


class TestDetectionScores:
    def test_perfect_detection(self):
        flags = np.array([True, False, True, False])
        out = detection_scores(flags, flags)
        assert (out.precision, out.recall, out.f1) == (1.0, 1.0, 1.0)

    def test_a_model_that_never_fires_scores_zero_not_undefined(self):
        out = detection_scores(np.array([True, True, False]), np.zeros(3, dtype=bool))
        assert out.precision == 0.0
        assert out.recall == 0.0
        assert out.f1 == 0.0

    def test_conservative_and_wrong_are_distinguished(self):
        """High precision with low recall is a different failure from noise."""
        truth = np.array([True] * 10 + [False] * 10)
        conservative = np.array([True] + [False] * 19)
        noisy = np.array([False] * 10 + [True] * 10)
        assert detection_scores(truth, conservative).precision == 1.0
        assert detection_scores(truth, conservative).recall == pytest.approx(0.1)
        assert detection_scores(truth, noisy).precision == 0.0

    def test_rates_are_reported_so_the_base_rate_is_visible(self):
        truth = np.array([True] * 3 + [False] * 7)
        out = detection_scores(truth, np.zeros(10, dtype=bool))
        assert out.true_rate == pytest.approx(0.3)
        assert out.pred_rate == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            detection_scores(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))


class TestSeparatesWhatScalarMaeCannot:
    """The audit's actual charge, tested head-on.

    Two predictions with the SAME scalar MAE — one compressing the vector toward
    its mean, one stretching it away by the same amount. Every scalar metric
    scores them identically; both vector metrics must not.

    A uniform additive shift would NOT work as the control: it moves the mean
    while leaving the standard deviation alone, so it changes the coefficient of
    variation too. Mean-preserving perturbations of equal L1 are the honest
    pair, and they differ only in the direction they move the shape.
    """

    TRUTH = [[2.0, 6.0, 16.0], [4.0, 6.0, 14.0], [2.0, 8.0, 14.0]]
    # Pulls the extremes 2 min toward the mean; mean unchanged.
    COMPRESSING = [[4.0, 6.0, 14.0], [6.0, 6.0, 12.0], [4.0, 8.0, 12.0]]
    # Pushes them 2 min away; mean unchanged, same total absolute error.
    EXPANDING = [[0.0, 6.0, 18.0], [2.0, 6.0, 16.0], [0.0, 8.0, 16.0]]

    def _frames(self, predicted):
        keys = ["2024-02-08T08:%02d:00" % i for i in range(len(self.TRUTH))]
        df = _residuals(dict(zip(keys, self.TRUTH)), dict(zip(keys, predicted)))
        return df, vector_frame(df, ["y_true", "y_pred_model"])

    def _mae(self, predicted):
        df, _ = self._frames(predicted)
        return float(
            (df.get_column("y_true") - df.get_column("y_pred_model")).abs().mean()
        )

    def test_the_two_predictions_have_identical_scalar_mae(self):
        """The premise. Without it the comparisons below prove nothing."""
        assert self._mae(self.COMPRESSING) == pytest.approx(self._mae(self.EXPANDING))

    def test_they_also_have_identical_mean_headways(self):
        """So no aggregate level statistic separates them either."""
        for predicted in (self.COMPRESSING, self.EXPANDING):
            _, frame = self._frames(predicted)
            assert frame.get_column("y_pred_model_mean").to_list() == pytest.approx(
                frame.get_column("y_true_mean").to_list()
            )

    def test_regularity_gives_them_opposite_signs(self):
        _, compressing = self._frames(self.COMPRESSING)
        _, expanding = self._frames(self.EXPANDING)
        bias_compressing = regularity_error(compressing, "y_pred_model")["cv_bias"]
        bias_expanding = regularity_error(expanding, "y_pred_model")["cv_bias"]
        assert bias_compressing < 0, "compression must read as a smoother corridor"
        assert bias_expanding > 0
        assert bias_compressing < bias_expanding

    def test_bunching_detection_separates_them(self):
        compressing_df, _ = self._frames(self.COMPRESSING)
        expanding_df, _ = self._frames(self.EXPANDING)
        truth = bunching_flags(compressing_df, "y_true").to_numpy()
        assert truth.any(), "the fixture must contain bunching to detect"

        f1_compressing = detection_scores(
            truth, bunching_flags(compressing_df, "y_pred_model").to_numpy()
        ).f1
        f1_expanding = detection_scores(
            truth, bunching_flags(expanding_df, "y_pred_model").to_numpy()
        ).f1
        assert f1_compressing < f1_expanding
        assert f1_compressing == 0.0, "a compressed vector fires no bunching flag"
