import numpy as np
import pytest

from src.evaluation.exante_terciles import (
    TercileThresholds,
    assign_terciles,
    compute_frozen_thresholds,
)


def test_thresholds_use_calibration_values_only_when_test_has_extremes() -> None:
    train_and_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    test_extremes = np.array([-1_000.0, 1_000.0])

    thresholds = compute_frozen_thresholds(train_and_val)

    assert thresholds.p33 == pytest.approx(np.percentile(train_and_val, 100 / 3))
    assert thresholds.p66 == pytest.approx(np.percentile(train_and_val, 200 / 3))
    assert thresholds.p33 != pytest.approx(
        np.percentile(np.concatenate([train_and_val, test_extremes]), 100 / 3)
    )
    assert thresholds.calib_split == "train+val"
    assert thresholds.calib_n == len(train_and_val)


def test_thresholds_match_train_and_val_percentiles_for_a_different_distribution() -> None:
    calibration = np.array([0.5, 1.5, 10.0, 20.0, 100.0, 200.0])

    thresholds = compute_frozen_thresholds(calibration)

    assert thresholds.p33 == pytest.approx(np.percentile(calibration, 100 / 3))
    assert thresholds.p66 == pytest.approx(np.percentile(calibration, 200 / 3))


def test_thresholds_exclude_nan_values_before_calibration() -> None:
    calibration = np.array([np.nan, 1.0, 2.0, np.nan, 4.0, 8.0])

    thresholds = compute_frozen_thresholds(calibration)

    assert thresholds.p33 == pytest.approx(np.percentile([1.0, 2.0, 4.0, 8.0], 100 / 3))
    assert thresholds.p66 == pytest.approx(np.percentile([1.0, 2.0, 4.0, 8.0], 200 / 3))
    assert np.isfinite(thresholds.p33)
    assert np.isfinite(thresholds.p66)
    assert thresholds.calib_n == 4


def test_thresholds_reject_exact_two_finite_values_plus_nan() -> None:
    calibration = np.array([1.0, 2.0, np.nan])

    with pytest.raises(ValueError, match="at least three finite calibration values"):
        compute_frozen_thresholds(calibration)


@pytest.mark.parametrize(
    "calibration",
    [np.array([np.nan, np.nan]), np.array([1.0, np.nan])],
)
def test_thresholds_reject_calibration_without_three_finite_values(
    calibration: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="at least three finite calibration values"):
        compute_frozen_thresholds(calibration)


def test_assign_terciles_uses_frozen_thresholds_and_excludes_nan_values() -> None:
    thresholds = TercileThresholds(p33=2.0, p66=4.0, calib_split="train+val", calib_n=6)

    codes = assign_terciles(np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]), thresholds)

    assert codes.tolist() == [0, 0, 1, 1, 2]


def test_assign_terciles_classifies_test_value_as_high_against_frozen_thresholds() -> None:
    thresholds = TercileThresholds(p33=2.0, p66=4.0, calib_split="train+val", calib_n=6)

    codes = assign_terciles(np.array([4.1, 100.0]), thresholds)

    assert codes.tolist() == [2, 2]


def test_thresholds_exclude_positive_and_negative_infinity_from_calibration() -> None:
    calibration = np.array([1.0, 2.0, 3.0, np.inf, -np.inf])

    thresholds = compute_frozen_thresholds(calibration)

    assert thresholds.p33 == pytest.approx(np.percentile([1.0, 2.0, 3.0], 100 / 3))
    assert thresholds.p66 == pytest.approx(np.percentile([1.0, 2.0, 3.0], 200 / 3))
    assert thresholds.calib_n == 3


def test_assign_terciles_excludes_positive_and_negative_infinity() -> None:
    thresholds = TercileThresholds(p33=2.0, p66=4.0, calib_split="train+val", calib_n=6)

    codes = assign_terciles(np.array([-np.inf, 1.0, 3.0, 5.0, np.inf]), thresholds)

    assert codes.tolist() == [0, 1, 2]
