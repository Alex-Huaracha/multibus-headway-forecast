"""Contracts for the Clopper-Pearson interval on detection precision.

Section V-C reports that the learner fired 14 times on E2 at ten minutes and hit
10 real events. A point precision off 14 trials is not a finding, so the paper
quotes an interval alongside it. That interval had no source of truth: this is
it.

The assertions below deliberately avoid re-implementing the beta quantile.
Clopper-Pearson has exact closed forms at the two degenerate counts, and an
exact complement symmetry, both of which pin the implementation without
restating it.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.evaluation.vector_metrics import precision_interval


def test_no_fires_has_no_precision_to_bound():
    """A detector that never fires has undefined precision, not precision 0."""
    with pytest.raises(ValueError):
        precision_interval(0, 0)


def test_hits_cannot_exceed_fires():
    with pytest.raises(ValueError):
        precision_interval(11, 10)


def test_all_fires_correct_pins_the_lower_bound_in_closed_form():
    """With tp == n the exact lower bound is (alpha/2)**(1/n) and high is 1."""
    low, high = precision_interval(14, 14, level=0.95)
    assert low == pytest.approx(0.025 ** (1 / 14), abs=1e-12)
    assert high == 1.0


def test_no_fire_correct_pins_the_upper_bound_in_closed_form():
    """With tp == 0 the exact upper bound is 1 - (alpha/2)**(1/n) and low is 0."""
    low, high = precision_interval(0, 5, level=0.95)
    assert low == 0.0
    assert high == pytest.approx(1 - 0.025 ** (1 / 5), abs=1e-12)


def test_interval_brackets_the_point_estimate():
    low, high = precision_interval(10, 14)
    assert low < 10 / 14 < high


def test_complement_symmetry():
    """Bounding hits and bounding misses are the same interval, reflected."""
    low, high = precision_interval(10, 14)
    miss_low, miss_high = precision_interval(4, 14)
    assert low == pytest.approx(1 - miss_high, abs=1e-12)
    assert high == pytest.approx(1 - miss_low, abs=1e-12)


def test_higher_confidence_widens_the_interval():
    tight = precision_interval(10, 14, level=0.90)
    loose = precision_interval(10, 14, level=0.99)
    assert loose[0] < tight[0]
    assert loose[1] > tight[1]


def test_more_trials_at_the_same_rate_narrows_the_interval():
    few = precision_interval(10, 14)
    many = precision_interval(1000, 1400)
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_level_must_be_a_probability():
    with pytest.raises(ValueError):
        precision_interval(10, 14, level=1.0)


def test_published_cell_matches_the_interval_quoted_in_section_v_c():
    """E2, ten minutes, the learner: 10 hits in 14 fires -> 42 % to 92 %."""
    low, high = precision_interval(10, 14, level=0.95)
    assert round(low * 100) == 42
    assert round(high * 100) == 92


def test_builder_emits_one_row_per_model_corridor_horizon():
    from src.build_detection_precision_ci import build

    frame = build()
    assert frame.height == frame.select("model", "corridor", "horizon").n_unique()
    assert set(frame.columns) == {
        "model", "corridor", "horizon", "fires", "hits",
        "precision", "ci_low", "ci_high", "level",
    }


def test_builder_leaves_bounds_null_where_the_detector_never_fired():
    """XGBoost never fires in some cells. A null bound is the honest value."""
    from src.build_detection_precision_ci import build

    frame = build()
    silent = frame.filter(pl.col("fires") == 0)
    assert silent.height > 0, "expected at least one cell with no fires"
    assert silent["ci_low"].null_count() == silent.height
    assert silent["ci_high"].null_count() == silent.height
    assert silent["precision"].null_count() == silent.height


def test_builder_bounds_bracket_the_precision_it_reports():
    from src.build_detection_precision_ci import build

    frame = build().filter(pl.col("fires") > 0)
    assert frame.height > 0
    for row in frame.iter_rows(named=True):
        assert row["ci_low"] <= row["precision"] <= row["ci_high"]


def test_builder_reproduces_the_section_v_c_cell():
    from src.build_detection_precision_ci import build

    row = build().filter(
        (pl.col("model") == "LSTM")
        & (pl.col("corridor") == "E2")
        & (pl.col("horizon") == 10)
    ).to_dicts()[0]
    assert row["fires"] == 14
    assert row["hits"] == 10
    assert round(row["ci_low"] * 100) == 42
    assert round(row["ci_high"] * 100) == 92


def test_builder_output_is_byte_identical_across_runs():
    from src.build_detection_precision_ci import build, render

    first = render(build())
    second = render(build())
    assert first == second
