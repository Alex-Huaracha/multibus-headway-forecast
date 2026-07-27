"""Tests for the ex-ante volatility stratifier.

The point of this stratifier is that it is computable from the input window
alone. If it ever reads a timestep the model did not see, the subgroup tests
built on it become as circular as the retrospective regime it replaces — so the
independence from the target is asserted directly, not assumed from the call
site.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.exante_volatility import window_dispersion


def _window(rows: list[list[float]], *, mask: list[list[bool]] | None = None):
    """One sample, one pair_rank: shape (1, T_in, 1) values + mask."""
    values = np.array(rows, dtype=np.float64).reshape(1, len(rows), 1)
    if mask is None:
        flags = np.ones_like(values, dtype=bool)
    else:
        flags = np.array(mask, dtype=bool).reshape(1, len(rows), 1)
    return values, flags


class TestValue:
    def test_matches_numpy_sample_std(self):
        vals = [3.0, 5.0, 9.0, 1.0, 4.0]
        values, mask = _window([[v] for v in vals])
        assert window_dispersion(values, mask)[0, 0] == pytest.approx(
            np.std(vals, ddof=1)
        )

    def test_constant_window_has_zero_dispersion(self):
        values, mask = _window([[7.0]] * 6)
        assert window_dispersion(values, mask)[0, 0] == pytest.approx(0.0)

    def test_uses_ddof_1_not_the_population_formula(self):
        """A frozen threshold read as minutes must not be biased low."""
        vals = [1.0, 2.0]
        values, mask = _window([[v] for v in vals])
        got = window_dispersion(values, mask)[0, 0]
        assert got == pytest.approx(np.std(vals, ddof=1))
        assert got != pytest.approx(np.std(vals))


class TestMasking:
    def test_masked_positions_are_excluded_from_the_mean(self):
        """The grid writes 0.0 under a False mask; that 0.0 must not count."""
        observed = [4.0, 6.0, 8.0]
        values, mask = _window(
            [[4.0], [0.0], [6.0], [0.0], [8.0]],
            mask=[[True], [False], [True], [False], [True]],
        )
        assert window_dispersion(values, mask)[0, 0] == pytest.approx(
            np.std(observed, ddof=1)
        )

    def test_masked_garbage_does_not_leak(self):
        """Whatever sits behind a False mask is irrelevant, however extreme."""
        values, mask = _window(
            [[4.0], [1e9], [6.0], [-1e9], [8.0]],
            mask=[[True], [False], [True], [False], [True]],
        )
        assert window_dispersion(values, mask)[0, 0] == pytest.approx(
            np.std([4.0, 6.0, 8.0], ddof=1)
        )

    def test_single_observation_is_nan_not_zero(self):
        """One point has no dispersion; reporting 0.0 would put it in `low`."""
        values, mask = _window(
            [[5.0], [0.0], [0.0]], mask=[[True], [False], [False]]
        )
        assert np.isnan(window_dispersion(values, mask)[0, 0])

    def test_fully_masked_cell_is_nan(self):
        values, mask = _window([[0.0]] * 4, mask=[[False]] * 4)
        assert np.isnan(window_dispersion(values, mask)[0, 0])

    def test_min_obs_threshold_is_honoured(self):
        values, mask = _window(
            [[1.0], [2.0], [3.0], [0.0]],
            mask=[[True], [True], [True], [False]],
        )
        assert np.isfinite(window_dispersion(values, mask, min_obs=3)[0, 0])
        assert np.isnan(window_dispersion(values, mask, min_obs=4)[0, 0])


class TestIndependenceFromTheTarget:
    """The property that makes subgroup inference legitimate here."""

    def test_signature_admits_no_target(self):
        """There is no argument through which the outcome could enter."""
        import inspect

        params = set(inspect.signature(window_dispersion).parameters)
        assert params == {"values", "mask", "min_obs"}

    def test_result_depends_only_on_the_window(self):
        """Two samples with identical windows get identical dispersion.

        Whatever happens after the window — the target, the realized change —
        is not an input, so it cannot move the bin a sample lands in.
        """
        window = [[2.0], [5.0], [3.0], [9.0]]
        values, mask = _window(window)
        twice_values = np.concatenate([values, values], axis=0)
        twice_mask = np.concatenate([mask, mask], axis=0)
        out = window_dispersion(twice_values, twice_mask)
        assert out[0, 0] == pytest.approx(out[1, 0])


class TestShapeContract:
    def test_reduces_the_time_axis_only(self):
        values = np.zeros((7, 12, 4))
        mask = np.ones_like(values, dtype=bool)
        assert window_dispersion(values, mask).shape == (7, 4)

    def test_pair_ranks_are_independent(self):
        """A quiet slot next to a volatile one must not borrow its dispersion."""
        values = np.zeros((1, 4, 2))
        values[0, :, 0] = [1.0, 1.0, 1.0, 1.0]
        values[0, :, 1] = [0.0, 10.0, 0.0, 10.0]
        mask = np.ones_like(values, dtype=bool)
        out = window_dispersion(values, mask)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] > 5.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            window_dispersion(np.zeros((2, 3, 1)), np.ones((2, 4, 1), dtype=bool))

    def test_wrong_rank_raises(self):
        with pytest.raises(ValueError, match=r"expected \(n, T_in, max_N\)"):
            window_dispersion(np.zeros((2, 3)), np.ones((2, 3), dtype=bool))

    def test_min_obs_below_two_raises(self):
        values, mask = _window([[1.0], [2.0]])
        with pytest.raises(ValueError, match="min_obs must be >= 2"):
            window_dispersion(values, mask, min_obs=1)
