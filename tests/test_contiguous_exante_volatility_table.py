"""Contracts on the committed ex-ante volatility table.

The table's whole claim to inferential validity rests on two properties that a
reader cannot check by eye: the thresholds were frozen on train+val, and the
bins are a genuine partition of the reported population. Both are asserted here
against the committed CSV, so a regenerated artifact that quietly violates them
fails the suite instead of shipping.

The regime itself is tested in ``tests/evaluation/test_exante_volatility.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_contiguous_volatility import OUT_CSV, TERCILE_NAMES  # noqa: E402

pytestmark = pytest.mark.skipif(
    not OUT_CSV.exists(),
    reason=f"{OUT_CSV.name} not generated — run src.build_contiguous_volatility",
)


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    return pl.read_csv(OUT_CSV)


class TestCoverage:
    def test_every_corridor_and_horizon_is_present(self, table):
        cells = set(
            zip(table.get_column("corridor"), table.get_column("horizon"))
        )
        assert cells == {(c, h) for c in CORRIDORS for h in HORIZONS}

    def test_three_terciles_per_cell(self, table):
        counts = table.group_by(["corridor", "horizon"]).len()
        assert counts.get_column("len").to_list() == [3] * counts.height
        assert set(table.get_column("tercile")) == set(TERCILE_NAMES)


class TestFrozenCalibration:
    """The ex-ante contract: thresholds come from train+val, never from test."""

    def test_calibration_split_is_train_plus_val(self, table):
        assert set(table.get_column("calib_split")) == {"train+val"}

    def test_thresholds_are_constant_within_a_cell(self, table):
        """One frozen pair per cell — a per-tercile threshold would mean the
        bins were re-derived from the data being reported."""
        spread = table.group_by(["corridor", "horizon"]).agg(
            pl.col("p33_threshold").n_unique().alias("p33"),
            pl.col("p66_threshold").n_unique().alias("p66"),
        )
        assert spread.get_column("p33").to_list() == [1] * spread.height
        assert spread.get_column("p66").to_list() == [1] * spread.height

    def test_thresholds_are_ordered(self, table):
        assert (
            table.get_column("p33_threshold") < table.get_column("p66_threshold")
        ).all()

    def test_calibration_is_larger_than_the_reported_split(self, table):
        """train+val spans ~130 days against test's 22; a calib_n at or below the
        reported n would mean the thresholds were fitted on test."""
        per_cell = table.group_by(["corridor", "horizon"]).agg(
            pl.col("n").sum().alias("test_n"), pl.col("calib_n").first()
        )
        assert (
            per_cell.get_column("calib_n") > per_cell.get_column("test_n")
        ).all()


class TestPartition:
    def test_shares_sum_to_one_per_cell(self, table):
        totals = (
            table.group_by(["corridor", "horizon"])
            .agg(pl.col("share").sum())
            .get_column("share")
            .to_numpy()
        )
        assert np.allclose(totals, 1.0)

    def test_dispersion_increases_across_terciles(self, table):
        """`low`, `mid`, `high` must actually be ordered by the stratifier."""
        for (corridor, horizon), sub in table.sort("tercile_order").group_by(
            ["corridor", "horizon"], maintain_order=True
        ):
            means = sub.get_column("mean_exante_std").to_numpy()
            assert np.all(np.diff(means) > 0), f"{corridor} h={horizon}: {means}"

    def test_no_bin_is_degenerate(self, table):
        """Frozen thresholds may unbalance the bins, but not empty them."""
        assert table.get_column("share").min() > 0.05


class TestReportedQuantities:
    def test_deltas_reconstruct_from_the_maes(self, table):
        lstm = table.get_column("mae_lstm").to_numpy()
        assert np.allclose(
            lstm - table.get_column("mae_persist").to_numpy(),
            table.get_column("delta_lstm_persist").to_numpy(),
        )
        assert np.allclose(
            lstm - table.get_column("mae_xgb").to_numpy(),
            table.get_column("delta_lstm_xgb").to_numpy(),
        )

    def test_service_day_count_matches_the_test_window(self, table):
        """The clustered variance's effective n. The test split is 2024-02-08 to
        2024-02-29, so a cell can hold at most 22 service days."""
        days = table.get_column("n_service_days")
        assert days.min() >= 20
        assert days.max() <= 22

    def test_p_values_are_in_range(self, table):
        for col in ("dm_p_lstm_persist", "dm_p_lstm_xgb"):
            values = table.get_column(col).to_numpy()
            assert np.all((values >= 0.0) & (values <= 1.0)), col


class TestTheFinding:
    """The claim this table exists to support, pinned so a rebuild must keep it.

    The crossover is not a horizon threshold — it is a VOLATILITY threshold that
    the horizon walks across. Within a horizon the learner does relatively better
    the more the input window moved; lengthening the horizon pushes its advantage
    down into calmer terciles.
    """

    # E59 h=1 has mid and high tied at +0.295 (they differ by 0.0003). Every
    # other pair is strictly ordered, so the ordering claim is stated as
    # non-increasing with a tolerance, and the endpoints strictly — rather than
    # asserting a strict monotonicity the data does not carry.
    FLAT_TOL = 1e-3

    def test_advantage_is_ordered_across_terciles(self, table):
        for (corridor, horizon), sub in table.sort("tercile_order").group_by(
            ["corridor", "horizon"], maintain_order=True
        ):
            deltas = sub.get_column("delta_lstm_persist").to_numpy()
            assert np.all(np.diff(deltas) < self.FLAT_TOL), (
                f"{corridor} h={horizon}: {deltas}"
            )

    def test_the_calm_to_volatile_gap_is_strict_everywhere(self, table):
        """low -> high is the claim itself, and it never merely ties."""
        for (corridor, horizon), sub in table.sort("tercile_order").group_by(
            ["corridor", "horizon"], maintain_order=True
        ):
            deltas = sub.get_column("delta_lstm_persist").to_numpy()
            assert deltas[-1] < deltas[0] - 0.05, f"{corridor} h={horizon}: {deltas}"

    def test_advantage_grows_with_the_horizon_within_a_tercile(self, table):
        for (corridor, tercile), sub in table.sort("horizon").group_by(
            ["corridor", "tercile"], maintain_order=True
        ):
            deltas = sub.get_column("delta_lstm_persist").to_numpy()
            assert np.all(np.diff(deltas) < 0), f"{corridor} {tercile}: {deltas}"

    def test_persistence_owns_the_calm_tercile_at_one_step(self, table):
        low_h1 = table.filter(
            (pl.col("horizon") == 1) & (pl.col("tercile") == "low")
        )
        assert (low_h1.get_column("delta_lstm_persist") > 0).all()
        assert (low_h1.get_column("dm_p_lstm_persist") < 0.01).all()
