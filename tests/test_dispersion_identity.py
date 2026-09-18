"""Is the compression the theorem, or is it this corpus's preprocessing?

Section V-B reports that the predicted vector is less dispersed than the observed
one in every cell. A reviewer can answer that with the corpus: this corridor has
no published geometry, the direction of travel is inferred, and every measurement
error inflates the prediction error. A noisier target produces a larger
compression for reasons that say nothing about forecasting.

The decomposition separates the two readings. Across the positions of one vector,

    Var(h) = Var(h_hat) + Var(e) + 2*Cov(h_hat, e).

If the forecast behaves like a conditional mean the covariance vanishes, and the
compression ratio is then a function of one measurable quantity — how large the
error is relative to the observed spread:

    Var(h_hat)/Var(h)  ==  1 - Var(e)/Var(h).

That relation holding cell by cell is what distinguishes the theorem operating
from an artifact: an artifact of the corpus has no reason to track the error term
across twelve cells whose ratios span an order of magnitude.

``TestTheIdentityIsAlgebra``
    The three terms close exactly. If this fails the columns are wrong, not the
    finding.

``TestCompressionTracksTheError``
    The result the manuscript rests on. The measured ratio follows the error term
    alone, so the compression is not free to be whatever the corpus makes it.

``TestTheDocumentDeclaresWhatTheModelExplains``
    The uncomfortable half of the same table, pinned so it cannot quietly leave
    the manuscript.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_dispersion_identity import (  # noqa: E402
    OUT_CSV,
    SCORING_ORIGIN,
    build,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV.name} not built yet")
    return pl.read_csv(OUT_CSV)


class TestTableShape:
    def test_one_row_per_cell(self, table):
        expected = len(CORRIDORS) * len(HORIZONS)
        assert table.height == expected
        assert table.unique(subset=["corridor", "horizon"]).height == expected

    def test_it_is_scored_where_the_rest_of_the_verdicts_are(self):
        """A different origin would make these columns incomparable with §V."""
        assert SCORING_ORIGIN == "main"


class TestTheIdentityIsAlgebra:
    """The three variance terms close. This is a check on the columns."""

    def test_the_measured_ratio_is_the_two_variances(self, table):
        for row in table.iter_rows(named=True):
            assert abs(row["ratio_measured"] - row["v_pred"] / row["v_true"]) < 1e-9

    def test_the_full_decomposition_closes(self, table):
        """With the covariance term the identity is exact, not approximate."""
        for row in table.iter_rows(named=True):
            total = row["v_pred"] + row["v_err"] + 2.0 * row["c_cross"]
            assert abs(total - row["v_true"]) < 1e-8, row

    def test_the_explained_share_is_a_share(self, table):
        share = table.get_column("explained").to_numpy()
        assert (share >= 0.0).all()
        assert (share <= 1.0).all()


class TestCompressionTracksTheError:
    """The answer to 'this is just your noisy corpus'."""

    def test_the_measured_ratio_follows_the_error_term(self, table):
        """One number predicts the compression of all twelve cells.

        The ratios span 0,04 to 0,56 across the corpus. A preprocessing artifact
        has no reason to move with the error term over that range.
        """
        measured = table.get_column("ratio_measured").to_numpy()
        identity = table.get_column("explained").to_numpy()
        assert np.corrcoef(measured, identity)[0, 1] > 0.95

    def test_the_correlation_travels_in_the_table(self, table):
        """The manuscript quotes it, so it needs a source of truth to quote."""
        column = table.get_column("r_identity")
        assert column.n_unique() == 1
        measured = table.get_column("ratio_measured").to_numpy()
        identity = table.get_column("explained").to_numpy()
        assert abs(column[0] - np.corrcoef(measured, identity)[0, 1]) < 1e-9

    def test_the_paper_prints_that_correlation(self, table):
        paper = (REPO_ROOT / "docs" / "paper" / "paper.md").read_text(
            encoding="utf-8"
        )
        printed = f"{table.get_column('r_identity')[0]:.3f}"
        assert printed in paper, printed

    def test_the_gap_left_by_dropping_the_covariance_is_small(self, table):
        """How far the forecast sits from a conditional mean, in the same units."""
        gap = table.get_column("gap_no_cov").to_numpy()
        assert np.abs(gap).max() < 0.15

    def test_the_forecast_never_compresses_more_than_a_conditional_mean(self, table):
        """A negative gap would mean the compression exceeds what the error
        explains, and the corpus would be doing work the theorem does not."""
        assert (table.get_column("gap_no_cov") >= 0.0).all()

    def test_compression_deepens_with_the_horizon_in_every_corridor(self, table):
        """Corollary 2 read on this axis: longer horizon, larger error, less
        surviving spread."""
        for corridor in CORRIDORS:
            cell = table.filter(pl.col("corridor") == corridor).sort("horizon")
            ratios = cell.get_column("ratio_measured").to_list()
            assert ratios == sorted(ratios, reverse=True), (corridor, ratios)


class TestTheDocumentDeclaresWhatTheModelExplains:
    """Publishing the compression and withholding its other reading is selective."""

    PAPER = REPO_ROOT / "docs" / "paper" / "paper.md"

    @pytest.fixture(scope="class")
    def paper(self) -> str:
        return self.PAPER.read_text(encoding="utf-8").replace("\r\n", "\n")

    def test_the_paper_reports_the_widest_and_the_narrowest_share(self, table):
        """Both ends, so the range cannot be read as uniformly good or bad."""
        share = table.get_column("explained")
        assert share.min() < 0.05
        assert share.max() > 0.45

    def test_the_paper_prints_both_ends_as_percentages(self, table, paper):
        """With the unit attached, so the figure cannot pass by coincidence."""
        share = table.get_column("explained")
        for value in (share.min(), share.max()):
            printed = "{:.1f} %".format(100.0 * value)
            assert printed in paper, printed

    def test_the_paper_names_the_threat_the_identity_answers(self, paper):
        """Section VI owes the reader the corpus reading, not only the theorem."""
        assert "ruido de medición" in paper


class TestBuildIsDeterministic:
    def test_two_builds_agree(self):
        assert build().equals(build())
