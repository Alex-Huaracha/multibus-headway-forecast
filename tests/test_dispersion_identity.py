"""Is the compression the theorem, or is it this corpus's preprocessing?

Section V-A reports that the predicted vector is less dispersed than the observed
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

A second reading is available to the reviewer once the first is closed: that the
compression is a habit of recurrent networks rather than of forecasts fitted by
squared error. The decomposition therefore runs over both forecasters the paper
publishes, scored on the same vectors, and the manuscript rests on the two agreeing
rather than on the one it happens to feature.

``TestTheIdentityIsAlgebra``
    The three terms close exactly. If this fails the columns are wrong, not the
    finding.

``TestCompressionTracksTheError``
    The result the manuscript rests on. The measured ratio follows the error term
    alone, so the compression is not free to be whatever the corpus makes it.

``TestTheCompressionIsNotTheArchitecture``
    The answer to 'this is a pathology of your LSTM'. A recurrent network and a
    boosted-tree ensemble share no inductive bias, and they compress alike.

``TestTheDocumentDeclaresWhatTheModelExplains``
    The uncomfortable half of the same table, pinned so it cannot quietly leave
    the manuscript.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_dispersion_identity import (  # noqa: E402
    MODELS,
    OUT_CSV,
    PUBLISHED_MODEL,
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
    def test_one_row_per_cell_and_model(self, table):
        expected = len(MODELS) * len(CORRIDORS) * len(HORIZONS)
        assert table.height == expected
        assert table.unique(subset=["model", "corridor", "horizon"]).height == expected

    def test_both_forecasters_cover_the_same_cells(self, table):
        """A model missing a cell would let the agreement be an artifact of
        which cells each one happens to be scored on."""
        cells = {
            name: set(
                table.filter(pl.col("model") == name)
                .select("corridor", "horizon")
                .iter_rows()
            )
            for name in MODELS
        }
        assert len(set(map(frozenset, cells.values()))) == 1, cells

    def test_the_featured_model_is_one_of_the_two(self):
        assert PUBLISHED_MODEL in MODELS

    def test_it_is_scored_where_the_rest_of_the_verdicts_are(self):
        """A different origin would make these columns incomparable with §V."""
        assert SCORING_ORIGIN == "main"

    def test_the_two_forecasters_are_scored_on_the_same_vectors(self, table):
        """The comparison is a paired one, so it owes the same population.

        Section IV-C fixes one sample population and the manuscript's own
        contract requires every A-beats-B claim to trace to identical samples.
        An unequal vector count here would mean the two ratios are averages over
        different corpora.
        """
        for corridor in CORRIDORS:
            for horizon in HORIZONS:
                cell = table.filter(
                    (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
                )
                assert cell.get_column("n_vectors").n_unique() == 1, cell


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

    @pytest.mark.parametrize("model", MODELS)
    def test_the_measured_ratio_follows_the_error_term(self, table, model):
        """One number predicts the compression of all twelve cells.

        The ratios span 0.04 to 0.56 across the corpus. A preprocessing artifact
        has no reason to move with the error term over that range.
        """
        cell = table.filter(pl.col("model") == model)
        measured = cell.get_column("ratio_measured").to_numpy()
        identity = cell.get_column("explained").to_numpy()
        assert np.corrcoef(measured, identity)[0, 1] > 0.95

    @pytest.mark.parametrize("model", MODELS)
    def test_the_correlation_travels_in_the_table(self, table, model):
        """The manuscript quotes it, so it needs a source of truth to quote."""
        cell = table.filter(pl.col("model") == model)
        column = cell.get_column("r_identity")
        assert column.n_unique() == 1
        measured = cell.get_column("ratio_measured").to_numpy()
        identity = cell.get_column("explained").to_numpy()
        assert abs(column[0] - np.corrcoef(measured, identity)[0, 1]) < 1e-9

    @pytest.mark.parametrize("model", MODELS)
    def test_the_paper_prints_that_correlation(self, table, model):
        """Both, because publishing only the featured one would leave the
        agreement between architectures unsourced."""
        paper = (REPO_ROOT / "docs" / "paper" / "paper.md").read_text(
            encoding="utf-8"
        )
        row = table.filter(pl.col("model") == model)
        printed = f"{row.get_column('r_identity')[0]:.3f}"
        assert printed in paper, (model, printed)

    def test_the_gap_left_by_dropping_the_covariance_is_small(self, table):
        """How far the forecast sits from a conditional mean, in the same units."""
        gap = table.get_column("gap_no_cov").to_numpy()
        assert np.abs(gap).max() < 0.15

    def test_the_forecast_never_compresses_more_than_a_conditional_mean(self, table):
        """A negative gap would mean the compression exceeds what the error
        explains, and the corpus would be doing work the theorem does not."""
        assert (table.get_column("gap_no_cov") >= 0.0).all()

    @pytest.mark.parametrize("model", MODELS)
    def test_compression_deepens_with_the_horizon_in_every_corridor(
        self, table, model
    ):
        """Corollary 2 read on this axis: longer horizon, larger error, less
        surviving spread."""
        for corridor in CORRIDORS:
            cell = table.filter(
                (pl.col("model") == model) & (pl.col("corridor") == corridor)
            ).sort("horizon")
            ratios = cell.get_column("ratio_measured").to_list()
            assert ratios == sorted(ratios, reverse=True), (model, corridor, ratios)


class TestTheCompressionIsNotTheArchitecture:
    """The answer to 'this is a pathology of your LSTM'.

    A recurrent network and a boosted-tree ensemble share no inductive bias and
    no optimizer. What they share is a squared-error objective, which is what the
    corollary is about. If the compression were a habit of the architecture, the
    two would not land on the same curve.
    """

    @pytest.mark.parametrize("model", MODELS)
    def test_every_cell_compresses_under_both_forecasters(self, table, model):
        cell = table.filter(pl.col("model") == model)
        assert cell.height == len(CORRIDORS) * len(HORIZONS)
        assert (cell.get_column("ratio_measured") < 1.0).all(), cell

    def test_the_two_forecasters_compress_alike_cell_by_cell(self, table):
        """Not the same number — the same ordering across an order of magnitude.

        The claim is that the compression is a property of the objective, so the
        cell where one forecaster keeps the most spread has to be the cell where
        the other one does too.
        """
        pair = (
            table.filter(pl.col("model") == MODELS[0])
            .select("corridor", "horizon", "ratio_measured")
            .join(
                table.filter(pl.col("model") == MODELS[1]).select(
                    "corridor", "horizon", pl.col("ratio_measured").alias("other")
                ),
                on=["corridor", "horizon"],
            )
        )
        assert pair.height == len(CORRIDORS) * len(HORIZONS)
        correlation = np.corrcoef(
            pair.get_column("ratio_measured").to_numpy(),
            pair.get_column("other").to_numpy(),
        )[0, 1]
        assert correlation > 0.90, correlation

    def test_the_identity_holds_for_both_and_not_only_the_featured_one(self, table):
        """If it held for one architecture alone the decomposition would be
        describing that model, not the objective it was fitted with."""
        for model in MODELS:
            r = table.filter(pl.col("model") == model).get_column("r_identity")[0]
            assert r > 0.95, (model, r)

    def test_the_paper_states_that_both_architectures_compress(self, table):
        """A measurement taken and left out of the document does not answer the
        objection it was taken to answer."""
        paper = re.sub(
            r"\s+",
            " ",
            (REPO_ROOT / "docs" / "paper" / "paper.md").read_text(encoding="utf-8"),
        )
        assert "XGBoost" in paper
        worst = (
            table.filter(pl.col("model") != PUBLISHED_MODEL)
            .sort("ratio_measured")
            .row(0, named=True)
        )
        assert f"{worst['ratio_measured']:.3f}" in paper, worst


class TestTheDocumentDeclaresWhatTheModelExplains:
    """Publishing the compression and withholding its other reading is selective."""

    PAPER = REPO_ROOT / "docs" / "paper" / "paper.md"

    @pytest.fixture(scope="class")
    def paper(self) -> str:
        # Whitespace collapsed: these guards assert what the paper states, not
        # where its lines happen to wrap.
        return re.sub(r"\s+", " ", self.PAPER.read_text(encoding="utf-8"))

    def test_the_paper_reports_the_widest_and_the_narrowest_share(self, table):
        """Both ends, so the range cannot be read as uniformly good or bad."""
        share = table.filter(pl.col("model") == PUBLISHED_MODEL).get_column("explained")
        assert share.min() < 0.05
        assert share.max() > 0.45

    def test_the_paper_prints_both_ends_as_percentages(self, table, paper):
        """With the unit attached, so the figure cannot pass by coincidence."""
        share = table.filter(pl.col("model") == PUBLISHED_MODEL).get_column("explained")
        for value in (share.min(), share.max()):
            printed = "{:.1f} %".format(100.0 * value)
            assert printed in paper, printed

    def test_the_paper_names_the_threat_the_identity_answers(self, paper):
        """Section VI owes the reader the corpus reading, not only the theorem."""
        assert "ruido de medición" in paper


class TestBuildIsDeterministic:
    def test_two_builds_agree(self):
        assert build().equals(build())
