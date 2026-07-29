"""The dissociation must be measured in three windows, not one read three times.

This mirrors ``tests/test_rolling_origin_significance.py`` and exists for the
same reason: a ``fold`` argument that silently selects nothing still produces a
full, plausible, perfectly formatted table — one where every origin agrees
because every origin IS the published window.
``test_each_origin_reads_a_different_population`` is the guard that the table
claiming temporal robustness was built from different data.

The behavioural tests below are the ones the paper would cite, and they now
guard a CORRECTED claim. The original version of this file asserted that
persistence out-detects the learner in all 36 cells and treated that as the
finding. It is reproducible and it is an artifact of the fixed 0.5x cut — see
``tests/test_detection_calibrated.py`` for the dismantling. What this file now
pins is the split:

* the fixed-cut sweep, kept and labelled as the artifact it is,
* the flattening (CV bias negative in 36 of 36), which survives untouched,
* the threshold-free crossover, which must hold at every origin or the
  correction is itself a February story.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS, load_lstm  # noqa: E402
from src.build_rolling_origin_dissociation import (  # noqa: E402
    MODELS,
    OUT_CSV,
    OUT_SUMMARY_CSV,
    agreement,
)
from src.build_rolling_origin_significance import ORIGINS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV.name} not built yet")
    return pl.read_csv(OUT_CSV)


@pytest.fixture(scope="module")
def summary() -> pl.DataFrame:
    if not OUT_SUMMARY_CSV.exists():
        pytest.skip(f"{OUT_SUMMARY_CSV.name} not built yet")
    return pl.read_csv(OUT_SUMMARY_CSV)


class TestTheOriginsAreReallyDifferent:
    def test_each_origin_reads_a_different_population(self):
        """Three windows of different length over different months cannot share
        a row count by accident; equality means the fold argument did not select."""
        counts = {origin: load_lstm(origin).height for origin in ORIGINS}
        assert len(set(counts.values())) == len(ORIGINS), (
            f"origins share a row count, so at least two read the same files: {counts}"
        )

    def test_every_cell_scores_a_real_population(self, table):
        assert table.get_column("n_vectors").min() > 0
        assert table.get_column("n_cells").min() > 0


class TestTableShape:
    def test_one_row_per_origin_cell_and_model(self, table):
        expected = len(ORIGINS) * len(CORRIDORS) * len(HORIZONS) * len(MODELS)
        assert table.height == expected
        assert table.unique(
            subset=["origin", "corridor", "horizon", "model"]
        ).height == expected

    def test_only_the_pairable_models_appear(self, table):
        """XGBoost is not re-run at the rolling origins, so it cannot be here."""
        assert set(table.get_column("model").unique()) == {n for n, _ in MODELS}

    def test_agreement_is_recomputable_from_the_detail_table(self, table, summary):
        assert agreement(table).equals(summary)

    def test_agrees_matches_the_three_winners(self, summary):
        for row in summary.iter_rows(named=True):
            winners = {row[f"winner_{origin}"] for origin in ORIGINS}
            assert row["agrees"] == (len(winners) == 1), row
            assert row["winner"] == (
                next(iter(winners)) if len(winners) == 1 else "SPLIT"
            )


class TestWhatTheTableSaysAboutTheClaim:
    """Behaviour, not formatting: the findings the paper would cite."""

    def test_the_fixed_cut_hands_persistence_every_origin(self, table):
        """The artifact, pinned so it stays reproducible. Persistence wins all 36
        cells at the published 0.5x cut. This is NOT evidence that it detects
        better — see ``test_the_learner_out_discriminates_at_h10_everywhere``
        for the same cells judged without a cut. It is kept because a correction
        that cannot reproduce the thing it corrects is not a correction."""
        lstm = table.filter(pl.col("model") == "LSTM").sort(
            ["corridor", "horizon", "origin"]
        )
        persist = table.filter(pl.col("model") == "Persistence").sort(
            ["corridor", "horizon", "origin"]
        )
        assert lstm.height == persist.height == 36

        losses = [
            (c, h, o)
            for c, h, o, a, b in zip(
                lstm.get_column("corridor"),
                lstm.get_column("horizon"),
                lstm.get_column("origin"),
                lstm.get_column("bunching_f1"),
                persist.get_column("bunching_f1"),
            )
            if not b > a
        ]
        assert not losses, f"the learner out-detects persistence in {losses}"

    def test_the_fixed_cut_verdict_loses_to_a_constant_at_long_horizons(self, table):
        """Why that sweep is not a finding: at h=10 the declared winner scores
        BELOW flagging every cell, at every origin. Nine of nine."""
        long_h = table.filter(
            (pl.col("model") == "Persistence") & (pl.col("horizon") == 10)
        )
        assert long_h.height == len(CORRIDORS) * len(ORIGINS)
        below = long_h.filter(pl.col("bunching_f1") < pl.col("trivial_f1"))
        assert below.height == long_h.height, (
            "persistence now beats the trivial detector somewhere at h=10; the "
            "document's Section 5.3 argument needs rewriting, not this test"
        )

    def test_the_learner_out_discriminates_at_h10_everywhere(self, table):
        """The correction, and it must hold in all three windows. If the AUC
        reversal were a February story, this fails instead of being averaged."""
        for origin in ORIGINS:
            for corridor in CORRIDORS:
                cell = table.filter(
                    (pl.col("origin") == origin)
                    & (pl.col("corridor") == corridor)
                    & (pl.col("horizon") == 10)
                )
                rows = {r["model"]: r for r in cell.iter_rows(named=True)}
                assert rows["LSTM"]["auc"] > rows["Persistence"]["auc"], (
                    f"{corridor}@{origin} h=10: learner AUC "
                    f"{rows['LSTM']['auc']:.4f} <= persistence "
                    f"{rows['Persistence']['auc']:.4f}"
                )

    def test_persistence_out_discriminates_at_h1_everywhere(self, table):
        """The other half of the crossover. The correction is not "the learner
        wins everything" — at h=1 persistence genuinely leads, in all nine."""
        for origin in ORIGINS:
            for corridor in CORRIDORS:
                cell = table.filter(
                    (pl.col("origin") == origin)
                    & (pl.col("corridor") == corridor)
                    & (pl.col("horizon") == 1)
                )
                rows = {r["model"]: r for r in cell.iter_rows(named=True)}
                assert rows["Persistence"]["auc"] > rows["LSTM"]["auc"], (
                    f"{corridor}@{origin} h=1: persistence does not lead"
                )

    def test_the_learner_is_never_at_chance(self, table):
        """Blindness would mean AUC near 0.5 somewhere. It never happens, at any
        origin, in any cell — which is what rules out the information reading."""
        auc = table.filter(pl.col("model") == "LSTM").get_column("auc")
        assert auc.len() == 36
        assert auc.min() > 0.55, f"minimum learner AUC is {auc.min()}"

    def test_the_learner_under_reports_irregularity_at_every_origin(self, table):
        """CV bias negative in all 36 cells: the learner always predicts a
        smoother corridor than the real one."""
        bias = table.filter(pl.col("model") == "LSTM").get_column("cv_bias")
        assert bias.len() == 36
        assert (bias < 0).all(), f"positive CV bias in {bias.to_list()}"

    def test_persistence_preserves_the_shape_it_copies(self, table):
        """Persistence propagates the observed vector, so its CV bias must sit
        near zero. If it drifted, the mechanism attributed to MAE would instead
        be an artifact of how the vectors are assembled."""
        bias = table.filter(pl.col("model") == "Persistence").get_column("cv_bias")
        assert bias.abs().max() < 0.05, (
            f"persistence CV bias is not ~0: max |bias| = {bias.abs().max()}"
        )

    def test_the_artifact_widens_with_the_horizon_at_every_origin(self, table):
        """The fixed-cut gap grows monotonically with the horizon in all nine
        (corridor, origin) pairs. Read correctly this is a property of the CUT,
        not of detection: the horizon compresses the forecast further, so the
        0.5x cut sits deeper in its tail. It is pinned because that monotonicity
        is what makes the units explanation predictive rather than post hoc."""
        for corridor in CORRIDORS:
            for origin in ORIGINS:
                cell = table.filter(
                    (pl.col("corridor") == corridor) & (pl.col("origin") == origin)
                )
                ratios = []
                for horizon in HORIZONS:
                    at = {
                        r["model"]: r
                        for r in cell.filter(pl.col("horizon") == horizon).iter_rows(
                            named=True
                        )
                    }
                    ratios.append(
                        at["Persistence"]["bunching_f1"] / at["LSTM"]["bunching_f1"]
                    )
                assert ratios == sorted(ratios), (
                    f"{corridor}@{origin} detection gap is not monotone: {ratios}"
                )

    def test_the_collapse_is_recall_not_precision(self, table):
        """The mechanism claim. A learner that were simply WRONG would lose
        precision too; regression to the mean shows up as preserved precision
        with recall near zero, and that distinction is what rules out noise."""
        lstm = table.filter(pl.col("model") == "LSTM")
        long_h = lstm.filter(pl.col("horizon") == 10)
        assert long_h.height == len(CORRIDORS) * len(ORIGINS)
        assert (long_h.get_column("bunching_recall") < 0.05).all()
        # Precision is only estimable where the model fires enough times: at
        # E2/r1 it fires three times in the whole window, and a ratio over three
        # events carries no information either way.
        firing = long_h.filter(pl.col("bunching_tp") + pl.col("bunching_fp") >= 30)
        assert firing.height >= 6, "too few cells fire to test precision at all"
        assert (firing.get_column("bunching_precision") > 0.3).all(), (
            "precision collapsed too — this is not the mean-reversion signature"
        )


class TestTheDocumentQuotesTheTable:
    """Every number narrated in the prose must come off the CSV."""

    DOC = REPO_ROOT / "docs" / "resultados" / "documento-resultados.md"
    HEADING = "### 5.5 Nada de esto es de febrero"

    @pytest.fixture(scope="class")
    def section(self) -> str:
        text = self.DOC.read_text(encoding="utf-8")
        assert self.HEADING in text, f"{self.HEADING!r} is gone from the document"
        return text.split(self.HEADING)[1].split("## 6.")[0]

    def test_the_cv_bias_count_is_not_stale(self, section, table):
        n = int(
            (table.filter(pl.col("model") == "LSTM").get_column("cv_bias") < 0).sum()
        )
        assert f"{n} celdas" in section, (
            f"CV bias is negative in {n} cells; the section does not say so"
        )

    def test_the_auc_agreement_count_is_not_stale(self, section, summary):
        """The correction's robustness claim. Stated as a count so it cannot
        drift into "it holds everywhere" if a cell starts disagreeing."""
        n_agree = int(summary.get_column("agrees_auc").sum())
        assert f"{n_agree} de 12" in section, (
            f"{n_agree} of {summary.height} cells agree on the threshold-free "
            "winner; the section does not state that count"
        )

    def test_the_split_cell_is_named(self, section, summary):
        """A reader must be told WHICH cell disagrees, not just how many. Hiding
        the identity of a split is how a limitation becomes a footnote."""
        split = summary.filter(~pl.col("agrees_auc"))
        for row in split.iter_rows(named=True):
            label = f"{row['corridor']} h={row['horizon']}"
            assert label in section, f"the split cell {label} is not named"

    def test_the_extreme_ratio_is_quoted_and_labelled_an_artifact(
        self, section, summary
    ):
        """The largest fixed-cut ratio is the number a reader remembers, so it
        must be quoted exactly AND framed as the artifact. Quoting 2299x without
        that framing is how the retracted claim would come back."""
        import re

        ratios = [
            summary.get_column(f"f1_ratio_{origin}").max() for origin in ORIGINS
        ]
        top = max(r for r in ratios if r is not None)
        quoted = [int(x) for x in re.findall(r"(\d{3,5})×", section)]
        assert quoted, "the section quotes no F1 ratio"
        assert max(quoted) == round(top), (
            f"largest ratio in the CSV is {top:.1f}x; the section quotes "
            f"{max(quoted)}x"
        )
        assert "artefacto" in section or "corte de 0.5" in section, (
            "the section quotes the ratio without saying what produced it"
        )
