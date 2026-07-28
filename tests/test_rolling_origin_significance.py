"""The rolling-origin table must compare three DIFFERENT windows.

The failure this file is built around is not a wrong number, it is a silent
no-op: a `fold` parameter that selects nothing still produces a full, plausible,
perfectly formatted table — one where all three origins are the published window
read three times, and every cell agrees because every cell is the same cell.
`test_each_origin_reads_a_different_population` is the guard that a table
claiming temporal robustness was actually built from different data.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS, load_lstm  # noqa: E402
from src.build_rolling_origin_significance import (  # noqa: E402
    ORIGINS,
    OUT_CSV,
    OUT_SUMMARY_CSV,
    agreement,
)

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
        """Three origins, three sample counts — not one window read three times.

        Windows of different length over different months cannot yield the same
        row count by accident; equality here means the fold parameter did not
        select, and the whole table is the published window in triplicate.
        """
        counts = {origin: load_lstm(origin).height for origin in ORIGINS}
        assert len(set(counts.values())) == len(ORIGINS), (
            f"origins share a row count, so at least two read the same files: {counts}"
        )

    def test_the_published_origin_is_the_default(self):
        assert load_lstm().height == load_lstm("main").height

    def test_an_unknown_origin_finds_nothing(self):
        with pytest.raises(FileNotFoundError, match="no LSTM residuals"):
            load_lstm("r99")

    def test_a_partial_origin_refuses_to_report(self, tmp_path, monkeypatch):
        """Seven files of eight must raise, not quietly score seven."""
        import src.build_contiguous_significance as mod

        header = ("corridor,direction,horizon,split,start_ts,target_ts,"
                  "pair_rank,y_true,y_pred_model,y_pred_persist\n")
        row = ("E2,-1,1,test,2024-01-01T00:00:00.000000,"
               "2024-01-01T00:12:00.000000,1,10.0,10.0,10.0\n")
        (tmp_path / "lstm_contig_r1_residuals_h1.csv").write_text(header + row)
        monkeypatch.setattr(mod, "LSTM_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="incomplete"):
            mod.load_lstm("r1")


class TestTableShape:
    def test_one_row_per_origin_cell_and_metric(self, table):
        expected = len(ORIGINS) * len(CORRIDORS) * len(HORIZONS) * 2
        assert table.height == expected
        assert table.unique(
            subset=["origin", "corridor", "horizon", "metric"]
        ).height == expected

    def test_every_origin_is_present(self, table):
        assert set(table.get_column("origin").unique()) == set(ORIGINS)

    def test_only_the_pairable_comparison_is_reported(self, table):
        """XGBoost is not re-run at the rolling origins, so it cannot appear."""
        assert set(table.get_column("comparison").unique()) == {"LSTM_vs_PERSIST"}

    def test_every_cell_scores_a_real_population(self, table):
        assert table.get_column("n").min() > 0


class TestAgreement:
    def test_one_row_per_cell(self, summary):
        assert summary.height == len(CORRIDORS) * len(HORIZONS)

    def test_agrees_matches_the_three_winners(self, summary):
        for row in summary.iter_rows(named=True):
            winners = {row[f"winner_{origin}"] for origin in ORIGINS}
            assert row["agrees"] == (len(winners) == 1), row
            assert row["winner"] == (
                next(iter(winners)) if len(winners) == 1 else "SPLIT"
            )

    def test_the_winner_column_follows_the_sign_of_delta(self, summary):
        for row in summary.iter_rows(named=True):
            for origin in ORIGINS:
                delta = row[f"delta_mae_{origin}"]
                expected = "LSTM" if delta < 0 else "PERSIST"
                assert row[f"winner_{origin}"] == expected, (origin, row)

    def test_agreement_is_recomputable_from_the_detail_table(self, table, summary):
        assert agreement(table).equals(summary)


class TestTheDocumentQuotesTheTable:
    """Every number in the prose must come off the CSV.

    The section is transcribed by hand, which is exactly how a table and its
    narration drift apart: the CSV is regenerated, the markdown is not, and the
    document keeps quoting numbers no artifact produces any more.
    """

    DOC = REPO_ROOT / "docs" / "resultados" / "documento-resultados.md"
    HEADING = "### ¿Y si el mes fuera otro?"

    @pytest.fixture(scope="class")
    def section(self) -> str:
        text = self.DOC.read_text(encoding="utf-8")
        assert self.HEADING in text, f"{self.HEADING!r} is gone from the document"
        return text.split(self.HEADING)[1].split("## 5.")[0]

    def test_every_delta_in_the_table_matches_the_csv(self, section, summary):
        import re

        pattern = re.compile(
            r"\|\s*\*{0,2}(E\d+) h=(\d+)\*{0,2}\s*\|"
            r"\s*\*{0,2}([+−-][\d.]+)\*{0,2}\s*\|"
            r"\s*\*{0,2}([+−-][\d.]+)\*{0,2}\s*\|"
            r"\s*\*{0,2}([+−-][\d.]+)\*{0,2}\s*\|"
            r"\s*\*{0,2}(sí|no)\*{0,2}\s*\|"
        )
        seen = 0
        for corridor, horizon, *deltas, agrees in pattern.findall(section):
            cell = summary.filter(
                (pl.col("corridor") == corridor)
                & (pl.col("horizon") == int(horizon))
            ).to_dicts()
            assert cell, f"document quotes {corridor} h={horizon}, absent from the CSV"
            cell = cell[0]
            for origin, quoted in zip(ORIGINS, deltas):
                # The document writes minus as U+2212 and marks wins with '+'.
                value = float(quoted.replace("−", "-").lstrip("+"))
                assert abs(cell[f"delta_mae_{origin}"] - value) < 5e-4, (
                    f"{corridor} h={horizon} @{origin}: document says {value}, "
                    f"CSV says {cell[f'delta_mae_{origin}']}"
                )
            assert (agrees == "sí") == cell["agrees"], (corridor, horizon)
            seen += 1
        assert seen == summary.height, (
            f"document tabulates {seen} cells, the CSV has {summary.height}"
        )

    def test_the_headline_count_is_not_stale(self, section, summary):
        n_agree = int(summary.get_column("agrees").sum())
        assert f"{n_agree} de las {summary.height} celdas" in section, (
            f"{n_agree} of {summary.height} cells agree; the section does not "
            "open with that count"
        )

    def test_the_long_horizon_count_is_not_stale(self, section, table):
        """The 'h>=5 holds everywhere' claim counts cells, and the count is
        corridors x horizons x origins — easy to quote as one horizon's worth."""
        n_cells = table.filter(
            (pl.col("metric") == "MAE") & (pl.col("horizon") >= 5)
        ).height
        assert f"{n_cells} celdas" in section, (
            f"h>=5 covers {n_cells} cells; the section does not say so"
        )


class TestWhatTheTableSaysAboutTheClaim:
    """Behaviour, not formatting: the findings the paper would cite."""

    def test_the_long_horizon_advantage_holds_at_every_origin(self, table):
        """h=10 is where the LSTM's case is strongest — it must not depend on
        the published month."""
        long_h = table.filter(
            (pl.col("metric") == "MAE") & (pl.col("horizon") == 10)
        )
        assert long_h.height == len(CORRIDORS) * len(ORIGINS)
        assert (long_h.get_column("delta_mae") < 0).all()
        assert (long_h.get_column("dm_p_clustered") < 0.05).all()

    def test_persistence_keeps_the_shortest_horizon_at_every_origin(self, table):
        """The other half of the crossover: at h=1 the LSTM never wins."""
        short_h = table.filter(
            (pl.col("metric") == "MAE") & (pl.col("horizon") == 1)
        )
        assert (short_h.get_column("delta_mae") > 0).all()

    def test_the_crossover_is_a_transition_not_a_step(self, table):
        """delta_mae must fall monotonically with the horizon in every
        (corridor, origin): the crossover is the horizon axis doing the work,
        and a non-monotone corridor would mean something else is."""
        mae = table.filter(pl.col("metric") == "MAE")
        for corridor in CORRIDORS:
            for origin in ORIGINS:
                deltas = (
                    mae.filter(
                        (pl.col("corridor") == corridor)
                        & (pl.col("origin") == origin)
                    )
                    .sort("horizon")
                    .get_column("delta_mae")
                    .to_list()
                )
                assert deltas == sorted(deltas, reverse=True), (
                    f"{corridor}@{origin} is not monotone in the horizon: {deltas}"
                )
