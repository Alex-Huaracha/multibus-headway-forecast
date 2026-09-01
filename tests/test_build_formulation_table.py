"""Contract for the headway-formulation comparison table.

Section III-A of the paper states that the adopted headway definition was picked
over three candidates, and quotes the two dimensions that decided the discards.
Those figures came out of the ``03-headway-viability`` probe and used to live
only in a hand-typed note, which is the failure mode ``build_paper_tables``
exists to prevent. These tests pin the pivot from the probe's per-corridor rows
and the figures the manuscript quotes.
"""

from __future__ import annotations

import polars as pl
import pytest

import src.build_paper_tables as build_paper_tables
from src.build_paper_tables import FormulationError, formulation_rows


def _probe(rows: list[dict]) -> pl.DataFrame:
    """A minimal probe matrix: one row per (formulation, corridor)."""
    return pl.DataFrame(rows)


def _row(
    formulation: str,
    empresa: int,
    *,
    autocorr: float,
    mi: float,
    passed: int,
) -> dict:
    return {
        "formulation": formulation,
        "empresa": empresa,
        "autocorr_5min": autocorr,
        "mi_bits": mi,
        "pass_count_total": passed,
    }


def _both(formulation: str, **kwargs) -> list[dict]:
    return [
        _row(formulation, 2, **kwargs),
        _row(formulation, 59, **kwargs),
    ]


class TestFormulationRows:
    def test_pivots_the_two_corridors_onto_one_row(self) -> None:
        frame = _probe(
            [
                _row("C2", 2, autocorr=0.313, mi=0.358, passed=6),
                _row("C2", 59, autocorr=0.603, mi=1.256, passed=6),
            ]
        )

        assert formulation_rows(frame, ("C2",)) == [
            {
                "formulation": "C2",
                "autocorr_e2": 0.313,
                "autocorr_e59": 0.603,
                "mi_e2": 0.358,
                "mi_e59": 1.256,
                "passed": 6,
            }
        ]

    def test_formulations_come_out_in_declared_order(self) -> None:
        frame = _probe(
            _both("C2", autocorr=0.3, mi=0.3, passed=6)
            + _both("A", autocorr=0.1, mi=0.1, passed=5)
        )

        rows = formulation_rows(frame, ("A", "C2"))

        assert [row["formulation"] for row in rows] == ["A", "C2"]

    def test_missing_formulation_is_refused(self) -> None:
        """A silently absent candidate would turn four compared definitions
        into three without changing a word of the manuscript."""
        frame = _probe(_both("C2", autocorr=0.3, mi=0.3, passed=6))

        with pytest.raises(FormulationError, match="A"):
            formulation_rows(frame, ("A", "C2"))

    def test_missing_corridor_is_refused(self) -> None:
        frame = _probe([_row("C2", 2, autocorr=0.3, mi=0.3, passed=6)])

        with pytest.raises(FormulationError, match="59"):
            formulation_rows(frame, ("C2",))

    def test_disagreeing_pass_counts_are_refused(self) -> None:
        """The table carries one pass count per formulation. If the corridors
        disagree, collapsing them would report a number neither one measured."""
        frame = _probe(
            [
                _row("C2", 2, autocorr=0.3, mi=0.3, passed=6),
                _row("C2", 59, autocorr=0.3, mi=0.3, passed=5),
            ]
        )

        with_disagreement = pytest.raises(FormulationError, match="pass_count")
        with with_disagreement:
            formulation_rows(frame, ("C2",))

    def test_missing_column_is_refused(self) -> None:
        frame = pl.DataFrame({"formulation": ["C2"], "empresa": [2]})

        with pytest.raises(FormulationError, match="autocorr_5min"):
            formulation_rows(frame, ("C2",))


class TestTabla5:
    def test_figures_are_read_from_the_probe(self, monkeypatch) -> None:
        """A hardcoded cell would survive a re-run of the probe in silence."""
        monkeypatch.setattr(
            build_paper_tables,
            "_load",
            lambda name: _probe(
                _both("A", autocorr=0.5, mi=0.5, passed=7)
                + _both("B", autocorr=0.5, mi=0.5, passed=7)
                + _both("C1", autocorr=0.5, mi=0.5, passed=7)
                + _both("C2", autocorr=0.125, mi=0.5, passed=7)
            ),
        )

        assert "0,125" in build_paper_tables.tabla_5()

    def test_matches_the_committed_probe(self) -> None:
        """Section III-A quotes these eight figures; the CSV owns them.

        If the probe is ever re-run, this test fails and the manuscript has to
        be updated with it, instead of drifting out of sync unnoticed.
        """
        table = build_paper_tables.tabla_5()

        # The adopted definition: the two dimensions that decided the discards.
        assert "0,313" in table and "0,603" in table
        assert "0,358" in table and "1,256" in table
        # Virtual points on the axis, discarded on autocorrelation.
        assert "0,167" in table and "-0,005" in table
        # Forward projection, discarded on neighbour mutual information.
        assert "0,226" in table and "0,326" in table

    def test_carries_the_four_candidates(self) -> None:
        table = build_paper_tables.tabla_5()

        assert table.count("\n") == 5  # header, rule, four rows
        assert "adoptada" in table
