"""The step between the builder and the manuscript, which nothing used to check.

``src/build_paper_tables.py`` writes ``docs/paper/tablas/tabla-N-*.md`` and its
own docstring records what happens next: the tables are *pasted* into
``docs/paper/paper.md``. Nothing in ``src/`` writes the manuscript, so that paste
is a manual step, and a manual step with no test is a step that drifts.

It drifted. Table 2 was restructured by hand — a positional-floor column and a
Matthews interval were added, a "winner" column was dropped — and ``tabla_2``
was left emitting the old eight-column shape. The manuscript and its builder
disagreed on every one of the twelve rows, and nothing said so.

These tests cover two things and deliberately stop there. The paste has to be
faithful, and every figure printed has to be the one its CSV published. How the
table marks its winners is presentation, it does not survive retypesetting into
the venue's template unchanged, and pinning it would cost more than it protects.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.build_paper_tables import (
    CORRIDORS,
    HORIZONS,
    _load,
    tabla_1,
    tabla_2,
    tabla_3,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER = REPO_ROOT / "docs" / "paper" / "paper.md"

# tabla_3 (robustez) is no longer pasted: the manuscript replaced it with the
# V-C paragraph, so only its shape is checked below, not its paste.
PASTED_TABLES = [("Tabla 1", tabla_1), ("Tabla 2", tabla_2)]
ALL_TABLES = PASTED_TABLES + [("robustez", tabla_3)]


@pytest.fixture(scope="module")
def paper() -> str:
    # The builders emit LF; the manuscript is stored CRLF. Comparing raw bytes
    # would fail on the line ending and say nothing about the figures.
    return PAPER.read_text(encoding="utf-8").replace("\r\n", "\n")


def _rows(table: str) -> list[str]:
    """Data rows only: the header and the alignment rule carry no figures."""
    return [line for line in table.splitlines()[2:] if line.startswith("|")]


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


class TestTheManuscriptCarriesWhatTheBuilderEmits:
    """The paste is faithful, or the test names which table stopped being so."""

    @pytest.mark.parametrize(("name", "builder"), PASTED_TABLES)
    def test_every_emitted_row_appears_verbatim(self, name, builder, paper) -> None:
        missing = [row for row in _rows(builder()) if row not in paper]
        assert not missing, (
            f"{name}: {len(missing)} de sus filas no están en paper.md. "
            f"Corre `uv run python -m src.build_paper_tables` y vuelve a pegarla. "
            f"Primera: {missing[0]}"
        )

    @pytest.mark.parametrize(("name", "builder"), PASTED_TABLES)
    def test_the_header_appears_verbatim(self, name, builder, paper) -> None:
        """A column added on one side only is the drift that already happened."""
        header = builder().splitlines()[0]
        assert header in paper, f"{name}: la cabecera pegada no es la que emite el builder"

    @pytest.mark.parametrize(("name", "builder"), ALL_TABLES)
    def test_each_table_carries_one_row_per_cell(self, name, builder) -> None:
        assert len(_rows(builder())) == len(CORRIDORS) * len(HORIZONS)


class TestTablaDosPrintsWhatItsSourcesPublished:
    """Every figure in the new columns traces to the CSV that produced it."""

    @pytest.fixture(scope="class")
    def rows(self) -> list[list[str]]:
        return [_cells(row) for row in _rows(tabla_2())]

    def test_the_floor_is_the_figure_the_null_builder_published(self, rows) -> None:
        printed = {
            (cells[0], int(cells[1])): cells[4].replace("**", "").split("&nbsp;")[0]
            for cells in rows
        }
        for row in _load("positional_null.csv").iter_rows(named=True):
            key = (row["corridor"], row["horizon"])
            assert printed[key] == f"{row['auc_null']:.3f}", key

    def test_every_interval_is_the_one_the_bootstrap_published(self, rows) -> None:
        printed = {(cells[0], int(cells[1])): (cells[5], cells[8]) for cells in rows}
        intervals = _load("detection_ranking_ci.csv").filter(pl.col("origin") == "main")
        for row in intervals.iter_rows(named=True):
            key = (row["corridor"], row["horizon"])
            for band, columns in (
                (printed[key][0], ("delta_auc", "auc_ci_low", "auc_ci_high")),
                (
                    printed[key][1],
                    (
                        "delta_mcc_calibrated",
                        "mcc_calibrated_ci_low",
                        "mcc_calibrated_ci_high",
                    ),
                ),
            ):
                for column in columns:
                    figure = f"{row[column]:+.3f}"
                    if float(figure) == 0.0:
                        figure = f"{0.0:.3f}"
                    assert figure in band, (key, column)
