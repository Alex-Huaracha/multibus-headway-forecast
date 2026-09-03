"""Contract for the architecture-ablation table.

Section V of the paper justifies carrying a flat LSTM for a vector-valued task,
and the justification is that two architectures which model the relation between
neighbouring positions did not move the scalar error. Those runs belong to the
frozen generation, so the table exists to keep their figures separated from the
recertified ones and to keep them off a hand-typed note.

The pooled row is the load-bearing detail. Each frozen results file carries one
row per direction plus an ``aggregate`` row, and averaging all three double
counts the corridor. These tests pin that the table reads the pooled row.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.build_paper_tables import (
    ARCHITECTURES,
    AblationError,
    ablation_rows,
    tabla_6,
)


def _row(
    corridor: str,
    direction: str,
    architecture: str,
    horizon: int,
    mae: float,
) -> dict:
    return {
        "corridor": corridor,
        "direction": direction,
        "baseline": architecture,
        "metric": "MAE",
        "value": mae,
        "horizon": horizon,
    }


def _cell(corridor: str, horizon: int, architecture: str, *, pooled: float) -> list[dict]:
    """One corridor x horizon x architecture, with its two directions."""
    return [
        _row(corridor, "-1", architecture, horizon, pooled + 1.0),
        _row(corridor, "+1", architecture, horizon, pooled - 1.0),
        _row(corridor, "aggregate", architecture, horizon, pooled),
    ]


def _frame(cells: list[list[dict]]) -> pl.DataFrame:
    return pl.DataFrame([row for cell in cells for row in cell])


def _full(corridor: str, horizon: int, maes: tuple[float, float, float]) -> list[list[dict]]:
    return [
        _cell(corridor, horizon, name, pooled=mae)
        for (name, _), mae in zip(ARCHITECTURES, maes)
    ]


class TestAblationRows:
    def test_reads_the_pooled_row_and_not_the_per_direction_ones(self) -> None:
        """Averaging the three rows would report 4.0 here, not 4.0 by luck."""
        frame = _frame(_full("E2", 10, (4.0, 5.0, 6.0)))
        rows = ablation_rows(frame, corridors=("E2",), horizons=(10,))
        assert rows[0][2:5] == ["4,000", "5,000", "6,000"]

    def test_one_row_per_corridor_and_horizon(self) -> None:
        cells = []
        for corridor in ("E2", "E4"):
            for horizon in (1, 10):
                cells += _full(corridor, horizon, (4.0, 4.1, 4.2))
        rows = ablation_rows(_frame(cells), corridors=("E2", "E4"), horizons=(1, 10))
        assert len(rows) == 4
        assert [row[0] for row in rows] == ["E2", "E2", "E4", "E4"]
        assert [row[1] for row in rows] == ["1", "10", "1", "10"]

    def test_the_spread_is_the_widest_gap_of_the_row(self) -> None:
        frame = _frame(_full("E2", 10, (4.0, 4.25, 4.1)))
        rows = ablation_rows(frame, corridors=("E2",), horizons=(10,))
        assert rows[0][5] == "0,250"

    def test_a_missing_architecture_is_refused(self) -> None:
        cells = _full("E2", 10, (4.0, 4.1, 4.2))
        frame = _frame(cells[:2])
        with pytest.raises(AblationError, match="SpatialTransformer"):
            ablation_rows(frame, corridors=("E2",), horizons=(10,))

    def test_a_missing_pooled_row_is_refused(self) -> None:
        """Falling back to a per-direction value would silently change the unit."""
        frame = _frame(_full("E2", 10, (4.0, 4.1, 4.2))).filter(
            pl.col("direction") != "aggregate"
        )
        with pytest.raises(AblationError, match="aggregate"):
            ablation_rows(frame, corridors=("E2",), horizons=(10,))


class TestTabla6:
    def test_carries_every_cell_and_architecture(self) -> None:
        table = tabla_6()
        for name, _ in ARCHITECTURES:
            assert name in table
        # Header, rule, and one row per corridor x horizon.
        assert len(table.splitlines()) == 2 + 12

    def test_matches_the_committed_frozen_results(self) -> None:
        """Every figure printed is the pooled MAE of its own results file."""
        from src.build_paper_tables import load_frozen_results

        frame = load_frozen_results()
        pooled = frame.filter(
            (pl.col("direction") == "aggregate") & (pl.col("metric") == "MAE")
        )
        table = tabla_6()
        for row in pooled.iter_rows(named=True):
            printed = f"{row['value']:.3f}".replace(".", ",")
            assert printed in table, f"{row['corridor']} h{row['horizon']} missing"
