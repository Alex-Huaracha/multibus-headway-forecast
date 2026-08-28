"""Contract for the headway-coverage builder.

Section IV-A of the paper reports what fraction of the evaluable bus pairs
actually produced a headway. That number used to live only in prose, which is
exactly the failure mode ``build_paper_tables`` exists to prevent: a figure in
the manuscript with no regenerable source. These tests pin the counting rule and
the fact that the percentage is derived from counts, never typed.
"""

from __future__ import annotations

import polars as pl
import pytest

import src.build_paper_tables as build_paper_tables
from src.build_headway_coverage import (
    CoverageError,
    aggregate_coverage,
    coverage_rows,
)


def _pairs(rows: list[tuple[int, float | None]]) -> pl.DataFrame:
    """A minimal headway frame: one row per bus pair, plus its headway."""
    return pl.DataFrame(
        {
            "direction": [direction for direction, _ in rows],
            "delta_t_min": [delta for _, delta in rows],
        },
        schema={"direction": pl.Int64, "delta_t_min": pl.Float64},
    )


class TestCoverageRows:
    def test_counts_valid_against_every_pair_attempted(self) -> None:
        frame = _pairs([(-1, 4.0), (-1, None), (-1, 6.5), (1, 2.0)])

        rows = coverage_rows(frame, "E2")

        assert rows == [
            {"corridor": "E2", "direction": -1, "valid_pairs": 2, "total_pairs": 3},
            {"corridor": "E2", "direction": 1, "valid_pairs": 1, "total_pairs": 1},
        ]

    def test_null_headway_stays_in_the_denominator(self) -> None:
        """A pair with no crossing is a pair that was attempted and failed.

        Dropping it from the denominator would report 100 % coverage for a
        corridor that answered nothing, which is the opposite of the claim.
        """
        frame = _pairs([(1, None), (1, None)])

        assert coverage_rows(frame, "E4") == [
            {"corridor": "E4", "direction": 1, "valid_pairs": 0, "total_pairs": 2}
        ]

    def test_directions_come_out_in_ascending_order(self) -> None:
        frame = _pairs([(1, 3.0), (-1, 3.0), (1, 3.0)])

        assert [row["direction"] for row in coverage_rows(frame, "E59")] == [-1, 1]

    def test_direction_zero_is_refused(self) -> None:
        """``compute_pairs`` drops direction 0, so its presence means the frame
        is not a pair table and the denominator would be inflated in silence."""
        frame = _pairs([(0, 3.0), (1, 3.0)])

        with pytest.raises(CoverageError, match="direction 0"):
            coverage_rows(frame, "E2")

    def test_missing_column_is_refused(self) -> None:
        frame = pl.DataFrame({"direction": [1]})

        with pytest.raises(CoverageError, match="delta_t_min"):
            coverage_rows(frame, "E2")

    def test_empty_frame_is_refused(self) -> None:
        with pytest.raises(CoverageError, match="no rows"):
            coverage_rows(_pairs([]), "E2")


class TestAggregateCoverage:
    def test_sums_directions_and_derives_the_percentage(self) -> None:
        frame = pl.DataFrame(
            [
                {"corridor": "E2", "direction": -1, "valid_pairs": 3, "total_pairs": 4},
                {"corridor": "E2", "direction": 1, "valid_pairs": 1, "total_pairs": 4},
            ]
        )

        aggregate = aggregate_coverage(frame)

        assert aggregate.to_dicts() == [
            {
                "corridor": "E2",
                "valid_pairs": 4,
                "total_pairs": 8,
                "coverage_pct": 50.0,
            }
        ]

    def test_corridors_come_out_in_declared_order(self) -> None:
        frame = pl.DataFrame(
            [
                {"corridor": "E59", "direction": 1, "valid_pairs": 1, "total_pairs": 2},
                {"corridor": "E2", "direction": 1, "valid_pairs": 1, "total_pairs": 2},
                {"corridor": "E4", "direction": 1, "valid_pairs": 1, "total_pairs": 2},
            ]
        )

        assert aggregate_coverage(frame)["corridor"].to_list() == ["E2", "E4", "E59"]

    def test_zero_denominator_is_refused(self) -> None:
        frame = pl.DataFrame(
            [{"corridor": "E2", "direction": 1, "valid_pairs": 0, "total_pairs": 0}]
        )

        with pytest.raises(CoverageError, match="no pairs"):
            aggregate_coverage(frame)


class TestTabla4:
    def test_percentage_is_derived_from_the_counts(self, monkeypatch) -> None:
        """A hardcoded percentage would survive a change of corpus in silence."""
        monkeypatch.setattr(
            build_paper_tables,
            "_load",
            lambda name: pl.DataFrame(
                [
                    {
                        "corridor": "E2",
                        "direction": 1,
                        "valid_pairs": 1,
                        "total_pairs": 8,
                    }
                ]
            ),
        )

        assert "12,5 %" in build_paper_tables.tabla_4()

    def test_matches_the_committed_counts(self) -> None:
        """The manuscript quotes these three figures; the CSV owns them.

        If the corpus is ever rebuilt, this test fails and Section IV-A has to
        be updated with it, instead of drifting out of sync unnoticed.
        """
        table = build_paper_tables.tabla_4()

        assert "| E2 |" in table and "63,5 %" in table
        assert "| E4 |" in table and "64,8 %" in table
        assert "| E59 |" in table and "77,1 %" in table
