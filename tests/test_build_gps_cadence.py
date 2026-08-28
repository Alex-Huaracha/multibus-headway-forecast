"""Contract for the GPS cadence builder.

Section IV-A opens by claiming the fleet reports every 20 seconds and that the
cadence is regular, evidenced by a median and a 95th percentile that coincide.
That claim traced only to a phase note and a notebook line, neither of which is
a source of truth the manuscript is allowed to quote. These tests pin how the
gap between consecutive pings is counted.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.build_gps_cadence import CadenceError, cadence_row


def _pings(rows: list[tuple[int, int]]) -> pl.DataFrame:
    """(bus, second offset) pairs, as the cleaned GPS frame carries them."""
    base = datetime(2023, 10, 1)
    return pl.DataFrame(
        {
            "unidadid": [bus for bus, _ in rows],
            "t": [base + timedelta(seconds=offset) for _, offset in rows],
        }
    )


class TestCadenceRow:
    def test_gaps_are_measured_within_each_bus(self) -> None:
        """Two buses parked at different hours must not produce a gap between
        them; that phantom gap would be the corridor's idle time, not a cadence."""
        frame = _pings([(1, 0), (1, 20), (1, 40), (2, 1000), (2, 1020)])

        row = cadence_row(frame, "E2")

        assert row["n_buses"] == 2
        assert row["n_gaps"] == 3
        assert row["median_gap_s"] == 20.0
        assert row["p95_gap_s"] == 20.0

    def test_the_fleet_is_counted_off_the_same_frame(self) -> None:
        """Section IV-A quotes the fleet size next to the cadence, so it comes
        from the same measurement rather than from a separate hand count."""
        frame = _pings([(7, 0), (7, 20), (9, 0), (9, 20), (9, 40)])

        assert cadence_row(frame, "E59")["n_buses"] == 2

    def test_input_order_does_not_matter(self) -> None:
        shuffled = _pings([(1, 40), (2, 1020), (1, 0), (2, 1000), (1, 20)])

        ordered = _pings([(1, 0), (1, 20), (1, 40), (2, 1000), (2, 1020)])

        assert cadence_row(shuffled, "E4") == cadence_row(ordered, "E4")

    def test_a_rare_long_gap_does_not_move_the_percentile(self) -> None:
        """One overnight break per bus per day sits far below the 95th
        percentile of a 20-second cadence, and the paper's claim depends on it."""
        rows = [(1, 20 * i) for i in range(21)] + [(1, 20 * 20 + 3600)]

        row = cadence_row(_pings(rows), "E59")

        assert row["median_gap_s"] == 20.0
        assert row["p95_gap_s"] == 20.0

    def test_emissions_per_minute_is_derived_from_the_median(self) -> None:
        frame = _pings([(1, 0), (1, 20), (1, 40)])

        assert cadence_row(frame, "E2")["emissions_per_minute"] == 3.0

    def test_missing_column_is_refused(self) -> None:
        with pytest.raises(CadenceError, match="unidadid"):
            cadence_row(pl.DataFrame({"t": [datetime(2023, 10, 1)]}), "E2")

    def test_empty_frame_is_refused(self) -> None:
        with pytest.raises(CadenceError, match="no rows"):
            cadence_row(_pings([]), "E2")

    def test_a_frame_with_no_consecutive_pings_is_refused(self) -> None:
        """One ping per bus yields no interval at all, so there is no cadence to
        report — returning a null would put an empty cell in the manuscript."""
        with pytest.raises(CadenceError, match="no consecutive"):
            cadence_row(_pings([(1, 0), (2, 500)]), "E2")
