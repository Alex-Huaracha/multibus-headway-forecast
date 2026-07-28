"""Tests for the rolling-origin fold definitions.

Two things have to hold. First, that adding folds changed nothing for the
published result — every frozen digest and committed table is keyed to the main
split, so a shifted boundary would invalidate work already done and the tests
here are what makes that impossible to do silently.

Second, that the folds are what they claim: contiguous, ordered, and never
training on their own test period. Those invariants are enforced in ``Fold``
itself rather than left to whoever writes the next fold by hand.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.evaluation.splits import (
    MAIN_FOLD,
    ROLLING_FOLDS,
    SPLIT_TEST_END,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_TRAIN_START,
    SPLIT_VAL_END,
    SPLIT_VAL_START,
    Fold,
    fold_by_name,
    split_temporal,
)

DAY = timedelta(days=1)


def _frame_from_dates(days: list[date]) -> pl.DataFrame:
    """A `t` column at 08:00 on each given day.

    Built from ``datetime.datetime`` values, not ``pl.datetime`` — the latter is
    an EXPRESSION, and handing expressions to the DataFrame constructor yields an
    Object column that fails to cast.
    """
    import datetime as _dt

    return pl.DataFrame({"t": [_dt.datetime(d.year, d.month, d.day, 8) for d in days]})


def _frame(start: date, end: date) -> pl.DataFrame:
    """One row per day at 08:00 across an inclusive date range."""
    return _frame_from_dates(
        [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    )


class TestBackwardCompatibility:
    """The published split must be untouched."""

    def test_main_fold_matches_the_module_constants(self):
        assert MAIN_FOLD.train_start == SPLIT_TRAIN_START
        assert MAIN_FOLD.train_end == SPLIT_TRAIN_END
        assert MAIN_FOLD.val_start == SPLIT_VAL_START
        assert MAIN_FOLD.val_end == SPLIT_VAL_END
        assert MAIN_FOLD.test_start == SPLIT_TEST_START
        assert MAIN_FOLD.test_end == SPLIT_TEST_END

    def test_split_temporal_defaults_to_the_main_fold(self):
        frame = _frame_from_dates(
            [date(2023, 10, 1), date(2024, 1, 20), date(2024, 2, 15)]
        )
        assert split_temporal(frame).get_column("split").to_list() == [
            "train", "val", "test"
        ]

    def test_the_default_and_the_explicit_main_fold_agree(self):
        frame = _frame(date(2023, 10, 1), date(2024, 2, 29))
        assert (
            split_temporal(frame).get_column("split").to_list()
            == split_temporal(frame, MAIN_FOLD).get_column("split").to_list()
        )

    def test_the_main_fold_covers_every_day_of_the_dataset(self):
        """It has no null split — the property the harness relies on."""
        frame = _frame(date(2023, 10, 1), date(2024, 2, 29))
        assert split_temporal(frame).get_column("split").null_count() == 0


class TestFoldInvariants:
    def test_ranges_must_be_contiguous(self):
        with pytest.raises(ValueError, match="val must start the day after train"):
            Fold(
                name="gap",
                train_start=date(2024, 1, 1), train_end=date(2024, 1, 10),
                val_start=date(2024, 1, 12), val_end=date(2024, 1, 20),
                test_start=date(2024, 1, 21), test_end=date(2024, 1, 30),
            )

    def test_test_must_follow_validation(self):
        with pytest.raises(ValueError, match="test must start the day after val"):
            Fold(
                name="gap",
                train_start=date(2024, 1, 1), train_end=date(2024, 1, 10),
                val_start=date(2024, 1, 11), val_end=date(2024, 1, 20),
                test_start=date(2024, 1, 25), test_end=date(2024, 1, 30),
            )

    def test_an_inverted_range_is_rejected(self):
        with pytest.raises(ValueError, match="train range is inverted"):
            Fold(
                name="inverted",
                train_start=date(2024, 1, 10), train_end=date(2024, 1, 1),
                val_start=date(2024, 1, 2), val_end=date(2024, 1, 20),
                test_start=date(2024, 1, 21), test_end=date(2024, 1, 30),
            )

    def test_overlap_is_impossible_by_construction(self):
        """Contiguity plus ordering forbids overlap; assert it on every fold."""
        for fold in ROLLING_FOLDS:
            assert fold.train_end < fold.val_start
            assert fold.val_end < fold.test_start

    def test_day_counts_are_inclusive(self):
        fold = Fold(
            name="tiny",
            train_start=date(2024, 1, 1), train_end=date(2024, 1, 10),
            val_start=date(2024, 1, 11), val_end=date(2024, 1, 15),
            test_start=date(2024, 1, 16), test_end=date(2024, 1, 20),
        )
        assert (fold.train_days, fold.val_days, fold.test_days) == (10, 5, 5)


class TestRollingSequence:
    def test_there_are_three_origins_and_the_last_is_the_published_one(self):
        assert len(ROLLING_FOLDS) == 3
        assert ROLLING_FOLDS[-1] is MAIN_FOLD

    def test_origins_advance_in_time(self):
        starts = [fold.test_start for fold in ROLLING_FOLDS]
        assert starts == sorted(starts)
        assert len(set(starts)) == 3

    def test_test_windows_do_not_overlap(self):
        for earlier, later in zip(ROLLING_FOLDS, ROLLING_FOLDS[1:]):
            assert earlier.test_end < later.test_start

    def test_each_test_window_is_twenty_two_days(self):
        """Equal evidence per verdict, so a difference between folds is about
        the period and not about sample size."""
        assert {fold.test_days for fold in ROLLING_FOLDS} == {22}

    def test_training_expands_rather_than_slides(self):
        assert {fold.train_start for fold in ROLLING_FOLDS} == {date(2023, 10, 1)}
        lengths = [fold.train_days for fold in ROLLING_FOLDS]
        assert lengths == sorted(lengths)
        assert len(set(lengths)) == 3

    def test_the_synthetic_folds_chain_into_each_other(self):
        """r1's test window is r2's validation window.

        The chain stops at the published fold on purpose: its boundaries are
        frozen by every digest and table already committed, so r2 cannot be
        aligned to it without either moving the published split or starving r1's
        training window. Perfect chaining is cosmetic — what a rolling origin
        actually requires is asserted by the two tests below.
        """
        r1, r2 = ROLLING_FOLDS[0], ROLLING_FOLDS[1]
        assert r2.val_start == r1.test_start
        assert r2.val_end == r1.test_end

    def test_no_fold_trains_on_its_own_test_period(self):
        """The only invariant that would invalidate a result outright."""
        for fold in ROLLING_FOLDS:
            assert fold.train_end < fold.test_start

    def test_every_fold_fits_inside_the_dataset(self):
        first, last = date(2023, 10, 1), date(2024, 2, 29)
        for fold in ROLLING_FOLDS:
            assert fold.train_start >= first
            assert fold.test_end <= last


class TestSplitAssignment:
    def test_rolling_folds_leave_later_days_unassigned(self):
        """Expected, not a defect: r1's windows end in January."""
        frame = _frame(date(2023, 10, 1), date(2024, 2, 29))
        tagged = split_temporal(frame, ROLLING_FOLDS[0])
        assert tagged.get_column("split").null_count() > 0

    def test_each_fold_assigns_exactly_its_own_days(self):
        frame = _frame(date(2023, 10, 1), date(2024, 2, 29))
        for fold in ROLLING_FOLDS:
            tagged = split_temporal(frame, fold)
            counts = (
                tagged.group_by("split").len().sort("split").to_dict(as_series=False)
            )
            by_split = dict(zip(counts["split"], counts["len"]))
            assert by_split["train"] == fold.train_days
            assert by_split["val"] == fold.val_days
            assert by_split["test"] == fold.test_days

    def test_boundary_days_land_on_the_expected_side(self):
        for fold in ROLLING_FOLDS:
            frame = _frame_from_dates(
                [fold.train_end, fold.val_start, fold.val_end, fold.test_start]
            )
            assert split_temporal(frame, fold).get_column("split").to_list() == [
                "train", "val", "val", "test"
            ]


class TestLookup:
    def test_finds_every_declared_fold(self):
        for fold in ROLLING_FOLDS:
            assert fold_by_name(fold.name) is fold

    def test_a_typo_raises_instead_of_falling_back(self):
        """A silent fallback would label results as one origin and compute them
        on another — the exact class of defect this pipeline was rebuilt over."""
        with pytest.raises(KeyError, match="unknown fold"):
            fold_by_name("r3")

    def test_the_error_lists_the_known_names(self):
        with pytest.raises(KeyError) as excinfo:
            fold_by_name("nope")
        for fold in ROLLING_FOLDS:
            assert fold.name in str(excinfo.value)
