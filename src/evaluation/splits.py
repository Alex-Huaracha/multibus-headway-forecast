"""Temporal split and winsorization helpers for headway evaluation — Fase 3.

Public API:
    split_temporal(df: pl.DataFrame, fold: Fold = MAIN_FOLD) -> pl.DataFrame
    winsorize_train_p99(df: pl.DataFrame) -> tuple[pl.DataFrame, float]
    Fold, MAIN_FOLD, ROLLING_FOLDS, fold_by_name

Constants (split date ranges, locked in spec §3 and design §5):
    SPLIT_TRAIN_START, SPLIT_TRAIN_END
    SPLIT_VAL_START,   SPLIT_VAL_END
    SPLIT_TEST_START,  SPLIT_TEST_END
    WINSOR_QUANTILE

Design decisions (locked in design §5 and §9):
  - Split key is pl.col("t").dt.date() membership, NOT row index.
  - Three ranges are exhaustive and mutually exclusive.
  - Rows outside all three ranges receive None (split column = null).
  - Winsorization threshold is computed on train rows only (AC-WINSOR-1, AC-WINSOR-2).
  - Null delta_t_min rows are NOT clipped (AC-WINSOR-3).
  - Rows above threshold are clipped (not dropped) (AC-WINSOR-4).
  - Constants live here (not PRODUCTIVE_PARAMS) — evaluation protocol concern.
  - WINSOR_QUANTILE and split dates are not added to pyproject.toml.

Rolling origin
--------------
Every published result rests on ONE test window of 22 days (February 2024), so
nothing distinguishes "the method works" from "those 22 days happened to
cooperate" — the standard objection to a forecasting evaluation, and the one the
results document still declares open.

``ROLLING_FOLDS`` answers it by re-running the whole protocol at three origins,
each with its own 22-day test window. The window **expands** rather than slides:
every fold trains from the first day of data up to its own cutoff. That is the
usual "rolling origin with recalibration" design, and it has a second payoff
here — because the folds differ in training length as well as in period, a
result that holds across all three is robust to both.

The last fold is **exactly** the main split. That is deliberate: it makes the
published result the final origin of the sequence rather than a separate
analysis, and it means only two additional folds need training.

Fold boundaries are contiguous by construction: one fold's test window becomes
the next fold's validation window. No fold ever sees its own test period during
training, which is the only property that matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

# ---------------------------------------------------------------------------
# Split date range constants (spec §3, inclusive on both ends)
# ---------------------------------------------------------------------------

SPLIT_TRAIN_START: date = date(2023, 10, 1)
SPLIT_TRAIN_END:   date = date(2024, 1, 15)

SPLIT_VAL_START:   date = date(2024, 1, 16)
SPLIT_VAL_END:     date = date(2024, 2, 7)

SPLIT_TEST_START:  date = date(2024, 2, 8)
SPLIT_TEST_END:    date = date(2024, 2, 29)

WINSOR_QUANTILE: float = 0.99


@dataclass(frozen=True)
class Fold:
    """One evaluation origin: three contiguous, ordered date ranges.

    Validated on construction. The invariants are not stylistic — a gap between
    ranges would silently drop days, and an overlap would train a model on its
    own test period, which is the failure this whole class exists to make
    impossible to introduce by hand.
    """

    name: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        for label, start, end in (
            ("train", self.train_start, self.train_end),
            ("val", self.val_start, self.val_end),
            ("test", self.test_start, self.test_end),
        ):
            if start > end:
                raise ValueError(
                    f"fold {self.name!r}: {label} range is inverted "
                    f"({start} > {end})"
                )
        day = timedelta(days=1)
        if self.val_start != self.train_end + day:
            raise ValueError(
                f"fold {self.name!r}: val must start the day after train ends; "
                f"train ends {self.train_end}, val starts {self.val_start}"
            )
        if self.test_start != self.val_end + day:
            raise ValueError(
                f"fold {self.name!r}: test must start the day after val ends; "
                f"val ends {self.val_end}, test starts {self.test_start}"
            )

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days + 1

    @property
    def val_days(self) -> int:
        return (self.val_end - self.val_start).days + 1

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days + 1

    def bounds(self) -> dict[str, tuple[date, date]]:
        """Split-name → (start, end), the shape the builders iterate."""
        return {
            "train": (self.train_start, self.train_end),
            "val": (self.val_start, self.val_end),
            "test": (self.test_start, self.test_end),
        }


#: The published split. Every frozen digest and every committed table is keyed
#: to it, so its dates must keep matching the module constants above.
MAIN_FOLD = Fold(
    name="main",
    train_start=SPLIT_TRAIN_START, train_end=SPLIT_TRAIN_END,
    val_start=SPLIT_VAL_START,     val_end=SPLIT_VAL_END,
    test_start=SPLIT_TEST_START,   test_end=SPLIT_TEST_END,
)

#: Three origins, oldest first. Each test window is 22 days, matching the main
#: split, so the folds differ in WHEN they are evaluated and in how much history
#: they were given — not in how much evidence each verdict rests on.
#:
#: Christmas and New Year land inside fold 1's test window and fold 2's
#: validation window. That is not a flaw to design around: if the result depends
#: on the holiday period, this is the analysis that has to reveal it.
ROLLING_FOLDS: tuple[Fold, ...] = (
    Fold(
        name="r1",
        train_start=date(2023, 10, 1), train_end=date(2023, 11, 30),
        val_start=date(2023, 12, 1),   val_end=date(2023, 12, 22),
        test_start=date(2023, 12, 23), test_end=date(2024, 1, 13),
    ),
    Fold(
        name="r2",
        train_start=date(2023, 10, 1), train_end=date(2023, 12, 22),
        val_start=date(2023, 12, 23),  val_end=date(2024, 1, 13),
        test_start=date(2024, 1, 14),  test_end=date(2024, 2, 4),
    ),
    MAIN_FOLD,
)


def fold_by_name(name: str) -> Fold:
    """Look up a fold by name, failing loudly on a typo.

    Builders take the fold as a string (CLI argument, notebook parameter), and a
    silent fallback to the main fold would produce results labelled as one origin
    and computed on another.
    """
    for fold in ROLLING_FOLDS:
        if fold.name == name:
            return fold
    known = ", ".join(fold.name for fold in ROLLING_FOLDS)
    raise KeyError(f"unknown fold {name!r}; known folds: {known}")


def split_temporal(df: pl.DataFrame, fold: Fold = MAIN_FOLD) -> pl.DataFrame:
    """Add a `split` column (Utf8) with values {"train", "val", "test"}.

    Membership is determined by pl.col("t").dt.date() against ``fold``'s six
    dates. Rows outside all three ranges receive null — expected for the rolling
    folds, whose windows end before the data does, and not expected for the main
    fold (the harness raises if found there).

    Parameters
    ----------
    df:
        headways DataFrame containing at least a `t` (Datetime) column.
    fold:
        Evaluation origin. Defaults to :data:`MAIN_FOLD`, so every existing
        caller keeps its exact behaviour and every frozen digest stays valid.

    Returns
    -------
    pl.DataFrame — input frame with one added column `split: Utf8`.
    """
    day = pl.col("t").dt.date()
    return df.with_columns(
        pl.when((day >= fold.train_start) & (day <= fold.train_end))
          .then(pl.lit("train"))
          .when((day >= fold.val_start) & (day <= fold.val_end))
          .then(pl.lit("val"))
          .when((day >= fold.test_start) & (day <= fold.test_end))
          .then(pl.lit("test"))
          .otherwise(None)
          .alias("split")
    )


def winsorize_train_p99(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, float]:
    """Clip delta_t_min to the 99th-percentile threshold computed on train rows only.

    The threshold is computed once as a scalar from non-null train-split rows.
    It is then applied as a clip ceiling to ALL rows (train + val + test).
    Null delta_t_min values are never clipped — they remain null (AC-WINSOR-3).

    Parameters
    ----------
    df:
        headways DataFrame that already has a `split` column (added by
        split_temporal) and a `delta_t_min` (Float64 nullable) column.

    Returns
    -------
    (clipped_df, threshold)
        clipped_df: same schema as df, delta_t_min clipped.
        threshold: the scalar train-p99 value used as the clip ceiling.

    Design note (AC-WINSOR-2 leakage guard):
        The filter `split == "train"` is applied BEFORE computing the quantile,
        so extreme outliers in val or test rows cannot shift the threshold.
    """
    threshold = float(
        df.filter(
            (pl.col("split") == "train") & pl.col("delta_t_min").is_not_null()
        )["delta_t_min"]
        .quantile(WINSOR_QUANTILE)
    )

    # Clip: preserve null rows; clip non-null rows to threshold from above.
    # pl.min_horizontal(col, lit(threshold)) would coerce null → 0 in some
    # polars versions, so we use the explicit when/then pattern (design §5).
    clipped = df.with_columns(
        pl.when(pl.col("delta_t_min").is_null())
          .then(None)
          .otherwise(
              pl.min_horizontal(pl.col("delta_t_min"), pl.lit(threshold))
          )
          .alias("delta_t_min")
    )
    return clipped, threshold
