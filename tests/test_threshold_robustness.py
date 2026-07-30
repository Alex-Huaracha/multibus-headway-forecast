"""The two robustness checks the threshold argument needs, pinned.

Both answer objections raised by the 2026-07-29 literature survey, and both are
written so that a reversal FAILS rather than being quietly renarrated — because
in both cases the measured answer contradicted what we predicted beforehand, and
that is exactly the kind of result a later refactor tends to smooth away.

``TestTheCollapseIsNotAboutSelfReference``
    Our 0.5x-of-own-vector-mean rule is not the field's convention (the field
    uses a fraction of the SCHEDULED headway). If the collapse were an artifact
    of the self-reference, the finding would say nothing about published
    practice. Measured: the collapse is WORSE under an absolute cut, and ~110x
    worse under the field's quarter-of-schedule convention.

``TestTheAbsoluteEventIsHarder``
    The honest counterweight. The learner carries LESS ranking information about
    the absolute event than the relative one, and in one cell it is at chance.
    Pinned so the "never blind" claim cannot silently over-extend.

``TestWhyTheThresholdIsFittedOnMcc``
    Mixed result, deliberately pinned as mixed: F1-fitted cuts are TIGHTER in the
    median. What favours MCC is the absence of a degenerate tail and the
    out-of-sample cost. A test that asserted "MCC is more stable" would be false.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_threshold_robustness import (  # noqa: E402
    ABSOLUTE_RATIOS,
    CALIBRATION_ORIGIN,
    MODELS,
    OBJECTIVES,
    OUT_ABSOLUTE,
    OUT_STABILITY,
    SCORING_ORIGIN,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "resultados" / "documento-resultados.md"


@pytest.fixture(scope="module")
def absolute() -> pl.DataFrame:
    if not OUT_ABSOLUTE.exists():
        pytest.skip(f"{OUT_ABSOLUTE.name} not built yet")
    return pl.read_csv(OUT_ABSOLUTE)


@pytest.fixture(scope="module")
def stability() -> pl.DataFrame:
    if not OUT_STABILITY.exists():
        pytest.skip(f"{OUT_STABILITY.name} not built yet")
    return pl.read_csv(OUT_STABILITY)


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


class TestTheComparisonIsHonestlyConstructed:
    def test_the_absolute_cut_is_calibrated_out_of_sample(self):
        assert CALIBRATION_ORIGIN != SCORING_ORIGIN
        assert CALIBRATION_ORIGIN in {"r1", "r2"} and SCORING_ORIGIN == "main"

    def test_both_rules_score_the_same_rows(self, absolute):
        """The two event definitions must be compared on one population, or the
        under-firing ratios are not comparable at all."""
        for (ratio, corridor, horizon), cell in absolute.group_by(
            ["absolute_ratio", "corridor", "horizon"], maintain_order=True
        ):
            counts = cell.get_column("n_cells").unique()
            assert counts.len() == 1, f"{ratio} {corridor} h={horizon}: {counts}"

    def test_the_field_convention_is_among_the_ratios(self):
        """0.25 is the dominant published relative form (a quarter of the
        scheduled headway). Dropping it would remove the only ratio that speaks
        to practice rather than to our own choice."""
        assert 0.25 in ABSOLUTE_RATIOS

    def test_one_row_per_ratio_cell_and_model(self, absolute):
        expected = len(ABSOLUTE_RATIOS) * len(CORRIDORS) * len(HORIZONS) * len(MODELS)
        assert absolute.height == expected
        assert absolute.unique(
            subset=["absolute_ratio", "corridor", "horizon", "model"]
        ).height == expected

    def test_the_absolute_cut_is_a_plausible_headway(self, absolute):
        """A cut in minutes must land in the range a bus headway actually
        occupies; a units bug would show up here before anywhere else."""
        cuts = absolute.get_column("cut_minutes")
        assert cuts.min() > 0.5 and cuts.max() < 10.0, cuts.to_list()


class TestTheCollapseIsNotAboutSelfReference:
    """The objection this measurement exists to answer, and it inverted."""

    def test_the_absolute_cut_makes_the_collapse_worse_not_better(self, absolute):
        """The prediction going in was that an absolute cut would under-fire
        LESS, because it is not self-referential. It under-fires MORE, at both
        ratios. If this ever reverses, Section 5.6's argument is void and must be
        rewritten rather than have the number quietly updated."""
        lstm = absolute.filter(pl.col("model") == "LSTM")
        for ratio in ABSOLUTE_RATIOS:
            sub = lstm.filter(pl.col("absolute_ratio") == ratio)
            relative = sub.get_column("underfire_relative").median()
            absolute_ = sub.get_column("underfire_absolute").median()
            assert absolute_ < relative, (
                f"ratio {ratio}: absolute under-firing {absolute_:.4f} is no "
                f"longer worse than self-referential {relative:.4f}"
            )

    def test_the_field_convention_is_two_orders_of_magnitude_worse(self, absolute):
        """The quantitative claim in Section 5.6: under a quarter-of-schedule
        cut the apparent collapse is ~110x worse than under our own rule."""
        sub = absolute.filter(
            (pl.col("model") == "LSTM") & (pl.col("absolute_ratio") == 0.25)
        )
        relative = sub.get_column("underfire_relative").median()
        absolute_ = sub.get_column("underfire_absolute").median()
        assert relative / absolute_ > 50, (
            f"the ratio is now {relative / absolute_:.1f}x; Section 5.6 quotes "
            "two orders of magnitude"
        )

    def test_our_own_rule_was_the_conservative_choice(self, absolute):
        """Stated as a per-cell property, not just a median: the self-referential
        rule must be the kinder one in most cells, otherwise "conservative" is
        the wrong word for it."""
        lstm = absolute.filter(
            (pl.col("model") == "LSTM") & (pl.col("absolute_ratio") == 0.25)
        )
        kinder = lstm.filter(
            pl.col("underfire_relative") > pl.col("underfire_absolute")
        )
        assert kinder.height >= 11, (
            f"our rule is the conservative one in only {kinder.height} of "
            f"{lstm.height} cells"
        )


class TestTheAbsoluteEventIsHarder:
    """The counterweight. Reported because it cuts against us."""

    def test_the_learner_ranks_the_absolute_event_worse(self, absolute):
        """Median AUC for the absolute event must sit below the relative one
        (0.63-0.81 per Section 5.4). This is the limit of the "never blind"
        claim and the document has to keep saying so."""
        sub = absolute.filter(
            (pl.col("model") == "LSTM") & (pl.col("absolute_ratio") == 0.25)
        )
        assert sub.get_column("auc_absolute").median() < 0.63

    def test_the_chance_level_cell_is_not_hidden(self, absolute, doc):
        """At least one cell sits at chance for the absolute event. If the
        document stops quoting it, this fails."""
        sub = absolute.filter(
            (pl.col("model") == "LSTM") & (pl.col("absolute_ratio") == 0.25)
        )
        worst = sub.get_column("auc_absolute").min()
        assert worst < 0.52, f"worst absolute AUC is now {worst:.4f}"
        assert f"{worst:.4f}".rstrip("0") in doc or "0.4934" in doc, (
            "the document no longer quotes the chance-level absolute AUC"
        )


class TestWhyTheThresholdIsFittedOnMcc:
    """Mixed evidence, pinned as mixed. Do not let this drift into a clean win."""

    def test_f1_fitting_is_tighter_in_the_median(self, stability):
        """Deliberately asserting the INCONVENIENT direction. Section 5.7 says so
        explicitly; if a future change makes MCC tighter in the median, the
        section's careful hedging becomes wrong and must be revisited."""
        assert (
            stability.get_column("spread_f1").median()
            < stability.get_column("spread_mcc").median()
        ), "MCC is now tighter in the median — Section 5.7 needs rewriting"

    def test_f1_fitting_has_a_degenerate_tail_that_mcc_does_not(self, stability):
        """The real reason for the choice: F1's worst case is catastrophic."""
        f1_bad = stability.filter(pl.col("spread_f1") > 0.5).height
        mcc_bad = stability.filter(pl.col("spread_mcc") > 0.5).height
        assert f1_bad > mcc_bad, (
            f"F1 has {f1_bad} wild cells, MCC has {mcc_bad} — the tail argument "
            "no longer holds"
        )
        assert (
            stability.get_column("spread_f1").max()
            > 3 * stability.get_column("spread_mcc").max()
        )

    def test_the_out_of_sample_cost_favours_mcc(self, stability):
        """What decides it: how much deployed performance moves with the choice
        of calibration window."""
        mcc_cost = stability.get_column("mcc_spread_mcc").median()
        f1_cost = stability.get_column("mcc_spread_f1").median()
        assert mcc_cost < f1_cost, (
            f"MCC fitting no longer costs less out of sample: {mcc_cost:.5f} vs "
            f"{f1_cost:.5f}"
        )
        assert f1_cost / mcc_cost > 2.0

    def test_the_degenerate_cells_are_the_ones_section_5_3_predicted(self, stability):
        """Mechanism check: F1's blow-ups must land on persistence at E2/E59,
        which is exactly where Section 5.3 showed the F1-optimal cut collapsing
        to always-fire. If they landed elsewhere, the explanation is wrong."""
        wild = stability.filter(pl.col("spread_f1") > 1.0)
        assert wild.height >= 3, "the degenerate cells vanished"
        assert set(wild.get_column("model").unique()) == {"Persistence"}

    def test_both_objectives_are_recorded(self, stability):
        for objective in OBJECTIVES:
            assert f"spread_{objective}" in stability.columns
            assert f"mcc_spread_{objective}" in stability.columns


class TestTheDocumentQuotesTheTables:
    def test_section_5_6_exists_and_names_the_inversion(self, doc):
        assert "### 5.6 Tampoco es de nuestro umbral" in doc
        section = doc.split("### 5.6")[1].split("### 5.7")[0]
        assert "110" in section, (
            "Section 5.6 no longer quotes the field-convention factor"
        )
        assert "empeora" in section, (
            "Section 5.6 no longer states that the collapse gets WORSE — that "
            "inversion is the whole point of the measurement"
        )

    def test_section_5_7_does_not_overclaim_stability(self, doc):
        """The sentence that must survive: F1 is tighter in the median. A version
        that only says "MCC is more stable" is false and this catches it."""
        assert "### 5.7" in doc
        section = doc.split("### 5.7")[1].split("## 6.")[0]
        assert "más estable" in section and "mediana" in section
        assert "no es \"el MCC es más estable\"" in section or (
            "sería falso en la mediana" in section
        ), "Section 5.7 lost the hedge that makes it honest"

    def test_the_scope_correction_reached_section_1(self, doc):
        """Section 1 previously scoped the finding to self-referential
        thresholds. That scope was too narrow and the correction must be there,
        not only in 5.6."""
        opening = doc.split("### La corrección")[0]
        assert "conservadora" in opening, (
            "Section 1 still under-states the scope of the finding"
        )
