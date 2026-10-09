"""Every headline number in the results document must trace to an artifact.

The previous version of that document drifted from the tables it cited — stale
figures, a superseded XGBoost, a significance footnote that was hardcoded rather
than computed — and nothing caught it because prose is not executable.

These tests make the prose executable. Each claim below is stated as
(what the document says) vs (what the CSV holds), so a regenerated table that
moves a number fails here instead of leaving the document quietly wrong.
"""
from __future__ import annotations

import os
import re

os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "docs" / "resultados" / "documento-resultados.md"
CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"

pytestmark = pytest.mark.skipif(
    not DOC.exists(), reason="documento-resultados.md missing"
)


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


def _csv(name: str) -> pl.DataFrame:
    path = CSV_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return pl.read_csv(path)


def _cell(frame: pl.DataFrame, column: str, **filters) -> float:
    predicate = pl.lit(True)
    for key, value in filters.items():
        predicate = predicate & (pl.col(key) == value)
    return float(frame.filter(predicate).get_column(column).item())


def _footnotes(text: str) -> tuple[set[str], set[str]]:
    """(referenced, defined).

    A definition is a ``[^x]:`` at the START of a line. Testing for the colon
    alone misclassifies prose like "divididos temporalmente[^split]:" — a real
    reference that happens to precede a colon — as a definition, and then the
    footnote looks orphaned. Definitions are stripped before scanning for uses.
    """
    definition = re.compile(r"^\[\^([a-z0-9]+)\]:", flags=re.MULTILINE)
    defined = set(definition.findall(text))
    body = "\n".join(
        line for line in text.splitlines() if not definition.match(line)
    )
    return set(re.findall(r"\[\^([a-z0-9]+)\]", body)), defined


class TestDocumentStructure:
    def test_every_footnote_reference_is_defined(self, text):
        used, defined = _footnotes(text)
        assert used - defined == set(), f"undefined footnotes: {sorted(used - defined)}"

    def test_no_footnote_is_defined_and_never_used(self, text):
        used, defined = _footnotes(text)
        assert defined - used == set(), f"orphan footnotes: {sorted(defined - used)}"

    def test_it_declares_which_pipeline_the_numbers_come_from(self, text):
        assert "pipeline contiguo" in text.lower()
        assert "21-lstm-contiguous" in text

    def test_the_router_section_stays_compressed(self, text):
        """Audit pending #9: two paragraphs, as a demonstration of feasibility.

        Counts prose blocks only — the heading remainder and the ``---`` rule are
        not paragraphs, and counting them turned a compliant section into a
        failure.
        """
        section = text.split("El enrutador ex-ante")[1].split("## 7.")[0]
        paragraphs = [
            block.strip()
            for block in section.split("\n\n")
            if block.strip()
            and not block.startswith("#")
            and block.strip() != "---"
            and not block.strip().startswith(":")
        ]
        assert len(paragraphs) <= 2, (
            f"router section grew to {len(paragraphs)} paragraphs"
        )

    def test_stale_figures_are_flagged_not_cited_as_current(self, text):
        for figure in ("curva-degradacion.png", "volatilidad-crossover.png"):
            if figure not in text:
                continue
            context = text[text.index(figure) - 400 : text.index(figure) + 400]
            assert "congeladas" in context or "regenerar" in context, (
                f"{figure} cited without flagging it as stale"
            )

    def test_every_embedded_figure_exists(self, text):
        embedded = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        assert embedded, "the document embeds no figure at all"
        missing = [
            name for name in embedded if not (DOC.parent / name).exists()
        ]
        assert missing == [], f"embedded figures not on disk: {missing}"

    def test_the_contiguous_figures_are_the_ones_embedded(self, text):
        embedded = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))
        assert embedded == {
            "contiguo-artefacto-threshold.png",
            "contiguo-deteccion-sin-threshold.png",
            "contiguo-degradacion.png",
            "contiguo-volatilidad.png",
        }, f"unexpected figure set: {sorted(embedded)}"

    def test_the_retracted_figure_is_gone(self, text):
        """``contiguo-disociacion.png`` plotted fixed-cut bunching F1 as if it
        measured the models. That is the artifact Section 5.3 dismantles, so a
        document that still embeds it argues against itself."""
        assert "contiguo-disociacion.png" not in text or "eliminada" in text[
            text.index("contiguo-disociacion.png") - 200 :
            text.index("contiguo-disociacion.png") + 200
        ], "the retracted figure is cited without saying it was retracted"
        assert not (DOC.parent / "contiguo-disociacion.png").exists(), (
            "the retracted figure is still on disk and will be picked up by "
            "anyone browsing the results directory"
        )

    def test_the_artifact_and_its_correction_lead_as_a_pair(self, text):
        """The two must appear together and BEFORE the scalar result: shown
        alone, the first misrepresents the models and the second overstates
        them. The scalar figures are supporting evidence, not the headline."""
        order = [
            text.index(name)
            for name in (
                "contiguo-artefacto-threshold.png",
                "contiguo-deteccion-sin-threshold.png",
                "contiguo-degradacion.png",
                "contiguo-volatilidad.png",
            )
        ]
        assert order == sorted(order), (
            "the artifact/correction pair no longer leads the document"
        )


class TestScalarClaims:
    def test_the_headline_h10_margins(self, text):
        audit = _csv("contiguous_paired_audit.csv").filter(
            pl.col("direction") == "aggregate"
        )
        for corridor, claimed in (("E2", -1.548), ("E59", -1.259), ("E4", -1.381)):
            actual = _cell(audit, "delta_lstm_persist", corridor=corridor, horizon=10)
            assert actual == pytest.approx(claimed, abs=0.001)
            assert f"{abs(claimed):.3f}" in text
        best = audit.filter(pl.col("horizon") == 10).get_column("delta_lstm_persist").min()
        assert f"hasta **{abs(best):.2f} min de MAE**" in text

    def test_persistence_wins_at_one_step_except_on_e2(self, text):
        """At h=1 persistence wins on E4 and E59; on E2 the two tie.

        The E2 tie is a claim, not an absence of one: the margin must stay
        below a hundredth of a minute in size AND fail the day-clustered test.
        A regenerated table that makes E2 a real win for either side fails here.
        """
        audit = _csv("contiguous_paired_audit.csv").filter(
            pl.col("direction") == "aggregate"
        )
        significance = _csv("contiguous_significance.csv").filter(
            (pl.col("metric") == "MAE") & (pl.col("comparison") == "LSTM_vs_PERSIST")
        )
        for corridor in ("E4", "E59"):
            assert _cell(audit, "delta_lstm_persist", corridor=corridor, horizon=1) > 0
            assert _cell(
                significance, "dm_p_clustered", corridor=corridor, horizon=1
            ) < 0.05
        e2_delta = _cell(audit, "delta_lstm_persist", corridor="E2", horizon=1)
        e2_p = _cell(significance, "dm_p_clustered", corridor="E2", horizon=1)
        assert abs(e2_delta) < 0.02 and e2_p > 0.05
        assert f"Δ = −{abs(e2_delta):.3f} min, *p* = {e2_p:.2f}" in text
        assert "En E2 empatan" in text

    def test_xgboost_reproduces_the_crossover(self, text):
        audit = _csv("contiguous_paired_audit.csv").filter(
            pl.col("direction") == "aggregate"
        )
        for corridor, claimed in (("E2", -1.566), ("E59", -0.972), ("E4", -1.085)):
            actual = _cell(audit, "delta_xgb_persist", corridor=corridor, horizon=10)
            assert actual == pytest.approx(claimed, abs=0.001)
            assert f"−{abs(claimed):.3f} en {corridor}" in text

    def test_the_framing_bias_figure(self, text):
        audit = _csv("contiguous_paired_audit.csv")
        worst = max(
            audit.get_column("framing_delta_lstm").abs().max(),
            audit.get_column("framing_delta_xgb").abs().max(),
        )
        assert worst < 0.0025
        assert f"{worst:.4f} min" in text

    def test_the_contiguity_cost_range(self, text):
        """Scoped to the PUBLISHED fold.

        The manifest now carries one row set per rolling origin, so an unscoped
        filter mixes windows the document does not describe — and the range it
        quotes would silently start meaning something else.
        """
        manifest = _csv("sample_index_manifest.csv").filter(
            (pl.col("fold") == "main") & (pl.col("split") == "test")
        )
        assert manifest.height == 12, "expected 3 corridors x 4 horizons"
        usable = manifest.get_column("pct_snapshots_usable")
        low, high = round(float(usable.min()), 1), round(float(usable.max()), 1)
        assert (low, high) == (81.9, 91.2)
        assert f"entre el {low} % y el {high} %" in text


class TestSignificanceClaims:
    @pytest.fixture(scope="class")
    def significance(self) -> pl.DataFrame:
        return _csv("contiguous_significance.csv").filter(
            (pl.col("metric") == "MAE") & (pl.col("comparison") == "LSTM_vs_PERSIST")
        )

    def test_the_test_window_is_twenty_two_service_days(self, significance, text):
        assert set(significance.get_column("n_service_days")) == {22}
        assert "22 días" in text

    def test_the_two_verdicts_that_fall(self, text):
        """Exactly two comparisons against persistence lose significance when
        the variance is clustered by service day: LSTM on E4 h=3 and XGBoost on
        E2 h=1. Derived from the CSV, so a third one appearing fails here."""
        mae = _csv("contiguous_significance.csv").filter(
            (pl.col("metric") == "MAE")
            & pl.col("comparison").is_in(["LSTM_vs_PERSIST", "XGB_vs_PERSIST"])
        )
        falling = mae.filter(
            (pl.col("dm_p_hac") < 0.05) & (pl.col("dm_p_clustered") >= 0.05)
        )
        assert set(
            zip(
                falling.get_column("comparison"),
                falling.get_column("corridor"),
                falling.get_column("horizon"),
            )
        ) == {("LSTM_vs_PERSIST", "E4", 3), ("XGB_vs_PERSIST", "E2", 1)}
        for row in falling.iter_rows(named=True):
            assert f"**{row['dm_p_clustered']:.4f}**" in text
        assert "dos veredictos se caen" in text

    def test_e2_one_step_was_never_significant(self, significance, text):
        """The old third verdict (LSTM on E2 h=1) no longer falls: it is not
        significant even before clustering, which the document must say."""
        hac = _cell(significance, "dm_p_hac", corridor="E2", horizon=1)
        clustered = _cell(significance, "dm_p_clustered", corridor="E2", horizon=1)
        assert 0.05 <= hac <= clustered
        assert f"| {hac:.4f} | **{clustered:.4f}** |" in text
        assert "ya no era significativo sin agrupar" in text

    def test_the_clustering_is_what_kills_them(self, significance):
        hac = _cell(significance, "dm_p_hac", corridor="E4", horizon=3)
        clustered = _cell(significance, "dm_p_clustered", corridor="E4", horizon=3)
        assert hac < 0.05 <= clustered

    def test_long_horizons_survive_clustering(self, significance):
        long = significance.filter(pl.col("horizon") >= 5)
        assert long.get_column("dm_p_clustered").max() < 1e-9

    def test_the_h3_win_rates(self, significance, text):
        for corridor, claimed in (("E4", 0.4598), ("E59", 0.4572)):
            actual = _cell(significance, "win_rate", corridor=corridor, horizon=3)
            assert actual == pytest.approx(claimed, abs=0.0005)
            assert f"{100 * actual:.1f} %" in text

    def test_the_h3_wilcoxon_contradicts_the_mean(self, significance, text):
        for corridor in ("E4", "E59"):
            assert _cell(
                significance, "delta_loss", corridor=corridor, horizon=3
            ) < 0
            assert _cell(
                significance, "wilcoxon_p_one_sided", corridor=corridor, horizon=3
            ) == pytest.approx(1.0, abs=1e-6)
        assert "*p* = 1.000 en los dos" in text


class TestVectorClaims:
    @pytest.fixture(scope="class")
    def vector(self) -> pl.DataFrame:
        return _csv("contiguous_vector_metrics.csv")

    def test_the_headline_f1_pair(self, vector, text):
        """E2 h=10: the LSTM never fires, so its F1 is zero and the ratio
        against persistence is not finite. The largest finite ratio is E2 h=5."""
        persistence = _cell(
            vector, "bunching_f1", model="Persistence", corridor="E2", horizon=10
        )
        e2 = vector.filter(
            (pl.col("model") == "LSTM") & (pl.col("corridor") == "E2")
            & (pl.col("horizon") == 10)
        ).row(0, named=True)
        assert e2["bunching_tp"] + e2["bunching_fp"] == 0
        assert e2["bunching_f1"] == 0.0
        assert persistence == pytest.approx(0.315, abs=0.001)
        assert f"| 0.000 | **{persistence:.3f}** |" in text
        h5_ratio = _cell(
            vector, "bunching_f1", model="Persistence", corridor="E2", horizon=5
        ) / _cell(vector, "bunching_f1", model="LSTM", corridor="E2", horizon=5)
        assert round(h5_ratio, 1) == pytest.approx(209.0, abs=0.5)
        assert "209" in text

    def test_the_flattening_figures(self, vector, text):
        true_cv = _cell(vector, "mean_cv_true", model="LSTM", corridor="E2", horizon=10)
        pred_cv = _cell(vector, "mean_cv_pred", model="LSTM", corridor="E2", horizon=10)
        assert true_cv == pytest.approx(0.742, abs=0.001)
        assert pred_cv == pytest.approx(0.179, abs=0.001)
        assert f"CV de {pred_cv:.2f} cuando el real es {true_cv:.2f}" in text

    def test_precision_holds_while_recall_collapses(self, vector, text):
        """Where the LSTM fires, it is right more often than persistence and
        than the base rate; where it does not fire (E2 h=10) precision is
        undefined, and the document has to say so rather than average it in."""
        lstm = vector.filter(pl.col("model") == "LSTM")
        persistence = vector.filter(pl.col("model") == "Persistence").select(
            "corridor", "horizon",
            pl.col("bunching_precision").alias("persist_precision"),
        )
        fires = lstm.filter(pl.col("bunching_tp") + pl.col("bunching_fp") > 0)
        silent = lstm.filter(pl.col("bunching_tp") + pl.col("bunching_fp") == 0)
        assert set(zip(silent.get_column("corridor"), silent.get_column("horizon"))) == {
            ("E2", 10)
        }
        assert fires.height == 11
        joined = fires.join(persistence, on=["corridor", "horizon"])
        assert (joined.get_column("bunching_precision")
                > joined.get_column("persist_precision")).all()
        assert (joined.get_column("bunching_precision")
                > joined.get_column("bunching_rate_true")).all()
        firing = lstm.filter(pl.col("bunching_tp") + pl.col("bunching_fp") > 100)
        low = firing.get_column("bunching_precision").min()
        assert round(100 * low) == 45
        assert "**45 % de precisión**" in text
        assert "precisión no está definida" in text
        assert lstm.filter(pl.col("horizon") == 10).get_column(
            "bunching_recall"
        ).max() < 0.02

    def test_the_bunching_base_rate_range(self, vector, text):
        rates = vector.get_column("bunching_rate_true")
        low, high = round(100 * rates.min()), round(100 * rates.max())
        assert (low, high) == (17, 29)
        assert f"**{low} % al {high} %**" in text

    def test_persistence_wins_every_vector_cell(self, vector):
        best = (
            vector.sort("bunching_f1", descending=True)
            .group_by(["corridor", "horizon"], maintain_order=True)
            .first()
        )
        assert set(best.get_column("model")) == {"Persistence"}


class TestRobustnessClaims:
    def test_the_clipping_footprint(self, text):
        sensitivity = _csv("contiguous_winsorization_sensitivity.csv")
        pct = sensitivity.get_column("pct_clipped_targets")
        low, high = round(float(pct.min()), 2), round(float(pct.max()), 2)
        assert (low, high) == (0.84, 1.10)
        assert f"**{low:.2f} % y {high:.2f} %**" in text

    def test_no_margin_moves_by_a_hundredth_of_a_minute(self, text):
        sensitivity = _csv("contiguous_winsorization_sensitivity.csv").filter(
            pl.col("model") != "Persistence"
        )
        shift = (
            sensitivity.get_column("delta_vs_persist_raw_fair").to_numpy()
            - sensitivity.get_column("delta_vs_persist_clipped").to_numpy()
        )
        assert abs(shift).max() < 0.01

    def test_the_router_gains_and_where_they_survive(self, text):
        router = _csv("contiguous_router.csv").filter(
            pl.col("split_mode") == "temporal"
        )
        for corridor, horizon, claimed in (
            ("E2", 1, -0.053), ("E4", 3, -0.073), ("E59", 3, -0.094),
        ):
            assert _cell(
                router, "gain_vs_best_pure", corridor=corridor, horizon=horizon
            ) == pytest.approx(claimed, abs=0.001)
            assert f"{corridor} h={horizon} (−{abs(claimed):.3f} min)" in text
        degenerate = int(router.get_column("policy_degenerate").sum())
        assert degenerate == 6
        assert f"{degenerate} de 12" in text

    def test_where_the_router_beats_seed_noise(self, text):
        """The cells the document names are exactly the ones whose temporal
        gain exceeds the random-split spread, split by sign."""
        from src.build_contiguous_router import seed_sweep_summary

        summary = seed_sweep_summary(_csv("contiguous_router.csv"))
        beats = summary.filter(pl.col("exceeds_seed_noise"))
        helps = beats.filter(pl.col("gain_temporal") < 0)
        hurts = beats.filter(pl.col("gain_temporal") > 0)
        assert set(zip(helps.get_column("corridor"), helps.get_column("horizon"))) == {
            ("E2", 1), ("E4", 3), ("E59", 3)
        }
        assert set(zip(hurts.get_column("corridor"), hurts.get_column("horizon"))) == {
            ("E2", 10)
        }
        assert hurts.get_column("fails_forward_in_time").all()
        assert f"{helps.height} de 12 celdas" in text

    def test_the_tuning_asymmetry_is_declared(self, text):
        from src.baselines.fitted import SEARCH_N_CONFIGS

        assert SEARCH_N_CONFIGS == 24
        assert "24 configuraciones" in text
        assert "no es atribuible" in text
