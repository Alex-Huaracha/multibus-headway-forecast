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


class TestScalarClaims:
    def test_the_headline_h10_margins(self, text):
        audit = _csv("contiguous_paired_audit.csv").filter(
            pl.col("direction") == "aggregate"
        )
        for corridor, claimed in (("E2", -1.473), ("E59", -1.173), ("E4", -1.381)):
            actual = _cell(audit, "delta_lstm_persist", corridor=corridor, horizon=10)
            assert actual == pytest.approx(claimed, abs=0.001)
            assert f"{abs(claimed):.3f}" in text

    def test_persistence_wins_at_one_step_everywhere(self, text):
        audit = _csv("contiguous_paired_audit.csv").filter(
            pl.col("direction") == "aggregate"
        )
        h1 = audit.filter(pl.col("horizon") == 1)
        assert (h1.get_column("delta_lstm_persist") > 0).all()

    def test_xgboost_reproduces_the_crossover(self, text):
        audit = _csv("contiguous_paired_audit.csv").filter(
            pl.col("direction") == "aggregate"
        )
        for corridor, claimed in (("E2", -1.585), ("E59", -0.787), ("E4", -1.085)):
            actual = _cell(audit, "delta_xgb_persist", corridor=corridor, horizon=10)
            assert actual == pytest.approx(claimed, abs=0.001)

    def test_the_framing_bias_figure(self, text):
        audit = _csv("contiguous_paired_audit.csv")
        worst = max(
            audit.get_column("framing_delta_lstm").abs().max(),
            audit.get_column("framing_delta_xgb").abs().max(),
        )
        assert worst < 0.001 + 1e-9
        assert "0.001 min" in text

    def test_the_contiguity_cost_range(self, text):
        manifest = _csv("sample_index_manifest.csv").filter(pl.col("split") == "test")
        usable = manifest.get_column("pct_snapshots_usable")
        assert round(float(usable.min()), 1) == 81.9
        assert round(float(usable.max()), 1) == 90.2
        assert "81.9" in text and "90.2" in text


class TestSignificanceClaims:
    @pytest.fixture(scope="class")
    def significance(self) -> pl.DataFrame:
        return _csv("contiguous_significance.csv").filter(
            (pl.col("metric") == "MAE") & (pl.col("comparison") == "LSTM_vs_PERSIST")
        )

    def test_the_test_window_is_twenty_two_service_days(self, significance, text):
        assert set(significance.get_column("n_service_days")) == {22}
        assert "22 días" in text

    def test_the_two_verdicts_that_fall(self, significance, text):
        e2 = _cell(significance, "dm_p_clustered", corridor="E2", horizon=1)
        e4 = _cell(significance, "dm_p_clustered", corridor="E4", horizon=3)
        assert e2 == pytest.approx(0.0619, abs=0.0005)
        assert e4 == pytest.approx(0.1849, abs=0.0005)
        assert "0.0619" in text and "0.1849" in text

    def test_the_clustering_is_what_kills_them(self, significance):
        for corridor, horizon in (("E2", 1), ("E4", 3)):
            hac = _cell(significance, "dm_p_hac", corridor=corridor, horizon=horizon)
            clustered = _cell(
                significance, "dm_p_clustered", corridor=corridor, horizon=horizon
            )
            assert hac < 0.05 <= clustered

    def test_long_horizons_survive_clustering(self, significance):
        long = significance.filter(pl.col("horizon") >= 5)
        assert long.get_column("dm_p_clustered").max() < 1e-9

    def test_the_h3_win_rates(self, significance, text):
        for corridor, claimed in (("E4", 0.4598), ("E59", 0.4726)):
            actual = _cell(significance, "win_rate", corridor=corridor, horizon=3)
            assert actual == pytest.approx(claimed, abs=0.0005)

    def test_the_h3_wilcoxon_contradicts_the_mean(self, significance, text):
        assert _cell(
            significance, "wilcoxon_p_one_sided", corridor="E4", horizon=3
        ) == pytest.approx(1.0, abs=1e-6)
        assert _cell(
            significance, "wilcoxon_p_one_sided", corridor="E59", horizon=3
        ) == pytest.approx(0.952, abs=0.001)
        assert "0.952" in text


class TestVectorClaims:
    @pytest.fixture(scope="class")
    def vector(self) -> pl.DataFrame:
        return _csv("contiguous_vector_metrics.csv")

    def test_the_headline_f1_pair(self, vector, text):
        persistence = _cell(
            vector, "bunching_f1", model="Persistence", corridor="E2", horizon=10
        )
        lstm = _cell(vector, "bunching_f1", model="LSTM", corridor="E2", horizon=10)
        assert persistence == pytest.approx(0.332, abs=0.001)
        assert lstm == pytest.approx(0.0013, abs=0.0001)
        assert round(persistence / lstm, 1) == pytest.approx(253.4, abs=0.5)
        assert "253" in text

    def test_the_flattening_figures(self, vector, text):
        true_cv = _cell(vector, "mean_cv_true", model="LSTM", corridor="E2", horizon=10)
        pred_cv = _cell(vector, "mean_cv_pred", model="LSTM", corridor="E2", horizon=10)
        assert true_cv == pytest.approx(0.787, abs=0.001)
        assert pred_cv == pytest.approx(0.161, abs=0.001)
        assert "0.16" in text and "0.79" in text

    def test_precision_holds_while_recall_collapses(self, vector, text):
        lstm = vector.filter(pl.col("model") == "LSTM")
        firing = lstm.filter(pl.col("bunching_tp") + pl.col("bunching_fp") > 100)
        assert firing.get_column("bunching_precision").min() > 0.49
        assert firing.get_column("bunching_precision").max() < 0.74
        assert lstm.filter(pl.col("horizon") == 10).get_column(
            "bunching_recall"
        ).max() < 0.02

    def test_the_bunching_base_rate_range(self, vector, text):
        rates = vector.get_column("bunching_rate_true")
        assert 0.17 < rates.min() and rates.max() < 0.31
        assert "17 %" in text and "30 %" in text

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
        assert round(float(pct.min()), 2) == 0.78
        assert round(float(pct.max()), 2) == 1.11
        assert "0.78" in text and "1.11" in text

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
        assert _cell(
            router, "gain_vs_best_pure", corridor="E4", horizon=3
        ) == pytest.approx(-0.073, abs=0.001)
        assert _cell(
            router, "gain_vs_best_pure", corridor="E59", horizon=3
        ) == pytest.approx(-0.042, abs=0.001)
        assert int(router.get_column("policy_degenerate").sum()) == 7
        assert "7 de 12" in text

    def test_the_tuning_asymmetry_is_declared(self, text):
        from src.baselines.fitted import SEARCH_N_CONFIGS

        assert SEARCH_N_CONFIGS == 24
        assert "24 configuraciones" in text
        assert "no es atribuible" in text
