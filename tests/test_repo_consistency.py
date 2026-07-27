"""The repo must not claim things it did not build — audit pending #8.

Three contradictions shipped for months because nothing checked them: the README
advertised a GNN+LSTM that was never built, four corridors were declared while
only three exist, and the Kaggle credential path disagreed between documents.
None of those is a code defect, which is exactly why no test caught them.

``docs/propuesta.md`` is exempt from the content rules: it is the original
proposal, kept unrewritten as the record of what was planned. Its exemption is
conditional on carrying the warning header that says so, which is itself
asserted here.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROPOSAL = REPO_ROOT / "docs" / "propuesta.md"
RESULTS_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"

# Documents that describe the project AS DELIVERED. The proposal is deliberately
# absent — see the module docstring.
CURRENT_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "docs" / "correr-kaggle.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks.

    Claims are checked per paragraph, not per line: markdown wraps sentences, so
    "**No se construyó ninguna GNN.** La propuesta original planteaba una
    arquitectura / GNN+LSTM; ..." puts the denial and the mention on different
    lines and a line-wise check flags its own correction.
    """
    return [block for block in text.split("\n\n") if block.strip()]


class TestTheProposalIsMarkedHistorical:
    def test_it_carries_the_warning_header(self):
        head = _read(PROPOSAL)[:3000]
        assert "Documento histórico" in head, (
            "propuesta.md keeps GNN and four-corridor claims; without the header "
            "saying it is historical, those become live contradictions again"
        )

    def test_the_header_names_all_three_divergences(self):
        head = _read(PROPOSAL)[:3000]
        for marker in ("GNN", "E58", "bunching"):
            assert marker in head, f"the header does not mention {marker}"

    def test_the_header_points_at_the_current_document(self):
        assert "documento-resultados.md" in _read(PROPOSAL)[:3000]


class TestNoGnnIsClaimed:
    """No GNN was ever implemented; the spatial architectures are ConvLSTM and
    Transformer, and neither beats the plain LSTM."""

    def test_no_current_doc_claims_a_gnn_was_built(self):
        for path in CURRENT_DOCS:
            for block in _paragraphs(_read(path)):
                if "GNN" not in block:
                    continue
                # A paragraph may mention the GNN only to say it does not exist.
                denies = any(
                    word in block.lower()
                    for word in ("nunca", "no se construyó", "not built", "never")
                )
                assert denies, f"{path.name}: unqualified GNN claim -> {block.strip()[:120]}"

    def test_no_gnn_module_exists_to_contradict_that(self):
        hits = list((REPO_ROOT / "src").rglob("*gnn*"))
        assert hits == [], f"a GNN module exists after all: {hits}"


class TestCorridorScope:
    IN_SCOPE = {"E2", "E4", "E59"}

    def test_e58_has_no_processed_data(self):
        assert not (REPO_ROOT / "data" / "processed" / "headways_E58.parquet").exists()

    def test_e58_appears_in_no_result_table(self):
        if not RESULTS_DIR.exists():
            pytest.skip("results directory not present")
        offenders = []
        for csv in sorted(RESULTS_DIR.glob("*.csv")):
            frame = pl.read_csv(csv, infer_schema_length=0)
            if "corridor" not in frame.columns:
                continue
            found = set(frame.get_column("corridor").unique().to_list())
            if not found <= self.IN_SCOPE:
                offenders.append((csv.name, sorted(found - self.IN_SCOPE)))
        assert offenders == [], f"out-of-scope corridors in results: {offenders}"

    def test_the_executable_scope_constants_exclude_e58(self):
        """The authoritative definition of scope is the constant the builders
        iterate, not a docstring. ``preprocessing/config.py`` legitimately names
        E58 while stating which companies report a heading field — that is a
        fact about the raw dataset, not a scope claim, so a textual scan of
        ``src`` would flag the wrong thing."""
        from src.build_contiguous_significance import CORRIDORS as ANALYSIS_SCOPE
        from src.build_sample_index import CORRIDORS as INDEX_SCOPE

        assert set(ANALYSIS_SCOPE) == self.IN_SCOPE
        assert {label for _empresaid, label in INDEX_SCOPE} == self.IN_SCOPE
        assert 58 not in {empresaid for empresaid, _label in INDEX_SCOPE}

    def test_no_module_reads_an_e58_parquet(self):
        hits = [
            path.name
            for path in (REPO_ROOT / "src").rglob("*.py")
            if "headways_E58" in path.read_text(encoding="utf-8")
        ]
        assert hits == [], f"modules loading E58 data: {hits}"

    def test_current_docs_qualify_any_four_corridor_mention(self):
        for path in CURRENT_DOCS:
            for block in _paragraphs(_read(path)):
                if "58" not in block:
                    continue
                qualifies = any(
                    word in block.lower()
                    for word in ("nunca", "never", "stale", "no tiene", "no entró")
                )
                assert qualifies, f"{path.name}: unqualified E58 claim -> {block.strip()[:120]}"


class TestCredentialPathIsConsistent:
    """The docs disagreed on where the Kaggle token lives; only one is real."""

    def test_no_current_doc_points_at_kaggle_json(self):
        for path in CURRENT_DOCS:
            for line in _read(path).splitlines():
                if "kaggle.json" not in line:
                    continue
                denies = any(
                    word in line.lower() for word in ("no es", "not ", "antiguo")
                )
                assert denies, f"{path.name}: stale credential path -> {line.strip()}"

    def test_the_real_path_is_documented(self):
        assert "access_token" in _read(REPO_ROOT / "README.md")
