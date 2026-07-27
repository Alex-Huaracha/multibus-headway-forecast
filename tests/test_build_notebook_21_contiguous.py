"""Integrity guard for the retrained-LSTM notebooks.

The notebooks are generated artifacts, so the thing worth protecting is not the
bytes but the contracts they are supposed to carry. Each assertion below maps to
a defect the retrain exists to remove; a regeneration that quietly drops one has
to turn this file red.

Note: running these tests regenerates the notebooks in place (the repo-wide
convention — see CLAUDE.md). They are written to
``notebooks/21_lstm_contiguous/``, which no other builder touches.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import polars as pl
import pytest

from src.build_notebook_21_lstm_contiguous import (
    GROUPS,
    HORIZONS,
    MANIFEST_CSV,
    build_notebook,
)

ROOT = Path(__file__).resolve().parent.parent
NB_ROOT = ROOT / "notebooks" / "21_lstm_contiguous"


@pytest.fixture(scope="module", autouse=True)
def built():
    for key in GROUPS:
        for h in HORIZONS:
            build_notebook(key, h)


def _notebook(group_key: str, horizon: int):
    path = (
        NB_ROOT / group_key / f"h{horizon}"
        / f"21_lstm_contiguous_{group_key}_h{horizon}.ipynb"
    )
    assert path.exists(), path
    return nbf.read(path, as_version=4)


def _source(group_key: str, horizon: int) -> str:
    return "\n".join(
        c["source"] for c in _notebook(group_key, horizon)["cells"]
        if c["cell_type"] == "code"
    )


ALL_CASES = [(k, h) for k in GROUPS for h in HORIZONS]


class TestCellsAreValidPython:
    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_every_code_cell_compiles(self, group_key, horizon):
        for i, cell in enumerate(_notebook(group_key, horizon)["cells"]):
            if cell["cell_type"] != "code":
                continue
            try:
                compile(cell["source"], f"<cell {i}>", "exec")
            except SyntaxError as exc:
                pytest.fail(f"{group_key}/h{horizon} cell {i} ({cell['id']}): {exc}")


class TestAtypicalFlagIsGone:
    """Contract C3 — the flag is a whole-day aggregate, hence leakage."""

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_not_a_required_input(self, group_key, horizon):
        src = _source(group_key, horizon)
        assert '"atypical_days.csv":' not in src, (
            "atypical_days.csv must not be in INPUT_HASHES"
        )
        assert '_resolve_input("atypical_days.csv")' not in src, (
            "the leaking flag must not be resolved as a training input"
        )

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_context_dim_is_four(self, group_key, horizon):
        src = _source(group_key, horizon)
        assert "CONTEXT_DIM = len(CTX_COLS)" in src
        assert "assert CONTEXT_DIM == 4" in src, (
            "the model must be sized for 4 causal context columns"
        )
        assert 'assert "atypical_flag" not in CTX_COLS' in src

    @pytest.mark.parametrize("group_key", list(GROUPS))
    def test_eda_kernel_source_dropped(self, group_key):
        meta = json.loads(
            (NB_ROOT / group_key / "h3" / "kernel-metadata.json").read_text()
        )
        assert "alexhuaracha/02-eda-corridors" not in meta["kernel_sources"], (
            "02-eda-corridors only provided atypical_days.csv, which is gone"
        )


class TestSharedPopulationGate:
    """Contract that replaces the post-hoc join of audit §2.1."""

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_gate_is_present_and_fails_closed(self, group_key, horizon):
        src = _source(group_key, horizon)
        assert "INDEX_DIGESTS" in src
        assert "SHARED-POPULATION GATE FAILED" in src
        assert "raise ValueError(" in src

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_injected_digests_match_the_frozen_manifest(self, group_key, horizon):
        """A stale digest would gate against the wrong population."""
        manifest = pl.read_csv(MANIFEST_CSV)
        src = _source(group_key, horizon)
        for name, _emp in GROUPS[group_key]["corridors"]:
            for split in ("train", "val", "test"):
                row = manifest.filter(
                    (pl.col("corridor") == name)
                    & (pl.col("split") == split)
                    & (pl.col("horizon") == horizon)
                )
                assert row.height == 1
                digest = row.row(0, named=True)["sha256"]
                assert f'"{name}|{split}": "{digest}"' in src, (
                    f"{group_key}/h{horizon}: digest for {name}/{split} is stale"
                )


class TestPopulationContracts:
    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_anchors_on_the_sample_index(self, group_key, horizon):
        src = _source(group_key, horizon)
        assert "make_sample_index(" in src
        # `make_window_index` is embedded (compute_max_N lives beside it) but must
        # never be the anchor — that is the positional defect being removed.
        assert "make_window_index(" not in src.split("def make_window_index")[-1].split(
            "return index"
        )[-1], "make_window_index must not be called for anchoring"

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_materialization_comes_from_the_tested_module(self, group_key, horizon):
        src = _source(group_key, horizon)
        assert "materialize_arrays(" in src
        assert "def fast_materialize" not in src, (
            "the notebook must not carry its own untested materializer"
        )

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_horizon_is_injected_consistently(self, group_key, horizon):
        src = _source(group_key, horizon)
        assert f"HORIZON = {horizon}\n" in src


class TestFullKeyExport:
    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_residuals_carry_the_full_key_and_are_verified(self, group_key, horizon):
        src = _source(group_key, horizon)
        assert "build_keyed_residuals(" in src
        assert "assert_key_is_unique(residuals)" in src, (
            "the key must be verified, not merely declared — harness.py:71 "
            "declared `t` a join key and nothing checked it"
        )


class TestEmbeddedModules:
    REQUIRED = [
        "def make_sample_index",
        "def materialize_arrays",
        "def build_keyed_residuals",
        "def compute_max_N",
        "def winsorize_train_p99",
        "class HeadwayLSTM",
    ]

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_required_definitions_are_embedded(self, group_key, horizon):
        src = _source(group_key, horizon)
        missing = [d for d in self.REQUIRED if d not in src]
        assert not missing, f"{group_key}/h{horizon} missing: {missing}"

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_no_intra_package_imports_survive(self, group_key, horizon):
        src = _source(group_key, horizon)
        assert "from src." not in src, "src.* imports break the flat Kaggle namespace"


class TestKernelMetadata:
    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_ids_are_distinct_and_gpu_enabled(self, group_key, horizon):
        meta = json.loads(
            (NB_ROOT / group_key / f"h{horizon}" / "kernel-metadata.json").read_text()
        )
        assert meta["enable_gpu"] is True
        assert str(horizon) in meta["id"]
        assert meta["code_file"].endswith(f"_h{horizon}.ipynb")

    def test_no_id_collides_with_the_frozen_notebooks(self):
        ids = set()
        for key in GROUPS:
            for h in HORIZONS:
                meta = json.loads(
                    (NB_ROOT / key / f"h{h}" / "kernel-metadata.json").read_text()
                )
                ids.add(meta["id"])
        assert len(ids) == len(GROUPS) * len(HORIZONS)
        for frozen in ("11-lstm-multihorizon", "17-e4-lstm", "12-spatialconvlstm"):
            assert not any(frozen in i for i in ids), (
                "a new kernel id must never overwrite a frozen one"
            )
