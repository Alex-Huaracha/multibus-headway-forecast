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
from src.evaluation.splits import MAIN_FOLD, ROLLING_FOLDS

ROOT = Path(__file__).resolve().parent.parent
NB_ROOT = ROOT / "notebooks" / "21_lstm_contiguous"


@pytest.fixture(scope="module", autouse=True)
def built():
    for fold in ROLLING_FOLDS:
        for key in GROUPS:
            for h in HORIZONS:
                build_notebook(key, h, fold)


def _dir(group_key: str, horizon: int, fold=MAIN_FOLD) -> Path:
    base = NB_ROOT / group_key
    return base / f"h{horizon}" if fold.name == "main" else base / fold.name / f"h{horizon}"


def _notebook(group_key: str, horizon: int, fold=MAIN_FOLD):
    suffix = "" if fold.name == "main" else f"_{fold.name}"
    path = (
        _dir(group_key, horizon, fold)
        / f"21_lstm_contiguous_{group_key}{suffix}_h{horizon}.ipynb"
    )
    assert path.exists(), path
    return nbf.read(path, as_version=4)


def _source(group_key: str, horizon: int, fold=MAIN_FOLD) -> str:
    return "\n".join(
        c["source"] for c in _notebook(group_key, horizon, fold)["cells"]
        if c["cell_type"] == "code"
    )


# Output stem per (corridor group, origin), spelled out rather than recomputed.
# Deriving it from the builder's own expression is what let the group coordinate
# go missing unnoticed: the test agreed with the bug. The two `main` entries are
# the published filenames that `docs/correr-kaggle.md` and
# `build_contiguous_significance.load_lstm` already read — they must not move.
EXPECTED_STEMS = {
    ("e2e59", "main"): "lstm_contig",
    ("e2e59", "r1"): "lstm_contig_r1",
    ("e2e59", "r2"): "lstm_contig_r2",
    ("e4", "main"): "lstm_contig_E4",
    ("e4", "r1"): "lstm_contig_E4_r1",
    ("e4", "r2"): "lstm_contig_E4_r2",
}


def _expected_stem(group_key: str, fold) -> str:
    return EXPECTED_STEMS[(group_key, fold.name)]


ALL_CASES = [(k, h) for k in GROUPS for h in HORIZONS]
ALL_FOLD_CASES = [(f, k, h) for f in ROLLING_FOLDS for k in GROUPS for h in HORIZONS]


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
        """A stale digest would gate against the wrong population.

        The fold is part of the lookup. The manifest carries one row set per
        evaluation origin, so a filter without it matches every origin and the
        notebook could end up frozen against a population it never trained on.
        """
        manifest = pl.read_csv(MANIFEST_CSV)
        src = _source(group_key, horizon)
        for name, _emp in GROUPS[group_key]["corridors"]:
            for split in ("train", "val", "test"):
                row = manifest.filter(
                    (pl.col("fold") == "main")
                    & (pl.col("corridor") == name)
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


class TestRollingOriginNotebooks:
    """Every origin gets its own kernel, its own gate and its own outputs.

    The risk this class guards is not that a rolling notebook fails — a failure
    is visible. It is that a rolling notebook succeeds while silently behaving
    like the published one, producing a "robustness check" that re-measures the
    same window and therefore proves nothing.
    """

    @pytest.mark.parametrize("fold,group_key,horizon", ALL_FOLD_CASES)
    def test_every_code_cell_compiles(self, fold, group_key, horizon):
        for cell in _notebook(group_key, horizon, fold)["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "<nb>", "exec")

    @pytest.mark.parametrize("fold,group_key,horizon", ALL_FOLD_CASES)
    def test_the_notebook_declares_its_own_fold(self, fold, group_key, horizon):
        src = _source(group_key, horizon, fold)
        assert f'FOLD_NAME = "{fold.name}"' in src

    @pytest.mark.parametrize("fold,group_key,horizon", ALL_FOLD_CASES)
    def test_the_split_is_taken_from_that_fold(self, fold, group_key, horizon):
        """`split_temporal(frame)` would silently fall back to the main split."""
        src = _source(group_key, horizon, fold)
        assert "FOLD = fold_by_name(FOLD_NAME)" in src
        assert "split_temporal(frame, FOLD)" in src
        assert "split_temporal(frame)" not in src

    @pytest.mark.parametrize("fold,group_key,horizon", ALL_FOLD_CASES)
    def test_outputs_do_not_collide_between_folds(self, fold, group_key, horizon):
        """Downloading r1 must not overwrite the published residuals."""
        src = _source(group_key, horizon, fold)
        stem = _expected_stem(group_key, fold)
        assert f'{stem}_residuals_h{{HORIZON}}.csv' in src
        assert f'{stem}_results_h{{HORIZON}}.csv' in src

    def test_every_output_filename_is_unique(self):
        """The property the per-case assertions cannot see.

        Every kernel of family 21 — both corridor groups, all three origins,
        four horizons — downloads into the SAME residual directory. A stem that
        drops either coordinate makes two runs land on one path and the second
        pull overwrites the first, with no error anywhere: the analysis layer
        (``build_contiguous_significance.load_lstm``) skips a missing file and
        reports metrics over whatever survived.

        Asserting the stem case by case cannot catch that — a formula shared by
        the test and the builder is a formula nobody checks. This asserts the
        collision itself.
        """
        # `ROLLING_FOLDS` already carries MAIN_FOLD as its last entry
        # (`src/evaluation/splits.py`), so it is the complete set of origins.
        names = []
        for fold in ROLLING_FOLDS:
            for group_key in GROUPS:
                for horizon in HORIZONS:
                    src = _source(group_key, horizon, fold)
                    stem = _expected_stem(group_key, fold)
                    for kind in ("residuals", "results"):
                        name = f"{stem}_{kind}_h{{HORIZON}}.csv"
                        assert name in src, (
                            f"{group_key}/{fold.name}/h{horizon} does not emit {name}"
                        )
                        names.append(name.replace("{HORIZON}", str(horizon)))

        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert not duplicates, (
            f"two runs of family 21 would write these same filenames and "
            f"overwrite each other on download: {duplicates}"
        )
        assert len(names) == 2 * len(GROUPS) * len(HORIZONS) * len(ROLLING_FOLDS)

    def test_every_kernel_id_is_unique(self):
        ids = []
        for fold in ROLLING_FOLDS:
            for key in GROUPS:
                for h in HORIZONS:
                    meta = json.loads(
                        (_dir(key, h, fold) / "kernel-metadata.json").read_text()
                    )
                    ids.append(meta["id"])
        assert len(ids) == len(set(ids)), "two folds would overwrite each other on Kaggle"
        assert len(ids) == len(ROLLING_FOLDS) * len(GROUPS) * len(HORIZONS)

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_the_published_fold_keeps_its_paths_and_slug(self, group_key, horizon):
        """Its kernel already exists on Kaggle and its residual filenames are
        referenced by the runbook and by every analysis builder."""
        meta = json.loads(
            (_dir(group_key, horizon) / "kernel-metadata.json").read_text()
        )
        expected = GROUPS[group_key]["kernel_id"].format(h=horizon)
        assert meta["id"] == f"alexhuaracha/{expected}"
        assert "r1" not in meta["id"] and "r2" not in meta["id"]

    @pytest.mark.parametrize("fold,group_key,horizon", ALL_FOLD_CASES)
    def test_the_gate_uses_that_folds_digests(self, fold, group_key, horizon):
        manifest = pl.read_csv(MANIFEST_CSV)
        src = _source(group_key, horizon, fold)
        for name, _emp in GROUPS[group_key]["corridors"]:
            for split in ("train", "val", "test"):
                digest = manifest.filter(
                    (pl.col("fold") == fold.name)
                    & (pl.col("corridor") == name)
                    & (pl.col("split") == split)
                    & (pl.col("horizon") == horizon)
                ).row(0, named=True)["sha256"]
                assert f'"{name}|{split}": "{digest}"' in src

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_no_two_folds_gate_on_the_same_training_population(
        self, group_key, horizon
    ):
        """The point of the exercise. Identical train digests would mean the
        three origins trained on the same days and the comparison is vacuous."""
        manifest = pl.read_csv(MANIFEST_CSV)
        digests = set()
        for fold in ROLLING_FOLDS:
            for name, _emp in GROUPS[group_key]["corridors"]:
                digests.add(
                    manifest.filter(
                        (pl.col("fold") == fold.name)
                        & (pl.col("corridor") == name)
                        & (pl.col("split") == "train")
                        & (pl.col("horizon") == horizon)
                    ).row(0, named=True)["sha256"]
                )
        assert len(digests) == len(ROLLING_FOLDS) * len(GROUPS[group_key]["corridors"])

    @pytest.mark.parametrize("group_key,horizon", ALL_CASES)
    def test_rolling_notebooks_announce_they_are_not_the_published_result(
        self, group_key, horizon
    ):
        for fold in ROLLING_FOLDS:
            markdown = "\n".join(
                c["source"]
                for c in _notebook(group_key, horizon, fold)["cells"]
                if c["cell_type"] == "markdown"
            )
            if fold.name == "main":
                assert "rolling origin" not in markdown.lower()
            else:
                assert "no** produce el resultado publicado" in markdown
                assert str(fold.test_start) in markdown
