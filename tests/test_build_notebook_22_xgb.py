"""Integrity guard for the XGBoost refit notebook.

The notebook embeds ``baselines/fitted.py`` for its frozen search sampler, but
that module also carries ``_build_features`` — the positional-lag builder that is
the whole reason for the refit. Shipping the defect next to its replacement is
acceptable only while something asserts the defect is never invoked.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import polars as pl
import pytest

from src.build_notebook_22_xgb_contiguous import (
    CORRIDORS,
    HORIZONS,
    KERNEL_ID,
    MANIFEST_CSV,
    NOTEBOOK_NAME,
    OUT_DIR,
    build,
)

# Cells authored by this builder, as opposed to verbatim embedded modules.
AUTHORED_CELLS = {"cell-22-setup", "cell-22-prepare", "cell-22-fit"}


@pytest.fixture(scope="module", autouse=True)
def built():
    build()


@pytest.fixture(scope="module")
def notebook():
    return nbf.read(OUT_DIR / NOTEBOOK_NAME, as_version=4)


@pytest.fixture(scope="module")
def all_code(notebook):
    return "\n".join(
        c["source"] for c in notebook["cells"] if c["cell_type"] == "code"
    )


@pytest.fixture(scope="module")
def authored_code(notebook):
    return "\n".join(
        c["source"] for c in notebook["cells"]
        if c["cell_type"] == "code" and c.get("id") in AUTHORED_CELLS
    )


class TestCompiles:
    def test_every_code_cell_is_valid_python(self, notebook):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            try:
                compile(cell["source"], f"<{cell['id']}>", "exec")
            except SyntaxError as exc:
                pytest.fail(f"{cell['id']}: {exc}")

    def test_no_intra_package_imports_survive(self, all_code):
        assert "from src." not in all_code
        assert "from ..data" not in all_code


class TestDefectiveBuilderIsNeverCalled:
    """`fitted._build_features` uses positional shifts — the §3 defect."""

    @pytest.mark.parametrize(
        "symbol", ["_build_features(", "fit_predict_b5_xgb(", "predict_b5_xgb("]
    )
    def test_not_invoked_by_authored_cells(self, authored_code, symbol):
        assert symbol not in authored_code, (
            f"{symbol} carries the positional-lag defect this notebook exists to remove"
        )

    def test_the_contiguous_builder_is_the_one_used(self, authored_code):
        assert "build_contiguous_features(" in authored_code


class TestSharedPopulationGate:
    def test_gate_present_and_fails_closed(self, authored_code):
        assert "INDEX_DIGESTS" in authored_code
        assert "SHARED-POPULATION GATE FAILED" in authored_code
        assert "raise ValueError(" in authored_code

    def test_covers_every_corridor_split_horizon(self, all_code):
        manifest = pl.read_csv(MANIFEST_CSV)
        for name, _emp in CORRIDORS:
            for split in ("train", "val", "test"):
                for horizon in HORIZONS:
                    row = manifest.filter(
                        (pl.col("fold") == "main")
                        & (pl.col("corridor") == name)
                        & (pl.col("split") == split)
                        & (pl.col("horizon") == horizon)
                    )
                    assert row.height == 1
                    digest = row.row(0, named=True)["sha256"]
                    assert f'"{name}|{split}|{horizon}": "{digest}"' in all_code, (
                        f"stale or missing digest for {name}/{split}/h{horizon}"
                    )

    def test_digests_match_the_lstm_notebooks(self, all_code):
        """Both families must gate against the same frozen digests.

        If these ever diverge, "same population" silently stops being true —
        which is exactly the failure audit §2.1 patched with a join.
        """
        lstm_nb = nbf.read(
            Path("notebooks/21_lstm_contiguous/e2e59/h3/21_lstm_contiguous_e2e59_h3.ipynb"),
            as_version=4,
        )
        lstm_code = "\n".join(
            c["source"] for c in lstm_nb["cells"] if c["cell_type"] == "code"
        )
        manifest = pl.read_csv(MANIFEST_CSV)
        for name in ("E2", "E59"):
            for split in ("train", "val", "test"):
                digest = manifest.filter(
                    (pl.col("fold") == "main")
                    & (pl.col("corridor") == name)
                    & (pl.col("split") == split)
                    & (pl.col("horizon") == 3)
                ).row(0, named=True)["sha256"]
                assert f'"{name}|{split}": "{digest}"' in lstm_code
                assert f'"{name}|{split}|3": "{digest}"' in all_code


class TestFrozenSearchIsPreserved:
    def test_search_constants_are_not_overridden(self, authored_code):
        assert "sample_search_configs(SEARCH_N_CONFIGS, seed=SEARCH_SEED)" in authored_code
        for redefinition in ("SEARCH_SEED =", "SEARCH_N_CONFIGS =", "SEARCH_SPACE ="):
            assert redefinition not in authored_code, (
                "the frozen search must not be re-rolled by the notebook"
            )

    def test_selection_never_sees_test(self, authored_code):
        """The test DMatrix must not appear in any `evals` list."""
        for line in authored_code.splitlines():
            if "evals=" in line:
                assert "dte" not in line, f"test split leaked into selection: {line}"

    def test_search_provenance_is_written(self, authored_code):
        assert "SEARCH_OUT" in authored_code
        assert "search_seed" in authored_code


class TestExportContract:
    def test_residuals_use_the_canonical_schema_and_are_verified(self, authored_code):
        assert "RESIDUAL_COLUMNS" in authored_code
        assert "assert_key_is_unique(residuals)" in authored_code

    def test_persistence_is_exported_alongside(self, authored_code):
        assert "y_pred_persist" in authored_code


class TestAtypicalFlagIsGone:
    def test_not_a_required_input(self, authored_code):
        assert '"atypical_days.csv":' not in authored_code
        assert '_resolve_input("atypical_days.csv")' not in authored_code

    def test_no_atypical_calendar_is_passed(self, authored_code):
        assert "atypical_dates=" not in authored_code


class TestKernelMetadata:
    def test_cpu_kernel_with_the_right_source(self):
        meta = json.loads((OUT_DIR / "kernel-metadata.json").read_text())
        assert meta["id"] == KERNEL_ID
        assert meta["enable_gpu"] is False, "the XGBoost refit runs on CPU"
        assert meta["kernel_sources"] == ["alexhuaracha/04-preprocessing"]
        assert "alexhuaracha/02-eda-corridors" not in meta["kernel_sources"]

    def test_does_not_collide_with_frozen_kernels(self):
        meta = json.loads((OUT_DIR / "kernel-metadata.json").read_text())
        for frozen in ("10-baselines", "16-e4-data", "20-xgb-paired"):
            assert frozen not in meta["id"]
