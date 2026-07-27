"""Tests for src/build_notebook_20_xgb_paired.py — keyed XGB paired export builder.

Covers:
  AC-NB20-1: the builder exits 0 and writes
             notebooks/20_xgb_paired/20_xgb_paired.ipynb.
  AC-NB20-2: two consecutive runs are byte-identical (no cell-ID flutter), for
             both the notebook and kernel-metadata.json.
  AC-NB20-3: every cell ID matches 'cell-20-*'.
  AC-NB20-4: kernel-metadata.json is well formed, CPU-only, private, and pins the
             THREE required kernel sources.
  AC-NB20-5: the frozen input-hash gate is present for all four inputs, resolved
             through _resolve_input, and the empty-atypical-set guard is emitted.
  AC-NB20-6: output filenames contain neither '_results_' (globbed by
             build_degradation_curve.py) nor '_residuals_h' (globbed by
             evaluation/paired_audit.py).
  AC-NB20-7: the notebook writes NO parquet — in particular it can never rewrite
             the hash-frozen headways_E4.parquet that NB17/NB18/NB19 pin.
  AC-NB20-8: the embedded modules appear in dependency order and the export loop
             covers all three corridors at all four horizons.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import nbformat
import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks" / "20_xgb_paired"
NOTEBOOK_20_PATH = NB_DIR / "20_xgb_paired.ipynb"
KERNEL_META_PATH = NB_DIR / "kernel-metadata.json"
BUILDER_20 = ROOT / "src" / "build_notebook_20_xgb_paired.py"

PAIRED_CSV_NAME = "xgb_paired_persample_test.csv"
PROVENANCE_CSV_NAME = "xgb_paired_search_provenance.csv"

REQUIRED_KERNEL_SOURCES = [
    "alexhuaracha/04-preprocessing",
    "alexhuaracha/02-eda-corridors",
    "alexhuaracha/16-e4-data-baselines",
]

# The same frozen digests the DL notebooks pin (tests/test_notebook_input_gate.py).
FROZEN_HASHES = {
    "headways_E2.parquet": "82a34eaffc79cd82346d4595a2e72f5d3ffb751ed37fa0fc0cde3a8f8fb345d4",
    "headways_E59.parquet": "0b5f5593caaa94e4e6af7da672bc2cad7b49b69b7cbd0a22092f15700a89a448",
    "headways_E4.parquet": "1dde7f38eea9bc7d9941c17cbc3d326cb864e70be815a1a7e3d0ae2691f19273",
    "atypical_days.csv": "2054245cc830e58b9397b75ea3b55d034581046b64e73b1630ca7d464e3ecb86",
}


def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_20_xgb_paired"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


@pytest.fixture(scope="module")
def notebook() -> nbformat.NotebookNode:
    result = _run_builder()
    assert result.returncode == 0, (
        f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    with open(NOTEBOOK_20_PATH, encoding="utf-8") as handle:
        return nbformat.read(handle, as_version=4)


@pytest.fixture(scope="module")
def code_source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(
        cell["source"] for cell in notebook.cells if cell["cell_type"] == "code"
    )


class TestNotebook20Builder:
    """AC-NB20-1/2/3: the builder runs, is idempotent, and uses stable cell IDs."""

    def test_builder_exits_zero_and_writes_notebook(self):
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_20_xgb_paired failed with return code "
            f"{result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert NOTEBOOK_20_PATH.exists(), (
            f"Expected notebook at {NOTEBOOK_20_PATH} but file does not exist"
        )

    def test_cell_ids_have_cell_20_prefix(self, notebook):
        stable_pattern = re.compile(r"^cell-20-")
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in notebook.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs do not match 'cell-20-*': {bad_ids}. All cell "
            "IDs must be explicit to prevent git flutter on re-generation."
        )

    def test_cell_ids_are_unique(self, notebook):
        ids = [cell.get("id") for cell in notebook.cells]
        duplicates = sorted({cell_id for cell_id in ids if ids.count(cell_id) > 1})
        assert not duplicates, f"Duplicate cell IDs: {duplicates}"

    def test_second_run_byte_identical(self):
        result1 = _run_builder()
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        nb_first = NOTEBOOK_20_PATH.read_bytes()
        meta_first = KERNEL_META_PATH.read_bytes()

        result2 = _run_builder()
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )

        assert nb_first == NOTEBOOK_20_PATH.read_bytes(), (
            "build_notebook_20_xgb_paired is NOT idempotent: two consecutive runs "
            "produced different notebook bytes."
        )
        assert meta_first == KERNEL_META_PATH.read_bytes(), (
            "kernel-metadata.json must be byte-identical across consecutive runs"
        )


class TestNotebook20KernelMetadata:
    """AC-NB20-4: metadata is well formed, CPU-only, private, three sources."""

    @pytest.fixture(scope="class")
    def meta(self, notebook) -> dict:
        return json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))

    def test_code_file_matches_notebook(self, meta):
        assert meta["code_file"] == NOTEBOOK_20_PATH.name

    def test_id_and_title_agree(self, meta):
        assert meta["id"] == "alexhuaracha/20-xgb-paired-export"
        # Kaggle slugifies the title; a title that does not slugify to the id
        # silently creates a second kernel (see the NB16 note).
        slug = re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-")
        assert slug == meta["id"].split("/", 1)[1], (
            f"title {meta['title']!r} slugifies to {slug!r}, which does not match "
            f"the kernel id {meta['id']!r}"
        )

    def test_kernel_sources_cover_all_three_inputs(self, meta):
        assert meta["kernel_sources"] == REQUIRED_KERNEL_SOURCES, (
            "kernel_sources must pin 04-preprocessing (E2/E59 parquets), "
            "02-eda-corridors (atypical_days.csv) and 16-e4-data-baselines "
            f"(E4 parquet); got {meta['kernel_sources']!r}"
        )

    def test_cpu_only(self, meta):
        assert meta["enable_gpu"] is False, (
            f"enable_gpu must be False (CPU baselines), got {meta['enable_gpu']!r}"
        )
        assert "accelerator" not in meta, (
            f"accelerator must be absent for a CPU kernel, got {meta.get('accelerator')!r}"
        )

    def test_base_fields(self, meta):
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_internet"] is True
        assert meta["dataset_sources"] == []
        assert meta["competition_sources"] == []


class TestNotebook20InputGate:
    """AC-NB20-5: frozen hash gate + required, non-empty atypical calendar."""

    def test_all_four_inputs_are_hash_pinned(self, code_source):
        assert "INPUT_HASHES" in code_source, "setup cell must pin frozen input hashes"
        assert "_resolve_input(" in code_source, (
            "inputs must resolve through the hash-verifying gate"
        )
        for name, digest in FROZEN_HASHES.items():
            assert name in code_source, f"input missing from the gate: {name}"
            assert digest in code_source, (
                f"frozen hash missing for {name}: {digest[:12]}…"
            )

    def test_gate_fails_closed(self, code_source):
        assert "raise FileNotFoundError(f\"Required input not found anywhere" in code_source
        assert "matches its frozen SHA-256" in code_source, (
            "a candidate whose bytes differ must raise, not be used"
        )

    def test_headways_resolve_through_the_gate(self, code_source):
        assert '_resolve_input(f"headways_E{empresa_id}.parquet")' in code_source, (
            "the three headways parquets must be resolved through the hash gate, "
            "not by a bare rglob — the export is only comparable to the DL "
            "residuals if XGB was fitted on byte-identical inputs"
        )

    def test_atypical_days_is_required_and_non_empty(self, code_source):
        assert 'atypical_path = _resolve_input("atypical_days.csv")' in code_source
        assert "atypical_path = None" not in code_source, (
            "no silent atypical fallback may exist"
        )
        assert "if not atypical_dates:" in code_source, (
            "the notebook must hard-fail when the atypical set parses empty"
        )
        assert "raise ValueError(f\"atypical_days.csv parsed to an empty date set" in (
            code_source
        )


class TestNotebook20Outputs:
    """AC-NB20-6/7: safe output filenames, CSV only, never a parquet."""

    def test_output_filenames_are_declared(self, code_source):
        assert PAIRED_CSV_NAME in code_source
        assert PROVENANCE_CSV_NAME in code_source

    @pytest.mark.parametrize("name", [PAIRED_CSV_NAME, PROVENANCE_CSV_NAME])
    def test_output_filename_avoids_analysis_layer_globs(self, name: str):
        assert "_results_" not in name, (
            f"{name} must not contain '_results_': build_degradation_curve.py globs "
            "'*_results_*.csv' and evaluation/degradation.py hard-requires the tidy "
            "metrics schema, so this file would crash the degradation build or "
            "contaminate consolidated_multihorizon.csv and Figure 1"
        )
        assert "_residuals_h" not in name, (
            f"{name} must not match '*_residuals_h*.csv', which "
            "evaluation/paired_audit.py globs and parses as a DL residual file"
        )
        assert "_multiseed_" not in name, (
            f"{name} must not match '*_multiseed_*.csv' (evaluation/multiseed.py)"
        )

    def test_every_emitted_output_path_avoids_the_analysis_globs(self, code_source):
        """Scan the actual `OUTPUT_DIR / "..."` targets, not incidental prose.

        Docstrings legitimately NAME the contaminated artifacts they explain
        (e.g. baselines_results_multih.csv), so the guard must look at emitted
        paths only.
        """
        emitted = re.findall(r'OUTPUT_DIR\s*/\s*"([^"]+)"', code_source)
        assert emitted, "no OUTPUT_DIR targets found — did the setup cell change?"
        assert sorted(emitted) == sorted([PAIRED_CSV_NAME, PROVENANCE_CSV_NAME]), (
            f"unexpected emitted output paths: {emitted}"
        )
        for name in emitted:
            assert "_results_" not in name
            assert "_residuals_h" not in name
            assert "_multiseed_" not in name
            assert name.endswith(".csv"), (
                f"{name} is not a CSV — this notebook writes CSV only"
            )

    def test_notebook_never_writes_parquet(self, code_source):
        assert "write_parquet" not in code_source, (
            "this notebook must not write any parquet. headways_E4.parquet in "
            "particular is hash-frozen in the INPUT_HASHES of NB17/NB18/NB19: a "
            "non-byte-identical rewrite makes all three E4 DL notebooks fail "
            "closed and forces a full E4 GPU retrain."
        )

    def test_notebook_writes_only_the_two_expected_csvs(self, code_source):
        write_targets = set(re.findall(r"\.write_csv\((\w+)\)", code_source))
        assert write_targets == {"PAIRED_OUT", "PROVENANCE_OUT"}, (
            f"unexpected CSV write targets: {sorted(write_targets)}"
        )


class TestNotebook20Structure:
    """AC-NB20-8: embed order and full corridor x horizon coverage."""

    EMBEDS = [
        "cell-20-embed-splits",
        "cell-20-embed-metrics",
        "cell-20-embed-statistical",
        "cell-20-embed-context",
        "cell-20-embed-fitted",
        "cell-20-embed-harness",
        "cell-20-embed-paired-export",
    ]

    def test_embedded_modules_appear_in_dependency_order(self, notebook):
        ids = [cell.get("id") for cell in notebook.cells]
        positions = []
        for embed_id in self.EMBEDS:
            assert embed_id in ids, f"missing embed cell: {embed_id}"
            positions.append(ids.index(embed_id))
        assert positions == sorted(positions), (
            "embeds must be ordered splits → metrics → statistical → "
            "context_features → fitted → harness → paired_export; "
            f"got positions {positions}"
        )

    def test_embedded_modules_are_import_free(self, notebook):
        """_strip_relative_imports must remove intra-package imports."""
        by_id = {cell.get("id"): cell for cell in notebook.cells}
        for embed_id in self.EMBEDS:
            src = by_id[embed_id]["source"]
            assert "from ." not in src, f"{embed_id} kept a relative import"
            assert "from src." not in src, f"{embed_id} kept an absolute src import"

    def test_paired_export_module_is_embedded_verbatim(self, notebook):
        """The embedded copy must be the real module, not a notebook-local rewrite."""
        by_id = {cell.get("id"): cell for cell in notebook.cells}
        src = by_id["cell-20-embed-paired-export"]["source"]
        assert "def paired_xgb_test_frame(" in src
        assert "def export_paired_xgb(" in src
        assert "XGB_PAIRED_COLUMNS" in src
        assert "pair_rank" in src

    def test_embedded_modules_execute_in_a_flat_namespace(self, notebook):
        """The Kaggle cell namespace is FLAT.

        Executing the embeds in the emitted order must therefore be enough to get
        a working `paired_xgb_test_frame` — this catches a wrong embed order or a
        stripped import that the source-text assertions above cannot see.
        """
        by_id = {cell.get("id"): cell for cell in notebook.cells}
        namespace: dict = {}
        for embed_id in self.EMBEDS:
            exec(compile(by_id[embed_id]["source"], embed_id, "exec"), namespace)

        # Two test rows sharing (t, direction) — the case `t` alone cannot key.
        shared_t = datetime(2024, 2, 10, 8, 0, 0)
        frame = pl.DataFrame(
            [
                {"empresaid": 2, "t": shared_t, "direction": -1, "pair_rank": 1,
                 "delta_t_min": 6.0, "split": "test",
                 "y_pred_b1": 5.0, "y_pred_b5_xgb": 5.5},
                {"empresaid": 2, "t": shared_t, "direction": -1, "pair_rank": 0,
                 "delta_t_min": 5.0, "split": "test",
                 "y_pred_b1": 4.0, "y_pred_b5_xgb": 4.5},
                {"empresaid": 2, "t": datetime(2023, 11, 1, 8, 0, 0), "direction": 1,
                 "pair_rank": 0, "delta_t_min": 7.0, "split": "train",
                 "y_pred_b1": 6.0, "y_pred_b5_xgb": 6.5},
            ]
        ).with_columns(pl.col("pair_rank").cast(pl.Int32))

        out = namespace["paired_xgb_test_frame"](frame, "E2", horizon=3)
        assert out.columns == namespace["XGB_PAIRED_COLUMNS"]
        assert out.height == 2, "only the two TEST rows may survive"
        assert out["pair_rank"].to_list() == [0, 1], "sorted by the full key"
        assert out.select(namespace["XGB_PAIRED_KEY"]).n_unique() == 2

    def test_export_loop_covers_three_corridors_and_four_horizons(self, notebook):
        by_id = {cell.get("id"): cell for cell in notebook.cells}
        setup = by_id["cell-20-setup"]["source"]
        assert "HORIZONS = [1, 3, 5, 10]" in setup
        assert 'CORRIDORS = [("E2", 2), ("E59", 59), ("E4", 4)]' in setup

        loop = by_id["cell-20-run-export"]["source"]
        assert "for label, empresa_id in CORRIDORS:" in loop
        assert "for h in HORIZONS:" in loop
        assert "export_paired_xgb(" in loop
        assert "horizon=h" in loop
        assert "atypical_dates=atypical_dates" in loop

    def test_key_uniqueness_is_asserted_at_runtime(self, notebook):
        by_id = {cell.get("id"): cell for cell in notebook.cells}
        verify = by_id["cell-20-verify"]["source"]
        assert "XGB_PAIRED_KEY" in verify
        assert "n_unique()" in verify
        assert "raise ValueError(" in verify, (
            "a non-unique exported key must abort the run — that is the defect "
            "this notebook exists to fix"
        )

    def test_builder_docstring_documents_the_parquet_prohibition(self):
        builder_src = BUILDER_20.read_text(encoding="utf-8")
        assert "headways_E4.parquet" in builder_src
        assert PAIRED_CSV_NAME in builder_src
