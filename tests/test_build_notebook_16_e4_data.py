"""Tests for src/build_notebook_16_e4_data.py — E4 data + baselines builder.

E4 is the third (validation) corridor for external validity. NB16 is a single
CPU kernel that does E4 preprocessing AND E4 baselines, mirroring the embed
pattern of build_notebook_04.py (preprocessing) + build_notebook_10.py
(baselines), but scoped to empresaid=4 only.

Covers:
  AC-NB16-1: python -m src.build_notebook_16_e4_data exits 0 and produces
              notebooks/16_e4_data/16_e4_data.ipynb.
  AC-NB16-2: two consecutive runs are byte-identical (idempotency / no flutter).
  AC-NB16-3: every cell ID matches 'cell-16-*' (no random UUIDs).
  AC-NB16-4: setup cell defines EMPRESAS = [4] (E4 ONLY — frozen corridors untouched).
  AC-NB16-5: builder embeds the preprocessing AND baselines library modules.
  AC-NB16-6: run-harness cell loops over HORIZONS and calls run_corridor
              with corridor "E4" and horizon=h.
  AC-NB16-7: builder references the output filename 'baselines_E4_results_multih.csv'.
  AC-NB16-8: kernel-metadata.json is written next to the notebook with correct fields.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks" / "16_e4_data"
NOTEBOOK_16_PATH = NB_DIR / "16_e4_data.ipynb"
KERNEL_META_PATH = NB_DIR / "kernel-metadata.json"
BUILDER_16 = ROOT / "src" / "build_notebook_16_e4_data.py"


def _run_builder():
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_16_e4_data"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _ensure_built() -> None:
    if not NOTEBOOK_16_PATH.exists() or not KERNEL_META_PATH.exists():
        result = _run_builder()
        assert result.returncode == 0, (
            f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestNotebook16Builder:
    """Verify build_notebook_16_e4_data.py produces a correct, stable notebook."""

    def test_builder_exits_zero_and_writes_notebook(self):
        """AC-NB16-1: builder exits 0 and writes the notebook at the expected path."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_16_e4_data failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert NOTEBOOK_16_PATH.exists(), (
            f"Expected notebook at {NOTEBOOK_16_PATH} but file does not exist"
        )

    def test_second_run_byte_identical(self):
        """AC-NB16-2: two consecutive runs must produce byte-identical output."""
        result1 = _run_builder()
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        bytes_after_first = NOTEBOOK_16_PATH.read_bytes()

        result2 = _run_builder()
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        bytes_after_second = NOTEBOOK_16_PATH.read_bytes()

        assert bytes_after_first == bytes_after_second, (
            "build_notebook_16_e4_data is NOT idempotent: two consecutive runs "
            "produced different byte content. Each cell must have an explicit "
            "stable ID like 'cell-16-setup', not a random uuid4."
        )

    def test_cell_ids_have_cell_16_prefix(self):
        """AC-NB16-3: every cell ID must match the pattern 'cell-16-*' (no uuid4)."""
        _ensure_built()
        stable_pattern = re.compile(r"^cell-16-")
        with open(NOTEBOOK_16_PATH, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs do not match 'cell-16-*' pattern: {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-16-<name>' to prevent "
            "git flutter on re-generation."
        )

    def test_setup_cell_scopes_e4_only(self):
        """AC-NB16-4: setup cell must define EMPRESAS = [4] (E4 ONLY)."""
        _ensure_built()
        with open(NOTEBOOK_16_PATH, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        setup_cell = next(
            (cell for cell in nb.cells if cell.get("id") == "cell-16-setup"),
            None,
        )
        assert setup_cell is not None, (
            "No cell with id='cell-16-setup' found in the notebook."
        )
        assert "EMPRESAS = [4]" in setup_cell["source"], (
            "setup cell must define EMPRESAS = [4] — NB16 must NOT touch the "
            "frozen corridors E2/E59."
        )

    def test_builder_embeds_library_modules(self):
        """AC-NB16-5: builder must embed preprocessing AND baselines modules."""
        builder_src = BUILDER_16.read_text(encoding="utf-8")
        # Preprocessing modules (mirror of build_notebook_04.py)
        for mod in (
            "config.py",
            "corridor.py",
            "projection.py",
            "direction.py",
            "trips.py",
            "headways.py",
            "pipeline.py",
        ):
            assert mod in builder_src, (
                f"builder must embed preprocessing module {mod!r}"
            )
        # Baselines / evaluation modules (mirror of build_notebook_10.py)
        for mod in (
            "evaluation/splits.py",
            "evaluation/metrics.py",
            "baselines/statistical.py",
            "baselines/fitted.py",
            "baselines/harness.py",
        ):
            assert mod in builder_src, (
                f"builder must embed evaluation/baselines module {mod!r}"
            )

    def test_run_harness_cell_evaluates_e4_multihorizon(self):
        """AC-NB16-6: run-harness cell must loop horizons and evaluate corridor 'E4'."""
        _ensure_built()
        with open(NOTEBOOK_16_PATH, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        harness_cell = next(
            (cell for cell in nb.cells if cell.get("id") == "cell-16-run-harness"),
            None,
        )
        assert harness_cell is not None, (
            "No cell with id='cell-16-run-harness' found in the notebook."
        )
        src = harness_cell["source"]
        assert "HORIZONS = [1, 3, 5, 10]" in src, (
            "run-harness cell must define HORIZONS = [1, 3, 5, 10]"
        )
        assert "horizon=h" in src, (
            "run-harness cell must pass 'horizon=h' to run_corridor"
        )
        assert '"E4"' in src, (
            "run-harness cell must call run_corridor with corridor label 'E4'"
        )
        assert "run_corridor" in src, (
            "run-harness cell must call run_corridor"
        )
        assert '.alias("horizon")' in src, (
            'run-harness cell must add a "horizon" column via .alias("horizon")'
        )

    def test_builder_references_e4_results_csv(self):
        """AC-NB16-7: builder must reference 'baselines_E4_results_multih.csv'."""
        builder_src = BUILDER_16.read_text(encoding="utf-8")
        assert "baselines_E4_results_multih.csv" in builder_src, (
            "build_notebook_16_e4_data.py must reference "
            "'baselines_E4_results_multih.csv' as the output CSV filename."
        )


class TestNotebook16KernelMetadata:
    """Verify build_notebook_16_e4_data.py writes a correct kernel-metadata.json."""

    def test_kernel_metadata_exists(self):
        """AC-NB16-8a: kernel-metadata.json must exist next to the notebook."""
        _ensure_built()
        assert KERNEL_META_PATH.exists(), (
            f"Expected kernel-metadata.json at {KERNEL_META_PATH} but file does not exist"
        )

    def test_kernel_metadata_code_file_matches_notebook(self):
        """AC-NB16-8b: code_file field must match the actual .ipynb filename."""
        _ensure_built()
        meta = json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))
        assert meta["code_file"] == "16_e4_data.ipynb", (
            f"code_file must be '16_e4_data.ipynb', got: {meta['code_file']!r}"
        )

    def test_kernel_metadata_id(self):
        """AC-NB16-8c: id must be 'alexhuaracha/16-e4-data-baselines'.

        The id must slugify from the title, else Kaggle creates the kernel under
        a title-derived slug. Title "16 E4 data baselines" → 16-e4-data-baselines.
        """
        _ensure_built()
        meta = json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))
        assert meta["id"] == "alexhuaracha/16-e4-data-baselines", (
            f"id must be 'alexhuaracha/16-e4-data-baselines', got: {meta['id']!r}"
        )

    def test_kernel_metadata_cpu_and_dataset_source(self):
        """AC-NB16-8d: CPU kernel reading the clean dataset (not a kernel source)."""
        _ensure_built()
        meta = json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))
        assert meta["enable_gpu"] is False, (
            f"enable_gpu must be False (CPU kernel), got: {meta['enable_gpu']!r}"
        )
        assert "accelerator" not in meta, (
            f"accelerator field must NOT be present for CPU kernel, got: {meta.get('accelerator')!r}"
        )
        assert meta["dataset_sources"] == [
            "alexhuaracha/multibus-headway-forecast-clean"
        ], (
            "dataset_sources must be ['alexhuaracha/multibus-headway-forecast-clean'] "
            f"(same clean dataset NB04 reads), got: {meta['dataset_sources']!r}"
        )

    def test_kernel_metadata_base_fields(self):
        """AC-NB16-8e: language, kernel_type, is_private, enable_internet correct."""
        _ensure_built()
        meta = json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_internet"] is True
        assert meta["competition_sources"] == []

    def test_kernel_metadata_deterministic(self):
        """AC-NB16-8f: two consecutive runs produce byte-identical kernel-metadata.json."""
        result1 = _run_builder()
        assert result1.returncode == 0
        bytes_first = KERNEL_META_PATH.read_bytes()

        result2 = _run_builder()
        assert result2.returncode == 0
        bytes_second = KERNEL_META_PATH.read_bytes()

        assert bytes_first == bytes_second, (
            "kernel-metadata.json must be byte-identical across consecutive runs"
        )
