"""Tests for src/build_notebook_10.py — Baselines multi-horizonte builder.

Covers:
  AC-NB10-1: python -m src.build_notebook_10 exits 0 and produces
              notebooks/10_baselines_multihorizon/10_baselines_multihorizon.ipynb.
  AC-NB10-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB10-3: every cell ID matches 'cell-10-*' (no random UUIDs).
  AC-NB10-4: cell 'cell-10-run-harness' loops over HORIZONS and adds 'horizon' column.
  AC-NB10-5: setup cell references 'baselines_results_multih.csv' as output filename.
  AC-NB10-6: kernel-metadata.json is written next to the notebook with correct fields.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks" / "10_baselines_multihorizon"
NOTEBOOK_10_PATH = NB_DIR / "10_baselines_multihorizon.ipynb"
KERNEL_META_PATH = NB_DIR / "kernel-metadata.json"
BUILDER_10 = ROOT / "src" / "build_notebook_10.py"


class TestNotebook10Builder:
    """Verify build_notebook_10.py produces a correct, stable multi-horizon notebook.

    AC-NB10-1: builder exits 0 and produces the notebook at the expected path.
    AC-NB10-2: two consecutive runs leave the working tree clean (byte-identical).
    AC-NB10-3: all cell IDs have the 'cell-10-' prefix (no uuid4 IDs).
    AC-NB10-4: run-harness cell loops over HORIZONS and appends 'horizon' column.
    AC-NB10-5: setup cell (or builder source) references 'baselines_results_multih.csv'.
    """

    def test_builder_exits_zero_and_writes_notebook(self):
        """AC-NB10-1: python -m src.build_notebook_10 exits 0 and writes the notebook."""
        result = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_10"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, (
            f"src.build_notebook_10 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert NOTEBOOK_10_PATH.exists(), (
            f"Expected notebook at {NOTEBOOK_10_PATH} but file does not exist"
        )

    def test_cell_ids_have_cell_10_prefix(self):
        """AC-NB10-3: every cell ID must match the pattern 'cell-10-*' (no uuid4)."""
        if not NOTEBOOK_10_PATH.exists():
            subprocess.run(
                [sys.executable, "-m", "src.build_notebook_10"],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                check=True,
            )
        stable_pattern = re.compile(r"^cell-10-")
        with open(NOTEBOOK_10_PATH, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs do not match 'cell-10-*' pattern: {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-10-<name>' to prevent "
            "git flutter on re-generation."
        )

    def test_second_run_byte_identical(self):
        """AC-NB10-2: two consecutive runs must produce byte-identical output."""
        result1 = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_10"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        bytes_after_first = NOTEBOOK_10_PATH.read_bytes()

        result2 = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_10"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        bytes_after_second = NOTEBOOK_10_PATH.read_bytes()

        assert bytes_after_first == bytes_after_second, (
            "build_notebook_10 is NOT idempotent: two consecutive runs produced "
            "different byte content. Each cell must have an explicit stable ID "
            "like 'cell-10-setup', not a random uuid4."
        )

    def test_run_harness_cell_loops_horizons(self):
        """AC-NB10-4: cell 'cell-10-run-harness' must contain the multi-horizon loop."""
        if not NOTEBOOK_10_PATH.exists():
            subprocess.run(
                [sys.executable, "-m", "src.build_notebook_10"],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                check=True,
            )
        with open(NOTEBOOK_10_PATH, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        harness_cell = next(
            (cell for cell in nb.cells if cell.get("id") == "cell-10-run-harness"),
            None,
        )
        assert harness_cell is not None, (
            "No cell with id='cell-10-run-harness' found in the notebook."
        )
        src = harness_cell["source"]
        assert "HORIZONS = [1, 3, 5, 10]" in src, (
            "run-harness cell must define HORIZONS = [1, 3, 5, 10]"
        )
        assert "horizon=h" in src, (
            "run-harness cell must pass 'horizon=h' to evaluate_corridor calls"
        )
        assert '.alias("horizon")' in src, (
            "run-harness cell must add a 'horizon' column via .alias(\"horizon\")"
        )

    def test_write_csv_targets_multih_filename(self):
        """AC-NB10-5: builder or setup cell must reference 'baselines_results_multih.csv'."""
        # Check in builder source (always accessible, no need to generate notebook first)
        builder_src = BUILDER_10.read_text(encoding="utf-8")
        assert "baselines_results_multih.csv" in builder_src, (
            "build_notebook_10.py must reference 'baselines_results_multih.csv' "
            "as the output CSV filename (CSV_OUT variable)."
        )


# ---------------------------------------------------------------------------
# AC-NB10-6: kernel-metadata.json written next to the notebook
# ---------------------------------------------------------------------------

class TestNotebook10KernelMetadata:
    """Verify build_notebook_10.py writes a correct kernel-metadata.json."""

    def _ensure_built(self) -> None:
        if not KERNEL_META_PATH.exists() or not NOTEBOOK_10_PATH.exists():
            result = subprocess.run(
                [sys.executable, "-m", "src.build_notebook_10"],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def test_kernel_metadata_exists(self):
        """AC-NB10-6a: kernel-metadata.json must exist in the same dir as the notebook."""
        self._ensure_built()
        assert KERNEL_META_PATH.exists(), (
            f"Expected kernel-metadata.json at {KERNEL_META_PATH} but file does not exist"
        )

    def test_kernel_metadata_code_file_matches_notebook(self):
        """AC-NB10-6b: code_file field must match the actual .ipynb filename."""
        self._ensure_built()
        meta = json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))
        assert meta["code_file"] == "10_baselines_multihorizon.ipynb", (
            f"code_file must be '10_baselines_multihorizon.ipynb', got: {meta['code_file']!r}"
        )

    def test_kernel_metadata_id(self):
        """AC-NB10-6c: id must be 'alexhuaracha/10-baselines-multihorizon'."""
        self._ensure_built()
        meta = json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))
        assert meta["id"] == "alexhuaracha/10-baselines-multihorizon", (
            f"id must be 'alexhuaracha/10-baselines-multihorizon', got: {meta['id']!r}"
        )

    def test_kernel_metadata_kernel_sources_cpu_only(self):
        """AC-NB10-6d: kernel_sources == ["alexhuaracha/04-preprocessing"] and enable_gpu is False."""
        self._ensure_built()
        meta = json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))
        assert meta["kernel_sources"] == ["alexhuaracha/04-preprocessing"], (
            f"kernel_sources must be ['alexhuaracha/04-preprocessing'], got: {meta['kernel_sources']!r}"
        )
        assert meta["enable_gpu"] is False, (
            f"enable_gpu must be False (CPU baselines), got: {meta['enable_gpu']!r}"
        )
        assert "accelerator" not in meta, (
            f"accelerator field must NOT be present for CPU kernel, got: {meta.get('accelerator')!r}"
        )

    def test_kernel_metadata_base_fields(self):
        """AC-NB10-6e: language, kernel_type, is_private, enable_internet have correct values."""
        self._ensure_built()
        meta = json.loads(KERNEL_META_PATH.read_text(encoding="utf-8"))
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_internet"] is True
        assert meta["dataset_sources"] == []
        assert meta["competition_sources"] == []

    def test_kernel_metadata_deterministic(self):
        """AC-NB10-6f: two consecutive runs produce byte-identical kernel-metadata.json."""
        result1 = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_10"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result1.returncode == 0
        bytes_first = KERNEL_META_PATH.read_bytes()

        result2 = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_10"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result2.returncode == 0
        bytes_second = KERNEL_META_PATH.read_bytes()

        assert bytes_first == bytes_second, (
            "kernel-metadata.json must be byte-identical across consecutive runs"
        )
