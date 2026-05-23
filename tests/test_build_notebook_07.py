"""Tests for src/build_notebook_07.py — W4-C1 RED.

Covers:
  AC-NB-1: python -m src.build_notebook_07 exits 0 and produces
            notebooks/07_lstm/07_lstm.ipynb.
  AC-NB-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB-3: every cell ID matches 'cell-07-*' (no random UUIDs).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).parent.parent
NOTEBOOK_07_PATH = ROOT / "notebooks" / "07_lstm" / "07_lstm.ipynb"
BUILDER_07 = ROOT / "src" / "build_notebook_07.py"


class TestNotebook07Builder:
    """W4 — Verify build_notebook_07.py produces a correct, stable notebook.

    AC-NB-1: builder exits 0 and produces notebooks/07_lstm/07_lstm.ipynb.
    AC-NB-2: two consecutive runs leave the working tree clean (byte-identical).
    AC-NB-3: all cell IDs have the 'cell-07-' prefix (no uuid4 IDs).
    """

    def test_builder_exits_zero_and_creates_notebook(self):
        """AC-NB-1: python -m src.build_notebook_07 exits 0 and writes the notebook.

        Failure mode: build_notebook_07.py does not exist yet (RED), or exits non-zero
        due to import error / missing dependency.
        """
        result = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_07"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result.returncode == 0, (
            f"src.build_notebook_07 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert NOTEBOOK_07_PATH.exists(), (
            f"Expected notebook at {NOTEBOOK_07_PATH} but file does not exist"
        )

    def test_cell_ids_have_cell_07_prefix(self):
        """AC-NB-3: every cell ID must match the pattern 'cell-07-*' (no uuid4).

        Failure mode: nbformat assigns random UUIDs when cell['id'] is not set
        explicitly; those IDs change on every re-generation causing git flutter.
        """
        # Ensure the notebook exists (run builder if needed)
        if not NOTEBOOK_07_PATH.exists():
            subprocess.run(
                [sys.executable, "-m", "src.build_notebook_07"],
                capture_output=True, text=True, cwd=str(ROOT),
                check=True,
            )
        import re
        stable_pattern = re.compile(r"^cell-07-")
        with open(NOTEBOOK_07_PATH, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs do not match 'cell-07-*' pattern: {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-07-<name>' to prevent "
            "git flutter on re-generation."
        )

    def test_second_run_is_byte_identical(self):
        """AC-NB-2: two consecutive runs must produce byte-identical output.

        Failure mode: cell IDs are generated with uuid4() so each run produces
        different IDs — the working tree would show uncommitted changes after
        the second run (notebook flutter).
        """
        # First run
        result1 = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_07"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        bytes_after_first = NOTEBOOK_07_PATH.read_bytes()

        # Second run
        result2 = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_07"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        bytes_after_second = NOTEBOOK_07_PATH.read_bytes()

        assert bytes_after_first == bytes_after_second, (
            "build_notebook_07 is NOT idempotent: two consecutive runs produced "
            "different byte content. Each cell must have an explicit stable ID "
            "like 'cell-07-setup', not a random uuid4."
        )
