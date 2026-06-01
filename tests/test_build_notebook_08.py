"""Tests for src/build_notebook_08.py — W3-C1 RED.

Covers:
  AC-NB08-1: python -m src.build_notebook_08 exits 0 and produces both per-corridor
             notebooks (08a E2 and 08b E59).
  AC-NB08-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB08-3: every cell ID matches 'cell-08-*' (no random UUIDs) in both notebooks.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).parent.parent

# Both per-corridor notebook paths produced by build_notebook_08.py.
NOTEBOOK_08A_PATH = (
    ROOT / "notebooks" / "08a_spatial_conv_lstm_e2" / "08a_spatial_conv_lstm_e2.ipynb"
)
NOTEBOOK_08B_PATH = (
    ROOT / "notebooks" / "08b_spatial_conv_lstm_e59" / "08b_spatial_conv_lstm_e59.ipynb"
)
BOTH_PATHS = [NOTEBOOK_08A_PATH, NOTEBOOK_08B_PATH]

BUILDER_08 = ROOT / "src" / "build_notebook_08.py"


def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_08"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


class TestNotebook08Builder:
    """W3 — Verify build_notebook_08.py produces correct, stable per-corridor notebooks.

    AC-NB08-1: builder exits 0 and produces both
               notebooks/08a_spatial_conv_lstm_e2/08a_spatial_conv_lstm_e2.ipynb and
               notebooks/08b_spatial_conv_lstm_e59/08b_spatial_conv_lstm_e59.ipynb.
    AC-NB08-2: two consecutive runs leave the working tree clean (byte-identical).
    AC-NB08-3: all cell IDs have the 'cell-08-' prefix (no uuid4 IDs) in both notebooks.
    """

    def test_builder_exits_zero_and_writes_notebooks(self):
        """AC-NB08-1: python -m src.build_notebook_08 exits 0 and writes both notebooks.

        Failure mode: build_notebook_08.py does not exist yet (RED), or exits non-zero
        due to import error / missing dependency.
        """
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_08 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        for nb_path in BOTH_PATHS:
            assert nb_path.exists(), (
                f"Expected notebook at {nb_path} but file does not exist"
            )

    @pytest.mark.parametrize("nb_path", BOTH_PATHS, ids=["08a-E2", "08b-E59"])
    def test_cell_ids_cell_08_prefix(self, nb_path: Path):
        """AC-NB08-3: every cell ID must match the pattern 'cell-08-*' (no uuid4).

        Failure mode: nbformat assigns random UUIDs when cell['id'] is not set
        explicitly; those IDs change on every re-generation causing git flutter.
        """
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        import re
        stable_pattern = re.compile(r"^cell-08-")
        with open(nb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs in {nb_path.name} do not match 'cell-08-*' "
            f"pattern: {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-08-<name>' to prevent "
            "git flutter on re-generation."
        )

    @pytest.mark.parametrize("nb_path", BOTH_PATHS, ids=["08a-E2", "08b-E59"])
    def test_second_run_byte_identical(self, nb_path: Path):
        """AC-NB08-2: two consecutive runs must produce byte-identical output.

        Failure mode: cell IDs are generated with uuid4() so each run produces
        different IDs — the working tree would show uncommitted changes after
        the second run (notebook flutter).
        """
        # First run
        result1 = _run_builder()
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        assert nb_path.exists(), f"{nb_path} not written after first run"
        bytes_after_first = nb_path.read_bytes()

        # Second run
        result2 = _run_builder()
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        bytes_after_second = nb_path.read_bytes()

        assert bytes_after_first == bytes_after_second, (
            f"build_notebook_08 is NOT idempotent for {nb_path.name}: two consecutive "
            "runs produced different byte content. Each cell must have an explicit "
            "stable ID like 'cell-08-setup', not a random uuid4."
        )
