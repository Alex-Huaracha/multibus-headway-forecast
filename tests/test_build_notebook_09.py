"""Tests for src/build_notebook_09.py.

Covers:
  AC-NB09-1: python -m src.build_notebook_09 exits 0 and produces all per-corridor
             notebooks (09a E2 full grid, 09b1 E59 tanda 1, 09b2 E59 tanda 2).
  AC-NB09-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB09-3: every cell ID matches 'cell-09-*' (no random UUIDs, no 'cell-08-*') in
             all notebooks.
  AC-NB09-4: E59 grid is split into two tandas — 09b1 runs TRANSFORMER_GRID[0:16],
             09b2 runs TRANSFORMER_GRID[16:32]; 09a (E2) runs the full grid. This keeps
             each Kaggle session under the 12h timeout for the larger corridor while
             preserving the exact 32-config search (argmin val_loss over both tandas
             == argmin over the full grid).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).parent.parent

# Per-corridor notebook paths produced by build_notebook_09.py.
NOTEBOOK_09A_PATH = (
    ROOT / "notebooks" / "09_spatial_transformer" / "09a_e2" / "notebook.ipynb"
)
NOTEBOOK_09B1_PATH = (
    ROOT / "notebooks" / "09_spatial_transformer" / "09b1_e59" / "notebook.ipynb"
)
NOTEBOOK_09B2_PATH = (
    ROOT / "notebooks" / "09_spatial_transformer" / "09b2_e59" / "notebook.ipynb"
)
BOTH_PATHS = [NOTEBOOK_09A_PATH, NOTEBOOK_09B1_PATH, NOTEBOOK_09B2_PATH]

BUILDER_09 = ROOT / "src" / "build_notebook_09.py"


def _train_cell_source(nb_path: Path) -> str:
    """Return the source of the grid-search train cell (cell-09-train)."""
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if cell.get("id") == "cell-09-train":
            return cell.source
    raise AssertionError(f"cell-09-train not found in {nb_path}")


def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_09"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


class TestNotebook09Builder:
    """W3 — Verify build_notebook_09.py produces correct, stable per-corridor notebooks.

    AC-NB09-1: builder exits 0 and produces both
               notebooks/09_spatial_transformer/09a_e2/notebook.ipynb and
               notebooks/09_spatial_transformer/09b_e59/notebook.ipynb.
    AC-NB09-2: two consecutive runs leave the working tree clean (byte-identical).
    AC-NB09-3: all cell IDs have the 'cell-09-' prefix (no uuid4 IDs, no 'cell-08-')
               in both notebooks.
    """

    def test_builder_exits_zero_and_writes_notebooks(self):
        """AC-NB09-1: python -m src.build_notebook_09 exits 0 and writes both notebooks.

        Failure mode: build_notebook_09.py does not exist yet (RED), or exits non-zero
        due to import error / missing dependency.
        """
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_09 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        for nb_path in BOTH_PATHS:
            assert nb_path.exists(), (
                f"Expected notebook at {nb_path} but file does not exist"
            )

    @pytest.mark.parametrize(
        "nb_path", BOTH_PATHS, ids=["09a-E2", "09b1-E59", "09b2-E59"]
    )
    def test_cell_ids_cell_09_prefix(self, nb_path: Path):
        """AC-NB09-3: every cell ID must match the pattern 'cell-09-*' (no uuid4, no cell-08-*).

        Failure mode: nbformat assigns random UUIDs when cell['id'] is not set
        explicitly; those IDs change on every re-generation causing git flutter.
        Also ensures no IDs from notebook 08 were accidentally copied.
        """
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        import re
        stable_pattern = re.compile(r"^cell-09-")
        with open(nb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs in {nb_path.name} do not match 'cell-09-*' "
            f"pattern: {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-09-<name>' to prevent "
            "git flutter on re-generation."
        )

    @pytest.mark.parametrize(
        "nb_path", BOTH_PATHS, ids=["09a-E2", "09b1-E59", "09b2-E59"]
    )
    def test_second_run_byte_identical(self, nb_path: Path):
        """AC-NB09-2: two consecutive runs must produce byte-identical output.

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
            f"build_notebook_09 is NOT idempotent for {nb_path.name}: two consecutive "
            "runs produced different byte content. Each cell must have an explicit "
            "stable ID like 'cell-09-setup', not a random uuid4."
        )


class TestNotebook09GridSplit:
    """AC-NB09-4 — E59 grid is split into two tandas to dodge the 12h Kaggle timeout.

    The larger corridor (E59) cannot finish the full 32-config grid in one 12h
    session. Splitting into TRANSFORMER_GRID[0:16] (09b1) and [16:32] (09b2) keeps
    each session under the cap. Selecting the lower best_val_loss across the two
    tandas is identical to argmin over the full 32-config grid, so the experiment
    is unchanged. E2 (09a) is small enough to run the full grid in one session.
    """

    def test_e2_runs_full_grid(self):
        """09a (E2) must pass the complete TRANSFORMER_GRID (no slice)."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        src = _train_cell_source(NOTEBOOK_09A_PATH)
        assert "configs=TRANSFORMER_GRID," in src, (
            "09a (E2) must run the full grid: expected 'configs=TRANSFORMER_GRID,' "
            f"in cell-09-train, got:\n{src}"
        )

    def test_e59_tanda1_runs_first_half(self):
        """09b1 (E59 tanda 1) must run TRANSFORMER_GRID[0:16]."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        src = _train_cell_source(NOTEBOOK_09B1_PATH)
        assert "TRANSFORMER_GRID[0:16]" in src, (
            "09b1 (E59 tanda 1) must run the first half of the grid: expected "
            f"'TRANSFORMER_GRID[0:16]' in cell-09-train, got:\n{src}"
        )

    def test_e59_tanda2_runs_second_half(self):
        """09b2 (E59 tanda 2) must run TRANSFORMER_GRID[16:32]."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        src = _train_cell_source(NOTEBOOK_09B2_PATH)
        assert "TRANSFORMER_GRID[16:32]" in src, (
            "09b2 (E59 tanda 2) must run the second half of the grid: expected "
            f"'TRANSFORMER_GRID[16:32]' in cell-09-train, got:\n{src}"
        )
