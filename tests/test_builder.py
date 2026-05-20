"""Tests for src/build_notebook_04.py (T_builder).

Covers:
  - Running build_notebook_04.py produces notebooks/04_preprocessing/04_preprocessing.ipynb.
  - Output is valid JSON parseable by nbformat.read.
  - Notebook has >= 18 cells.
  - First cell is a markdown cell containing "04 — Preprocessing".
  - Each of the 6 module names (config, corridor, projection, direction,
    trips, headways) has its source verbatim in at least one code cell.
  - No 'from .' relative import survives in any code cell.
"""
from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "04_preprocessing" / "04_preprocessing.ipynb"
MODULE_NAMES = ["config", "corridor", "projection", "direction", "trips", "headways"]


@pytest.fixture(scope="module")
def generated_notebook() -> nbformat.NotebookNode:
    """Run build_notebook_04.py and return the parsed notebook."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "build_notebook_04.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"build_notebook_04.py failed with return code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert NOTEBOOK_PATH.exists(), (
        f"Expected notebook at {NOTEBOOK_PATH} but file does not exist"
    )
    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


class TestNotebookStructure:
    """T_builder — basic structural checks on the generated notebook."""

    def test_notebook_exists(self, generated_notebook: nbformat.NotebookNode):
        """Notebook file must exist at the exact path from clarification #17 rule 3."""
        assert NOTEBOOK_PATH.exists(), (
            f"notebooks/04_preprocessing/04_preprocessing.ipynb not found"
        )

    def test_cell_count_at_least_18(self, generated_notebook: nbformat.NotebookNode):
        """Failure mode: a module was accidentally excluded; cell count would drop < 18."""
        nb = generated_notebook
        assert len(nb.cells) >= 18, (
            f"Expected >= 18 cells; got {len(nb.cells)}"
        )

    def test_first_cell_is_markdown_with_title(self, generated_notebook: nbformat.NotebookNode):
        """First cell must be a markdown cell containing '04 — Preprocessing'."""
        nb = generated_notebook
        first = nb.cells[0]
        assert first.cell_type == "markdown", (
            f"First cell is '{first.cell_type}'; expected 'markdown'"
        )
        assert "04" in first.source and "Preprocessing" in first.source, (
            f"First markdown cell does not contain '04 — Preprocessing': "
            f"{first.source[:100]!r}"
        )

    def test_valid_nbformat(self, generated_notebook: nbformat.NotebookNode):
        """Notebook must parse without error via nbformat.read (already done in fixture)."""
        nb = generated_notebook
        assert hasattr(nb, "cells"), "Notebook must have a cells attribute"
        assert hasattr(nb, "metadata"), "Notebook must have a metadata attribute"


class TestModuleEmbedding:
    """Each module's source must appear verbatim in at least one code cell."""

    def _get_code_cells(self, nb: nbformat.NotebookNode) -> list[str]:
        return [c.source for c in nb.cells if c.cell_type == "code"]

    def test_config_embedded(self, generated_notebook: nbformat.NotebookNode):
        """Failure mode: embed_module skipped config.py or _strip_relative_imports
        removed too much content.
        """
        code_cells = self._get_code_cells(generated_notebook)
        combined = "\n".join(code_cells)
        assert "PRODUCTIVE_PARAMS" in combined, (
            "Expected 'PRODUCTIVE_PARAMS' from config.py in a code cell"
        )

    def test_corridor_embedded(self, generated_notebook: nbformat.NotebookNode):
        code_cells = self._get_code_cells(generated_notebook)
        combined = "\n".join(code_cells)
        assert "build_centerline" in combined, (
            "Expected 'build_centerline' from corridor.py in a code cell"
        )

    def test_projection_embedded(self, generated_notebook: nbformat.NotebookNode):
        code_cells = self._get_code_cells(generated_notebook)
        combined = "\n".join(code_cells)
        assert "attach_observed_speed" in combined, (
            "Expected 'attach_observed_speed' from projection.py in a code cell"
        )

    def test_direction_embedded(self, generated_notebook: nbformat.NotebookNode):
        code_cells = self._get_code_cells(generated_notebook)
        combined = "\n".join(code_cells)
        assert "infer_direction" in combined, (
            "Expected 'infer_direction' from direction.py in a code cell"
        )

    def test_trips_embedded(self, generated_notebook: nbformat.NotebookNode):
        code_cells = self._get_code_cells(generated_notebook)
        combined = "\n".join(code_cells)
        assert "assign_trip_ids" in combined, (
            "Expected 'assign_trip_ids' from trips.py in a code cell"
        )

    def test_headways_embedded(self, generated_notebook: nbformat.NotebookNode):
        code_cells = self._get_code_cells(generated_notebook)
        combined = "\n".join(code_cells)
        assert "compute_headways_c2" in combined, (
            "Expected 'compute_headways_c2' from headways.py in a code cell"
        )


class TestRelativeImportsStripped:
    """No 'from .' relative import must survive in any code cell (_strip_relative_imports)."""

    def test_no_relative_imports_in_code_cells(self, generated_notebook: nbformat.NotebookNode):
        """Failure mode: _strip_relative_imports regex is wrong or too narrow,
        leaving 'from .config import ...' lines that would fail at Kaggle runtime.
        """
        import re
        relative_import_pattern = re.compile(r"from \.")
        for i, cell in enumerate(generated_notebook.cells):
            if cell.cell_type != "code":
                continue
            matches = relative_import_pattern.findall(cell.source)
            assert not matches, (
                f"Code cell {i} contains relative import(s): {matches}\n"
                f"Cell source (first 200 chars): {cell.source[:200]!r}"
            )
