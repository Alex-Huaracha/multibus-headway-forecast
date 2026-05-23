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

    def test_null_buckets_sidecar_embedded(self, generated_notebook: nbformat.NotebookNode):
        """AC-NB-1: the generated notebook must contain the null_buckets sidecar write.

        Asserts that:
        1. The string 'headway_null_buckets_E' is present (output filename pattern).
        2. A second '.write_parquet(' call is present for the sidecar DataFrame.

        Failure mode (RED): build_notebook_04.py does not yet unpack the tuple
        or write the sidecar parquet — both strings are absent.
        """
        code_cells = self._get_code_cells(generated_notebook)
        combined = "\n".join(code_cells)
        assert "headway_null_buckets_E" in combined, (
            "Expected 'headway_null_buckets_E' in generated notebook code cells (AC-NB-1). "
            "Update src/build_notebook_04.py to unpack the tuple and write the sidecar parquet."
        )
        # The notebook must have at least 2 .write_parquet( calls in the headways section.
        write_parquet_count = combined.count(".write_parquet(")
        assert write_parquet_count >= 2, (
            f"Expected >= 2 '.write_parquet(' calls in generated notebook (headways + sidecar); "
            f"got {write_parquet_count}. Update src/build_notebook_04.py to add the sidecar write."
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


class TestAllCellsCompile:
    """Every code cell in the generated notebook must be syntactically valid Python.

    This is the permanent guard for CRITICAL-1: _strip_relative_imports must
    leave every code cell in a state that Python can compile without SyntaxError.
    Catches multi-line import blocks where stripping only the opening line leaves
    dangling names + closing paren as orphaned syntax.
    """

    def test_all_cells_compile(self, generated_notebook: nbformat.NotebookNode):
        """Failure mode: _strip_relative_imports strips only the 'from .x import ('
        line but leaves indented names and closing paren in the cell — Python raises
        SyntaxError: unexpected indent on those lines.

        Reads notebooks/04_preprocessing/04_preprocessing.ipynb, compiles every
        code cell, and asserts no SyntaxError is raised.
        """
        errors: list[str] = []
        for i, cell in enumerate(generated_notebook.cells):
            if cell.cell_type != "code":
                continue
            source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            try:
                compile(source, f"<cell-{i}>", "exec")
            except SyntaxError as exc:
                errors.append(
                    f"Cell {i} SyntaxError at line {exc.lineno}: {exc.msg}\n"
                    f"  Source snippet: {source.splitlines()[max(0,(exc.lineno or 1)-1)]!r}"
                )
        assert not errors, (
            f"{len(errors)} code cell(s) failed to compile:\n" + "\n".join(errors)
        )


# ---------------------------------------------------------------------------
# Tests for src/build_notebook_06.py  (T8.1 — RED)
#   AC-NB-1: builder exits 0, notebook exists, valid JSON
#   AC-NB-2: byte-identical second run (stable cell IDs, flutter-free)
# ---------------------------------------------------------------------------

NOTEBOOK_06_PATH = ROOT / "notebooks" / "06_baselines_stat" / "06_baselines_stat.ipynb"
BUILDER_06 = ROOT / "src" / "build_notebook_06.py"


@pytest.fixture(scope="module")
def generated_notebook_06() -> nbformat.NotebookNode:
    """Run build_notebook_06.py once and return the parsed notebook.

    AC-NB-1: builder must exit 0 and produce a valid .ipynb at
    notebooks/06_baselines_stat/06_baselines_stat.ipynb.
    """
    result = subprocess.run(
        [sys.executable, str(BUILDER_06)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"build_notebook_06.py failed with return code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert NOTEBOOK_06_PATH.exists(), (
        f"Expected notebook at {NOTEBOOK_06_PATH} but file does not exist"
    )
    with open(NOTEBOOK_06_PATH, encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


class TestNotebook06Builder:
    """T8.1 — Verify build_notebook_06.py produces a correct, stable notebook."""

    def test_notebook_06_exists_and_valid(
        self, generated_notebook_06: nbformat.NotebookNode
    ):
        """AC-NB-1: Notebook file must exist and be valid nbformat.

        Failure mode: builder script raises an exception or writes invalid JSON.
        """
        assert NOTEBOOK_06_PATH.exists()
        nb = generated_notebook_06
        assert hasattr(nb, "cells"), "Notebook must have a cells attribute"
        assert len(nb.cells) >= 8, (
            f"Expected >= 8 cells (title + setup + 4 embeds + sanity + run + write + summary); "
            f"got {len(nb.cells)}"
        )

    def test_notebook_06_first_cell_markdown_with_title(
        self, generated_notebook_06: nbformat.NotebookNode
    ):
        """First cell must be a markdown title cell referencing notebook 06."""
        first = generated_notebook_06.cells[0]
        assert first.cell_type == "markdown", (
            f"First cell is '{first.cell_type}'; expected 'markdown'"
        )
        assert "06" in first.source, (
            f"First markdown cell does not mention '06': {first.source[:100]!r}"
        )

    def test_notebook_06_embeds_all_modules(
        self, generated_notebook_06: nbformat.NotebookNode
    ):
        """All four modules must be inlined as code cells.

        Checks for at least one unique symbol from each module:
          - splits.py  → SPLIT_TRAIN_START
          - metrics.py → def mae
          - statistical.py → predict_b0
          - harness.py → evaluate_corridor
        """
        code_cells = [
            c.source for c in generated_notebook_06.cells if c.cell_type == "code"
        ]
        combined = "\n".join(code_cells)
        assert "SPLIT_TRAIN_START" in combined, (
            "Expected 'SPLIT_TRAIN_START' from evaluation/splits.py in a code cell"
        )
        assert "def mae" in combined, (
            "Expected 'def mae' from evaluation/metrics.py in a code cell"
        )
        assert "predict_b0" in combined, (
            "Expected 'predict_b0' from baselines/statistical.py in a code cell"
        )
        assert "evaluate_corridor" in combined, (
            "Expected 'evaluate_corridor' from baselines/harness.py in a code cell"
        )

    def test_notebook_06_no_relative_imports(
        self, generated_notebook_06: nbformat.NotebookNode
    ):
        """No relative import may survive in any code cell after _strip_relative_imports.

        Failure mode: the helper misses 'from .config import X' type lines,
        which would fail at Kaggle runtime.
        """
        import re
        relative_import_pattern = re.compile(r"from \.")
        for i, cell in enumerate(generated_notebook_06.cells):
            if cell.cell_type != "code":
                continue
            matches = relative_import_pattern.findall(cell.source)
            assert not matches, (
                f"Code cell {i} contains relative import(s): {matches}\n"
                f"Cell source (first 200 chars): {cell.source[:200]!r}"
            )

    def test_notebook_06_references_e2_and_e59_only(
        self, generated_notebook_06: nbformat.NotebookNode
    ):
        """No reference to E4 or E58 may appear in any cell.

        This guards B3-CORRIDOR-SCOPE / AC-CSV-3.
        """
        for i, cell in enumerate(generated_notebook_06.cells):
            src = cell.source
            assert "E4" not in src and "E58" not in src, (
                f"Cell {i} references E4 or E58 (out of scope): {src[:200]!r}"
            )

    def test_notebook_06_stable_cell_ids(self):
        """AC-NB-2: Running the builder twice must produce identical cell IDs.

        This is the regression test for B3-NOTEBOOK-FLUTTER: if cell IDs are
        random (uuid4), each regen produces a dirty working tree. With explicit
        stable IDs (cell-06-N) the second run is byte-identical.
        """
        # Run builder a second time
        result2 = subprocess.run(
            [sys.executable, str(BUILDER_06)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result2.returncode == 0, (
            f"Second run of build_notebook_06.py failed.\n"
            f"stdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        # Read both versions (first was written by the fixture; second just written)
        first_bytes = NOTEBOOK_06_PATH.read_bytes()
        # Write again and compare (the fixture already ran the first pass)
        result3 = subprocess.run(
            [sys.executable, str(BUILDER_06)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result3.returncode == 0
        second_bytes = NOTEBOOK_06_PATH.read_bytes()
        assert first_bytes == second_bytes, (
            "build_notebook_06.py is NOT idempotent: two consecutive runs produced "
            "different byte content. This indicates non-stable cell IDs (flutter). "
            "Each cell must have an explicit id= like 'cell-06-setup'."
        )


# ---------------------------------------------------------------------------
# Tests for src/build_notebook_05.py  (W5 — RED)
#   AC-NB-1: builder exits 0, notebook exists at notebooks/05_dataset/05_dataset.ipynb
#   AC-NB-2: byte-identical second run (idempotency / no cell-ID flutter)
#   AC-NB-3: all cell IDs match pattern cell-05-* (no random UUIDs)
# ---------------------------------------------------------------------------

NOTEBOOK_05_PATH = ROOT / "notebooks" / "05_dataset" / "05_dataset.ipynb"
BUILDER_05 = ROOT / "src" / "build_notebook_05.py"


class TestNotebook05Builder:
    """W5 — Verify build_notebook_05.py produces a correct, stable notebook.

    AC-NB-1: builder exits 0 and produces notebooks/05_dataset/05_dataset.ipynb.
    AC-NB-2: two consecutive runs leave the working tree clean (byte-identical).
    AC-NB-3: all cell IDs have the 'cell-05-' prefix (no uuid4 IDs).
    """

    def test_builder_runs_clean_exit_0(self):
        """AC-NB-1: python -m src.build_notebook_05 exits 0 and writes the notebook.

        Failure mode: build_notebook_05.py does not exist yet (RED), or exits non-zero
        due to import error / missing dependency.
        """
        result = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_05"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result.returncode == 0, (
            f"src.build_notebook_05 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert NOTEBOOK_05_PATH.exists(), (
            f"Expected notebook at {NOTEBOOK_05_PATH} but file does not exist"
        )

    def test_builder_is_idempotent(self):
        """AC-NB-2: two consecutive runs must produce byte-identical output.

        Failure mode: cell IDs are generated with uuid4() so each run produces
        different IDs — the working tree would show uncommitted changes after
        the second run (notebook flutter).
        """
        # First run
        result1 = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_05"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        bytes_after_first = NOTEBOOK_05_PATH.read_bytes()

        # Second run
        result2 = subprocess.run(
            [sys.executable, "-m", "src.build_notebook_05"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        bytes_after_second = NOTEBOOK_05_PATH.read_bytes()

        assert bytes_after_first == bytes_after_second, (
            "build_notebook_05 is NOT idempotent: two consecutive runs produced "
            "different byte content. Each cell must have an explicit stable ID "
            "like 'cell-05-setup', not a random uuid4."
        )

    def test_notebook_has_stable_cell_ids(self):
        """AC-NB-3: every cell ID must match the pattern 'cell-05-*' (no uuid4).

        Failure mode: nbformat assigns random UUIDs when cell['id'] is not set
        explicitly; those IDs change on every re-generation causing git flutter.
        """
        # Ensure the notebook exists (run builder if needed)
        if not NOTEBOOK_05_PATH.exists():
            subprocess.run(
                [sys.executable, "-m", "src.build_notebook_05"],
                capture_output=True, text=True, cwd=str(ROOT),
                check=True,
            )
        import re
        stable_pattern = re.compile(r"^cell-05-")
        with open(NOTEBOOK_05_PATH, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs do not match 'cell-05-*' pattern: {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-05-<name>' to prevent "
            "git flutter on re-generation."
        )
