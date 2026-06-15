"""Tests for src/build_notebook_11.py — Paso 3a RED.

Covers:
  AC-NB11-1: python -m src.build_notebook_11 exits 0 and produces all 4 per-horizon
             notebooks (h∈{1,3,5,10}) under notebooks/11_lstm_multihorizon/.
  AC-NB11-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB11-3: every cell ID matches 'cell-11-*' (no random UUIDs) in all notebooks.
  AC-NB11-4: dataset cell is horizon-aware — HORIZON constant injected, window_size uses
             HORIZON (not T_OUT), make_window_index receives horizon= kwarg.
  AC-NB11-5: train cell uses WINNING_CONFIGS dict (no grid search over GRID).
  AC-NB11-6: compare cell reads baselines_results_multih.csv and filters by HORIZON.
  AC-NB11-7: results cell has horizon column (schema: corridor, direction, baseline,
             metric, value, horizon).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).parent.parent

# One notebook per horizon — all live under the same parent directory.
NB_DIR = ROOT / "notebooks" / "11_lstm_multihorizon"

HORIZONS = [1, 3, 5, 10]

NB_PATHS: dict[int, Path] = {
    h: NB_DIR / f"11_lstm_h{h}.ipynb" for h in HORIZONS
}

ALL_NB_PATHS = list(NB_PATHS.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_11"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _cell_source(nb_path: Path, cell_id: str) -> str:
    """Return the source of the cell with the given ID."""
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if cell.get("id") == cell_id:
            return cell.source
    raise AssertionError(f"{cell_id} not found in {nb_path}")


# ---------------------------------------------------------------------------
# AC-NB11-1: builder exits 0 and writes all 4 notebooks
# ---------------------------------------------------------------------------

class TestNotebook11Builder:
    """Verify build_notebook_11.py produces correct, stable per-horizon notebooks."""

    def test_builder_exits_zero_and_writes_notebooks(self):
        """AC-NB11-1: python -m src.build_notebook_11 exits 0 and writes all 4 .ipynb files."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_11 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        for h, nb_path in NB_PATHS.items():
            assert nb_path.exists(), (
                f"Expected notebook for h={h} at {nb_path} but file does not exist"
            )

    # -----------------------------------------------------------------------
    # AC-NB11-3: cell IDs have cell-11- prefix
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "nb_path", ALL_NB_PATHS, ids=[f"h{h}" for h in HORIZONS]
    )
    def test_cell_ids_have_cell_11_prefix(self, nb_path: Path):
        """AC-NB11-3: every cell ID must match 'cell-11-*' (no uuid4, no cell-07-*)."""
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        stable_pattern = re.compile(r"^cell-11-")
        with open(nb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs in {nb_path.name} do not match 'cell-11-*' "
            f"pattern: {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-11-<name>' to prevent "
            "git flutter on re-generation."
        )

    # -----------------------------------------------------------------------
    # AC-NB11-2: byte-identical on second run
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "nb_path", ALL_NB_PATHS, ids=[f"h{h}" for h in HORIZONS]
    )
    def test_second_run_byte_identical(self, nb_path: Path):
        """AC-NB11-2: two consecutive runs must produce byte-identical output."""
        result1 = _run_builder()
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        assert nb_path.exists(), f"{nb_path} not written after first run"
        bytes_after_first = nb_path.read_bytes()

        result2 = _run_builder()
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        bytes_after_second = nb_path.read_bytes()

        assert bytes_after_first == bytes_after_second, (
            f"build_notebook_11 is NOT idempotent for {nb_path.name}: two consecutive "
            "runs produced different byte content."
        )


# ---------------------------------------------------------------------------
# AC-NB11-4: dataset cell is horizon-aware (checked on h=5 notebook)
# ---------------------------------------------------------------------------

class TestNotebook11DatasetCell:
    """Verify the dataset cell injects HORIZON and uses it correctly."""

    def test_dataset_cell_is_horizon_aware(self):
        """AC-NB11-4: h=5 notebook dataset cell contains HORIZON=5, window_size=T_IN+HORIZON,
        and horizon=HORIZON in make_window_index. Must NOT contain T_IN + T_OUT for window_size.
        """
        nb_path = NB_PATHS[5]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-11-dataset")

        assert "HORIZON = 5" in src, (
            f"Expected 'HORIZON = 5' in cell-11-dataset of h=5 notebook, got:\n{src[:500]}"
        )
        assert "window_size = T_IN + HORIZON" in src, (
            f"Expected 'window_size = T_IN + HORIZON' in cell-11-dataset, got:\n{src[:500]}"
        )
        assert "horizon=HORIZON" in src, (
            f"Expected 'horizon=HORIZON' in make_window_index call in cell-11-dataset, "
            f"got:\n{src[:500]}"
        )
        # Must NOT use T_IN + T_OUT for the window_size calculation
        assert "T_IN + T_OUT" not in src, (
            "cell-11-dataset must NOT use 'T_IN + T_OUT' for window_size (that's the "
            "h=1-only formula). Use 'T_IN + HORIZON' instead."
        )


# ---------------------------------------------------------------------------
# AC-NB11-5: train cell uses WINNING_CONFIGS (no grid over GRID)
# ---------------------------------------------------------------------------

class TestNotebook11TrainCell:
    """Verify the train cell uses the winning configs dict, not a full grid search."""

    def test_train_cell_uses_winning_config(self):
        """AC-NB11-5: train cell contains WINNING_CONFIGS dict and passes it as a
        single-element list, not the full GRID.
        """
        # Check on h=1 notebook (any would work, they all have the same train cell)
        nb_path = NB_PATHS[1]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-11-train")

        assert "WINNING_CONFIGS" in src, (
            f"Expected 'WINNING_CONFIGS' in cell-11-train, got:\n{src[:500]}"
        )
        assert "configs=[WINNING_CONFIGS[label]]" in src, (
            f"Expected 'configs=[WINNING_CONFIGS[label]]' in cell-11-train, "
            f"got:\n{src[:500]}"
        )
        assert "configs=GRID" not in src, (
            "cell-11-train must NOT use 'configs=GRID' (full grid search). "
            "Use the single winning config instead."
        )


# ---------------------------------------------------------------------------
# AC-NB11-6: compare cell reads multih baselines and filters by HORIZON
# ---------------------------------------------------------------------------

class TestNotebook11CompareCell:
    """Verify the compare cell reads the multi-horizon baselines CSV."""

    def test_compare_reads_multih_baselines(self):
        """AC-NB11-6: compare cell references baselines_results_multih.csv
        and filters by pl.col("horizon") == HORIZON.
        """
        nb_path = NB_PATHS[3]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-11-compare")

        assert "baselines_results_multih.csv" in src, (
            f"Expected 'baselines_results_multih.csv' in cell-11-compare, "
            f"got:\n{src[:500]}"
        )
        assert 'pl.col("horizon") == HORIZON' in src, (
            f"Expected 'pl.col(\"horizon\") == HORIZON' filter in cell-11-compare, "
            f"got:\n{src[:500]}"
        )


# ---------------------------------------------------------------------------
# AC-NB11-7: results cell has horizon column
# ---------------------------------------------------------------------------

class TestNotebook11ResultsCell:
    """Verify the results cell adds horizon column to the output CSV."""

    def test_results_csv_has_horizon_column(self):
        """AC-NB11-7: results cell contains 'horizon': HORIZON in the row dict
        and writes to lstm_results_h{H}.csv.
        """
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-11-results")

        assert '"horizon": HORIZON' in src, (
            f"Expected '\"horizon\": HORIZON' in cell-11-results, got:\n{src[:500]}"
        )
        assert "lstm_results_h" in src, (
            f"Expected 'lstm_results_h' (horizon-discriminated filename) in "
            f"cell-11-results, got:\n{src[:500]}"
        )
