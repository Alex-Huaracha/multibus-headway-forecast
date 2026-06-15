"""Tests for src/build_notebook_13.py — Paso 3c RED.

Covers:
  AC-NB13-1: python -m src.build_notebook_13 exits 0 and produces all 4 per-horizon
             notebooks (h∈{1,3,5,10}) under notebooks/13_spatial_transformer_multihorizon/.
  AC-NB13-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB13-3: every cell ID matches 'cell-13-*' (no random UUIDs) in all notebooks.
  AC-NB13-4: dataset cell is horizon-aware — HORIZON constant injected, window_size uses
             HORIZON (not T_OUT), make_window_index receives horizon= kwarg.
  AC-NB13-5: train cell uses WINNING_CONFIGS dict with nhead= and d_model= fields, and
             passes a single-element list (no full grid search over TRANSFORMER_GRID).
  AC-NB13-6: compare cell reads baselines_results_multih.csv and filters by HORIZON.
  AC-NB13-7: results cell has horizon column (schema: corridor, direction, baseline,
             metric, value, horizon) and writes to spatial_transformer_results_h{H}.csv.
  AC-NB13-8: evaluate cell uses SpatialTransformer( construction and model(inp, ctx, input_mask)
             forward signature (3-arg spatial dispatch).
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
NB_DIR = ROOT / "notebooks" / "13_spatial_transformer_multihorizon"

HORIZONS = [1, 3, 5, 10]

NB_PATHS: dict[int, Path] = {
    h: NB_DIR / f"13_spatial_transformer_h{h}.ipynb" for h in HORIZONS
}

ALL_NB_PATHS = list(NB_PATHS.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_13"],
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
# AC-NB13-1: builder exits 0 and writes all 4 notebooks
# ---------------------------------------------------------------------------

class TestNotebook13Builder:
    """Verify build_notebook_13.py produces correct, stable per-horizon notebooks."""

    def test_builder_exits_zero_and_writes_notebooks(self):
        """AC-NB13-1: python -m src.build_notebook_13 exits 0 and writes all 4 .ipynb files."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_13 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        for h, nb_path in NB_PATHS.items():
            assert nb_path.exists(), (
                f"Expected notebook for h={h} at {nb_path} but file does not exist"
            )

    # -----------------------------------------------------------------------
    # AC-NB13-3: cell IDs have cell-13- prefix
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "nb_path", ALL_NB_PATHS, ids=[f"h{h}" for h in HORIZONS]
    )
    def test_cell_ids_have_cell_13_prefix(self, nb_path: Path):
        """AC-NB13-3: every cell ID must match 'cell-13-*' (no uuid4, no cell-09-*)."""
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        stable_pattern = re.compile(r"^cell-13-")
        with open(nb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs in {nb_path.name} do not match 'cell-13-*' "
            f"pattern: {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-13-<name>' to prevent "
            "git flutter on re-generation."
        )

    # -----------------------------------------------------------------------
    # AC-NB13-2: byte-identical on second run
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "nb_path", ALL_NB_PATHS, ids=[f"h{h}" for h in HORIZONS]
    )
    def test_second_run_byte_identical(self, nb_path: Path):
        """AC-NB13-2: two consecutive runs must produce byte-identical output."""
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
            f"build_notebook_13 is NOT idempotent for {nb_path.name}: two consecutive "
            "runs produced different byte content."
        )


# ---------------------------------------------------------------------------
# AC-NB13-4: dataset cell is horizon-aware (checked on h=5 notebook)
# ---------------------------------------------------------------------------

class TestNotebook13DatasetCell:
    """Verify the dataset cell injects HORIZON and uses it correctly."""

    def test_dataset_cell_is_horizon_aware(self):
        """AC-NB13-4: h=5 notebook dataset cell contains HORIZON=5, window_size=T_IN+HORIZON,
        and horizon=HORIZON in make_window_index. Must NOT contain T_IN + T_OUT for window_size.
        """
        nb_path = NB_PATHS[5]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-13-dataset")

        assert "HORIZON = 5" in src, (
            f"Expected 'HORIZON = 5' in cell-13-dataset of h=5 notebook, got:\n{src[:500]}"
        )
        assert "window_size = T_IN + HORIZON" in src, (
            f"Expected 'window_size = T_IN + HORIZON' in cell-13-dataset, got:\n{src[:500]}"
        )
        assert "horizon=HORIZON" in src, (
            f"Expected 'horizon=HORIZON' in make_window_index call in cell-13-dataset, "
            f"got:\n{src[:500]}"
        )
        # Must NOT use T_IN + T_OUT for the window_size calculation
        assert "T_IN + T_OUT" not in src, (
            "cell-13-dataset must NOT use 'T_IN + T_OUT' for window_size (that's the "
            "h=1-only formula). Use 'T_IN + HORIZON' instead."
        )


# ---------------------------------------------------------------------------
# AC-NB13-5: train cell uses WINNING_CONFIGS with nhead/d_model (no grid over TRANSFORMER_GRID)
# ---------------------------------------------------------------------------

class TestNotebook13TrainCell:
    """Verify the train cell uses the Transformer winning configs dict, not a full grid search."""

    def test_train_cell_uses_winning_config_with_nhead(self):
        """AC-NB13-5: train cell contains WINNING_CONFIGS dict with nhead= and d_model= fields,
        and passes it as a single-element list.
        """
        # Check on h=1 notebook (any would work, they all have the same train cell)
        nb_path = NB_PATHS[1]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-13-train")

        assert "WINNING_CONFIGS" in src, (
            f"Expected 'WINNING_CONFIGS' in cell-13-train, got:\n{src[:500]}"
        )
        assert "nhead=" in src, (
            f"Expected 'nhead=' in WINNING_CONFIGS inside cell-13-train (Transformer-specific), "
            f"got:\n{src[:500]}"
        )
        assert "d_model=" in src, (
            f"Expected 'd_model=' in WINNING_CONFIGS inside cell-13-train (Transformer-specific), "
            f"got:\n{src[:500]}"
        )
        assert "configs=[WINNING_CONFIGS[label]]" in src, (
            f"Expected 'configs=[WINNING_CONFIGS[label]]' in cell-13-train, "
            f"got:\n{src[:500]}"
        )
        assert "configs=TRANSFORMER_GRID" not in src, (
            "cell-13-train must NOT use 'configs=TRANSFORMER_GRID' (full grid search). "
            "Use the single winning config instead."
        )


# ---------------------------------------------------------------------------
# AC-NB13-6: compare cell reads multih baselines and filters by HORIZON
# ---------------------------------------------------------------------------

class TestNotebook13CompareCell:
    """Verify the compare cell reads the multi-horizon baselines CSV."""

    def test_compare_reads_multih_baselines(self):
        """AC-NB13-6: compare cell references baselines_results_multih.csv
        and filters by pl.col("horizon") == HORIZON.
        """
        nb_path = NB_PATHS[3]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-13-compare")

        assert "baselines_results_multih.csv" in src, (
            f"Expected 'baselines_results_multih.csv' in cell-13-compare, "
            f"got:\n{src[:500]}"
        )
        assert 'pl.col("horizon") == HORIZON' in src, (
            f"Expected 'pl.col(\"horizon\") == HORIZON' filter in cell-13-compare, "
            f"got:\n{src[:500]}"
        )


# ---------------------------------------------------------------------------
# AC-NB13-7: results cell has horizon column and correct filename pattern
# ---------------------------------------------------------------------------

class TestNotebook13ResultsCell:
    """Verify the results cell adds horizon column to the output CSV."""

    def test_results_csv_has_horizon_column(self):
        """AC-NB13-7: results cell contains 'horizon': HORIZON in the row dict,
        writes to spatial_transformer_results_h{H}.csv, and uses "SpatialTransformer" baseline.
        """
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-13-results")

        assert '"horizon": HORIZON' in src, (
            f"Expected '\"horizon\": HORIZON' in cell-13-results, got:\n{src[:500]}"
        )
        assert "spatial_transformer_results_h" in src, (
            f"Expected 'spatial_transformer_results_h' (horizon-discriminated filename) in "
            f"cell-13-results, got:\n{src[:500]}"
        )
        assert '"SpatialTransformer"' in src, (
            f"Expected '\"SpatialTransformer\"' baseline label in cell-13-results, "
            f"got:\n{src[:500]}"
        )


# ---------------------------------------------------------------------------
# AC-NB13-8: evaluate cell uses SpatialTransformer construction and 3-arg forward
# ---------------------------------------------------------------------------

class TestNotebook13EvaluateCell:
    """Verify the evaluate cell uses SpatialTransformer and spatial 3-arg forward."""

    def test_evaluate_uses_spatial_transformer_and_3arg_forward(self):
        """AC-NB13-8: h=5 evaluate cell must instantiate SpatialTransformer(
        and call model(inp, ctx, input_mask) — not model(x) like LSTM.
        """
        nb_path = NB_PATHS[5]
        if not nb_path.exists():
            result = _run_builder()
            assert result.returncode == 0, (
                f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        src = _cell_source(nb_path, "cell-13-evaluate")

        assert "SpatialTransformer(" in src, (
            f"Expected 'SpatialTransformer(' construction in cell-13-evaluate, "
            f"got:\n{src[:500]}"
        )
        assert "model(inp, ctx, input_mask)" in src, (
            f"Expected 'model(inp, ctx, input_mask)' (3-arg spatial forward) in "
            f"cell-13-evaluate, got:\n{src[:500]}"
        )
        # Must NOT use the LSTM-style cat forward
        assert "torch.cat([inp, ctx]" not in src, (
            "cell-13-evaluate must NOT use 'torch.cat([inp, ctx])' (LSTM forward). "
            "SpatialTransformer uses the 3-arg spatial dispatch."
        )
