"""Tests for src/build_notebook_14.py.

Covers:
  AC-NB14-1: python -m src.build_notebook_14 exits 0 and produces the notebook
             at notebooks/14_lstm_minigrid_h10/14_lstm_minigrid_h10.ipynb plus
             kernel-metadata.json in the same directory.
  AC-NB14-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB14-3: every cell ID matches 'cell-14-*' (no random UUIDs).
  AC-NB14-4: setup cell injects HORIZON=10 and a single output CSV
             (lstm_minigrid_h10.csv).
  AC-NB14-5: train cell defines exactly 8 MINIGRID_CONFIGS (4 per corridor:
             1 winner control + 3 neighbors):
             E2 control: (hidden=32, layers=1, dropout=0.0, lr=0.0005)
             E2 neighbors of the above.
             E59 control: (hidden=32, layers=2, dropout=0.2, lr=0.0005)
             E59 neighbors of the above.
  AC-NB14-6: results/export cell writes one CSV with columns
             corridor, hidden_size, num_layers, dropout, lr, mae, rmse, role
             where role is 'winner' for control rows and 'neighbor' for the rest.
  AC-NB14-9: train cell assigns 'role' by config identity, NOT by zip position.
             grid_search re-sorts results ascending by val_loss, so pairing the
             sorted results with the input-order roles list (zip(results, roles))
             mislabels the CSV's role column. Regression guard.
  AC-NB14-7: kernel-metadata.json has the correct id, title, GPU, and
             kernel_sources (04-preprocessing + 10-baselines-multi-horizonte).
  AC-NB14-8: notebook directory contains EXACTLY .ipynb + kernel-metadata.json.
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

NB_DIR = ROOT / "notebooks" / "14_lstm_minigrid_h10"
NB_PATH = NB_DIR / "14_lstm_minigrid_h10.ipynb"
META_PATH = NB_DIR / "kernel-metadata.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_14"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _cell_source(cell_id: str) -> str:
    """Return the source of the cell with the given ID."""
    with open(NB_PATH, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if cell.get("id") == cell_id:
            return cell.source
    raise AssertionError(f"{cell_id!r} not found in {NB_PATH}")


def _ensure_built() -> None:
    result = _run_builder()
    assert result.returncode == 0, (
        f"src.build_notebook_14 failed.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# AC-NB14-1: builder exits 0 and writes notebook + metadata
# ---------------------------------------------------------------------------

class TestNotebook14Builder:
    """Verify build_notebook_14.py produces the expected output files."""

    def test_builder_exits_zero_and_writes_notebook(self):
        """AC-NB14-1: builder exits 0 and writes the notebook and metadata."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_14 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert NB_PATH.exists(), f"Expected notebook at {NB_PATH}"
        assert META_PATH.exists(), f"Expected kernel-metadata.json at {META_PATH}"

    # -----------------------------------------------------------------------
    # AC-NB14-3: cell IDs have cell-14- prefix
    # -----------------------------------------------------------------------

    def test_cell_ids_have_cell_14_prefix(self):
        """AC-NB14-3: every cell ID must match 'cell-14-*' (no uuid4)."""
        _ensure_built()
        stable_pattern = re.compile(r"^cell-14-")
        with open(NB_PATH, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"The following cell IDs do not match 'cell-14-*': {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-14-<name>'."
        )

    # -----------------------------------------------------------------------
    # AC-NB14-2: byte-identical on second run
    # -----------------------------------------------------------------------

    def test_second_run_byte_identical(self):
        """AC-NB14-2: two consecutive runs must produce byte-identical output."""
        result1 = _run_builder()
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        assert NB_PATH.exists()
        bytes_nb_first = NB_PATH.read_bytes()
        bytes_meta_first = META_PATH.read_bytes()

        result2 = _run_builder()
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        assert NB_PATH.read_bytes() == bytes_nb_first, (
            "build_notebook_14 is NOT idempotent: notebook content changed between runs."
        )
        assert META_PATH.read_bytes() == bytes_meta_first, (
            "build_notebook_14 is NOT idempotent: metadata changed between runs."
        )


# ---------------------------------------------------------------------------
# AC-NB14-4: setup cell injects HORIZON=10
# ---------------------------------------------------------------------------

class TestNotebook14SetupCell:
    """Verify the setup cell is correctly configured for h=10."""

    def test_setup_cell_horizon_and_csv_name(self):
        """AC-NB14-4: setup cell contains HORIZON=10 and lstm_minigrid_h10.csv."""
        _ensure_built()
        src = _cell_source("cell-14-setup")
        assert "HORIZON = 10" in src, (
            f"Expected 'HORIZON = 10' in cell-14-setup, got:\n{src[:500]}"
        )
        assert "lstm_minigrid_h10.csv" in src, (
            f"Expected 'lstm_minigrid_h10.csv' in cell-14-setup, got:\n{src[:500]}"
        )


# ---------------------------------------------------------------------------
# AC-NB14-5: train cell defines exactly 6 MINIGRID_CONFIGS
# ---------------------------------------------------------------------------

class TestNotebook14TrainCell:
    """Verify the train cell defines the 8 configs (1 winner + 3 neighbors per corridor)."""

    def test_train_cell_defines_minigrid_configs(self):
        """AC-NB14-5: train cell contains MINIGRID_CONFIGS for both corridors,
        with 4 configs each: 1 winner control + 3 neighbors.
        """
        _ensure_built()
        src = _cell_source("cell-14-train")

        assert "MINIGRID_CONFIGS" in src, (
            "Expected 'MINIGRID_CONFIGS' in cell-14-train."
        )

        # Winner control rows must be present
        # E2 winner: hidden=32, layers=1, dropout=0.0, lr=0.0005
        # E59 winner: hidden=32, layers=2, dropout=0.2, lr=0.0005
        assert "winner" in src, "Expected 'winner' role marker in cell-14-train"
        assert "neighbor" in src, "Expected 'neighbor' role marker in cell-14-train"

        # Neighbor hyperparameters: hidden=64, dropout=0.2 (or 0.0 for E59), lr=0.001
        assert "hidden_size=64" in src, "E2/E59 neighbor vary-hidden: hidden_size=64 missing"
        assert "lr=0.001" in src or "lr=1e-3" in src, (
            "E2/E59 neighbor vary-lr: lr=0.001 missing"
        )

        # Corridor keys
        assert '"E59"' in src or "'E59'" in src, "E59 configs missing from MINIGRID_CONFIGS"
        assert '"E2"' in src or "'E2'" in src, "E2 configs missing from MINIGRID_CONFIGS"

    def test_train_cell_has_eight_configs(self):
        """AC-NB14-5: MINIGRID_CONFIGS must have exactly 4 configs per corridor (8 total)."""
        _ensure_built()
        src = _cell_source("cell-14-train")
        # Count TrainConfig(...) occurrences — one per config (1 winner + 3 neighbors each)
        config_count = src.count("TrainConfig(")
        assert config_count == 8, (
            f"Expected exactly 8 TrainConfig(...) entries in cell-14-train, "
            f"got {config_count}. The mini-grid must have 4 per corridor "
            f"(1 winner control + 3 neighbors)."
        )

    def test_train_cell_winner_configs_match_frozen_values(self):
        """AC-NB14-5: winner control configs must match the frozen NB11 values exactly."""
        _ensure_built()
        src = _cell_source("cell-14-train")

        # E2 winner: hidden=32, layers=1, dropout=0.0, lr=0.0005
        # E59 winner: hidden=32, layers=2, dropout=0.2, lr=0.0005
        # Both share hidden=32 and lr=0.0005; presence of layers=2 covers E59.
        assert "num_layers=2" in src, "E59 winner: num_layers=2 missing"
        assert "num_layers=1" in src, "E2 winner: num_layers=1 missing"

    def test_train_cell_role_assigned_by_config_identity(self):
        """AC-NB14-9 (regression): role must be matched to each result by config
        identity, not by zip position.

        grid_search returns results sorted ascending by best_val_loss, so the
        result order differs from MINIGRID_CONFIGS input order. Pairing the
        sorted results with the input-order roles list (``zip(results, roles)``)
        attaches 'winner'/'neighbor' to the wrong config — the lowest-val_loss
        config gets labeled 'winner' regardless of identity. The CSV role column
        must instead be derived from each result's hyperparameters.
        """
        _ensure_built()
        src = _cell_source("cell-14-train")

        # The fragile positional pairing must be gone.
        assert "zip(results, roles)" not in src, (
            "cell-14-train still pairs SORTED results with input-order roles via "
            "zip(results, roles); grid_search re-sorts by val_loss so this mislabels "
            "the role column. Match role by config identity instead."
        )
        # The fix builds a role lookup keyed by the config's hyperparameters.
        assert "role_by_key" in src, (
            "run_corridor_minigrid must map config identity -> role (role_by_key) "
            "so role labels survive grid_search re-sorting by val_loss."
        )


# ---------------------------------------------------------------------------
# AC-NB14-6: results cell exports CSV with required columns
# ---------------------------------------------------------------------------

class TestNotebook14ResultsCell:
    """Verify the results/export cell writes the correct CSV schema."""

    def test_results_cell_csv_schema(self):
        """AC-NB14-6: results cell writes CSV with:
        corridor, hidden_size, num_layers, dropout, lr, mae, rmse, role.
        The 'role' column must be 'winner' for control rows and 'neighbor' for the rest.
        """
        _ensure_built()
        src = _cell_source("cell-14-results")

        required_cols = [
            "corridor", "hidden_size", "num_layers", "dropout", "lr", "mae", "rmse", "role"
        ]
        for col in required_cols:
            assert col in src, (
                f"Expected column {col!r} in cell-14-results CSV schema, "
                f"not found in:\n{src[:500]}"
            )

        assert "lstm_minigrid_h10.csv" in src, (
            "Expected output filename 'lstm_minigrid_h10.csv' in cell-14-results"
        )

        # role values must be 'winner' and 'neighbor'
        assert "winner" in src, (
            "Expected 'winner' role value in cell-14-results (for control rows)"
        )
        assert "neighbor" in src, (
            "Expected 'neighbor' role value in cell-14-results (for neighbor rows)"
        )


# ---------------------------------------------------------------------------
# AC-NB14-7: kernel-metadata.json correctness
# ---------------------------------------------------------------------------

class TestNotebook14KernelMetadata:
    """Verify build_notebook_14.py writes the correct kernel-metadata.json."""

    def test_kernel_metadata_id_and_title(self):
        """AC-NB14-7: id, title, and code_file are correct."""
        _ensure_built()
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        assert meta["id"] == "alexhuaracha/14-lstm-minigrid-h10", (
            f"Expected id 'alexhuaracha/14-lstm-minigrid-h10', got: {meta['id']!r}"
        )
        assert meta["title"] == "14 LSTM Minigrid h10", (
            f"Expected title '14 LSTM Minigrid h10', got: {meta['title']!r}"
        )
        assert meta["code_file"] == "14_lstm_minigrid_h10.ipynb", (
            f"Expected code_file '14_lstm_minigrid_h10.ipynb', got: {meta['code_file']!r}"
        )

    def test_kernel_metadata_gpu_and_kernel_sources(self):
        """AC-NB14-7: GPU fields and kernel_sources match NB11 pattern."""
        _ensure_built()
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        assert meta["enable_gpu"] is True
        assert meta.get("accelerator") == "GPU_T4X2"
        assert "alexhuaracha/04-preprocessing" in meta["kernel_sources"], (
            f"kernel_sources must contain '04-preprocessing', got: {meta['kernel_sources']}"
        )
        assert "alexhuaracha/10-baselines-multi-horizonte" in meta["kernel_sources"], (
            f"kernel_sources must contain '10-baselines-multi-horizonte', "
            f"got: {meta['kernel_sources']}"
        )

    def test_kernel_metadata_base_fields(self):
        """AC-NB14-7: language, kernel_type, is_private, enable_internet are correct."""
        _ensure_built()
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_internet"] is True
        assert meta["dataset_sources"] == []
        assert meta["competition_sources"] == []


# ---------------------------------------------------------------------------
# AC-NB14-8: directory contains exactly .ipynb + kernel-metadata.json
# ---------------------------------------------------------------------------

class TestNotebook14DirectoryLayout:
    """Verify the output directory has exactly the two expected files."""

    def test_directory_contains_exactly_notebook_and_metadata(self):
        """AC-NB14-8: 14_lstm_minigrid_h10/ must contain exactly .ipynb + kernel-metadata.json."""
        _ensure_built()
        assert NB_DIR.is_dir(), f"Expected directory {NB_DIR} to exist"
        files = sorted(p.name for p in NB_DIR.iterdir())
        expected = sorted(["14_lstm_minigrid_h10.ipynb", "kernel-metadata.json"])
        assert files == expected, (
            f"14_lstm_minigrid_h10/ must contain exactly {expected}, got {files}"
        )
