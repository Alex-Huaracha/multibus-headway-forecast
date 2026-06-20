"""Tests for src/build_notebook_15.py.

NB15 is a MULTI-SEED ROBUSTNESS study of the frozen winning LSTM config.
It re-trains the SAME winning config N times with N different seeds and exports
per-seed metrics, so confidence intervals can be drawn on the degradation curve.

It is a faithful clone of build_notebook_11.py's scaffolding (per-horizon loop,
one notebook per horizon h in {1, 3, 5, 10}) with these surgical differences:
  - output: notebooks/15_lstm_multiseed/h{H}/15_lstm_multiseed_h{H}.ipynb
  - cell IDs: cell-15-*
  - kernel id/title/code_file: 15-lstm-multiseed-h{H} / 15 LSTM Multiseed h{H}
  - train cell loops over SEEDS = [42, 123, 456, 789, 999], building a per-seed
    config via dataclasses.replace(WINNING_CONFIGS[label], seed=s)
  - results CSV lstm_multiseed_h{H}.csv has an extra `seed` column
  - NB11's residuals + compare cells are dropped

Covers:
  AC-NB15-1: python -m src.build_notebook_15 exits 0 and produces, per horizon,
             the notebook + kernel-metadata.json.
  AC-NB15-2: two consecutive runs are byte-identical (idempotency).
  AC-NB15-3: every cell ID matches 'cell-15-*' (no random UUIDs).
  AC-NB15-4: setup/dataset cell injects HORIZON={H} and lstm_multiseed_h{H}.csv.
  AC-NB15-5: train cell injects SEEDS = [42, 123, 456, 789, 999], loops over
             them and uses dataclasses.replace(..., seed=...).
  AC-NB15-6: train cell carries the frozen WINNING_CONFIGS (num_layers=1 and
             num_layers=2, hidden_size=32, lr=5e-4).
  AC-NB15-7: results cell schema includes seed plus
             corridor/direction/baseline/metric/value/horizon.
  AC-NB15-8: kernel-metadata.json has the correct id/title/code_file/GPU/sources.
  AC-NB15-9: each horizon dir contains EXACTLY .ipynb + kernel-metadata.json.
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

HORIZONS = [1, 3, 5, 10]


def _nb_dir(h: int) -> Path:
    return ROOT / "notebooks" / "15_lstm_multiseed" / f"h{h}"


def _nb_path(h: int) -> Path:
    return _nb_dir(h) / f"15_lstm_multiseed_h{h}.ipynb"


def _meta_path(h: int) -> Path:
    return _nb_dir(h) / "kernel-metadata.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_15"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _cell_source(h: int, cell_id: str) -> str:
    """Return the source of the cell with the given ID for horizon h's notebook."""
    with open(_nb_path(h), encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if cell.get("id") == cell_id:
            return cell.source
    raise AssertionError(f"{cell_id!r} not found in {_nb_path(h)}")


def _ensure_built() -> None:
    result = _run_builder()
    assert result.returncode == 0, (
        f"src.build_notebook_15 failed.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# AC-NB15-1: builder exits 0 and writes notebook + metadata per horizon
# ---------------------------------------------------------------------------

class TestNotebook15Builder:
    """Verify build_notebook_15.py produces the expected output files."""

    def test_builder_exits_zero(self):
        """AC-NB15-1: builder exits 0."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_15 failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    @pytest.mark.parametrize("h", HORIZONS)
    def test_builder_writes_notebook_and_metadata(self, h):
        """AC-NB15-1: each horizon has its notebook + kernel-metadata.json."""
        _ensure_built()
        assert _nb_path(h).exists(), f"Expected notebook at {_nb_path(h)}"
        assert _meta_path(h).exists(), f"Expected kernel-metadata.json at {_meta_path(h)}"

    @pytest.mark.parametrize("h", HORIZONS)
    def test_cell_ids_have_cell_15_prefix(self, h):
        """AC-NB15-3: every cell ID must match 'cell-15-*' (no uuid4)."""
        _ensure_built()
        stable_pattern = re.compile(r"^cell-15-")
        with open(_nb_path(h), encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, (
            f"h={h}: the following cell IDs do not match 'cell-15-*': {bad_ids}. "
            "All cell IDs must be explicitly set to 'cell-15-<name>'."
        )

    def test_second_run_byte_identical(self):
        """AC-NB15-2: two consecutive runs must produce byte-identical output."""
        result1 = _run_builder()
        assert result1.returncode == 0, (
            f"First run failed.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
        )
        first_nb = {h: _nb_path(h).read_bytes() for h in HORIZONS}
        first_meta = {h: _meta_path(h).read_bytes() for h in HORIZONS}

        result2 = _run_builder()
        assert result2.returncode == 0, (
            f"Second run failed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
        )
        for h in HORIZONS:
            assert _nb_path(h).read_bytes() == first_nb[h], (
                f"build_notebook_15 is NOT idempotent: h={h} notebook changed between runs."
            )
            assert _meta_path(h).read_bytes() == first_meta[h], (
                f"build_notebook_15 is NOT idempotent: h={h} metadata changed between runs."
            )


# ---------------------------------------------------------------------------
# AC-NB15-4: setup/dataset cells inject HORIZON and output filename
# ---------------------------------------------------------------------------

class TestNotebook15SetupCell:
    """Verify per-horizon constant injection."""

    @pytest.mark.parametrize("h", HORIZONS)
    def test_setup_cell_horizon_and_csv_name(self, h):
        """AC-NB15-4: setup cell contains HORIZON={h} and lstm_multiseed_h{h}.csv."""
        _ensure_built()
        src = _cell_source(h, "cell-15-setup")
        assert f"HORIZON = {h}" in src, (
            f"Expected 'HORIZON = {h}' in cell-15-setup, got:\n{src[:500]}"
        )
        assert "lstm_multiseed_h" in src, (
            f"Expected horizon-discriminated 'lstm_multiseed_h*.csv' in cell-15-setup, "
            f"got:\n{src[:500]}"
        )

    @pytest.mark.parametrize("h", HORIZONS)
    def test_dataset_cell_injects_horizon(self, h):
        """AC-NB15-4: dataset cell also injects HORIZON={h}."""
        _ensure_built()
        src = _cell_source(h, "cell-15-dataset")
        assert f"HORIZON = {h}" in src, (
            f"Expected 'HORIZON = {h}' in cell-15-dataset, got:\n{src[:500]}"
        )


# ---------------------------------------------------------------------------
# AC-NB15-5 / AC-NB15-6: train cell seed loop + frozen winning configs
# ---------------------------------------------------------------------------

class TestNotebook15TrainCell:
    """Verify the train cell loops over seeds with the frozen winning config."""

    @pytest.mark.parametrize("h", HORIZONS)
    def test_train_cell_defines_seeds(self, h):
        """AC-NB15-5: train cell injects SEEDS = [42, 123, 456, 789, 999]."""
        _ensure_built()
        src = _cell_source(h, "cell-15-train")
        assert "SEEDS = [42, 123, 456, 789, 999]" in src, (
            f"Expected 'SEEDS = [42, 123, 456, 789, 999]' in cell-15-train, "
            f"got:\n{src[:800]}"
        )

    @pytest.mark.parametrize("h", HORIZONS)
    def test_train_cell_loops_over_seeds_with_replace(self, h):
        """AC-NB15-5: train cell loops over SEEDS and uses dataclasses.replace(..., seed=...)."""
        _ensure_built()
        src = _cell_source(h, "cell-15-train")
        assert "import dataclasses" in src, (
            "cell-15-train must 'import dataclasses' to build per-seed configs."
        )
        assert "for" in src and "SEEDS" in src, (
            "cell-15-train must loop over SEEDS."
        )
        assert "dataclasses.replace(" in src, (
            "cell-15-train must build per-seed configs via dataclasses.replace(...)."
        )
        assert "seed=" in src, (
            "cell-15-train must pass seed= to dataclasses.replace(...)."
        )

    @pytest.mark.parametrize("h", HORIZONS)
    def test_train_cell_winning_configs(self, h):
        """AC-NB15-6: train cell carries the frozen WINNING_CONFIGS."""
        _ensure_built()
        src = _cell_source(h, "cell-15-train")
        assert "WINNING_CONFIGS" in src, "Expected 'WINNING_CONFIGS' in cell-15-train."
        # E2 winner: hidden=32, layers=1, dropout=0.0, lr=5e-4
        # E59 winner: hidden=32, layers=2, dropout=0.2, lr=5e-4
        assert "num_layers=1" in src, "E2 winner: num_layers=1 missing"
        assert "num_layers=2" in src, "E59 winner: num_layers=2 missing"
        assert "hidden_size=32" in src, "winners share hidden_size=32"
        assert "lr=5e-4" in src or "lr=0.0005" in src, "winners share lr=5e-4"

    @pytest.mark.parametrize("h", HORIZONS)
    def test_train_cell_collects_per_seed_results(self, h):
        """AC-NB15-5: each corridor must yield a list of (seed, TrainResult)."""
        _ensure_built()
        src = _cell_source(h, "cell-15-train")
        # grid_search is called with a single per-seed config list
        assert "grid_search(" in src, "cell-15-train must call grid_search per seed."
        assert "configs=[" in src, (
            "cell-15-train must pass a single per-seed config list to grid_search "
            "(configs=[cfg])."
        )


# ---------------------------------------------------------------------------
# AC-NB15-7: results cell schema includes the seed column
# ---------------------------------------------------------------------------

class TestNotebook15ResultsCell:
    """Verify the results/export cell writes the correct long-form CSV schema."""

    @pytest.mark.parametrize("h", HORIZONS)
    def test_results_cell_csv_schema(self, h):
        """AC-NB15-7: results cell writes CSV with
        corridor, direction, baseline, metric, value, horizon, seed.
        """
        _ensure_built()
        src = _cell_source(h, "cell-15-results")
        required_cols = [
            "corridor", "direction", "baseline", "metric", "value", "horizon", "seed"
        ]
        for col in required_cols:
            assert f'"{col}"' in src, (
                f"Expected column {col!r} in cell-15-results CSV schema, "
                f"not found in:\n{src[:800]}"
            )
        assert "lstm_multiseed_h" in src, (
            "Expected output filename 'lstm_multiseed_h*.csv' in cell-15-results"
        )
        # baseline stays 'LSTM'
        assert '"LSTM"' in src, "baseline value must be 'LSTM' in cell-15-results"
        # aggregate direction still present per seed
        assert "aggregate" in src, (
            "cell-15-results must still emit the 'aggregate' direction per seed."
        )

    @pytest.mark.parametrize("h", HORIZONS)
    def test_residuals_and_compare_cells_dropped(self, h):
        """NB15 drops NB11's residuals + compare cells."""
        _ensure_built()
        with open(_nb_path(h), encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        ids = {cell.get("id") for cell in nb.cells}
        assert "cell-15-residuals" not in ids, "NB15 must NOT contain a residuals cell."
        assert "cell-15-compare" not in ids, "NB15 must NOT contain a compare cell."


# ---------------------------------------------------------------------------
# AC-NB15-8: kernel-metadata.json correctness
# ---------------------------------------------------------------------------

class TestNotebook15KernelMetadata:
    """Verify build_notebook_15.py writes correct kernel-metadata.json per horizon."""

    @pytest.mark.parametrize("h", HORIZONS)
    def test_kernel_metadata_id_title_code_file(self, h):
        """AC-NB15-8: id, title, and code_file are correct per horizon."""
        _ensure_built()
        meta = json.loads(_meta_path(h).read_text(encoding="utf-8"))
        assert meta["id"] == f"alexhuaracha/15-lstm-multiseed-h{h}", (
            f"Expected id 'alexhuaracha/15-lstm-multiseed-h{h}', got: {meta['id']!r}"
        )
        assert meta["title"] == f"15 LSTM Multiseed h{h}", (
            f"Expected title '15 LSTM Multiseed h{h}', got: {meta['title']!r}"
        )
        assert meta["code_file"] == f"15_lstm_multiseed_h{h}.ipynb", (
            f"Expected code_file '15_lstm_multiseed_h{h}.ipynb', got: {meta['code_file']!r}"
        )

    @pytest.mark.parametrize("h", HORIZONS)
    def test_kernel_metadata_gpu_and_kernel_sources(self, h):
        """AC-NB15-8: GPU fields and kernel_sources match NB11 pattern."""
        _ensure_built()
        meta = json.loads(_meta_path(h).read_text(encoding="utf-8"))
        assert meta["enable_gpu"] is True
        assert meta.get("accelerator") == "GPU_T4X2"
        assert "alexhuaracha/04-preprocessing" in meta["kernel_sources"], (
            f"kernel_sources must contain '04-preprocessing', got: {meta['kernel_sources']}"
        )
        assert "alexhuaracha/10-baselines-multi-horizonte" in meta["kernel_sources"], (
            f"kernel_sources must contain '10-baselines-multi-horizonte', "
            f"got: {meta['kernel_sources']}"
        )

    @pytest.mark.parametrize("h", HORIZONS)
    def test_kernel_metadata_base_fields(self, h):
        """AC-NB15-8: language, kernel_type, is_private, enable_internet are correct."""
        _ensure_built()
        meta = json.loads(_meta_path(h).read_text(encoding="utf-8"))
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_internet"] is True
        assert meta["dataset_sources"] == []
        assert meta["competition_sources"] == []


# ---------------------------------------------------------------------------
# AC-NB15-9: each horizon dir contains exactly .ipynb + kernel-metadata.json
# ---------------------------------------------------------------------------

class TestNotebook15DirectoryLayout:
    """Verify each output directory has exactly the two expected files."""

    @pytest.mark.parametrize("h", HORIZONS)
    def test_directory_contains_exactly_notebook_and_metadata(self, h):
        """AC-NB15-9: h{H}/ must contain exactly .ipynb + kernel-metadata.json."""
        _ensure_built()
        d = _nb_dir(h)
        assert d.is_dir(), f"Expected directory {d} to exist"
        files = sorted(p.name for p in d.iterdir())
        expected = sorted([f"15_lstm_multiseed_h{h}.ipynb", "kernel-metadata.json"])
        assert files == expected, (
            f"{d} must contain exactly {expected}, got {files}"
        )
