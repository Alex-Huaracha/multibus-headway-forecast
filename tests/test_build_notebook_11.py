"""Tests for src/build_notebook_11.py.

Covers:
  AC-NB11-1: python -m src.build_notebook_11 exits 0 and produces all 4 per-horizon
             notebooks (h∈{1,3,5,10}) each in its own h{H}/ subdir under
             notebooks/11_lstm_multihorizon/.
  AC-NB11-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB11-3: every cell ID matches 'cell-11-*' (no random UUIDs) in all notebooks.
  AC-NB11-4: dataset cell is horizon-aware — HORIZON constant injected, window_size uses
             HORIZON (not T_OUT), make_window_index receives horizon= kwarg.
  AC-NB11-5: train cell uses WINNING_CONFIGS dict (no grid search over GRID).
  AC-NB11-6: compare cell reads baselines_results_multih.csv and filters by HORIZON.
  AC-NB11-7: results cell has horizon column (schema: corridor, direction, baseline,
             metric, value, horizon).
  AC-NB11-8: each h{H}/ subdir contains EXACTLY its .ipynb and its kernel-metadata.json;
             code_file matches the .ipynb in that subdir; all 4 ids are distinct;
             GPU fields and kernel_sources are correct.
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

# Parent directory for all NB11 per-horizon subdirs.
NB_DIR = ROOT / "notebooks" / "11_lstm_multihorizon"

HORIZONS = [1, 3, 5, 10]

# Each horizon now lives in its own h{H}/ subdir.
NB_PATHS: dict[int, Path] = {
    h: NB_DIR / f"h{h}" / f"11_lstm_h{h}.ipynb" for h in HORIZONS
}

META_PATHS: dict[int, Path] = {
    h: NB_DIR / f"h{h}" / "kernel-metadata.json" for h in HORIZONS
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
# AC-NB11-1: builder exits 0 and writes all 4 notebooks in h{H}/ subdirs
# ---------------------------------------------------------------------------

class TestNotebook11Builder:
    """Verify build_notebook_11.py produces correct, stable per-horizon notebooks."""

    def test_builder_exits_zero_and_writes_notebooks(self):
        """AC-NB11-1: builder exits 0; each h{H}/ subdir contains its .ipynb."""
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
    # AC-NB11-2: byte-identical on second run (per subdir)
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


# ---------------------------------------------------------------------------
# AC-NB11-8: one kernel-metadata.json per h{H}/ subdir — each correct and distinct
# ---------------------------------------------------------------------------

class TestNotebook11KernelMetadata:
    """Verify build_notebook_11.py writes one kernel-metadata.json per h{H}/ subdir."""

    def _ensure_built(self) -> None:
        result = _run_builder()
        assert result.returncode == 0, (
            f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # -----------------------------------------------------------------------
    # AC-NB11-8a: each subdir contains EXACTLY its .ipynb + kernel-metadata.json
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_subdir_contains_exactly_notebook_and_metadata(self, horizon: int):
        """AC-NB11-8a: h{H}/ subdir must contain exactly the .ipynb and kernel-metadata.json."""
        self._ensure_built()
        subdir = NB_DIR / f"h{horizon}"
        assert subdir.is_dir(), f"Expected subdir {subdir} to exist"
        files = sorted(p.name for p in subdir.iterdir())
        expected = sorted([f"11_lstm_h{horizon}.ipynb", "kernel-metadata.json"])
        assert files == expected, (
            f"h{horizon}/ subdir must contain exactly {expected}, got {files}"
        )

    # -----------------------------------------------------------------------
    # AC-NB11-8b: code_file matches the .ipynb in that subdir
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_code_file_matches_notebook(self, horizon: int):
        """AC-NB11-8b: code_file in each h{H}/kernel-metadata.json matches that subdir's .ipynb."""
        self._ensure_built()
        meta_path = META_PATHS[horizon]
        assert meta_path.exists(), f"Expected {meta_path} to exist"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_code_file = f"11_lstm_h{horizon}.ipynb"
        assert meta["code_file"] == expected_code_file, (
            f"h{horizon}/kernel-metadata.json code_file must be {expected_code_file!r}, "
            f"got: {meta['code_file']!r}"
        )

    # -----------------------------------------------------------------------
    # AC-NB11-8c: all 4 metadata have distinct ids
    # -----------------------------------------------------------------------

    def test_all_metadata_ids_are_distinct(self):
        """AC-NB11-8c: each h{H}/kernel-metadata.json has a unique id."""
        self._ensure_built()
        ids = []
        for h in HORIZONS:
            meta = json.loads(META_PATHS[h].read_text(encoding="utf-8"))
            ids.append(meta["id"])
        assert len(ids) == len(set(ids)), (
            f"All 4 kernel-metadata.json files must have distinct ids, got: {ids}"
        )
        for h in HORIZONS:
            meta = json.loads(META_PATHS[h].read_text(encoding="utf-8"))
            expected_id = f"alexhuaracha/11-lstm-multihorizon-h{h}"
            assert meta["id"] == expected_id, (
                f"h{h}/kernel-metadata.json id must be {expected_id!r}, got: {meta['id']!r}"
            )

    # -----------------------------------------------------------------------
    # AC-NB11-8d: GPU fields and kernel_sources
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_gpu_and_kernel_sources(self, horizon: int):
        """AC-NB11-8d: kernel_sources contains NB04 and NB10, enable_gpu is True."""
        self._ensure_built()
        meta = json.loads(META_PATHS[horizon].read_text(encoding="utf-8"))
        assert "alexhuaracha/04-preprocessing" in meta["kernel_sources"], (
            f"kernel_sources must contain 'alexhuaracha/04-preprocessing', "
            f"got: {meta['kernel_sources']!r}"
        )
        assert "alexhuaracha/10-baselines-multi-horizonte" in meta["kernel_sources"], (
            f"kernel_sources must contain 'alexhuaracha/10-baselines-multi-horizonte', "
            f"got: {meta['kernel_sources']!r}"
        )
        assert meta["enable_gpu"] is True, (
            f"enable_gpu must be True for DL notebook, got: {meta['enable_gpu']!r}"
        )
        assert meta.get("accelerator") == "GPU_T4X2", (
            f"accelerator must be 'GPU_T4X2', got: {meta.get('accelerator')!r}"
        )

    # -----------------------------------------------------------------------
    # AC-NB11-8e: base fields
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_base_fields(self, horizon: int):
        """AC-NB11-8e: language, kernel_type, is_private, enable_internet have correct values."""
        self._ensure_built()
        meta = json.loads(META_PATHS[horizon].read_text(encoding="utf-8"))
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_internet"] is True
        assert meta["dataset_sources"] == []
        assert meta["competition_sources"] == []

    # -----------------------------------------------------------------------
    # AC-NB11-8f: deterministic metadata per subdir
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_deterministic(self, horizon: int):
        """AC-NB11-8f: two consecutive runs produce byte-identical kernel-metadata.json per subdir."""
        result1 = _run_builder()
        assert result1.returncode == 0
        bytes_first = META_PATHS[horizon].read_bytes()

        result2 = _run_builder()
        assert result2.returncode == 0
        bytes_second = META_PATHS[horizon].read_bytes()

        assert bytes_first == bytes_second, (
            f"h{horizon}/kernel-metadata.json must be byte-identical across consecutive runs"
        )

    # -----------------------------------------------------------------------
    # AC-NB11-8g: no flat .ipynb or kernel-metadata.json at the NB_DIR root
    # -----------------------------------------------------------------------

    def test_no_flat_notebooks_at_root(self):
        """AC-NB11-8g: NB_DIR root must NOT contain any .ipynb or kernel-metadata.json directly."""
        self._ensure_built()
        flat_ipynb = list(NB_DIR.glob("*.ipynb"))
        flat_meta = list(NB_DIR.glob("kernel-metadata.json"))
        assert flat_ipynb == [], (
            f"NB_DIR root must not contain flat .ipynb files, found: {flat_ipynb}"
        )
        assert flat_meta == [], (
            f"NB_DIR root must not contain a flat kernel-metadata.json, found: {flat_meta}"
        )
