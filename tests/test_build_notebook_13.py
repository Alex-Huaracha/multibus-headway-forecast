"""Tests for src/build_notebook_13.py.

Covers:
  AC-NB13-1: python -m src.build_notebook_13 exits 0 and produces all 4 per-horizon
             notebooks (h∈{1,3,5,10}) each in its own h{H}/ subdir under
             notebooks/13_spatial_transformer_multihorizon/.
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
  AC-NB13-9: each h{H}/ subdir contains EXACTLY its .ipynb and its kernel-metadata.json;
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

# Parent directory for all NB13 per-horizon subdirs.
NB_DIR = ROOT / "notebooks" / "13_spatial_transformer_multihorizon"

HORIZONS = [1, 3, 5, 10]

# Each horizon now lives in its own h{H}/ subdir.
NB_PATHS: dict[int, Path] = {
    h: NB_DIR / f"h{h}" / f"13_spatial_transformer_h{h}.ipynb" for h in HORIZONS
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
# AC-NB13-1: builder exits 0 and writes all 4 notebooks in h{H}/ subdirs
# ---------------------------------------------------------------------------

class TestNotebook13Builder:
    """Verify build_notebook_13.py produces correct, stable per-horizon notebooks."""

    def test_builder_exits_zero_and_writes_notebooks(self):
        """AC-NB13-1: builder exits 0; each h{H}/ subdir contains its .ipynb."""
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
    # AC-NB13-2: byte-identical on second run (per subdir)
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
# AC-NB13-10: residuals cell exports per-sample paired errors for significance
# ---------------------------------------------------------------------------

class TestNotebook13ResidualsCell:
    """Verify the residuals cell + persistence capture for significance tests."""

    def test_evaluate_cell_captures_persistence(self):
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-13-evaluate")
        compile(src, "cell-13-evaluate", "exec")
        assert "all_persist" in src and "inp[:, T_IN - 1, :]" in src

    def test_residuals_cell_schema_and_output(self):
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-13-residuals")
        compile(src, "cell-13-residuals", "exec")
        assert "spatial_transformer_residuals_h" in src
        for col in ("y_true", "y_pred_dl", "y_pred_persist", "corridor",
                    "direction", "horizon"):
            assert col in src, f"residuals schema missing {col!r}"
        assert "tmask & pmask" in src


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


# ---------------------------------------------------------------------------
# AC-NB13-9: one kernel-metadata.json per h{H}/ subdir — each correct and distinct
# ---------------------------------------------------------------------------

class TestNotebook13KernelMetadata:
    """Verify build_notebook_13.py writes one kernel-metadata.json per h{H}/ subdir."""

    def _ensure_built(self) -> None:
        result = _run_builder()
        assert result.returncode == 0, (
            f"Builder failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # -----------------------------------------------------------------------
    # AC-NB13-9a: each subdir contains EXACTLY its .ipynb + kernel-metadata.json
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_subdir_contains_exactly_notebook_and_metadata(self, horizon: int):
        """AC-NB13-9a: h{H}/ subdir must contain exactly the .ipynb and kernel-metadata.json."""
        self._ensure_built()
        subdir = NB_DIR / f"h{horizon}"
        assert subdir.is_dir(), f"Expected subdir {subdir} to exist"
        files = sorted(p.name for p in subdir.iterdir())
        expected = sorted([f"13_spatial_transformer_h{horizon}.ipynb", "kernel-metadata.json"])
        assert files == expected, (
            f"h{horizon}/ subdir must contain exactly {expected}, got {files}"
        )

    # -----------------------------------------------------------------------
    # AC-NB13-9b: code_file matches the .ipynb in that subdir
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_code_file_matches_notebook(self, horizon: int):
        """AC-NB13-9b: code_file in each h{H}/kernel-metadata.json matches that subdir's .ipynb."""
        self._ensure_built()
        meta_path = META_PATHS[horizon]
        assert meta_path.exists(), f"Expected {meta_path} to exist"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_code_file = f"13_spatial_transformer_h{horizon}.ipynb"
        assert meta["code_file"] == expected_code_file, (
            f"h{horizon}/kernel-metadata.json code_file must be {expected_code_file!r}, "
            f"got: {meta['code_file']!r}"
        )

    # -----------------------------------------------------------------------
    # AC-NB13-9c: all 4 metadata have distinct ids
    # -----------------------------------------------------------------------

    def test_all_metadata_ids_are_distinct(self):
        """AC-NB13-9c: each h{H}/kernel-metadata.json has a unique id."""
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
            expected_id = f"alexhuaracha/13-spatialtransformer-multihorizon-h{h}"
            assert meta["id"] == expected_id, (
                f"h{h}/kernel-metadata.json id must be {expected_id!r}, got: {meta['id']!r}"
            )

    # -----------------------------------------------------------------------
    # AC-NB13-9d: GPU fields and kernel_sources
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_gpu_and_kernel_sources(self, horizon: int):
        """AC-NB13-9d: kernel_sources contains NB04 and NB10, enable_gpu is True."""
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
    # AC-NB13-9e: base fields
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_base_fields(self, horizon: int):
        """AC-NB13-9e: base fields have correct values."""
        self._ensure_built()
        meta = json.loads(META_PATHS[horizon].read_text(encoding="utf-8"))
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_internet"] is True
        assert meta["dataset_sources"] == []
        assert meta["competition_sources"] == []

    # -----------------------------------------------------------------------
    # AC-NB13-9f: deterministic metadata per subdir
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_deterministic(self, horizon: int):
        """AC-NB13-9f: two consecutive runs produce byte-identical kernel-metadata.json per subdir."""
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
    # AC-NB13-9g: no flat .ipynb or kernel-metadata.json at the NB_DIR root
    # -----------------------------------------------------------------------

    def test_no_flat_notebooks_at_root(self):
        """AC-NB13-9g: NB_DIR root must NOT contain any .ipynb or kernel-metadata.json directly."""
        self._ensure_built()
        flat_ipynb = list(NB_DIR.glob("*.ipynb"))
        flat_meta = list(NB_DIR.glob("kernel-metadata.json"))
        assert flat_ipynb == [], (
            f"NB_DIR root must not contain flat .ipynb files, found: {flat_ipynb}"
        )
        assert flat_meta == [], (
            f"NB_DIR root must not contain a flat kernel-metadata.json, found: {flat_meta}"
        )
