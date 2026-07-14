"""Tests for src/build_notebook_18_e4_convlstm.py (E4-only SpatialConvLSTM).

Covers (mirror of test_build_notebook_11.py, E4-scoped):
  AC-NB18-1: builder exits 0 and produces all 4 per-horizon notebooks.
  AC-NB18-2: two consecutive runs are byte-identical.
  AC-NB18-3: every cell ID matches 'cell-18-*'.
  AC-NB18-4: dataset cell is horizon-aware.
  AC-NB18-5: train cell uses the single REUSED E2 config (no grid, no minigrid).
  AC-NB18-6: compare cell reads baselines_results_multih.csv and filters by HORIZON.
  AC-NB18-7: results cell has horizon column and writes spatial_conv_lstm_E4_results_h{H}.csv.
  AC-NB18-8: kernel-metadata per subdir — code_file/ids/slug/GPU/kernel_sources correct.
  AC-NB18-9: run cells are E4-only (E4 present, E2/E59 absent).
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

NB_DIR = ROOT / "notebooks" / "18_e4_convlstm"

HORIZONS = [1, 3, 5, 10]

NB_PATHS: dict[int, Path] = {
    h: NB_DIR / f"h{h}" / f"18_e4_convlstm_h{h}.ipynb" for h in HORIZONS
}

META_PATHS: dict[int, Path] = {
    h: NB_DIR / f"h{h}" / "kernel-metadata.json" for h in HORIZONS
}

ALL_NB_PATHS = list(NB_PATHS.values())

RUN_CELL_IDS = [
    "cell-18-load", "cell-18-split", "cell-18-norm", "cell-18-context",
    "cell-18-dataset", "cell-18-train", "cell-18-evaluate",
    "cell-18-results", "cell-18-residuals", "cell-18-compare",
]


def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_18_e4_convlstm"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _cell_source(nb_path: Path, cell_id: str) -> str:
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if cell.get("id") == cell_id:
            return cell.source
    raise AssertionError(f"{cell_id} not found in {nb_path}")


class TestNotebook18Builder:
    def test_builder_exits_zero_and_writes_notebooks(self):
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_18_e4_convlstm failed with code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        for h, nb_path in NB_PATHS.items():
            assert nb_path.exists(), f"Expected notebook for h={h} at {nb_path}"

    @pytest.mark.parametrize("nb_path", ALL_NB_PATHS, ids=[f"h{h}" for h in HORIZONS])
    def test_cell_ids_have_cell_18_prefix(self, nb_path: Path):
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        stable_pattern = re.compile(r"^cell-18-")
        with open(nb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, f"Cell IDs not matching 'cell-18-*' in {nb_path.name}: {bad_ids}"

    @pytest.mark.parametrize("nb_path", ALL_NB_PATHS, ids=[f"h{h}" for h in HORIZONS])
    def test_second_run_byte_identical(self, nb_path: Path):
        assert _run_builder().returncode == 0
        assert nb_path.exists()
        bytes_first = nb_path.read_bytes()
        assert _run_builder().returncode == 0
        bytes_second = nb_path.read_bytes()
        assert bytes_first == bytes_second, f"Not idempotent for {nb_path.name}"


class TestNotebook18DatasetCell:
    def test_dataset_cell_is_horizon_aware(self):
        nb_path = NB_PATHS[5]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-18-dataset")
        assert "HORIZON = 5" in src
        assert "window_size = T_IN + HORIZON" in src
        assert "horizon=HORIZON" in src
        assert "T_IN + T_OUT" not in src


class TestNotebook18TrainCell:
    def test_train_cell_uses_reused_single_config(self):
        """AC-NB18-5: single reused E2 SpatialConvLSTM config (no grid/minigrid)."""
        nb_path = NB_PATHS[1]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-18-train")
        compile(src, "cell-18-train", "exec")
        # Exactly one config (the reused E2 winner), passed as a single-element list.
        assert src.count("TrainConfig(") == 1, "must reuse exactly one config"
        assert "configs=[E4_CONFIG]" in src
        # E2's SpatialConvLSTM winner: conv_channels=1, hidden=32, layers=1, dropout=0.2
        assert "conv_channels=1" in src
        assert "hidden_size=32" in src
        assert "num_layers=1" in src
        assert "dropout=0.2" in src
        assert "E4_MINIGRID" not in src
        assert "configs=SPATIAL_GRID" not in src


class TestNotebook18CompareCell:
    def test_compare_reads_multih_baselines(self):
        nb_path = NB_PATHS[3]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-18-compare")
        assert "baselines_results_multih.csv" in src
        assert 'pl.col("horizon") == HORIZON' in src


class TestNotebook18ResultsCell:
    def test_results_csv_has_horizon_column(self):
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-18-results")
        assert '"horizon": HORIZON' in src
        assert "spatial_conv_lstm_E4_results_h" in src


class TestNotebook18ResidualsCell:
    def test_evaluate_cell_captures_persistence(self):
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-18-evaluate")
        compile(src, "cell-18-evaluate", "exec")
        assert "all_persist" in src and "inp[:, T_IN - 1, :]" in src
        # spatial 3-arg forward dispatch
        assert "model(inp, ctx, input_mask)" in src

    def test_residuals_cell_schema_and_output(self):
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-18-residuals")
        compile(src, "cell-18-residuals", "exec")
        assert "spatial_conv_lstm_E4_residuals_h" in src
        for col in ("y_true", "y_pred_dl", "y_pred_persist", "corridor",
                    "direction", "horizon"):
            assert col in src, f"residuals schema missing {col!r}"
        assert "tmask & pmask" in src


class TestNotebook18E4Only:
    @pytest.mark.parametrize("cell_id", RUN_CELL_IDS)
    def test_run_cells_are_e4_only(self, cell_id: str):
        nb_path = NB_PATHS[1]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, cell_id)
        assert not re.search(r"\bE2\b", src), f"{cell_id} must not mention E2"
        assert not re.search(r"\bE59\b", src), f"{cell_id} must not mention E59"

    def test_e4_present_somewhere(self):
        nb_path = NB_PATHS[1]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-18-load")
        assert "E4" in src and '_resolve_input("headways_E4.parquet")' in src


class TestNotebook18KernelMetadata:
    def _ensure_built(self) -> None:
        assert _run_builder().returncode == 0

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_subdir_contains_exactly_notebook_and_metadata(self, horizon: int):
        self._ensure_built()
        subdir = NB_DIR / f"h{horizon}"
        assert subdir.is_dir()
        files = sorted(p.name for p in subdir.iterdir())
        expected = sorted([f"18_e4_convlstm_h{horizon}.ipynb", "kernel-metadata.json"])
        assert files == expected

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_code_file_matches_notebook(self, horizon: int):
        self._ensure_built()
        meta = json.loads(META_PATHS[horizon].read_text(encoding="utf-8"))
        assert meta["code_file"] == f"18_e4_convlstm_h{horizon}.ipynb"

    def test_all_metadata_ids_distinct_and_slug_consistent(self):
        self._ensure_built()
        ids = []
        for h in HORIZONS:
            meta = json.loads(META_PATHS[h].read_text(encoding="utf-8"))
            ids.append(meta["id"])
            assert meta["id"] == f"alexhuaracha/18-e4-convlstm-h{h}"
            slug = meta["title"].lower().replace(" ", "-")
            assert meta["id"] == f"alexhuaracha/{slug}", (
                f"title {meta['title']!r} must slugify to id {meta['id']!r}"
            )
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_gpu_and_kernel_sources(self, horizon: int):
        self._ensure_built()
        meta = json.loads(META_PATHS[horizon].read_text(encoding="utf-8"))
        assert meta["kernel_sources"] == [
            "alexhuaracha/16-e4-data-baselines",
            "alexhuaracha/02-eda-corridors",
        ]
        assert meta["enable_gpu"] is True
        assert meta.get("accelerator") == "GPU_T4X2"

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_base_fields(self, horizon: int):
        self._ensure_built()
        meta = json.loads(META_PATHS[horizon].read_text(encoding="utf-8"))
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_internet"] is True
        assert meta["dataset_sources"] == []
        assert meta["competition_sources"] == []

    def test_no_flat_notebooks_at_root(self):
        self._ensure_built()
        assert list(NB_DIR.glob("*.ipynb")) == []
        assert list(NB_DIR.glob("kernel-metadata.json")) == []
