"""Tests for src/build_notebook_17_e4_lstm.py (E4-only LSTM, external validity).

Covers:
  AC-NB17-1: python -m src.build_notebook_17_e4_lstm exits 0 and produces all 4
             per-horizon notebooks (h∈{1,3,5,10}) each in its own h{H}/ subdir
             under notebooks/17_e4_lstm/.
  AC-NB17-2: two consecutive runs are byte-identical (idempotency / no cell-ID flutter).
  AC-NB17-3: every cell ID matches 'cell-17-*' (no random UUIDs) in all notebooks.
  AC-NB17-4: dataset cell is horizon-aware — HORIZON constant injected, window_size uses
             HORIZON (not T_OUT), make_window_index receives horizon= kwarg.
  AC-NB17-5: train cell uses E4_MINIGRID through grid_search and takes results[0]
             (no WINNING_CONFIGS dict, no role/zip metadata).
  AC-NB17-6: compare cell reads baselines_results_multih.csv and filters by HORIZON.
  AC-NB17-7: results cell has horizon column and writes lstm_E4_results_h{H}.csv.
  AC-NB17-8: each h{H}/ subdir contains EXACTLY its .ipynb and its kernel-metadata.json;
             code_file matches the .ipynb; ids distinct and slug-consistent;
             GPU fields and kernel_sources (16-e4-data-baselines) correct.
  AC-NB17-9: run cells are E4-only — contain "E4" and NOT "E2"/"E59".
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

NB_DIR = ROOT / "notebooks" / "17_e4_lstm"

HORIZONS = [1, 3, 5, 10]

NB_PATHS: dict[int, Path] = {
    h: NB_DIR / f"h{h}" / f"17_e4_lstm_h{h}.ipynb" for h in HORIZONS
}

META_PATHS: dict[int, Path] = {
    h: NB_DIR / f"h{h}" / "kernel-metadata.json" for h in HORIZONS
}

ALL_NB_PATHS = list(NB_PATHS.values())

# Cells that contain the actual E4 pipeline (load/split/norm/context/dataset/
# train/evaluate/results/residuals/compare). Embed cells are excluded because
# they are verbatim library code that legitimately may mention nothing of E4.
RUN_CELL_IDS = [
    "cell-17-load", "cell-17-split", "cell-17-norm", "cell-17-context",
    "cell-17-dataset", "cell-17-train", "cell-17-evaluate",
    "cell-17-results", "cell-17-residuals", "cell-17-compare",
]


def _run_builder() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.build_notebook_17_e4_lstm"],
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


class TestNotebook17Builder:
    def test_builder_exits_zero_and_writes_notebooks(self):
        """AC-NB17-1: builder exits 0; each h{H}/ subdir contains its .ipynb."""
        result = _run_builder()
        assert result.returncode == 0, (
            f"src.build_notebook_17_e4_lstm failed with code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        for h, nb_path in NB_PATHS.items():
            assert nb_path.exists(), f"Expected notebook for h={h} at {nb_path}"

    @pytest.mark.parametrize("nb_path", ALL_NB_PATHS, ids=[f"h{h}" for h in HORIZONS])
    def test_cell_ids_have_cell_17_prefix(self, nb_path: Path):
        """AC-NB17-3: every cell ID must match 'cell-17-*'."""
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        stable_pattern = re.compile(r"^cell-17-")
        with open(nb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bad_ids = [
            cell.get("id", "<no-id>")
            for cell in nb.cells
            if not stable_pattern.match(cell.get("id", ""))
        ]
        assert not bad_ids, f"Cell IDs not matching 'cell-17-*' in {nb_path.name}: {bad_ids}"

    @pytest.mark.parametrize("nb_path", ALL_NB_PATHS, ids=[f"h{h}" for h in HORIZONS])
    def test_second_run_byte_identical(self, nb_path: Path):
        """AC-NB17-2: two consecutive runs produce byte-identical output."""
        assert _run_builder().returncode == 0
        assert nb_path.exists()
        bytes_first = nb_path.read_bytes()
        assert _run_builder().returncode == 0
        bytes_second = nb_path.read_bytes()
        assert bytes_first == bytes_second, f"Not idempotent for {nb_path.name}"


class TestNotebook17DatasetCell:
    def test_dataset_cell_is_horizon_aware(self):
        """AC-NB17-4: h=5 dataset cell has HORIZON=5, window_size=T_IN+HORIZON,
        horizon=HORIZON; must NOT use T_IN + T_OUT."""
        nb_path = NB_PATHS[5]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-17-dataset")
        assert "HORIZON = 5" in src
        assert "window_size = T_IN + HORIZON" in src
        assert "horizon=HORIZON" in src
        assert "T_IN + T_OUT" not in src


class TestNotebook17TrainCell:
    def test_train_cell_uses_minigrid_through_grid_search(self):
        """AC-NB17-5: train cell defines E4_MINIGRID, runs it through grid_search,
        and takes results[0] as the winner. No WINNING_CONFIGS dict, no role/zip."""
        nb_path = NB_PATHS[1]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-17-train")
        compile(src, "cell-17-train", "exec")  # syntax check

        assert "E4_MINIGRID" in src
        assert "configs=E4_MINIGRID" in src, "mini-grid must be passed to grid_search"
        assert "grid_search(" in src
        assert "results[0]" in src, "results[0] is the mini-grid winner (sorted by val_loss)"
        # Must be a list of 3 candidate configs.
        assert src.count("TrainConfig(") == 3, "E4_MINIGRID must have exactly 3 configs"
        # No reused-config dict and no role/zip selection bug.
        assert "WINNING_CONFIGS" not in src
        assert "zip(" not in src, "no role/zip metadata selection — results[0] is the winner"


class TestNotebook17CompareCell:
    def test_compare_reads_multih_baselines(self):
        """AC-NB17-6: compare cell references baselines_results_multih.csv and
        filters by horizon."""
        nb_path = NB_PATHS[3]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-17-compare")
        assert "baselines_results_multih.csv" in src
        assert 'pl.col("horizon") == HORIZON' in src


class TestNotebook17ResultsCell:
    def test_results_csv_has_horizon_column(self):
        """AC-NB17-7: results cell has 'horizon': HORIZON and writes
        lstm_E4_results_h{H}.csv (E4-scoped filename)."""
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-17-results")
        assert '"horizon": HORIZON' in src
        assert "lstm_E4_results_h" in src


class TestNotebook17ResidualsCell:
    def test_evaluate_cell_captures_persistence(self):
        """AC-NB17-9: evaluate cell captures persistence from the last input step."""
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-17-evaluate")
        compile(src, "cell-17-evaluate", "exec")
        assert "all_persist" in src and "inp[:, T_IN - 1, :]" in src

    def test_residuals_cell_schema_and_output(self):
        """AC-NB17-9: residuals cell writes lstm_E4_residuals_h{H}.csv with the
        6-column paired schema."""
        nb_path = NB_PATHS[10]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-17-residuals")
        compile(src, "cell-17-residuals", "exec")
        assert "lstm_E4_residuals_h" in src
        for col in ("y_true", "y_pred_dl", "y_pred_persist", "corridor",
                    "direction", "horizon"):
            assert col in src, f"residuals schema missing {col!r}"
        assert "tmask & pmask" in src


class TestNotebook17E4Only:
    @pytest.mark.parametrize("cell_id", RUN_CELL_IDS)
    def test_run_cells_are_e4_only(self, cell_id: str):
        """AC-NB17-9: run cells contain E4 and never E2/E59."""
        nb_path = NB_PATHS[1]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, cell_id)
        assert not re.search(r"\bE2\b", src), f"{cell_id} must not mention E2"
        assert not re.search(r"\bE59\b", src), f"{cell_id} must not mention E59"

    def test_e4_present_somewhere(self):
        """E4 must appear in the run pipeline (load cell at minimum)."""
        nb_path = NB_PATHS[1]
        if not nb_path.exists():
            assert _run_builder().returncode == 0
        src = _cell_source(nb_path, "cell-17-load")
        assert "E4" in src and "_find_parquet(4)" in src


class TestNotebook17KernelMetadata:
    def _ensure_built(self) -> None:
        assert _run_builder().returncode == 0

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_subdir_contains_exactly_notebook_and_metadata(self, horizon: int):
        self._ensure_built()
        subdir = NB_DIR / f"h{horizon}"
        assert subdir.is_dir()
        files = sorted(p.name for p in subdir.iterdir())
        expected = sorted([f"17_e4_lstm_h{horizon}.ipynb", "kernel-metadata.json"])
        assert files == expected

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_code_file_matches_notebook(self, horizon: int):
        self._ensure_built()
        meta = json.loads(META_PATHS[horizon].read_text(encoding="utf-8"))
        assert meta["code_file"] == f"17_e4_lstm_h{horizon}.ipynb"

    def test_all_metadata_ids_distinct_and_slug_consistent(self):
        """AC-NB17-8: distinct ids; title slugifies to the id."""
        self._ensure_built()
        ids = []
        for h in HORIZONS:
            meta = json.loads(META_PATHS[h].read_text(encoding="utf-8"))
            ids.append(meta["id"])
            expected_id = f"alexhuaracha/17-e4-lstm-h{h}"
            assert meta["id"] == expected_id
            # title slugifies to the id slug part
            slug = meta["title"].lower().replace(" ", "-")
            assert meta["id"] == f"alexhuaracha/{slug}", (
                f"title {meta['title']!r} must slugify to id {meta['id']!r}"
            )
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_kernel_metadata_gpu_and_kernel_sources(self, horizon: int):
        self._ensure_built()
        meta = json.loads(META_PATHS[horizon].read_text(encoding="utf-8"))
        assert meta["kernel_sources"] == ["alexhuaracha/16-e4-data-baselines"]
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
