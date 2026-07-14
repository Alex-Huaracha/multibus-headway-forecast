"""Determinism contracts for local report-builder entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import polars as pl

from src.build_paired_audit import build


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_BUILDERS = (
    "src.build_paired_audit",
    "src.build_exante_curve",
    "src.build_volatility_table",
    "src.build_volatility_curve",
)


def _polars_threads_at_first_import(module: str, threads: str | None) -> str:
    env = os.environ.copy()
    env.pop("POLARS_MAX_THREADS", None)
    if threads is not None:
        env["POLARS_MAX_THREADS"] = threads
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import importlib.abc",
                    "import importlib.machinery",
                    "import os",
                    "import sys",
                    "",
                    "class PolarsImportProbe(importlib.abc.MetaPathFinder):",
                    "    def find_spec(self, fullname, path=None, target=None):",
                    "        if fullname == 'polars':",
                    "            value = os.environ.get('POLARS_MAX_THREADS', '<unset>')",
                    "            print(f'POLARS_THREADS_AT_FIRST_IMPORT={value}', flush=True)",
                    "            sys.meta_path.remove(self)",
                    "            return importlib.machinery.PathFinder.find_spec(fullname, path, target)",
                    "        return None",
                    "",
                    "sys.meta_path.insert(0, PolarsImportProbe())",
                    f"import {module}",
                ]
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    observations = [
        line.removeprefix("POLARS_THREADS_AT_FIRST_IMPORT=")
        for line in result.stdout.splitlines()
        if line.startswith("POLARS_THREADS_AT_FIRST_IMPORT=")
    ]
    assert observations == [threads or "1"]
    return observations[0]


@pytest.mark.parametrize("module", REPORT_BUILDERS)
def test_report_builder_import_pins_polars_to_one_thread(module: str) -> None:
    assert _polars_threads_at_first_import(module, threads=None) == "1"


@pytest.mark.parametrize("module", REPORT_BUILDERS)
def test_report_builder_import_preserves_explicit_polars_thread_setting(module: str) -> None:
    assert _polars_threads_at_first_import(module, threads="2") == "2"


def test_paired_audit_rerun_writes_byte_identical_csvs(tmp_path: Path) -> None:
    resid_dir = tmp_path / "residuals"
    residual_path = resid_dir / "11-lstm" / "h3" / "lstm_residuals_h3.csv"
    residual_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "corridor": ["E2", "E2"],
            "direction": [-1, 1],
            "horizon": [3, 3],
            "y_true": [10.0, 20.0],
            "y_pred_dl": [11.0, 18.0],
            "y_pred_persist": [13.0, 17.0],
        }
    ).write_csv(residual_path)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result_columns = ["corridor", "direction", "baseline", "metric", "value", "horizon"]
    pl.DataFrame(
        [
            ("E2", "aggregate", "LSTM", "MAE", 1.5, 3),
            ("E2", "aggregate", "LSTM", "RMSE", 1.5811388300841898, 3),
        ],
        schema=result_columns,
        orient="row",
    ).write_csv(results_dir / "lstm_results_h3.csv")
    pl.DataFrame(
        [
            ("E2", "aggregate", "B1", "MAE", 3.0, 3),
            ("E2", "aggregate", "B1", "RMSE", 3.0, 3),
        ],
        schema=result_columns,
        orient="row",
    ).write_csv(results_dir / "baselines_results_multih.csv")

    paired_path, audit_path = build(resid_dir, results_dir)
    first_paired = paired_path.read_bytes()
    first_audit = audit_path.read_bytes()
    build(resid_dir, results_dir)

    assert first_paired
    assert first_audit
    assert paired_path.read_bytes() == first_paired
    assert audit_path.read_bytes() == first_audit
