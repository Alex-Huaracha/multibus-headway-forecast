"""Integration tests for frozen ex-ante calibration in both report builders."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src import build_exante_volatility
from src import build_exante_correlation
from src.build_exante_correlation import build_csv_row, compute_exante_terciles
from src.data.normalization import NormalizationStats
from src.evaluation.exante_terciles import compute_frozen_thresholds


@pytest.mark.parametrize(
    "module_name",
    ["src.build_exante_volatility", "src.build_exante_correlation"],
)
def test_builder_import_pins_polars_to_one_thread(module_name: str) -> None:
    environment = os.environ.copy()
    environment.pop("POLARS_MAX_THREADS", None)
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}; import os; print(os.environ['POLARS_MAX_THREADS'])"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_materialize_corridor_combines_requested_train_and_val_splits(monkeypatch) -> None:
    df = pl.DataFrame(
        {
            "split": ["train", "val", "test"],
            "direction": [-1, 1, -1],
        }
    )
    seen_splits: list[set[str]] = []

    monkeypatch.setattr(
        build_exante_volatility,
        "compute_max_N",
        lambda _frame, quantile: {-1: 1, 1: 1},
    )

    def fake_materialize_direction(frame, _max_n, _horizon, _stats, _empresaid, _direction):
        seen_splits.append(set(frame["split"].to_list()))
        size = frame.height
        values = np.arange(size, dtype=float)
        return values, values + 10.0, np.ones(size, dtype=bool), np.ones(size, dtype=bool), values + 20.0

    monkeypatch.setattr(build_exante_volatility, "materialize_direction", fake_materialize_direction)

    targets, persist, ex_ante = build_exante_volatility.materialize_corridor(
        df, stats=None, empresaid=2, horizon=3, splits=("train", "val")
    )

    assert seen_splits == [{"train", "val"}, {"train", "val"}]
    assert targets.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert persist.tolist() == [10.0, 11.0, 10.0, 11.0]
    assert ex_ante.tolist() == [20.0, 21.0, 20.0, 21.0]


def _prepared_corridor_frame(
    test_multiplier: float = 1.0,
) -> tuple[pl.DataFrame, NormalizationStats]:
    """Build a small real-window fixture with deliberately distinct split sigmas."""
    rows = []
    split_values = {
        "train": np.arange(16, dtype=float),
        "val": np.arange(16, dtype=float) * 2.0,
        "test": np.arange(16, dtype=float) ** 2 * 100.0 * test_multiplier,
    }
    split_starts = {"train": 0, "val": 100, "test": 200}
    for split, values in split_values.items():
        for timestamp, value in enumerate(values):
            rows.append(
                {
                    "empresaid": 2,
                    "direction": -1,
                    "pair_rank": 0,
                    "t": split_starts[split] + timestamp,
                    "split": split,
                    "n_buses": 2,
                    "delta_t_min": value,
                    "delta_t_min_z": value,
                }
            )
    return pl.DataFrame(rows), NormalizationStats(
        means={(2, -1): 0.0, (2, 1): 0.0},
        stds={(2, -1): 1.0, (2, 1): 1.0},
    )


def test_default_test_materialization_excludes_test_rows_from_calibration() -> None:
    """D2: default test output is distinct from real train+val calibration windows."""
    df, stats = _prepared_corridor_frame()

    _, _, calibration_ex_ante = build_exante_volatility.materialize_corridor(
        df, stats, empresaid=2, horizon=3, splits=("train", "val")
    )
    _, _, default_test_ex_ante = build_exante_volatility.materialize_corridor(
        df, stats, empresaid=2, horizon=3
    )

    calibration = compute_frozen_thresholds(calibration_ex_ante)
    assert calibration.calib_n == 18
    assert len(default_test_ex_ante) == 2
    assert np.all(np.isfinite(default_test_ex_ante))
    assert calibration.p66 < float(default_test_ex_ante.min())


def test_builder_entrypoints_calibrate_from_real_train_val_materialization(tmp_path, monkeypatch) -> None:
    """EVC4: public entrypoint thresholds ignore perturbed test-only windows."""
    def run_entrypoints(
        df: pl.DataFrame, stats: NormalizationStats, name: str
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        targets, persist, _ = build_exante_volatility.materialize_corridor(
            df, stats, empresaid=2, horizon=3
        )
        residual_path = tmp_path / f"residuals-{name}.csv"
        pl.DataFrame(
            {
                "y_true": targets,
                "y_pred_persist": persist,
                "y_pred_dl": targets,
            }
        ).write_csv(residual_path)

        monkeypatch.setattr(
            build_exante_volatility, "prepare_df", lambda _empresaid: (df, stats)
        )
        volatility_rows = build_exante_volatility.run_corridor(
            "E2", 2, [3], lambda _horizon: residual_path
        )

        captured_thresholds = []
        real_build_csv_row = build_exante_correlation.build_csv_row

        def capture_calibrated_row(*args, **kwargs):
            captured_thresholds.append(args[4])
            return real_build_csv_row(*args, **kwargs)

        monkeypatch.setattr(
            build_exante_correlation, "prepare_df", lambda _empresaid: (df, stats)
        )
        monkeypatch.setattr(
            build_exante_correlation, "build_csv_row", capture_calibrated_row
        )
        monkeypatch.setattr(build_exante_correlation, "CORRIDORS", [("E2", 2)])
        monkeypatch.setattr(build_exante_correlation, "HORIZONS", [3])
        monkeypatch.setattr(build_exante_correlation, "OUT_DIR", tmp_path)
        build_exante_correlation.main()

        volatility_thresholds = {
            (row["p33_threshold"], row["p66_threshold"]) for row in volatility_rows
        }
        assert len(volatility_thresholds) == 1
        assert len(captured_thresholds) == 1
        return volatility_thresholds.pop(), (
            captured_thresholds[0].p33,
            captured_thresholds[0].p66,
        )

    df, stats = _prepared_corridor_frame()
    perturbed_df, perturbed_stats = _prepared_corridor_frame(test_multiplier=1_000_000.0)
    _, _, expected_calibration = build_exante_volatility.materialize_corridor(
        df, stats, empresaid=2, horizon=3, splits=("train", "val")
    )
    _, _, baseline_test_ex_ante = build_exante_volatility.materialize_corridor(
        df, stats, empresaid=2, horizon=3
    )
    _, _, perturbed_test_ex_ante = build_exante_volatility.materialize_corridor(
        perturbed_df, perturbed_stats, empresaid=2, horizon=3
    )
    expected_thresholds = compute_frozen_thresholds(expected_calibration)
    expected_pair = (expected_thresholds.p33, expected_thresholds.p66)

    assert float(perturbed_test_ex_ante.min()) > float(baseline_test_ex_ante.max())
    with pytest.raises(ValueError, match="at least three finite calibration values"):
        compute_frozen_thresholds(perturbed_test_ex_ante)

    baseline_volatility, baseline_correlation = run_entrypoints(df, stats, "baseline")
    perturbed_volatility, perturbed_correlation = run_entrypoints(
        perturbed_df, perturbed_stats, "perturbed"
    )

    assert baseline_volatility == expected_pair
    assert baseline_correlation == expected_pair
    assert perturbed_volatility == baseline_volatility
    assert perturbed_correlation == baseline_correlation


def test_volatility_stratification_emits_frozen_calibration_metadata() -> None:
    thresholds = compute_frozen_thresholds(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    rows = build_exante_volatility.compute_stratification(
        "E2",
        3,
        np.array([1.0, 2.0, 3.0]),
        np.array([1.5, 2.5, 3.5]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 3.0, 6.0]),
        thresholds,
    )

    assert {row["tercile"] for row in rows} == {"low", "mid", "high"}
    assert {(row["p33_threshold"], row["p66_threshold"]) for row in rows} == {
        (thresholds.p33, thresholds.p66)
    }
    assert {row["calib_split"] for row in rows} == {"train+val"}
    assert {row["calib_n"] for row in rows} == {6}


def test_builders_classify_fixture_with_identical_frozen_thresholds() -> None:
    calibration = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    test_ex_ante = np.array([1.0, 3.0, 6.0, 1_000.0])
    thresholds = compute_frozen_thresholds(calibration)

    volatility_rows = build_exante_volatility.compute_stratification(
        "E2",
        3,
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.5, 2.5, 3.5, 4.5]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        test_ex_ante,
        thresholds,
    )
    correlation_row = build_csv_row(
        "E2", 3, test_ex_ante, np.array([1.0, 2.0, 3.0, 4.0]), thresholds
    )
    correlation_codes = compute_exante_terciles(test_ex_ante, thresholds)

    assert {(row["p33_threshold"], row["p66_threshold"]) for row in volatility_rows} == {
        (thresholds.p33, thresholds.p66)
    }
    assert correlation_codes.tolist() == [0, 1, 2, 2]
    assert correlation_row["n"] == 4
