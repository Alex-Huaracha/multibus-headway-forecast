"""Tests for paired residual metric audits."""
from __future__ import annotations

import math

import polars as pl
import pytest

from src.evaluation.paired_audit import (
    audit_against_reported,
    build_paired_metrics,
    discover_residual_files,
    paired_metrics_table,
    parse_residual_file,
)


RESIDUAL_SCHEMA = [
    "corridor",
    "direction",
    "horizon",
    "y_true",
    "y_pred_dl",
    "y_pred_persist",
]
RESULT_SCHEMA = ["corridor", "direction", "baseline", "metric", "value", "horizon"]


def _residual_frame(corridor, horizon, y_true, y_dl, y_persist):
    n = len(y_true)
    return pl.DataFrame(
        {
            "corridor": [corridor] * n,
            "direction": [-1 if i % 2 == 0 else 1 for i in range(n)],
            "horizon": [horizon] * n,
            "y_true": list(y_true),
            "y_pred_dl": list(y_dl),
            "y_pred_persist": list(y_persist),
        }
    )


def _write_results(path, rows):
    pl.DataFrame(rows, schema=RESULT_SCHEMA, orient="row").write_csv(path)


class TestResidualDiscovery:
    def test_discovers_residuals_and_parses_model_context(self, tmp_path):
        residual = tmp_path / "11-lstm" / "h3" / "lstm_residuals_h3.csv"
        residual.parent.mkdir(parents=True)
        _residual_frame("E2", 3, [10.0], [11.0], [13.0]).write_csv(residual)

        e4_residual = (
            tmp_path
            / "13-spatialtransformer"
            / "h10"
            / "spatial_transformer_E4_residuals_h10.csv"
        )
        e4_residual.parent.mkdir(parents=True)
        _residual_frame("E4", 10, [10.0], [9.0], [12.0]).write_csv(e4_residual)

        records = discover_residual_files(tmp_path)

        assert [(r.model, r.horizon) for r in records] == [
            ("LSTM", 3),
            ("SpatialTransformer", 10),
        ]
        assert parse_residual_file(e4_residual).model == "SpatialTransformer"

    def test_does_not_include_co_located_result_csvs(self, tmp_path):
        residual_dir = tmp_path / "12-spatialconvlstm" / "h3"
        residual_dir.mkdir(parents=True)
        _residual_frame("E2", 3, [10.0], [11.0], [13.0]).write_csv(
            residual_dir / "spatial_conv_lstm_residuals_h3.csv"
        )
        pl.DataFrame({"wrong": ["schema"]}).write_csv(
            residual_dir / "spatial_conv_lstm_results_h3.csv"
        )

        records = discover_residual_files(tmp_path)
        metrics = build_paired_metrics(tmp_path)

        assert [record.path.name for record in records] == [
            "spatial_conv_lstm_residuals_h3.csv"
        ]
        assert metrics.height == 1

    def test_rejects_residual_rows_with_horizon_mismatching_file_context(
        self, tmp_path
    ):
        residual = tmp_path / "11-lstm" / "h3" / "lstm_residuals_h3.csv"
        residual.parent.mkdir(parents=True)
        _residual_frame("E2", 5, [10.0], [11.0], [13.0]).write_csv(residual)

        with pytest.raises(
            ValueError,
            match="residual horizon values .* do not match parsed/discovered horizon h3",
        ):
            build_paired_metrics(tmp_path)


class TestPairedMetrics:
    def test_calculates_paired_mae_rmse_and_deltas(self):
        df = _residual_frame(
            "E2",
            3,
            y_true=[10.0, 20.0],
            y_dl=[11.0, 18.0],
            y_persist=[13.0, 17.0],
        )

        row = paired_metrics_table(df, model="LSTM").row(0, named=True)

        assert row["model"] == "LSTM"
        assert row["n"] == 2
        assert row["mae_dl"] == pytest.approx(1.5)
        assert row["rmse_dl"] == pytest.approx(math.sqrt(2.5))
        assert row["mae_persist"] == pytest.approx(3.0)
        assert row["rmse_persist"] == pytest.approx(3.0)
        assert row["delta_mae"] == pytest.approx(-1.5)
        assert row["delta_rmse"] == pytest.approx(math.sqrt(2.5) - 3.0)


class TestAuditJoin:
    def _paired(self):
        return pl.DataFrame(
            {
                "model": ["LSTM"],
                "corridor": ["E2"],
                "horizon": [3],
                "n": [2],
                "mae_dl": [1.0],
                "rmse_dl": [2.0],
                "mae_persist": [2.0],
                "rmse_persist": [4.0],
                "delta_mae": [-1.0],
                "delta_rmse": [-2.0],
            }
        )

    def test_reports_differences_and_sign_mismatches(self, tmp_path):
        paired = self._paired()
        _write_results(
            tmp_path / "lstm_results_h3.csv",
            [
                ("E2", "aggregate", "LSTM", "MAE", 3.0, 3),
                ("E2", "aggregate", "LSTM", "RMSE", 3.5, 3),
            ],
        )
        _write_results(
            tmp_path / "baselines_results_multih.csv",
            [
                ("E2", "aggregate", "B1", "MAE", 2.0, 3),
                ("E2", "aggregate", "B1", "RMSE", 4.0, 3),
            ],
        )

        audit = audit_against_reported(paired, tmp_path)
        mae = audit.filter(pl.col("metric") == "MAE").row(0, named=True)
        rmse = audit.filter(pl.col("metric") == "RMSE").row(0, named=True)

        assert mae["paired_dl"] == pytest.approx(1.0)
        assert mae["reported_dl"] == pytest.approx(3.0)
        assert mae["abs_diff_dl"] == pytest.approx(2.0)
        assert mae["abs_diff_persist"] == pytest.approx(0.0)
        assert mae["paired_dl_better"] is True
        assert mae["reported_dl_better"] is False
        assert mae["sign_mismatch"] is True

        assert rmse["reported_delta"] == pytest.approx(-0.5)
        assert rmse["sign_mismatch"] is False

    def test_rejects_missing_reported_metric_keys(self, tmp_path):
        paired = self._paired()
        _write_results(
            tmp_path / "lstm_results_h3.csv",
            [
                ("E2", "aggregate", "LSTM", "MAE", 3.0, 3),
                ("E2", "aggregate", "LSTM", "RMSE", 3.5, 3),
            ],
        )
        _write_results(
            tmp_path / "baselines_results_multih.csv",
            [("E2", "aggregate", "B1", "MAE", 2.0, 3)],
        )

        with pytest.raises(
            ValueError, match="missing reported persistence metric keys"
        ):
            audit_against_reported(paired, tmp_path)

    def test_rejects_duplicate_reported_metric_keys(self, tmp_path):
        paired = self._paired()
        _write_results(
            tmp_path / "lstm_results_h3.csv",
            [
                ("E2", "aggregate", "LSTM", "MAE", 3.0, 3),
                ("E2", "aggregate", "LSTM", "MAE", 3.1, 3),
                ("E2", "aggregate", "LSTM", "RMSE", 3.5, 3),
            ],
        )
        _write_results(
            tmp_path / "baselines_results_multih.csv",
            [
                ("E2", "aggregate", "B1", "MAE", 2.0, 3),
                ("E2", "aggregate", "B1", "RMSE", 4.0, 3),
            ],
        )

        with pytest.raises(ValueError, match="duplicate DL metric keys"):
            audit_against_reported(paired, tmp_path)
