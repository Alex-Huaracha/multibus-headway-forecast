"""Tests for src/baselines/paired_export.py — keyed per-sample B5_XGB export.

The defect this module fixes is a NON-UNIQUE KEY: `harness.XGB_RESIDUAL_COLUMNS`
drops `pair_rank`, so `t` is treated as a join key even though the headways frame
is keyed on `(t, direction, pair_rank)`. These tests pin:

  AC-PX-1: the export schema is exactly XGB_PAIRED_COLUMNS (pair_rank + empresaid
           present, extra source columns dropped).
  AC-PX-2: only TEST rows survive, and only where the target AND both predictions
           are non-null — the same semantics as harness._build_xgb_residuals.
  AC-PX-3: (corridor, direction, horizon, t, pair_rank) is unique even when two
           pair_ranks share one (t, direction); dropping pair_rank collapses them.
  AC-PX-4: direction labels are byte-equal to harness._direction_label ("-1"/"+1"),
           so the join keys stay string-compatible with the DL residual exports.
  AC-PX-5: output is deterministically sorted by the full key.
  AC-PX-6: a frame without pair_rank fails loudly instead of exporting a
           non-unique key.
  AC-PX-7: end-to-end via run_corridor — the export is the SAME paired sample set
           as harness's own residual export, and the train-only p99 winsorization
           contract still applies to the exported test targets.
"""
from __future__ import annotations

import sys
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from src.baselines.harness import _direction_label, run_corridor
from src.baselines.paired_export import (
    XGB_PAIRED_COLUMNS,
    XGB_PAIRED_KEY,
    export_paired_xgb,
    paired_xgb_from_run,
    paired_xgb_test_frame,
    search_provenance_row,
)
from src.evaluation.splits import WINSOR_QUANTILE
from tests.fixtures.headways_factory import make_headways_fixture


# ---------------------------------------------------------------------------
# Synthetic stand-in for B5FitResult.predictions
# ---------------------------------------------------------------------------

def _row(
    *,
    t: datetime,
    direction: int,
    pair_rank: int,
    delta_t_min: float | None,
    split: str,
    y_pred_b1: float | None,
    y_pred_b5_xgb: float | None,
) -> dict:
    return {
        "empresaid": 2,
        "t": t,
        "direction": direction,
        "pair_rank": pair_rank,
        "delta_t_min": delta_t_min,
        "split": split,
        # An extra baseline column that must NOT reach the export.
        "y_pred_b0": 1.0,
        "y_pred_b1": y_pred_b1,
        "y_pred_b5_xgb": y_pred_b5_xgb,
    }


# Two test rows share (t, direction) and differ only in pair_rank — the exact
# situation that makes `t` unusable as a key.
_T_SHARED = datetime(2024, 2, 10, 8, 0, 0)


def _make_predictions_frame() -> pl.DataFrame:
    """A small `B5FitResult.predictions`-shaped frame, deliberately unsorted."""
    rows = [
        # Second slot of the shared timestamp listed FIRST so the sort is tested.
        _row(t=_T_SHARED, direction=-1, pair_rank=1, delta_t_min=6.0,
             split="test", y_pred_b1=5.0, y_pred_b5_xgb=5.5),
        _row(t=_T_SHARED, direction=-1, pair_rank=0, delta_t_min=5.0,
             split="test", y_pred_b1=4.0, y_pred_b5_xgb=4.5),
        # Positive direction → label "+1".
        _row(t=datetime(2024, 2, 10, 8, 1, 0), direction=1, pair_rank=0,
             delta_t_min=7.0, split="test", y_pred_b1=6.0, y_pred_b5_xgb=6.5),
        # Non-test splits must be excluded.
        _row(t=datetime(2023, 11, 1, 8, 0, 0), direction=-1, pair_rank=0,
             delta_t_min=3.0, split="train", y_pred_b1=2.0, y_pred_b5_xgb=2.5),
        _row(t=datetime(2024, 1, 20, 8, 0, 0), direction=-1, pair_rank=0,
             delta_t_min=3.0, split="val", y_pred_b1=2.0, y_pred_b5_xgb=2.5),
        # One null per column → each row must be dropped.
        _row(t=datetime(2024, 2, 11, 8, 0, 0), direction=-1, pair_rank=0,
             delta_t_min=None, split="test", y_pred_b1=1.0, y_pred_b5_xgb=1.5),
        _row(t=datetime(2024, 2, 11, 8, 1, 0), direction=-1, pair_rank=0,
             delta_t_min=8.0, split="test", y_pred_b1=1.0, y_pred_b5_xgb=None),
        _row(t=datetime(2024, 2, 11, 8, 2, 0), direction=-1, pair_rank=0,
             delta_t_min=9.0, split="test", y_pred_b1=None, y_pred_b5_xgb=1.5),
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("direction").cast(pl.Int64),
        pl.col("pair_rank").cast(pl.Int32),
        pl.col("delta_t_min").cast(pl.Float64),
        pl.col("split").cast(pl.Utf8),
        pl.col("y_pred_b1").cast(pl.Float64),
        pl.col("y_pred_b5_xgb").cast(pl.Float64),
    )


class TestPairedExportColumnContract:
    """AC-PX-1 / AC-PX-4: schema, dtypes and direction-label convention."""

    def test_columns_are_exactly_the_declared_contract(self):
        out = paired_xgb_test_frame(_make_predictions_frame(), "E2", horizon=3)

        assert out.columns == XGB_PAIRED_COLUMNS, f"Unexpected columns: {out.columns}"
        assert "pair_rank" in out.columns, (
            "pair_rank is the whole point of this export — without it the key "
            "collapses to ~4.2 rows per (t, direction)"
        )
        assert "empresaid" in out.columns
        assert "y_pred_b0" not in out.columns, "extra source columns must be dropped"

    def test_dtypes_and_literal_columns(self):
        out = paired_xgb_test_frame(_make_predictions_frame(), "E2", horizon=3)

        assert out.schema["corridor"] == pl.Utf8
        assert out.schema["empresaid"] == pl.Int64
        assert out.schema["direction"] == pl.Utf8
        assert out.schema["horizon"] == pl.Int64
        assert out.schema["pair_rank"] == pl.Int32
        assert out.schema["y_true"] == pl.Float64
        assert out.schema["y_pred_xgb"] == pl.Float64
        assert out.schema["y_pred_persist"] == pl.Float64

        assert out["corridor"].unique().to_list() == ["E2"]
        assert out["horizon"].unique().to_list() == [3]

    def test_direction_labels_match_harness_convention(self):
        """AC-PX-4: the vectorised label must equal harness._direction_label."""
        out = paired_xgb_test_frame(_make_predictions_frame(), "E2", horizon=1)

        assert set(out["direction"].to_list()) == {"-1", "+1"}
        for direction_val in (-1, 1):
            source = _make_predictions_frame().filter(
                (pl.col("split") == "test") & (pl.col("direction") == direction_val)
            )
            expected = _direction_label(direction_val)
            got = paired_xgb_test_frame(source, "E2", horizon=1)["direction"].unique()
            assert got.to_list() == [expected], (
                f"direction {direction_val} exported as {got.to_list()} but "
                f"harness._direction_label says {expected!r}"
            )


class TestPairedExportFiltering:
    """AC-PX-2: test-split-only, and only fully paired samples."""

    def test_only_fully_paired_test_rows_survive(self):
        out = paired_xgb_test_frame(_make_predictions_frame(), "E2", horizon=1)

        assert out.height == 3, (
            "expected the 3 test rows with target and both predictions present; "
            f"got {out.height} rows: {out.to_dicts()}"
        )
        assert out["y_true"].null_count() == 0
        assert out["y_pred_xgb"].null_count() == 0
        assert out["y_pred_persist"].null_count() == 0

    def test_non_test_splits_are_excluded(self):
        source = _make_predictions_frame()
        out = paired_xgb_test_frame(source, "E2", horizon=1)

        train_val_timestamps = set(
            source.filter(pl.col("split") != "test")["t"].to_list()
        )
        assert not train_val_timestamps & set(out["t"].to_list()), (
            "train/val rows leaked into the TEST export"
        )

    def test_empty_input_yields_empty_frame_with_contract_columns(self):
        empty = _make_predictions_frame().head(0)
        out = paired_xgb_test_frame(empty, "E2", horizon=1)

        assert out.height == 0
        assert out.columns == XGB_PAIRED_COLUMNS


class TestPairedExportKey:
    """AC-PX-3 / AC-PX-5: key uniqueness and deterministic ordering."""

    def test_full_key_is_unique_and_pair_rank_is_what_makes_it_unique(self):
        out = paired_xgb_test_frame(_make_predictions_frame(), "E2", horizon=5)

        assert out.select(XGB_PAIRED_KEY).n_unique() == out.height, (
            "the exported key must be unique row-for-row"
        )
        key_without_pair_rank = [c for c in XGB_PAIRED_KEY if c != "pair_rank"]
        assert out.select(key_without_pair_rank).n_unique() < out.height, (
            "fixture must contain at least two pair_ranks sharing one "
            "(t, direction) — otherwise this test proves nothing"
        )

    def test_output_is_sorted_by_the_full_key(self):
        out = paired_xgb_test_frame(_make_predictions_frame(), "E2", horizon=1)

        assert out.to_dicts() == out.sort(XGB_PAIRED_KEY).to_dicts(), (
            "export must be deterministically sorted by the full key"
        )

    def test_deterministic_across_input_row_order(self):
        source = _make_predictions_frame()
        out_a = paired_xgb_test_frame(source, "E2", horizon=1)
        out_b = paired_xgb_test_frame(source.reverse(), "E2", horizon=1)

        assert out_a.to_dicts() == out_b.to_dicts()


class TestPairedExportGuards:
    """AC-PX-6: fail loudly rather than export a silently broken key."""

    @pytest.mark.parametrize(
        "dropped", ["pair_rank", "empresaid", "split", "y_pred_b1", "y_pred_b5_xgb"]
    )
    def test_missing_required_column_raises(self, dropped: str):
        source = _make_predictions_frame().drop(dropped)
        with pytest.raises(ValueError, match="missing required"):
            paired_xgb_test_frame(source, "E2", horizon=1)

    def test_from_run_requires_a_fitted_model(self):
        headways = _make_corridor_frame()
        run = run_corridor(headways, "E2", horizon=1, include_fitted=False)
        with pytest.raises(ValueError, match="include_fitted"):
            paired_xgb_from_run(run, "E2", horizon=1)

    def test_search_provenance_requires_a_fitted_model(self):
        headways = _make_corridor_frame()
        run = run_corridor(headways, "E2", horizon=1, include_fitted=False)
        with pytest.raises(ValueError, match="include_fitted"):
            search_provenance_row(run, "E2", horizon=1)


# ---------------------------------------------------------------------------
# AC-PX-7: end-to-end through the real harness (fits xgboost)
# ---------------------------------------------------------------------------

_N_TRAIN = 20
_N_TEST = 5
# Test targets far above the train p99 so the winsorization ceiling is visible.
_TEST_OUTLIER = 999.0


def _xgboost_loadable() -> bool:
    """True when xgboost's native library can actually be used.

    `import xgboost` succeeds even when libxgboost.dylib cannot be dlopen'd (a
    macOS box without the OpenMP runtime), so the probe has to touch the C API.
    B5_XGB cannot run without it, which is an environment gap, not a defect.
    """
    try:
        import xgboost

        xgboost.DMatrix(np.zeros((1, 1), dtype=np.float64))
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


requires_xgboost = pytest.mark.skipif(
    not _xgboost_loadable(),
    reason="xgboost native library unavailable (install the OpenMP runtime)",
)


def _make_corridor_frame() -> pl.DataFrame:
    """Synthetic headways frame: 20 train + 5 test rows per slot, 2 slots.

    Test targets are extreme (999.0) so the train-only p99 clip is observable in
    the exported `y_true`.
    """
    train_dates = [date(2023, 11, 1 + i) for i in range(_N_TRAIN)]
    test_dates = [date(2024, 2, 10 + i) for i in range(_N_TEST)]

    train_vals_a = [float(v) for v in range(1, _N_TRAIN + 1)]
    train_vals_b = [float(v) for v in range(2, _N_TRAIN + 2)]
    outliers = [_TEST_OUTLIER] * _N_TEST

    return make_headways_fixture(
        empresaid=2,
        train_dates=train_dates,
        test_dates=test_dates,
        delta_values_per_slot={
            (-1, 1): train_vals_a + outliers,
            (1, 1): train_vals_b + outliers,
        },
    )


class _StubBooster:
    """Deterministic stand-in for a trained XGBoost booster.

    Predicts the mean of the training labels for every row: a constant, finite,
    non-null prediction. That is exactly what these tests need — the export's
    filtering semantics must depend only on the target and the B1 persistence
    prediction, never on which learner produced `y_pred_b5_xgb`.
    """

    best_score = 0.5
    best_iteration = 7

    def __init__(self, constant: float) -> None:
        self._constant = constant

    def predict(self, dmatrix: "_StubDMatrix") -> np.ndarray:
        return np.full(dmatrix.n_rows, self._constant, dtype=np.float64)


class _StubDMatrix:
    def __init__(self, data, label=None, missing=None) -> None:
        self.n_rows = int(np.asarray(data).shape[0])
        self.label = None if label is None else np.asarray(label, dtype=np.float64)


class _StubXGBoost:
    """Minimal `xgboost` surface used by `fitted.fit_predict_b5_xgb`."""

    DMatrix = _StubDMatrix

    @staticmethod
    def train(params, dtrain, num_boost_round=None, **kwargs) -> _StubBooster:
        return _StubBooster(float(np.mean(dtrain.label)))


@pytest.fixture
def stub_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the stub learner. `fitted` imports xgboost lazily, inside the fit."""
    monkeypatch.setitem(sys.modules, "xgboost", _StubXGBoost)


class TestPairedExportEndToEndStubbedLearner:
    """AC-PX-7 with a stubbed learner, so it runs on any machine.

    Everything under test here is composition, not learning: the temporal split,
    the train-only p99 winsorization applied to all splits, B1 persistence, the
    paired filtering and the key. Swapping the booster for a constant predictor
    keeps all of that intact while removing the native-library dependency (the
    same assertions run against real xgboost in
    :class:`TestPairedExportEndToEnd`).

    The fixture has no val rows, so `fit_predict_b5_xgb` takes the no-search
    branch; the random search itself is covered by tests/baselines/test_fitted.py.
    """

    def test_export_matches_harness_residual_sample_set(self, stub_xgboost):
        headways = _make_corridor_frame()
        run = run_corridor(headways, "E2", horizon=1)
        paired = paired_xgb_from_run(run, "E2", horizon=1)

        assert paired.height > 0
        assert paired.height == run.residuals.height, (
            "the keyed export must keep EXACTLY the samples harness keeps — "
            f"export={paired.height}, harness={run.residuals.height}"
        )
        for column in ("y_true", "y_pred_xgb", "y_pred_persist"):
            assert sorted(paired[column].to_list()) == sorted(
                run.residuals[column].to_list()
            ), f"{column} values diverge from harness's own residual export"

    def test_winsorization_ceiling_flows_through(self, stub_xgboost):
        headways = _make_corridor_frame()
        threshold = float(
            headways.filter(pl.col("t").dt.date() <= date(2024, 1, 15))[
                "delta_t_min"
            ].quantile(WINSOR_QUANTILE)
        )
        paired, _run = export_paired_xgb(headways, "E2", horizon=1)

        assert paired.height > 0
        assert paired["y_true"].max() <= threshold, (
            "exported test targets must be clipped at the TRAIN p99 threshold "
            f"({threshold}); got max={paired['y_true'].max()}"
        )
        assert paired["y_true"].max() < _TEST_OUTLIER, (
            "raw test outlier survived — the winsorization contract is broken"
        )

    def test_key_is_unique_and_schema_is_the_contract(self, stub_xgboost):
        paired, run = export_paired_xgb(_make_corridor_frame(), "E59", horizon=3)

        assert paired.columns == XGB_PAIRED_COLUMNS
        assert paired.select(XGB_PAIRED_KEY).n_unique() == paired.height
        assert paired["corridor"].unique().to_list() == ["E59"]
        assert paired["horizon"].unique().to_list() == [3]
        assert paired["empresaid"].unique().to_list() == [2]
        assert search_provenance_row(run, "E59", horizon=3)["corridor"] == "E59"


@requires_xgboost
class TestPairedExportEndToEnd:
    """AC-PX-7: same paired sample set as harness, winsorization intact."""

    def test_export_matches_harness_residual_sample_set(self):
        headways = _make_corridor_frame()
        run = run_corridor(headways, "E2", horizon=1)
        paired = paired_xgb_from_run(run, "E2", horizon=1)

        assert paired.height == run.residuals.height, (
            "the keyed export must keep EXACTLY the samples harness keeps — "
            f"export={paired.height}, harness={run.residuals.height}"
        )
        for column in ("y_true", "y_pred_xgb", "y_pred_persist"):
            assert sorted(paired[column].to_list()) == sorted(
                run.residuals[column].to_list()
            ), f"{column} values diverge from harness's own residual export"

    def test_winsorization_ceiling_flows_through(self):
        """Train-only p99 threshold must clip the EXPORTED test targets."""
        headways = _make_corridor_frame()
        threshold = float(
            headways.filter(pl.col("t").dt.date() <= date(2024, 1, 15))[
                "delta_t_min"
            ].quantile(WINSOR_QUANTILE)
        )
        paired, _run = export_paired_xgb(headways, "E2", horizon=1)

        assert paired.height > 0
        assert paired["y_true"].max() <= threshold, (
            "exported test targets must be clipped at the TRAIN p99 threshold "
            f"({threshold}); got max={paired['y_true'].max()}"
        )
        assert paired["y_true"].max() < _TEST_OUTLIER, (
            "raw test outlier survived — the winsorization contract is broken"
        )

    def test_export_paired_xgb_returns_usable_provenance(self):
        headways = _make_corridor_frame()
        paired, run = export_paired_xgb(headways, "E2", horizon=3)

        assert paired.columns == XGB_PAIRED_COLUMNS
        assert paired["horizon"].unique().to_list() == [3]

        row = search_provenance_row(run, "E2", horizon=3)
        assert row["corridor"] == "E2"
        assert row["horizon"] == 3
        assert row["search_seed"] == run.fit_result.search_seed
        assert row["used_atypical_flag"] is False

    def test_key_is_unique_on_real_pipeline_output(self):
        headways = _make_corridor_frame()
        paired, _run = export_paired_xgb(headways, "E59", horizon=1)

        assert paired.select(XGB_PAIRED_KEY).n_unique() == paired.height
        assert paired["corridor"].unique().to_list() == ["E59"]
