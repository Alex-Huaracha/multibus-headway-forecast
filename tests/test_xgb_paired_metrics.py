"""Contracts for the paired DL-vs-XGBoost comparison (builder + emitted CSVs).

Three groups of tests:

* *naming* — the emitted filenames must stay invisible to the globs the other
  report pipelines run over ``docs/resultados/csv-multihorizon/``;
* *unit* — the gates and the population algebra, on synthetic frames;
* *golden* — the committed CSVs must still carry the 12 verified restricted
  deltas, so a silent regression in the join, the population choice or the input
  store cannot pass.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl
import pytest

from src.evaluation.xgb_paired import (
    AUDIT_COLUMNS,
    PAIRED_METRIC_COLUMNS,
    SIGNIFICANCE_COLUMNS,
    audit_against_reported,
    distinct_population,
    join_xgb,
    load_xgb_export,
    residual_csv_path,
    significance_row,
    validate_inputs,
    xgb_cell,
    xgb_export_path,
    xgb_minus_dl_differential,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"

METRICS_CSV = CSV_DIR / "xgb_paired_dl_metrics.csv"
AUDIT_CSV = CSV_DIR / "xgb_paired_vs_reported_audit.csv"
SIGNIFICANCE_CSV = CSV_DIR / "xgb_paired_significance.csv"
OUTPUT_NAMES = (METRICS_CSV.name, AUDIT_CSV.name, SIGNIFICANCE_CSV.name)
# The 208 MB per-sample XGB export lives INSIDE the residual tree that
# build_significance_table and paired_audit sweep recursively, so its filename is
# under the same constraint as the emitted CSVs.
EXPORT_NAME = xgb_export_path("/store").name
GLOB_CONSTRAINED_NAMES = (*OUTPUT_NAMES, EXPORT_NAME)

# Substrings that other pipelines glob on. A new CSV in csv-multihorizon whose
# name contains one of them is loaded with an incompatible schema:
#   *_results_*      -> src/build_degradation_curve.py:111 (load_results)
#   *_residuals_*    -> src/build_significance_table.py (RESIDUALS_GLOB)
#   *_multiseed_*    -> src/evaluation/multiseed.py (default pattern)
BANNED_SUBSTRINGS = ("_results_", "_residuals_h", "_multiseed_")
FOREIGN_GLOBS = (
    "*_results_*.csv",
    "*_results_h*.csv",
    "baselines*_results_multih.csv",
    "*_residuals_*.csv",
    "**/*_residuals_*.csv",
    "*_multiseed_*.csv",
)

# The verified restricted comparison: mae_dl, mae_xgb, delta (= xgb - dl, so
# POSITIVE means the LSTM wins) and the LSTM population size, per cell.
# Reproduced independently before this builder existed; if the builder disagrees,
# the builder is wrong, not this table.
GOLDEN = {
    ("E2", 1): (4.2741, 4.3100, +0.0359, 654_303),
    ("E2", 3): (4.8572, 4.8624, +0.0052, 599_117),
    ("E2", 5): (5.0245, 5.0199, -0.0046, 583_733),
    ("E2", 10): (5.1631, 5.1627, -0.0004, 566_186),
    ("E59", 1): (3.1625, 3.1685, +0.0060, 2_264_126),
    ("E59", 3): (3.7213, 3.7826, +0.0613, 2_169_833),
    ("E59", 5): (3.9300, 4.0596, +0.1297, 2_142_718),
    ("E59", 10): (4.1876, 4.4296, +0.2420, 2_088_148),
    ("E4", 1): (3.3669, 3.0916, -0.2753, 577_255),
    ("E4", 3): (4.3979, 4.1921, -0.2057, 535_061),
    ("E4", 5): (4.8319, 4.7814, -0.0506, 514_587),
    ("E4", 10): (5.3597, 5.4988, +0.1391, 487_011),
}
# The golden values are quoted to 4 decimals, so 5e-4 is the tightest honest
# tolerance — strictly sharper than the "3 decimals" the regression asks for.
GOLDEN_TOL = 5e-4

requires_metrics_csv = pytest.mark.skipif(
    not METRICS_CSV.exists(), reason=f"{METRICS_CSV.name} not built yet"
)


# ---------------------------------------------------------------------------
# Naming: the outputs must not be swept up by the other pipelines' globs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", GLOB_CONSTRAINED_NAMES)
def test_output_name_avoids_the_globbed_substrings(name: str) -> None:
    hits = [s for s in BANNED_SUBSTRINGS if s in name]
    assert not hits, (
        f"{name} contains {hits}; another report pipeline globs that substring and "
        "would load this CSV with an incompatible schema"
    )


def test_outputs_are_invisible_to_the_foreign_globs(tmp_path: Path) -> None:
    """Glob for real, against the actual filenames, in an isolated directory."""
    for name in GLOB_CONSTRAINED_NAMES:
        (tmp_path / name).write_text("corridor\nE2\n", encoding="utf-8")
    for pattern in FOREIGN_GLOBS:
        matched = sorted(p.name for p in tmp_path.glob(pattern))
        assert matched == [], f"pattern {pattern!r} picked up {matched}"


# ---------------------------------------------------------------------------
# Unit: gates and population algebra
# ---------------------------------------------------------------------------

def _dl_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "direction": ["+1", "+1", "+1"],
            "t": [1, 2, 2],
            "pair_rank": [0, 0, 0],
            "y_true": [10.0, 20.0, 20.0],
            "y_pred_dl": [11.0, 26.0, 22.0],
            "y_pred_persist": [13.0, 17.0, 17.0],
        }
    )


def _xgb_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "direction": ["+1", "+1"],
            "t": [1, 2],
            "pair_rank": [0, 0],
            "y_pred_xgb": [12.0, 23.0],
        }
    )


def test_join_preserves_the_dl_multiplicity() -> None:
    joined, coverage = join_xgb(_dl_frame(), _xgb_frame())
    assert joined.height == 3  # the duplicated target stays duplicated
    assert coverage == 100.0
    assert joined.get_column("y_pred_xgb").to_list() == [12.0, 23.0, 23.0]


def test_join_fails_when_a_dl_row_has_no_xgb_counterpart() -> None:
    partial = _xgb_frame().head(1)
    with pytest.raises(ValueError, match="have no XGB counterpart"):
        join_xgb(_dl_frame(), partial)


def test_xgb_cell_rejects_a_non_unique_key() -> None:
    export = pl.DataFrame(
        {
            "corridor": ["E2", "E2"],
            "empresaid": [2, 2],
            "direction": ["+1", "+1"],
            "horizon": [1, 1],
            "t": [1, 1],
            "pair_rank": [0, 0],
            "y_true": [10.0, 10.0],
            "y_pred_xgb": [12.0, 13.0],
            "y_pred_persist": [13.0, 13.0],
        }
    )
    with pytest.raises(ValueError, match="not per-sample unique"):
        xgb_cell(export, "E2", 1)


def test_xgb_cell_rejects_a_missing_cell() -> None:
    export = pl.DataFrame(
        {
            "corridor": ["E2"],
            "empresaid": [2],
            "direction": ["+1"],
            "horizon": [1],
            "t": [1],
            "pair_rank": [0],
            "y_true": [10.0],
            "y_pred_xgb": [12.0],
            "y_pred_persist": [13.0],
        }
    )
    with pytest.raises(ValueError, match="no XGB rows for corridor"):
        xgb_cell(export, "E2", 10)


def test_distinct_population_keeps_one_row_per_target_deterministically() -> None:
    joined, _ = join_xgb(_dl_frame(), _xgb_frame())
    first = distinct_population(joined)
    assert first.height == 2
    # Sorted by (direction, t, pair_rank), so the kept replica of t=2 is the one
    # that appeared first in temporal order: y_pred_dl == 26.0.
    assert first.get_column("t").to_list() == [1, 2]
    assert first.get_column("y_pred_dl").to_list() == [11.0, 26.0]
    # Repeated calls must agree byte for byte (maintain_order=True).
    assert distinct_population(joined).to_dicts() == first.to_dicts()


def test_differential_is_positive_when_the_dl_model_wins() -> None:
    joined, _ = join_xgb(_dl_frame(), _xgb_frame())
    d = xgb_minus_dl_differential(joined, "MAE")
    # row 0: |10-12| - |10-11| = +1 (DL wins)
    # row 1: |20-23| - |20-26| = -3 (XGB wins — the DL's worse replica of t=2)
    # row 2: |20-23| - |20-22| = +1 (DL wins — its better replica of the SAME target,
    #        which is exactly why the distinct-target population needs a defined
    #        replica rule instead of an accidental one)
    assert d.tolist() == [1.0, -3.0, 1.0]


def test_significance_row_records_a_one_sided_wilcoxon_and_its_direction() -> None:
    joined, _ = join_xgb(_dl_frame(), _xgb_frame())
    row = significance_row("E2", 1, distinct_population(joined))
    assert set(row) == set(SIGNIFICANCE_COLUMNS)
    assert row["wilcoxon_alternative"] == "greater"
    assert row["population"] == "distinct_target"
    assert 0.0 <= row["wilcoxon_p_one_sided"] <= 1.0
    assert 0.0 <= row["win_rate"] <= 1.0
    # A directional claim needs the one-sided p-value; the two-sided one is kept
    # only so the pair can be compared, never quoted alone for the sign.
    assert "wilcoxon_p_two_sided" in row


# ---------------------------------------------------------------------------
# Fail fast on missing per-sample inputs
# ---------------------------------------------------------------------------

def test_validate_inputs_names_every_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no per-sample inputs found"):
        validate_inputs(tmp_path)


def test_load_xgb_export_fails_fast_when_absent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no XGB paired export found in"):
        load_xgb_export(xgb_export_path(tmp_path))


def test_residual_path_layout() -> None:
    root = Path("/store")
    assert residual_csv_path(root, "E2", 3).name == "lstm_residuals_h3.csv"
    assert residual_csv_path(root, "E59", 3).name == "lstm_residuals_h3.csv"
    assert residual_csv_path(root, "E4", 3).name == "lstm_E4_residuals_h3.csv"
    assert residual_csv_path(root, "E2", 3).parent.name == "h3"
    with pytest.raises(ValueError, match="unknown corridor"):
        residual_csv_path(root, "E58", 3)


# ---------------------------------------------------------------------------
# Determinism of the reconciliation layer (the part testable without parquet)
# ---------------------------------------------------------------------------

def _synthetic_paired() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "corridor": ["E2"],
            "horizon": [1],
            "n_dl_rows": [10],
            "n_xgb_matched": [10],
            "coverage_pct": [100.0],
            "n_distinct_targets": [4],
            "align_max_abs_diff_target": [1e-6],
            "align_max_abs_diff_persist": [1e-6],
            "align_tol": [1e-2],
            "mae_persist_matched": [6.0],
            "mae_dl_matched": [4.0],
            "mae_xgb_matched": [5.0],
            "delta_xgb_minus_dl_matched": [1.0],
            "dl_better_matched": [True],
            "mae_dl_distinct": [4.1],
            "mae_xgb_distinct": [5.1],
            "delta_xgb_minus_dl_distinct": [1.0],
            "dl_better_distinct": [True],
        }
    ).select(PAIRED_METRIC_COLUMNS)


def _write_reported(results_dir: Path, xgb_mae: float, dl_mae: float) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    columns = ["corridor", "direction", "baseline", "metric", "value", "horizon"]
    pl.DataFrame(
        [("E2", "aggregate", "B5_XGB", "MAE", xgb_mae, 1)],
        schema=columns,
        orient="row",
    ).write_csv(results_dir / "baselines_results_multih.csv")
    pl.DataFrame(
        [("E4", "aggregate", "B5_XGB", "MAE", xgb_mae, 1)],
        schema=columns,
        orient="row",
    ).write_csv(results_dir / "baselines_E4_results_multih.csv")
    for horizon in (1, 3, 5, 10):
        pl.DataFrame(
            [("E2", "aggregate", "LSTM", "MAE", dl_mae, horizon)],
            schema=columns,
            orient="row",
        ).write_csv(results_dir / f"lstm_results_h{horizon}.csv")
        pl.DataFrame(
            [("E4", "aggregate", "LSTM", "MAE", dl_mae, horizon)],
            schema=columns,
            orient="row",
        ).write_csv(results_dir / f"lstm_E4_results_h{horizon}.csv")


def test_audit_rerun_is_byte_identical(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_reported(results_dir, xgb_mae=3.5, dl_mae=4.0)
    paired = _synthetic_paired()

    first = audit_against_reported(paired, results_dir)
    out = tmp_path / "audit.csv"
    first.write_csv(out)
    first_bytes = out.read_bytes()
    audit_against_reported(paired, results_dir).write_csv(out)

    assert first_bytes
    assert out.read_bytes() == first_bytes
    assert first.columns == AUDIT_COLUMNS


def test_audit_flags_the_verdict_flip_caused_by_the_framing_bias(tmp_path: Path) -> None:
    """Restricted says the LSTM wins; the aggregate XGB MAE says the opposite."""
    results_dir = tmp_path / "results"
    _write_reported(results_dir, xgb_mae=3.5, dl_mae=4.0)
    audit = audit_against_reported(_synthetic_paired(), results_dir)
    row = audit.to_dicts()[0]
    assert row["restricted_dl_better"] is True
    assert row["reported_dl_better"] is False
    assert row["sign_flip"] is True
    assert row["abs_diff"] == pytest.approx(1.5)


def test_audit_reports_no_flip_when_both_framings_agree(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_reported(results_dir, xgb_mae=5.2, dl_mae=4.0)
    row = audit_against_reported(_synthetic_paired(), results_dir).to_dicts()[0]
    assert row["reported_dl_better"] is True
    assert row["sign_flip"] is False


def test_audit_rejects_a_duplicated_reported_row(tmp_path: Path) -> None:
    """Two reported rows per cell would fan the join out and double the audit."""
    results_dir = tmp_path / "results"
    _write_reported(results_dir, xgb_mae=5.2, dl_mae=4.0)
    columns = ["corridor", "direction", "baseline", "metric", "value", "horizon"]
    pl.DataFrame(
        [
            ("E2", "aggregate", "B5_XGB", "MAE", 5.2, 1),
            ("E2", "aggregate", "B5_XGB", "MAE", 5.3, 1),
        ],
        schema=columns,
        orient="row",
    ).write_csv(results_dir / "baselines_results_multih.csv")
    with pytest.raises(ValueError, match="duplicate reported B5_XGB keys"):
        audit_against_reported(_synthetic_paired(), results_dir)


def test_audit_rejects_a_missing_reported_cell(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_reported(results_dir, xgb_mae=5.2, dl_mae=4.0)
    columns = ["corridor", "direction", "baseline", "metric", "value", "horizon"]
    pl.DataFrame(
        [("E2", "aggregate", "B5_XGB", "MAE", 5.2, 3)],  # h=1 is what paired needs
        schema=columns,
        orient="row",
    ).write_csv(results_dir / "baselines_results_multih.csv")
    with pytest.raises(ValueError, match="missing reported B5_XGB MAE"):
        audit_against_reported(_synthetic_paired(), results_dir)


# ---------------------------------------------------------------------------
# Golden: the committed CSVs
# ---------------------------------------------------------------------------

@requires_metrics_csv
def test_metrics_csv_has_twelve_cells_and_the_declared_schema() -> None:
    table = pl.read_csv(METRICS_CSV)
    assert table.columns == PAIRED_METRIC_COLUMNS
    assert table.height == 12
    cells = set(zip(table.get_column("corridor"), table.get_column("horizon")))
    assert cells == set(GOLDEN)


@requires_metrics_csv
def test_metrics_csv_reproduces_the_verified_restricted_deltas() -> None:
    rows = {
        (r["corridor"], r["horizon"]): r
        for r in pl.read_csv(METRICS_CSV).iter_rows(named=True)
    }
    for cell, (mae_dl, mae_xgb, delta, n_rows) in GOLDEN.items():
        row = rows[cell]
        assert row["n_dl_rows"] == n_rows, f"{cell}: LSTM population size changed"
        assert row["mae_dl_matched"] == pytest.approx(mae_dl, abs=GOLDEN_TOL), cell
        assert row["mae_xgb_matched"] == pytest.approx(mae_xgb, abs=GOLDEN_TOL), cell
        assert row["delta_xgb_minus_dl_matched"] == pytest.approx(
            delta, abs=GOLDEN_TOL
        ), cell
        assert row["dl_better_matched"] == (delta > 0), cell


@requires_metrics_csv
def test_metrics_csv_records_that_every_gate_passed() -> None:
    for row in pl.read_csv(METRICS_CSV).iter_rows(named=True):
        cell = (row["corridor"], row["horizon"])
        assert row["coverage_pct"] == pytest.approx(100.0, abs=1e-9), cell
        assert row["n_xgb_matched"] == row["n_dl_rows"], cell
        assert row["align_max_abs_diff_target"] < row["align_tol"], cell
        assert row["align_max_abs_diff_persist"] < row["align_tol"], cell
        assert 0 < row["n_distinct_targets"] <= row["n_dl_rows"], cell


@requires_metrics_csv
def test_the_two_populations_are_actually_different() -> None:
    """Guard the documented reason both populations exist: replication is real."""
    table = pl.read_csv(METRICS_CSV)
    ratios = (
        table.get_column("n_dl_rows") / table.get_column("n_distinct_targets")
    ).to_list()
    assert all(ratio > 3.0 for ratio in ratios), (
        f"expected ~4.5x window replication in the DL population, got {ratios}"
    )


@requires_metrics_csv
def test_audit_csv_matches_the_metrics_csv() -> None:
    audit = pl.read_csv(AUDIT_CSV)
    assert audit.columns == AUDIT_COLUMNS
    assert audit.height == 12
    metrics = {
        (r["corridor"], r["horizon"]): r
        for r in pl.read_csv(METRICS_CSV).iter_rows(named=True)
    }
    for row in audit.iter_rows(named=True):
        source = metrics[(row["corridor"], row["horizon"])]
        assert row["restricted_xgb"] == pytest.approx(source["mae_xgb_matched"])
        assert row["restricted_dl"] == pytest.approx(source["mae_dl_matched"])
        assert row["abs_diff"] == pytest.approx(
            abs(row["restricted_xgb"] - row["reported_xgb"])
        )
        assert row["sign_flip"] == (
            row["restricted_dl_better"] != row["reported_dl_better"]
        )


@requires_metrics_csv
def test_significance_csv_is_one_sided_and_sits_on_the_distinct_population() -> None:
    significance = pl.read_csv(SIGNIFICANCE_CSV)
    assert significance.columns == SIGNIFICANCE_COLUMNS
    assert significance.height == 12
    metrics = {
        (r["corridor"], r["horizon"]): r
        for r in pl.read_csv(METRICS_CSV).iter_rows(named=True)
    }
    for row in significance.iter_rows(named=True):
        cell = (row["corridor"], row["horizon"])
        assert row["population"] == "distinct_target", cell
        assert row["wilcoxon_alternative"] == "greater", cell
        assert row["n"] == metrics[cell]["n_distinct_targets"], cell
        assert 0.0 <= row["wilcoxon_p_one_sided"] <= 1.0, cell
        assert 0.0 <= row["dm_p_one_sided"] <= 1.0, cell
        assert 0.0 <= row["win_rate"] <= 1.0, cell
        assert row["dl_better"] == (row["delta_loss"] > 0), cell
        assert row["delta_mae"] == pytest.approx(
            metrics[cell]["delta_xgb_minus_dl_distinct"], abs=1e-9
        ), cell
