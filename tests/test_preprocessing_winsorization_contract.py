"""Regression tests for train-p99 winsorization contract in DL preprocessing."""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta

import polars as pl
import pytest

from src.evaluation.splits import split_temporal, winsorize_train_p99


BUILDER_SPLIT_CELLS = [
    ("src.build_notebook_11", "cell-11-split"),
    ("src.build_notebook_12", "cell-12-split"),
    ("src.build_notebook_13", "cell-13-split"),
    ("src.build_notebook_17_e4_lstm", "cell-17-split"),
    ("src.build_notebook_18_e4_convlstm", "cell-18-split"),
    ("src.build_notebook_19_e4_transformer", "cell-19-split"),
]


def _generated_split_cell_source(module_name: str, cell_id: str) -> str:
    module = importlib.import_module(module_name)
    module._reset()
    module._add_split_cell()

    for cell in module._cells:
        if cell.get("id") == cell_id:
            return cell.source

    raise AssertionError(f"{cell_id} not found in generated cells for {module_name}")


@pytest.mark.parametrize(
    ("module_name", "cell_id"),
    BUILDER_SPLIT_CELLS,
    ids=[module_name.rsplit(".", maxsplit=1)[-1] for module_name, _ in BUILDER_SPLIT_CELLS],
)
def test_dl_builder_split_cell_winsorizes_full_split_frame(
    module_name: str,
    cell_id: str,
) -> None:
    """DL builders must pass the full split frame to winsorize_train_p99."""
    src = _generated_split_cell_source(module_name, cell_id)
    compile(src, cell_id, "exec")

    assert "df_split = split_temporal(hw)" in src
    assert "df_winsor, threshold = winsorize_train_p99(df_split)" in src
    assert "return df_winsor" in src

    assert "winsorize_train_p99(train_df)" not in src
    assert "non_train" not in src
    assert "pl.concat([df_winsor, non_train])" not in src


def _exante_headways_fixture() -> pl.DataFrame:
    train_start = datetime(2023, 12, 1, 0, 0, 0)
    rows = [
        {
            "empresaid": 2,
            "t": train_start + timedelta(minutes=i),
            "direction": -1,
            "pair_rank": 1,
            "delta_t_min": float(i + 1),
        }
        for i in range(100)
    ]
    rows.extend(
        [
            {
                "empresaid": 2,
                "t": datetime(2024, 1, 20, 0, 0, 0),
                "direction": -1,
                "pair_rank": 1,
                "delta_t_min": 10_000.0,
            },
            {
                "empresaid": 2,
                "t": datetime(2024, 2, 10, 0, 0, 0),
                "direction": -1,
                "pair_rank": 1,
                "delta_t_min": 9_000.0,
            },
        ]
    )

    return pl.DataFrame(rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("direction").cast(pl.Int64),
        pl.col("pair_rank").cast(pl.Int32),
        pl.col("delta_t_min").cast(pl.Float64),
    )


def test_exante_prepare_df_applies_train_p99_to_full_split_frame(monkeypatch) -> None:
    """Ex-ante volatility preprocessing must clip val/test with the train-p99 ceiling."""
    from src import build_exante_volatility

    raw = _exante_headways_fixture()
    expected, threshold = winsorize_train_p99(split_temporal(raw))

    def fake_load_parquet(empresaid: int) -> pl.DataFrame:
        assert empresaid == 2
        return raw

    monkeypatch.setattr(build_exante_volatility, "load_parquet", fake_load_parquet)

    result, _stats = build_exante_volatility.prepare_df(2)

    for split_name in ("val", "test"):
        clipped_value = result.filter(pl.col("split") == split_name)["delta_t_min"][0]
        assert clipped_value == pytest.approx(threshold, abs=1e-9)

    result_values = result.sort("t")["delta_t_min"].to_list()
    expected_values = expected.sort("t")["delta_t_min"].to_list()
    assert result_values == pytest.approx(expected_values, abs=1e-9)
