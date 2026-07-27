"""Dataset backed by the canonical sample index — the retrained pipeline's loader.

``HeadwayDataset`` (``src/data/dataset.py``) anchors every window inside a
``(empresaid, direction, pair_rank)`` slot and slices it by row position. Two
consequences, both audit findings:

  * the same snapshot target is emitted once per anchoring slot, so the reported
    MAE is weighted by fleet density (#13);
  * row positions are not checked to be consecutive minutes, so the nominal
    horizon is a row offset rather than a time offset (§3).

This loader consumes ``sample_index.make_sample_index`` instead. Anchors are
timestamps, each target appears exactly once, and contiguity is guaranteed by
construction — so the window timestamps can be derived arithmetically rather
than read off a slot frame.

``HeadwayDataset`` is left in place and untouched: notebooks 12/13/18/19 must
keep reproducing the frozen architecture comparison, whose validity rests on all
three architectures sharing the same flaw.

Context features
----------------
``CAUSAL_CONTEXT_FEATURE_NAMES`` drops ``atypical_flag`` from the five-column
set. The flag is a whole-day aggregate whose threshold was fitted over all 152
days including test, so classifying a day requires that day's full record count
— information unavailable at 08:00 on the day being predicted. That is leakage
by design, not by parametrization, so the feature is removed rather than
recalibrated (see ``docs/plan-reentrenamiento.md`` §2, C3).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

# Cyclical time only. `atypical_flag` is deliberately absent — see module docstring.
CAUSAL_CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

_SERIES_COLS: list[str] = ["empresaid", "direction"]


class _SeriesGrid:
    """Dense (n_timesteps, max_N) view of one ``(empresaid, direction)`` series.

    Built once per series and reused by every sample that anchors into it. This
    replaces the per-timestep ``filter`` the legacy loader performed inside
    ``__getitem__``, which cost one full scan per timestep of every window.

    The grid holds only distinct snapshots, so its footprint is
    ``n_timesteps x max_N`` — independent of how many windows anchor into it.
    """

    __slots__ = ("ts_index", "ts_values", "values", "masks", "context")

    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        max_N: int,
        value_col: str,
        context_cols: list[str],
    ) -> None:
        timestamps = (
            frame.select("t").unique().sort("t").get_column("t").to_numpy()
        )
        self.ts_index: dict[np.datetime64, int] = {
            ts: i for i, ts in enumerate(timestamps)
        }
        # Positional view of the same axis, so a target row can be verified
        # against the timestamp the sample index recorded.
        self.ts_values = timestamps
        n = timestamps.size

        self.values = np.zeros((n, max_N), dtype=np.float32)
        self.masks = np.zeros((n, max_N), dtype=bool)
        self.context = np.zeros((n, len(context_cols)), dtype=np.float32)

        rows = frame.select(["t", "pair_rank", value_col] + context_cols)
        ts_col = rows.get_column("t").to_numpy()
        pr_col = rows.get_column("pair_rank").to_numpy()
        val_col = np.asarray(rows.get_column(value_col).to_numpy(), dtype=np.float64)

        row_of = np.array([self.ts_index[ts] for ts in ts_col], dtype=np.int64)

        # Values and masks: only in-range pair_ranks with an observed value.
        # Out-of-range slots are truncated at max_N (AC-MAXN-2); missing values
        # keep value 0.0 and mask False, the legacy null convention (AC-MASK-3).
        #
        # Missingness MUST be tested with np.isnan, not polars' is_null. Once a
        # column is converted to numpy its nulls become NaN and the null flag is
        # gone, so `pl.Series(arr).is_null()` answers False for every one of them
        # — polars treats NaN and null as different things. Getting this wrong
        # writes NaN into the tensor with mask=True, which makes the loss NaN,
        # leaves every epoch un-improved, and yields an empty state_dict.
        in_range = (pr_col >= 0) & (pr_col < max_N)
        present = in_range & ~np.isnan(val_col)
        self.values[row_of[present], pr_col[present]] = val_col[present].astype(np.float32)
        self.masks[row_of[present], pr_col[present]] = True

        # Context is per snapshot, identical across pair_ranks — last write wins.
        for c_idx, name in enumerate(context_cols):
            col = rows.get_column(name).fill_null(0.0).to_numpy()
            self.context[row_of, c_idx] = col.astype(np.float32)


def materialize_arrays(
    df: pl.DataFrame,
    sample_index: pl.DataFrame,
    *,
    max_N: int,
    T_in: int,
    horizon: int,
    value_col: str = "delta_t_min_z",
    context_cols: tuple[str, ...] = CAUSAL_CONTEXT_FEATURE_NAMES,
) -> dict[str, np.ndarray]:
    """Dense arrays for every sample, in sample-index row order.

    The Kaggle notebooks materialize the whole split up front rather than
    streaming through a DataLoader, so this is the path they take. It lives here
    — beside the Dataset and covered by the same tests — instead of being
    reimplemented inside a notebook cell, because an untested copy inside a
    generated artifact is how the pipeline drifted from its contracts before.

    Returns
    -------
    dict with ``input`` (n, T_in, max_N), ``target`` (n, 1, max_N),
    ``input_mask``, ``target_mask``, ``context`` (n, T_in, n_ctx) — the same key
    names ``collate_fn`` and ``train.py`` already consume.

    Raises
    ------
    ValueError
        When a sample's target does not land on the timestamp the index
        declared, i.e. contract C2 is violated for that row.
    """
    if "atypical_flag" in context_cols:
        raise ValueError(
            "atypical_flag is a leaking feature and must not be materialized "
            "in the retrained pipeline (plan-reentrenamiento.md C3)"
        )

    n = sample_index.height
    n_ctx = len(context_cols)
    out = {
        "input": np.zeros((n, T_in, max_N), dtype=np.float32),
        "target": np.zeros((n, 1, max_N), dtype=np.float32),
        "input_mask": np.zeros((n, T_in, max_N), dtype=bool),
        "target_mask": np.zeros((n, 1, max_N), dtype=bool),
        "context": np.zeros((n, T_in, n_ctx), dtype=np.float32),
    }
    if n == 0:
        return out

    emp_col = sample_index.get_column("empresaid").to_numpy()
    dir_col = sample_index.get_column("direction").to_numpy()
    start_col = sample_index.get_column("start_ts").to_numpy()
    target_col = sample_index.get_column("target_ts").to_numpy()

    order = np.arange(n)
    for (empresaid, direction) in sorted({(int(e), int(d)) for e, d in zip(emp_col, dir_col)}):
        rows = order[(emp_col == empresaid) & (dir_col == direction)]
        if rows.size == 0:
            continue
        series = df.filter(
            (pl.col("empresaid") == empresaid) & (pl.col("direction") == direction)
        )
        grid = _SeriesGrid(
            series,
            max_N=max_N,
            value_col=value_col,
            context_cols=list(context_cols),
        )
        starts = np.array([grid.ts_index[ts] for ts in start_col[rows]], dtype=np.int64)
        target_rows = starts + (T_in - 1 + horizon)

        if target_rows.max() >= grid.ts_values.size:
            raise ValueError(
                f"target row out of range for series ({empresaid}, {direction})"
            )
        declared = target_col[rows]
        if not np.array_equal(grid.ts_values[target_rows], declared):
            bad = int(np.argmax(grid.ts_values[target_rows] != declared))
            raise ValueError(
                f"contiguity violated for ({empresaid}, {direction}) "
                f"start={start_col[rows][bad]}: grid holds "
                f"{grid.ts_values[target_rows][bad]}, index declared {declared[bad]}"
            )

        # Window rows are start..start+T_in-1, contiguous by C2.
        window_rows = starts[:, None] + np.arange(T_in)[None, :]
        out["input"][rows] = grid.values[window_rows]
        out["input_mask"][rows] = grid.masks[window_rows]
        out["context"][rows] = grid.context[window_rows]
        out["target"][rows, 0] = grid.values[target_rows]
        out["target_mask"][rows, 0] = grid.masks[target_rows]

    return out


class ContiguousHeadwayDataset(Dataset):
    """Sample-index-backed dataset. One item per canonical sample.

    Tensor contract matches ``HeadwayDataset`` so ``train.py`` and ``collate_fn``
    need no changes:
        input       : (T_in, max_N)   float32
        target      : (1, max_N)      float32
        input_mask  : (T_in, max_N)   bool
        target_mask : (1, max_N)      bool
        context     : (T_in, n_ctx)   float32   — 4 columns, no atypical flag

    Mask polarity: True = VALID (INV-5).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        sample_index: pl.DataFrame,
        *,
        max_N_by_direction: dict[tuple[int, int], int],
        T_in: int,
        horizon: int,
        value_col: str = "delta_t_min_z",
        context_cols: tuple[str, ...] = CAUSAL_CONTEXT_FEATURE_NAMES,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        missing = [c for c in context_cols if c not in df.columns]
        if missing:
            raise ValueError(f"df is missing context columns: {missing}")
        if "atypical_flag" in context_cols:
            raise ValueError(
                "atypical_flag is a leaking feature and must not be a context "
                "column in the retrained pipeline (plan-reentrenamiento.md C3)"
            )

        self._df = df
        self._index = sample_index
        self._max_N_by_direction = max_N_by_direction
        self._T_in = T_in
        self._horizon = horizon
        self._value_col = value_col
        self._context_cols = list(context_cols)

        # Materialized lazily per series — never per window (INV-7).
        self._grids: dict[tuple[int, int], _SeriesGrid] = {}

        # Column-oriented access beats row(named=True) inside __getitem__.
        self._emp = sample_index.get_column("empresaid").to_numpy()
        self._dir = sample_index.get_column("direction").to_numpy()
        self._start = sample_index.get_column("start_ts").to_numpy()
        self._target = sample_index.get_column("target_ts").to_numpy()

    def __len__(self) -> int:
        return self._index.height

    def _grid(self, empresaid: int, direction: int) -> _SeriesGrid:
        key = (empresaid, direction)
        if key not in self._grids:
            frame = self._df.filter(
                (pl.col("empresaid") == empresaid)
                & (pl.col("direction") == direction)
            )
            self._grids[key] = _SeriesGrid(
                frame,
                max_N=self._max_N_by_direction[key],
                value_col=self._value_col,
                context_cols=self._context_cols,
            )
        return self._grids[key]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        empresaid = int(self._emp[idx])
        direction = int(self._dir[idx])
        start_ts = self._start[idx]
        target_ts = self._target[idx]

        grid = self._grid(empresaid, direction)
        i = grid.ts_index[start_ts]
        target_row = i + self._T_in - 1 + self._horizon

        # Defense in depth: C2 guarantees the run is contiguous, so the target
        # must land on exactly the timestamp the index recorded. If it does not,
        # the index and the frame disagree and the sample is silently wrong —
        # precisely the failure mode this pipeline exists to prevent.
        if target_row >= grid.ts_values.size:
            raise IndexError(
                f"target row {target_row} out of range for series "
                f"({empresaid}, {direction}) with {grid.ts_values.size} snapshots"
            )
        if grid.ts_values[target_row] != target_ts:
            raise ValueError(
                f"contiguity violated for ({empresaid}, {direction}) "
                f"start={start_ts}: target row holds "
                f"{grid.ts_values[target_row]}, index declared {target_ts}"
            )

        values = torch.from_numpy(grid.values[i : i + self._T_in].copy())
        masks = torch.from_numpy(grid.masks[i : i + self._T_in].copy())
        context = torch.from_numpy(grid.context[i : i + self._T_in].copy())
        target = torch.from_numpy(grid.values[target_row : target_row + 1].copy())
        target_mask = torch.from_numpy(grid.masks[target_row : target_row + 1].copy())

        return {
            "input": values,
            "target": target,
            "input_mask": masks,
            "target_mask": target_mask,
            "context": context,
        }
