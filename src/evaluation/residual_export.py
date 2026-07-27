"""Canonical per-sample residual export — the full-key contract.

Every question this project could not answer from disk traces to an export that
threw the key away:

  * ``harness.py`` exported XGBoost residuals keyed on ``t`` alone, which is not
    unique (~4.49 rows per ``(t, direction)``). Repairing the DL-vs-XGBoost
    comparison therefore needed a whole new Kaggle kernel, ``20-xgb-paired-export``
    (audit §2.1).
  * The DL residual CSVs carry only
    ``corridor, direction, horizon, y_true, y_pred_dl, y_pred_persist`` — no
    ``t``, no ``pair_rank`` — which is why clustering by service day (#6) and a
    per-position error profile (#5) cannot be computed locally at all.

Both are the same mistake: a lossy export turns every new question into another
GPU run. This module fixes it once, for every model family.

The key
-------
``(corridor, direction, horizon, split, start_ts, target_ts, pair_rank)`` is
unique by construction: the sample index guarantees one row per
``(empresaid, direction, start_ts, horizon)`` (contract C1) and ``pair_rank``
indexes position within that sample's predicted vector.

Reconstruction
--------------
Model outputs arrive as dense ``(n_samples, max_N)`` arrays whose row order is
the sample index's row order and whose column order is ``pair_rank``. The key is
therefore recoverable exactly: repeat each index row ``max_N`` times, tile
``pair_rank`` across it, then drop masked-out cells. No join, no ambiguity.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# Canonical column order. Key first, then values — so a `head` on the CSV shows
# what identifies a row before what it measured.
RESIDUAL_KEY_COLUMNS: list[str] = [
    "corridor",
    "direction",
    "horizon",
    "split",
    "start_ts",
    "target_ts",
    "pair_rank",
]

RESIDUAL_VALUE_COLUMNS: list[str] = [
    "y_true",
    "y_pred_model",
    "y_pred_persist",
]

RESIDUAL_COLUMNS: list[str] = RESIDUAL_KEY_COLUMNS + RESIDUAL_VALUE_COLUMNS


def direction_label(direction_val: int) -> str:
    """Signed direction label ("-1" / "+1").

    Kept identical to ``baselines.harness._direction_label`` and the legacy DL
    exports so the new residuals stay readable by the existing analysis layer.
    """
    return f"+{direction_val}" if direction_val > 0 else str(direction_val)


def build_keyed_residuals(
    sample_index: pl.DataFrame,
    *,
    corridor: str,
    split: str,
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_persist: np.ndarray,
    target_mask: np.ndarray,
    persist_mask: np.ndarray,
) -> pl.DataFrame:
    """Per-sample paired residuals carrying the full key.

    Parameters
    ----------
    sample_index:
        The frame from ``make_sample_index``, in the order the model consumed
        it. Row ``i`` of every array below corresponds to row ``i`` here.
    corridor:
        Corridor label ("E2", "E59", "E4").
    split:
        Split label ("train", "val", "test").
    y_true, y_pred_model, y_pred_persist:
        Dense ``(n_samples, max_N)`` arrays in original (un-z-scored) units.
    target_mask, persist_mask:
        Dense ``(n_samples, max_N)`` boolean arrays, True = VALID (INV-5).
        A cell is exported only where BOTH are valid — the paired set the
        significance tests require.

    Returns
    -------
    DataFrame with ``RESIDUAL_COLUMNS``, sorted deterministically.

    Raises
    ------
    ValueError
        If any array's shape disagrees with the index height, which would mean
        the export is silently misaligned with the population it claims.
    """
    n = sample_index.height
    arrays = {
        "y_true": y_true,
        "y_pred_model": y_pred_model,
        "y_pred_persist": y_pred_persist,
        "target_mask": target_mask,
        "persist_mask": persist_mask,
    }
    shapes = {name: a.shape for name, a in arrays.items()}
    if len({s for s in shapes.values()}) != 1:
        raise ValueError(f"arrays disagree in shape: {shapes}")
    n_rows, max_N = y_true.shape
    if n_rows != n:
        raise ValueError(
            f"array rows ({n_rows}) != sample index height ({n}); the export "
            "is misaligned with its population"
        )

    keep = target_mask & persist_mask
    if not keep.any():
        return pl.DataFrame(schema={c: pl.Utf8 for c in RESIDUAL_COLUMNS})

    # Row i of the flattened arrays maps to index row i // max_N, pair_rank i % max_N.
    rows, cols = np.nonzero(keep)

    directions = sample_index.get_column("direction").to_numpy()[rows]
    starts = sample_index.get_column("start_ts").to_numpy()[rows]
    targets = sample_index.get_column("target_ts").to_numpy()[rows]
    horizons = sample_index.get_column("horizon").to_numpy()[rows]

    frame = pl.DataFrame(
        {
            "corridor": np.full(rows.size, corridor),
            "direction": [direction_label(int(d)) for d in directions],
            "horizon": horizons.astype(np.int64),
            "split": np.full(rows.size, split),
            "start_ts": starts,
            "target_ts": targets,
            "pair_rank": cols.astype(np.int64),
            "y_true": y_true[rows, cols].astype(np.float64),
            "y_pred_model": y_pred_model[rows, cols].astype(np.float64),
            "y_pred_persist": y_pred_persist[rows, cols].astype(np.float64),
        }
    )

    return frame.select(RESIDUAL_COLUMNS).sort(
        ["corridor", "direction", "horizon", "start_ts", "pair_rank"]
    )


def assert_key_is_unique(residuals: pl.DataFrame) -> None:
    """Fail closed when the exported key does not identify a row.

    This is the check whose absence cost a Kaggle kernel: ``harness.py``'s
    docstring declared ``t`` a join key and nothing verified it.
    """
    keys = residuals.select(RESIDUAL_KEY_COLUMNS)
    if keys.height != keys.unique().height:
        dupes = (
            keys.group_by(RESIDUAL_KEY_COLUMNS)
            .len()
            .filter(pl.col("len") > 1)
            .sort("len", descending=True)
        )
        raise ValueError(
            f"residual key is not unique: {dupes.height} duplicated keys, "
            f"worst multiplicity {dupes.get_column('len').max()}"
        )
