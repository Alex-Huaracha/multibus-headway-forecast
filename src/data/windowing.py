"""Windowing module for supervised dataset construction — Fase 3 DL.

AC-WIN-1: Build window index per slot.
AC-WIN-2: Stride-parametrized index generation.
AC-WIN-3: Deterministic slot-boundary-respecting index.
AC-WIN-4: Exported constants DEFAULT_T_IN, DEFAULT_T_OUT, DEFAULT_STRIDE.
AC-WIN-5: Empty-slot guard (returns zero entries when N < T_in + T_out).
AC-WIN-6: Zero torch imports at module level.
AC-MAXN-1: compute_max_N returns train-p99 of (n_buses-1) per (empresaid, direction).
AC-MAXN-2: compute_max_N is called on train-only df; leakage is caller responsibility.

Design decisions (locked in design §2.2 and §5):
  - WindowIndexEntry: TypedDict with empresaid, direction, pair_rank, start_idx.
  - start_idx is relative to the sorted slot frame (not the full df).
  - Slot key: (empresaid, direction, pair_rank).
  - No torch imports anywhere in this module (INV-10, DL-10).
"""
from __future__ import annotations

import math
from typing import TypedDict

import polars as pl

# ---------------------------------------------------------------------------
# Constants (locked in design §5 — DL-1)
# ---------------------------------------------------------------------------

DEFAULT_T_IN: int = 12
DEFAULT_T_OUT: int = 1
DEFAULT_STRIDE: int = 1

_SLOT_COLS: list[str] = ["empresaid", "direction", "pair_rank"]


class WindowIndexEntry(TypedDict):
    """Single window anchor.

    empresaid: int — corridor identifier.
    direction: int — bus direction (-1 or +1).
    pair_rank: int — positional slot index within a snapshot.
    start_idx: int — row index into the sorted slot frame where this window starts.
                     The window covers rows [start_idx, start_idx + T_in + T_out).
    """

    empresaid: int
    direction: int
    pair_rank: int
    start_idx: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slot_lengths(df: pl.DataFrame) -> pl.DataFrame:
    """Return a DataFrame with (empresaid, direction, pair_rank, n_rows).

    Used by make_window_index to determine how many windows each slot produces.
    The count is over ALL rows (null delta_t_min counts — windowing does not
    drop null rows; the Dataset layer handles null masking later).
    """
    return (
        df.group_by(_SLOT_COLS)
        .agg(pl.len().alias("n_rows"))
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_max_N(
    train_df: pl.DataFrame,
    *,
    quantile: float = 0.99,
) -> dict[tuple[int, int], int]:
    """Train-p99 of (n_buses - 1) per (empresaid, direction). DL-5. AC-MAXN-1..2.

    Parameters
    ----------
    train_df:
        DataFrame filtered to train rows only (caller responsibility).
        Must have columns: empresaid (Int64), direction (Int64), n_buses (Int32).
    quantile:
        Percentile for the cap (default 0.99 per DL-5).

    Returns
    -------
    dict[(empresaid, direction), int] — the maximum slot index (0-based max_N).
    Returned values are Python int (not np.int64) so they can be used as tensor
    dimensions directly.
    """
    # Compute quantile of (n_buses - 1) per (empresaid, direction).
    # We use unique snapshots: each row in the windowing context represents one
    # (empresaid, direction, snapshot) combination. n_buses is per snapshot.
    result: dict[tuple[int, int], int] = {}

    # Group by (empresaid, direction) and compute the p99 of (n_buses - 1).
    stats = (
        train_df
        .with_columns(
            (pl.col("n_buses") - 1).alias("_n_slots")
        )
        .group_by(["empresaid", "direction"])
        .agg(
            pl.col("_n_slots").quantile(quantile).alias("max_N_float")
        )
    )

    for row in stats.iter_rows(named=True):
        key = (int(row["empresaid"]), int(row["direction"]))
        result[key] = int(math.floor(row["max_N_float"]))

    return result


def make_window_index(
    df: pl.DataFrame,
    *,
    T_in: int = DEFAULT_T_IN,
    T_out: int = DEFAULT_T_OUT,
    horizon: int | None = None,
    stride: int = DEFAULT_STRIDE,
) -> list[WindowIndexEntry]:
    """Deterministic per-slot window index. DL-1, DL-11. AC-WIN-1..5, AC-WIN-H1..H3.

    Produces a list of WindowIndexEntry dicts where each entry anchors one
    training window. Entries are sorted by (empresaid, direction, pair_rank,
    start_idx) for determinism.

    Parameters
    ----------
    df:
        headways DataFrame sorted (or sortable) by (slot_cols, t).
        Columns required: empresaid, direction, pair_rank, t.
    T_in:
        Input sequence length (number of timesteps fed to model).
    T_out:
        Prediction sequence length (number of future timesteps). Retained for
        backward compatibility. Default 1.
    horizon:
        DIRECT-horizon prediction offset. When provided, ``window_size = T_in + horizon``
        (overrides the T_out contribution). Default ``None`` falls back to T_out semantics
        so existing callers are unaffected. ``horizon=1`` produces results identical to
        ``T_out=1`` (AC-WIN-H3).
    stride:
        Step between consecutive window starts (default 1 = every timestep).

    Returns
    -------
    list[WindowIndexEntry] — may be empty if no slot has enough rows.
    """
    window_size = T_in + (horizon if horizon is not None else T_out)
    index: list[WindowIndexEntry] = []

    # Partition by slot to keep slot boundaries clean (AC-WIN-3).
    slots = df.sort(_SLOT_COLS + ["t"]).partition_by(_SLOT_COLS, maintain_order=True)

    for slot_df in slots:
        if slot_df.is_empty():
            continue

        n_rows = len(slot_df)
        if n_rows < window_size:
            # AC-WIN-5: not enough rows for even one window — skip.
            continue

        # Extract slot key from first row.
        first = slot_df.row(0, named=True)
        emp: int = int(first["empresaid"])
        direction: int = int(first["direction"])
        pr: int = int(first["pair_rank"])

        # Generate start indices with stride.
        # Number of valid windows: floor((n_rows - window_size) / stride) + 1
        n_windows = math.floor((n_rows - window_size) / stride) + 1
        for w in range(n_windows):
            start_idx = w * stride
            index.append(
                WindowIndexEntry(
                    empresaid=emp,
                    direction=direction,
                    pair_rank=pr,
                    start_idx=start_idx,
                )
            )

    return index
