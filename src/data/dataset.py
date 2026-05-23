"""HeadwayDataset — torch adapter for supervised dataset construction.

This is the ONLY module in src/data/ that imports torch (INV-10, DL-10).
All other modules (windowing, normalization, context_features) are torch-free.

ACs covered:
    AC-DS-1: __getitem__ returns dict with keys {input, target, input_mask, target_mask, context}.
    AC-DS-2: tensor shapes per item: input (T_in, max_N), target (T_out, max_N),
             masks same as data, context (T_in, 5).
    AC-DS-3: float32 for input/target/context; bool for masks.
    AC-DS-4: len(dataset) == len(window_index).
    AC-DS-5: collate_fn stacks dicts into batched tensors on dim 0.
    AC-DS-6: DataLoader(dataset, collate_fn=collate_fn) iterates without error.
    AC-MASK-1: present non-null slot → mask True (True = VALID, INV-5).
    AC-MASK-2: absent slot → mask False, value 0.0.
    AC-MASK-3: present but null delta_t_min → mask False, value 0.0.
    AC-MASK-4: mask convention identical between input_mask and target_mask.
    AC-DS-NOMAT-1 / INV-7: __init__ MUST NOT call __getitem__ or iterate windows.

Design refs: spec §4 (AC-DS-*, AC-MASK-*), design §2.5, §4, §5, INV-4, INV-5, INV-7.
"""
from __future__ import annotations

from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset

from .windowing import WindowIndexEntry

# Re-export CONTEXT_FEATURE_NAMES for __init__.py convenience.
from .context_features import CONTEXT_FEATURE_NAMES

_SLOT_COLS: list[str] = ["empresaid", "direction", "pair_rank"]


class HeadwayDataset(Dataset):
    """Snapshot-as-set dataset backed by a precomputed window index.

    Returns one dict of 5 tensors per __getitem__ call; windows are
    materialized on-the-fly (never at __init__ time per INV-7 / DL-11).

    Tensor contract (per item, before DataLoader batching):
        input       : (T_in, max_N)   float32   — z-scored delta_t_min; 0 where absent
        target      : (T_out, max_N)  float32   — same for the horizon
        input_mask  : (T_in, max_N)   bool      — True where slot present AND non-null
        target_mask : (T_out, max_N)  bool      — same convention for horizon
        context     : (T_in, 5)       float32   — cyclical time + atypical flag

    Mask polarity: True = VALID (PyTorch attention_mask convention, INV-5).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        window_index: list[WindowIndexEntry],
        *,
        max_N_by_direction: dict[tuple[int, int], int],
        T_in: int,
        T_out: int,
        value_col: str = "delta_t_min_z",
        context_cols: tuple[str, ...] = CONTEXT_FEATURE_NAMES,
    ) -> None:
        """Construct lightweight wrapper. AC-DS-NOMAT-1: MUST NOT iterate windows.

        Parameters
        ----------
        df:
            Full headways DataFrame (any split) that has already been:
            winsorized, z-scored (delta_t_min_z column present), and
            context-feature encoded (CONTEXT_FEATURE_NAMES columns present).
        window_index:
            Precomputed list of WindowIndexEntry dicts from make_window_index.
        max_N_by_direction:
            Per-(empresaid, direction) maximum slot count (0-indexed, train-p99).
        T_in:
            Number of input timesteps per window.
        T_out:
            Number of target timesteps per window.
        value_col:
            Name of the z-scored column in df (default: "delta_t_min_z").
        context_cols:
            Ordered tuple of context column names (must be 5 columns, float64).
        """
        # Store metadata — NO window materialization here (INV-7).
        self._df = df
        self._window_index = window_index
        self._max_N_by_direction = max_N_by_direction
        self._T_in = T_in
        self._T_out = T_out
        self._value_col = value_col
        self._context_cols = list(context_cols)

        # Cache per-slot partitions lazily (populated on first access per slot).
        # Key: (empresaid, direction, pair_rank) → sorted slot DataFrame.
        self._slot_cache: dict[tuple[int, int, int], pl.DataFrame] = {}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """AC-DS-4: total number of windows across all slots."""
        return len(self._window_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Materialize one window. AC-DS-1, AC-DS-2, AC-DS-3.

        Returns
        -------
        dict with keys: input, target, input_mask, target_mask, context.
        """
        entry = self._window_index[idx]
        empresaid: int = entry["empresaid"]
        direction: int = entry["direction"]
        pair_rank: int = entry["pair_rank"]
        start_idx: int = entry["start_idx"]

        return self._materialize_window(
            empresaid=empresaid,
            direction=direction,
            pair_rank=pair_rank,
            start_idx=start_idx,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slot_frame(self, empresaid: int, direction: int, pair_rank: int) -> pl.DataFrame:
        """Return the sorted slot frame, cached per slot key.

        The frame is sorted by 't' once and reused across all windows
        that share the same slot.
        """
        key = (empresaid, direction, pair_rank)
        if key not in self._slot_cache:
            slot_df = (
                self._df
                .filter(
                    (pl.col("empresaid") == empresaid)
                    & (pl.col("direction") == direction)
                    & (pl.col("pair_rank") == pair_rank)
                )
                .sort("t")
            )
            self._slot_cache[key] = slot_df
        return self._slot_cache[key]

    def _materialize_window(
        self,
        *,
        empresaid: int,
        direction: int,
        pair_rank: int,
        start_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Build input/target/mask/context tensors for one window.

        Design note: the window_index records start_idx relative to the sorted
        slot frame for (empresaid, direction, pair_rank). We slice T_in rows for
        the input and T_out rows for the target; then pad the N dimension
        to max_N using per-snapshot presence detection.
        """
        window_size = self._T_in + self._T_out
        max_N = self._max_N_by_direction[(empresaid, direction)]

        # Get the sorted slot frame.
        slot_df = self._slot_frame(empresaid, direction, pair_rank)

        # Slice the window rows.
        window_df = slot_df.slice(start_idx, window_size)

        # Extract value column and context columns.
        # The slot frame holds ONE pair_rank at a time; we need all pair_ranks
        # for this snapshot to build the full (T, max_N) tensors.
        # We use the snapshot timestamps from this slot to locate all pair_ranks.
        timestamps = window_df["t"].to_list()

        # For each snapshot timestep, gather all pair_ranks in [0, max_N).
        # This requires a lookup in the full df filtered to (empresaid, direction).
        # We cache the direction-level frame for efficiency.
        dir_key = (empresaid, direction)
        if not hasattr(self, "_dir_cache"):
            self._dir_cache: dict[tuple[int, int], pl.DataFrame] = {}
        if dir_key not in self._dir_cache:
            self._dir_cache[dir_key] = (
                self._df
                .filter(
                    (pl.col("empresaid") == empresaid)
                    & (pl.col("direction") == direction)
                )
            )
        dir_df = self._dir_cache[dir_key]

        # Build tensors by iterating over the window timesteps.
        # We build dense (T, max_N) matrices where absent pair_ranks are zero / False.
        T = window_size

        # Pre-allocate: float tensors default 0.0, bool mask default False.
        values = torch.zeros((T, max_N), dtype=torch.float32)
        masks = torch.zeros((T, max_N), dtype=torch.bool)
        context = torch.zeros((T, len(self._context_cols)), dtype=torch.float32)

        for t_idx, ts in enumerate(timestamps):
            # Filter dir_df to this exact snapshot timestamp.
            snap = dir_df.filter(pl.col("t") == ts)

            # Context is the same per snapshot across pair_ranks — take first row.
            if not snap.is_empty():
                ctx_row = snap.row(0, named=True)
                for c_idx, col_name in enumerate(self._context_cols):
                    if col_name in ctx_row and ctx_row[col_name] is not None:
                        context[t_idx, c_idx] = float(ctx_row[col_name])

            # Fill values and masks per pair_rank present in this snapshot.
            for pr in snap["pair_rank"].to_list():
                if pr < 0 or pr >= max_N:
                    # Truncate pair_ranks beyond max_N (AC-MAXN-2).
                    continue
                pr_row = snap.filter(pl.col("pair_rank") == pr)
                if pr_row.is_empty():
                    continue
                val = pr_row[self._value_col][0]
                if val is not None:
                    values[t_idx, pr] = float(val)
                    masks[t_idx, pr] = True
                # If val is None: value stays 0.0, mask stays False (AC-MASK-3).

        return {
            "input": values[: self._T_in],
            "target": values[self._T_in :],
            "input_mask": masks[: self._T_in],
            "target_mask": masks[self._T_in :],
            "context": context[: self._T_in],
        }


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack a list of __getitem__ outputs into batched tensors (B-axis prepended).

    All tensors in a batch share the same shape (T, max_N or 5) since max_N is
    fixed per (empresaid, direction) and all items in a batch should come from
    the same direction. torch.stack is used (not pad_sequence) because shapes
    are guaranteed equal.

    AC-DS-5: batch dimension is dim 0 for all tensors.
    AC-DS-6: compatible with torch.utils.data.DataLoader.
    """
    keys = list(batch[0].keys())
    return {k: torch.stack([item[k] for item in batch], dim=0) for k in keys}
