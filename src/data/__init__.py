"""src.data — supervised dataset construction package (Fase 3 DL).

Public re-exports (full surface completed in PR2 when dataset.py is added):
    windowing: make_window_index, compute_max_N, DEFAULT_T_IN, DEFAULT_T_OUT, DEFAULT_STRIDE
    normalization: compute_normalization_stats, apply_zscore, NormalizationStats
    context_features: encode_context, load_atypical_days
    dataset: HeadwayDataset, collate_fn  (PR2 — requires torch)

AC-DEP-3: all pure modules (windowing, normalization, context_features) are
importable without torch. Only dataset.py imports torch.
"""
from __future__ import annotations

from .normalization import NormalizationStats, apply_zscore, compute_normalization_stats
from .windowing import (
    DEFAULT_STRIDE,
    DEFAULT_T_IN,
    DEFAULT_T_OUT,
    WindowIndexEntry,
    compute_max_N,
    make_window_index,
)
