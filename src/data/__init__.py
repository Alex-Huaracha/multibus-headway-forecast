"""src.data — supervised dataset construction package (Fase 3 DL).

Public re-exports:
    windowing: make_window_index, compute_max_N, DEFAULT_T_IN, DEFAULT_T_OUT, DEFAULT_STRIDE
    normalization: compute_normalization_stats, apply_zscore, NormalizationStats
    context_features: encode_context, load_atypical_days, CONTEXT_FEATURE_NAMES
    dataset: HeadwayDataset, collate_fn  (requires torch — DL-10, INV-10)

AC-DEP-3: windowing, normalization, and context_features are importable without
torch. Only dataset.py imports torch; importing src.data therefore pulls in
torch transitively — callers that need torch-free imports must import the
sub-modules directly.
"""
from __future__ import annotations

from .context_features import (
    CONTEXT_FEATURE_NAMES,
    encode_context,
    load_atypical_days,
)
from .normalization import NormalizationStats, apply_zscore, compute_normalization_stats
from .windowing import (
    DEFAULT_STRIDE,
    DEFAULT_T_IN,
    DEFAULT_T_OUT,
    WindowIndexEntry,
    compute_max_N,
    make_window_index,
)
from .dataset import HeadwayDataset, collate_fn
