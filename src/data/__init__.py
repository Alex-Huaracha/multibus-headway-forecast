"""src.data — supervised dataset construction package (Fase 3 DL).

Public re-exports (full surface completed in PR2 when dataset.py is added):
    windowing: make_window_index, compute_max_N, DEFAULT_T_IN, DEFAULT_T_OUT, DEFAULT_STRIDE
    normalization: compute_normalization_stats, apply_zscore, NormalizationStats
    context_features: encode_context, load_atypical_days
    dataset: HeadwayDataset, collate_fn  (PR2 — requires torch)
"""
from __future__ import annotations
