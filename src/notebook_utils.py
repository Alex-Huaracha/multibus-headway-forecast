"""Shared utilities for notebook builder scripts.

This module centralises helpers that all build_notebook_NN.py scripts need
so that each builder is not a private copy with drift risk.

Public API
----------
_strip_relative_imports(src: str) -> str
    Remove all ``from .xxx import ...`` statements so inlined modules work
    in a flat Kaggle cell namespace.

The function was extracted from ``build_notebook_04.py`` (commit cd6d70c).
If a third builder ever appears, add further shared helpers here.
"""
from __future__ import annotations

import ast


def _strip_relative_imports(src: str) -> str:
    """Remove intra-package imports so inlined modules work in a flat Kaggle cell.

    Strips two kinds of imports:
      1. Relative: ``from .config import X`` (level > 0)
      2. Absolute from ``src.*``: ``from src.models.lstm import X`` (level == 0)

    Both would raise ``ModuleNotFoundError`` in a Kaggle notebook where all
    modules are embedded sequentially into the same flat namespace.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    segments_to_remove: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            is_relative = node.level > 0
            is_src_absolute = node.level == 0 and node.module and node.module.startswith("src.")
            if is_relative or is_src_absolute:
                segment = ast.get_source_segment(src, node)
                if segment:
                    segments_to_remove.append(segment)

    result = src
    for segment in segments_to_remove:
        result = result.replace(segment + "\n", "")
        result = result.replace(segment, "")

    return result
