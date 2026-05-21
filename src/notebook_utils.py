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
    """Remove all relative ``from .xxx import ...`` statements from source.

    Uses ``ast.get_source_segment`` to locate and remove the EXACT text for
    every relative ImportFrom node (level > 0).  This handles both
    single-line and parenthesized multi-line import blocks without regex
    fragility.

    Single-line:  ``from .config import X``
    Multi-line::

        from .config import (
            X,
            Y,
        )

    Inside the notebook all modules are inlined into the same flat namespace
    so relative imports would raise ``ImportError`` at Kaggle runtime.  This
    is the only transformation applied to module source code before embedding.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # If the source itself is broken, return as-is so the compile test
        # catches it with a clear error.
        return src

    # Collect source segments for all relative ImportFrom nodes.
    segments_to_remove: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            segment = ast.get_source_segment(src, node)
            if segment:
                segments_to_remove.append(segment)

    result = src
    for segment in segments_to_remove:
        # Remove the segment plus its trailing newline (if any).
        result = result.replace(segment + "\n", "")
        result = result.replace(segment, "")

    return result
