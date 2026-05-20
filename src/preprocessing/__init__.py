"""Preprocessing package for Fase 2 — corridor centerline, projection, direction,
trip segmentation, and headway computation.

All modules use polars + numpy exclusively (no pandas in production paths).
Parameters are frozen in config.py from docs/decisiones-headway-fase2.md §3.
"""
