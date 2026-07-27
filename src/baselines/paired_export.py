"""Per-sample B5_XGB TEST predictions keyed compatibly with the DL population.

Why this module exists
----------------------
The paper's claim "LSTM beats the leveled XGBoost baseline in 8/8 cells" compares
MAEs computed over DIFFERENT sample populations:

  * ``baselines_results_multih.csv`` (and its E4 twin) aggregate over EVERY test
    row that has a non-null prediction.
  * The DL metrics aggregate over the DL WINDOW population: cold-start rows are
    dropped and every target is replicated once per anchoring window slot.

The project's own measured aggregate-vs-paired framing bias for persistence is
0.28-0.53 min, which is larger than 7 of the 8 claimed XGB margins (+0.05 to
+0.41), so the claim cannot be defended from aggregate metrics. Re-scoring XGB
over exactly the DL's rows requires per-sample XGB predictions carrying the
FULL unique key of the headways frame.

``harness.XGB_RESIDUAL_COLUMNS`` cannot serve that purpose: it exports
``[corridor, direction, horizon, t, y_true, y_pred_xgb, y_pred_persist]`` and
drops ``pair_rank`` at the final ``select``. ``t`` is NOT a unique key — the
headways frame is keyed on ``(t, direction, pair_rank)`` and carries roughly 4.2
rows per ``(t, direction)`` — so those residuals cannot be joined row-for-row
against anything.

Why this module is additive instead of a fix in place
----------------------------------------------------
``harness.py``, ``fitted.py`` and ``statistical.py`` are inlined VERBATIM into
the NB10 and NB16 notebooks by ``build_notebook_*.embed_module``. Editing any of
them changes those generated notebooks' bytes and breaks their byte-identity
guards, which would force a re-push of frozen Kaggle artifacts. So nothing here
touches them. The escape hatch is ``B5FitResult.predictions``, which is the
input frame plus ``y_pred_b5_xgb`` and therefore still carries ``pair_rank``,
``empresaid`` and ``split``.

Contracts inherited unchanged (nothing is reimplemented here)
------------------------------------------------------------
Everything methodological is delegated to ``harness.run_corridor``:

  * temporal split (``split_temporal``),
  * winsorization — the p99 threshold of ``delta_t_min`` is computed on TRAIN
    only and applied to ALL splits, because ``run_corridor`` hands the full
    split-tagged frame to ``winsorize_train_p99`` before any baseline runs,
  * the B5_XGB feature construction, the seeded 24-configuration random search
    selected strictly on VALIDATION, and the leakage contract (the test split is
    absent from every DMatrix).

This module only RESHAPES the resulting test rows. It applies the exact same
filtering semantics as ``harness._build_xgb_residuals`` (keep a sample only when
the target AND both predictions are present) and the exact same ``"-1"`` / ``"+1"``
direction-label convention the DL residual exports use, so the string join keys
are compatible on both sides.

Reproducibility note
--------------------
An export run REFITS B5_XGB. That is safe by design: ``fitted.py`` pins the
search seed, the training seed and ``nthread``, so a rerun on the same inputs
selects the same configuration and produces the same predictions.
``search_provenance_row`` exports the winning configuration precisely so the
refit can be diffed against the frozen ``xgb_search_config_multih.csv`` /
``xgb_search_config_E4_multih.csv``; a mismatch there invalidates the export.

Output-shape decision: ONE combined long CSV
--------------------------------------------
The notebook that drives this module writes a SINGLE combined CSV for all three
corridors and all four horizons, with ``corridor`` and ``horizon`` as columns,
because:

  1. the consumer is one join against the DL per-sample residuals, so a single
     frame avoids a discover-and-concat step that could silently miss a file;
  2. ``corridor`` and ``horizon`` are already part of the join key, so splitting
     by them would encode key material in filenames instead of columns;
  3. every split-by-file scheme risks colliding with an existing glob. The
     analysis layer discovers artifacts by pattern — ``degradation.load_results``
     globs ``*_results_*.csv`` and ``paired_audit`` globs ``*_residuals_h*.csv``
     — and a foreign-schema file caught by either glob would crash the
     degradation build or silently contaminate ``consolidated_multihorizon.csv``
     and Figure 1. One file with a name matching NEITHER pattern is the smallest
     surface for that failure mode.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from .harness import CorridorRun, run_corridor

# Full export schema. The four columns beyond `XGB_RESIDUAL_COLUMNS` are the
# point of this module: `pair_rank` completes the headways unique key and
# `empresaid` makes the composite data key `(empresaid, unidadid)`-compatible
# corridor identity explicit instead of implied by the `corridor` label.
XGB_PAIRED_COLUMNS: list[str] = [
    "corridor",
    "empresaid",
    "direction",
    "horizon",
    "t",
    "pair_rank",
    "y_true",
    "y_pred_xgb",
    "y_pred_persist",
]

# The unique key of one exported sample, and the deterministic sort order.
# `(t, direction, pair_rank)` is the unique key of the headways frame; the
# `corridor` and `horizon` prefix scopes it across the 12 (corridor, horizon)
# runs that land in one combined export.
XGB_PAIRED_KEY: list[str] = [
    "corridor",
    "direction",
    "horizon",
    "t",
    "pair_rank",
]

# Columns `paired_xgb_test_frame` needs from a `B5FitResult.predictions` frame.
_REQUIRED_SOURCE_COLUMNS: list[str] = [
    "empresaid",
    "t",
    "direction",
    "pair_rank",
    "delta_t_min",
    "split",
    "y_pred_b1",
    "y_pred_b5_xgb",
]

# Signed direction label, vectorised. Semantically identical to
# `harness._direction_label` ("+1" / "-1"), which the DL residual exports also
# use; `tests/baselines/test_paired_export.py` pins that equivalence so the two
# conventions cannot drift apart.
_DIRECTION_LABEL = (
    pl.when(pl.col("direction") > 0)
    .then(pl.concat_str([pl.lit("+"), pl.col("direction").cast(pl.Utf8)]))
    .otherwise(pl.col("direction").cast(pl.Utf8))
)


def paired_xgb_test_frame(
    predictions: pl.DataFrame, corridor_name: str, *, horizon: int
) -> pl.DataFrame:
    """Reshape a ``B5FitResult.predictions`` frame into the paired TEST export.

    Parameters
    ----------
    predictions:
        The frame returned as ``B5FitResult.predictions`` — the split-tagged,
        winsorized headways frame with the baseline prediction columns added.
        Must still carry ``pair_rank`` (it does: ``fit_predict_b5_xgb`` selects
        the caller's original columns before appending its prediction).
    corridor_name:
        Corridor label written to the ``corridor`` column (e.g. "E2", "E59", "E4").
    horizon:
        Forecast horizon in steps, written to the ``horizon`` column.

    Returns
    -------
    pl.DataFrame with exactly :data:`XGB_PAIRED_COLUMNS`, restricted to TEST rows
    where the target AND both predictions are non-null, sorted by
    :data:`XGB_PAIRED_KEY`.

    Raises
    ------
    ValueError
        If any column in :data:`_REQUIRED_SOURCE_COLUMNS` is absent — in
        particular if ``pair_rank`` was already dropped upstream, which would
        silently produce a non-unique key.
    """
    missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(
            "paired_xgb_test_frame: predictions frame is missing required "
            f"columns {missing}. Pass B5FitResult.predictions from a "
            "run_corridor(..., include_fitted=True) call, not the residual export."
        )

    return (
        predictions.filter(
            (pl.col("split") == "test")
            # Same paired-sample semantics as harness._build_xgb_residuals.
            & pl.col("delta_t_min").is_not_null()
            & pl.col("y_pred_b5_xgb").is_not_null()
            & pl.col("y_pred_b1").is_not_null()
        )
        .with_columns(
            pl.lit(corridor_name, dtype=pl.Utf8).alias("corridor"),
            pl.col("empresaid").cast(pl.Int64),
            _DIRECTION_LABEL.alias("direction"),
            pl.lit(horizon, dtype=pl.Int64).alias("horizon"),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64).alias("y_true"),
            pl.col("y_pred_b5_xgb").cast(pl.Float64).alias("y_pred_xgb"),
            pl.col("y_pred_b1").cast(pl.Float64).alias("y_pred_persist"),
        )
        .select(XGB_PAIRED_COLUMNS)
        .sort(XGB_PAIRED_KEY)
    )


def paired_xgb_from_run(
    run: CorridorRun, corridor_name: str, *, horizon: int
) -> pl.DataFrame:
    """Paired TEST export for an already-computed :class:`CorridorRun`.

    Use this when the caller already ran ``run_corridor`` and wants both the
    metrics and the paired export without refitting.

    Raises
    ------
    ValueError
        If the run was produced with ``include_fitted=False`` (no fitted model,
        hence nothing to export).
    """
    if run.fit_result is None:
        raise ValueError(
            "paired_xgb_from_run: run has no fit_result — call run_corridor with "
            "include_fitted=True (the default) to fit B5_XGB."
        )
    return paired_xgb_test_frame(
        run.fit_result.predictions, corridor_name, horizon=horizon
    )


def export_paired_xgb(
    headways: pl.DataFrame,
    corridor_name: str,
    *,
    horizon: int = 1,
    atypical_dates: set[date] | None = None,
) -> tuple[pl.DataFrame, CorridorRun]:
    """Fit B5_XGB for one corridor x horizon and return the paired TEST export.

    Thin composition over ``harness.run_corridor`` — the split, the train-only
    p99 winsorization applied to all splits, the feature construction and the
    validation-only random search all happen there, unchanged.

    Parameters
    ----------
    headways:
        Raw headways frame (no ``split`` column), with ``empresaid`` present.
    corridor_name:
        Corridor label for the export.
    horizon:
        Forecast horizon in steps.
    atypical_dates:
        Atypical-day calendar forwarded to B5_XGB. An explicit empty set raises
        inside ``fitted._build_features`` (fail closed).

    Returns
    -------
    (paired_export, run)
        ``paired_export`` has :data:`XGB_PAIRED_COLUMNS`; ``run`` is returned so
        the caller can also persist metrics and the search provenance without
        paying for a second fit.
    """
    run = run_corridor(
        headways,
        corridor_name,
        horizon=horizon,
        include_fitted=True,
        atypical_dates=atypical_dates,
    )
    return paired_xgb_from_run(run, corridor_name, horizon=horizon), run


def search_provenance_row(
    run: CorridorRun, corridor_name: str, *, horizon: int
) -> dict:
    """Flat audit row describing the fitted configuration behind one export.

    Mirrors the row NB10/NB16 persist to ``xgb_search_config*.csv`` so the refit
    performed by the export notebook can be diffed against the frozen search
    provenance of the original runs.
    """
    if run.fit_result is None:
        raise ValueError(
            "search_provenance_row: run has no fit_result — call run_corridor "
            "with include_fitted=True (the default)."
        )
    fit = run.fit_result
    return {
        "corridor": corridor_name,
        "horizon": horizon,
        "n_configs_evaluated": fit.n_configs_evaluated,
        "search_seed": fit.search_seed,
        "best_val_rmse": fit.best_val_rmse,
        "best_iteration": fit.best_iteration,
        "used_atypical_flag": fit.used_atypical_flag,
        **{f"param_{k}": str(v) for k, v in sorted(fit.best_params.items())},
    }
