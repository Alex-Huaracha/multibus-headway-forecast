"""Sensitivity sweep for ``PRODUCTIVE_PARAMS.centerline_n_bins``.

Why this exists. The parameter table in ``docs/decisiones-headway-fase2.md`` §3
justified ``CENTERLINE_N_BINS = 50`` with "probado estable contra 10 y 40
(KL < 0.01)". That justification was a mis-attribution: the ``{10, 20, 40}``
sweep with a KL criterion belongs to ``N_POINTS_VARIANTS`` in
``src/build_notebook_03.py``, a different parameter (base 20) of formulation A,
which was discarded. The bin count of the corridor axis was never swept.

This script runs the sweep that was missing, so §3.2 of that document cites a
number someone can reproduce instead of one nobody can check.

What it measures. For each corridor it rebuilds the centerline at several bin
counts and projects a held-out sample of moving pings onto each one, reporting:

- the vertex count actually produced (bins with < 5 pings are dropped, so the
  effective count is below ``n_bins``);
- the resulting corridor length;
- the lateral offset distribution, which is how tightly the polyline tracks the
  point cloud;
- the pairwise KL divergence between the arc-length distributions each
  centerline induces, matching the metric family the original claim invoked.

Two caveats travel with the output and belong in any reading of it:

1. ``cleaned_gps_*.parquet`` is already off-route filtered at 300 m against the
   50-bin centerline, so the evaluation sample favours 50 and its neighbours.
   A finding that some other bin count fits *better* runs against that bias and
   survives it; the absolute ``|lateral|`` magnitudes do not, and are not a raw
   goodness-of-fit measure.
2. The original claim never said which distribution its KL was computed over, so
   this script does not reproduce that metric — it defines one. The ``< 0.01``
   threshold is carried over only to make the comparison against the old claim
   possible.

The centerline and projection routines are imported from ``src.preprocessing``
rather than reimplemented. ``build_notebook_03.py`` keeps its own copy of both,
and that duplication is how the mis-attribution survived in the first place.

Requires ``data/processed/cleaned_gps_E{2,4,59}.parquet``, which is gitignored
and lives in the Kaggle datasets (see ``docs/dataset-manifest.md``). The script
reports which corridors it could not read rather than failing outright.

Usage
-----
    uv run python -m src.build_centerline_sweep
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.stats import entropy  # noqa: E402

from src.preprocessing.config import EMPRESA_CONFIG, PRODUCTIVE_PARAMS  # noqa: E402
from src.preprocessing.corridor import _build_centerline_from_points  # noqa: E402
from src.preprocessing.projection import _project_arc_length  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_METRICS = OUT_DIR / "centerline_bins_sweep.csv"
OUT_KL = OUT_DIR / "centerline_bins_sweep_kl.csv"

CORRIDORS = (2, 4, 59)
#: Bin counts to compare. Brackets the production value on both sides so the
#: sweep can distinguish "50 sits on a plateau" from "more bins keep helping".
BIN_VARIANTS = (10, 20, 40, 50, 80)

#: Pings projected onto each candidate centerline. Independent of the fitting
#: sample so the comparison is not scored on the points that defined the line.
EVAL_SAMPLE = 200_000
#: Seed for the fitting sample. Mirrors ``build_centerline``'s default so the
#: n_bins=50 row reproduces the production centerline.
FIT_SEED = 42
EVAL_SEED = 7
#: Histogram resolution for the KL divergence, over arc-length normalised to
#: [0, 1] — the corridor length itself changes with n_bins, so the raw metre
#: scale is not comparable across variants.
KL_BINS = 50
PROJECTION_CHUNK = 10_000
#: The threshold the superseded claim asserted. Kept only for contrast.
LEGACY_KL_THRESHOLD = 0.01


def _load_samples(empresaid: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (fitting sample, evaluation sample, total moving pings).

    Reproduces ``build_centerline``'s sampling: moving pings only, capped at the
    empresa's ``centerline_sample_cap`` with a fixed seed.
    """
    cfg = EMPRESA_CONFIG[empresaid]
    moving = (
        pl.scan_parquet(DATA_DIR / f"cleaned_gps_E{empresaid}.parquet")
        .filter(
            pl.col("speed_kmh") >= PRODUCTIVE_PARAMS.min_speed_for_centerline_kmh
        )
        .select(["lat", "lon"])
        .collect()
        .to_numpy()
    )

    fit = moving
    if len(fit) > cfg.centerline_sample_cap:
        idx = np.random.default_rng(FIT_SEED).choice(
            len(fit), size=cfg.centerline_sample_cap, replace=False
        )
        fit = fit[idx]

    evaluation = moving
    if len(evaluation) > EVAL_SAMPLE:
        idx = np.random.default_rng(EVAL_SEED).choice(
            len(evaluation), size=EVAL_SAMPLE, replace=False
        )
        evaluation = evaluation[idx]

    return fit, evaluation, len(moving)


def _density(values: np.ndarray) -> np.ndarray:
    """Normalised histogram of arc-length, smoothed off zero for the KL."""
    hist, _ = np.histogram(values, bins=np.linspace(0, 1, KL_BINS), density=True)
    hist = hist + 1e-9
    return hist / hist.sum()


def sweep_corridor(empresaid: int) -> tuple[list[dict], list[dict]]:
    """Build every centerline variant for one corridor and score it."""
    fit, evaluation, n_moving = _load_samples(empresaid)
    corridor = f"E{empresaid}"

    metrics: list[dict] = []
    densities: dict[int, np.ndarray] = {}

    for n_bins in BIN_VARIANTS:
        centerline = _build_centerline_from_points(
            fit,
            n_bins=n_bins,
            trim_pct=PRODUCTIVE_PARAMS.centerline_trim_pct,
            smooth_win=PRODUCTIVE_PARAMS.centerline_smooth_win,
        )
        s, lateral = _project_arc_length(
            evaluation, centerline, chunk_size=PROJECTION_CHUNK
        )
        length_m = float(s.max())
        densities[n_bins] = _density(s / length_m)
        lateral_abs = np.abs(lateral)

        metrics.append(
            {
                "corridor": corridor,
                "n_bins": n_bins,
                "n_vertices": len(centerline),
                "n_moving_pings": n_moving,
                "n_fit_sample": len(fit),
                "n_eval_sample": len(evaluation),
                "corridor_length_km": round(length_m / 1000, 3),
                "lateral_median_m": round(float(np.median(lateral_abs)), 1),
                "lateral_p95_m": round(float(np.percentile(lateral_abs, 95)), 1),
                "is_production": n_bins == PRODUCTIVE_PARAMS.centerline_n_bins,
            }
        )

    divergences = [
        {
            "corridor": corridor,
            "n_bins_a": a,
            "n_bins_b": b,
            "kl": round(float(entropy(densities[a], densities[b])), 4),
        }
        for i, a in enumerate(BIN_VARIANTS)
        for b in BIN_VARIANTS[i + 1:]
    ]
    for row in divergences:
        row["below_legacy_threshold"] = row["kl"] < LEGACY_KL_THRESHOLD

    return metrics, divergences


def main() -> None:
    all_metrics: list[dict] = []
    all_kl: list[dict] = []
    missing: list[str] = []

    for empresaid in CORRIDORS:
        if not (DATA_DIR / f"cleaned_gps_E{empresaid}.parquet").exists():
            missing.append(f"E{empresaid}")
            continue
        metrics, divergences = sweep_corridor(empresaid)
        all_metrics.extend(metrics)
        all_kl.extend(divergences)

    if missing:
        print(
            f"sin datos locales para {', '.join(missing)} — "
            f"faltan parquets en {DATA_DIR.relative_to(REPO_ROOT)} "
            "(ver docs/dataset-manifest.md)"
        )
    if not all_metrics:
        print("ningún corredor disponible; no se escribió nada")
        return

    metrics_table = pl.DataFrame(all_metrics).sort(["corridor", "n_bins"])
    kl_table = pl.DataFrame(all_kl).sort(["corridor", "n_bins_a", "n_bins_b"])
    metrics_table.write_csv(OUT_METRICS)
    kl_table.write_csv(OUT_KL)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(metrics_table.drop("n_fit_sample", "n_eval_sample"))
        print()
        print(
            kl_table.group_by("corridor")
            .agg(
                pl.col("kl").max().alias("kl_maxima"),
                pl.col("below_legacy_threshold").sum().alias("pares_bajo_0.01"),
                pl.len().alias("pares_totales"),
            )
            .sort("corridor")
        )

    print(f"\nescrito en {OUT_METRICS.relative_to(REPO_ROOT)}")
    print(f"escrito en {OUT_KL.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
