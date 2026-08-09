"""Re-test of the neighbour mutual information across the four headway formulations.

Why this exists. ``docs/decisiones-headway-fase2.md`` §2.2 records the viability
probe's verdict for the four candidate headway definitions, including a neighbour
mutual-information column. The paper then compressed that verdict into "C.2 was
adopted for 6 of 7 criteria **and the best relationship between neighbouring
buses**". The MI column does not support the second half: formulation A scores
higher than C.2 in E2. Those numbers lived only in a markdown table, produced by a
Kaggle run that was never repeated, so the claim could not be checked.

This script checks it, by running the probe's own definitions against local data.

What it does NOT do: reproduce the probe. The probe built its own single-pass
centerline inside the notebook, over raw GPS. This runs over
``cleaned_gps_*.parquet``, whose ``s`` and ``direction`` come from the current
production pipeline (two-pass centerline, direction-conditional sort key). So the
magnitudes move. What carries over is the ordering between formulations, which is
what the claim is about — and E59's C.2 lands at 1.268 against the recorded 1.26,
which is the evidence that the two settings are comparable at all.

The formulation and diagnostic functions are extracted from the ``code()`` string
blocks of ``src/build_notebook_03.py`` and executed, rather than reimplemented.
That builder is not importable — its logic lives inside notebook-source strings —
and reimplementing it is exactly how the mis-attribution this script investigates
came about. Blocks are located by the functions they define, not by position, so
reordering the notebook does not silently change what runs here.

Requires ``data/processed/cleaned_gps_E{2,4,59}.parquet``, which is gitignored and
lives in the Kaggle datasets (see ``docs/dataset-manifest.md``). Corridors whose
parquet is missing are reported and skipped.

Usage
-----
    uv run python -m src.build_mi_recheck
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import ast  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.feature_selection import mutual_info_regression  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
PROBE_BUILDER = REPO_ROOT / "src" / "build_notebook_03.py"
OUT_CSV = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon" / "mi_recheck.csv"

CORRIDORS = (2, 4, 59)
#: Formulation id → the function that computes it, in the probe's vocabulary.
FORMULATIONS = {
    "A": "compute_headways_A",
    "B": "compute_headways_B",
    "C1": "compute_headways_C1",
    "C2": "compute_headways_C2",
}
#: Every function the probe blocks must yield for this script to run.
#: ``_value_col`` is the column resolver ``diag_mi_neighbours`` calls internally.
REQUIRED = tuple(FORMULATIONS.values()) + (
    "build_snapshots",
    "diag_mi_neighbours",
    "_value_col",
)


def probe_namespace() -> dict:
    """Execute the probe's constants and formulation functions into a namespace.

    Block 0 mixes the constants with Kaggle path discovery, so its statements run
    one at a time and the ones that need ``/kaggle/input`` are skipped. The
    function blocks also contain driver code that calls them against the probe's
    own frames, so only their definitions are executed.
    """
    tree = ast.parse(PROBE_BUILDER.read_text(encoding="utf-8"))
    blocks = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "code"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]

    ns: dict = {
        "np": np,
        "pl": pl,
        "date": date,
        "time": time,
        "mutual_info_regression": mutual_info_regression,
    }

    # Constants. Anything that reaches for /kaggle/input fails and is skipped;
    # the same statements print a Kaggle directory listing, muted here.
    for block in blocks:
        if "RNG_SEED" not in block:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            for node in ast.parse(block).body:
                try:
                    exec(compile(ast.Module([node], []), "<probe>", "exec"), ns)
                except Exception:  # noqa: BLE001 — Kaggle-only statements
                    continue
        break
    ns["log"] = lambda *args, **kwargs: None

    # Definitions only, located by the names they bind.
    for block in blocks:
        for node in ast.parse(block).body:
            if isinstance(node, ast.FunctionDef) and node.name in REQUIRED:
                exec(compile(ast.Module([node], []), "<probe>", "exec"), ns)

    missing = [name for name in REQUIRED if name not in ns]
    if missing:
        raise RuntimeError(
            f"no se pudieron extraer de {PROBE_BUILDER.name}: {missing}. "
            "El builder cambió de forma; revisar los bloques code()."
        )
    return ns


def load_gps(empresaid: int, target_dates: list[date]) -> pl.DataFrame:
    """Cleaned pings for one corridor, on the probe's days, in its schema."""
    return (
        pl.scan_parquet(DATA_DIR / f"cleaned_gps_E{empresaid}.parquet")
        .with_columns(
            [
                pl.lit(empresaid).cast(pl.Int64).alias("empresaid"),
                pl.col("t").alias("time"),
                pl.col("t").dt.date().alias("day"),
            ]
        )
        .filter(pl.col("day").is_in(target_dates))
        .select(
            [
                "empresaid",
                "unidadid",
                "day",
                "time",
                "lat",
                "lon",
                "s",
                "direction",
                "speed_kmh",
            ]
        )
        .sort(["empresaid", "unidadid", "time"])
        .collect()
    )


def recheck_corridor(empresaid: int, ns: dict) -> list[dict]:
    """Every formulation's neighbour MI for one corridor."""
    gps = load_gps(empresaid, ns["TARGET_DATES"])
    if gps.height == 0:
        return []

    snapshots = ns["build_snapshots"](gps)
    frames = {
        "A": ns["compute_headways_A"](gps),
        "B": ns["compute_headways_B"](snapshots),
        "C1": ns["compute_headways_C1"](snapshots),
        "C2": ns["compute_headways_C2"](snapshots, gps),
    }

    rows = []
    for fid, frame in frames.items():
        mi, sd = ns["diag_mi_neighbours"](frame, fid)
        rows.append(
            {
                "corridor": f"E{empresaid}",
                "formulation": fid,
                "n_pings": gps.height,
                "n_days": gps.get_column("day").n_unique(),
                "n_rows": frame.height,
                "mi_bits": None if np.isnan(mi) else round(mi, 3),
                "mi_sd": None if np.isnan(sd) else round(sd, 3),
                "is_adopted": fid == "C2",
            }
        )
    return rows


def main() -> None:
    ns = probe_namespace()
    records: list[dict] = []
    missing: list[str] = []

    for empresaid in CORRIDORS:
        if not (DATA_DIR / f"cleaned_gps_E{empresaid}.parquet").exists():
            missing.append(f"E{empresaid}")
            continue
        records.extend(recheck_corridor(empresaid, ns))

    if missing:
        print(
            f"sin datos locales para {', '.join(missing)} — "
            f"faltan parquets en {DATA_DIR.relative_to(REPO_ROOT)} "
            "(ver docs/dataset-manifest.md)"
        )
    if not records:
        print("ningún corredor disponible; no se escribió nada")
        return

    table = pl.DataFrame(records).sort(["corridor", "formulation"])
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table.drop("n_pings", "n_days"))
        print()
        print(
            table.pivot(values="mi_bits", index="formulation", on="corridor").sort(
                "formulation"
            )
        )

    winners = (
        table.sort("mi_bits", descending=True)
        .group_by("corridor")
        .first()
        .sort("corridor")
    )
    print("\nformulación con mayor MI por corredor:")
    for row in winners.iter_rows(named=True):
        print(f"   {row['corridor']}: {row['formulation']} ({row['mi_bits']} bits)")

    print(f"\nescrito en {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
