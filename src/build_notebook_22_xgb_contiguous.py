"""Generate the XGBoost refit notebook on the canonical shared sample index.

  notebooks/22_xgb_contiguous/  → alexhuaracha/22-xgb-contiguous

One notebook covers all three corridors x four horizons: every cell reads the
same three parquets from ``04-preprocessing``, so splitting it would only risk
the corridors drifting apart in code.

Why this exists
---------------
``baselines/fitted.py`` builds its lags with
``.forward_fill().shift(horizon + k - 1).over(_SLOT_COLS)`` — a **positional**
step that never checks consecutive rows are consecutive minutes. That is the
audit's §3 defect reaching the fitted baseline through a different mechanism, so
"levelled competitor" was never quite true: the two families were mis-specified
differently over different populations, and audit §2.1 could only reconcile them
with a post-hoc join.

This notebook instead:

1. recomputes the canonical sample index and **fails closed** unless its
   SHA-256 matches the frozen ``sample_index_manifest.csv`` — the same gate the
   LSTM notebooks run, which is what makes "same population" verifiable rather
   than asserted;
2. reads ``lag_k`` off the contiguous window via ``build_contiguous_features``,
   so with ``N_LAGS == T_in == 12`` the XGBoost sees exactly the twelve
   observations the LSTM sees;
3. keeps the frozen 24-configuration validation search (``SEARCH_SEED`` and
   ``SEARCH_SPACE`` unchanged) so selection provenance stays auditable;
4. drops the atypical flag — a whole-day aggregate, hence unknowable at
   prediction time (plan §2, C3);
5. exports residuals under the full key so pairing never needs another run.

``fitted.py`` is embedded for its search sampler and frozen constants. Its
``_build_features`` carries the positional defect and MUST NOT be called here; a
guard test asserts the notebook never does.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_utils import _strip_relative_imports  # noqa: E402

MANIFEST_CSV = ROOT / "docs" / "resultados" / "csv-multihorizon" / "sample_index_manifest.csv"
OUT_DIR = ROOT / "notebooks" / "22_xgb_contiguous"
NOTEBOOK_NAME = "22_xgb_contiguous.ipynb"
KERNEL_ID = "alexhuaracha/22-xgb-contiguous"

CORRIDORS = [("E2", 2), ("E59", 59), ("E4", 4)]
HORIZONS = [1, 3, 5, 10]

PARQUET_HASHES = {
    "headways_E2.parquet": "82a34eaffc79cd82346d4595a2e72f5d3ffb751ed37fa0fc0cde3a8f8fb345d4",
    "headways_E59.parquet": "0b5f5593caaa94e4e6af7da672bc2cad7b49b69b7cbd0a22092f15700a89a448",
    "headways_E4.parquet": "1dde7f38eea9bc7d9941c17cbc3d326cb864e70be815a1a7e3d0ae2691f19273",
}

KERNEL_META = {
    "id": KERNEL_ID,
    "title": "22 XGB Contiguous",
    "code_file": NOTEBOOK_NAME,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": ["alexhuaracha/04-preprocessing"],
    "competition_sources": [],
}

_cells: list = []
_nb = None


def _reset() -> None:
    global _cells, _nb
    _cells = []
    _nb = nbf.v4.new_notebook()


def md(text: str, cell_id: str) -> None:
    cell = nbf.v4.new_markdown_cell(text.strip())
    cell["id"] = cell_id
    _cells.append(cell)


def code(src: str, cell_id: str) -> None:
    cell = nbf.v4.new_code_cell(src.rstrip())
    cell["id"] = cell_id
    _cells.append(cell)


def embed_module(rel_path: str, header_md: str, slug: str) -> None:
    md(header_md, cell_id=f"cell-22-embed-{slug}-md")
    raw = (ROOT / "src" / rel_path).read_text(encoding="utf-8")
    code(_strip_relative_imports(raw), cell_id=f"cell-22-embed-{slug}")


def _all_digests() -> dict:
    """`{"E2|test|3": sha}` for every corridor x split x horizon."""
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"{MANIFEST_CSV} missing — run: uv run python -m src.build_sample_index"
        )
    manifest = pl.read_csv(MANIFEST_CSV)
    out = {}
    for name, _emp in CORRIDORS:
        for split in ("train", "val", "test"):
            for horizon in HORIZONS:
                row = manifest.filter(
                    (pl.col("corridor") == name)
                    & (pl.col("split") == split)
                    & (pl.col("horizon") == horizon)
                )
                if row.height != 1:
                    raise ValueError(f"manifest missing {name}/{split}/h{horizon}")
                out[f"{name}|{split}|{horizon}"] = row.row(0, named=True)["sha256"]
    return out


def build() -> None:
    _reset()

    md(
        """
# 22 — XGBoost sobre el índice compartido (auto-generado)

Generado por `build_notebook_22_xgb_contiguous.py`. **No editar a mano.**

Reajusta el baseline B5_XGB sobre **exactamente la misma población** que consume
el LSTM reentrenado (notebooks 21). La garantía no es un archivo compartido: cada
kernel recomputa el índice canónico desde el mismo parquet hash-pineado y
verifica que su SHA-256 coincida con `sample_index_manifest.csv`. Mismo código
más mismos bytes da el mismo índice.

Con `N_LAGS == T_IN == 12`, el XGBoost ve **las mismas doce observaciones** que
la red. Nivelar deja de ser un argumento y pasa a ser una propiedad verificable.

**Qué cambia respecto de NB10/NB16.** Los lags ya no se construyen con
`shift()` posicional sobre el slot —que nunca verificó contigüidad temporal— sino
leyéndolos de la ventana contigua. Y se retira la bandera de día atípico: es un
agregado del día completo y por lo tanto no se conoce al momento de predecir.

**La búsqueda de 24 configuraciones se conserva sin cambios** (`SEARCH_SEED`,
`SEARCH_SPACE`), seleccionada estrictamente sobre validación.
""",
        cell_id="cell-22-title",
    )

    code(
        f"""
import hashlib, time

import numpy as np
import polars as pl
import xgboost as xgb
from pathlib import Path

INPUT_HASHES = {json.dumps(PARQUET_HASHES, indent=4)}

INDEX_DIGESTS = {json.dumps(_all_digests(), indent=4)}

CORRIDORS = {json.dumps([[n, e] for n, e in CORRIDORS])}
HORIZONS = {json.dumps(HORIZONS)}

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _resolve_input(name: str) -> Path:
    roots = [Path("/kaggle/input"), Path(".")]
    candidates = [p for root in roots if root.exists() for p in sorted(root.rglob(name))]
    if not candidates:
        raise FileNotFoundError(f"Required input not found anywhere: {{name}}")
    for path in candidates:
        if _sha256_file(path) == INPUT_HASHES[name]:
            return path
    raise ValueError(
        f"No copy of {{name}} matches its frozen SHA-256 — "
        f"candidates: {{[str(p) for p in candidates]}}"
    )

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_OUT = OUTPUT_DIR / "xgb_contig_results.csv"
RESID_OUT = OUTPUT_DIR / "xgb_contig_residuals.csv"
SEARCH_OUT = OUTPUT_DIR / "xgb_contig_search_config.csv"
print(f"Output dir: {{OUTPUT_DIR}}")
""",
        cell_id="cell-22-setup",
    )

    for rel, header, slug in [
        ("evaluation/splits.py",
         "## Module: evaluation/splits\n\n`split_temporal` + `winsorize_train_p99` (umbral solo de train, aplicado a todos los splits).",
         "splits"),
        ("evaluation/metrics.py", "## Module: evaluation/metrics", "metrics"),
        ("data/windowing.py",
         "## Module: data/windowing\n\nEmbebido solo por `compute_max_N`. `make_window_index` NO se usa acá.",
         "windowing"),
        ("data/sample_index.py",
         "## Module: data/sample_index  (contratos C1 + C2)", "sampleindex"),
        ("data/context_features.py",
         "## Module: data/context_features\n\nEmbebido porque `fitted.py` lo referencia. La bandera atípica **no** se usa.",
         "context"),
        ("baselines/fitted.py",
         "## Module: baselines/fitted\n\n⚠️ Embebido **solo** por `sample_search_configs`, `SEARCH_SPACE`, "
         "`SEARCH_SEED` y `SEARCH_N_CONFIGS`. Su `_build_features` arrastra el defecto de "
         "lags posicionales y **no se llama** en este notebook.",
         "fitted"),
        ("baselines/contiguous_features.py",
         "## Module: baselines/contiguous_features\n\n`build_contiguous_features` — lags leídos de la ventana contigua.",
         "contigfeat"),
        ("evaluation/residual_export.py",
         "## Module: evaluation/residual_export\n\nClave completa + verificación de unicidad.",
         "residexport"),
    ]:
        embed_module(rel, header, slug)

    md(
        """## Preparación + portón de población compartida

Para cada corredor: split temporal, winsorización con umbral de train, y luego
—por horizonte— se reconstruye el índice canónico y se verifica su dígito. Si
alguno no coincide, la corrida se detiene: significa que esta corrida ya no es
comparable con la del LSTM.
""",
        cell_id="cell-22-prepare-md",
    )
    code(
        """
T_IN = DEFAULT_T_IN  # 12

def _index_digest(index: pl.DataFrame) -> str:
    canonical = index.select(
        ["empresaid", "direction", "start_ts", "target_ts", "horizon"]
    ).sort(["empresaid", "direction", "horizon", "start_ts"])
    payload = canonical.write_csv(datetime_format="%Y-%m-%dT%H:%M:%S")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

PREPARED, MAXN = {}, {}
for name, emp in CORRIDORS:
    raw = pl.read_parquet(_resolve_input(f"headways_{name}.parquet")).with_columns(
        pl.lit(emp, dtype=pl.Int64).alias("empresaid")
    )
    df_split = split_temporal(raw)
    df_winsor, threshold = winsorize_train_p99(df_split)
    PREPARED[name] = df_winsor
    MAXN[name] = compute_max_N(df_winsor.filter(pl.col("split") == "train"), quantile=0.99)
    print(f"{name}: {raw.height:,} rows, winsor threshold={threshold:.4f} min, max_N={MAXN[name]}")

INDEX = {}
for name, _emp in CORRIDORS:
    for split in ["train", "val", "test"]:
        part = PREPARED[name].filter(pl.col("split") == split)
        for horizon in HORIZONS:
            idx = make_sample_index(part, horizon=horizon, T_in=T_IN)
            got = _index_digest(idx)
            expected = INDEX_DIGESTS[f"{name}|{split}|{horizon}"]
            if got != expected:
                raise ValueError(
                    f"SHARED-POPULATION GATE FAILED for {name}/{split}/h{horizon}: "
                    f"{got} != frozen {expected}. This run is NOT comparable with the "
                    f"LSTM notebooks — stop and re-freeze the manifest."
                )
            INDEX[(name, split, horizon)] = idx
print("\\nShared-population gate: PASSED for every corridor x split x horizon.")
""",
        cell_id="cell-22-prepare",
    )

    md(
        """## Búsqueda de 24 configuraciones + ajuste final

Selección **estrictamente sobre validación**: el split de test nunca entra en
ninguna `DMatrix` de la búsqueda. La configuración ganadora por
`(corredor, horizonte)` queda registrada para que sea auditable.
""",
        cell_id="cell-22-fit-md",
    )
    code(
        """
def features_for(name, split, horizon):
    part = PREPARED[name].filter(pl.col("split") == split)
    return build_contiguous_features(
        part, INDEX[(name, split, horizon)],
        horizon=horizon, T_in=T_IN, max_N_by_direction=MAXN[name],
    )

rows, resid_frames, search_rows = [], [], []

for name, _emp in CORRIDORS:
    for horizon in HORIZONS:
        t0 = time.time()
        Xtr, ytr, _ = features_for(name, "train", horizon)
        Xva, yva, _ = features_for(name, "val", horizon)
        Xte, yte, kte = features_for(name, "test", horizon)

        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURE_NAMES)
        dva = xgb.DMatrix(Xva, label=yva, feature_names=FEATURE_NAMES)
        dte = xgb.DMatrix(Xte, label=yte, feature_names=FEATURE_NAMES)

        best = None
        for cfg in sample_search_configs(SEARCH_N_CONFIGS, seed=SEARCH_SEED):
            params = {"objective": "reg:squarederror", "seed": 42, "nthread": 4, **cfg}
            booster = xgb.train(
                params, dtr, num_boost_round=800, evals=[(dva, "val")],
                early_stopping_rounds=40, verbose_eval=False,
            )
            score = float(booster.best_score)
            if best is None or score < best[0]:
                best = (score, cfg, booster)

        val_rmse, cfg, booster = best
        pred = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))
        persist = kte.get_column("y_pred_persist").to_numpy()

        mae_x, rmse_x = mae(yte, pred), rmse(yte, pred)
        mae_p, rmse_p = mae(yte, persist), rmse(yte, persist)
        print(f"{name} h={horizon:2d}: n={len(yte):7,}  MAE_xgb={mae_x:.4f}  "
              f"MAE_pers={mae_p:.4f}  d={mae_x - mae_p:+.4f}  "
              f"val_rmse={val_rmse:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        for model_name, m_mae, m_rmse in [("B5_XGB_CONTIG", mae_x, rmse_x),
                                          ("B1_PERSIST", mae_p, rmse_p)]:
            for metric_name, value in [("MAE", m_mae), ("RMSE", m_rmse)]:
                rows.append({"corridor": name, "horizon": horizon, "baseline": model_name,
                             "metric": metric_name, "value": float(value), "n": len(yte)})

        search_rows.append({"corridor": name, "horizon": horizon, "val_rmse": val_rmse,
                            "best_iteration": booster.best_iteration,
                            "n_configs": SEARCH_N_CONFIGS, "search_seed": SEARCH_SEED,
                            **{f"param_{k}": v for k, v in cfg.items()}})

        n_rows = len(yte)
        resid_frames.append(pl.DataFrame({
            "corridor": [name] * n_rows,
            "direction": [f"+{d}" if d > 0 else str(d)
                          for d in kte.get_column("direction").to_list()],
            "horizon": [horizon] * n_rows,
            "split": ["test"] * n_rows,
            "start_ts": kte.get_column("start_ts"),
            "target_ts": kte.get_column("target_ts"),
            "pair_rank": kte.get_column("pair_rank"),
            "y_true": yte.astype("float64"),
            "y_pred_model": pred.astype("float64"),
            "y_pred_persist": persist.astype("float64"),
        }).select(RESIDUAL_COLUMNS))

results = pl.DataFrame(rows)
results.write_csv(RESULTS_OUT)
print(f"\\nResults: {RESULTS_OUT} ({results.height} rows)")
print(results)

search = pl.DataFrame(search_rows)
search.write_csv(SEARCH_OUT)
print(f"Search provenance: {SEARCH_OUT} ({search.height} rows)")

residuals = pl.concat(resid_frames)
assert_key_is_unique(residuals)
residuals.write_csv(RESID_OUT)
print(f"Residuals: {RESID_OUT} ({residuals.height:,} rows, key verified unique)")
""",
        cell_id="cell-22-fit",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _nb["cells"] = _cells
    _nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    (OUT_DIR / NOTEBOOK_NAME).write_text(nbf.writes(_nb), encoding="utf-8")
    print(f"Notebook written: {OUT_DIR / NOTEBOOK_NAME}  ({len(_cells)} cells)")

    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(KERNEL_META, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Kernel metadata written: {OUT_DIR / 'kernel-metadata.json'}")


if __name__ == "__main__":
    build()
