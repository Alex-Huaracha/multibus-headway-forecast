"""Generate the retrained-LSTM notebooks (contiguous pipeline).

Emits one notebook per (corridor group, horizon):

  notebooks/21_lstm_contiguous/e2e59/h{H}/  → alexhuaracha/21-lstm-contiguous-h{H}
  notebooks/21_lstm_contiguous/e4/h{H}/     → alexhuaracha/21-lstm-contiguous-e4-h{H}

Why a new builder instead of editing build_notebook_11 / _17
------------------------------------------------------------
Notebooks 11 and 17 produce the LSTM arm of the frozen architecture comparison
(against 12/13 and 18/19). That comparison stays valid precisely because all
three architectures share the same pipeline defect — so those builders must keep
emitting byte-comparable notebooks. Editing them would make the published null
result unreproducible from the repo. See ``docs/plan-reentrenamiento.md`` §4.

What changes relative to NB11/NB17
----------------------------------
1. **Population.** ``make_sample_index`` replaces ``make_window_index``: anchors
   are timestamps, windows are verified contiguous, and each target is emitted
   once instead of once per ``pair_rank`` slot (contracts C1 + C2).
2. **Shared-population gate.** The notebook recomputes the index digest and
   fails closed unless it matches the frozen ``sample_index_manifest.csv``. The
   XGBoost refit runs the same check, so both families provably consume the same
   samples rather than being reconciled by a post-hoc join (audit §2.1).
3. **Atypical flag removed.** ``CONTEXT_DIM`` drops 5 → 4 and
   ``atypical_days.csv`` leaves the input set entirely; the flag is a whole-day
   aggregate and therefore unknowable at prediction time (plan §2, C3). The
   ``02-eda-corridors`` kernel source is no longer needed.
4. **Full-key residual export.** ``build_keyed_residuals`` writes
   ``(corridor, direction, horizon, split, start_ts, target_ts, pair_rank)``
   alongside the values, so pending questions #5 and #6 become answerable from
   disk instead of requiring another GPU run.

Materialization lives in ``src/data/contiguous_dataset.materialize_arrays`` and
is embedded verbatim — the notebook does not carry its own untested copy, which
is how the generated artifacts drifted from their contracts before.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.splits import MAIN_FOLD, ROLLING_FOLDS  # noqa: E402
from src.notebook_utils import _strip_relative_imports  # noqa: E402

MANIFEST_CSV = ROOT / "docs" / "resultados" / "csv-multihorizon" / "sample_index_manifest.csv"

HORIZONS = [1, 3, 5, 10]

# Frozen SHA-256 of the parquets, mirroring docs/dataset-manifest.md:190-192.
PARQUET_HASHES = {
    "headways_E2.parquet": "82a34eaffc79cd82346d4595a2e72f5d3ffb751ed37fa0fc0cde3a8f8fb345d4",
    "headways_E59.parquet": "0b5f5593caaa94e4e6af7da672bc2cad7b49b69b7cbd0a22092f15700a89a448",
    "headways_E4.parquet": "1dde7f38eea9bc7d9941c17cbc3d326cb864e70be815a1a7e3d0ae2691f19273",
}

# Winning configs carried over from Fase 5 (docs/resultados/configuraciones-ganadoras.md).
# E4 never had a frozen winner, so it keeps its 3-config validation mini-grid.
GROUPS = {
    "e2e59": {
        "corridors": [("E2", 2), ("E59", 59)],
        "kernel_id": "21-lstm-contiguous-h{h}",
        "title": "21 LSTM Contiguous h{h}",
        "kernel_sources": [
            "alexhuaracha/04-preprocessing",
            "alexhuaracha/10-baselines-multi-horizonte",
        ],
        "baselines_csv": "baselines_results_multih.csv",
        "configs": {
            "E2": "[TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=5e-4)]",
            "E59": "[TrainConfig(hidden_size=32, num_layers=2, dropout=0.2, lr=5e-4)]",
        },
    },
    "e4": {
        "corridors": [("E4", 4)],
        "kernel_id": "21-lstm-contiguous-e4-h{h}",
        "title": "21 LSTM Contiguous E4 h{h}",
        "kernel_sources": ["alexhuaracha/04-preprocessing"],
        "baselines_csv": "baselines_E4_results_multih.csv",
        "configs": {
            "E4": (
                "[TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=5e-4),\n"
                "     TrainConfig(hidden_size=32, num_layers=2, dropout=0.2, lr=5e-4),\n"
                "     TrainConfig(hidden_size=64, num_layers=1, dropout=0.0, lr=5e-4)]"
            ),
        },
    },
}

_KERNEL_META_BASE = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "accelerator": "GPU_T4X2",
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
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


def embed_module(rel_path: str, header_md: str, cell_id_md: str, cell_id_code: str) -> None:
    md(header_md, cell_id=cell_id_md)
    raw = (ROOT / "src" / rel_path).read_text(encoding="utf-8")
    code(_strip_relative_imports(raw), cell_id=cell_id_code)


def _frozen_digests(corridors, horizon: int, fold: str = "main") -> dict:
    """Expected index digests per (corridor, split), read from the manifest.

    Injecting them at build time is what turns "same population" into something
    the kernel can verify: same code plus same input bytes must reproduce these
    exact digests, or the run stops before training.

    ``fold`` selects the evaluation origin. It is REQUIRED in the filter, not
    optional: the manifest holds one row set per fold, so matching on
    corridor/split/horizon alone now returns three rows, and a lookup that
    tolerated that would pick an arbitrary origin's digest.
    """
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"{MANIFEST_CSV} missing — run: uv run python -m src.build_sample_index"
        )
    manifest = pl.read_csv(MANIFEST_CSV)
    if "fold" not in manifest.columns:
        raise ValueError(
            f"{MANIFEST_CSV.name} predates rolling origin (no `fold` column) — "
            "regenerate it: uv run python -m src.build_sample_index"
        )
    out = {}
    for name, _emp in corridors:
        for split in ("train", "val", "test"):
            row = manifest.filter(
                (pl.col("fold") == fold)
                & (pl.col("corridor") == name)
                & (pl.col("split") == split)
                & (pl.col("horizon") == horizon)
            )
            if row.height != 1:
                raise ValueError(
                    f"manifest has {row.height} rows for "
                    f"{fold}/{name}/{split}/h{horizon}, expected exactly 1"
                )
            out[f"{name}|{split}"] = row.row(0, named=True)["sha256"]
    return out


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def _fold_note(fold) -> str:
    """Header paragraph describing the evaluation origin, when it is not the main one."""
    if fold.name == "main":
        return ""
    return f"""
> ### ⚠️ Origen de evaluación `{fold.name}` — corte de *rolling origin*
>
> Este notebook **no** produce el resultado publicado. Entrena y evalúa sobre un
> corte temporal distinto, para responder si el hallazgo se sostiene fuera de la
> ventana de febrero:
>
> | | Desde | Hasta | Días |
> |---|---|---|---|
> | Entrenamiento | {fold.train_start} | {fold.train_end} | {fold.train_days} |
> | Validación | {fold.val_start} | {fold.val_end} | {fold.val_days} |
> | Prueba | {fold.test_start} | {fold.test_end} | {fold.test_days} |
>
> Sus salidas llevan el sufijo `{fold.name}` para que no se confundan con las del
> corte publicado al descargarlas.
"""


def _add_title_cell(group_key: str, corridors, horizon: int, fold) -> None:
    names = " + ".join(n for n, _ in corridors)
    md(
        f"""
# 21 — LSTM sobre ventanas contiguas · {names} · h={horizon} · corte `{fold.name}`
{_fold_note(fold)}

Auto-generado por `build_notebook_21_lstm_contiguous.py`. **No editar a mano.**

Reentrena el LSTM sobre la población canónica definida en
`docs/plan-reentrenamiento.md`:

- **C1** — una muestra es `(empresaid, direction, start_ts, horizon)`; el ancla es
  un instante, no un índice de fila, y cada objetivo se emite **una sola vez**.
- **C2** — la ventana solo es válida si sus marcas de tiempo son minutos
  consecutivos, de modo que el objetivo cae exactamente a `h` minutos del final.
- **C3** — sin bandera de día atípico: es un agregado del día completo y por lo
  tanto no se conoce al momento de predecir.

El portón de dígitos verifica que el índice reconstruido acá coincida con
`sample_index_manifest.csv`. El refit del XGBoost corre el mismo portón, así que
ambas familias consumen las mismas muestras **por construcción**, no por un join
posterior.

Los notebooks 11/12/13 y 17/18/19 **no se tocan**: siguen sosteniendo la
comparación entre arquitecturas sobre el pipeline anterior.
""",
        cell_id="cell-21-title",
    )


def _add_setup_cell(group: dict, corridors, horizon: int, fold) -> None:
    hashes = {f"headways_{name}.parquet": PARQUET_HASHES[f"headways_{name}.parquet"]
              for name, _ in corridors}
    digests = _frozen_digests(corridors, horizon, fold.name)
    # Output names carry the fold for every origin except the published one,
    # whose filenames are already referenced by the download runbook and by the
    # analysis builders. Without the suffix, pulling r1's outputs into the
    # residual tree would overwrite the published residuals in place.
    stem = "lstm_contig" if fold.name == "main" else f"lstm_contig_{fold.name}"
    code(
        f"""
import hashlib

import polars as pl
import numpy as np
from pathlib import Path

# Frozen SHA-256 of every required training input. The run stops BEFORE training
# when a required file is missing or its bytes differ from the pinned snapshot.
# `atypical_days.csv` is deliberately absent: the flag was removed (plan C3).
INPUT_HASHES = {json.dumps(hashes, indent=4)}

# Frozen digests of the canonical sample index, from sample_index_manifest.csv.
# Recomputing them here is the shared-population gate.
INDEX_DIGESTS = {json.dumps(digests, indent=4)}

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

# Report-only comparison input (NOT a training input): outside the frozen gate.
def _find_baselines_csv() -> Path | None:
    name = "{group['baselines_csv']}"
    for root in (Path("/kaggle/input"), Path(".")):
        if root.exists():
            found = list(root.rglob(name))
            if found:
                return found[0]
    return None

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)
HORIZON = {horizon}
GROUP = "{group['baselines_csv'].split('_')[0]}"

# Evaluation origin. Resolved against the embedded `splits` module in the prepare
# cell — this cell runs before the embeds, so only the NAME can live here.
FOLD_NAME = "{fold.name}"

RESULTS_OUT = OUTPUT_DIR / f"{stem}_results_h{{HORIZON}}.csv"
RESID_OUT = OUTPUT_DIR / f"{stem}_residuals_h{{HORIZON}}.csv"

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"Output dir: {{OUTPUT_DIR}}")
print(f"Horizon:    {{HORIZON}}")
print(f"Fold:       {{FOLD_NAME}}")
print(f"Device:     {{DEVICE}}")
""",
        cell_id="cell-21-setup",
    )


def _add_embed_cells() -> None:
    for rel, header, slug in [
        ("evaluation/splits.py",
         "## Module: evaluation/splits\n\n`split_temporal` + `winsorize_train_p99` (train-only threshold, all splits).",
         "splits"),
        ("evaluation/metrics.py",
         "## Module: evaluation/metrics\n\n`mae` and `rmse` in minutes.",
         "metrics"),
        ("data/windowing.py",
         "## Module: data/windowing\n\nEmbedded for `compute_max_N` only. `make_window_index` is NOT used here — "
         "the retrained pipeline anchors on `make_sample_index` instead.",
         "windowing"),
        ("data/sample_index.py",
         "## Module: data/sample_index  (contracts C1 + C2)\n\n`make_sample_index` — timestamp-anchored, contiguity-checked, one row per target.",
         "sampleindex"),
        ("data/normalization.py",
         "## Module: data/normalization\n\nz-score stats from TRAIN ONLY; applied to every split.",
         "normalization"),
        ("data/context_features.py",
         "## Module: data/context_features\n\n`encode_context` still emits the 5-column set; the retrained pipeline "
         "consumes only the 4 causal ones (`CAUSAL_CONTEXT_FEATURE_NAMES`).",
         "context"),
        ("data/contiguous_dataset.py",
         "## Module: data/contiguous_dataset  (first torch import)\n\n`materialize_arrays` — the path this notebook takes, covered by "
         "`tests/data/test_contiguous_dataset.py` rather than reimplemented in a cell.",
         "dataset"),
        ("evaluation/residual_export.py",
         "## Module: evaluation/residual_export\n\n`build_keyed_residuals` + `assert_key_is_unique` — the full-key contract.",
         "residexport"),
        ("models/lstm.py",
         "## Module: models/lstm\n\n`HeadwayLSTM` + `masked_mse_loss`.",
         "lstm"),
        ("train.py",
         "## Module: train\n\n`TrainConfig`, `set_seed`, `train_model`, `grid_search`, `denormalize_predictions`.",
         "train"),
    ]:
        embed_module(rel, header, f"cell-21-embed-{slug}-md", f"cell-21-embed-{slug}")


def _add_context_dim_cell() -> None:
    md(
        """## `CONTEXT_DIM` 5 → 4

`train.py` dimensiona el modelo como `max_N + CONTEXT_DIM`. Al retirar la bandera
de día atípico el contexto pasa de 5 a 4 columnas, así que la constante se
rebindea **antes** de construir cualquier modelo. En el espacio de nombres plano
del notebook, `grid_search` y `train_model` leen esta misma global.
""",
        cell_id="cell-21-ctxdim-md",
    )
    code(
        """
CTX_COLS = list(CAUSAL_CONTEXT_FEATURE_NAMES)
assert "atypical_flag" not in CTX_COLS, "the leaking flag must not come back"

CONTEXT_DIM = len(CTX_COLS)
assert CONTEXT_DIM == 4, CONTEXT_DIM
print(f"CONTEXT_DIM rebound to {CONTEXT_DIM}: {CTX_COLS}")
""",
        cell_id="cell-21-ctxdim",
    )


def _add_prepare_cell(corridors) -> None:
    loads = "\n".join(
        f'RAW["{name}"] = pl.read_parquet(_resolve_input("headways_{name}.parquet"))'
        f'.with_columns(pl.lit({emp}, dtype=pl.Int64).alias("empresaid"))'
        for name, emp in corridors
    )
    md(
        """## Cargar, partir, winsorizar, normalizar, contexto

Idéntico al pipeline anterior salvo por el contexto: se codifican las 5 columnas
pero solo se consumen las 4 causales.

El corte temporal se resuelve por **nombre** contra el módulo `splits` embebido.
Un nombre desconocido levanta `KeyError` acá mismo, antes de tocar la GPU: caer
en silencio al corte publicado produciría resultados etiquetados como un origen
y calculados sobre otro.
""",
        cell_id="cell-21-prepare-md",
    )
    code(
        f"""
FOLD = fold_by_name(FOLD_NAME)
print(f"Fold {{FOLD.name}}: train {{FOLD.train_start}}..{{FOLD.train_end}} "
      f"({{FOLD.train_days}}d) | val {{FOLD.val_start}}..{{FOLD.val_end}} "
      f"({{FOLD.val_days}}d) | test {{FOLD.test_start}}..{{FOLD.test_end}} "
      f"({{FOLD.test_days}}d)")

RAW = {{}}
{loads}
for name, frame in RAW.items():
    print(f"{{name}}: {{frame.height:,}} rows")

PREPARED = {{}}
STATS = {{}}
for name, frame in RAW.items():
    df_split = split_temporal(frame, FOLD)
    df_winsor, threshold = winsorize_train_p99(df_split)
    stats = compute_normalization_stats(df_winsor.filter(pl.col("split") == "train"))
    df_z = apply_zscore(df_winsor, stats)
    # No atypical calendar is passed: the flag column is emitted as all-zero and
    # then never selected (CTX_COLS excludes it).
    PREPARED[name] = encode_context(df_z)
    STATS[name] = stats
    print(f"{{name}}: winsor threshold={{threshold:.4f}} min, "
          f"splits={{df_split.group_by('split').agg(pl.len()).sort('split').to_dicts()}}")
""",
        cell_id="cell-21-prepare",
    )


def _add_index_gate_cell(horizon: int) -> None:
    md(
        f"""## Índice canónico + portón de población compartida

Construye el índice para h={horizon} en cada split y **verifica su SHA-256**
contra `sample_index_manifest.csv`. Si no coincide, la corrida se detiene antes
de entrenar: significa que el código o los bytes de entrada cambiaron y que esta
corrida ya no es comparable con el refit del XGBoost.
""",
        cell_id="cell-21-index-md",
    )
    code(
        """
T_IN = DEFAULT_T_IN   # 12
BATCH_SIZE = 128

def _index_digest(index: pl.DataFrame) -> str:
    canonical = index.select(
        ["empresaid", "direction", "start_ts", "target_ts", "horizon"]
    ).sort(["empresaid", "direction", "horizon", "start_ts"])
    payload = canonical.write_csv(datetime_format="%Y-%m-%dT%H:%M:%S")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

INDEX = {}
for name, df in PREPARED.items():
    for split in ["train", "val", "test"]:
        part = df.filter(pl.col("split") == split)
        idx = make_sample_index(part, horizon=HORIZON, T_in=T_IN)
        digest = _index_digest(idx)
        expected = INDEX_DIGESTS[f"{name}|{split}"]
        if digest != expected:
            raise ValueError(
                f"SHARED-POPULATION GATE FAILED for {name}/{split}/h{HORIZON}: "
                f"digest {digest} != frozen {expected}. This run is NOT comparable "
                f"with the XGBoost refit — stop and re-freeze the manifest."
            )
        INDEX[(name, split)] = idx
        print(f"  {name}/{split}: {idx.height:,} samples  digest OK")
print("\\nShared-population gate: PASSED for every split.")
""",
        cell_id="cell-21-index",
    )


def _add_materialize_cell() -> None:
    md(
        """## Materialización

`materialize_arrays` vive en `data/contiguous_dataset` y está cubierto por tests
que verifican que coincide muestra por muestra con el `Dataset`. El notebook no
lleva su propia copia.
""",
        cell_id="cell-21-materialize-md",
    )
    code(
        """
import torch
import time as _time

MAX_N = {}
BATCHES = {}
TEST_ARRAYS = {}

def _to_batches(arrays: dict) -> list:
    n = arrays["input"].shape[0]
    tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
    return [
        {k: v[s:min(s + BATCH_SIZE, n)] for k, v in tensors.items()}
        for s in range(0, n, BATCH_SIZE)
    ]

for name, df in PREPARED.items():
    max_n_by_dir = compute_max_N(df.filter(pl.col("split") == "train"), quantile=0.99)
    global_max_N = max(max_n_by_dir.values())
    MAX_N[name] = global_max_N
    print(f"\\n{name}: max_N per direction={max_n_by_dir}, global={global_max_N}")

    loaders = {}
    for split in ["train", "val"]:
        t0 = _time.time()
        arrays = materialize_arrays(
            df.filter(pl.col("split") == split), INDEX[(name, split)],
            max_N=global_max_N, T_in=T_IN, horizon=HORIZON, context_cols=tuple(CTX_COLS),
        )
        loaders[split] = _to_batches(arrays)
        print(f"  {name} {split}: {arrays['input'].shape[0]:,} samples -> "
              f"{len(loaders[split]):,} batches in {_time.time()-t0:.1f}s")
    BATCHES[name] = loaders

    # Test is kept per-direction: denormalization uses direction-specific stats.
    test_df = df.filter(pl.col("split") == "test")
    per_dir = {}
    for direction in [-1, 1]:
        sub_idx = INDEX[(name, "test")].filter(pl.col("direction") == direction)
        arrays = materialize_arrays(
            test_df.filter(pl.col("direction") == direction), sub_idx,
            max_N=global_max_N, T_in=T_IN, horizon=HORIZON, context_cols=tuple(CTX_COLS),
        )
        per_dir[direction] = (sub_idx, arrays, _to_batches(arrays))
        print(f"  {name} test dir={direction:+d}: {arrays['input'].shape[0]:,} samples")
    TEST_ARRAYS[name] = per_dir
""",
        cell_id="cell-21-materialize",
    )


def _add_train_cell(group: dict) -> None:
    config_lines = "\n".join(
        f'    "{name}": {cfg},' for name, cfg in group["configs"].items()
    )
    md(
        """## Entrenamiento

Se reusa la configuración ganadora congelada por corredor. E4 nunca tuvo una
ganadora congelada, así que conserva su mini-grid de 3 configuraciones
seleccionadas **solo sobre validación**.
""",
        cell_id="cell-21-train-md",
    )
    code(
        f"""
CONFIGS = {{
{config_lines}
}}

RESULTS = {{}}
for name, loaders in BATCHES.items():
    print(f"\\n{{name}} training: max_N={{MAX_N[name]}}, device={{DEVICE}}")
    res = grid_search(
        train_dl=loaders["train"],
        val_dl=loaders["val"],
        max_N=MAX_N[name],
        configs=CONFIGS[name],
        device=DEVICE,
    )
    best = res[0]
    print(f"  {{name}} winner: hidden={{best.config.hidden_size}}, "
          f"layers={{best.config.num_layers}}, dropout={{best.config.dropout}}, "
          f"lr={{best.config.lr}}, val_loss={{best.best_val_loss:.6f}} "
          f"(epoch {{best.best_epoch}}, of {{len(res)}} configs)")
    RESULTS[name] = res
""",
        cell_id="cell-21-train",
    )


def _add_evaluate_cell() -> None:
    md(
        """## Evaluación + residuos con clave completa

La persistencia (B1) es el último paso observado de la ventana, así que se
compara contra **exactamente las mismas muestras** — pareado por construcción.
Los residuos salen con la clave completa, de modo que agrupar por día de servicio
o construir un perfil por posición del vector ya no exige otra corrida.
""",
        cell_id="cell-21-evaluate-md",
    )
    code(
        """
def evaluate_corridor(name):
    best = RESULTS[name][0]
    stats = STATS[name]
    empresa_id = int(list(stats.means.keys())[0][0])
    max_N = MAX_N[name]

    model = HeadwayLSTM(
        input_size=max_N + CONTEXT_DIM,
        hidden_size=best.config.hidden_size,
        output_size=max_N,
        num_layers=best.config.num_layers,
        dropout=best.config.dropout,
    )
    model.load_state_dict(best.state_dict)
    model.eval()
    model.to(torch.device(DEVICE))

    rows, resid_frames = [], []
    pooled = {"pred": [], "true": []}

    for direction in [-1, 1]:
        sub_idx, arrays, batches = TEST_ARRAYS[name][direction]
        mean_val = stats.means[(empresa_id, direction)]
        std_val = stats.stds[(empresa_id, direction)]

        preds = []
        with torch.no_grad():
            for batch in batches:
                x = torch.cat([batch["input"].to(DEVICE), batch["context"].to(DEVICE)], dim=-1)
                preds.append(
                    denormalize_predictions(model(x), mean_val, std_val).cpu().numpy()
                )
        pred_min = np.concatenate(preds) if preds else np.zeros((0, max_N), dtype=np.float32)

        target_min = denormalize_predictions(
            torch.from_numpy(arrays["target"][:, 0]), mean_val, std_val
        ).numpy()
        persist_min = denormalize_predictions(
            torch.from_numpy(arrays["input"][:, T_IN - 1, :]), mean_val, std_val
        ).numpy()
        tmask = arrays["target_mask"][:, 0]
        pmask = arrays["input_mask"][:, T_IN - 1, :]

        valid = tmask
        mae_val = mae(target_min[valid], pred_min[valid])
        rmse_val = rmse(target_min[valid], pred_min[valid])
        print(f"{name} dir={direction:+d}: MAE={mae_val:.4f} RMSE={rmse_val:.4f} "
              f"(n_valid={int(valid.sum()):,})")
        pooled["pred"].append(pred_min[valid])
        pooled["true"].append(target_min[valid])

        dir_str = f"+{direction}" if direction > 0 else str(direction)
        for metric_name, metric_val in [("MAE", mae_val), ("RMSE", rmse_val)]:
            rows.append({"corridor": name, "direction": dir_str, "baseline": "LSTM_CONTIG",
                         "metric": metric_name, "value": float(metric_val), "horizon": HORIZON})

        resid_frames.append(build_keyed_residuals(
            sub_idx, corridor=name, split="test",
            y_true=target_min, y_pred_model=pred_min, y_pred_persist=persist_min,
            target_mask=tmask, persist_mask=pmask,
        ))

    all_pred = np.concatenate(pooled["pred"])
    all_true = np.concatenate(pooled["true"])
    agg_mae, agg_rmse = mae(all_true, all_pred), rmse(all_true, all_pred)
    print(f"{name} aggregate: MAE={agg_mae:.4f} RMSE={agg_rmse:.4f} (n={len(all_pred):,})")
    for metric_name, metric_val in [("MAE", agg_mae), ("RMSE", agg_rmse)]:
        rows.append({"corridor": name, "direction": "aggregate", "baseline": "LSTM_CONTIG",
                     "metric": metric_name, "value": float(metric_val), "horizon": HORIZON})

    return rows, pl.concat(resid_frames)

all_rows, all_resid = [], []
for name in PREPARED:
    r, res = evaluate_corridor(name)
    all_rows.extend(r)
    all_resid.append(res)

results = pl.DataFrame(all_rows)
results.write_csv(RESULTS_OUT)
print(f"\\nResults written: {RESULTS_OUT} ({results.height} rows)")
print(results)

residuals = pl.concat(all_resid)
assert_key_is_unique(residuals)
residuals.write_csv(RESID_OUT)
print(f"Residuals written: {RESID_OUT} ({residuals.height:,} rows, key verified unique)")
""",
        cell_id="cell-21-evaluate",
    )


def _add_compare_cell() -> None:
    md(
        """## Comparación con los baselines del pipeline anterior

⚠️ **Lectura con cuidado.** Los baselines vienen del pipeline **anterior**, con
otra población. La comparación es orientativa; la cifra canónica sale del refit
del XGBoost sobre el índice compartido.
""",
        cell_id="cell-21-compare-md",
    )
    code(
        """
baselines_csv = _find_baselines_csv()
if baselines_csv is not None:
    baselines = pl.read_csv(baselines_csv).filter(pl.col("horizon") == HORIZON)
    print(f"Baselines (OLD population) from {baselines_csv}: {baselines.height} rows")
    print(pl.concat([
        baselines.filter(pl.col("metric") == "MAE"),
        results.filter(pl.col("metric") == "MAE"),
    ]).sort(["corridor", "direction", "baseline"]))
    print("\\nNOTE: baselines above ran on the OLD population — orientation only.")
else:
    print("Baselines CSV not found — skipping the orientation comparison.")
""",
        cell_id="cell-21-compare",
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_notebook(group_key: str, horizon: int, fold=MAIN_FOLD) -> None:
    """Emit one notebook + kernel metadata for a (group, horizon, fold).

    The published fold keeps its exact paths and kernel ids; the rolling folds
    get their own directory level and their own kernel slug. Sharing either
    would mean a rolling run overwriting the published kernel on Kaggle.
    """
    group = GROUPS[group_key]
    corridors = group["corridors"]
    _reset()

    base = ROOT / "notebooks" / "21_lstm_contiguous" / group_key
    out_dir = base / f"h{horizon}" if fold.name == "main" else base / fold.name / f"h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if fold.name == "main" else f"_{fold.name}"
    filename = f"21_lstm_contiguous_{group_key}{suffix}_h{horizon}.ipynb"

    _add_title_cell(group_key, corridors, horizon, fold)
    _add_setup_cell(group, corridors, horizon, fold)
    _add_embed_cells()
    _add_context_dim_cell()
    _add_prepare_cell(corridors)
    _add_index_gate_cell(horizon)
    _add_materialize_cell()
    _add_train_cell(group)
    _add_evaluate_cell()
    _add_compare_cell()

    _nb["cells"] = _cells
    _nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    (out_dir / filename).write_text(nbf.writes(_nb), encoding="utf-8")
    print(f"Notebook written: {out_dir / filename}  ({len(_cells)} cells)")

    kernel_id = group["kernel_id"].format(h=horizon)
    title = group["title"].format(h=horizon)
    if fold.name != "main":
        # Slug and title carry the fold so the two never collide on Kaggle and so
        # the run list stays readable at a glance.
        kernel_id = f"{kernel_id}-{fold.name}"
        title = f"{title} [{fold.name}]"

    kernel_meta = {
        "id": f"alexhuaracha/{kernel_id}",
        "title": title,
        "code_file": filename,
        "kernel_sources": group["kernel_sources"],
        **_KERNEL_META_BASE,
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Kernel metadata written: {out_dir / 'kernel-metadata.json'}")


if __name__ == "__main__":
    # Every origin, published one included. ROLLING_FOLDS ends with MAIN_FOLD, so
    # the published notebooks are emitted exactly once and in their usual paths.
    for fold in ROLLING_FOLDS:
        for key in GROUPS:
            for h in HORIZONS:
                build_notebook(key, h, fold)
