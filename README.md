# multibus-headway-forecast

Predicción del vector completo de headways en corredores de transporte público urbano usando GNN+LSTM, con datos GPS reales del SIT Arequipa.

Publicación objetivo: IJACSA. Propuesta completa en [`docs/propuesta.md`](docs/propuesta.md).

## Convenciones del proyecto

- **Clave compuesta**: siempre `(empresaid, unidadid)` — los `unidadid` se reutilizan entre empresas (34 de 150 aparecen en 3+ empresas). Nunca usar `unidadid` solo.
- **Corredores incluidos**: empresas 2, 4, 58, 59. El resto fue descartado por viabilidad (ver propuesta sección 4.3).
- **Formato de datos procesados**: Parquet. Nada de CSV en el pipeline interno.
- **Datos no van a Git** — viven en Kaggle Datasets (versionados allá) y localmente bajo `data/` (gitignored).

## Estructura

```
data/raw/         GPS crudo (gitignored, descarga local del dataset Kaggle)
data/processed/   Parquets limpios por corredor (gitignored)
notebooks/        Pipeline por fases (EDA, preprocesamiento, headways, baselines, modelos)
src/              Utilidades reusables entre notebooks
kaggle/           Metadata de notebooks y datasets Kaggle
docs/             Propuesta y notas
```

## Setup

Gestión de Python y dependencias con [`uv`](https://docs.astral.sh/uv/). Versión de Python pinneada en `.python-version` (3.12).

Instalar `uv` (una sola vez):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reproducir el entorno del proyecto:

```bash
uv sync                    # crea .venv con Python 3.12 + deps del lockfile
uv run jupyter lab         # abre notebooks
uv run python src/...      # corre cualquier script
uv add <paquete>            # agregar dependencias
```

Token de Kaggle en `~/.kaggle/access_token` (chmod 600). En Windows: `%USERPROFILE%\.kaggle\access_token`.

## Reproducción de resultados (recertificación)

El pipeline es **builders → Kaggle (GPU, entrenamiento) → análisis local**. Los notebooks
bajo `notebooks/` son **artefactos generados**: se emiten desde `src/build_notebook_*.py` y
nunca se editan a mano. Los datos crudos/procesados viven en Kaggle Datasets (pinneados en
[`docs/dataset-manifest.md`](docs/dataset-manifest.md)), no en Git.

**1 — Entorno**

```bash
uv sync                        # .venv con Python 3.12 + deps del lockfile
uv run pytest -q               # suite completa (~880 tests)
```

> ⚠️ Correr la suite completa **reescribe los notebooks generados** (los tests de
> `test_build_notebook_*` ejecutan los builders). Tras `uv run pytest`, revisá `git status`
> y revertí cambios no deseados con `git checkout -- notebooks/`.

**2 — Descargar residuos frescos desde Kaggle** (fuente de verdad del análisis)

Las 6 familias DL recertificadas (11/12/13 en E2/E59; 17/18/19 en E4) × h∈{1,3,5,10} se
re-corrieron con el pipeline corregido. Bajar sus outputs a `docs/resultados/recertificado/`
(runbook completo en [`docs/correr-kaggle.md`](docs/correr-kaggle.md)):

```bash
uv run kaggle kernels output alexhuaracha/11-lstm-multihorizon-h3 -p <destino>/
# … repetir por familia × horizonte; ver docs/dataset-manifest.md § Recertificación
```

**3 — Regenerar tablas/figuras localmente** (no requiere GPU ni reentrenar)

```bash
uv run python src/build_exante_volatility.py      # estratificación ex-ante (lento, ~min)
uv run python src/build_exante_correlation.py     # chequeo anti-circularidad
# … resto de builders de reporte (significancia, degradación, paired-audit) — Fase 10
```

Los builders de reporte apuntan a `docs/resultados/recertificado/` y fijan
`POLARS_MAX_THREADS=1` para salidas byte-idénticas.

**4 — Lanzar una corrida en Kaggle** (opcional, requiere credenciales)

```bash
uv run python src/build_notebook_11.py                          # regenerar notebook
uv run kaggle kernels push -p notebooks/11_lstm_multihorizon/h3/  # subir versión + correr
```
