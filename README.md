# multibus-headway-forecast

Pronóstico multi-horizonte de headways en corredores de transporte público urbano
con datos GPS reales del SIT Arequipa, comparando LSTM, SpatialConvLSTM y
SpatialTransformer contra baselines de persistencia, estadísticos y XGBoost.

**No se construyó ninguna GNN.** La propuesta original planteaba una arquitectura
GNN+LSTM; lo que se implementó fueron dos arquitecturas espaciales alternativas
(SpatialConvLSTM y SpatialTransformer), y ninguna supera al LSTM plano. Ese nulo
espacial es un resultado del trabajo, no un pendiente.

Publicación objetivo: IJACSA. La propuesta original se conserva en
[`docs/propuesta.md`](docs/propuesta.md) como registro de lo planificado; los
resultados y el alcance real están en
[`docs/resultados/documento-resultados.md`](docs/resultados/documento-resultados.md).

## Convenciones del proyecto

- **Clave compuesta**: siempre `(empresaid, unidadid)` — los `unidadid` se reutilizan entre empresas (34 de 150 aparecen en 3+ empresas). Nunca usar `unidadid` solo.
- **Corredores incluidos**: empresas **2, 4 y 59**. La propuesta declaraba también
  la 58, pero **E58 nunca entró al pipeline**: no tiene parquet procesado, no
  aparece en ningún resultado y ningún builder la referencia. El alcance real del
  trabajo son tres corredores. El resto de empresas se descartó por viabilidad
  (ver propuesta sección 4.3).
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

**2 — Descargar residuos desde Kaggle** (fuente de verdad del análisis)

Hay **dos** conjuntos de residuos y no son intercambiables:

| Conjunto | Familias | Para qué sirve |
|---|---|---|
| **Pipeline contiguo** (vigente) | `21-lstm-contiguous-{h1,h3,h5,h10}` y `-e4-{h1,h3,h5,h10}`, más `22-xgb-contiguous` | Todo verdicto del paper. Cumple los contratos C1/C2/C3: una muestra por `(empresa, sentido, start_ts, horizonte)`, ventanas de minutos consecutivos, sin features que filtren. |
| **Comparativa de arquitecturas** (congelada) | 11/12/13 en E2/E59, 17/18/19 en E4 | Solo el ranking LSTM vs SpatialConvLSTM vs SpatialTransformer, cuya validez descansa en que las tres comparten el mismo sesgo. **No usar para claims contra persistencia o XGBoost.** |

```bash
uv run kaggle kernels output alexhuaracha/21-lstm-contiguous-h3 \
  -p docs/resultados/residuos-multihorizon/21-lstm-contiguous/
uv run kaggle kernels output alexhuaracha/22-xgb-contiguous \
  -p docs/resultados/residuos-multihorizon/22-xgb-contiguous/
# … repetir por horizonte y grupo; runbook completo en docs/correr-kaggle.md
```

**3 — Regenerar tablas localmente** (no requiere GPU ni reentrenar)

```bash
uv run python -m src.build_sample_index                       # congela la población compartida
uv run python -m src.build_contiguous_significance            # DM/Wilcoxon con varianza por día
uv run python -m src.build_contiguous_paired_audit            # sesgo de encuadre (debe dar ~0)
uv run python -m src.build_contiguous_volatility              # estratificación ex-ante (lento, ~min)
uv run python -m src.build_contiguous_router                  # router + corte temporal + semillas
uv run python -m src.build_contiguous_vector_metrics          # métricas vectoriales
uv run python -m src.build_contiguous_winsorization_sensitivity  # robustez al techo p99
```

Leen `docs/resultados/residuos-multihorizon/` y escriben en
`docs/resultados/csv-multihorizon/`, fijando `POLARS_MAX_THREADS=1` para salidas
byte-idénticas. Varios **fallan cerrado** si la población no coincide con la
congelada: eso es intencional, no un bug.

Los builders `build_exante_*`, `build_volatility_*` y `build_router*` (sin
`contiguous`) corresponden a la comparativa congelada y se conservan por
reproducibilidad de esa tabla.

**4 — Lanzar una corrida en Kaggle** (opcional, requiere credenciales)

```bash
uv run python src/build_notebook_21_lstm_contiguous.py            # regenerar los 8 notebooks
uv run kaggle kernels push -p notebooks/21_lstm_contiguous/h3/    # subir versión + correr
```

Kaggle solo admite **2 sesiones GPU simultáneas**: lanzar de a dos. Si un kernel
falla con `no kernel image is available for execution on the device`, es un
desajuste de entorno (P100 vs T4×2) que se corrige desde la web, no desde el CLI
ni el builder.
