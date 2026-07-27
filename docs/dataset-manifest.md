# Dataset Manifest

Pin autoritativo de los artefactos de datos que el pipeline consume. Cada vez que un dataset cambia de versión en Kaggle, este manifest se re-emite y el cambio se documenta abajo. Política aplicable: [`decisiones-limpieza-fase2.md`](./decisiones-limpieza-fase2.md) §1 y la política de reproducibilidad acordada para el proyecto (no regenerar Fase 0; el parquet existente ES la fuente de verdad).

---

## `clean_gps.parquet` — input canónico desde Fase 2 en adelante

### Pin Kaggle (autoritativo)

| Campo | Valor |
|---|---|
| Dataset ID | `alexhuaracha/multibus-headway-forecast-clean` |
| Internal dataset ID | `10461549` |
| Visibility | private |
| License | CC BY 4.0 |
| Last updated (UTC) | 2026-05-17 19:17:15 |
| File name | `clean_gps.parquet` |
| File size | 678,033,135 bytes (~647 MB) |
| Compression | zstd |

> Kaggle versiona inmutablemente: cualquier consumidor que descargue el dataset con el `lastUpdated` indicado obtiene los mismos bytes. Si `lastUpdated` cambia, este manifest queda invalidado y debe re-emitirse.

### Contenido

| Campo | Valor |
|---|---|
| Filas | 47,681,656 |
| Empresas | 2, 4, 58, 59 |
| Período cubierto | 2023-10-01 → 2024-02-29 (151 días) |
| Buses únicos `(empresaid, unidadid)` | 102 operacionales (120 en raw; 18 excluidas por estacionarios) |
| Schema | `id, direccion, excesoVelocidad, fueraDeRuta, lat, lon, time, velocidad, empresaid, unidadid` |

### Provenance

| Campo | Valor |
|---|---|
| Productor | `notebooks/01_viability_and_filter/01_viability_and_filter.ipynb` |
| Builder | `src/build_notebook_01.py` |
| Git commit de producción | `45a752b` — "feat(phase-0): complete viability analysis with corrected methodology" (2026-05-17) |
| Dataset fuente | `alexhuaracha/multibus-headway-forecast-raw` (832 MB, ~100M filas crudas, 12 empresas) |
| Metodología | filtro a empresas viables (2, 4, 58, 59) + dedup con clave compuesta `(empresaid, unidadid, time)` + exclusión de buses estacionarios |

### Cómo descargar

```bash
mkdir -p data/processed
uv run kaggle datasets download -d alexhuaracha/multibus-headway-forecast-clean -p data/processed/ --unzip
```

### SHA256

- **Estado**: no requerido para este proyecto.
- **Razón**: la ejecución del pipeline vive en Kaggle (ver convención del proyecto). No se mantiene copia local del parquet, por lo que el SHA no tiene contraparte que verificar. Kaggle versiona inmutablemente y eso ya garantiza la identidad del dataset.
- **Cuándo registrarlo igualmente**: si en algún momento alguien necesita una copia local persistente (backup off-Kaggle, archivado a largo plazo), ejecutar `sha256sum data/processed/clean_gps.parquet` y completar este campo entonces.

---

## `multibus-headway-forecast-raw` — fuente raw (referencia)

No se consume directamente desde Fase 2; lo registramos para trazabilidad.

| Campo | Valor |
|---|---|
| Dataset ID | `alexhuaracha/multibus-headway-forecast-raw` |
| Last updated (UTC) | 2026-05-07 16:20:03 |
| Tamaño | 832,517,408 bytes |
| Filas (dedup) | 98,968,817 |
| Empresas | 12 |
| Fuente original | Municipalidad Provincial de Arequipa — Sistema Integrado de Transporte (SIT) |
| Licencia | CC BY 4.0 |

---

## Artefactos derivados — Fase 2 (productos del NB04 v8)

Todos los artefactos derivados producidos por Fase 2 viven como outputs del Kaggle kernel `alexhuaracha/04-preprocessing` (versión vigente v8, ejecutada 2026-05-23). No se publican como Kaggle Dataset — el kernel versiona inmutablemente sus outputs, así que el pin autoritativo es la versión del kernel.

### Pin Kaggle (kernel productor)

| Campo | Valor |
|---|---|
| Kernel ID | `alexhuaracha/04-preprocessing` |
| Versión vigente | v8 |
| Última ejecución (UTC) | 2026-05-23 14:08 |
| Status | COMPLETE |
| Tiempo de corrida | ~36 min |
| dataset_sources | `alexhuaracha/multibus-headway-forecast-clean` |
| kernel_sources | — |
| Builder | `src/build_notebook_04.py` |
| Commit de producción | `8674a5e` — "chore(nb04): regenerate notebook after H7 sort-key fix" (2026-05-23) |

### Outputs del kernel — Fase 2 paso 1: GPS limpio por empresa

| Artefacto | Tamaño | Schema |
|---|---|---|
| `cleaned_gps_E2.parquet` | ~151 MB | R7 v4 — añade `s`, `lateral_m`, `direction`, `trip_id` sobre el schema crudo de `clean_gps.parquet` |
| `cleaned_gps_E59.parquet` | ~197 MB | R7 v4 — idem |

### Outputs del kernel — Fase 2 paso 2: Headways por empresa

| Artefacto | Tamaño | Rows válidos (delta_t_min not null) | Schema |
|---|---|---|---|
| `headways_E2.parquet` | ~64 MB | 1,009,284 (495,562 dir=-1 + 513,722 dir=+1) | `t, direction, pair_rank, bus_front, bus_back, s_front, s_back, speed_front_kmh, speed_back_kmh, delta_t_min, n_buses, lateral_m_front, lateral_m_back` |
| `headways_E59.parquet` | ~106 MB | 2,069,193 (1,155,295 dir=-1 + 913,898 dir=+1) | idem |

### Outputs del kernel — Fase 2 sidecar: diagnostics de NULL

| Artefacto | Tamaño | Filas | Propósito |
|---|---|---|---|
| `headway_null_buckets_E2.parquet` | ~2 KB | 12 (6 buckets × 2 directions) | Adjudicación AC-DATA-2 — counts por `(empresaid, direction, bucket)` |
| `headway_null_buckets_E59.parquet` | ~2 KB | 12 | idem |

### Otros artefactos

| Artefacto | Estado | Fase |
|---|---|---|
| `splits/<empresa>/{train,val,test}.parquet` | **no producido — excluido por DL-4** (splits se computan en-kernel en NB05; no se escriben parquets intermedios) | Fase 3 |
| `atypical_days.csv` | producido por kernel `alexhuaracha/02-eda-corridors` (Fase 1), última corrida 2026-05-19 | Fase 1 → consumido por Fase 3 y Fase 7 |
| `quality_gps.csv` | producido por kernel `02-eda-corridors`, contiene métricas de calidad GPS por empresa | Fase 1 |

### Cómo descargar (snapshot local)

```bash
mkdir -p /tmp/nb04_v8
uv run kaggle kernels output alexhuaracha/04-preprocessing -p /tmp/nb04_v8/
```

Los outputs descargados incluyen los 6 parquets listados arriba más el log de ejecución `04-preprocessing.log`.

---

## Artefactos derivados — Fase 3 (productos del NB05, pendiente ejecución Kaggle)

El kernel `alexhuaracha/05-dataset` construye el dataset supervisado sobre los outputs de NB04 v8. La ejecución en Kaggle está **diferida** (DL-12): el notebook está generado y los módulos están integrados al venv; el kernel se lanzará una vez que la cadena NB04 → NB05 se valide localmente.

### Pin Kaggle (kernel productor)

| Campo | Valor |
|---|---|
| Kernel ID | `alexhuaracha/05-dataset` |
| Versión vigente | **Pendiente de ejecución Kaggle (DL-12)** |
| kernel_sources | `alexhuaracha/04-preprocessing` |
| dataset_sources | — (vacío; datos llegan vía kernel_sources) |
| Builder | `src/build_notebook_05.py` |
| Commit de producción | `4bdd7cf` — "feat(builder): GREEN — build_notebook_05 with stable cell IDs and embedded modules" (2026-05-23) |

### Outputs esperados del kernel — Fase 3

> **Nota DL-4**: los splits train/val/test se computan íntegramente dentro del kernel a partir de `headways_E{2,59}.parquet`. **No se escriben archivos `splits/*.parquet`** en `/kaggle/working`. Esta es una decisión de diseño deliberada para evitar artefactos intermedios redundantes (la fuente de verdad es el kernel NB04 v8 más el código de NB05).

| Artefacto | Descripción | Estado |
|---|---|---|
| `dataset_stats.csv` | Métricas por `(corridor, direction, split)`: `n_rows`, `n_windows`, `max_N`, `mean_delta_t_min`, `std_delta_t_min`, `truncation_rate`. 7–9 columnas. | Pendiente de ejecución Kaggle (DL-12) |

El `HeadwayDataset` y los `DataLoader` son objetos Python en memoria; no se serializan a disco (no hay `*.pt` outputs). Los tensores se reconstruyen en cada kernel run a partir de los parquets de NB04.

### Splits (en-kernel, no en disco)

Los splits temporales se derivan en-kernel mediante `split_temporal` de `src/evaluation/splits.py`. Fechas de corte (bloqueadas en el código):

| Split | Período |
|---|---|
| train | 2023-10-01 → 2024-01-15 |
| val   | 2024-01-16 → 2024-02-07 |
| test  | 2024-02-08 → 2024-02-29 |

> No existen `splits/{train,val,test}.parquet` en disco ni en `/kaggle/working`. Si se necesitan extraer, volver a ejecutar NB05 y agregar un `write_parquet` ad-hoc dentro del kernel.

---

## Recertificación DL (Fase 9, re-corridas Kaggle 2026-07-15)

Las 6 familias de notebooks DL se re-ejecutaron en Kaggle con el pipeline corregido
(winsorización p99 aplicada a **todos** los splits + feature de día atípico **activa**
+ fix del loader `day`/`date`). El pin autoritativo de cada familia es la versión del
kernel productor; los outputs frescos ya son la fuente canónica bajo
`docs/resultados/residuos-multihorizon/` (residuos) y `docs/resultados/csv-multihorizon/`
(CSV chicos) — reemplazaron in situ a los de junio (pre-fix), preservados en el historial git.

### Inputs de entrenamiento congelados (verificados por hash, fail-closed)

Cada notebook DL fija el SHA-256 de cada input y aborta antes de entrenar si los bytes
no coinciden (`_resolve_input`; ver `tests/test_notebook_input_gate.py`). Los hashes son
independientes del punto de montaje: da igual de qué fuente Kaggle provenga el archivo,
si sus bytes no matchean, la corrida falla.

| Input | Familias | SHA-256 |
|---|---|---|
| `headways_E2.parquet` | 11/12/13 | `82a34eaffc79cd82346d4595a2e72f5d3ffb751ed37fa0fc0cde3a8f8fb345d4` |
| `headways_E59.parquet` | 11/12/13 | `0b5f5593caaa94e4e6af7da672bc2cad7b49b69b7cbd0a22092f15700a89a448` |
| `headways_E4.parquet` | 17/18/19 | `1dde7f38eea9bc7d9941c17cbc3d326cb864e70be815a1a7e3d0ae2691f19273` |
| `atypical_days.csv` | todas | `2054245cc830e58b9397b75ea3b55d034581046b64e73b1630ca7d464e3ecb86` |

Validación por-log de las 24 corridas: `Atypical days loaded: 17 dates`, umbral de
winsorización `delta_t_min` p99 = **28.4679** (E2) / **27.9969** (E59) / **29.0984** (E4),
sin traceback. El umbral idéntico entre modelos de un mismo corredor prueba que los inputs
congelados son byte-idénticos.

### Pins por familia (kernel productor)

| Familia | Arquitectura | Corredores | Kernel ID (por horizonte h∈{1,3,5,10}) | Fuente de datos |
|---|---|---|---|---|
| 11 | LSTM | E2, E59 | `alexhuaracha/11-lstm-multihorizon-h{H}` | `04-preprocessing` (headways) + `10-baselines-multi-horizonte` |
| 12 | SpatialConvLSTM | E2, E59 | `alexhuaracha/12-spatialconvlstm-multihorizon-h{H}` — **h10 usa el slug `-h10b`** (el `-h10` original quedó corrupto en Kaggle) | idem |
| 13 | SpatialTransformer | E2, E59 | `alexhuaracha/13-spatialtransformer-multihorizon-h{H}` | idem |
| 17 | LSTM | E4 | `alexhuaracha/17-e4-lstm-h{H}` | `16-e4-data-baselines` (headways **y** baselines E4) |
| 18 | SpatialConvLSTM | E4 | `alexhuaracha/18-e4-convlstm-h{H}` | idem |
| 19 | SpatialTransformer | E4 | `alexhuaracha/19-e4-transformer-h{H}` | idem |

> Los kernels `14-lstm-minigrid-h10` y `15-lstm-multiseed-h{1,3,5,10}` (estudios de
> sensibilidad por hiperparámetro y por seed) se **recertificaron** en una extensión de
> scope (2026-07-16): sus builders arrastraban el mismo bug de winsorización train-only y
> carecían del gate de hash; se corrigieron y re-corrieron con el pipeline validado (mismos
> umbrales E2 28.4679 / E59 27.9969, `17 dates`). Sus CSV frescos son `lstm_minigrid_h10.csv`
> y `lstm_multiseed_h{1,3,5,10}.csv`. El agregado `multiseed_ci_multihorizon.csv` **no** es
> output de kernel: se regenera localmente con `src/build_multiseed_table.py`.

### Desviación del montaje de `atypical_days.csv`

La `kernel-metadata.json` de cada notebook declara `alexhuaracha/02-eda-corridors` como
`kernel_source` (la **intención**). Pero las re-corridas montaron el CSV desde el **Kaggle
Dataset `alexhuaracha/atypical-days-frozen`** (hash-pinneado, agregado por única vez vía web
"Add Input"), porque un `push` por CLI no adjunta de forma confiable un `kernel_source`
nuevo nunca antes adjuntado. Como el gate verifica por hash, ambas rutas de montaje son
equivalentes: la única garantía real es que `atypical_days.csv` matchee `2054245c…`.

### Inventario de artefactos de salida (recertificado, canónico)

| Artefacto | Ruta | Git |
|---|---|---|
| Residuos por-muestra (`{lstm,…}_residuals_h{H}.csv`, E4 como `{…}_E4_residuals_h{H}.csv`) | `docs/resultados/residuos-multihorizon/<familia>/h{H}/` | **gitignored** (pesado) |
| Predicciones XGBoost por-muestra de test (`xgb_paired_persample_test.csv`, ~208 MB, 2 248 396 filas) | `docs/resultados/residuos-multihorizon/20-xgb-paired/` | **gitignored** (pesado) |
| CSV de resultados/análisis (`*_results_h{H}.csv`, `exante_*_multihorizon.csv`, `xgb_paired_*.csv`) | `docs/resultados/csv-multihorizon/` | trackeado |

Los residuos son la fuente de verdad para el análisis local (Fases 5 y 10): schema
`corridor,direction,horizon,y_true,y_pred_dl,y_pred_persist` en todos los horizontes.

#### Export por-muestra de XGBoost (`20-xgb-paired`)

Salida del kernel `alexhuaracha/20-xgb-paired-export` (builder
`src/build_notebook_20_xgb_paired.py`, módulo `src/baselines/paired_export.py`).
Schema: `corridor,empresaid,direction,horizon,t,pair_rank,y_true,y_pred_xgb,y_pred_persist`.

- **Por qué no está versionado**: 208 MB. `docs/resultados/csv-multihorizon/` es la
  única carpeta de resultados trackeada y está reservada para CSV chicos de métricas;
  este archivo va al árbol gitignored de residuos, junto al resto de las salidas
  por-muestra.
- **Por qué el nombre no contiene `_residuals_`**: los globs
  `**/*_residuals_*.csv` (`src/build_significance_table.py`) y `*_residuals_h*.csv`
  (`src/evaluation/paired_audit.py`) recorren ese mismo árbol y cargarían este schema
  incompatible.
- **Para qué sirve**: `pair_rank` completa la clave única
  `(direction, t, pair_rank)` de la tabla de headways, lo que permite re-puntuar
  XGBoost exactamente sobre las filas evaluadas por el DL
  (`src/build_xgb_paired_metrics.py` → `xgb_paired_dl_metrics.csv`,
  `xgb_paired_vs_reported_audit.csv`, `xgb_paired_significance.csv`). Los residuos
  de `harness.XGB_RESIDUAL_COLUMNS` no sirven: descartan `pair_rank`.
- **Reproducibilidad**: un re-export re-ajusta B5_XGB. Es seguro por diseño —
  `fitted.py` fija la semilla de búsqueda, la de entrenamiento y `nthread` — y la
  corrida verificada reprodujo las 12 celdas de `xgb_search_config_multih.csv` /
  `xgb_search_config_E4_multih.csv` en todas las columnas a precisión float completa.

### Nota de sensibilidad: clipping del test (winsorización)

**Contrato aplicado en la recertificación.** El umbral p99 de `delta_t_min` se calcula
**solo sobre train** y se aplica a **todos** los splits, incluido el **ground-truth de test**
(`winsorize_train_p99` en `src/evaluation/splits.py`; guardado por
`tests/test_preprocessing_winsorization_contract.py`). Esto es lo correcto para la
comparación pareada: el DL y la persistencia se evalúan sobre exactamente los mismos targets
clipeados, así que ningún modelo obtiene ventaja por el techo. El umbral efectivo es
~28–29 min por corredor (28.4679 E2 / 27.9969 E59 / 29.0984 E4).

**Implicación de sensibilidad.** Winsorizar el ground-truth de test **acota los headways
reales extremos** al techo de train (~28–29 min). Como la ventaja del DL se concentra en el
régimen de alta volatilidad (headways que dan saltos grandes), el clipping toca justamente la
cola donde más se juega la comparación. El efecto neto sobre la brecha DL-vs-persistencia en
esa cola **no está medido** en esta recertificación —es precisamente lo que el chequeo
no-clipping cuantificaría—; lo único garantizado por el contrato es que ambos modelos se
evalúan sobre los **mismos** targets clipeados, sin ventaja para ninguno.

**Plan no-clipping (chequeo planeado, no ejecutado este ciclo).** Un variante que **no**
clipee el ground-truth de test —reportando la comparación pareada sobre targets crudos— queda
**documentada como chequeo de sensibilidad planeado** para una futura re-corrida Kaggle, no
implementada en esta recertificación (confirmado en `proposal.md`). Mediría cuánto y en qué
dirección cambia la brecha en la cola al remover el techo; hasta correrlo, el resultado queda
abierto.

---

## Política de pinning

1. La versión Kaggle (`lastUpdated`) es el **pin autoritativo**. Si cambia, el manifest se re-emite.
2. El SHA256 es **complementario** — verificación de integridad de la copia local, no de identidad del dataset.
3. Cualquier cambio metodológico en Fase 0 (criterios de viabilidad, dedup, filtros) obliga a:
   - Nueva versión del Kaggle Dataset `multibus-headway-forecast-clean`.
   - Nuevo commit del builder `src/build_notebook_01.py`.
   - Nueva entrada en este manifest reemplazando la vigente.
   - El manifest anterior se preserva en historia git pero NO en este archivo.

---

## Historial de revisiones

| Fecha | Cambio | Commit |
|---|---|---|
| 2026-05-19 | Manifest inicial al cierre de Fase 1 | (pendiente) |
| 2026-05-23 | Registro de artefactos derivados de Fase 2 (NB04 v8 post H7 fix) | `86c4834` |
| 2026-05-23 | Registro de artefactos de Fase 3 (NB05, DL-12 deferred); nota DL-4 splits en-kernel | `86c4834` |
| 2026-07-16 | Sección de recertificación DL (Fase 9): hashes congelados, pins por familia, desviación del montaje atypical, inventario de salidas recertificadas | (este commit) |
| 2026-07-26 | Registro del export por-muestra de XGBoost (`20-xgb-paired`, no versionado) y de los CSV `xgb_paired_*` derivados | (pendiente) |
