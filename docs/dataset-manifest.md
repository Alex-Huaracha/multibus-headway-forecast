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
| 2026-05-23 | Registro de artefactos de Fase 3 (NB05, DL-12 deferred); nota DL-4 splits en-kernel | (este commit) |
