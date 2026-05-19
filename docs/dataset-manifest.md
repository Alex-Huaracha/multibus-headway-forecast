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

## Artefactos derivados (placeholder — Fase 2 los producirá)

| Artefacto | Estado | Fase |
|---|---|---|
| `cleaned_gps_<empresa>.parquet` | no producido aún | Fase 2 paso 1 |
| `headways_<empresa>.parquet` | no producido aún | Fase 2 paso 2 |
| `splits/<empresa>/{train,val,test}.parquet` | no producido aún | Fase 3 |
| `atypical_days.csv` | producido por notebook 02 (Fase 1) — pendiente de re-correr en Kaggle con bug fix | Fase 1 → consumido por Fase 3 y Fase 7 |

Cuando Fase 2 cierre, cada artefacto se agrega arriba con su propio bloque "Pin Kaggle" y "Provenance".

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
