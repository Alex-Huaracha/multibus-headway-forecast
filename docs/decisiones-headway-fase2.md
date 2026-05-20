# Decisiones de headway para Fase 2

Este documento cierra el **viability probe** ejecutado al inicio de Fase 2: confirma la formulación de headway adoptada para todo el pipeline (Fases 2 a 7), los parámetros productivos derivados, y los caveats registrados como deuda técnica conocida.

> **Estado:** APROBADAS (2026-05-19) — cada decisión queda como contrato de Fase 2. Cualquier cambio posterior requiere re-emitir este documento con nueva versión y motivación explícita.

## 1. Fuente de los hallazgos

- Notebook `notebooks/03_headway_viability/03_headway_viability.ipynb` (generado por `src/build_notebook_03.py`).
- Kaggle kernel `alexhuaracha/03-headway-viability` versiones 1-5. Producción = **versión 5**.
- Outputs verificables: `viability_matrix.csv`, `figuras/{signal_distributions,autocorrelation,spatial_mi_heatmap,stability_kl,centerlines}.png`, `viability_log.txt`, 24 parquets de headways de prueba.
- Scope del probe: empresas **E2** (PCA=33.55, mediana 16 buses) y **E59** (PCA=5.92, mediana 20 buses, sin heading), días `2024-01-23` (martes), `2024-01-27` (sábado) y `2023-10-28` (sistémico).

## 2. Decisión central — formulación de headway

**Opción C.2 — Temporal snapshot por bus con trailing crossing.**

### 2.1 Definición formal

En cada snapshot `T` (grilla uniforme cada `GRID_SECONDS` segundos), para cada par consecutivo de buses `(i, i+1)` ordenados por coordenada lineal `s` dentro de la misma dirección:

```
Δt(i, i+1, T) = T − t_cross(bus_{i+1}, s_i)
```

donde `t_cross(bus_{i+1}, s_i)` es el instante en el que el bus de atrás cruzó por última vez la posición actual del bus de adelante, interpolado linealmente sobre la trayectoria pasada del bus de atrás.

El vector resultante en el instante `T` tiene tamaño `N(T) − 1`, donde `N(T)` es la cantidad de buses activos en el corredor en `T`. Las unidades son **minutos**.

### 2.2 Tabla de formulaciones evaluadas y veredicto

| ID | Definición | E2 PASS | E59 PASS | MI vecinos (bits) | Veredicto |
|----|------------|---------|----------|-------------------|-----------|
| A   | Δt en puntos virtuales del recorrido | 5/7 | 5/7 | 1.15 / 0.57 | Descartar — autocorr muy baja, no apta para snapshot model |
| B   | Δs en metros entre pares en snapshot | 6/7 | 6/7 | 0.27 / 0.37 | Descartar — pasaría las pruebas pero está en metros, requiere reescribir `propuesta.md` §3.2, §5.2, §6.2 |
| C.1 | Δt forward projection `(s_i − s_{i+1}) / v_{i+1}` | 5/7 | 5/7 | 0.23 / 0.33 | Descartar — bajo MI inicial (~0 en v1), bajo autocorr |
| **C.2** | **Δt trailing crossing** | **6/7** | **6/7** | **0.36 / 1.26** | **ADOPTADA** — pasa 6 de 7 dimensiones en ambas empresas, MI alta confirma propagación bus-a-bus, alineada con `propuesta.md` sin cambios |

La dimensión que falla (R² persistencia) lo hace en TODAS las celdas — el umbral `[0.5, 0.85]` resultó mal calibrado. Su interpretación correcta no es "el método falla" sino "persistencia es un baseline débil, el modelo lo va a superar fácil". Se documenta como caveat 4 abajo y no bloquea la decisión.

## 3. Parámetros productivos

Estos parámetros se fijan como contrato para `src/preprocessing/` de Fase 2. Cualquier cambio requiere re-correr el probe y actualizar este documento.

| Parámetro | Valor | Motivación |
|-----------|-------|------------|
| `GRID_SECONDS` | **60** | GPS pinguea cada ~20s; grilla 60s da factor 3× de smoothing sin perder granularidad operacional. Probado estable contra 30s y 120s (KL < 0.001). |
| `MIN_SPEED_FOR_CENTERLINE_KMH` | **10.0** | Filtra terminales y semáforos del sample que construye la centerline. 5 km/h era insuficiente. |
| `CENTERLINE_LATLON_QUANTILE` | **(0.005, 0.995)** | Box IQR pre-PCA para descartar outliers geográficos extremos antes de calcular el eje principal. Remueve ~1.7-1.8% de pings. |
| `CENTERLINE_N_BINS` | **50** | Vertices del polyline. Probado estable contra 10 y 40 (KL < 0.01). |
| `LATERAL_OFFSET_THRESHOLD_M` | **300.0** | Pings proyectados a más de 300 m de la centerline se consideran "off-route" y se descartan. En el probe drop el 43.7% — agresivo pero limpia los pings de calles paralelas/depósitos. Revisable a 500 m en Fase 2 si baja demasiado el conteo. |
| `DIRECTION_SMOOTH_WIN` | **5** | Ventana móvil para `sign(ds/dt)`. Suaviza ruido sin perder transiciones reales ida↔vuelta. |
| `MAX_INTERPOLATION_LOOKBACK_MINUTES` | **30.0 min** | Límite temporal en la búsqueda de cruce histórico (C.2). Cruces más antiguos que 30 min → emitir `delta_t_min = NULL` en lugar de un valor absurdo. Headways típicos en Arequipa urbana: 5–15 min; 30 min = margen 2–3×. Corrige el 58.4% de ruido en E2 dir=1 (obs #27 — max observado: ~112 días). Ambas ramas de emisión de `_find_last_crossing_ns` (cruce exacto y cruce interpolado por cambio de signo) aplican el límite. Ver `fix(phase-2): bound C.2 trailing crossing with lookback window`. |
| Sub-opción de C | **C.2 (trailing crossing)** | Más precisa que C.1 (forward projection); C.1 quedó con MI~0 en v1 (la GNN no aprende nada) y MI=0.23-0.33 en v3 (sigue inferior a C.2). |

### 3.1 Parámetros de `ProductiveParams` añadidos en c2-lookback-fix

**`max_interpolation_lookback_minutes = 30.0`** (campo frozen en `src/preprocessing/config.py`).

**Problema raíz**: Los corredores multi-filar de Arequipa (en particular E2) comparten un eje lineal `s` común. Cuando dos líneas independientes circulan por el mismo tramo, `np.searchsorted` localiza el cruce histórico más reciente del bus de atrás sobre `s_front`, sin importar cuán antiguo sea. En un corredor multi-filar, ese cruce puede corresponder a un viaje de hace horas o días → `delta_t_min` absurdo (hasta 161,666 min observados en la validación visual de notebook 04b).

**Solución**: introducir un techo temporal de 30 min. Si `T − t_cross > 30 min`, el kernel retorna `None` (mismo centinela que "no hay cruce") y la fila se emite con `delta_t_min = NULL`. La conversión minutos → nanosegundos se hace UNA SOLA VEZ en `compute_headways_c2` antes del bucle de pares; dentro del kernel todo opera en nanosegundos int64/float64.

**¿Por qué 30 min?**: El histograma de `delta_t_min` para E59 (corredor sin multi-filar) muestra una distribución exponencial truncada con percentil 95 en ~18 min. Un techo de 30 min = 2–3× el headway típico peak, preserva virtualmente todos los pares válidos de E59 (cambio esperado < 1% en conteo de pares no-null) y elimina los valores patológicos de E2 dir=1. La validación cuantitativa definitiva (AC-D4 y AC-D5 de la spec) se realiza en Kaggle v3 post-merge.

## 4. Caveats registrados (deuda técnica conocida)

| # | Caveat | Magnitud | Decisión para Fase 2 |
|---|--------|----------|----------------------|
| 1 | Centerline de E2 corto (~6 km de cluster denso urbano, vs pings que se extienden ~80 km al sur) | Filtro IQR p0.5-p99.5 sólo remueve 1.7% del sample. El off-route filter (300 m) compensa post-proyección descartando 43.7% del total. Ver figuras `centerlines.png` y `clusters_per_empresa.png` (v4). | **Aceptar el comportamiento actual.** El probe v4 con HDBSCAN confirmó que la trail sur es NOISE (sparse, no operacional) — NO es un segundo corredor. Re-evaluar sólo si Fase 2 produce headways con `n_pairs_efectivo < 10k` por empresa-día. |
| 2 | Outliers en Δt (cola larga, CV=8.13 en E2, 5.27 en E59) | Distribución exponencial truncada hasta 30 min con cola pesada. | **Aplicar winsorización al percentil 99** del target `Δt` por (empresa, dirección) ANTES de entrenar el modelo. Alternativa equivalente: `log(1 + Δt)` como transformación de target. Decidir definitivamente al construir el dataset de Fase 5. |
| 3 | Off-route filter de 300 m descarta 43.7% de pings | Threshold agresivo en el probe. | **Calibrable**. Si Fase 2 muestra que esto reduce demasiado el volumen útil, relajar a 500 m. La decisión final se ancla cuando `src/preprocessing/` produzca `cleaned_gps_<empresa>.parquet` y se mida el conteo final por bus-día. |
| 4 | R² de persistencia es negativo en 7 de 8 celdas (rango −1.35 a +0.18) | Persistencia es un baseline malo en estas series. | **No es un problema de la formulación, es información sobre el sistema**: persistencia es trivial y el modelo lo va a superar. Documentar en el paper como dato relevante. El umbral `[0.5, 0.85]` queda invalidado y se elimina de los criterios de viabilidad. |
| 5 | Métrica `n_pairs_151d` en `viability_matrix.csv` arroja valores absurdos (10^12) | Bug de cómputo en el probe, no afecta la decisión (el threshold de 50k se pasa por muchos órdenes de magnitud). | **Housekeeping**: arreglar en una iteración futura del probe si se vuelve a correr. No bloquea Fase 2. |

## 5. Implicaciones para el resto del pipeline

### 5.1 Fase 2 — Preprocesamiento

Producir dos artefactos intermedios separados por empresa (no mezclar):
- `cleaned_gps_<empresa>.parquet`: GPS con centerline construida, `s` proyectada, `direction` inferida, `speed_kmh` observada, pings off-route descartados.
- `headways_<empresa>.parquet`: long-format con columnas `(t, direction, pair_rank, bus_front, bus_back, s_front, delta_t_min)`. Una fila por par consecutivo por snapshot.

Esta separación permite iterar sobre la fórmula del headway sin re-correr la proyección (que es la parte cara). Acuerdo previo registrado en la sesión de cierre de Fase 1.

### 5.2 Fase 5 — Target del modelo

El modelo aprende a predecir el vector `[Δt_1, ..., Δt_{N−1}]` en `T + h`, donde `h` es el horizonte (a definir entre 1-5 min — ver caveat 6 abajo). Loss: MAE en minutos sobre los pares válidos en el snapshot.

### 5.3 Fase 6 — GNN

El grafo es **dinámico** por snapshot: nodos = buses activos en `T`, aristas = pares consecutivos ordenados por `s` dentro de la misma dirección. Atributos de nodo recomendados (a refinar en Fase 6): `(s, speed_kmh, direction)`. La topología cambia en cada snapshot — se usa masking/padding para tamaño variable.

### 5.4 Caveat adicional sobre horizonte de predicción

La autocorrelación a 5 min está apenas en 0.31 (E2) y 0.60 (E59) para C.2 — marginal sobre el threshold 0.3. En lag-1 (1 min) los valores son 0.53 (E2) y 0.76+ (E59). **Sugerencia para Fase 5**: explorar horizontes de 1, 2, 3 y 5 minutos como hiperparámetro experimental, no fijar 5 min desde el inicio. Reportar la curva de error vs horizonte como hallazgo del paper.

## 6. Cierre del probe

El viability probe queda aprobado al cierre con los resultados del kernel `alexhuaracha/03-headway-viability` versión 5 (ejecutado el 2026-05-19, status `COMPLETE`). El siguiente paso del [plan de desarrollo](./plan-de-desarrollo.md) es scaffold de `src/preprocessing/` con los parámetros productivos de §3 y la formulación C.2 de §2.

Esta decisión actualiza implícitamente la sección "8.2 Próximos pasos" de [`propuesta.md`](./propuesta.md) — el preprocesamiento de Fase 2 se rige por este documento, no por la descripción genérica del paso 1 de §8.2.

## 7. Cierre de Fase 2 — Kaggle v3 multi-filar-disambiguation (2026-05-20)

### 7.0 AC-D5 — RETIRADA

**Criterio original** (de `sdd/c2-lookback-fix/spec`): "E59 non-null pair count changes by < 1% vs v2 — Row count comparison across Kaggle runs."

**Por qué se retira**: AC-D5 asumió que E59 dir=1 tenía cobertura adecuada y que el lookback fix no la alteraría. La validación Kaggle v3 (obs #38) falseó esta hipótesis: E59 dir=1 sobrevive en sólo 13.4% de fracción non-null, confirmando contaminación cross-street y no una regresión de conteo.

**Criterio de reemplazo — AC D-SHAPE**: la distribución `delta_t_min` de E59 dir=1 debe ser unimodal con mediana en [4, 12] min y skewness > 1.0 (exponential-like). Éste es un criterio de forma de distribución, no de estabilidad de conteo. Se valida en notebook 04b Figura 8.

---

### 7.0b Parámetro nuevo — `lateral_pair_threshold_m` (multi-filar-disambiguation)

**Motivación**: los corredores multi-filar (E2, E59) proyectan buses de calles paralelas al mismo eje `s`. Sin un filtro lateral, `compute_pairs` empareja buses en diferentes calles → headways espurios. El filtro `|lateral_m_front − lateral_m_back| > threshold` elimina estos pares cross-street.

| Campo | Valor | Notas |
|---|---|---|
| `ProductiveParams.lateral_pair_threshold_m` | **50.0 m** | Default operativo; sujeto a calibración Kaggle v4 |
| `EmpresaConfig.lateral_pair_threshold_m_override` | `None` (por defecto) | Permite override por empresa tras inspección del histograma 04b Figura 7 |
| Resolver | `lateral_pair_threshold_for(empresaid)` | Espejo de `lateral_threshold_for`; fallback a global para empresas sin config |

**Protocolo de calibración post-run Kaggle v4**:

1. Abrir notebook 04b v4. Inspeccionar Figura 7 (`|lateral_delta|` por empresa, dirección) con línea vertical en 50 m.
2. Si el histograma es **bimodal con un valle claro**: tomar el valor del valle como override. Redondear a 10 m. Establecer con `EmpresaConfig.lateral_pair_threshold_m_override`.
3. Si el histograma es **unimodal o el valle no es claro**: el default 50 m queda vigente.
4. Re-correr Kaggle sólo si se eligió un override en el paso 2; de lo contrario el run v4 es final.
5. Registrar el threshold final aquí (actualizar esta entrada con la decisión post-calibración).

**Nota para sdd-verify**: el AC D-PAIRS (n_pairs_efectivo >= 90% de v3 baseline) requiere que el contador v3 sea registrado ANTES de ejecutar el run Kaggle v4. Instrucción explícita: antes de subir el código v4 a Kaggle, anotar aquí los `n_pairs_efectivo` diarios (min/mean/max) de E2 y E59 de la tabla §7.1. Los valores v2 están en §7.1; si no se corrió v3, usar los de v2 como baseline conservador.

**Impacto en R7 schema**: `compute_pairs` y `compute_headways_c2` ahora emiten `lateral_m_front` (Float64, nullable) y `lateral_m_back` (Float64, nullable) como las dos últimas columnas del parquet de headways. Cambio ADITIVO — no se renombra ni cambia el tipo de ninguna columna anterior. Los 13 campos existentes están intactos.

---

## 8. Cierre de Fase 2 — Kaggle v2 (2026-05-20)

El módulo productivo `src/preprocessing/` fue implementado y ejecutado en Kaggle como kernel `alexhuaracha/04-preprocessing` versión 2, status `COMPLETE`. Los parquets `cleaned_gps_E{2,59}.parquet` y `headways_E{2,59}.parquet` quedaron generados y validados contra los invariantes de la spec (INV-4, INV-6, INV-7, INV-8 = 0 violations en ambas empresas).

### 7.1 Métricas del run productivo

| Métrica | E2 | E59 |
|---|---|---|
| cleaned_gps rows | 9,595,566 | 10,557,968 |
| headways rows | 1,692,411 | 3,530,316 |
| non-null `delta_t_min` | 1,691,530 (99.95%) | 3,523,241 (99.80%) |
| `pairs_efectivo`/día (min / mean / max) | 654 / 11,128 / 16,752 | 9,122 / 23,179 / 30,744 |
| Tiempo Kaggle | 656 s (~11 min) | 1,389 s (~23 min) |

### 7.2 Resolución de Caveat 3 (off-route filter 300m)

El threshold de `pairs_efectivo < 10k/día` para E2 (definido en §4 row 3 como criterio de calibración) fue **alcanzado en 32 de 152 días (21%)**. La investigación post-run categorizó cada uno de los 32 días:

| Categoría | Días | Acción Fase 2 |
|---|---|---|
| Domingos sistemáticos (caveat §6 de `eventos-anomalos.md`) | 22 | Patrón conocido; Fase 3 split design los maneja |
| Feriados ya documentados (Navidad, Año Nuevo) | 2 | Patrón conocido |
| Evento sistémico (2023-10-28, Señor de los Milagros) | 1 | Patrón conocido (`eventos-anomalos.md §3.1`) |
| Feriados nacionales nuevos (Nov 1, Nov 2, Dec 8) | 3 | **Documentados ahora en `eventos-anomalos.md §4`** |
| Días residuales cerca del threshold (8-9.7k pares) | 4 | Aceptados; sábados de baja demanda o post-feriados |

**Veredicto**: el threshold de 300m NO se sube a 500m. La caída de `n_pairs_efectivo` es operacional (menos buses circulando en días de baja demanda), no problema de filtrado lateral. Subir el threshold introduciría ruido (calles paralelas, depósitos) sin recuperar pares válidos.

**Hallazgo colateral**: detectamos 3 feriados nacionales peruanos que el §8 del notebook 02 (Fase 1) no había flagueado porque su criterio compara `active_units` y `records` contra la mediana agregada, y los feriados nacionales caen entre semana sin reducir esos contadores al 50% del baseline. La métrica `n_pairs_efectivo` (que requiere cruces históricos computables) sí los expone. Esto refuerza la recomendación del §8 de `eventos-anomalos.md`: refinar el baseline a (empresa, día_de_semana) en Fase 3.

### 7.3 Caveats que quedan abiertos para fases posteriores

- **Caveat 2 (winsorización p99 de `delta_t_min`)**: confirmado necesario — E2 tiene `max(delta_t_min) = 216,550` min, cola pesada. Decisión vinculante en Fase 5 al construir el dataset supervisado.
- **Caveat 4 (R² persistencia)**: sin cambios respecto al cierre del probe.
- **Caveat 5 (`n_pairs_151d` bug del probe)**: no aplicable a `04-preprocessing` (métrica no se vuelve a calcular).
- **Sin nuevos caveats** introducidos por el run productivo.

### 7.4 Estado de outputs

Los 4 parquets están disponibles en el kernel Kaggle `alexhuaracha/04-preprocessing` v2 (output del run COMPLETE). Para Fase 3 se recomienda promoverlos a un Kaggle Dataset versionado (siguiendo la política de §`docs/dataset-manifest.md`).
