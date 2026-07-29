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

En cada snapshot `T` (grilla uniforme cada `GRID_SECONDS` segundos), para cada par de buses consecutivos en la misma dirección, sea **L** el bus que va adelante (*leader*) y **F** el que lo sigue (*follower*):

```
Δt(L, F, T) = T − t_cross(L, s_F(T))
```

donde `t_cross(L, s_F(T))` es el instante en el que **el bus de adelante cruzó por última vez la posición donde el bus de atrás se encuentra ahora**, interpolado linealmente sobre la trayectoria pasada de **L**. En palabras: hace cuánto tiempo el bus líder estuvo donde está ahora el bus que lo sigue. Eso es el *headway*.

El vector resultante en el instante `T` tiene tamaño `N(T) − 1`, donde `N(T)` es la cantidad de buses activos en el corredor en `T`. Las unidades son **minutos**.

> ⚠️ **Las columnas del parquet tienen los nombres invertidos respecto de esta ecuación.** En `headways_E*.parquet` y en `src/preprocessing/headways.py`:
>
> | Columna | Bus |
> |---|---|
> | `bus_back`, `s_back` | **L** — el que va **adelante** (líder) |
> | `bus_front`, `s_front` | **F** — el que va **atrás** (seguidor) |
>
> Así que el código calcula `T − t_cross(bus_back, s_front)`, que es exactamente la ecuación de arriba con los nombres al revés. **La aritmética es correcta; solo las etiquetas mienten.**
>
> **Por qué se invierten, en las dos direcciones.** Esto sale de una **definición**, así que no puede desalinearse con el tiempo. `direction` se define como `sign(rolling_mean(ds))` (ver `src/preprocessing/direction.py::infer_direction`), así que dentro de un grupo de dirección el sentido en que se mueve `s` queda fijado por construcción:
>
> | Dirección | `s` a medida que el bus avanza | El líder tiene… |
> |---|---|---|
> | −1 | **decrece** | el `s` **menor** |
> | +1 | **crece** | el `s` **mayor** |
>
> El *sort key* es `s` para `direction = −1` y `−s` para `direction = +1` (`CALIBRATED_INVERTED_DIRECTION`), así que el orden ascendente pone al líder **primero** en los dos casos. `shift(1)` entrega esa primera fila a las columnas `back`. La inversión es uniforme entre direcciones, y eso es exactamente lo que hace correcto al pipeline a pesar de los nombres.
>
> **Verificación empírica** (sobre `data/processed/`, los tres corredores). Recalculando ambas lecturas con el mismo *helper*, sobre las mismas filas:
>
> | | Como está implementado | Como sugieren los nombres |
> |---|---|---|
> | Cobertura (no nulos) | **72 %** | 29 % |
> | *Headway* mediano | **4.96 min** | 11.65 min |
> | Velocidad implícita mediana | **9.6 km/h** | 2.0 km/h |
> | Fracción con velocidad plausible (5–40 km/h) | **70 %** | 27 % |
>
> Los 5–15 min de *headway* típico en Arequipa urbana y los ~10 km/h de un bus urbano confirman la primera columna. La segunda pregunta a un bus cuándo pasó por un lugar al que todavía no llegó.
>
> **Los nombres no se cambian a propósito.** Están en el esquema de los parquets procesados, en los *builders* de notebooks y en los residuos ya descargados de Kaggle; renombrarlos obliga a regenerar toda esa cadena para ganancia analítica nula. Se documenta la inversión en lugar de propagarla.

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

El viability probe queda aprobado al cierre con los resultados del kernel `alexhuaracha/03-headway-viability` versión 5 (ejecutado el 2026-05-19, status `COMPLETE`). El siguiente paso del [plan de desarrollo](./historico/plan-de-desarrollo.md) es scaffold de `src/preprocessing/` con los parámetros productivos de §3 y la formulación C.2 de §2.

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

**Baseline v3 para AC D-PAIRS** (registrado 2026-05-20 desde `/tmp/kernel-04-v3-outputs/headways_E{2,59}.parquet`, post c2-lookback-fix, pre multi-filar-disambiguation):

| Métrica | E2 v3 | E59 v3 |
|---|---|---|
| headways rows | 1,692,411 | 3,530,316 |
| n_pairs_efectivo (delta_t_min non-null) | 818,661 | 1,243,516 |
| pairs_efectivo/día (min / mean / median / max) | 270 / 5,386 / 5,935 / 8,651 | 2,385 / 8,181 / 8,818 / 10,545 |
| n_pairs_efectivo dir=-1 | 638,828 | 968,766 |
| n_pairs_efectivo dir=+1 | 179,833 | 274,750 |
| Cobertura dir=+1 | 24.0% | 13.3% |
| Cobertura dir=-1 | 67.7% | 65.9% |
| n_días | 152 | 152 |

**Umbrales D-PAIRS** (drop ≤ 10% relativo a v3 baseline):

| Empresa | v3 n_pairs_efectivo | Mínimo aceptable v4 (90%) |
|---|---|---|
| E2 | 818,661 | ≥ 736,795 |
| E59 | 1,243,516 | ≥ 1,119,164 |

Nota: el baseline v3 es ~50% (E2) y ~65% (E59) inferior al baseline v2 (§7.1) porque v2 incluía pares con bound C.2 sin acotar (`max_interpolation_lookback_minutes`). v3 es el baseline correcto post-c2-lookback-fix.

**Impacto en R7 schema**: `compute_pairs` y `compute_headways_c2` ahora emiten `lateral_m_front` (Float64, nullable) y `lateral_m_back` (Float64, nullable) como las dos últimas columnas del parquet de headways. Cambio ADITIVO — no se renombra ni cambia el tipo de ninguna columna anterior. Los 13 campos existentes están intactos.

### 7.0b.1 — Default OFF (2026-05-21)

**Decisión**: `ProductiveParams.lateral_pair_threshold_m` cambió de `50.0` a `float('inf')`.

**Evidencia**: Figura 7 del kernel Kaggle 04b v4 (ejecutado 2026-05-21) muestra una distribución `|lateral_m_front − lateral_m_back|` monotónicamente decreciente de 0 a 50 m para E2/E59 × dir=±1. No existe valle bimodal. Sin un valle bimodal no hay umbral calibrable: cualquier valor entre 0 y 50 m discriminaría de forma arbitraria, no estructural.

**Consecuencia**: el filtro lateral queda como infraestructura opt-in. Con el default `float('inf')`, la condición `|delta| > inf` es siempre falsa y `compute_pairs` se comporta idéntico al estado pre-multi-filar-disambiguation. Los tests, el schema R7 v4 y `EmpresaConfig.lateral_pair_threshold_m_override` se preservan para activación futura por empresa si la evidencia cambia.

**Avance**: la causa raíz de la contaminación cross-street es upstream (centerline único + proyección sin separación por dirección). El SDD `multi-filar-direction-balanced-centerline` (Option D) es la ruta que aborda este problema estructuralmente.

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

---

## 8. Centerline por dirección (multi-filar-direction-balanced-centerline, Option D)

**Estado:** IMPLEMENTADO (2026-05-21) — SDD `multi-filar-direction-balanced-centerline`. Validación Kaggle v5 pendiente (post-merge manual).

### 8.1 Causa raíz

El SDD predecesor `multi-filar-disambiguation` (archivado PARTIAL-SUCCESS) demostró que el centroide único construido por `build_centerline` sobre todos los pings (ida + vuelta mezclados) cae **entre las dos calles paralelas** de los corredores multi-filares E2 y E59. Esto produce:

- `lateral_m` ruidoso (cada ping proyectado contra el centroide del corredor en lugar de su propia calle).
- `s` ruidoso → `ds/dt` ruidoso → etiquetas de dirección erróneas.
- Contaminación cruzada de pares (bus de calle A emparejado con bus de calle B).

La distribución `|lateral_delta|` mostrada en Figura 7 del notebook 04b v4 es monotónicamente decreciente, sin valle bimodal. No es posible calibrar un threshold post-emparejamiento; la corrección debe ser upstream.

### 8.2 Estrategia two-pass

**Opción D — Two-pass PCA centerline** (rechazadas: B=DBSCAN, C=displacement-heuristic).

```
Pass 1 (existente):
  build_centerline → project_to_centerline → infer_direction
  [produce etiquetas de dirección crudas, dir=-1 ≈ 50-63% cobertura]

Pass 2 (nuevo, E2 y E59):
  build_centerline_per_direction  → project_per_direction
  → infer_direction               → assign_trip_ids
  [cada ping proyectado contra el centroide de SU calle, no el centroide global]
```

### 8.3 Política de fallback

Threshold mínimo: **1,000 pings por subconjunto de dirección** para intentar PCA pass-2. Justificación: `_build_centerline_from_points` requiere ≥ 5 muestras por bin × 50 bins mínimos = 250 pings absolutos; 1,000 provee un margen de seguridad 4× para el recorte IQR (0.5% por cola) y el paso de mediana binada.

Cuando un subconjunto cae por debajo del threshold (o PCA lanza `ValueError`), `build_centerline_per_direction` vuelve al centroide single-pass para esa dirección y emite un `FallbackEvent` WARNING estructurado con `empresaid`, `direction` y `pings`. El comportamiento downstream es transparente: `s` y `lateral_m` se escriben con dtype Float64 sin introducir nulls.

### 8.4 Garantía de esquema R7 v4

El esquema de los parquets `cleaned_gps` y `headways` **no cambia**. Las columnas `s`, `lateral_m`, `lateral_m_front`, `lateral_m_back`, `direction`, `delta_t_min`, etc. conservan nombres, dtypes y semántica. Solo los **valores** de `s` y `lateral_m` mejoran (cada ping proyectado contra la calle correcta).

### 8.5 Invariante de ordenamiento (R-PIPE2)

`assign_trip_ids` se ejecuta DESPUÉS de la segunda llamada a `infer_direction` en el path two-pass. Este ordenamiento se verifica con un test de integración (`TestTwoPassPipeline::test_two_pass_call_order`).

### 8.6 Validación Kaggle v5 (post-merge, manual)

| AC | Métrica | Target |
|---|---|---|
| D2-EXPO-DIR1 | Skewness `delta_t_min` E59 dir=+1 | > 1.0 |
| D2-SHAPE-E59-DIR1 | Mediana `delta_t_min` E59 dir=+1 | ∈ [4, 12] min |
| D2-ASYMMETRY | Ratio cobertura dir=+1 / dir=-1 (E2 y E59) | ∈ [0.77, 1.30] |
| D2-NPAIRS-REGRESS | `n_pairs_efectivo` vs baseline v3 | ≥ 0.90 × baseline |
| D2-DETERMINISM | Hash centerlines numpy en 2 runs consecutivos | bit-identical |

**Bloqueantes**: D2-NPAIRS-REGRESS y D2-DETERMINISM. Si cualquiera falla, el SDD no puede archivarse.

### 8.7 Rollback

El two-pass está gateado por `centerline_strategy_for(empresaid)`. Revertir el merge commit o forzar `centerline_strategy_override=None` en `EMPRESA_CONFIG` restaura el comportamiento v4 completamente. El esquema R7 v4 no cambia, por lo que los parquets anteriores son legibles sin migración.

---

## 9. Cierre de Fase 2 — Kaggle v8 direction-conditional sort (2026-05-23)

Cross-references: SDD `dir1-pair-ordering-h7` (archive engram obs #148), SDD `e2-short-headways-audit` (archive engram obs #153), predecesor fallido `dir1-pair-coverage-recovery`, SDD instrumentación `headway-null-diagnostics`.

### 9.1 Causa raíz (H7)

Después del two-pass centerline (§8), la cobertura `dir=+1` seguía estructuralmente rota: E2 dir+1 cubría 15.6% válido y E59 dir+1 cubría 7.3%, vs ~70% en dir=-1. El SDD `headway-null-diagnostics` instrumentó counters de buckets de NULL y reveló que el bucket dominante en dir+1 era **stale-crossing** (E2 83.5%, E59 92.6%) — el algoritmo C.2 encontraba crossings pero con timestamps anteriores a `max_lookback_minutes = 30`, así que los descartaba.

Las hipótesis H1-H6 (multi-filar, heading, ds-zero, etc.) quedaron descartadas por evidencia empírica. La hipótesis **H7** se confirmó por lectura de código + datos:

> `compute_pairs` en `src/preprocessing/headways.py:71` ordenaba unconditionally por `s` ascending para asignar `bus_front` (último s) y `bus_back` (penúltimo s). Pero el PCA centerline del two-pass (§8) no tiene orientación canónica — para una de las dos direcciones físicas, `s` decrece con el sentido de marcha. En esa dirección, `s` ascending equivale a **back-to-front** físico, no front-to-back, así que `bus_front` se asignaba al bus que va FÍSICAMENTE DETRÁS. `_find_last_crossing_ns` encontraba el cruce HISTÓRICO (cuando ese bus, antes de ser sobrepasado, sí estaba adelante), con timestamp > 30 min, y `max_lookback` lo filtraba.

### 9.2 Solución — direction-conditional sort key (Encoding A)

Se introduce el parámetro `CALIBRATED_INVERTED_DIRECTION: Literal[1, -1]` en `src/preprocessing/config.py:132-147` con valor `1` hardcoded. `compute_pairs` ahora calcula el sort key como:

```python
s_sort = pl.when(pl.col("direction") == CALIBRATED_INVERTED_DIRECTION) \
           .then(-pl.col("s")) \
           .otherwise(pl.col("s"))
```

Y ordena por `s_sort` ascending. El efecto es que para `direction == +1` el orden se invierte (mayor `s` físico queda al final → `bus_front` recibe el bus físicamente adelantado). Para `direction == -1` el comportamiento no cambia.

**Esquema R7 v4 sin cambios**. La firma de `compute_headways_c2` (tupla `(headways_df, null_buckets_df)` post-SDD `headway-null-diagnostics`) se preserva. No hay migración de parquets.

### 9.3 Calibración observacional

Los parquets v7 no estaban disponibles localmente para correr la calibración directa `corr(s, t_ns)` por bus que el design proponía. Se eligió **calibración observacional**: la distribución de buckets de NULL del SDD predecesor (`headway_null_buckets_E{2,59}.parquet` v7) identificó inequívocamente dir+1 como la dirección con stale-crossing dominante en ambas empresas (E2 83.5%, E59 92.6%). Por construcción, esa es la dirección con `s` invertido. Decisión registrada en engram obs #135.

`AC-PROC-1` satisfecho por la decisión documentada. `AC-PROC-2` (stop on ambiguity) N/A.

### 9.4 Validación Kaggle v8 — adjudicación AC-DATA-2

Kaggle kernel `alexhuaracha/04-preprocessing` v8 ejecutado 2026-05-23 (~36 min). Resultados:

| Empresa | Dir | Métrica | v7 (pre-fix) | v8 (post-fix) | Δ |
|---|---|---|---|---|---|
| E2 | +1 | success | 15.6% | **57.85%** | **+42.25pp** |
| E2 | +1 | stale-crossing | 83.5% | 42.06% | −41.44pp |
| E59 | +1 | success | 7.3% | **74.00%** | **+66.70pp** |
| E59 | +1 | stale-crossing | 92.6% | 25.97% | −66.63pp |
| E2 | -1 | success | 70.5% | 70.53% | +0.03pp (sin regresión) |
| E59 | -1 | success | ~70% | 79.69% | +9.69pp |

Adjudicación contra targets del spec:

| AC | Veredicto | Evidencia |
|----|-----------|-----------|
| AC-DATA-2a (dir+1 stale <30%) | **PARTIAL** | E59 PASS (26%), E2 FAIL (42%) |
| AC-DATA-2b (dir+1 success ≥50%) | **PASS** | E2 58%, E59 74% |
| AC-DATA-2c (dir-1 success ≥65%) | **PASS** | E2 70.5%, E59 79.7% |
| AC-DATA-2d (dir-1 stale <15%) | **FAIL-aspirational** | E2 28%, E59 20%; target no cumplido por baseline v7 tampoco |
| AC-DATA-2e (discrimination invariant) | **PASS** | sum_check OK en las 4 (empresa, dir) groups |

Verdict del SDD: **PASS-WITH-WARNINGS**. El objetivo funcional (cobertura dir+1 paper-credible en ambas empresas) está conseguido. Los targets stale missed eran aspiracionales sin baseline medido.

### 9.5 Audit complementario — E2 short-headway tail (NO-FIX)

Tras el fix, el dataset v8 mostró que **E2 tiene 10% de headways < 1 min** vs 3% en E59. SDD audit-only `e2-short-headways-audit` investigó cinco hipótesis empíricamente:

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| H1 — Multi-filar cross-lane | **REFUTED** | mean(\|lat_delta\|) para <1min < para ≥1min (ratio 0.78 E2, 0.87 E59). Si fuera multi-filar el ratio sería >1. |
| H2 — Buses detenidos apareados | PARTIAL (solo E59) | E59 <1min: 46% min_speed <2 km/h. E2 <1min: 18%, indistinguible de baseline. |
| H3 — Rush-hour real | Rechazada como principal | E2 short% plano 6-12% todo el día (5h-21h); no concentrado en picos. |
| H4 — pair_rank artifact | No es artefacto | Decay gracioso 1→8 en ambas empresas. |
| H5 — Mismo bus | Imposible | algebraicamente bloqueado por `shift(1).over(group_cols)`. |
| H6 — Densidad operacional real | **SUPPORTED** | E2 mediana dir+1 = 5.0 min vs E59 7.5 min; corredor E2 opera con frecuencia genuinamente más alta. |

**Conclusión**: el 10% de E2 es operación real, no defecto. **Sin cambio de pipeline.** Cero commits. Filtros opcionales (`delta_t_min >= 1.0` o flag `is_short`) a discreción del notebook ML downstream, no del preprocesamiento.

Archive engram obs #153.

### 9.6 Cobertura final lista para Fase 3

| Empresa | Dir (semántica) | success_count | total_pairs | success_fraction |
|---|---|---|---|---|
| E2 | -1 (vuelta) | 495,562 | 702,619 | 70.53% |
| E2 | +1 (ida) | 513,722 | 888,040 | 57.85% |
| E59 | -1 (vuelta) | 1,155,295 | 1,449,818 | 79.69% |
| E59 | +1 (ida) | 913,898 | 1,235,060 | 74.00% |

**~3 millones de observaciones válidas** distribuidas en 5 meses contiguos (2023-10-01 → 2024-02-29, 152 días sin gaps). **IDA Y VUELTA confirmadas para ambas empresas** — habilitación del objetivo del paper IJACSA.

### 9.7 Rollback

El fix está concentrado en una expresión Polars en `headways.py:71`. Para revertir:

1. Forzar `CALIBRATED_INVERTED_DIRECTION = None` o eliminar la rama `when().then(-s)`, restaurando el sort unconditional.
2. Re-correr NB04 en Kaggle para regenerar `headways_<empresa>.parquet`.
3. Esquema R7 v4 no cambia → no hay migración.

Costo del rollback: 1 línea de código + 1 re-run de Kaggle (~36 min). El predecesor `dir1-pair-coverage-recovery` está documentado como ejemplo de qué NO hacer (orientation flip sin direction-awareness, validado por tests pero no funcionalmente).

### 9.8 Items abiertos (no bloquean Fase 3)

- **E2 dir+1 residual stale-crossing 42%**: si el paper requiere paridad estricta entre empresas, evaluar promover `CALIBRATED_INVERTED_DIRECTION` a `Mapping[int, int]` per-empresa. Hipótesis: el corredor E2 tiene segmentos con orientación no monotónica.
- **E59 stopped buses overnight**: caracterización informativa, no afecta el pipeline.
- **Asimetría flota/frecuencia E2 vs E59**: E59 (40 unidades) opera con headway mediano 7.5 min; E2 (31 unidades) con 5 min. Diferencia operacional real, no investigada a fondo (largo de corredor, tiempo de vuelta).
- **Spec inicial — targets aspiracionales**: lección registrada — futuros SDDs deben derivar thresholds de baselines medidos, no de aspiraciones.
