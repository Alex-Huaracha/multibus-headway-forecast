# Decisiones de limpieza para Fase 2

Este documento cierra el criterio de Fase 1 del [plan de desarrollo](./plan-de-desarrollo.md): consolida los problemas de calidad detectados en el EDA (notebook `02_eda_corredores`, ejecutado en Kaggle sobre `clean_gps.parquet`) y registra la decisión de limpieza adoptada para cada uno antes de iniciar Fase 2.

> **Estado:** APROBADAS (2026-05-19) — cada decisión queda como contrato de Fase 2. Cualquier cambio posterior requiere re-emitir este documento con nueva versión y motivación explícita.

## 1. Fuente de los hallazgos

- Notebook `notebooks/02_eda_corredores/02_eda_corredores.ipynb` (generado por `src/build_notebook_02.py`).
- Salidas reales del run en Kaggle: `quality_gps.csv` y `figuras/`.
- Filas totales de entrada: **47,681,656** (4 corredores, post-dedup de Fase 0).
- Filas tras limpieza row-level del preflight: **47,633,619** (descartadas: 48,037).

## 2. Tabla de decisiones

| # | Problema | Magnitud observada | Decisión | Justificación |
|---|---|---|---|---|
| 1 | `time` null | 19 filas | **Descartar fila** | Sin timestamp no se puede ubicar el registro en la serie temporal. Aplicado ya en el preflight. |
| 2 | `lat` o `lon` null | 19 filas | **Descartar fila** | Sin coordenada no hay análisis espacial. Aplicado ya en el preflight. |
| 3 | `(lat, lon) == (0, 0)` | 48,018 filas | **Descartar fila** | (0, 0) está en el océano Atlántico — es valor centinela de GPS no fijado, no posición real. Aplicado ya en el preflight. |
| 4 | Velocidad observada > 80 km/h entre pares consecutivos | E2: 0.044%, E4: 0.021%, E58: 0.031%, E59: 0.015% | **Descartar el par para el cálculo de step/headway** (no la fila completa) | Supera el techo urbano realista de Arequipa. La velocidad observada se calcula como `step_m / dt_s`; descartar el par evita propagar el salto a la serie de headways. La fila se conserva para otros usos. |
| 5 | Salto espacial: `step_m > 500m` con `dt_s ≤ 60s` | E2: 0.008%, E4: 0.012%, E58: 0.017%, E59: 0.010% | **Descartar el par** | Teleport GPS (>30 km/h por geometría, pero criterio geométrico independiente del tiempo). Mismo razonamiento que (4). |
| 6 | Campo `velocidad` reportado = 0 mientras el bus se movió ≥ 50 m | E2: 1.89%, E4: 3.26%, E58: 0.00%, E59: 0.00% | **Ignorar el campo `velocidad` reportado en todo el pipeline; usar siempre velocidad observada `step_m / dt_s`** | El campo está descalibrado o no se reporta. Usar la velocidad observada es la convención ya documentada en el notebook 02. Para E58/E59 ver §3. |
| 7 | Campo `direccion` = 0 (sentinel) | E2: 0.80%, E4: 0.58%, E58: 0.00%, E59: 0.00% | **Tratar `direccion == 0` como "indefinido" (NULL semántico) al usar heading** | Es valor centinela mientras el bus está estacionado. No representa rumbo norte real. |
| 8 | `direccion` no reportada (NULL) en E58 y E59 | 100% efectivo (n_zero = 0, n_null > 0) | **Cambiar el método primario de detección ida/vuelta a la derivada de `s`; heading queda como confirmación opcional sólo en E2/E4** | Ver §3 — implicación crítica para Fase 2. |
| 9 | Gaps inter-registro > 30 min | E2: 2,228 gaps, E4: 1,918, E58: 977, E59: 2,985 (sobre 17.7M, 7.8M, 4.2M, 17.9M pares) | **Cortar la segmentación de viajes en ese punto** (el bus no estaba reportando — efectivamente fuera de servicio) | Un gap > 30 min indica blackout GPS, fin de viaje o falla de comunicación; no debe interpolarse como continuidad operacional. |
| 10 | Duplicados residuales con clave `(empresaid, unidadid, time)` | 0 en las 4 empresas | **Sin acción** | Fase 0 ya dedupea correctamente. Verificación pasa. |

## 3. Implicaciones para Fase 2

### 3.1 Detección ida/vuelta (cambio de método primario)

El plan original de Fase 2 incluía `Identificación de sentido ida/vuelta (heading + derivada de s)`. El EDA reveló que **dos de las cuatro empresas (58 y 59) no reportan `direccion`**, y E59 es **empresa obligatoria** según [`objetivo.md`](./objetivo.md).

Decisión: **el método primario de detección de sentido será la derivada signada de la coordenada lineal `s = proyección sobre el corredor`.** El heading se usará únicamente como verificación cruzada en E2 y E4 cuando esté disponible y no sea valor centinela.

Esto debe declararse explícitamente en el paper como decisión metodológica forzada por la realidad del dataset — refuerza, no debilita, la generalización del método.

### 3.2 Cálculo de velocidad

El campo `velocidad` reportado por el GPS se descarta del pipeline. Toda velocidad usada en Fase 2 (para detectar parado/moviendo, segmentar viajes, calcular tiempos de recorrido) se computa como `step_m / dt_s` con los descartes de los puntos (4) y (5).

### 3.3 Segmentación de viajes

Un viaje termina cuando ocurre cualquiera de:
- El bus alcanza un extremo del corredor (definido por la centerline reconstruida).
- Hay un gap > 30 min entre registros consecutivos (decisión 9).
- El bus cambia de sentido según la derivada de `s` (decisión 3.1).

### 3.4 Conteo de unidades por empresa (resuelto)

El notebook 02 confirmó que `clean_gps.parquet` tiene **102 unidades operacionales** distribuidas como 31 (E2), 19 (E4), 12 (E58), 40 (E59). Esto difiere del conteo original en `propuesta.md` §4.2 (120 unidades) por el **filtro de buses estacionarios aplicado en Fase 0** (commit `047ed85`, notebook 01), no por el preflight de 48k filas como se anticipó en versiones anteriores de este documento. Las 18 unidades excluidas no aportan al análisis de headways. `propuesta.md` §4.2 y §7 fueron actualizados con la cifra operacional al cierre de Fase 1.

### 3.5 Días atípicos (input para Fase 3 y Fase 7)

El §8 del notebook 02 identifica 22 (empresa, día) flagged. Su análisis cualitativo y las decisiones derivadas viven en [`eventos-anomalos.md`](./eventos-anomalos.md). Hallazgo clave: **2023-10-28** es sistémico en los 4 corredores (procesión del Señor de los Milagros) y debe tratarse como caso de estudio de robustez, no como ruido.

La metodología del §8 sobre-flagea fines de semana (compara contra mediana de todos los días). Fase 3 refinará el baseline a (empresa, día_de_semana) antes de usar la lista para split design. Ver [`eventos-anomalos.md`](./eventos-anomalos.md) §6.

## 4. Cierre de Fase 1

Aprobado al cierre de Fase 1 con los hallazgos del notebook `02_eda_corredores` (kernel version 3 en Kaggle, ejecutado el 2026-05-19). Los checkboxes de Fase 1 en [`plan-de-desarrollo.md`](./plan-de-desarrollo.md) se consideran satisfechos y Fase 2 puede iniciar bajo estas decisiones como contrato.
