# Eventos anómalos en el dataset GPS

Este documento registra los eventos identificados en el EDA (Fase 1, notebook 02) que se manifiestan como días atípicos en el comportamiento operacional. La lista informa decisiones de Fase 3 (split design) y Fase 7 (robustness analysis).

## 1. Fuente

- Notebook `02_eda_corredores`, sección §8 ("Atypical days detection").
- Output: `atypical_days.csv` — 22 (empresa, día) flagged sobre las 4 empresas.

## 2. Criterio operativo de "día atípico"

Un día (empresa, día) es flagged cuando:

- `records < 50% × mediana_records_diaria_por_empresa` (low_records), o
- `active_units < 50% × mediana_active_units_diaria_por_empresa` (low_fleet).

No todo día flagged es operacionalmente significativo. Distinguimos tres categorías:

| Categoría | Característica | Tratamiento |
|---|---|---|
| Evento sistémico | Mismo día flagged en múltiples empresas. Causa externa identificable. | Documentar; usar como caso de estudio (F7). |
| Feriado conocido | Caída esperada por calendario. | Documentar; opcionalmente excluir de train o anotar como `is_holiday`. |
| Ramp-up | Días iniciales con flota incompleta antes de que la empresa operara a régimen. | NO es anomalía; tratar como período de calentamiento o excluir. |

## 3. Eventos sistémicos

### 3.1 2023-10-28 — Procesión del Señor de los Milagros y sismo en Caylloma

**Empresas afectadas**: E2, E4, E58, E59 (los 4 corredores).

**Causa operacional dominante**: sexto y último recorrido procesional del Señor de los Milagros por las calles del Cercado de Arequipa y la Plaza de Armas. La procesión cierra arterias principales del centro, donde los 4 corredores convergen. La caída simultánea en todas las empresas es consistente con un cierre de calles externo, no con un problema interno de cada empresa.

**Causa secundaria (impacto operacional diurno presumiblemente bajo)**: sismo con epicentro a 12 km al sureste de Cabanaconde, provincia de Caylloma, registrado a las 04:03 a.m. (hora local) por el IGP y el CISMID. La hora (madrugada) y la distancia al área urbana de Arequipa sugieren que su efecto sobre el transporte diurno fue menor. Lo registramos para completitud.

**Decisión Fase 3 (split design)**: este día NO debe caer en test si las métricas pretenden representar comportamiento típico. Recomendado: incluir en train con tag de evento, o excluir explícitamente del análisis.

**Decisión Fase 7 (robustness)**: caso de estudio natural para "comportamiento del sistema bajo perturbación externa sincrónica". Útil para demostrar que el modelo no solo predice el régimen normal sino que también degrada de forma interpretable bajo eventos identificables.

## 4. Feriados conocidos

| Fecha | Empresas afectadas | Evento |
|---|---|---|
| 2023-12-25 | E4, E58 | Navidad |
| 2023-12-31 | E4 | Víspera de Año Nuevo |
| 2024-01-01 | E2, E4 | Año Nuevo |

Estos días son **predecibles** y representan operación reducida pero no estructuralmente anómala. No deben tratarse como outliers en F7. Pueden incluirse en train, val o test sin manipulación especial; opcionalmente se anotan como `is_holiday` si Fase 5+ usa esa señal como feature.

## 5. Períodos de ramp-up (NO son anomalías)

### 5.1 E58 — 2023-10-01 a 2023-10-04

Cuatro días consecutivos con 2–3 unidades activas (vs mediana 9) y ~25% de records vs mediana.

**Interpretación**: la empresa 58 no estaba operando a régimen al inicio del período del dataset. Esto NO es un evento anómalo operacional — es la forma del dataset: la empresa "se llena" durante las primeras semanas.

**Decisión Fase 3 (split design)**: considerar excluir las primeras 1-2 semanas de E58 del análisis, o tratar explícitamente como warm-up. Si se incluyen en cualquier split, no presentarlas como métricas representativas del modelo.

**Decisión Fase 7**: NO usar este período como caso de estudio de robustez — no es robustez, es escasez de datos.

## 6. Caveat metodológico — el §8 sobre-flagea fines de semana

El criterio del §8 compara cada día contra la mediana **agregada sobre todos los días** de la empresa. Esto sobre-flagea los domingos en empresas con caída sistemática de fin de semana:

- E2: de sus 7 días flagged, **5 son domingos** (2023-11-05, 2023-11-12, 2023-11-26, 2023-12-03, 2024-01-21).
- E58: varios de los Sundays flagged caen en su ramp-up (Oct 1, 8, 22).

Estos días NO son eventos anómalos — son días de menor demanda dentro de un patrón semanal estable. La metodología del §8 es un **primer pase** útil para señalizar eventos verdaderamente atípicos (sistémicos como 2023-10-28). Fase 3 debería refinar el baseline a **mediana por (empresa, día_de_semana)** antes de usar la lista para split design.

## 7. Mantenimiento

Este documento se re-emite cuando:

- Se descubre un nuevo evento sistémico con impacto cross-empresa.
- Se identifica una nueva categoría de anomalía.
- Fase 3 refina el criterio del §8 (baseline weekday-aware) y la lista cambia.
- El dataset crece con nuevos meses y aparecen días flagged adicionales.

---
*Generado al cierre de Fase 1 (2026-05-19) a partir del análisis del notebook 02_eda_corredores ejecutado en Kaggle como kernel version 3.*
