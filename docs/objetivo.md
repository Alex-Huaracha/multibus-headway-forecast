# Objetivo del Proyecto

## Objetivo general

Publicar en IJACSA un paper que presente y evalúe un método de Deep Learning para predecir el vector completo de headways de corredores de transporte público urbano integrado, usando exclusivamente datos GPS, validado sobre corredores reales de Arequipa.

## Criterios de éxito

El proyecto se considera exitoso cuando se cumplen, simultáneamente:

1. El mejor modelo de DL supera a los baselines estadísticos (naive y promedio móvil) en MAE y RMSE, con significancia estadística (p < 0.05) usando test de Diebold-Mariano o Wilcoxon pareado.
2. La evaluación se realiza sobre al menos 2 corredores reales (Empresas 2 y 59), con resultados consistentes en ambos.
3. El paper completo está redactado, revisado por asesores, y enviado a IJACSA.

## Alcance

**Empresas obligatorias:** 2 y 59 (las dos de mayor flota simultánea).

**Empresas deseables:** 4 y 58 (entran si el preprocesamiento es viable y robusto; fortalecen el argumento de generalización).

**Modelos a comparar (mínimo):**

- Baselines estadísticos: predicción ingenua y promedio móvil.
- Baseline DL: LSTM sobre el vector aplanado.
- Modelo espacial-temporal: la arquitectura final se decide empíricamente entre GNN+LSTM, Transformer con atención entre buses, STGCN u otras. El compromiso no es con una arquitectura específica sino con la comparativa rigurosa.

## Contribución central

La novedad del paper es **aplicada**, no metodológica:

- El problema (predicción del vector completo de headways en corredores reales).
- El dataset (~99M registros GPS tras deduplicación, 5 meses, 4 corredores reales con flotas de tamaño variado).
- La comparativa rigurosa entre enfoques con y sin modelado espacial entre buses.

No se reclama una arquitectura nueva. Se aplica y adapta arquitecturas existentes al problema, lo cual encaja con el perfil de IJACSA.

## Lo que NO es objetivo del proyecto

- Detección de anomalías como tarea principal (es un subproducto).
- Despliegue en producción o dashboards para operadores.
- Optimización de rutas, flotas o frecuencias.
- Comparación con sistemas BRT formales (TransMilenio, Metropolitano, etc.).
- Predicción de demanda de pasajeros.

## Entregable final

Manuscrito enviado a IJACSA + repositorio reproducible con datos procesados, código y checkpoints.
