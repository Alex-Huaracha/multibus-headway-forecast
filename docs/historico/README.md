# Histórico

Documentos **superados** que se conservan como rastro de decisiones del proyecto
(el *por qué* de cómo llegamos al resultado final), no como referencia vigente.
La fuente de verdad actual vive en `docs/resultados/documento-resultados.md`,
`docs/dataset-manifest.md` y `docs/plan-de-desarrollo.md`.

| Archivo | Qué fue | Superado por |
|---|---|---|
| `diagnostico-y-plan-paper.md` | Diagnóstico (jun-2026) que reencuadró el paper: resultado nulo espacial + el problema de la persistencia a 1 min. Definió el plan multi-horizonte. | Ejecutado — resultados en `documento-resultados.md` |
| `mejoras-resultados.md` | Plan de trazabilidad que definió la recertificación (evaluación pareada, winsorización full-split, terciles ex-ante, XGBoost condicional, reproducibilidad). | Ejecutado — change OpenSpec `paper-recertification` (archivado) |
| `fase-5-lstm-baseline.md` | Reporte de la fase LSTM single-horizonte (h=1). | `documento-resultados.md` (multi-horizonte) |
| `fase-6-spatial-conv-lstm.md` | Reporte de la fase SpatialConvLSTM single-horizonte. | idem |
| `fase-6b-spatial-transformer.md` | Reporte de la fase SpatialTransformer single-horizonte. | idem |
| `INTERPRETACION.md` | Guía de lectura de figuras/tablas (qué muestra cada una, qué número citar). | `documento-resultados.md` embebe la interpretación; sus cifras son pre-recertificación |

> Contienen **números pre-recertificación** (antes de winsorizar el test sobre todos
> los splits). No citar de acá para el paper — usar `documento-resultados.md`.
