# Plan de Desarrollo

Este plan ejecuta el objetivo definido en [`objetivo.md`](./objetivo.md). Cada fase tiene un objetivo propio, una lista de tareas con casilleros para marcar avance, un artefacto verificable y un criterio de cierre. No hay fechas: una fase se cierra cuando cumple su criterio.

---

## Fase 0 — Viabilidad y filtro de empresas

**Objetivo de la fase:** Identificar qué empresas tienen corredores viables para el problema.

- [x] Análisis PCA de linealidad de rutas (sobre puntos en movimiento).
- [x] Conteo de buses simultáneos por empresa (datos deduplicados con clave compuesta).
- [x] Análisis de sensibilidad sobre umbrales.
- [x] Selección de 4 empresas (2, 4, 58, 59) acotada por alcance.
- [x] Notebook `01_viability_and_filter` ejecutado en Kaggle con outputs verificados.
- [x] `clean_gps.parquet` producido (47.68M filas — 4 corredores filtrados del dataset crudo de 98.97M dedup de las 12 empresas).
- [x] Propuesta actualizada con números reales del análisis.

**Artefacto:** `notebooks/01_viability_and_filter/01_viability_and_filter.ipynb` + `data/processed/{viability,sensitivity}.csv` + `data/processed/figuras/` + `data/processed/clean_gps.parquet`.

**Estado:** Completada.

---

## Fase 1 — EDA dirigido sobre los corredores seleccionados

**Objetivo de la fase:** Entender la calidad y dinámica temporal/espacial de los datos GPS de cada corredor antes de procesarlos. Detectar problemas de calidad que impactarán el preprocesamiento.

- [x] Distribución temporal: registros por hora/día/mes por empresa.
- [x] Detección de gaps de servicio y días atípicos.
- [x] Heatmaps espaciales por empresa (validar trazado del corredor).
- [x] Estadísticas por unidad: viajes/día, horas activas, distancia recorrida.
- [x] Diagnóstico de calidad GPS: velocidades imposibles, saltos, duplicados.
- [x] Análisis de heading: ¿permite distinguir ida/vuelta de forma confiable?
- [x] Documentar problemas encontrados y decisiones de limpieza para Fase 2.

**Artefacto:** `notebooks/02_eda_corredores/02_eda_corredores.ipynb` + `quality_gps.csv` + `atypical_days.csv` (en Kaggle) + [`docs/decisiones-limpieza-fase2.md`](./decisiones-limpieza-fase2.md) + [`docs/eventos-anomalos.md`](./eventos-anomalos.md) + [`docs/dataset-manifest.md`](./dataset-manifest.md).

**Criterio de cierre:** Tabla de problemas de calidad por empresa documentada y decisiones de limpieza aprobadas.

**Estado:** Completada (2026-05-19).

---

## Fase 2 — Preprocesamiento y cálculo de headways

**Objetivo de la fase:** Producir series temporales limpias de headways por corredor y sentido, listas para alimentar al modelo. Esta es la fase más crítica: la calidad de los headways determina el techo de rendimiento de todos los modelos.

**Estado:** Completada (2026-05-20).

- [x] Reconstrucción del trazado del corredor (median path / centerline).
- [x] Proyección lineal: convertir (lat, lon) en distancia acumulada `s`.
- [x] Identificación de sentido ida/vuelta: método primario = derivada signada de `s`; el heading se usa solo como verificación cruzada en E2 y E4 (E58 y E59 no reportan `direccion`, ver `decisiones-limpieza-fase2.md` §3.1).
- [x] Segmentación de viajes (terminal a terminal).
- [x] Definición operativa de headway (espacial vs. temporal — decidir y documentar).
- [x] Cálculo de headways en grilla temporal regular.
- [x] Estrategia de cardinalidad variable (buses que entran/salen del corredor).
- [x] Validación visual sobre muestras (lado a lado con GPS crudo).
- [x] Módulo `src/preprocessing/` reutilizable.
- [x] Notebook documentando decisiones por empresa.
- [x] Dataset intermedio `data/processed/headways_<empresa>.parquet`.

**Artefacto:** Módulo `src/preprocessing/` + notebook `04_preprocessing` + datasets parquet por empresa.

**Criterio de cierre:** Para Empresas 2 y 59, series de headways validadas, sin huecos no documentados, listas para Fase 3.

---

## Fase 3 — Construcción del dataset supervisado

**Objetivo de la fase:** Transformar las series de headways en pares (X, y) para entrenamiento, con splits temporales sin leakage.

- [x] Definición de ventana de entrada `T_in` y horizonte `T_out`.
- [x] Generación de ventanas deslizantes por corredor y sentido.
- [x] Split train/val/test temporal (cronológico, no aleatorio).
- [x] Normalización ajustada solo sobre train.
- [x] Codificación de features de contexto (hora cíclica, día de semana).
- [x] Implementación de máscara para cardinalidad variable.
- [x] DataLoader / Dataset reutilizable.
- [x] Notebook con estadísticas de splits y verificación de sanidad.

**Artefacto:** Módulo `src/data/dataset.py` + notebook `05_dataset`.

**Criterio de cierre:** DataLoaders reproducibles entregando tensores con shapes documentados, sin leakage entre splits.

**Estado:** Completada (2026-05-23).

---

## Fase 4 — Baselines estadísticos

> **Nota sobre orden de ejecución**: esta fase se implementó ANTES de la Fase 3 (SDD `phase-3-baselines-classical`, cerrado 2026-05-21, commit `fe12cdd`). Los baselines clásicos operan directamente sobre `headways_E{2,59}.parquet` por slot `(empresaid, direction, pair_rank)` sin requerir ventanas X/y prefabricadas, por lo que no dependen de Fase 3. Fase 5 (LSTM) y Fase 6 (GNN+LSTM) sí requieren Fase 3 cerrada. La numeración se conserva por convención académica (dataset → baselines → DL es el orden estándar esperado por el revisor del paper).

**Objetivo de la fase:** Establecer la línea base obligatoria que el DL debe superar. Sin esto no hay paper.

- [x] B0 — Media global por slot (train only).
- [x] B1 — Baseline ingenuo / persistencia (`y_pred(t+h) = y(t)`).
- [x] B2 — Promedio móvil (w ∈ {5, 10, 15}).
- [x] B3 — Suavizado exponencial simple (SES α=0.3).
- [x] B4 — Promedio histórico por hora-del-día (HA).
- [x] Evaluación sobre test con MAE y RMSE por horizonte.
- [x] Tabla de métricas baseline congelada (Kaggle NB06 v8, 84 filas, data v8 post-fix H7).

**Artefacto:** Módulo `src/baselines/statistical.py` + `src/baselines/harness.py` + notebook `06_baselines_stat` (Kaggle kernel `alexhuaracha/06-baselines-stat`).

**Criterio de cierre:** Tabla de métricas baseline publicada y congelada como referencia.

**Estado:** Completada (2026-05-23).

---

## Fase 5 — Baseline de Deep Learning (LSTM)

**Objetivo de la fase:** LSTM sobre el vector aplanado de headways. Aísla la contribución de cualquier componente espacial que se añada en Fase 6.

- [x] Arquitectura LSTM definida (encoder seq2one o seq2seq).
- [x] Loss con máscara para cardinalidad variable.
- [x] Loop de entrenamiento con early stopping y checkpointing.
- [x] Hyperparameter search acotado (hidden, capas, dropout, lr).
- [x] Logging de experimentos.
- [x] Métricas sobre test.

**Artefacto:** `src/models/lstm.py` + `src/train.py` + notebook `07_lstm` (Kaggle kernel `alexhuaracha/07-lstm`) + checkpoints.

**Criterio de cierre:** LSTM entrenado y evaluado en Kaggle. Métricas sobre test publicadas y comparadas con baselines (MAE y RMSE por horizonte, significancia verificada).

**Estado:** Completada (2026-05-27). LSTM supera todos los baselines: E2 aggregate MAE 4.47 min (-6.4% vs B3), E59 aggregate MAE 3.34 min (-4.8% vs B3). Grid search en Kaggle GPU T4, ~6h. Integridad verificada: split temporal, mismo test set que baselines, denormalización correcta.

---

## Fase 6 — Modelo principal espacial-temporal

**Objetivo de la fase:** Evaluar al menos una arquitectura que modele explícitamente las relaciones espaciales entre buses. La arquitectura ganadora será el modelo central del paper.

- [ ] Selección de candidatas (GNN+LSTM, Transformer con atención entre buses, STGCN).
- [ ] Implementación de la primera candidata.
- [ ] Definición de grafo / mecanismo de atención (vecindad en `s`).
- [ ] Manejo de cardinalidad variable en la dimensión espacial.
- [ ] Entrenamiento y hyperparameter search.
- [ ] Comparación con LSTM puro.
- [ ] (Si la primera no supera al LSTM) implementación de candidata alternativa.
- [ ] Selección de la arquitectura final para el paper.

**Artefacto:** Módulo `src/models/<arquitectura>.py` + notebook `08_modelo_espacial` + checkpoints.

**Criterio de cierre:** Al menos una arquitectura espacial-temporal entrenada con métricas registradas, comparable contra LSTM y baselines.

---

## Fase 7 — Evaluación comparativa

**Objetivo de la fase:** Comparar rigurosamente todos los modelos en las empresas seleccionadas y producir las tablas/figuras del paper.

- [ ] MAE y RMSE por modelo, empresa y horizonte.
- [ ] Test de significancia estadística (Diebold-Mariano o Wilcoxon pareado).
- [ ] Análisis por franja horaria (pico vs. valle).
- [ ] Análisis por posición en el vector de headways.
- [ ] Robustez frente a días atípicos.
- [ ] Tablas y figuras candidatas a paper, congeladas.

**Artefacto:** Notebook `09_evaluacion` + carpeta `results/` con CSVs crudos.

**Criterio de cierre:** Tablas y figuras del paper aprobadas; el criterio de éxito de `objetivo.md` (p<0.05) está verificado en Empresas 2 y 59.

---

## Fase 8 — Análisis cualitativo y casos de estudio

**Objetivo de la fase:** Demostrar narrativamente que el modelo detecta anticipadamente bunching, gaps y congestión. Sustento para la sección de discusión del paper.

- [ ] Identificación de eventos reales de bunching en el dataset.
- [ ] Identificación de gaps de servicio.
- [ ] Identificación de episodios de congestión colectiva.
- [ ] Visualizaciones espacio-temporales (diagramas tiempo–espacio).
- [ ] Métrica operacional simple (ej. minutos de anticipación con error < X).
- [ ] Al menos 3 casos por fenómeno.

**Artefacto:** Notebook `10_casos_estudio` + figuras para la discusión.

**Criterio de cierre:** 3+ casos por fenómeno documentados con predicción vs. realidad vs. baseline.

---

## Fase 9 — Redacción del paper y reproducibilidad

**Objetivo de la fase:** Producir el manuscrito y dejar el repositorio reproducible.

- [ ] Estructura IJACSA: intro, related work, datos, método, experimentos, resultados, discusión, conclusión.
- [ ] Primer borrador completo.
- [ ] Iteración con asesores.
- [ ] README final con instrucciones de reproducción end-to-end.
- [ ] Congelar versiones (`uv.lock`), seeds y publicar checkpoints.
- [ ] Revisión final de figuras y tablas.
- [ ] Submission a IJACSA.

**Artefacto:** Manuscrito en `paper/` + README de reproducibilidad.

**Criterio de cierre:** Manuscrito enviado.

---

## Estructura de archivos esperada al final del proyecto

```
src/
  preprocessing/{corridor,projection,direction,trips,headways,config}.py
  data/dataset.py
  baselines/statistical.py
  models/{lstm,<arquitectura_final>}.py
  train.py
  evaluate.py
notebooks/
  01_viability_and_filter/    [✓]
  02_eda_corredores/          [✓]
  03_headway_viability/       [✓]  (viability probe — Opción C.2)
  04_preprocessing/
  05_dataset/
  06_baselines_stat/
  07_lstm/
  08_modelo_espacial/
  09_evaluacion/
  10_casos_estudio/
data/
  raw/                        [✓]
  processed/cleaned_gps_<empresa>.parquet
  processed/headways_<empresa>.parquet
models/{lstm,modelo_final}/
results/
paper/
```
