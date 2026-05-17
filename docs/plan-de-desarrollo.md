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
- [x] `clean_gps.parquet` producido (~99M filas dedup, 4 corredores).
- [x] Propuesta actualizada con números reales del análisis.

**Artefacto:** `notebooks/01_viability_and_filter/01_viability_and_filter.ipynb` + `data/processed/{viability,sensitivity}.csv` + `data/processed/figuras/` + `data/processed/clean_gps.parquet`.

**Estado:** Completada.

---

## Fase 1 — EDA dirigido sobre los corredores seleccionados

**Objetivo de la fase:** Entender la calidad y dinámica temporal/espacial de los datos GPS de cada corredor antes de procesarlos. Detectar problemas de calidad que impactarán el preprocesamiento.

- [ ] Distribución temporal: registros por hora/día/mes por empresa.
- [ ] Detección de gaps de servicio y días atípicos.
- [ ] Heatmaps espaciales por empresa (validar trazado del corredor).
- [ ] Estadísticas por unidad: viajes/día, horas activas, distancia recorrida.
- [ ] Diagnóstico de calidad GPS: velocidades imposibles, saltos, duplicados.
- [ ] Análisis de heading: ¿permite distinguir ida/vuelta de forma confiable?
- [ ] Documentar problemas encontrados y decisiones de limpieza para Fase 2.

**Artefacto:** `notebooks/02_eda_corredores/02_eda_corredores.ipynb`.

**Criterio de cierre:** Tabla de problemas de calidad por empresa documentada y decisiones de limpieza aprobadas.

---

## Fase 2 — Preprocesamiento y cálculo de headways

**Objetivo de la fase:** Producir series temporales limpias de headways por corredor y sentido, listas para alimentar al modelo. Esta es la fase más crítica: la calidad de los headways determina el techo de rendimiento de todos los modelos.

- [ ] Reconstrucción del trazado del corredor (median path / centerline).
- [ ] Proyección lineal: convertir (lat, lon) en distancia acumulada `s`.
- [ ] Identificación de sentido ida/vuelta (heading + derivada de `s`).
- [ ] Segmentación de viajes (terminal a terminal).
- [ ] Definición operativa de headway (espacial vs. temporal — decidir y documentar).
- [ ] Cálculo de headways en grilla temporal regular.
- [ ] Estrategia de cardinalidad variable (buses que entran/salen del corredor).
- [ ] Validación visual sobre muestras (lado a lado con GPS crudo).
- [ ] Módulo `src/preprocessing/` reutilizable.
- [ ] Notebook documentando decisiones por empresa.
- [ ] Dataset intermedio `data/processed/headways_<empresa>.parquet`.

**Artefacto:** Módulo `src/preprocessing/` + notebook `03_preprocessing` + datasets parquet por empresa.

**Criterio de cierre:** Para Empresas 2 y 59, series de headways validadas, sin huecos no documentados, listas para Fase 3.

---

## Fase 3 — Construcción del dataset supervisado

**Objetivo de la fase:** Transformar las series de headways en pares (X, y) para entrenamiento, con splits temporales sin leakage.

- [ ] Definición de ventana de entrada `T_in` y horizonte `T_out`.
- [ ] Generación de ventanas deslizantes por corredor y sentido.
- [ ] Split train/val/test temporal (cronológico, no aleatorio).
- [ ] Normalización ajustada solo sobre train.
- [ ] Codificación de features de contexto (hora cíclica, día de semana).
- [ ] Implementación de máscara para cardinalidad variable.
- [ ] DataLoader / Dataset reutilizable.
- [ ] Notebook con estadísticas de splits y verificación de sanidad.

**Artefacto:** Módulo `src/data/dataset.py` + notebook `04_dataset`.

**Criterio de cierre:** DataLoaders reproducibles entregando tensores con shapes documentados, sin leakage entre splits.

---

## Fase 4 — Baselines estadísticos

**Objetivo de la fase:** Establecer la línea base obligatoria que el DL debe superar. Sin esto no hay paper.

- [ ] Implementación de baseline ingenuo (`y_pred(t+h) = y(t)`).
- [ ] Implementación de promedio móvil.
- [ ] (Opcional) Promedio histórico por hora-del-día.
- [ ] Evaluación sobre test con MAE y RMSE por horizonte.
- [ ] Tabla de métricas baseline congelada.

**Artefacto:** Módulo `src/baselines/statistical.py` + notebook `05_baselines_stat`.

**Criterio de cierre:** Tabla de métricas baseline publicada y congelada como referencia.

---

## Fase 5 — Baseline de Deep Learning (LSTM)

**Objetivo de la fase:** LSTM sobre el vector aplanado de headways. Aísla la contribución de cualquier componente espacial que se añada en Fase 6.

- [ ] Arquitectura LSTM definida (encoder seq2one o seq2seq).
- [ ] Loss con máscara para cardinalidad variable.
- [ ] Loop de entrenamiento con early stopping y checkpointing.
- [ ] Hyperparameter search acotado (hidden, capas, dropout, lr).
- [ ] Logging de experimentos.
- [ ] Métricas sobre test.

**Artefacto:** Módulo `src/models/lstm.py` + `src/train.py` + notebook `06_lstm` + checkpoints.

**Criterio de cierre:** LSTM supera baselines estadísticos en MAE/RMSE con significancia.

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

**Artefacto:** Módulo `src/models/<arquitectura>.py` + notebook `07_modelo_espacial` + checkpoints.

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

**Artefacto:** Notebook `08_evaluacion` + carpeta `results/` con CSVs crudos.

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

**Artefacto:** Notebook `09_casos_estudio` + figuras para la discusión.

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
  preprocessing/{corridor,projection,direction,trips,headways}.py
  data/dataset.py
  baselines/statistical.py
  models/{lstm,<arquitectura_final>}.py
  train.py
  evaluate.py
notebooks/
  01_viability_and_filter/    [✓]
  02_eda_corredores/
  03_preprocessing/
  04_dataset/
  05_baselines_stat/
  06_lstm/
  07_modelo_espacial/
  08_evaluacion/
  09_casos_estudio/
data/
  raw/                        [✓]
  processed/headways_<empresa>.parquet
models/{lstm,modelo_final}/
results/
paper/
```
