# Mejoras pendientes para blindar los resultados del paper

Este documento registra, desde cero, qué debe revisarse antes de usar `docs/resultados/documento-resultados.md` como base final del paper. Está escrito como **plan de trazabilidad pendiente**: no asume que las correcciones ya fueron aplicadas ni que los resultados ya fueron regenerados.

## Resumen ejecutivo

El core experimental es prometedor, pero para elevar la aceptabilidad del paper hacia un rango estimado de **82–85 %** se debe blindar la metodología de evaluación. El objetivo no es cambiar la tesis ni repetir todo el trabajo: es asegurar que las comparaciones DL vs baselines sean justas, pareadas y reproducibles.

La mejora principal consiste en:

1. Usar una evaluación pareada y trazable.
2. Garantizar el mismo preprocesamiento para DL, persistencia, XGBoost y baselines clásicos.
3. Regenerar los artefactos necesarios solo después de corregir el pipeline.
4. Actualizar el documento de resultados con los valores corregidos, si cambian.

## Qué estaba mal o débil

| Tema | Problema | Riesgo para el paper |
|---|---|---|
| Evaluación agregada vs pareada | Los resultados agregados y los tests de significancia pueden estar calculados sobre conjuntos de muestras distintos. | Un revisor puede cuestionar que la comparación DL vs persistencia no sea exactamente sobre las mismas observaciones. |
| Winsorización inconsistente | El contrato metodológico indica calcular el p99 en train y aplicar el umbral a todos los splits, pero los builders DL deben verificarse porque podrían estar aplicándolo solo a train. | Comparación injusta: baselines y DL podrían evaluar objetivos/preprocesamiento ligeramente distintos. |
| Notebooks generados | Los notebooks usados en Kaggle son artefactos generados desde builders. Si el builder cambia, el notebook queda desactualizado hasta regenerarse explícitamente. | Se puede creer que el código fuente está corregido, pero Kaggle seguiría ejecutando lógica vieja. |
| Volatilidad ex-ante | Los terciles de volatilidad ex-ante deben fijarse con train/validación, no ajustarse mirando test. | La afirmación “operativamente ejecutable en vivo” queda vulnerable si el umbral se calibra con test. |
| Reproducibilidad externa | Los residuos por muestra no están versionados en Git y los datos limpios dependen de Kaggle. | Un reviewer puede pedir una ruta clara para reproducir tablas, significancia y figuras. |
| Claim contra XGBoost | En E4, XGBoost compite fuerte y gana en horizontes cortos/medios. | El paper no debe vender “DL gana siempre”; debe formular la ventaja como condicional a horizonte, volatilidad y escala del corredor. |

## Problema 1 — Evaluación agregada vs evaluación pareada

### Qué se debe verificar

Los resultados principales reportan métricas agregadas por modelo, corredor y horizonte. En paralelo, la significancia estadística usa residuos por muestra, donde DL y persistencia predicen exactamente el mismo target.

La comparación más fuerte para publicación debe ser la pareada:

```text
misma muestra
mismo target real
predicción DL
predicción persistencia
misma métrica
```

### Archivos involucrados

| Archivo o ruta | Rol |
|---|---|
| `src/evaluation/significance.py` | Calcula diferencias pareadas de error y tests DM/Wilcoxon. |
| `src/evaluation/degradation.py` | Consolida métricas agregadas para la curva de degradación. |
| `docs/resultados/residuos-multihorizon/` | Contiene residuos por muestra necesarios para evaluación pareada. |
| `docs/resultados/csv-multihorizon/*_results_*.csv` | Contiene resultados agregados reportados. |
| `docs/resultados/documento-resultados.md` | Documento que debe quedar alineado con la métrica canónica. |

### Mejora requerida

Crear una auditoría que calcule, desde los residuos por muestra:

- MAE pareado DL.
- RMSE pareado DL.
- MAE pareado persistencia.
- RMSE pareado persistencia.
- Diferencia DL − persistencia.
- Comparación contra los valores agregados actualmente reportados.
- Marcador de cambio de signo cuando el ganador cambia.

### Artefactos esperados

| Artefacto propuesto | Propósito |
|---|---|
| `src/evaluation/paired_audit.py` | Módulo reutilizable para calcular métricas pareadas desde residuos. |
| `src/build_paired_audit.py` | Script para generar los CSV de auditoría. |
| `docs/resultados/csv-multihorizon/paired_dl_persistence_metrics.csv` | Tabla canónica de métricas pareadas DL vs persistencia. |
| `docs/resultados/csv-multihorizon/paired_vs_reported_audit.csv` | Comparación entre métricas pareadas y métricas agregadas reportadas. |
| `tests/evaluation/test_paired_audit.py` | Tests para discovery de residuos, cálculo de métricas y joins de auditoría. |

### Criterio de aceptación

- La auditoría no debe incluir archivos `*_results_*.csv` ni logs como si fueran residuos.
- Debe fallar si faltan columnas requeridas.
- Debe fallar si el horizonte dentro del CSV no coincide con el horizonte esperado.
- Debe fallar si faltan o se duplican métricas reportadas al comparar.
- Debe listar explícitamente cualquier celda donde cambie el ganador.

## Problema 2 — Winsorización inconsistente entre baselines y DL

### Qué se debe verificar

El contrato metodológico esperado es:

1. Dividir temporalmente en train/val/test.
2. Calcular el umbral p99 de `delta_t_min` usando solo train.
3. Aplicar ese umbral a todos los splits: train, val y test.

Esto evita leakage porque el umbral viene de train, pero mantiene la misma distribución capada para todos los modelos.

### Evidencia a revisar

| Archivo | Qué revisar |
|---|---|
| `src/evaluation/splits.py` | El contrato de `winsorize_train_p99`: calcular umbral en train y aplicarlo al frame recibido. |
| `src/baselines/harness.py` | Los baselines deben llamar `winsorize_train_p99` sobre el frame completo ya dividido. |
| `src/build_notebook_11.py` | Builder LSTM multi-horizonte. |
| `src/build_notebook_12.py` | Builder SpatialConvLSTM multi-horizonte. |
| `src/build_notebook_13.py` | Builder SpatialTransformer multi-horizonte. |
| `src/build_notebook_17_e4_lstm.py` | Builder LSTM para E4. |
| `src/build_notebook_18_e4_convlstm.py` | Builder SpatialConvLSTM para E4. |
| `src/build_notebook_19_e4_transformer.py` | Builder SpatialTransformer para E4. |
| `src/build_exante_volatility.py` | Reconstrucción de datos para análisis ex-ante. |

### Qué está mal si aparece el patrón viejo

Este patrón es metodológicamente riesgoso:

```python
df_split = split_temporal(hw)
train_df = df_split.filter(pl.col("split") == "train")
df_winsor, threshold = winsorize_train_p99(train_df)
non_train = df_split.filter(pl.col("split") != "train")
df_full = pl.concat([df_winsor, non_train])
```

El problema es que `val` y `test` quedan sin clipping, mientras que los baselines sí pueden estar usando clipping en todos los splits. Eso no necesariamente cambia la tesis, pero deja una grieta metodológica.

### Lógica correcta esperada

```python
df_split = split_temporal(hw)
df_winsor, threshold = winsorize_train_p99(df_split)
return df_winsor
```

### Mejora requerida

- Alinear todos los builders DL con el mismo contrato que usan los baselines.
- Agregar tests que fallen si vuelve a aparecer el patrón `train_df + non_train`.
- Después de corregir builders, regenerar notebooks de Kaggle de manera explícita.

### Criterio de aceptación

- Los builders DL deben aplicar `winsorize_train_p99` al frame completo con split.
- `src/build_exante_volatility.py` debe reconstruir con el mismo contrato.
- Deben existir tests que bloqueen regresión a winsorización solo en train.

## Problema 3 — Notebooks Kaggle desactualizados

### Qué se debe entender

Los notebooks en `notebooks/` son artefactos generados. Si se corrige un builder, eso no actualiza automáticamente los notebooks ya versionados.

Por lo tanto, hay dos niveles distintos:

| Nivel | Qué significa |
|---|---|
| Builder corregido | El código fuente que genera notebooks ya contiene la lógica correcta. |
| Notebook regenerado | El `.ipynb` que se subirá o ejecutará en Kaggle ya incluye esa lógica correcta. |

### Mejora requerida

Después de corregir builders:

1. Regenerar notebooks de forma explícita.
2. Revisar que no aparezcan cambios accidentales fuera de los notebooks esperados.
3. Subir/correr los kernels corregidos en Kaggle.
4. Descargar nuevos outputs.

### Archivos involucrados

| Ruta | Rol |
|---|---|
| `notebooks/11_lstm_multihorizon/` | Notebooks LSTM multi-horizonte. |
| `notebooks/12_spatial_conv_lstm_multihorizon/` | Notebooks SpatialConvLSTM multi-horizonte. |
| `notebooks/13_spatial_transformer_multihorizon/` | Notebooks SpatialTransformer multi-horizonte. |
| `notebooks/17_e4_lstm/` | Notebooks LSTM E4. |
| `notebooks/18_e4_convlstm/` | Notebooks SpatialConvLSTM E4. |
| `notebooks/19_e4_transformer/` | Notebooks SpatialTransformer E4. |

## Problema 4 — Volatilidad ex-ante con umbrales calibrados correctamente

### Qué se debe verificar

El análisis ex-ante busca demostrar que el régimen de volatilidad puede conocerse antes de predecir. Para que esa afirmación sea operativa, los umbrales de baja/media/alta volatilidad no deben depender del test.

### Riesgo

Si los terciles se calculan sobre test, el análisis sigue siendo descriptivo, pero la frase “ejecutable en vivo” queda más débil.

### Mejora requerida

- Calcular umbrales de volatilidad reciente usando train o validación.
- Congelar esos umbrales.
- Aplicarlos sobre test.
- Regenerar:
  - `exante_volatility_multihorizon.csv`
  - `exante_correlation_multihorizon.csv`, si depende de la misma asignación de régimen.
  - `volatilidad-exante.png`.

### Archivos involucrados

| Archivo | Rol |
|---|---|
| `src/build_exante_volatility.py` | Genera estratificación ex-ante. |
| `src/build_exante_correlation.py` | Evalúa correlación entre volatilidad ex-ante y régimen retrospectivo. |
| `docs/resultados/csv-multihorizon/exante_volatility_multihorizon.csv` | Tabla ex-ante actual. |
| `docs/resultados/csv-multihorizon/exante_correlation_multihorizon.csv` | Tabla de correlación actual. |
| `docs/resultados/volatilidad-exante.png` | Figura ex-ante del documento. |

## Problema 5 — Reproducibilidad para reviewers

### Qué falta blindar

El paper no necesita meter datos pesados en Git, pero sí debe dar una ruta clara para reconstruir resultados.

### Mejora requerida

Crear una sección o documento de reproducción con:

1. Comando para instalar entorno:
   ```bash
   uv sync
   ```
2. Comandos para descargar datos/outputs desde Kaggle.
3. Comandos para reconstruir tablas y figuras.
4. Lista de artefactos que no se versionan en Git y dónde obtenerlos.
5. Versiones de datasets/kernels Kaggle usados.

### Archivos involucrados

| Archivo | Rol |
|---|---|
| `README.md` | Debe tener ruta mínima de setup/reproducción. |
| `docs/dataset-manifest.md` | Pin de datasets y kernels. |
| `.gitignore` | Explica qué resultados pesados no van a Git. |
| `docs/resultados/residuos-multihorizon/` | Residuos necesarios para significancia y auditoría pareada. |

## Qué NO se debe hacer

No es necesario:

- Repetir el grid search completo.
- Cambiar de arquitectura.
- Rehacer todo desde cero.
- Abandonar el argumento central del paper.

Lo máximo esperable para dejar los resultados publicables es:

- Corregir preprocessing.
- Regenerar notebooks.
- Re-ejecutar kernels DL con configs congeladas.
- Regenerar CSVs/figuras/documento.

Eso es una recertificación del experimento, no otro mes de búsqueda de hiperparámetros.

## Orden recomendado de trabajo

### Fase 1 — Auditoría sin reentrenar

- [ ] Crear auditoría pareada desde residuos existentes.
- [ ] Comparar métricas pareadas vs métricas agregadas reportadas.
- [ ] Identificar celdas con cambio de signo.
- [ ] Documentar si el hallazgo cambia o no la tesis.

### Fase 2 — Corrección metodológica del código fuente

- [ ] Alinear winsorización de builders DL con baselines.
- [ ] Alinear reconstrucción ex-ante con el mismo contrato.
- [ ] Agregar tests de contrato de winsorización.
- [ ] Agregar tests de auditoría pareada.

### Fase 3 — Regeneración controlada

- [ ] Regenerar notebooks desde builders corregidos.
- [ ] Verificar que los notebooks ya no contienen el patrón viejo.
- [ ] Ejecutar kernels corregidos en Kaggle con configs congeladas.
- [ ] Descargar outputs nuevos.

### Fase 4 — Recalcular resultados

- [ ] Regenerar CSVs de resultados DL.
- [ ] Regenerar residuos por muestra.
- [ ] Regenerar significancia.
- [ ] Regenerar curva de degradación.
- [ ] Regenerar volatilidad retrospectiva y ex-ante.
- [ ] Actualizar `docs/resultados/documento-resultados.md`.

### Fase 5 — Decisión final de publicación

- [ ] Confirmar que la tesis central se sostiene.
- [ ] Confirmar si alguna celda cambia de ganador.
- [ ] Ajustar redacción para no sobre-vender.
- [ ] Preparar paquete reproducible para reviewer.

## Cómo sube la aceptabilidad

| Mejora | Impacto esperado |
|---|---|
| Evaluación pareada | Reduce el riesgo de crítica por comparación no equivalente. |
| Winsorización consistente | Cierra el principal flanco metodológico entre baselines y DL. |
| Notebooks regenerados | Garantiza que Kaggle ejecuta la lógica corregida. |
| Ex-ante con umbrales train/val | Fortalece la afirmación operativa del paper. |
| Reproducibilidad documentada | Facilita revisión y réplica externa. |
| Claim matizado contra XGBoost | Evita una objeción por sobre-generalización. |

## Decisión metodológica recomendada

Para el paper, la tabla canónica debería basarse en una comparación pareada cuando se compare DL vs persistencia. Los agregados globales pueden mantenerse como figura descriptiva, pero cualquier afirmación fuerte de “gana A sobre B” debe poder rastrearse a una comparación sobre las mismas muestras.

La conclusión correcta no debe ser “DL gana siempre”. Debe ser:

> El Deep Learning aporta valor en horizontes operativos y en regímenes de mayor volatilidad; frente a XGBoost, la ventaja es clara en corredores grandes y más débil en el corredor pequeño.

Esa formulación es más defendible y aumenta la probabilidad de aceptación.
