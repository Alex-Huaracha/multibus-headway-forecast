# Resultados — Fase 5: LSTM Baseline

## 1. Descripcion del modelo

**HeadwayLSTM** es un codificador LSTM plano (seq2one) entrenado para predecir el vector completo de headways de un corredor en el paso siguiente.

### Entradas

Cada muestra consiste en una secuencia de 12 snapshots consecutivos del corredor (1 hora de historia, muestreada cada 5 minutos). Cada snapshot contiene:

- Vector de headways de longitud `max_N`, z-normalizado por (corredor, direccion)
- 5 caracteristicas de contexto: sin/cos de la hora, sin/cos del dia de la semana, y conteo de buses activos

**Tensor de entrada:** `(B, 12, max_N + 5)`

### Salida

Vector de headways predicho para el paso 13: `(B, max_N)`

### Manejo de cardinalidad variable

Los buses entran y salen del corredor a lo largo del tiempo. Una mascara binaria indica que posiciones son validas en cada paso temporal. La funcion de perdida `masked_mse_loss` penaliza unicamente las posiciones validas, ignorando los huecos.

### Arquitectura

- Un modelo independiente por corredor (E2 y E59). Los modelos no comparten parametros.
- Arquitectura LSTM estandar; la configuracion final se selecciono mediante busqueda en grilla sobre el conjunto de validacion.

---

## 2. Division temporal

La division es estrictamente cronologica; no se aplica mezcla aleatoria en ningun momento.

| Conjunto | Periodo |
|----------|---------|
| Train    | 2023-10-01 a 2024-01-15 |
| Val      | 2024-01-16 a 2024-02-07 |
| Test     | 2024-02-08 a 2024-02-29 |

Las estadisticas de normalizacion (media y desviacion estandar) y el umbral de winsorization se calculan exclusivamente sobre el conjunto de entrenamiento. Se emplea la misma funcion `split_temporal` utilizada en los baselines (NB06), lo que garantiza comparabilidad directa.

---

## 3. Busqueda en grilla

Se evaluaron 24 configuraciones con las siguientes dimensiones:

| Hiperparametro | Valores |
|----------------|---------|
| hidden         | {32, 64, 128} |
| layers         | {1, 2} |
| dropout        | {0.0, 0.2} |
| lr             | {1e-3, 5e-4} |

- **EarlyStopping:** paciencia = 10 epocas, maximo 50 epocas
- **Criterio de seleccion:** menor val masked MSE loss
- **Metricas finales:** reportadas sobre el conjunto de TEST (no visto durante entrenamiento ni seleccion)
- **Entorno:** Kaggle GPU T4x2, tiempo total aproximado de 6 horas

### Configuraciones ganadoras

| Corredor | hidden | layers | dropout | lr     | Val loss | Epoca final |
|----------|--------|--------|---------|--------|----------|-------------|
| E2       | 32     | 1      | 0.0     | 0.0005 | 0.812    | 3/14        |
| E59      | 32     | 2      | 0.2     | 0.0005 | 0.523    | 15/26       |

Los modelos mas pequenos (hidden=32) ganaron en ambos corredores. E59 requirio dos capas y regularizacion via dropout, mientras que E2 convergio con una arquitectura mas simple y sin dropout. Esto sugiere que la senal temporal de headways no requiere alta capacidad de representacion.

---

## 4. Resultados

### 4.1 MAE por corredor y direccion (minutos)

| Corredor | Direccion | B0 MAE | B1 MAE | B2_w5 MAE | B3 MAE | B4_HA MAE | LSTM MAE |
|----------|-----------|--------|--------|-----------|--------|-----------|----------|
| E2       | +1        | 4.49   | 4.16   | 4.40      | 4.16   | 4.47      | **3.80** |
| E2       | -1        | 6.16   | 5.46   | 5.47      | 5.46   | 6.13      | **5.37** |
| E2       | aggregate | 5.29   | 4.78   | 4.91      | 4.78   | 5.26      | **4.47** |
| E59      | +1        | 4.97   | 3.54   | 3.32      | 3.27   | 4.91      | **3.10** |
| E59      | -1        | 4.77   | 4.00   | 3.84      | 3.69   | 4.72      | **3.50** |
| E59      | aggregate | 4.86   | 3.80   | 3.66      | 3.51   | 4.81      | **3.34** |

B0 = media global por slot (train only) | B1 = persistencia / naive (y_pred = y_actual) | B2_w5 = promedio movil 5 pasos | B3 = suavizado exponencial simple (SES, alpha=0.3) | B4_HA = promedio historico por hora del dia

### 4.2 RMSE (minutos) — LSTM vs. mejor baseline estadistico (B3)

| Corredor | Direccion | LSTM RMSE | B3 RMSE |
|----------|-----------|-----------|---------|
| E2       | +1        | 5.40      | 5.78    |
| E2       | -1        | 6.95      | 7.31    |
| E2       | aggregate | 6.11      | 6.55    |
| E59      | +1        | 4.36      | 4.63    |
| E59      | -1        | 4.87      | 5.09    |
| E59      | aggregate | 4.67      | 5.09    |

### 4.3 Mejora sobre el mejor baseline estadistico (B3)

| Corredor | MAE improvement | RMSE improvement |
|----------|-----------------|------------------|
| E2       | -6.4 %          | -6.7 %           |
| E59      | -4.8 %          | -8.3 %           |

El LSTM supera a todos los baselines en ambos corredores, en ambas direcciones y en ambas metricas.

### 4.4 Conteo de predicciones validas en test

El aggregate se calcula concatenando todos los escalares validos de ambas direcciones, no como promedio simple de los resultados por direccion.

| Corredor | Direccion | n_valid (escalares) |
|----------|-----------|---------------------|
| E2       | -1        | 416,942             |
| E2       | +1        | 559,696             |
| E59      | -1        | 1,615,636           |
| E59      | +1        | 1,121,843           |

---

## 5. Verificacion de integridad

| Punto | Estado | Descripcion |
|-------|--------|-------------|
| Data leakage | PASS | Split temporal estricto; estadisticas de normalizacion y umbral de winsorization calculados solo sobre train |
| Mismo test set que baselines | PASS | Misma funcion `split_temporal`, mismas fechas, mismos parquets |
| Denormalizacion correcta | PASS | Predicciones y targets denormalizados por (corredor, direccion) antes de calcular MAE y RMSE; valores en minutos reales |
| Aggregate correcto | PASS | Pooled scalars (concatenacion de validos de ambas direcciones), no promedio simple |
| n_valid coherente | PASS | Escalares individuales (cada slot bus-pair por timestep); magnitudes consistentes con el volumen de datos del corredor |
| Grid search vs. test | PASS | Seleccion por val loss; metricas finales sobre test set separado, nunca visto durante entrenamiento o seleccion |

---

## 6. Conclusion

Fase 5 demuestra que los patrones temporales de headways son predecibles con deep learning y que un LSTM plano supera de forma consistente a todos los baselines estadisticos evaluados en Fases 1-4.

Los modelos ganadores son llamativamente pequenos (hidden=32), lo que indica que la senal temporal de headways no requiere alta capacidad de representacion. La diferencia entre E2 y E59 (este ultimo necesitando dos capas y dropout) sugiere que la variabilidad intrinseca del corredor influye en la complejidad del modelo optimo.

Estos resultados justifican avanzar a Fase 6, donde se evaluara si incorporar estructura espacial (vecindad entre buses via GNN) mejora las predicciones sobre el LSTM plano. La pregunta abierta es cuanto aporta modelar las interacciones entre buses adyacentes sobre un modelo que solo usa la historia temporal de cada posicion de forma independiente.

---

## 7. Artefactos

| Tipo | Ruta / Identificador |
|------|----------------------|
| Modelo | `src/models/lstm.py` |
| Entrenamiento | `src/train.py` |
| Generador de notebook | `src/build_notebook_07.py` |
| Notebook | `notebooks/07_lstm/07_lstm.ipynb` |
| Kaggle kernel | `alexhuaracha/07-lstm-baseline` |
| Output principal | `lstm_results.csv` (12 filas, 5 columnas) |
| Tests | 253/253 passing (36 nuevos para Fase 5) |
