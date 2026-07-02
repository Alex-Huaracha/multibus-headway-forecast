# Resultados — Fase 6: SpatialConvLSTM

> **Estado**: COMPLETO — E2 y E59 corridos en Kaggle.
> **Última actualización**: 2026-06-02
> **Modelo**: `SpatialConvLSTM` (Conv1D espacial + LSTM temporal) — `src/models/spatial_conv_lstm.py`

## Resumen ejecutivo

**El SpatialConvLSTM predice prácticamente igual que el LSTM de Fase 5.** La diferencia entre
ambos es mínima en los dos corredores:

- **E2**: 4.4721 vs 4.4707 min de error (MAE) → **0.0014 min de diferencia**, ≈ 0.08 segundos.
- **E59**: 3.3371 vs 3.34 min → **≈ 0.003 min de diferencia**, ≈ 0.2 segundos.

Sobre predicciones que rondan los 3–5 minutos, esas diferencias son **fracciones de segundo**:
para cualquier uso práctico, los dos modelos dan el mismo resultado.

¿Por qué? Porque la búsqueda de hiperparámetros, **pudiendo usar más capacidad de convolución
espacial (8 o 16 canales), eligió el mínimo (`conv_channels=1`) en ambos corredores** — es decir,
el modelo casi "apagó" la parte espacial por sí solo. Esto indica que **agregar la relación
espacial entre buses vecinos no le sirve para predecir mejor**: toda la señal útil ya estaba en
la historia temporal de cada posición, que es lo que el LSTM plano ya usaba.

Los dos modelos de deep learning siguen ganándole a todos los baselines estadísticos; lo que no
ocurre es que la convolución espacial le saque ventaja al LSTM.

## Setup experimental

| Parámetro | Valor |
|---|---|
| Grid search | 48 configs: `conv_channels{1,8,16} × hidden{32,64} × layers{1,2} × dropout{0,0.2} × lr{1e-3,5e-4}` |
| Early stopping | `patience=10`, `max_epochs=50` |
| Batch size | 128 |
| Ventana | `T_in=12`, `T_out=1`, `stride=1` |
| Hardware | Kaggle GPU T4 x2 |
| Partición | Un kernel por corredor (`08a` E2, `08b` E59) para entrar en el límite de 12h/sesión |

> El grid, la patience, el test split y la normalización son **idénticos** a los usados en
> la comparación de Fase 5; la partición por corredor es puramente de ejecución y no altera
> los resultados.

## E2 — Resultados

**Corrida**: `alexhuaracha/08a-spatialconvlstm-e2`, completada en **4.47 h**.

**Mejor configuración** (menor val loss): `conv_channels=1, hidden=32, layers=1, dropout=0.2, lr=5e-4`
— val loss 0.8124 (epoch 5), 16 epochs entrenados.

### Comparativa (MAE agregado, minutos — menor es mejor)

| Modelo | MAE | RMSE |
|---|---|---|
| B4_HA | 5.259 | — |
| B2_w5 | 5.030 | — |
| B3 (SES) | 4.777 | — |
| **LSTM** (Fase 5) | **4.4707** | **6.110** |
| **SpatialConvLSTM** (Fase 6) | **4.4721** | **6.114** |

### Desglose por dirección — SpatialConvLSTM

| Dirección | MAE | RMSE | n válidos |
|---|---|---|---|
| −1 | 5.3646 | 6.9528 | 416,942 |
| +1 | 3.8073 | 5.4060 | 559,696 |
| **agregado** | **4.4721** | **6.1144** | 976,638 |

### Interpretación

- **SpatialConvLSTM ≈ LSTM**: diferencia de 0.0014 min (0.03 %), marginalmente a favor del LSTM. Empate.
- El grid, pudiendo usar 8 o 16 canales de convolución espacial, **eligió `conv_channels=1`** —
  la mínima capacidad espacial posible. El modelo efectivamente **apagó** la convolución.
- Lectura: en E2 la **adyacencia espacial entre pares de buses no mejora** la predicción del
  headway respecto al LSTM plano. La hipótesis del proposal de Fase 6 **no se sostiene en E2**.

## E59 — Resultados

**Corrida**: `alexhuaracha/08b-spatialconvlstm-e59`, completada (`status COMPLETE`, ~11.3 h).

**Mejor configuración** (menor val loss): `conv_channels=1, hidden=32, layers=2, dropout=0.2, lr=5e-4`
— val loss 0.5235 (epoch 18), 29 epochs entrenados.
**Idéntica a la del LSTM E59 de Fase 5** (val loss 0.523) → con `conv_channels=1` el modelo
colapsa al LSTM plano.

### Comparativa (MAE agregado, minutos — menor es mejor)

| Modelo | MAE | RMSE |
|---|---|---|
| B3 (SES) | 3.51 | 5.09 |
| **LSTM** (Fase 5) | **3.34** | **4.67** |
| **SpatialConvLSTM** (Fase 6) | **3.3371** | **4.6720** |

### Desglose por dirección — SpatialConvLSTM

| Dirección | MAE | RMSE | LSTM MAE (Fase 5) |
|---|---|---|---|
| −1 | 3.5041 | 4.8770 | 3.50 |
| +1 | 3.0966 | 4.3599 | 3.10 |
| **agregado** | **3.3371** | **4.6720** | **3.34** |

### Interpretación

- **SpatialConvLSTM ≈ LSTM** también en E59: diferencia ≈ 0.003 min, despreciable. Empate.
- El grid **volvió a elegir `conv_channels=1`**, replicando exactamente la configuración
  ganadora del LSTM de Fase 5. La predicción por dirección es casi idéntica a la del LSTM.
- Lectura: el comportamiento de E2 **se confirma en una topología distinta** (`max_N=19`,
  más pares por dirección). El resultado nulo **no es un artefacto de un solo corredor**.

## Conclusión

Con **los dos corredores cerrados**, la respuesta es clara: **agregar convolución espacial no
cambia las predicciones.** El SpatialConvLSTM y el LSTM dan el mismo error de predicción
(diferencias de centésimas de minuto, imperceptibles en la práctica), y el propio modelo lo
"reconoce" al elegir la mínima capacidad de convolución posible en ambos corredores.

Que esto pase en E2 **y** en E59 —dos corredores con topologías distintas— muestra que no es
casualidad de un caso particular: para predecir el próximo headway, **mirar a los buses vecinos
no aporta nada que la historia temporal de cada bus no tenga ya.**

Esto deja dos caminos:

1. **Probar un mecanismo que *aprenda* qué relaciones espaciales importan** en lugar de imponer
   un filtro local fijo, como el **SpatialTransformer** (atención) — Fase 6, modelo siguiente.
2. **Cerrar la línea espacial** y concluir que, en estos datos, la interacción entre buses
   adyacentes es marginal frente a la dinámica temporal de cada posición.
