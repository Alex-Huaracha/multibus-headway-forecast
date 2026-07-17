# Resultados — Fase 6b: SpatialTransformer

> **Estado**: COMPLETO — E2 (09a) y E59 (09b1 + 09b2) cerrados.
> **Última actualización**: 2026-06-14
> **Modelo**: `SpatialTransformer` (atención multi-cabeza espacial + LSTM temporal) — `src/models/spatial_transformer.py`

## Resumen ejecutivo

**La atención espacial no mejora la predicción en ninguno de los dos corredores.** El
SpatialTransformer queda marginalmente *por debajo* del LSTM en ambos: en E2, MAE agregado
**4.4949 min** vs **4.4707** del LSTM (+0.024 min); en E59, **3.3590 min** vs **3.3375**
(+0.022 min). Diferencias de ~1.3 s, empate técnico, pero siempre en la dirección equivocada.

Esto cierra la línea espacial con **dos corredores y dos mecanismos distintos**: ni la
convolución fija de la Fase 6a (que eligió `conv_channels=1`, apagando lo espacial) ni la
atención aprendida de la Fase 6b le sacan ventaja al LSTM plano. **La señal espacial entre
buses adyacentes es marginal frente a la dinámica temporal.**

> ### ⚠️ Hallazgo crítico para el paper: la persistencia (B1) gana en E59 sobre MAE
>
> Al consolidar la comparativa completa apareció algo que **debe reportarse con honestidad**:
> en **E59, el baseline de persistencia B1** (predecir el próximo headway = el último
> observado, `src/baselines/statistical.py:79`) da **MAE agregado 3.100 min**, **mejor que
> TODOS los modelos profundos** (LSTM/ConvLSTM/Transformer ≈ 3.337–3.359) y en ambas
> direcciones.
>
> El matiz que salva a los modelos profundos es el **RMSE**: B1 en E59 tiene RMSE agregado
> **5.712** contra **~4.67** de los modelos profundos — es decir, persistencia acierta el caso
> típico pero **explota en los outliers**. Los modelos profundos cambian un poco de MAE por
> mucha mejor robustez en la cola de la distribución.
>
> En **E2 el relato sí se sostiene**: el LSTM (MAE 4.471, RMSE 6.110) le gana a B1 (MAE 4.757,
> RMSE 7.636) en ambas métricas. Pero incluso ahí, **B1 es el mejor baseline por MAE** (4.757 <
> B3 4.777) — el "mejor baseline estadístico" del paper no es B3, es la persistencia.
>
> **Conclusión metodológica**: la afirmación "los modelos profundos superan a los baselines" es
> robusta **solo en E2 y solo cuando se mira RMSE**. En E59 sobre MAE, un baseline trivial gana.
> El paper debe enmarcar la contribución en términos de **robustez (RMSE) frente a outliers**,
> no de error promedio (MAE).

## Setup experimental

| Parámetro | Valor |
|---|---|
| Grid search | 32 configs: `nhead{1,2} × d_model{16,32} × hidden{32,64} × dropout{0,0.2} × lr{1e-3,5e-4}`, `num_layers=1` fijo (obs #417) |
| Early stopping | `patience=10`, `max_epochs=50` |
| Batch size | 128 |
| Ventana | `T_in=12`, `T_out=1`, `stride=1` |
| Hardware | Kaggle GPU T4 x2 |
| Partición | E2 en un kernel (`09a`); E59 dividido en dos tandas (`09b1` = grid[0:16], `09b2` = grid[16:32]) |

> El grid, la patience, el test split y la normalización son **idénticos** a los de Fase 5 y 6a;
> la partición por corredor/tanda es puramente de ejecución y no altera los resultados.

### Por qué E59 se dividió en dos tandas

Correr las 32 configs de E59 en un solo kernel **excedió el límite de 12h de Kaggle**
(`Version 2 was canceled after 43200.6s — timeout exceeded`; 43200s = 12h exactas). E59 es el
corredor más pesado: la atención escala con el número de celdas (N²), así que cada config tarda
más que en E2.

La solución sigue el **mismo patrón de partición que la Fase 6a** (`08b` ya se había dividido
por el límite de 12h): en lugar de un kernel con las 32 configs, dos kernels con 16 cada uno
(`09b1` = `TRANSFORMER_GRID[0:16]`, `09b2` = `TRANSFORMER_GRID[16:32]`).

**El split no fue solo correcto, fue necesario.** La tanda `09b1`, con apenas **16 configs,
tardó 10.47h** (37 674 s) — casi tocando el límite de 12h. El grid completo de 32 habría tardado
~21h, **casi el doble del límite**. No existía forma de meter E59 entero en una sola sesión.
(Para contraste, E2 con las 32 configs corrió en 6.54h: E59 es ~3× más pesado por config.)

> **Cuidado al cerrar E59**: cada tanda reporta su `best config` **dentro de su propio
> subconjunto** del grid. El verdadero mejor de E59 sale de **comparar el `best_val_loss` de
> 09b1 contra el de 09b2** y quedarse con el menor. No tomar el "best" de una sola tanda.

## E2 — Resultados

**Corrida**: `alexhuaracha/09a-spatialtransformer-e2`, completada en **6.54 h** (23 556 s),
`max_N=22`, 32 configs.

**Mejor configuración** (menor val loss): `nhead=1, d_model=16, hidden=64, dropout=0.0, lr=5e-4`
— val loss 0.8172 (epoch 1), 12 epochs entrenados.

### Comparativa (MAE agregado, minutos — menor es mejor)

| Modelo | MAE | RMSE |
|---|---|---|
| B4_HA | 5.259 | — |
| B2_w5 | 5.030 | — |
| B3 (SES) | 4.777 | — |
| B1 (persistencia — mejor baseline) | 4.757 | 7.636 |
| **LSTM** (Fase 5) | **4.4707** | **6.110** |
| **SpatialConvLSTM** (Fase 6a) | **4.4721** | **6.114** |
| **SpatialTransformer** (Fase 6b) | **4.4949** | **6.133** |

> En E2 los modelos profundos ganan limpio: el mejor baseline (B1, persistencia) da MAE 4.757 /
> RMSE 7.636, y los tres modelos profundos quedan por debajo en ambas métricas.

### Desglose por dirección — SpatialTransformer

| Dirección | MAE | RMSE | n válidos |
|---|---|---|---|
| −1 | 5.4090 | 6.9777 | 416,942 |
| +1 | 3.8140 | 5.4179 | 559,696 |
| **agregado** | **4.4949** | **6.1325** | 976,638 |

### Interpretación

- **SpatialTransformer ligeramente por debajo del LSTM**: +0.024 min de MAE (+0.5 %),
  marginalmente peor. En la práctica, empate técnico.
- El grid eligió `nhead=1` y `d_model=16` (la cabeza única y la dimensión de atención más
  chica del espacio de búsqueda) — la **mínima capacidad de atención** disponible, igual que en
  6a el grid había elegido `conv_channels=1`. El modelo vuelve a "apagar" lo espacial.
- Lectura: en E2, **ni siquiera una atención aprendida supera al LSTM plano**. La hipótesis
  espacial **no se sostiene en E2**, ahora confirmada con dos mecanismos espaciales distintos
  (convolución fija y atención).

## E59 — Resultados

Ambas tandas completaron en Kaggle. El grid de 32 configs se evaluó en dos kernels de 16:

| Tanda | Kernel | Mejor config | Val loss | Wall |
|---|---|---|---|---|
| 09b1 (grid[0:16]) | `alexhuaracha/09b1-spatialtransformer-e59` | `nhead=1, d_model=32, hidden=32, dropout=0.0, lr=5e-4` | 0.526799 (epoch 40) | 10.47 h |
| **09b2 (grid[16:32])** | `alexhuaracha/09b2-spatialtransformer-e59` | `nhead=2, d_model=32, hidden=64, dropout=0.2, lr=5e-4` | **0.526059 (epoch 3)** ← menor | — |

**Veredicto del merge**: `09b2` gana por `val_loss` (0.526059 < 0.526799), así que **su CSV es
el resultado oficial de E59**. En la práctica las dos tandas dan casi lo mismo (MAE agregado
3.359 vs 3.357), pero metodológicamente el ganador es el `argmin(val_loss)` sobre las 32 configs
= la mejor config de 09b2.

### Desglose por dirección — SpatialTransformer (oficial = 09b2)

| Dirección | MAE | RMSE | n válidos | LSTM MAE | ConvLSTM MAE |
|---|---|---|---|---|---|
| −1 | 3.5294 | 4.9022 | 1,615,636 | 3.5023 | 3.5041 |
| +1 | 3.1136 | 4.3793 | 1,121,843 | 3.1000 | 3.0966 |
| **agregado** | **3.3590** | **4.6949** | 2,737,479 | **3.3375** | **3.3371** |

### Comparativa completa (MAE / RMSE agregado, minutos — menor es mejor)

| Modelo | MAE | RMSE |
|---|---|---|
| **B1 (persistencia)** | **3.1004** | 5.7118 |
| SpatialConvLSTM (Fase 6a) | 3.3371 | **4.6720** |
| LSTM (Fase 5) | 3.3375 | 4.6671 |
| SpatialTransformer (Fase 6b) | 3.3590 | 4.6949 |
| B2_w5 | 3.6557 | — |
| B3 (SES) | 3.5052 | — |
| B4_HA | 4.8052 | — |
| B0 | 4.8618 | — |

### Interpretación

- **SpatialTransformer ligeramente por debajo del LSTM y del ConvLSTM** (3.3590 vs 3.3375 /
  3.3371): mismo empate técnico que en E2. La atención espacial no aporta.
- **B1 (persistencia) gana en MAE a todos los modelos profundos** (3.100 vs ~3.34) — ver el
  recuadro del resumen ejecutivo. Esto **no se observa en E2** y es específico de E59, un
  corredor de alta frecuencia donde "el próximo headway ≈ el actual" es un predictor muy fuerte
  del caso típico.
- **El RMSE invierte el ranking**: los modelos profundos (RMSE ~4.67) baten a B1 (5.712) por un
  margen amplio. Es la única métrica donde el aprendizaje profundo justifica su costo en E59.

## Conclusión

El SpatialTransformer **confirma en los dos corredores** lo que la Fase 6a había mostrado: la
relación espacial entre buses adyacentes no mejora la predicción del headway, ni con convolución
fija ni con atención aprendida. **La señal útil para predecir el próximo headway es temporal, no
espacial.**

Pero la consolidación de resultados deja una conclusión más fina y honesta para el paper:

1. **La línea espacial es un resultado nulo robusto** (dos corredores, dos mecanismos). Esto es
   un aporte válido: descarta una hipótesis razonable con evidencia limpia.
2. **El aprendizaje profundo justifica su costo solo vía RMSE.** Sobre MAE, la persistencia
   trivial (B1) gana en E59 y queda a centésimas en E2. El valor de los modelos profundos está
   en **acotar los errores grandes (outliers)**, no en bajar el error promedio. El paper debe
   enmarcarse así para no sobre-vender una mejora de MAE que en E59 directamente no existe.
