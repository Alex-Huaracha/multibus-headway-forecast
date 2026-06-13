# Resultados — Fase 6b: SpatialTransformer

> **Estado**: PARCIAL — E2 (09a) cerrado. E59: 09b1 corrido, 09b2 pendiente.
> **Última actualización**: 2026-06-08
> **Modelo**: `SpatialTransformer` (atención multi-cabeza espacial + LSTM temporal) — `src/models/spatial_transformer.py`

## Resumen ejecutivo

**En E2 la atención espacial tampoco mejora la predicción — de hecho queda marginalmente
por debajo del LSTM.** El SpatialTransformer da **MAE agregado 4.4949 min** contra los
**4.4707 min** del LSTM de Fase 5: **+0.024 min de error** (≈ 1.5 segundos peor), una
diferencia despreciable pero en la dirección equivocada.

Esto refuerza la lectura de la Fase 6a: en E2, **mirar a los buses vecinos no aporta señal
útil** que la historia temporal de cada posición no tenga ya. La Fase 6a lo mostró con
convolución espacial (el grid eligió `conv_channels=1`, apagando lo espacial); la Fase 6b lo
muestra con un mecanismo más expresivo —atención, que *aprende* qué relaciones importan— y aun
así no le saca ventaja al LSTM plano. Cuando ni una convolución fija ni una atención aprendida
mejoran el resultado, la conclusión es robusta: **la señal espacial entre buses adyacentes es
marginal frente a la dinámica temporal.**

Sigue ganándole a todos los baselines estadísticos (B3 = 4.777), pero ese ya no es el punto.

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
| B3 (mejor baseline estadístico) | 4.777 | — |
| **LSTM** (Fase 5) | **4.4707** | **6.110** |
| **SpatialConvLSTM** (Fase 6a) | **4.4721** | **6.114** |
| **SpatialTransformer** (Fase 6b) | **4.4949** | **6.133** |

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

## E59 — Resultados (PARCIAL — solo tanda 1)

> **Veredicto de E59 pendiente.** Lo de abajo es el mejor de `09b1` (configs 0:16). El ganador
> real de E59 sale de comparar este `val_loss` contra el de `09b2` (configs 16:32, aún sin
> correr). Si `09b2` trae una config con menor `val_loss`, su MAE reemplaza a este.

**Tanda 1** — `alexhuaracha/09b1-spatialtransformer-e59` (grid[0:16]), completada en **10.47 h**
(37 674 s), `max_N=19`, 16 configs.

**Mejor configuración de la tanda 1** (menor val loss): `nhead=1, d_model=32, hidden=32,
dropout=0.0, lr=5e-4` — val loss 0.5268 (epoch 40), 50 epochs entrenados.

### Desglose por dirección — SpatialTransformer (parcial, tanda 1)

| Dirección | MAE | RMSE | n válidos | LSTM MAE (Fase 5) |
|---|---|---|---|---|
| −1 | 3.5157 | 4.8871 | 1,615,636 | 3.50 |
| +1 | 3.1283 | 4.3892 | 1,121,843 | 3.10 |
| **agregado** | **3.3570** | **4.6894** | 2,737,479 | **3.34** |

### Lectura preliminar (sujeta a 09b2)

- El mejor de la tanda 1 da MAE agregado 3.3570 vs LSTM 3.34 → **mismo empate técnico** que en
  E2 y en la Fase 6a. El grid volvió a elegir `nhead=1` (cabeza única, mínima atención).
- Falta 09b2 para confirmar que ninguna config de la otra mitad mejora esto. Si 09b2 no baja del
  val loss 0.5268, este es el resultado final de E59.

**Pendiente**: `alexhuaracha/09b2-spatialtransformer-e59` (tanda 2, grid[16:32]).

## Conclusión (preliminar)

Con E2 cerrado, el SpatialTransformer **repite el resultado nulo de la Fase 6a**: la relación
espacial entre buses adyacentes no mejora la predicción del headway, ni con convolución fija ni
con atención aprendida. Falta E59 para confirmarlo en una segunda topología (como pasó en 6a),
pero la dirección es clara.

Si E59 confirma el patrón, la lectura de cierre de la línea espacial es sólida: **para predecir
el próximo headway, la dinámica temporal de cada posición ya contiene la señal útil; la
interacción con buses vecinos es marginal.**
