# Configuraciones ganadoras — grid search Fases 5, 6a y 6b

> **Qué es esto**: la mejor configuración de hiperparámetros (la que minimizó el
> `val_loss`) encontrada por el grid search de cada modelo, en cada corredor.
> **Por qué importa**: encontrar estas configuraciones costó ~1 mes de cuota GPU de
> Kaggle (24–32 versiones × 3 modelos × 2 corredores). Este archivo las blinda: ya no
> hay que volver a hacer el grid search. Para cualquier re-corrida (otro horizonte,
> otra métrica), se reusan estas perillas directamente.
> **Última actualización**: 2026-06-14
> **Fuente**: línea `Best config:` de los logs de cada kernel de Kaggle (ver columna "Kernel").

## Horizonte de estas corridas

Todas estas configuraciones se hallaron prediciendo a **1 paso = 1 minuto** de horizonte
(`T_in=12`, `T_out=1`, grilla de 60 s). Ver [`fase-6b-spatial-transformer.md`](../historico/fase-6b-spatial-transformer.md)
y el plan de re-corrida a 5 min en [`diagnostico-y-plan-paper.md`](../historico/diagnostico-y-plan-paper.md).

## Tabla de configuraciones ganadoras

| Modelo | Corredor | Configuración ganadora | Kernel Kaggle |
|---|---|---|---|
| **LSTM** (Fase 5) | E2 | `hidden=32, layers=1, dropout=0.0, lr=0.0005` | `alexhuaracha/07-lstm-baseline` |
| **LSTM** (Fase 5) | E59 | `hidden=32, layers=2, dropout=0.2, lr=0.0005` | `alexhuaracha/07-lstm-baseline` |
| **SpatialConvLSTM** (Fase 6a) | E2 | `conv_channels=1, hidden=32, layers=1, dropout=0.2, lr=0.0005` | `alexhuaracha/08a-spatialconvlstm-e2` |
| **SpatialConvLSTM** (Fase 6a) | E59 | `conv_channels=1, hidden=32, layers=2, dropout=0.2, lr=0.0005` | `alexhuaracha/08b-spatialconvlstm-e59` |
| **SpatialTransformer** (Fase 6b) | E2 | `nhead=1, d_model=16, hidden=64, dropout=0.0, lr=0.0005` | `alexhuaracha/09a-spatialtransformer-e2` |
| **SpatialTransformer** (Fase 6b) | E59 | `nhead=2, d_model=32, hidden=64, dropout=0.2, lr=0.0005` | `alexhuaracha/09b2-spatialtransformer-e59` |

### Notas por modelo

- **Tamaño del grid**: LSTM y ConvLSTM probaron **24 configuraciones**; el Transformer probó
  **32** (agrega el eje `nhead`).
- **`conv_channels=1` (ConvLSTM)** y **`nhead=1, d_model=16` (Transformer E2)**: el grid eligió
  la **mínima capacidad espacial** disponible — el modelo "apaga" lo espacial. Es una de las
  evidencias de que la señal espacial entre buses no aporta (ver doc de Fase 6b).
- **E59 SpatialTransformer**: el grid se partió en dos tandas (`09b1` = configs 0:16, `09b2` =
  16:32) por el límite de 12 h de Kaggle. Ganó la config de `09b2` (`val_loss` 0.526059 <
  0.526799 de `09b1`).

## Generación recertificada (notebook 21, LSTM contiguo)

> **Qué es esto**: las configuraciones que usa la línea publicada del paper. No salen del
> grid de 24 de la Fase 5. En **E2 y E59** se hereda la ganadora de aquella búsqueda como
> configuración única (`src/build_notebook_21_lstm_contiguous.py:79-80`), así que no hay
> búsqueda que registrar. En **E4** el builder ofrece **3 candidatas** (`:92-94`) y elige
> por horizonte sobre validación, de modo que la ganadora solo existía en el log del kernel.
> **Fuente**: línea `E4 winner:` de cada log, recuperada con
> `PYTHONUTF8=1 uv run --env-file .env kaggle kernels output alexhuaracha/21-lstm-contiguous-e4-<h>`.
> Sin `PYTHONUTF8=1` el cliente escribe el log vacío en Windows.
> **Última actualización**: 2026-08-29.

| Corredor | Horizonte | Configuración ganadora | `val_loss` | Época | Kernel Kaggle |
|---|---:|---|---:|---:|---|
| E2 | 1, 3, 5, 10 | `hidden=32, layers=1, dropout=0.0, lr=0.0005` (heredada) | — | — | `alexhuaracha/21-lstm-contiguous-h<h>` |
| E59 | 1, 3, 5, 10 | `hidden=32, layers=2, dropout=0.2, lr=0.0005` (heredada) | — | — | `alexhuaracha/21-lstm-contiguous-h<h>` |
| E4 | 1 | `hidden=32, layers=2, dropout=0.2, lr=0.0005` | 0.504767 | 48 | `alexhuaracha/21-lstm-contiguous-e4-h1` |
| E4 | 3 | `hidden=32, layers=1, dropout=0.0, lr=0.0005` | 0.690270 | 47 | `alexhuaracha/21-lstm-contiguous-e4-h3` |
| E4 | 5 | `hidden=32, layers=1, dropout=0.0, lr=0.0005` | 0.780777 | 49 | `alexhuaracha/21-lstm-contiguous-e4-h5` |
| E4 | 10 | `hidden=32, layers=1, dropout=0.0, lr=0.0005` | 0.899096 | 39 | `alexhuaracha/21-lstm-contiguous-e4-h10` |

- **E4 usa una capa en tres de los cuatro horizontes.** Solo h1 eligió la variante profunda
  con `dropout`. Las tres candidatas eran `(32, 1, 0.0)`, `(32, 2, 0.2)` y `(64, 1, 0.0)`,
  todas con `lr=0.0005`.
- **Asimetría de búsqueda frente al XGBoost**: la red recibió 3 candidatas por celda en E4 y
  ninguna en E2/E59, contra las 24 por celda de `src/baselines/fitted.py`
  (`SEARCH_N_CONFIGS`). El paper lo declara en la Sección IV-B y acota su alcance en la
  Sección VI.

## Cómo se halló cada una (recordatorio)

El grid search entrena todas las versiones y se queda con la de **menor `val_loss`** (error
sobre el conjunto de validación). Esa línea se imprime al final del entrenamiento de cada
corredor como `Best config: ...`. El detalle del espacio de búsqueda está en
`src/train.py` (`GRID`, `TRANSFORMER_GRID`).
