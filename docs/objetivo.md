# Objetivo del Proyecto

> **Reformulado el 2026-07-27.** La versión anterior definía el éxito como
> «el mejor modelo de DL supera a los baselines en MAE y RMSE con p < 0.05» y
> declaraba como aporte la predicción del vector completo de headways. La
> evidencia contradijo las dos cosas: el aprendiz gana el MAE pero **pierde toda
> capacidad de anticipar la irregularidad del servicio**, y esa disociación pasó
> a ser el aporte. Un documento de objetivos que define el éxito en términos que
> el trabajo ya refutó no es papel viejo — desalinea al autor de su propio paper.
> El registro de lo planificado está en [`propuesta.md`](./propuesta.md).

## Objetivo general

Publicar en IJACSA un paper que demuestre que **la métrica con la que se evalúa
un pronóstico de headways decide qué modelo gana**, y que una evaluación basada
en error escalar agregado es estructuralmente incapaz de detectar la pérdida de
información que ella misma premia. Validado sobre corredores reales de Arequipa
con datos exclusivamente GPS.

## Criterios de éxito

El proyecto se considera exitoso cuando se cumplen, simultáneamente:

1. **La disociación está medida y es robusta.** El aprendiz gana el MAE frente a
   la persistencia a horizontes largos, y pierde las métricas vectoriales
   (regularidad y detección conjunta de *bunching*) en las mismas celdas, con
   márgenes que crecen con el horizonte. — ✅ **Cumplido**
2. **La comparación es atribuible.** Los modelos comparados consumen la misma
   población por construcción, con el sesgo de encuadre medido y declarado.
   — ✅ **Cumplido** (0.001 min, contra 0.28–0.53 del pipeline anterior)
3. **Los verdictos resisten una prueba honesta.** Varianza agrupada por día de
   servicio, mediana reportada junto a la media, y las celdas sin victoria
   declarable identificadas como tales. — ✅ **Cumplido**
4. **El paper está redactado, revisado y enviado a IJACSA.** — ⬜ Pendiente

## Alcance

**Corredores evaluados:** E2, E59 y E4. E4 aporta validez externa acotada a la
escala de flota (19 buses), no geográfica.

La propuesta original declaraba también E58. **Nunca entró al pipeline**: no
tiene parquet procesado ni aparece en ningún resultado.

**Modelos comparados:**

| Familia | Rol |
|---|---|
| Persistencia (B1) | El rival serio, y —resultado inesperado— el mejor detector de *bunching* |
| XGBoost | El competidor aprendido. Muestra que el cruce es propiedad del problema, no del Deep Learning |
| LSTM | El modelo del titular |
| SpatialConvLSTM, SpatialTransformer | Nulo espacial: no superan al LSTM plano |

**No se construyó ninguna GNN.** La propuesta la planteaba; se implementaron dos
arquitecturas espaciales alternativas y ninguna aporta.

## Contribución central

La novedad es **metodológica**, no arquitectónica:

- Se muestra que el MAE **premia contraer**, que contraer aplana el vector de
  headways, y que el *bunching* **es** irregularidad — de modo que optimizar la
  métrica destruye justamente lo que el operador necesita ver.
- Se aporta el instrumental para detectarlo: perfil de error por posición, índice
  de regularidad e indicador de detección conjunta, sobre una población
  compartida verificable.
- El resultado es negativo y por eso vale: le sirve a cualquiera que evalúe
  pronóstico de transporte con error escalar, que es prácticamente todo el campo.

No se reclama una arquitectura nueva.

## Lo que NO es objetivo del proyecto

- **Anticipar *bunching*, huecos o congestión con estos modelos.** La propuesta
  original lo declaraba como aporte central; la medición mostró que no se
  sostiene, y decirlo con el número al lado es el aporte.
- Detección de anomalías como tarea principal.
- Despliegue en producción o dashboards para operadores.
- Optimización de rutas, flotas o frecuencias.
- Comparación con sistemas BRT formales.
- Predicción de demanda de pasajeros.

## Entregable final

Manuscrito enviado a IJACSA + repositorio reproducible: builders, contratos
verificados por tests, y cada cifra del documento de resultados atada a su CSV.
