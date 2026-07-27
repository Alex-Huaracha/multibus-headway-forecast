# Plan de reentrenamiento: contrato previo

**Fecha:** 2026-07-27 · **Estado:** propuesta, sin ejecutar

Este documento se escribe **antes** de correr nada. Esa es su razón de ser: la
causa raíz de todo el retrabajo registrado en [`auditoria-hallazgos.md`](./auditoria-hallazgos.md)
fue validar al final en vez de al principio. Acá se fija qué se va a construir,
se verifica que el contrato sea correcto, y recién después se entrena.

Ninguna corrida de Kaggle se lanza hasta que las tres verificaciones de la
sección 5 pasen en local.

---

## 1. Qué se reentrena y qué no

| Componente | Decisión | Motivo |
|---|---|---|
| **LSTM** | **Reentrenar** — 3 corredores × 4 horizontes = 12 corridas GPU | Es el modelo del titular |
| **XGBoost (B5)** | **Reajustar** sobre la población nueva — CPU | Debe pelear en las mismas condiciones |
| **Persistencia (B1)** | Recalcular — sin costo | No se entrena |
| **ConvLSTM / SpatialTransformer** | **No se tocan** | Ver sección 4 |

**Criterio de selección del LSTM, declarado:** las tres arquitecturas empatan
dentro del ruido de semilla (márgenes 0.017–0.038 min contra un desvío por
semilla de 0.001–0.037). No se elige el LSTM por ganar — se elige por **empatar
al menor costo**. La parsimonia es el criterio, y el resultado nulo es su
justificación.

---

## 2. Los tres contratos que faltaban

Estos son los contratos que el repo nunca escribió, y en cuyo hueco vivían los
defectos. Cada uno necesita un test que falle cerrado.

### C1 — Identidad de la muestra

**Una muestra es una tupla `(empresaid, direction, start_ts, horizon)`.**

- El ancla es un **instante**, no un índice de fila.
- Cada objetivo se emite **exactamente una vez**. Se elimina la replicación por
  `pair_rank`, que hoy cuenta cada objetivo 2.4–5.4 veces y pondera el MAE por
  densidad de flota (pendiente #13).
- El `pair_rank` sigue existiendo como **posición dentro del vector**, no como
  eje de anclaje.

### C2 — Contigüidad temporal

**Una ventana es válida solo si sus marcas de tiempo son minutos consecutivos, y
el objetivo cae exactamente `horizon` minutos después del fin de la ventana.**

- Hoy `make_window_index` (`src/data/windowing.py:154-188`) corta por índice
  posicional y nunca verifica contigüidad. El horizonte nominal es un
  desplazamiento de filas, no de tiempo.
- Toda ventana que cruce frontera de día, corte de viaje (`GAP_CUT_SECONDS`) o
  caída de flota se **descarta**, no se usa.

### C3 — Frontera de información

**Ninguna feature puede usar información no disponible en el instante de
predecir.**

La bandera de día atípico viola esto por tres vías (pendiente #11):

| Defecto | Decisión |
|---|---|
| Umbral calculado sobre los 152 días, incluyendo test | Recalcular **solo sobre train** |
| Agregado de día completo (a las 08:00 no se conoce el total del día) | **Eliminar la feature**, o reemplazarla por un agregado acumulado hasta `t` |
| `context_features.py` descarta el `empresaid` → un día marcado para una empresa marca los tres corredores | Marcar **por `(empresaid, día)`** |

**Decisión tomada (2026-07-27): eliminar la bandera.** Su efecto medido es
≤ 0.04 min (nivelar el XGBoost con ella movió esa cantidad), así que el costo de
sacarla es despreciable y elimina la fuga entera en vez de mitigarla.

Se evaluó y se descartó arreglarla: el defecto grave es que la bandera es un
agregado del **día completo**, así que clasificar el 12 de febrero exige los datos
de todo el 12 de febrero. Recalcular el umbral solo sobre train corrige (a) y (c)
pero **no** corrige (b) — la fuga es por diseño, no por parametrización.

También se evaluó y se descartó reemplazarla por una bandera de **calendario**
(feriados y Carnaval por fecha conocida de antemano, sin fuga). Es una feature
nueva, no un arreglo, y agranda el alcance justo en la fase de cierre.

**Consecuencias declaradas de eliminarla:**

| Dónde pega | Qué hay que hacer |
|---|---|
| `documento-resultados.md` §2 (nivelado del XGBoost) | Reescribir: el XGBoost sigue nivelado porque **ambos** pierden la bandera |
| `tests/test_notebook_input_gate.py` y 10 archivos más | El contrato pasa de "la bandera es obligatoria y falla cerrado" a "la bandera no existe" |
| Amenaza del Carnaval | Queda **declarada y abierta**. Sin la bandera no hay mecanismo para días anómalos |

---

## 3. La regla de condiciones idénticas

**Este es el punto central del plan y el que hoy no se cumple.**

Hoy las dos familias construyen poblaciones distintas:

| Modelo | Qué predice | Consecuencia |
|---|---|---|
| XGBoost | Una predicción **por fila** de `headways_E*.parquet` | Incluye filas que el LSTM nunca ve |
| LSTM | Una predicción **por ventana** de `T_in + horizon` filas | Descarta arranques en frío; replica objetivos |

El arreglo de §2.1 emparejó las dos poblaciones **después** del hecho, con un
*join*. Eso corrige el reporte pero no la causa: siguen siendo dos poblaciones
construidas por separado que casualmente se logran cruzar.

**El contrato nuevo:**

> Se construye **un único índice de muestras** que cumple C1 y C2. Ese índice se
> materializa **una vez**, se congela con un SHA-256, y **tanto el LSTM como el
> XGBoost y la persistencia consumen exactamente ese índice**. Ninguno construye
> su propia población.

Consecuencia práctica: el emparejamiento deja de ser un paso de análisis y pasa
a ser una propiedad estructural. No hay *join* que pueda fallar porque nunca hubo
dos poblaciones.

---

## 4. Por qué el nulo espacial no se reentrena

Las tres arquitecturas corrieron sobre el **mismo pipeline defectuoso**. El
horizonte posicional, la replicación de objetivos y la bandera con fuga les
afectan **por igual**. Una comparación A-vs-B bajo un defecto compartido sigue
siendo una comparación justa: el defecto no favorece a ninguna.

Por lo tanto el resultado nulo —la complejidad espacial no mejora el
pronóstico— **se sostiene sin reentrenar** y se reporta como está.

**Dos condiciones no negociables:**

1. **Las tablas nunca se mezclan.** La comparación entre arquitecturas vive en su
   propia tabla, sobre el pipeline viejo. El titular (LSTM vs XGBoost vs
   persistencia) vive en otra, sobre el pipeline nuevo. Poner un LSTM arreglado
   junto a un ConvLSTM sin arreglar **reintroduce el defecto §2.1**.
2. **Se declara la limitación:** el empate entre arquitecturas se midió bajo el
   pipeline con el defecto de contigüidad. No está medido si se sostendría con el
   arreglo.

---

## 5. Verificaciones previas — nada se lanza sin estas

Esta sección es el antídoto directo a la causa raíz. Las tres corren **en local**
y deben pasar antes de la primera corrida en Kaggle.

| # | Verificación | Criterio de aprobación |
|---|---|---|
| V1 | Test de contigüidad sobre el índice nuevo | **Cero** ventanas con marcas no consecutivas; horizonte efectivo == nominal en el 100 % |
| V2 | Test de unicidad de la muestra | Cada `(empresaid, direction, start_ts, horizon)` aparece **exactamente una vez** |
| V3 | Test de población compartida | El índice que consume el XGBoost es **byte-idéntico** al que consume el LSTM (mismo SHA-256) |

Además, antes de entrenar se mide **cuánto encoge el conjunto**: aplicar C1 y C2
descarta ventanas. Si el test se reduce por debajo de un umbral que haga
inestables los intervalos, hay que saberlo **antes**, no después.

---

## 6. Exportación: clave completa, siempre

El retrabajo del kernel `20-xgb-paired-export` existió porque la exportación tiró
la clave. Hoy los residuos del DL exportan solo
`corridor, direction, horizon, y_true, y_pred_dl, y_pred_persist` — sin `t` ni
`pair_rank`, lo que hoy bloquea los pendientes #5 y #6.

**Contrato de exportación:** todo residuo por muestra lleva

```
corridor, direction, horizon, split, start_ts, t_target, pair_rank,
y_true, y_pred_<modelo>, y_pred_persist
```

Y el docstring incorrecto de `harness.py:71` —que declara `t` clave de join
cuando hay ~4.49 filas por `(t, direction)`— se corrige en el mismo cambio.

---

## 7. Portón de procedencia en los builders locales

Once builders de notebook verifican el SHA-256 de sus inputs. Los **15 builders
locales de análisis y reporte verifican cero**. Ese hueco produjo la Figura 1
obsoleta y el árbol de residuos rancio: consumieron datos viejos y publicaron
números equivocados en silencio.

**Acción:** el mismo portón de hash que ya existe, aplicado a los builders
locales. Si el input cambió, que **fallen** en vez de reportar.

---

## 8. Orden de ejecución

1. Escribir C1, C2, C3 como código, con sus tests (V1, V2, V3). **Local.**
2. Materializar el índice compartido y congelar su SHA-256. **Local.**
3. Medir cuánto encoge el conjunto y decidir si sigue siendo viable. **Local.**
4. Reajustar el XGBoost sobre el índice congelado. **Kaggle CPU.**
5. Reentrenar el LSTM sobre el mismo índice. **Kaggle GPU, 12 corridas.**
6. Recalcular la persistencia y el paquete de significancia. **Local.**
7. Reescribir las secciones afectadas de `documento-resultados.md`.

Los pasos 1–3 son los que evitan repetir el error. Si algo está mal en el
contrato, se descubre en el paso 3 y no después de 12 corridas de GPU.

---

## 9. Lo que este plan no cubre

Declarado explícitamente para que no se confunda alcance con cobertura:

- **Origen rodante (pendiente #12).** Sigue siendo una única ventana de test de
  ~22 días. Se declara como limitación; no se resuelve acá.
- **Métrica vectorial (pendiente #5).** El contrato de exportación la vuelve
  computable al conservar `pair_rank`, pero definirla es trabajo aparte.
- **Cobertura de semillas de ConvLSTM y Transformer (pendiente #4).** Queda
  declarada, no se cierra.
