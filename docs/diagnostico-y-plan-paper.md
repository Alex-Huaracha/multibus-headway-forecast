# Diagnóstico y plan hacia un paper publicable (IJACSA)

> **Fecha**: 2026-06-14 · **Última actualización**: 2026-06-15
> **Propósito**: dejar por escrito, sin vueltas, cuál es el problema real que apareció al
> consolidar los resultados, qué necesitamos para tener un paper sólido, y los pasos concretos
> a seguir. Este documento manda sobre el orden de trabajo de acá en adelante.
>
> ⚠️ **El estado de avance vigente y cómo continuar están en la [§6](#6-estado-de-avance-y-decisiones-2026-06-15).**
> Las secciones 1–5 son el diagnóstico original; la §6 las actualiza donde difieren.

---

## 1. Dónde estamos (lo que YA está hecho y NO se repite)

- **Pipeline completo construido y testeado**: preprocesamiento, cálculo de headways,
  dataset supervisado, baselines estadísticos, y 3 modelos de deep learning (LSTM,
  SpatialConvLSTM, SpatialTransformer). Todo con suite de tests verde.
- **Grid search completo corrido en Kaggle** para los 3 modelos en los 2 corredores
  obligatorios (E2 y E59). Las configuraciones ganadoras están guardadas en
  [`resultados/configuraciones-ganadoras.md`](./resultados/configuraciones-ganadoras.md).
- **Resultados al horizonte de 1 minuto** consolidados (ver
  [`resultados/fase-6b-spatial-transformer.md`](./resultados/fase-6b-spatial-transformer.md)).

> **Importante**: encontrar las configuraciones ganadoras costó ~1 mes (no de programación,
> sino de **esperar la cuota semanal de GPU de Kaggle** al correr 24–32 versiones × 3 modelos).
> Ese trabajo **ya está hecho y no se vuelve a hacer.** Cualquier re-corrida futura reusa las
> perillas ganadoras = 1 entrenamiento por modelo, no 24–32.

---

## 2. El problema que apareció

Al consolidar los resultados surgieron **dos hallazgos** que obligan a reencuadrar el paper.

### 2.1 La señal espacial no aporta (resultado nulo, pero VÁLIDO)

Ni la convolución espacial (ConvLSTM) ni la atención espacial (Transformer) superan al LSTM
plano, **en ninguno de los dos corredores**. El grid incluso eligió la mínima capacidad
espacial (`conv_channels=1`, `nhead=1`). Conclusión: para predecir el próximo headway, la
dinámica temporal de cada posición ya contiene la señal útil; mirar a los buses vecinos no
agrega.

→ **Esto no es un fracaso**: un resultado nulo bien demostrado (2 corredores, 2 mecanismos
distintos) es publicable. Descarta una hipótesis razonable con evidencia limpia.

### 2.2 A 1 minuto, la persistencia trivial empata o gana (el problema REAL)

El baseline B1 (persistencia: "el próximo headway = el último observado") es durísimo de batir
al horizonte de **1 minuto**:

| Corredor | Métrica | Persistencia (B1) | Mejor DL | ¿Gana el DL? |
|---|---|---|---|---|
| E2 | MAE | 4.757 | 4.471 (LSTM) | ✅ sí |
| E2 | RMSE | 7.636 | 6.110 (LSTM) | ✅ sí |
| E59 | MAE | **3.100** | 3.337 | ❌ **no** |
| E59 | RMSE | 5.712 | 4.667 (LSTM) | ✅ sí |

**Causa raíz**: estamos prediciendo a **1 minuto** (`T_out=1`, grilla de 60 s). En 60 segundos
el headway casi no cambia — sobre todo en E59 (alta frecuencia) — así que "el próximo ≈ el
actual" es casi imbatible **por construcción**. No es una falla del modelo ni del código.

**Por qué importa para el paper**:
- El criterio de éxito del proyecto ([`objetivo.md`](./objetivo.md) §Criterios, #1) pide que el
  DL supere a los baselines **en MAE y RMSE** con consistencia en ambos corredores. Al horizonte
  de 1 minuto, **E59 no lo cumple** (la persistencia gana en MAE).
- Un revisor de IJACSA detectaría esto de inmediato: *"si predecís a 1 minuto y la persistencia
  te empata, ¿cuál es el aporte?"*.

---

## 3. Qué necesitamos para un paper SÓLIDO

El arreglo no es re-hacer el trabajo: es **medir al horizonte correcto** y cerrar la
comparación ahí.

1. **Re-correr a un horizonte de 5 minutos** (alineado con el caso de uso real: anticipar
   bunching/gaps con margen para intervenir). A 5 minutos el headway sí cambia, la persistencia
   se degrada, y los modelos profundos tienen espacio para ganar en MAE **y** RMSE.
2. **Correr los 3 modelos** (LSTM, ConvLSTM, Transformer), no solo el LSTM, para que la
   comparación espacial-vs-temporal quede cerrada **al horizonte de publicación** y no deje
   agujeros para el revisor.
3. **(Opcional, refuerza mucho)** Medir a varios horizontes (1, 3, 5, 10 min) y mostrar la
   **curva de degradación**: cómo la persistencia se cae y los modelos profundos aguantan al
   crecer el horizonte. Convierte la debilidad (persistencia gana a 1 min) en el mejor argumento
   del paper.

### Costo real de esto (NO es otro mes)

- El grid search **ya está hecho**: se reusan las configuraciones ganadoras → **1 entrenamiento
  por modelo**, no 24–32.
- 3 modelos × 2 corredores × 1 entrenamiento = **6 corridas**. Tiempo estimado de tus propios
  logs: ~12 min/corrida en E2, ~40 min/corrida en E59 → **~2–3 horas de GPU en total**.
- Eso entra en **una sola sesión de Kaggle**. No se toca casi nada de la cuota semanal. **No hay
  que esperar resets semana tras semana** — eso pasó porque antes se corrían 24–32 versiones.
- Los baselines a 5 min **no usan GPU** (corren en CPU).

---

## 4. Pasos a seguir (en orden)

- [ ] **Paso 1 — Re-corrida multi-horizonte**: ⚠️ el enfoque se refinó — ver [§6](#6-estado-de-avance-y-decisiones-2026-06-15).
      NO se re-generan los notebooks 07/08/09 (son artefactos congelados del experimento de 1 min);
      se crean builders NUEVOS que reusan la librería ya parametrizada con `horizon`. Correr en
      Kaggle los 3 modelos + baselines en E2 y E59 a 1/3/5/10 min.
- [ ] **Paso 2 — Consolidar resultados a 5 min**: tabla comparativa DL vs baselines (incluida la
      persistencia) en MAE y RMSE, ambos corredores. Verificar que se cumple el criterio de
      éxito (DL > baselines en ambas métricas).
- [ ] **Paso 3 — (Opcional) Curva multi-horizonte** (1/3/5/10 min) si el tiempo lo permite, para
      reforzar el argumento.
- [ ] **Paso 4 — Significancia estadística**: test de Diebold-Mariano o Wilcoxon pareado
      (p < 0.05) sobre las diferencias DL vs baselines, como exige [`objetivo.md`](./objetivo.md).
- [ ] **Paso 5 — Fase 8: detección de anomalías**: sobre las predicciones a 5 min, demostrar que
      el sistema marca bunching y gaps reales en los datos (análisis, no entrenamiento).
- [ ] **Paso 6 — Redacción del paper** con el encuadre honesto: (a) comparación rigurosa
      estadístico vs LSTM vs espacial; (b) resultado nulo espacial; (c) valor del DL al horizonte
      útil (5 min) y en robustez (RMSE); (d) demo de anticipación de anomalías.

---

## 5. El encuadre honesto del paper (para no caer ante un revisor)

- **NO** vender "el DL gana siempre en error promedio" — a horizonte corto no es cierto.
- **SÍ** vender: (1) comparación rigurosa sobre corredores reales latinoamericanos con solo GPS
  básico; (2) la relación espacial entre buses **no** mejora la predicción (hallazgo nulo
  robusto); (3) los modelos profundos aportan al **horizonte útil para operación (5 min)** y en
  **robustez frente a errores grandes (RMSE)**; (4) el sistema **anticipa anomalías colectivas**
  (bunching/gaps) con margen para intervenir.

> La honestidad metodológica acá no es un costo: es lo que hace al paper defendible. El hallazgo
> de que la persistencia es fuerte a horizonte corto, reportado y explicado, es una fortaleza,
> no una debilidad.

---

## 6. Estado de avance y decisiones (2026-06-15)

> Esta sección refleja el trabajo real hecho en la rama `feat/fase-6-5-multi-horizonte` y
> **manda sobre las secciones 1–5 donde difieran**. La curva multi-horizonte (antes "opcional"
> en §3.3) pasó a ser **el argumento central del paper**, no un extra.

### 6.1 Lo que YA está hecho (rama `feat/fase-6-5-multi-horizonte`)

- **Fase 6 cerrada y validada**: SpatialConvLSTM y SpatialTransformer dan resultado nulo en
  ambos corredores. Validado contra Kaggle al decimal (commit del plan `5c3e261`).
- **Librería parametrizada con `horizon`** (Olas 1-2, commits `d48e913` + `42f1aa5`):
  - `predict_b1` / `evaluate_corridor` aceptan `horizon` (persistencia = `shift(horizon)`).
  - `make_window_index` / `HeadwayDataset` aceptan `horizon`: el target es la única fila en
    `T_in + horizon - 1`, shape `(1, max_N)` (mantiene válido el `squeeze(1)` de `train.py`).
  - Backward-compatible: `horizon` por defecto reproduce **exactamente** el comportamiento de
    1 min. **337 tests verdes.**
- **Experimentos de 1 min INTACTOS**: builders `07/08/09` y sus notebooks sin tocar (idénticos
  a `main`). Modelos y resultados ya entrenados, sin tocar.
- **Librería de baselines horizon-aware completa** (Ola 3, 2026-06-15): `predict_b2` y
  `predict_b3` ahora aceptan `horizon` y `evaluate_corridor` lo propaga a B1/B2/B3. Resolvió el
  hallazgo de §6.3 (B2/B3 no eran horizon-correctos). **346 tests verdes** (337 + 9 nuevos).
- **Builders multi-horizonte completos** (2026-06-15, commits `4b8043a`/`9b5e501`/`0f563b1`/`8f24e44`):
  - **NB10** (`build_notebook_10.py`): baselines a h∈{1,3,5,10} → `baselines_results_multih.csv`
    (336 filas, columna `horizon`). Embebe el código Ola 3.
  - **NB11/12/13** (LSTM/ConvLSTM/Transformer): **un notebook por horizonte** (4 c/u),
    `fast_materialize` horizon-aware (target en `T_in+HORIZON-1`), reusan la config ganadora
    (1 entrenamiento vía `grid_search` con lista de 1), leen `baselines_results_multih.csv`
    filtrado por `horizon`, salida `*_results_h{N}.csv` con columna `horizon`.
  - **392 tests verdes** en total. Notebooks 01–09 (artefactos congelados de 1 min) intactos.

### 6.2 Decisiones de arquitectura (las que rigen de acá en adelante)

1. **Librería vs experimento — se tratan distinto** (buena práctica de ML reproducible):
   - **Librería** (`src/data/`, `src/baselines/`, `src/models/`): código vivo compartido →
     se **PARAMETRIZA** (perilla `horizon`), NUNCA se duplica (dos copias divergen).
   - **Notebooks de experimento** (builders): un experimento corrido es un **artefacto
     inmutable** (el registro reproducible del resultado de 1 min). El multi-horizonte es un
     experimento NUEVO → se crean **builders NUEVOS**, NO se mutan los de 1 min.

2. **Esquema DIRECTO por horizonte**: un modelo entrenado por cada `T_out` objetivo (no
   recursivo), para comparación limpia DL vs persistencia a cada horizonte.

3. **Reusar configs ganadoras** (`resultados/configuraciones-ganadoras.md`, validadas al
   decimal): 1 entrenamiento por modelo/corredor/horizonte. Riesgo aceptado: las configs son
   óptimas a 1 min; si un revisor lo cuestiona, se blinda con un mini-grid de 2-3 configs
   vecinas, no el grid completo.

4. **Sin PR**: rama + merge local a `main` (el usuario es el único en el repo).

### 6.3 Bug pendiente que el nuevo experimento DEBE resolver

Los notebooks DL leen los baselines de un CSV que genera **NB06**, y NB06 **no es
horizon-aware** → a 3/5/10 min el DL se compararía contra baselines a 1 min (comparación
tramposa que invalidaría la curva). El experimento multi-horizonte nuevo **debe generar sus
baselines al horizonte correcto** (los 5: B0–B4, no solo B1), sin tocar el NB06 viejo.

> **Actualización (Ola 3, 2026-06-15)**: al verificar el "los 5 baselines" se descubrió que
> el problema era mayor de lo escrito. Solo B1 estaba parametrizado (Ola 1). B0 (media por slot)
> y B4_HA (media por slot,hora) son **legítimamente horizon-agnósticos** (predictor constante /
> lookup por hora conocida). Pero **B2 (media móvil) y B3 (SES)** eran predictores de 1 paso: a
> 3/5/10 min usaban observaciones hasta `t-1` para predecir `t+h` = mirar al futuro, lo que los
> haría artificialmente fuertes y un revisor lo detectaría. La **Ola 3** los parametrizó con la
> regla unificada `shift(horizon-1)` (consistente con B1) y `evaluate_corridor` propaga `horizon`
> a B1/B2/B3. Hallazgo cerrado a nivel librería; el builder nuevo solo debe **pasar el `horizon`
> correcto**.

### 6.4 Cómo continuar (próxima sesión)

1. ~~Crear los **builders NUEVOS** de multi-horizonte~~ ✅ **HECHO** (NB10-13, §6.1).
2. ~~Tests TDD para cada builder nuevo~~ ✅ **HECHO**.
3. ~~Generar los notebooks, correr en Kaggle, bajar resultados~~ ✅ **HECHO** (2026-06-16):
   los 13 CSVs viven versionados en `docs/resultados/csv-multihorizon/` (excepción en
   `.gitignore`). El h=1 de los DL se bajó de los kernels viejos `07/08/09` y se normalizó al
   schema multi-h (`horizon=1`, corredores combinados).
4. ~~Consolidar 1/3/5/10 min → **curva de degradación**~~ ✅ **HECHO** → ver §6.6.
5. **Pendiente — significancia estadística**: bloqueada con los datos actuales. DM/Wilcoxon
   comparan errores **por-muestra**, pero los CSVs solo guardan MAE/RMSE agregados (y faltan
   `scipy`/`statsmodels`). Requiere **re-correr los kernels exportando residuos/predicciones por
   muestra**. Es un paso aparte, no un cómputo sobre lo que ya hay.
6. **Pendiente — redacción** del paper apoyada en la curva (§6.6) + Fase 8 (anomalías).

### 6.5 Lecciones operativas (NO repetir errores de esta sesión)

- **NO ejecutar los `src/build_notebook_*.py` durante el desarrollo**: regeneran los `.ipynb` y
  ensucian el árbol. `pytest tests/` también los regenera (los tests de builders escriben en la
  ubicación real). Tras testear: `git restore notebooks/`. (Pendiente de fondo: que esos tests
  escriban en `tmp_path`.)
- El trabajo de los builders nuevos se hace **inline, paso a paso**, mostrando cada archivo
  antes de escribirlo — sin sub-agentes que corran generadores.
- **Parche operativo NB12 h10 → slug `h10b`** (2026-06-16): el slug
  `alexhuaracha/12-spatialconvlstm-multihorizon-h10` quedó **corrupto en el backend de Kaggle**
  (`GetKernel` → 500, `status` → 404, no aparece en el listado de kernels). El push falla con
  "Notebook not found" aunque el `.ipynb` es válido (gemelo byte-a-byte del 13-h10, que pushea
  bien). Probado: el mismo archivo entra limpio en un slug nuevo. **Fix:** el kernel-metadata.json
  de `notebooks/12_spatial_conv_lstm_multihorizon/h10/` apunta a `12-spatialconvlstm-multihorizon-h10b`.
  El builder `build_notebook_12.py` **se deja limpio** (genera `h10` por convención); el `h10b` es
  un parche del artefacto generado, no de la lógica. Si se regenera NB12, re-aplicar el sufijo a
  mano en ese metadata. Causa raíz del zombie: probable push interrumpido previo (inferencia).

### 6.6 Resultado — curva de degradación (2026-06-16)

Consolidados los 13 CSVs de `csv-multihorizon/`, el módulo `src/evaluation/degradation.py`
(`load_results` + `degradation_table`, 7 tests TDD) y el script
`src/build_degradation_curve.py` producen la **figura central del paper**:
`docs/resultados/curva-degradacion.png` (2×2: MAE/RMSE × E2/E59).

> Reproducir: `uv run python -m src.build_degradation_curve`
> (NO `python src/build_degradation_curve.py` — el import `src` falla sin el rootdir en path).

**Hallazgo central — el deep learning gana al alejar el horizonte.** A **h=1** los tres modelos
profundos **empatan** con la persistencia (B1); ese era el "problema" de §2.2. Pero **B1 se
degrada mucho más rápido** que los DL al crecer el horizonte. Ejemplo E2, MAE agregado:

| Modelo | h=1 | h=3 | h=5 | h=10 |
|--------|-----|-----|-----|------|
| Persistencia (B1) | 4.76 | 6.07 | 6.49 | **7.03** |
| LSTM | 4.47 | 4.94 | 5.05 | **5.15** |
| ConvLSTM | 4.47 | 4.94 | 5.07 | **5.14** |
| Transformer | 4.49 | 4.94 | 5.08 | **5.17** |

A h=1 la diferencia es de ~0.3 min; a h=10 es de **~1.9 min** (B1 7.03 vs DL ~5.15). La brecha
se **abre a favor de los modelos profundos** a 3/5/10 min, en ambos corredores y en ambas
métricas. (Los tres DL son casi indistinguibles entre sí — coherente con el resultado nulo
espacial de Fase 6.) **Esto reencuadra el paper**: el aporte del DL no es ganar a 1 min (no lo hace),
sino **resistir la degradación** cuando la predicción se aleja — justo donde la persistencia
trivial colapsa. La curva de degradación es el argumento, no un horizonte aislado.

**Límite honesto (para el revisor)**: la comparación es sobre el **error agregado** (MAE/RMSE).
Falta el test de **significancia** (DM/Wilcoxon) que confirme que la brecha a h≥3 es
estadísticamente real — ver §6.4 paso 5 (requiere errores por-muestra, no disponibles aún).
