# Diagnóstico y plan hacia un paper publicable (IJACSA)

> **Fecha**: 2026-06-14
> **Propósito**: dejar por escrito, sin vueltas, cuál es el problema real que apareció al
> consolidar los resultados, qué necesitamos para tener un paper sólido, y los pasos concretos
> a seguir. Este documento manda sobre el orden de trabajo de acá en adelante.

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

- [ ] **Paso 1 — Re-corrida a 5 min**: cambiar el horizonte (`T_out`) y re-generar los notebooks
      07 / 08 / 09 usando las configuraciones ganadoras ya conocidas. Correr en Kaggle los 3
      modelos + baselines en E2 y E59 (~2–3 h GPU, una sesión).
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
