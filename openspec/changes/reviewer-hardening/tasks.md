# Tareas — reviewer-hardening

Reglas del repo que aplican a todo lo de abajo:

- Todo corre con `uv run`. Los builders van como `uv run python -m src.<modulo>`
  (la ruta pelada falla al importar `src`).
- **Nunca** editar un `.ipynb` a mano: se edita el builder y se regenera.
- **Nunca** correr la suite completa: reescribe los notebooks generados. Usar tests
  puntuales. Si se corre entera, revisar `git status` y revertir `notebooks/`.
- Contrato de winsorización: umbral p99 sobre **train**, aplicado a **todos** los splits
  vía `winsorize_train_p99` sobre el frame completo con la columna `split`.

---

## Tarea 1 — Restaurar la corrección de la línea 41 (REGRESIÓN, hacer primero)

**Problema:** `docs/resultados/documento-resultados.md:41` afirma hoy que el XGBoost
*"recibe **exactamente la misma información** que la red neuronal"*. **Es falso** y está
commiteado. La corrección se escribió el 2026-07-18 y se perdió cuando un agente
regeneró archivos.

Verificado: `rg -c "atypical" src/baselines/fitted.py` → **0**. Los modelos DL sí reciben
la bandera de día atípico (input obligatorio con hash en `src/build_notebook_11.py:246`).

**Hacer:** reemplazar esa frase por una que declare las dos asimetrías reales:

1. La red recibe una bandera de día atípico que el XGBoost no ve.
2. Los hiperparámetros del XGBoost se fijaron a priori (`fitted.py:50-60`, un dict
   hardcodeado); los de la red salieron de búsqueda en grilla.

Y encuadrar la comparación con B5 como **cota inferior** de lo que lograría un aprendiz
clásico bien ajustado — con el matiz de que en E4 el XGBoost ya supera al LSTM en
horizontes cortos y medios sin ninguna de esas ventajas.

**Nota:** si se hace la Tarea 2, esta frase hay que volver a actualizarla (las asimetrías
dejarían de existir). Aun así corregirla YA: el documento no puede tener una afirmación
falsa mientras tanto.

- [ ] Frase corregida
- [ ] Commit

---

## Tarea 2 — Nivelar el XGBoost

**Por qué:** el revisor de rigor lo marcó como la asimetría más seria. El XGBoost es el
único rival que sostiene la defensa "nuestros baselines no son de paja", y compite en
desventaja.

### 2a. Bandera de día atípico como feature

- Espejar cómo la cargan los builders DL (`load_atypical_days` / `encode_context` en
  `src/build_notebook_11.py`).
- Contrato existente: un set que parsea **vacío** debe **lanzar excepción**, no producir
  una bandera todo-ceros (ver `tests/test_notebook_input_gate.py`).
- SHA-256 de `atypical_days.csv`:
  `2054245cc830e58b9397b75ea3b55d034581046b64e73b1630ca7d464e3ecb86`

### 2b. Búsqueda de hiperparámetros

- **Exactamente 24 configuraciones** (búsqueda aleatoria). No más: los kernels de CPU en
  Kaggle tienen límite de tiempo y son 3 corredores × 4 horizontes.
- Espacio sobre al menos: `eta`, `max_depth`, `min_child_weight`, `subsample`,
  `colsample_bytree`, `lambda`.
- **Selección estricta sobre validación.** El test no puede influir — contrato duro.
- Semilla de muestreo fija en el fuente (reproducible, no re-sorteable en silencio).
- Registrar la configuración ganadora por (corredor, horizonte) en la salida: si no
  queda escrita, no es auditable.
- Mantener el corte temprano sobre validación que ya existe (`fitted.py:143-148`).
- `nthread` está hoy en **1** (`fitted.py:59`). Subirlo para el kernel de CPU.

### 2c. Builders y fuentes de Kaggle

- Editar `src/build_notebook_10.py` (E2/E59) y `src/build_notebook_16_e4_data.py` (E4)
  para que monten y usen `atypical_days.csv`; regenerar ambos notebooks.
- Agregar `alexhuaracha/02-eda-corridors` a `kernel_sources` en los dos
  `kernel-metadata.json`. Estado actual verificado:

  | Notebook | `kernel_sources` hoy |
  |---|---|
  | `10-baselines-multi-horizonte` | `["alexhuaracha/04-preprocessing"]` |
  | `16-e4-data-baselines` | `[]` |

- ⚠️ **PASO MANUAL, NO SE PUEDE POR CLI.** Adjuntar una fuente **nueva** requiere un
  "Add Input" desde la web de Kaggle, **una vez por notebook**. Un `push` por CLI la
  lista un momento y la suelta. Verificar después con `kaggle kernels pull -m`.
  Documentado en `CLAUDE.md`.
- Si esos notebooks tienen portón de hashes de entrada, sumar `atypical_days.csv` como
  input requerido con su SHA.

### 2d. Correr y bajar

- `uv run kaggle kernels push -p notebooks/10_baselines_multihorizon/`
- `uv run kaggle kernels push -p notebooks/16_e4_data/`
- Bajar resultados y regenerar tablas derivadas. Runbook: `docs/correr-kaggle.md`.

**No se reentrena ningún modelo DL.** Son 2 kernels de CPU (`enable_gpu: false`).
Los baselines naive B0–B4 son fórmulas deterministas: dan idéntico.

**Riesgo asumido:** un XGBoost nivelado puede ganarle al LSTM en más celdas de las que ya
le gana en E4. Si pasa, **el documento tiene que decirlo**. Ese es el precio de sacarle
la objeción al revisor.

- [ ] 2a bandera
- [ ] 2b búsqueda
- [ ] 2c builders + metadata
- [ ] **2c-manual: Add Input en la web (×2)** ← requiere persona
- [ ] 2d push, bajar, regenerar tablas
- [ ] Actualizar §2 y §3 del documento con los números nuevos
- [ ] Commit

---

## Tarea 3 — Test estadístico contra el XGBoost (depende de la 2)

**Problema:** `src/evaluation/significance.py` solo maneja `y_pred_dl` vs
`y_pred_persist`. El competidor más fuerte **nunca** recibió DM/Wilcoxon.

- Extender para permitir DL vs XGBoost con la misma maquinaria (Diebold-Mariano con
  Newey-West/HAC + Wilcoxon).
- **Verificar primero si los notebooks exportan predicciones por muestra del XGBoost.**
  Si no, hay que agregar ese export en los builders → entra en la corrida de la Tarea 2.
  Comprobar antes de planificar.

- [ ] Confirmado si existen las predicciones por muestra
- [ ] Test implementado
- [ ] Commit

---

## Tarea 4 — Router con corte temporal por bloques

**Problema (lo encontró el revisor de rigor):** `policy_eval_split()` en
`src/build_router.py` hace una **permutación uniforme**. Las muestras son ventanas
solapadas con stride 1 que comparten 11 de 12 pasos de entrada, así que caen ventanas
casi gemelas a ambos lados. Son disjuntas por índice (bien testeado) pero **no
independientes**. Por eso el router coincide con el oráculo en 12/12 con Δ = 0.0: la
prueba no discrimina.

**Hacer:** partición por **bloques de tiempo** — primer ~60 % del período de test calibra,
último ~40 % evalúa.

- `materialize_corridor` (`src/build_exante_volatility.py:209`) devuelve hoy
  `(targets, persist, ex_ante_std)`. Hay que exponer también la **marca temporal del
  objetivo** por muestra, **preservando el orden exacto** (`dir=-1` y después `dir=+1`,
  que es el orden del CSV de residuos). El índice de ventanas se arma sobre una columna
  `t` (`src/data/windowing.py`, `make_window_index`), así que el dato existe.
- No romper el portón de alineación: `verify_alignment` debe seguir pasando.
- Terciles siguen congelados en train+val (`compute_frozen_thresholds`).
- Escribir a `docs/resultados/csv-multihorizon/router_temporal_multihorizon.csv`.
  **No pisar** `router_multihorizon.csv`: reportar **las dos bases** lado a lado.
- Incluir por celda la ganancia sobre la **regla trivial** (persistencia a h=1, LSTM a
  h≥3), que es el benchmark honesto — no solo contra always-DL.
- Tests en `tests/test_router_temporal.py`.

⚠️ **REGLA DE HONESTIDAD INTELECTUAL.** El corte temporal puede dar **peor**: la política
puede cambiar en algunas celdas, el router puede dejar de igualar al oráculo, la ganancia
puede achicarse o desaparecer. **Eso es un resultado aceptable y esperable.** Prohibido
tunear, rebarajar o ajustar fracciones buscando que quede lindo. Se reporta lo que dé la
primera corrida honesta. Si empeora, se dice y se cuantifica.

- [ ] Timestamps expuestos, orden preservado
- [ ] Corte temporal implementado
- [ ] CSV generado con ambas bases
- [ ] Tests
- [ ] Documento actualizado con el resultado real, sea cual sea
- [ ] Commit

---

## Tarea 5 — Amenazas a la validez no declaradas

El revisor de rigor listó ocho que el documento no declara. Todas son texto en
"Alcance y limitaciones" (§6). Ninguna requiere recomputar nada.

| # | Amenaza |
|---|---|
| 1 | **El objetivo está censurado.** `winsorize_train_p99` recorta `delta_t_min` en el p99 de train sobre *todos* los splits, incluido test. El MAE se mide contra un objetivo truncado, y el 1 % superior —justo los eventos de bunching que §5 reclama como terreno del DL— queda comprimido contra un techo. |
| 2 | **No hay evaluación de origen rodante.** Una sola ventana de test fija de 22 días. Es la demanda estándar de un revisor en un paper de forecasting. |
| 3 | **Confusor del período de test.** Febrero 2024 en Arequipa incluye Carnaval (12–13 feb). No se dice si están en `atypical_days.csv` ni se caracteriza la composición del test. |
| 4 | **El n efectivo está muy sobreestimado.** Las observaciones son ventanas solapadas que comparten 11/12 entradas, agrupadas por bus en el mismo snapshot. Además se confunde un IC de **varianza de entrenamiento** (5 semillas, mismos datos) con uno de **muestreo**. |
| 5 | **Agrupamiento de direcciones.** `significance_table(direction="aggregate")` mezcla `-1` y `+1`; donde difieren, el MAE agregado es una mezcla con pesos distintos entre la base pareada y la agregada. |
| 6 | **Sin estratificar por magnitud del headway.** 1 min de error sobre un headway de 3 min y sobre uno de 15 min no son lo mismo. Todo se reporta en absoluto. |
| 7 | **El "valor operativo" se afirma, nunca se modela.** No hay función de costo ni modelo de intervención que muestre que −1.5 min de MAE a h=10 cambie una decisión de despacho. |
| 8 | **Sin corrección por comparaciones múltiples.** 54+ comparaciones reportadas, ningún Holm/BH. Con p≈0 es cosmético, pero es gratis. |

Extra sugerido por el revisor adversarial (verificar antes de escribir):

- `lstm_minigrid_h10.csv`: en E2 el "empate exacto" con una vecina sería un **duplicado
  degenerado** — `src/models/lstm.py:62` hace `dropout=dropout if num_layers > 1 else 0.0`,
  así que con `num_layers=1` la vecina con `dropout=0.2` **es el mismo modelo**. La grilla
  de E2 tendría 3 configuraciones distintas, no 4. **Verificar antes de corregir.**

- [ ] Ocho amenazas agregadas a §6
- [ ] Duplicado degenerado del mini-grid verificado y, si aplica, declarado
- [ ] Commit
