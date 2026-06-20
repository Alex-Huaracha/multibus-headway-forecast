# PENDIENTE — C2: multi-seed + intervalos de confianza

> **Estado:** ABIERTO · **Última actualización:** 2026-06-20
> **Rama:** main · **Notebook:** NB15 (`notebooks/15_lstm_multiseed/`) · **Commit:** `a015a2d`

## Por qué hacemos esto (el gap C2)

El audit del repo (de cara al paper IJACSA) marcó dos gaps críticos en los
resultados. **C1** (baseline ajustado XGBoost) ya está cerrado. **C2** es este:

> Todos los resultados de Deep Learning salen de **una sola corrida, un único
> seed, sin intervalos de confianza ni error bars**. Es el reclamo más
> automático de un revisor: *"¿y si ese seed tuvo suerte con la inicialización?"*

**Un *seed* es el número que fija el azar del entrenamiento** (pesos iniciales,
barajado de datos, dropout). Mismo seed → entrenamiento idéntico; distinto seed →
modelo ligeramente distinto. Con un solo seed tenemos **una sola muestra** del
rendimiento. La solución es entrenar el modelo con **varios seeds** y reportar
**media ± intervalo de confianza**, demostrando que el resultado es estable
frente al azar, no un golpe de suerte.

## Qué hace NB15

Re-entrena la **misma configuración ganadora congelada del LSTM** (la de NB11,
sin tocarla) con **5 seeds** `[42, 123, 456, 789, 999]`, en cada horizonte
`h ∈ {1, 3, 5, 10}`, sobre ambos corredores (E2, E59).

- Alcance decidido: **solo LSTM** (los 3 modelos DL son casi idénticos en la
  curva, spread < 0.03 min, así que la varianza por seed del LSTM acota a toda
  la familia). ~11 GPU-horas en total, entra en una semana de cuota Kaggle.
- Salida por kernel: `lstm_multiseed_h{H}.csv` con schema
  `corridor, direction, baseline, metric, value, horizon, seed` (60 filas =
  3 direcciones × 2 métricas × 2 corredores × 5 seeds).
- **NB11 queda intacto** como el resultado canónico single-run. NB15 se apoya
  sobre él, igual que NB14 (sensibilidad de hiperparámetros). La cadena
  replicable 01→15 se mantiene en orden.

## Estado de las corridas en Kaggle

Kaggle limita a **2 sesiones GPU (T4x2) batch simultáneas**, así que las 4
corridas van en dos tandas.

| Kernel | Slug | Estado |
|--------|------|--------|
| h=1  | `alexhuaracha/15-lstm-multiseed-h1`  | ⏳ en ejecución / verificar |
| h=3  | `alexhuaracha/15-lstm-multiseed-h3`  | ⏳ en ejecución / verificar |
| h=5  | `alexhuaracha/15-lstm-multiseed-h5`  | ⏳ **PENDIENTE de correr** (rebotó por límite de 2 GPU) |
| h=10 | `alexhuaracha/15-lstm-multiseed-h10` | ⏳ **PENDIENTE de correr** (rebotó por límite de 2 GPU) |

## Pasos pendientes para cerrar C2

1. [ ] **Correr h5 y h10** (cuando se libere cupo GPU). Push desde cada subdir:
   ```bash
   cd notebooks/15_lstm_multiseed/h5  && uv run kaggle kernels push -p .
   cd notebooks/15_lstm_multiseed/h10 && uv run kaggle kernels push -p .
   ```
2. [ ] **Verificar estado** de los 4 kernels:
   ```bash
   for h in 1 3 5 10; do uv run kaggle kernels status alexhuaracha/15-lstm-multiseed-h$h; done
   ```
3. [ ] **Bajar los 4 CSVs** (`kaggle kernels output ...`) y versionarlos en
   `docs/resultados/csv-multihorizon/lstm_multiseed_h{H}.csv`.
4. [ ] **Calcular media ± IC** sobre los 5 seeds por (corredor, dirección,
   métrica, horizonte) — script nuevo en `src/` con su test (TDD).
5. [ ] **Agregar error bars** a la curva de degradación
   (`src/build_degradation_curve.py` + `curva-degradacion.png`).
6. [ ] **Actualizar `documento-resultados.md`** (Sección 3): reportar que el
   resultado del LSTM es estable frente al seed (IC angosto) → C2 cerrado.
7. [ ] Cerrar este archivo (mover a histórico o borrar) cuando los 6 pasos estén
   hechos.

## Referencias

- Builder: `src/build_notebook_15.py` · Tests: `tests/test_build_notebook_15.py` (58, TDD)
- Resultado esperado: error bars angostos confirman que la ventaja del DL de la
  curva de degradación no depende de un seed afortunado.
