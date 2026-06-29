# Problemas encontrados — integridad de datos y resultados

> **Estado:** ABIERTO · **Fecha:** 2026-06-29
> **Alcance:** problemas técnicos del documento de resultados y de los artefactos de datos. **No** incluye tareas editoriales conocidas (traducción a inglés, formato IMRaD de paper), que se abordarán después.
> **Documento auditado:** `docs/resultados/documento-resultados.md` (corredores E2, E59, E4; horizontes 1–10 min; multi-seed integrado).

Cada problema indica severidad, evidencia (archivo/línea), estado de verificación y acción.

- **Verificado** = comprobado directamente contra el repo en esta sesión.
- **Reportado** = señalado por la auditoría; pendiente de comprobación directa.

---

## P1 — Se cita un CSV que no existe y que ningún script genera
- **Severidad:** Alta (rompe la reproducibilidad de un argumento central).
- **Evidencia:** `docs/resultados/documento-resultados.md:223` cita `csv-multihorizon/exante_correlation_multihorizon.csv` (9 filas: 3 corredores × 3 horizontes) como respaldo de la **refutación de circularidad** (Pearson r ≈ 0.25, Spearman ρ ≈ 0.22, contingencia 1.1–1.3×, "entre 28 % y 54 %").
  - El archivo **no existe** en el repo. El único CSV ex-ante presente es `exante_volatility_multihorizon.csv` (otro contenido: 27 filas de terciles), citado correctamente en la línea 221.
  - Ningún script lo genera: `src/build_exante_curve.py` y `src/build_exante_volatility.py` ambos escriben `exante_volatility_multihorizon.csv`. La búsqueda de `correlation/pearson/spearman` en `src/` y `notebooks/` solo devuelve `build_notebook_03.py`, `significance.py` y el notebook 03 — ninguno relacionado con ex-ante.
- **Estado:** **Verificado** en esta sesión.
- **Impacto:** los números r ≈ 0.25 / ρ ≈ 0.22 que sostienen la refutación de circularidad (Sección 5) no se pueden reproducir desde el repo.
- **Acción:** crear `src/build_exante_correlation.py` que emita el CSV y versionarlo, o eliminar las afirmaciones numéricas. Los insumos (σ de la ventana de entrada y el cambio realizado |y_real − persistencia|) podrían ya estar en `build_exante_volatility.py`; verificar antes de reescribir.

---

## P2 — Sesgo en la elección del baseline titular
- **Severidad:** Alta (el margen titular sobrerrepresenta la ventaja del DL).
- **Evidencia:** los márgenes estrella (−26.7 %, −24.5 %) se calculan contra la persistencia (B1), el baseline que más se degrada al alargar el horizonte. El mejor baseline plano simple, **B4_HA (media horaria), es agnóstico al horizonte**, y a h=10:
  - E2 h10: B4_HA ≈ 5.259 vs LSTM 5.153 → ventaja real ~2 % (no 26.7 %).
  - E59 h10: B4_HA ≈ 4.805 vs LSTM 4.225 → ventaja ~12 % (no 24.5 %).
  La Figura 1 etiqueta a **B3 como "el mejor baseline formulaico"**, pero a h=10 B3 (E2: 5.946) no es el mejor — B4_HA (5.259) y B0 (5.287) lo superan y están omitidos de la figura central.
- **Estado:** Reportado (verificar B4_HA/B0 a cada horizonte contra `csv-multihorizon/baselines_results_multih.csv`).
- **Acción:** incluir B4_HA y B0 en la Figura 1, dejar de llamar a B3 "el mejor formulaico", y reportar el margen del DL contra el **mejor** baseline simple a cada horizonte. Declarar como limitación que la ventaja sobre B4_HA es de ~2 % en E2 a h=10.

---

## P3 — Columna mal etiquetada en el CSV de significancia
- **Severidad:** Media (inconsistencia interna en datos publicados).
- **Evidencia:** en `significance_multihorizon.csv`, las filas RMSE reusan el `delta_mae` calculado para MAE (p. ej. Transformer/E4/h3/RMSE muestra un Δ de MAE junto a un `dm_stat` de RMSE — internamente inconsistente). El test de significancia corrió sobre RMSE, pero el Δ reportado en esa columna es el de MAE.
- **Estado:** Reportado.
- **Acción:** recalcular la columna de delta para las filas RMSE, o renombrarla y documentar que el delta mostrado es siempre el de MAE.

---

## P4 — Discrepancia numérica multi-seed (E2 h10)
- **Severidad:** Baja (error de redondeo en el documento).
- **Evidencia:** el documento (§4) reporta E2 MAE h10 media multi-seed = **5.146**; el CSV `multiseed_ci_multihorizon.csv` da **5.14462 → redondea a 5.145**. (E59 = 4.225 sí cuadra.)
- **Estado:** Verificado.
- **Acción:** corregir 5.146 → 5.145.

---

## P5 — CSV consolidado citado pero no versionado
- **Severidad:** Baja (regenerable).
- **Evidencia:** `consolidated_multihorizon.csv` (citado en `INTERPRETACION.md:148`) no está versionado, aunque lo genera `build_degradation_curve.py`.
- **Estado:** Reportado.
- **Acción:** versionar el archivo o dejar explícito en el texto que es regenerable y con qué script.

---

## P6 — `set_seed` no garantiza determinismo total
- **Severidad:** Baja (reproducibilidad incremental).
- **Evidencia:** `src/train.py:set_seed` omite `random.seed()` de Python y `torch.backends.cudnn.deterministic`; el propio docstring lo admite.
- **Estado:** Reportado.
- **Acción:** añadir `random.seed` y `cudnn.deterministic`, y mencionarlo en la sección de reproducibilidad.

---

## Resumen

| ID | Problema | Severidad | Estado |
|----|----------|-----------|--------|
| P1 | CSV `exante_correlation` citado pero inexistente | Alta | Verificado |
| P2 | Sesgo en baseline titular (persistencia vs. B4_HA/B0) | Alta | Reportado |
| P3 | Columna `delta_mae` mal etiquetada en filas RMSE | Media | Reportado |
| P4 | Discrepancia 5.146 vs 5.145 (E2 h10 multi-seed) | Baja | Verificado |
| P5 | `consolidated_multihorizon.csv` no versionado | Baja | Reportado |
| P6 | `set_seed` sin determinismo total | Baja | Reportado |
