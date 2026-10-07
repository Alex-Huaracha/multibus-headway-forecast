# Pendientes

Lo que falta tras corregir el pipeline de las rutas A y C (2026-10-07).

## Contexto

La segunda pasada de las rutas A y C (E2, E59) tenía dos defectos: no
eliminaba los registros a más de 300 m del eje y dejaba sin posición a los buses
detenidos. Se corrigió el código, se regeneraron los datos en Kaggle y se
reentrenó el LSTM (commit `10ce275`). El LSTM nuevo mantiene todas las
conclusiones del paper, pero cambian las cifras de A y C. La ruta B (E4) no
cambia.

## En curso

- [ ] **XGBoost en Kaggle** (`alexhuaracha/22-xgb-contiguous`, versión 3).
  Corre desde las 15:17. Al terminar:
  - bajar `xgb_contig_results.csv`, `xgb_contig_residuals.csv` y
    `xgb_contig_search_config.csv` a
    `docs/resultados/residuos-multihorizon/22-xgb-contiguous/`;
  - copiar los resultados y la configuración a `docs/resultados/csv-multihorizon/`;
  - revisar el log: sin fallas de los controles de datos, y winsorización de
    E2 en 28.3020 y de E59 en 28.5911.

## Después del XGBoost

- [ ] **Copiar los resultados nuevos del LSTM** de
  `docs/resultados/residuos-multihorizon/21-lstm-contiguous/` a
  `docs/resultados/csv-multihorizon/` (`lstm_contig_results_h*.csv` y los de
  r1 y r2).
- [ ] **Regenerar las tablas de análisis** con los residuos nuevos:
  significancia, métricas de vector, detección calibrada, identidad de la
  dispersión, intervalos del ranking, origen rodante, figuras.
- [ ] **Arreglar el test que falla**:
  `tests/test_threshold_denominators.py::test_the_table_carries_the_rate_each_quota_used`.
  Falla porque su CSV se calculó con los residuos viejos. Debe pasar al
  regenerar las tablas.
- [ ] **Vigilar E59 a 3 minutos.** El LSTM sigue ganando en error a la
  persistencia, pero por menos (de 0.19 a 0.07 min). Ver si la prueba de
  significancia la sostiene.
- [ ] **Recalcular `window_sensitivity.csv`** con los datos nuevos
  (`uv run python -m src.build_window_sensitivity`). Sostiene la frase de la
  §III-A sobre $L = 12$. Hoy está calculado con los datos viejos y sin comitear.

## En el paper

- [ ] **Actualizar a mano toda cifra de las rutas A y C**: tablas de la §V,
  resumen (por ejemplo «17 de 19»), rangos de $r$, introducción y conclusión.
  Ninguna se genera sola.
- [ ] **Revisar el apéndice A.1 ya reescrito** (commit `efcb303`). Su cobertura
  ya usa los datos nuevos: 65.0 % en A, 64.8 % en B, 66.5 % en C.

## Fuera de alcance por ahora

- Los demás notebooks de la ruta B no se tocan. Si se vuelve a correr el LSTM
  de B en Kaggle, necesita la fuente `alexhuaracha/16-e4-data-baselines`, como
  ya la tiene el XGBoost.
