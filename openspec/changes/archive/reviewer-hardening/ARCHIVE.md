# Reporte de Archivo — reviewer-hardening

**Archivado:** 2026-07-20 · **Estado:** completo (las 5 tareas cerradas)

## Qué hizo este change

Endureció `docs/resultados/documento-resultados.md` frente a las objeciones de un panel de
tres revisores IJACSA simulados (60 % / 72 % / 72 %). Corrigió afirmaciones que los datos no
sostenían, niveló el baseline aprendido para que compita en igualdad, cubrió la significancia
del contraste más fuerte de forma honesta, sometió el enrutador a una prueba más dura, y
declaró las amenazas a la validez que el documento callaba.

## Especificaciones promovidas a `openspec/specs/`

Ninguna. Fue un change de endurecimiento (correcciones + análisis de refuerzo sobre resultados
ya existentes), no de nuevas capacidades; no llevó delta specs.

## Resultado por tarea

1. **Línea 41 (regresión).** Se eliminó la afirmación falsa de que el XGBoost "recibe la misma
   información que la red" (`fitted.py` no usa la bandera de día atípico).
2. **Nivelar el XGBoost.** Se le dio la bandera de día atípico (mismo `encode_context`) y una
   búsqueda de 24 hiperparámetros seleccionada solo sobre validación (sin fuga, verificado).
   Corrido en Kaggle CPU (kernels 10 y 16). **Hallazgo:** nivelar apenas movió su MAE (≤ 0.03
   min) — la comparación previa ya era representativa. El LSTM le gana en las **8/8** celdas de
   E2+E59; E4 sigue siendo el matiz de escala.
3. **Significancia DL-vs-XGBoost.** El test pareado por muestra resultó **inviable** (residuos
   del DL sobrecontan cada objetivo ~4.5× vía ventanas solapadas, sin clave por muestra). Se
   adoptó un **test de signos a nivel de celda**, inmune al sobreconteo: E2+E59 el LSTM gana
   8/8 (**p = 0.004**); E4 1/4 (no significativo). La limitación quedó declarada.
4. **Router con corte temporal.** El corte uniforme dejaba ventanas casi gemelas a ambos lados.
   El corte por bloques de tiempo (primer 60 % calibra, último 40 % evalúa) **confirma** el
   router: política **idéntica en 12/12**, iguala al oráculo en 12/12, ganancia sobre la regla
   trivial −0.016 min (vs −0.018 uniforme). El 12/12 no era artefacto del solapamiento.
5. **Amenazas a la validez.** Se declararon 8 en §6 (objetivo censurado, sin origen rodante,
   confusor de Carnaval, n efectivo sobreestimado, agregación de direcciones, sin estratificar
   por magnitud, valor operativo no modelado, sin corrección por comparaciones múltiples). Se
   **descubrió y corrigió** un duplicado degenerado en el mini-grid de E2: `lstm.py:62` fuerza
   dropout=0 con `num_layers=1`, así que la "vecina" era bit a bit el mismo modelo (3 configs
   distintas, no 4).

## Intelectual-honestidad

Las Tareas 2 y 4 podían salir en contra y se corrieron sin tunear. Ambas terminaron
reforzando el resultado, pero se reportaron por su valor honesto: los aportes reales del
XGBoost nivelado (nulo) y de la señal de volatilidad (~1 seg) son chicos, y así se dice.

## Commits que lo completan

- `b28cc80` corregir afirmación falsa sobre paridad de información del XGBoost
- `629c8d5` nivelar B5_XGB (bandera de día atípico + búsqueda de hiperparámetros)
- `4098d32` integrar XGBoost nivelado y reencuadrar la comparación
- `7467f89` test de signos a nivel de celda para DL-vs-XGBoost
- `14402d7` declarar 8 amenazas a la validez y corregir el duplicado del mini-grid
- `d2683e5` corte temporal por bloques confirma el router
