# Change: reviewer-hardening

**Estado:** propuesto · **Creado:** 2026-07-18 · **Origen:** panel de 3 revisores IJACSA simulados

## Por qué

Un panel de tres revisores independientes (rigor metodológico / contribución / auditoría
adversarial de afirmaciones) evaluó `docs/resultados/documento-resultados.md` y dio
**60 % / 72 % / 72 %** de probabilidad de aceptación en IJACSA. Los tres coincidieron en
que la calidad de ingeniería está por encima de la media del venue, y los tres
encontraron defectos reales. Los verificados contra los CSV se corrigieron en el commit
`0eceb38`; lo que queda es este change.

Los dos revisores que estimaron probabilidad post-corrección la ubicaron en **80–82 %**.

## Qué ya se hizo (commit `0eceb38`, NO rehacer)

| Corrección | Detalle |
|---|---|
| Afirmación falsa en §5 | "aun ahí el DL gana" en la fracción calma del tercil alto ex-ante era imposible por construcción (el régimen estable se *define* como error de persistencia < 1 min). Reescrito. |
| El cruce es específico del MAE | Se afirmaba que "el RMSE confirma el mismo patrón"; bajo RMSE el DL gana a h=1 en 9/9 celdas. Declarado explícitamente. |
| Tabla de cruce de §3 | Reconstruida sobre la base **pareada**. La persistencia gana a h=1 en los tres corredores (antes E2·h=1 figuraba como victoria del DL por usar agregados sobre muestras distintas). |
| `paired_audit.HORIZONS` | De `(3,5,10)` a `(1,3,5,10)`. La auditoría canónica excluía justo el horizonte donde el veredicto se invierte. 72 celdas, 68 acuerdos de signo. |
| Router reencuadrado | Ablación contra la regla trivial (persistencia a h=1, LSTM a h≥3): la volatilidad aporta **−0.018 min**, no −0.100, y solo en 3 de 12 celdas. Se eliminó la comparación contra always-persistence por hombre de paja. |

## Alcance de este change

Cinco tareas independientes entre sí. Detalle ejecutable en `tasks.md`.

| # | Tarea | Bloquea | Costo | Riesgo para el paper |
|---|---|---|---|---|
| 1 | Restaurar la corrección de la línea 41 (regresión) | — | 2 min | Ninguno: hoy hay una afirmación falsa en el documento |
| 2 | Nivelar el XGBoost (bandera de día atípico + búsqueda de 24 configs) | Paso manual en Kaggle | 2 kernels CPU | **Alto**: un XGBoost bien ajustado puede ganarle al LSTM en más celdas |
| 3 | Test DM/Wilcoxon contra el XGBoost | Tarea 2 | Local | Medio |
| 4 | Router con corte temporal por bloques | — | Local, lento | **Alto**: la política puede dejar de ser estable |
| 5 | Amenazas a la validez no declaradas | — | 1 h | Ninguno: solo suma honestidad |

## Fuera de alcance (decidido explícitamente)

**Calibrar la política del router sobre train+val.** Exigiría reentrenar los 8 kernels
GPU: los kernels no guardan pesos (`save_checkpoint` nunca se invoca), y solo exportaron
predicciones por muestra del split de test. Costo ~8 GB de salida y riesgo de mover
números ya verificados. Queda declarado como limitación en el documento.

Si algún día se hace: el archivo nuevo **no debe** coincidir con el patrón
`*_residuals_*`, porque varios consumidores lo globean y concatenan sin filtrar por
split — se corromperían en silencio la significancia y la auditoría pareada.

## Estado del trabajo interrumpido

Dos agentes fueron cancelados a mitad de camino el 2026-07-18. Su trabajo parcial está
en `git stash` (`stash@{0}`), NO commiteado:

```
stash@{0}: WIP xgboost-nivelado + router-temporal (agentes interrumpidos 2026-07-18)
```

Toca 10 archivos: `src/baselines/fitted.py`, `src/baselines/harness.py`,
`src/build_notebook_10.py`, `src/build_notebook_16_e4_data.py`,
`src/build_exante_volatility.py`, `src/evaluation/significance.py`, los dos `.ipynb`
regenerados y sus dos `kernel-metadata.json`.

**Recomendación: revisarlo con `git stash show -p stash@{0}` antes de aplicarlo.** Quedó
a medias (el agente de XGBoost estaba por regenerar notebooks; el del router estaba
editando `materialize_corridor`). Aplicarlo a ciegas es más riesgoso que rehacerlo:
un `fitted.py` a medio editar puede producir resultados incorrectos sin fallar.
