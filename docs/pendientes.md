# Qué falta y en qué orden

Actualizado: 2026-07-27.

> **Lo único que bloquea la publicación es el manuscrito.** Todo lo demás de esta
> lista cierra objeciones de revisor o es deuda técnica. Ninguno cambia el
> hallazgo central.

---

## 1. Dónde estamos

| | |
|---|---|
| Rama `main` | Pipeline contiguo mergeado. **22 commits sin pushear.** |
| Rama `feat/rolling-origin` | 2 commits: cortes parametrizados + notebooks emitidos. **Sin mergear.** |
| Suite de tests | 1606 pasando |
| Hallazgo central | Medido, con guardián sobre cada cifra del documento |

El aporte del paper es **la disociación**: los modelos óptimos en MAE
sub-reportan sistemáticamente la irregularidad del servicio, y una evaluación
escalar es estructuralmente incapaz de detectarlo. A h=10 el LSTM gana el MAE por
1.47 min y pierde el F1 de detección de *bunching* por un factor de 253.

---

## 2. Para hacer en casa: correr r1 y r2 en Kaggle

**16 kernels, ≈1.8 h de GPU, ~1 h de reloj.** Runbook completo en
[`correr-kaggle.md`](./correr-kaggle.md) §2.

| Paso | Comando / acción | Verificar |
|---|---|---|
| 1 | `git checkout feat/rolling-origin` | estás en la rama correcta |
| 2 | `uv run python src/build_notebook_21_lstm_contiguous.py` | emite 24 notebooks |
| 3 | Push de a **2 kernels** (límite de Kaggle) | `correr-kaggle.md` §2 |
| 4 | Si falla con `no kernel image is available` | cambiar GPU a **T4×2 desde la web** |
| 5 | Revisar el log de cada corrida | `correr-kaggle.md` §3 — **6 chequeos** |
| 6 | Descargar outputs | `correr-kaggle.md` §4 |
| 7 | `git add docs/resultados/ && git commit` | 16 CSVs nuevos con sufijo `_r1_` / `_r2_` |

⚠️ **El chequeo que no se puede saltear:** que el log diga `Fold: r1` (o `r2`).
Si dice `main`, el kernel está re-midiendo febrero y el output no sirve.

Cuando los 16 estén bajados, avisá: falta el **paso 4**, el builder que compara
los tres cortes y responde si el hallazgo aguanta fuera de febrero.

---

## 3. Pendientes

### Bloquean la publicación

| # | Tarea | Costo | Estado |
|---|---|---|---|
| P1 | **El manuscrito** | 3–4 semanas | No empezado |

### Cierran objeciones de revisor (no bloquean)

| # | Tarea | Costo | Por qué importa |
|---|---|---|---|
| R1 | Rolling origin: correr r1/r2 + builder comparativo | 1.8 h GPU + 1 día local | Tu aporte es una afirmación **general**; una sola ventana no la sostiene |
| R2 | Semillas para ConvLSTM y Transformer | GPU | Solo el LSTM tiene barrido; el nulo espacial podría ser mala suerte de inicialización |
| R3 | Nivelar el tuning del LSTM (24 configs) | ~14 h GPU | Ya está **declarado** como no atribuible. Baja prioridad |

### Deuda técnica (no afectan al paper)

| # | Tarea | Costo |
|---|---|---|
| D1 | Mergear `feat/rolling-origin` a `main` | minutos |
| D2 | `git push` de los 22 commits de `main` | minutos |
| D3 | 9 notebooks legacy (04–09) rancios contra sus builders | horas |

---

## 4. Orden recomendado

1. **Escribir el manuscrito.** No depende de nada de lo anterior.
2. Correr R1 mientras escribís, o después del borrador.
3. R2 solo si sobra cuota de GPU.
4. R3 y D3, probablemente nunca.

**Por qué el manuscrito primero:** escribir te va a decir con precisión cuánto
pesa cada objeción en *tu* argumento. Ahora lo estamos estimando. Y si al
redactar la sección de limitaciones concluís que declarar la limitación alcanza,
R1 no se corre y no se perdió nada — la infraestructura ya está construida.

Lo que **no** conviene es correr R1 ahora "porque ya está hecho": eso es razonar
por costo hundido.

---

## 5. Lo que NO hay que hacer

| | Por qué |
|---|---|
| Re-correr las familias 11/12/13, 17/18/19 | Congeladas. Sostienen el ranking entre arquitecturas, válido porque las tres comparten el mismo sesgo |
| Re-correr el XGBoost para rolling origin | Su papel ya está establecido sobre el corte publicado; agregaría ~6 h de CPU por una pregunta que nadie hizo |
| Editar un `.ipynb` a mano | Son artefactos generados. Se edita el builder y se regenera |
| Reentrenar por el desajuste de `max_N` | 0.05 % de filas, efecto 0.00067 min, fuera de la intersección y de todo verdicto |
