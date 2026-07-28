# Qué falta y en qué orden

Actualizado: 2026-07-28.

> **Lo único que bloquea la publicación es el manuscrito.** Todo lo demás de esta
> lista cierra objeciones de revisor o es deuda técnica. Ninguno cambia el
> hallazgo central.

---

## 1. Dónde estamos

| | |
|---|---|
| Rama `main` | Pipeline contiguo mergeado. Al día con el remoto. |
| Rama `feat/rolling-origin` | 6 commits. **4 sin pushear. Sin mergear.** |
| Suite de tests | 1540 pasando (13 excluidos por entorno: `libomp`, ABI torch/numpy) |
| Hallazgo central | Medido, con guardián sobre cada cifra del documento |
| Rolling origin | **Cerrado.** 24 corridas, 11/12 celdas coinciden en los tres orígenes |

El aporte del paper es **la disociación**: los modelos óptimos en MAE
sub-reportan sistemáticamente la irregularidad del servicio, y una evaluación
escalar es estructuralmente incapaz de detectarlo. A h=10 el LSTM gana el MAE por
1.47 min y pierde el F1 de detección de *bunching* por un factor de 253.

---

## 2. Rolling origin: cerrado

Las 24 corridas de la familia 21 se relanzaron y validaron contra los 6 chequeos
del runbook. El builder comparativo está escrito y su salida vive en la Sección 4
del documento de resultados, sección *"¿Y si el mes fuera otro?"*.

**11 de 12 celdas ponen la victoria del mismo lado en los tres orígenes.** A h≥5
las 18 celdas dan ventaja al aprendiz y las 18 son significativas. La única que
se da vuelta, E4 h=3, ya era no significativa en la ventana publicada.

Dos cosas que salieron de ahí y conviene no olvidar:

- El re-entrenamiento reprodujo los 8 CSVs de resultados del corte publicado
  **byte por byte**. Es reproducibilidad demostrada, y sirve para el paper.
- El slug `21-lstm-contiguous-h10-r2` quedó podrido en Kaggle; esa corrida vive
  en `-h10-r2b` (ver `POISONED_SLUGS` en el builder y `correr-kaggle.md` §5).

---

## 3. Pendientes

### Bloquean la publicación

| # | Tarea | Costo | Estado |
|---|---|---|---|
| P1 | **El manuscrito** | 3–4 semanas | No empezado |

### Cierran objeciones de revisor (no bloquean)

| # | Tarea | Costo | Por qué importa | Estado |
|---|---|---|---|---|
| ~~R1~~ | ~~Rolling origin~~ | — | — | ✅ **Cerrado 2026-07-28** |
| R2 | Semillas para ConvLSTM y Transformer | GPU | Solo el LSTM tiene barrido (`15_lstm_multiseed`); el nulo espacial podría ser mala suerte de inicialización | Abierto |
| R3 | Nivelar el tuning del LSTM (24 configs) | ~14 h GPU | Ya está **declarado** como no atribuible. Baja prioridad | Abierto |

### Deuda técnica (no afectan al paper)

| # | Tarea | Costo |
|---|---|---|
| D1 | Mergear `feat/rolling-origin` a `main` | minutos |
| D2 | `git push` de los 4 commits de `feat/rolling-origin` — **el trabajo de hoy no está respaldado en ningún lado** | minutos |
| D3 | 9 notebooks legacy (04–09) rancios contra sus builders | horas |

---

## 4. Orden recomendado

1. **D1 y D2: mergear y pushear.** Minutos, y hoy el trabajo vive en un solo disco.
2. **Escribir el manuscrito.** No depende de nada de lo anterior.
3. R2 solo si sobra cuota de GPU.
4. R3 y D3, probablemente nunca.

**Por qué el manuscrito primero:** escribir te va a decir con precisión cuánto
pesa cada objeción en *tu* argumento. Ahora lo estamos estimando. Con R1 cerrado,
la investigación ya sostiene lo que el paper afirma; lo que queda son objeciones
declarables, no agujeros.

Y el recordatorio de calibración, que no cambió: en 8–9 papers recientes de
IJACSA, los tests formales de significancia aparecen en 2 y las secciones de
amenazas a la validez en 0. **El rigor ya está muy por encima de la mediana del
venue.** Agregar más tiene rendimiento decreciente; escribir no.

---

## 5. Lo que NO hay que hacer

| | Por qué |
|---|---|
| Re-correr las familias 11/12/13, 17/18/19 | Congeladas. Sostienen el ranking entre arquitecturas, válido porque las tres comparten el mismo sesgo |
| Re-correr el XGBoost para rolling origin | Su papel ya está establecido sobre el corte publicado; agregaría ~6 h de CPU por una pregunta que nadie hizo |
| Editar un `.ipynb` a mano | Son artefactos generados. Se edita el builder y se regenera |
| Reentrenar por el desajuste de `max_N` | 0.05 % de filas, efecto 0.00067 min, fuera de la intersección y de todo verdicto |
