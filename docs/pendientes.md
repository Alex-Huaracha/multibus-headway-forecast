# Qué falta y en qué orden

Actualizado: 2026-07-29.

> **Lo único que bloquea la publicación es el manuscrito.** Todo lo demás cierra
> objeciones de revisor o es deuda técnica.

Este archivo no lleva el estado de git ni el conteo de tests: eso se consulta,
no se documenta. Solo lo que hay que decidir o hacer.

---

## 1. Bloquean la publicación

| # | Tarea | Costo | Estado |
|---|---|---|---|
| P1 | **El manuscrito en inglés** | 3–4 semanas | No empezado — es lo único que queda bloqueando |
| P2 | **Reescribir las contribuciones C1–C4** de `docs/paper/manuscrito.md` | horas | ✅ **Cerrado 2026-07-30** |
| P3 | **Las referencias** | días | No arranca de cero — ver abajo |

**P2 se cerró el 2026-07-30**, después de leer los cuatro papers de *paywall*.
El manuscrito está reestructurado entero (título, vacío, C1–C4, §I-D, §II en
A–E, §IV reordenado) y las contribuciones ya incorporan los cuatro veredictos.
Lo que sigue es prosa, no estructura.

El 2026-07-29 se había retirado el titular *"la métrica decide el ganador"* y el
mecanismo *"el MAE premia contraer"*: el primero era un artefacto de umbral, el
segundo lo contradicen nuestros propios datos a h=1. El aporte ahora es otro y
está medido:

> Un pronóstico puntual está sub-disperso (sesgo de CV negativo en las 36
> celdas). Su costo es **de unidades, no de información**: una regla de evento
> calibrada sobre observaciones no es transportable al pronóstico, y
> trasplantarla fabrica una degradación aparente de hasta 253× que no existe.
> Sin umbral, el LSTM gana la detección a h=10 en los tres corredores y en los
> tres orígenes.

Detalle completo en `docs/resultados/documento-resultados.md` §1, §5.3 y §5.4.
Las C1–C4 actuales afirman lo contrario de eso.

**Sobre P3, corrigiendo una versión anterior de este archivo.** No hay ningún
archivo `.bib` en el repo, pero **no se arranca de cero**: `docs/paper/manuscrito.md`
§II ya tiene 10 referencias reales con identificador (arXiv / DOI / PII) y prosa
redactada alrededor. Lo que falta es (a) convertirlas al formato IJACSA,
(b) verificar a texto completo los campos marcados `[por confirmar]`, y
(c) sumar las fuentes del encuadre nuevo — sub-dispersión de pronósticos
puntuales, calibración de umbrales de evento, y métricas de detección con
desbalance moderado.

La referencia **[3]** (Li, Yang y Wang, *over-stationarization* en predicción de
arribos de buses, arXiv:2509.06979) pasa de nota al pie a **vecino más cercano**
del hallazgo: la *over-stationarization* es el mismo fenómeno de sub-dispersión
que medimos. Hay que leerla a texto completo y delimitar con precisión qué
afirma ella y qué agregamos nosotros.

---

## 2. Cierran objeciones de revisor (no bloquean)

| # | Tarea | Costo | Por qué importa | Estado |
|---|---|---|---|---|
| R2 | Semillas para ConvLSTM y Transformer | GPU | Solo el LSTM tiene barrido (`15_lstm_multiseed`); el nulo espacial podría ser mala suerte de inicialización | Abierto |
| R3 | Nivelar el tuning del LSTM (24 configs) | ~14 h GPU | Ya está **declarado** como no atribuible. Baja prioridad | Abierto |
| R4 | XGBoost en los orígenes de rolling | ~6 h GPU | Todo lo que involucra al árbol sigue apoyado en una sola ventana (limitación 2 del documento) | Abierto — **declarado, no bloquea** |

R1 (rolling origin) cerrado el 2026-07-28: 24 corridas, y el re-entrenamiento
reprodujo los 8 CSVs del corte publicado **byte por byte**. Eso sirve para el
paper como reproducibilidad demostrada.

---

## 2-bis. Papers de *paywall* — ✅ los cuatro críticos, leídos

PDFs en `docs/paper/papers/`. Hallazgos completos en
`docs/paper/fuentes-verificadas.md` §0. Resumen:

| Paper | Veredicto |
|---|---|
| Mayer & Yang 2022 | ⚠️ **La amenaza se confirma.** *"MSE-optimized forecasts are always underdispersed"* es literal. **C1 reformulado**: queda el CV transversal sobre el vector, la inversión del TCQSM y la consecuencia sobre la regla de evento. Ellos: cero umbrales, cero detección, y su objeto es una serie escalar |
| Santos et al. 2022 | ✅ **Sin barrido de umbral. C2 sobrevive.** Y su tabla de la literatura previa confirma que **nadie reporta AUC ni precisión media** |
| Rezazada et al. 2024 | ✅ **Verificado:** *"no existe un único valor de umbral"*, y el rango va de 20 s a ¼ del programado |
| Yu et al. 2016 | ✅ **Reconciliado, y nos ayuda.** Su >95 % es a **2 paradas**; su propia sensibilidad cae a **73 % a 5 paradas**. Y usan el *headway* observado de la primera parada como sustituto del horario ausente — mismo problema nuestro, misma clase de solución |

**Sigue faltando, prioridad media:** Jiao, Shen & Zhang, IEEE ICITE 2023,
doi:`10.1109/icite59717.2023.10733869`. Reclama 89 % con LSTM → umbral.
Probablemente se reconcilia igual que Yu et al., pero hay que verificarlo.

**Y uno nuevo que salió de Rezazada:** Gong et al. (2020), umbral **variable**
según tipo de servicio e información en tiempo real. Revisar que no amenace C2.

---

## 3. Deuda técnica (no afecta al paper)

| # | Tarea | Costo |
|---|---|---|
| D1 | 9 notebooks legacy (04–09) rancios contra sus builders | horas |
| D2 | Decidir si se borra `docs/resultados/residuos-multihorizon.stale/` (1.7 GB, gitignored, irreversible) | minutos |

---

## 4. Orden recomendado

1. ~~P2: reescribir C1–C4.~~ ✅ Cerrado 2026-07-30.
2. **P1 + P3: escribir, con las referencias en paralelo.** Dentro de P1, el
   orden es §II (A→E) primero, porque todo el resto del artículo cita hacia
   atrás a esa sección y sus fuentes ya están verificadas; después §III y §IV,
   que están casi redactadas en `documento-resultados.md`; después §V; y el
   Abstract y la Introducción **al final**, como dice el propio andamiaje.
3. R2 solo si sobra cuota de GPU. R3, R4 y D1: probablemente nunca.

Y el recordatorio de calibración, que no cambió: en 8–9 papers recientes de
IJACSA, los tests formales de significancia aparecen en 2 y las secciones de
amenazas a la validez en 0. **El rigor ya está muy por encima de la mediana del
venue.** Agregar más tiene rendimiento decreciente; escribir no.

---

## 5. Lo que NO hay que hacer

| | Por qué |
|---|---|
| Re-correr las familias 11/12/13, 17/18/19 | Congeladas. Sostienen el ranking entre arquitecturas, válido porque las tres comparten el mismo sesgo |
| Editar un `.ipynb` a mano | Son artefactos generados. Se edita el builder y se regenera |
| Reentrenar por el desajuste de `max_N` | 0.05 % de filas, efecto 0.00067 min, fuera de la intersección y de todo verdicto |
| **Renombrar `s_front`/`s_back`/`bus_front`/`bus_back`** | Las etiquetas están invertidas respecto del movimiento físico y **la aritmética es correcta**. Los nombres están en el esquema de los parquets, en los builders y en los residuos ya bajados. Renombrar obliga a regenerar todo por ganancia nula. Ver `src/preprocessing/headways.py` y `docs/decisiones-headway-fase2.md` §2.1 |
| Reentrenar por la corrección del titular | El artefacto era de **evaluación**, no de entrenamiento. Los residuos en disco sirven tal cual; todo se recalculó sobre ellos sin GPU |
| Volver a reportar el F1 de *bunching* con el corte fijo como si midiera al modelo | Es la afirmación retirada. Con tasa base del 17–30 %, "marcar todo" supera al ganador declarado en 5 de 12 celdas. Los veredictos van por AUC y MCC |
