# Papers de referencia estructural

Modelos de **estructura**, no fuentes. No se citan en `paper.md` y no van en
`fuentes-verificadas.md`.

El género es el mismo que el nuestro: el aporte es demostrar un defecto en una
práctica establecida, no proponer un método.

## La lista

| Paper | Venue | Qué demuestra |
|---|---|---|
| Kim, Choi, Choi, Lee, Yoon (2022) — *Towards a Rigorous Evaluation of Time-series Anomaly Detection* · [arXiv 2109.05257](https://arxiv.org/abs/2109.05257) | AAAI | El protocolo *point adjustment* sobreestima tanto la detección que un puntaje aleatorio queda como estado del arte |
| Musgrave, Belongie, Lim (2020) — *A Metric Learning Reality Check* · [arXiv 2003.08505](https://arxiv.org/abs/2003.08505) | ECCV | Las comparaciones del campo son injustas y sus métricas premian a lo que no discrimina |
| Dacrema, Cremonesi, Jannach (2019) — *Are We Really Making Much Progress?* · [arXiv 1907.06902](https://arxiv.org/abs/1907.06902) | RecSys | Seis de siete métodos neuronales no superan a líneas base bien ajustadas |
| Dacrema, Boglio, Cremonesi, Jannach (2021) — *A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research* · [arXiv 1911.07698](https://arxiv.org/abs/1911.07698) | ACM TOIS | Versión extendida del anterior, 49 páginas |

| Wu, Keogh (2021) — *Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress* · `papers/paper-ejemplo/wu2022.pdf` · DOI `10.1109/TKDE.2021.3112126` | IEEE TKDE | Cuatro defectos hacen que los archivos de referencia del campo no midan lo que se les atribuye, y que el progreso publicado sea ilusorio |

## Las anatomías, lado a lado

| | Montaje | El defecto | La reparación |
|---|---|---|---|
| Kim et al. 2022 | §2 Background | §3 *Pitfalls of the TAD evaluation* | §4 *Towards a rigorous evaluation* |
| Musgrave et al. 2020 | §1 Overview (*Related work* es **subsección**) | §2 *Flaws in the existing literature* | §3 *Proposed evaluation method* |
| Wu y Keogh 2021 | no tiene: *Related Work* es **§2.1**, dentro del defecto | §2 *A Taxonomy of Benchmark Flaws* | §3 *Introducing the UCR Anomaly Archive* · §4 *Recommendations* |
| Dacrema et al. 2019 | §2 Research Method | §3 *Validation Against Baselines* | — |
| Dacrema et al. 2021 | §2–3 Scope y Evaluation Methodology | §4 *Results — Analysis of…* | §5.4 Guidelines |

El de Wu y Keogh es el caso extremo y es de revista. Nueve páginas, sin sección
de antecedentes: la §2 arranca con «before discussing the four major flaws found
in many public archives, we will briefly discuss related work, to put our
observations into context», y de ahí en adelante **una subsección por defecto**,
cada una con el nombre del defecto:

```
2   A Taxonomy of Benchmark Flaws
2.1   Related Work
2.2   Triviality
2.3   Unrealistic Anomaly Density
2.4   Mislabeled Ground Truth
2.5   Run-to-failure Bias
2.6   Summary of Benchmark Flaws
3   Introducing the UCR Anomaly Archive
4   Recommendations
5   Conclusions
```

## Lo que los cinco comparten

1. **Ninguno tiene sección de trabajos relacionados ni de fundamentos teóricos
   separadas del montaje.** El montaje es una sección. En Musgrave y en Wu,
   *Related work* es una subsección, y en Wu está **dentro de la sección del
   defecto**: el lector se entera de lo previo mientras se le muestra qué falla.
2. **El defecto y su reparación llevan su nombre en el encabezado.** Un revisor
   que solo hojee el índice ya sabe qué se demuestra.
3. **El detalle experimental baja a apéndice.** Dacrema 2021 tiene 49 páginas en
   revista y sigue sin trabajos relacionados: al pasar de conferencia a revista
   ganó apéndices, no secciones de frente.

## Cómo escriben el contenido

- **Musgrave, §2:** tres subsecciones, **un párrafo cada una**. Cada una abre
  diciendo qué hace el campo —«most metric learning papers use Recall@K, NMI and
  F1»— y cierra mostrando el fallo con una tabla propia. Dos movimientos.
- **Kim, §3:** tres subsecciones, 15–20 % del paper. Una afirmación por
  subsección: formulación, «el protocolo sobreestima», «un modelo sin entrenar
  puntúa comparable».
- **Wu y Keogh, §2:** una subsección por defecto, con el nombre del defecto en el
  título —*Triviality*, *Mislabeled Ground Truth*—, y una §2.6 que cierra
  resumiéndolos. La reparación es una sección aparte y las recomendaciones otra.
- **Kim difiere datasets y métricas hasta después del defecto.** Dacrema no.
  Ahí no hay patrón, es una decisión.

La regla que sí es patrón: **una afirmación por subsección, con su evidencia**,
y no una narrativa que encadena matices.

## Lo que de Wu NO se traslada

Su §2 lleva una subsección por defecto porque tiene **cuatro defectos
independientes**: cualquiera de los cuatro se lee sin los otros tres. Lo nuestro
es un defecto con cadena causal —la compresión produce el colapso, el colapso se
localiza en el umbral, el umbral en minutos es la propiedad culpable—, y partirlo
en cuatro subsecciones con nombre de defecto inventaría cuatro hallazgos donde
hay uno. Para una cadena, el molde es el de Kim: formulación, afirmación,
afirmación.

## Pendientes que salen de estas anatomías

- [x] **Cierre tipo §2.6 de Wu al final de §III.** Aplicado: §III-D «Alcance del
  colapso» enuncia la cadena en limpio antes de pasar a §IV.
- [ ] **Apretar §III y §IV a una afirmación por subsección**, con el molde de
  Musgrave: qué hace el campo, la medición que lo desmiente, la tabla.
- [x] **Decidir si Trabajos previos se mueve junto al defecto.** Resuelto en la
  dirección de Kim, no la de Wu: los previos salieron de §III y viven en §II-D,
  dentro del montaje. El error escalar (ex §III-B) bajó a prosa de apertura de
  §III, y las ex §III-E y §III-F se fundieron en §III-C. §III quedó en
  compresión → colapso → ordenamiento → alcance.
