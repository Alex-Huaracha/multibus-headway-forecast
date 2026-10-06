# Papers de referencia estructural

Modelos de **estructura y redacción**, no fuentes. No se citan en `paper.md` y no
van en `fuentes-verificadas.md`.

El género es el mismo que el nuestro: el aporte es demostrar un defecto en una
práctica establecida, no proponer un método.

## La lista

| Paper | Venue | Qué demuestra | Qué tomamos |
|---|---|---|---|
| Kim, Choi, Choi, Lee, Yoon (2022) — *Towards a Rigorous Evaluation of Time-series Anomaly Detection* · [arXiv 2109.05257](https://arxiv.org/abs/2109.05257) · `papers/paper-ejemplo/kim2022.pdf` | AAAI | El protocolo *point adjustment* sobreestima tanto la detección que un puntaje aleatorio queda como estado del arte | La estructura del paper y la secuencia del resumen |
| Wu, Keogh (2021) — *Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress* · DOI `10.1109/TKDE.2021.3112126` · `papers/paper-ejemplo/wu2022.pdf` | IEEE TKDE | Cuatro defectos hacen que los archivos de referencia del campo no midan lo que se les atribuye | La redacción: cada cifra con su lectura |

## Kim: la estructura

| Montaje | El defecto | La reparación |
|---|---|---|
| §2 Background | §3 *Pitfalls of the TAD evaluation* | §4 *Towards a rigorous evaluation* |

- **El defecto y su reparación llevan su nombre en el encabezado.** Un revisor que
  solo hojee el índice ya sabe qué se demuestra. En el nuestro, §III es el defecto
  y §IV la reparación.
- **Una afirmación por subsección, con su evidencia.** La §3 de Kim tiene tres
  subsecciones: la formulación, «el protocolo sobreestima» y «un modelo sin
  entrenar puntúa comparable». Para un defecto con cadena causal, como el
  nuestro, el molde es ese: formulación, afirmación, afirmación.
- **El resumen sigue una secuencia fija**, sin cifras con decimales:
  1. lo que el campo cree («*giving the impression of clear improvements*»);
  2. la práctica sospechosa, con «However»;
  3. el defecto, en palabras («*even a random anomaly score can easily turn into
     a state-of-the-art method*»);
  4. la consecuencia («*misguided rankings*»);
  5. la reparación propuesta;
  6. la expectativa («*We expect that our study will help…*»).

## Wu y Keogh: la redacción

- **Cada cifra lleva su lectura al lado**, para que el lector no tenga que
  deducir si es buena o mala. Kim hace lo mismo: «*the reported F1 scores exceed
  0.9, giving an encouraging impression*».
- **El mecanismo se explica en palabras** antes que en fórmulas.

Su estructura **no** se traslada. Su §2 lleva una subsección por defecto porque
tiene cuatro defectos independientes. El nuestro es un solo defecto con cadena
causal, y partirlo en subsecciones inventaría hallazgos donde hay uno.
