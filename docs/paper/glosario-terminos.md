# Glosario de términos del paper

Registro de los nombres fijados para cada objeto técnico del manuscrito, con la
cuenta de usos verificada sobre `paper.md` y el nombre que le corresponde al
traducir.

Existe porque `reglas-redaccion.md` §4 obliga a un nombre por objeto pero no dice
cuál. Los defectos que se corrigieron no fueron términos mal elegidos: en cinco
casos el nombre correcto ya estaba escrito en el paper —`historial`,
`orígenes de evaluación`, `coordenada`, `denominador`, `coincidir`— y se
abandonaba a los pocos párrafos. La columna **Nunca** es la que hace el trabajo:
casi todo defecto fue una palabra correcta desplazada por un sinónimo, no un
término inventado.

La evidencia medida sobre las 11 referencias vive en `fuentes-verificadas.md`.
Aquí solo va la decisión.

## Términos fijados

| Objeto | Español | Usos | Inglés | Nunca |
| :--- | :--- | ---: | :--- | :--- |
| La línea que los buses siguen a lo largo del corredor | `eje` | 6 | `centerline` ¹ | trazado, línea central, curva principal, recorrido |
| Los buses de una empresa que circulan sobre una misma ruta | `corredor` | 65 | `corridor` | — |
| El trazado vial que el corredor recorre | `ruta` | — | `route` | — |
| El vehículo | `bus` | 51 | `bus` | unidad ², flota |
| La entidad dueña de los buses | `empresa` | 5 | `company` ³ | operador |
| Lo que el bus emite: identificador, instante y coordenada | `registro GPS` | 10 | `GPS record` | dato, emisión, ping, bitácora, registro ⁴ |
| La ubicación de un bus en el espacio | `coordenada` | 14 | `position` ⁵ | posición, punto ⁶, ubicación |
| El lado del eje hacia el que el bus avanza | `sentido` | 15 | `direction` ¹⁵ | dirección ¹⁶ |
| La recta que domina el desplazamiento de una empresa | `dirección` | 2 | `single direction` ¹⁵ | eje ¹⁷ |
| La retícula temporal de sesenta segundos | `rejilla` | 3 | ⁷ | — |
| La coordenada de todos los buses en un minuto | `snapshot` | 4 | `snapshot` | instantánea |
| Dos buses consecutivos del mismo sentido | `par` | 10 | `bus pair` | — |
| La casilla del vector de headways | `posición` | 40 | `position` ⁵ | slot, casilla, índice, componente ⁸ |
| La combinación de corredor y horizonte | `celda` | 30 | `case` ⁹ | par ¹⁴, combinación |
| Los $T$ minutos de historia que lee el modelo | `ventana de entrada` | 5 | `input window` ¹⁰ | ventana ¹¹ |
| Una re-ejecución completa del protocolo | `origen` | 32 | `evaluation origin`, `fold` | ventana |
| El instante en que el de adelante pasó por la coordenada del de atrás | `cruce` | 4 | `crossing` ¹⁸ | — |
| El punto a partir del cual el LSTM pasa a ganar | `frontera de régimen` | 5 | ¹⁹ | cruce, frontera ²⁰ |
| La tabla que cruza lo observado con lo predicho | `matriz de confusión` | 1 | `confusion matrix` ²¹ | cruce |
| La señal que el detector emite sobre una posición predicha | `trigger` | 19 | `trigger` ²² | disparo, alarma, alerta ²³ |
| Lo que hace la regla sobre lo observado al fijar el evento verdadero | `marcar` | 3 | `mark` ²² | trigger ²⁴ |
| El valor contra el que se compara el headway | `umbral` | 95 | `threshold` | corte, referencia |
| El valor del que el umbral es una fracción | `denominador` | 7 | `denominator` ¹² | referencia |
| El error que fijan los métodos sin ajuste | `error de referencia` | 2 | `reference` | línea base, benchmark |
| La realidad contra la que se compara la predicción | `observado` | — | `observed`, `observations` | de referencia |
| La dispersión entre los buses de un mismo instante | `dispersión transversal` | 4 | ⁷ | lateral ¹³ |

## Notas

1. No es un término heredado: `centerline`, `center line` y `centreline` dan
   **cero** usos en las 11 referencias, y los 28 de `axis` son ejes de gráficos en
   los papers de métricas. El nombre viene del código (`corridor.py`,
   `project_to_centerline`), y hay que declararlo al traducir.
2. Sobrevive un solo uso legítimo, el compuesto fijo «32 **unidades** ocultas» del
   LSTM. Cualquier otro es el defecto.
3. `operador` nombra un **rol** en el paper: «un operador solo dispone del pasado»,
   «lo que un operador llamaría bunching». La entidad es `empresa`, que además es
   la clave `empresaid` del corpus. **En inglés el choque se invierte**: la
   literatura usa `operator` para la entidad —Yu «transit **operators** use
   predictions», Boudabbous «tailored to a single city, **operator**, or
   corridor»—, 17 usos en 7 de 11. Por eso la entidad va a `company` (16 usos en
   4 de 11, en Santos y Moreira-Matias) y `operator` queda para el rol, como el
   paper ya lo reparte.
4. `registro` a secas no: `registro de incidentes` es otro objeto, el que VII
   declara ausente. Los dos compuestos van siempre completos.
5. **Las dos filas colapsan en `position` al traducir.** El código resuelve el
   choque como lo resolvería el inglés: reserva `position` para la casilla
   —«`pair_rank` indexes position WITHIN it», «this vector position»— y llama
   `ping` a la observación espacial. Al traducir hay que decidir esto
   explícitamente o la versión inglesa refunde los dos objetos que el español
   acaba de separar.
6. `punto` está tomado por **punto de operación**, que es el concepto del título.
7. Sin término que heredar. Para la rejilla se leyeron completos los dos trabajos
   de reconstrucción de trayectorias de transporte desde datos AVL —Huang,
   Abdelhalim, Stewart, Zhao y Koutsopoulos, «Reconstructing Transit Vehicle
   Trajectory Using High-Resolution GPS Data», y Robbennolt y Munira, «A
   Comparative Study of Spline-Based Trajectory Reconstruction Methods Across
   Varying Automatic Vehicle Location Data Densities»— y dan **cero** usos de
   `time grid`, `uniform`, `resample` y `discretize`: reconstruyen una trayectoria
   continua, así que nunca nombran una retícula. `interpolat*` sí, 54 y 11 usos.
   Acuñado y definido en el cuerpo, que es lo correcto cuando el campo no da
   nombre. Ninguno de los dos se cita en el paper; sostienen esta decisión y nada
   más.
8. `componente` no: III-C ya escribe el vector «por **componentes**».
9. **`celda` no se traduce como `cell`.** En español es un término acuñado y
   definido en IV-B, y eso basta. En inglés `cell` está tomado tres veces en
   nuestro propio corpus: la celda geográfica H3 de Boudabbous, el `cell state` de
   la LSTM en Jiao —y este paper habla de LSTM— y la celda de tabla en Patton.
   Ninguna referencia lo usa como unidad experimental. El término que sí existe es
   `case`: 126 usos en **11 de 11**, seguido de `condition` 53 y `combination` 47.
10. Solo respaldo del código (`windowing.py`, `T_in`). **Ninguno** de los catorce
    candidatos ingleses —`input window`, `lookback`, `sequence length`…— aparece en
    las 11 referencias; esos trabajos no alimentan secuencias, así que no nombran
    el objeto. No es un término heredado.
11. `ventana` va **siempre con calificador**. Es la regla del código, que escribe
    `test window` e `input window` pero nunca `window` a secas, y la de Boudabbous
    2026: «For each **fold**, models are trained on past data and evaluated on a
    subsequent validation and test **window**».
12. El campo no abstrae el objeto: nombra el valor concreto —`scheduled headway`,
    `planned headway`, `average headway`—. `denominator` es el nombre interno del
    paper, no una herencia.
13. `lateral` es otro objeto: el desvío de una coordenada respecto del eje, del
    paso 2 de III-A. No son intercambiables.
14. `par` no: nombra el par de buses.
15. **`sentido` y `dirección` son dos objetos, no un término y su sinónimo.** En
    castellano un vector tiene módulo, dirección —la recta, sin signo— y sentido
    —cuál de sus dos lados—. El código usa `direction` para ambos: el
    `direction: Int8 ∈ {+1, −1}` de `direction.py:51`, que es llave de partición
    de los datos, es el **sentido**; el «route shape is dominated by a single
    **direction**» de `build_notebook_01.py:41` —`PC1_var / PC2_var ≥ 4`,
    `pca_ratio`— es la **dirección**. Al traducir colapsan igual que `coordenada`
    y `posición` en la nota 5, y hay que calificar el segundo con la frase del
    propio notebook 01. Heredado hay poco: de los 25 usos de `direction` medidos
    en las 11 referencias, solo 2 documentos lo aplican a la marcha
    —Moreira-Matias «Each line has two **route-directions** A1, A2…» y Boudabbous,
    campo GTFS «vehicle identifier, route, **direction**, and timestamp»—; los
    otros 21 son la acepción abstracta («future research directions», «the
    **direction** of the suboptimality»). `travel direction`, `direction of
    travel`, `heading`, `inbound`, `outbound` y `bearing` dan **cero**. Y `sense`
    nunca es la marcha: sus 3 usos son «in a broader sense» y «in the sense
    that».
16. La marcha nunca es `dirección`. Dos usos lo eran —el paso 3 de III-A y V-F— y
    pasaron a `sentido`.
17. `eje` no: la dirección se mide **antes** de que el eje exista, sobre
    coordenadas crudas y por otro procedimiento —PCA sobre `(lat, lon)` en el
    notebook 01, contra la mediana por bins del 04—. El eje es la curva ajustada;
    la dirección es la recta que la hace posible.

18. Sin término que heredar: `crossing`, `crossed` y `crosses` dan **cero** usos
    en las 11 referencias, porque todas miden en paradas y nada cruza nada.
    Acuñado, pero definido en la 240 y respaldado por el nombre de la formulación
    en el código —«Opción C.2 — **trailing crossing**», `build_notebook_04.py:70`,
    y «a pair with no **crossing**», `build_headway_coverage.py:4`.
19. **En inglés este objeto no se nombra con un sustantivo: se dice con verbo**,
    «the LSTM **outperforms** persistence from three minutes on» —`outperform`, 31
    usos en 7 de 11—. Los sustantivos candidatos no existen: `crossover` **cero**,
    `cross over` **cero**, `break-even` **cero**, `tipping point` **cero**,
    `turning point` **cero**, `regime boundary` **cero**. Y `boundary`, que tiene
    40 usos, no sirve: 29 son `decision boundary` en un trabajo de clasificación, y
    este paper clasifica bunching. Traducir `cruce` por `crossing` y este objeto
    por `crossover` recrearía en inglés el defecto que el castellano acaba de
    quitar, porque comparten raíz.
20. `frontera` **nunca va sola**, como `ventana`. La mitad anclada del compuesto
    es `régimen`, que el código define: «The **regime** is the ex-ante
    input-window dispersion, binned with p33/p66 frozen on train+val»
    (`build_contiguous_router.py:27`), con su `assign_regime`. `frontera` sola
    solo tiene un docstring detrás (`build_contiguous_figures.py:300`), y en las
    11 referencias `frontier` da **cero**.
21. El concepto ya estaba construido sin nombre: la 557 define TP, FP, FN y TN una
    por una. `confusion matrix` da 46 usos, y están en Chicco (39) y Fawcett (7),
    que son justo las dos referencias que el paper cita para el MCC y la curva
    ROC. En el código, `vector_metrics.py:59`. No se presta sin traducir porque
    «matriz de confusión» es el nombre estándar en castellano; `headway` se presta
    porque no lo tiene.

22. **`trigger` es préstamo declarado, no traducción pendiente.** Se toma de
    Moreira-Matias, que es quien nombra este objeto: «Frequency-based threshold to
    **trigger** a BB alarm on stop Bj», «once a BB alarm is **triggered**». Hay que
    saber dos cosas al traducir. La primera: **en la fuente `trigger` es el verbo y
    `alarm` el sustantivo**, de modo que la versión inglesa dirá «the detector
    **triggers** an alert on position $i$» y no «a trigger». La segunda: **la
    `tasa de trigger` es acuñada** —el campo no nombra esa fracción— y hay que
    declararla. El resto del vocabulario del campo: Yu usa `alert` como verbo
    —«we aim to **alert** when and where the possible bus bunching will occur»— y
    `notice` como sustantivo; Jiao, «issue BB **warnings**». `fire` da **cero**
    usos y `flag` **uno**, así que el `fire_rate` del código no se traduce literal.
    **Corrección a una cuenta anterior**: los 38 de `alert` estaban inflados —21
    son nombres de variable de Waze en Santos (`alertSubtype`,
    `alertNThumbsUp`), no el detector—. Para lo observado, `mark` o `label`.
23. `alarma` y `alerta` **están vetadas por el propio paper**: V-H abre con «Eso no
    es una **alarma**. Una alarma tiene que sonar cuando ocurre el evento», y
    concluye que lo que queda es un filtro de prioridad. Usar el sustantivo que la
    sección niega contradiría su conclusión operativa. `disparo` se descartó por
    legibilidad fuera del gremio: se entiende en un contexto de software, no en uno
    de transporte.
24. **El paper ya trazaba la línea entre los dos objetos**, en dos frases seguidas
    de V-C: «La regla… aplicada a lo observado, **marcó** 15 245 eventos… Aplicada
    a lo predicho…, emitió catorce **triggers**». Doce usos la habían cruzado y
    volvieron. La frase que peor la cruzaba era la definición misma de la tasa, en
    IV-D, que decía «la fracción de posiciones que **marca** como evento». Ahora
    las dos se definen sobre su ecuación: `marcar` en la (5) y `trigger` en la (6),
    en lugar de deducirse doscientas líneas después.

## Nota sobre la concordancia

La locución «los dos cruces van en el mismo **sentido**» era un tercer objeto
colado en `sentido`: el acuerdo entre dos métricas. No hubo que acuñar nada,
porque el paper ya escribía `coincidir` ocho líneas antes —V-E: «Los dos
**coincidieron** en las doce celdas: en cada una ganó el mismo método»—, y también
en V-F, en el encabezado «Coinciden» de la Tabla 3 y en el suptítulo que
`build_contiguous_figures.py` genera («las dos métricas **coinciden**»).

Quedó corregido en V-E, en el pie de la Fig. 8, en
`build_contiguous_figures.py:522` y en el PNG que ese pie imprime. «Cruzó en
sentido contrario» pasó a «cruzó **a favor de la persistencia**», que además dice
cuál de los dos.

Al traducir, `direction` arrastra esa segunda acepción y con precedente —Patton
escribe «the **direction** of the suboptimality»—, de modo que la locución no debe
reintroducirse por esa vía: lo que corresponde es `agree` o `coincide`.
