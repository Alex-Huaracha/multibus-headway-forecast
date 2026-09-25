# Glosario de términos del paper

Registro de los nombres fijados para cada objeto técnico del manuscrito, con la
cuenta de usos verificada sobre `paper.md` y el nombre que le corresponde al
traducir.

Existe porque `reglas-redaccion.md` §5 obliga a un nombre por objeto pero no dice
cuál. Los defectos que se corrigieron no fueron términos mal elegidos: en seis
casos el nombre correcto ya estaba escrito en el paper —`historial`,
`orígenes de evaluación`, `coordenada`, `denominador`, `coincidir`, `celda`— y se
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
| La entidad dueña de los buses | `empresa` | 6 | `company` ³ | operador |
| La colección de registros de los tres corredores, cruda o procesada | `dataset` | 4 | `dataset` | corpus, conjunto |
| Lo que el bus emite: identificador, instante y coordenada | `registro GPS` | 10 | `GPS record` | dato, emisión, ping, bitácora, registro ⁴ |
| La ubicación de un bus en el espacio | `coordenada` | 14 | `position` ⁵ | posición, punto ⁶, ubicación |
| El lado del eje hacia el que el bus avanza | `sentido` | 15 | `direction` ¹⁵ | dirección ¹⁶ |
| La recta que domina el desplazamiento de una empresa | `dirección` | 2 | `single direction` ¹⁵ | eje ¹⁷ |
| La retícula temporal de sesenta segundos | `rejilla` | 3 | ⁷ | — |
| La coordenada de todos los buses en un minuto | `snapshot` | 4 | `snapshot` | instantánea |
| Dos buses consecutivos del mismo sentido | `par` | 10 | `bus pair` | — |
| La casilla del vector de headways | `posición` | 40 | `position` ⁵ | slot, casilla, índice, componente ⁸ |
| La combinación de corredor y horizonte | `celda` | 31 | `case` ⁹ | par ¹⁴, combinación |
| Los $T$ minutos de historia que lee el modelo | `ventana de entrada` | 5 | `input window` ¹⁰ | ventana ¹¹ |
| Una re-ejecución completa del protocolo | `origen` | 32 | `evaluation origin`, `fold` | ventana |
| El instante en que el de adelante pasó por la coordenada del de atrás | `cruce` | 4 | `crossing` ¹⁸ | — |
| El punto a partir del cual el LSTM pasa a ganar | `frontera de régimen` | 5 | ¹⁹ | cruce, frontera ²⁰ |
| La tabla que cruza lo observado con lo predicho | `matriz de confusión` | 1 | `confusion matrix` ²¹ | cruce |
| La señal que el detector emite sobre una posición predicha | `trigger` | 19 | `trigger` ²² | disparo, alarma, alerta ²³ |
| Lo que hace la regla sobre lo observado al fijar el evento verdadero | `marcar` | 3 | `mark` ²² | trigger ²⁴ |
| El valor contra el que se compara el headway | `umbral` | 99 | `threshold` | corte, referencia, punto de operación ²⁵ |
| El valor del que el umbral es una fracción | `denominador` | 7 | `denominator` ¹² | referencia |
| El error que fijan los métodos sin ajuste | `error de referencia` | 1 | `reference` | línea base, benchmark, baseline ²⁶ |
| La regla que marca bunching en toda posición, contra la que se lee todo F1 | `baseline siempre positivo` | 6 | `always-positive baseline` ²⁶ | detector trivial, piso trivial, regla vacía |
| El headway promedio de cada posición del vector en un período anterior, repetido cada minuto; contra él se lee todo AUC | `baseline de promedio histórico por posición`; después, `promedio histórico por posición` | 11 | `per-position historical average` ²⁷ | perfil posicional, piso posicional, baseline posicional, nulo posicional |
| La comparación de dos métodos sobre las mismas muestras bajo una métrica declarada | `comparación pareada` | 2 | `paired comparison` | veredicto ²⁵ |
| La realidad contra la que se compara la predicción | `observado` | — | `observed`, `observations` | de referencia |
| La dispersión entre los buses de un mismo instante | `dispersión transversal` | 4 | ⁷ | lateral ¹³ |
| Lo predicho con menos dispersión transversal que lo observado, por el ajuste con error cuadrático | `subdispersión`, `subdisperso` | 19 | `underdispersion`, `underdispersed` (Mayer y Yang) | compresión, comprimido, encogido, aplanado |

## Notas

1. No es un término heredado: `centerline`, `center line` y `centreline` dan
   **cero** usos en las 11 referencias, y los 28 de `axis` son ejes de gráficos en
   los papers de métricas. El nombre viene del código (`corridor.py`,
   `project_to_centerline`), y hay que declararlo al traducir.
2. Sobrevive un solo uso legítimo, el compuesto fijo «32 **unidades** ocultas» del
   LSTM. Cualquier otro es el defecto.
3. `operador` nombra un **rol** en el paper: «un operador solo dispone del pasado»,
   «lo que un operador llamaría bunching». La entidad es `empresa`, que además es
   la clave `empresaid` del corpus. **El inglés reparte igual**, medido sobre los
   20 usos de `operator` en 7 de las 11 referencias: unos 14 son el rol —Boudabbous
   «control center **operators** must identify delayed vehicles… and decide when to
   intervene», Yu «transit **operators** can adopt preventive countermeasures», Jiao
   «support transit **operators** to issue BB warnings»—, unos 5 la entidad
   —Boudabbous «tailored to a single city, **operator**, or corridor»— y uno el
   conductor —Yu «bus **operators** are a dominant factor in causing running time
   variation»—. Para la entidad el inglés dispone de `company`/`companies` (15 usos
   en 3) y `agency`/`agencies` (11 en 3). Así que `operator` traduce el rol y
   `company` la entidad. El único contagio venía del título de Trompet, «Regularity
   of Service between Urban Bus **Operators**», y ya se corrigió en el paper.
4. `registro` a secas no: `registro de incidentes` es otro objeto, el que VII
   declara ausente. Los dos compuestos van siempre completos.
5. **Las dos filas colapsan en `position` al traducir.** El código resuelve el
   choque como lo resolvería el inglés: reserva `position` para la casilla
   —«`pair_rank` indexes position WITHIN it», «this vector position»— y llama
   `ping` a la observación espacial. Al traducir hay que decidir esto
   explícitamente o la versión inglesa refunde los dos objetos que el español
   acaba de separar.
6. `punto` quedaba tomado por **punto de operación**. Ese término salió del
   manuscrito el 2026-09-24 (nota 25), así que la restricción ya no aplica.
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
23. `alarma` y `alerta` **están vetadas por el propio paper**: V-H abre con «Eso
    no es una **alarma**. Una alarma tiene que sonar cuando ocurre el evento», y
    concluye que lo que queda es un filtro de prioridad. Usar el sustantivo que
    la sección niega contradiría su conclusión operativa. **Corrección
    2026-09-24:** esa frase ya no existe en `paper.md` —la V-H se eliminó—, de
    modo que el veto perdió su fundamento. El sustantivo del campo es `alarm`
    (Moreira-Matias, Sun, Yu); la decisión de cambiar `trigger` sigue abierta.
    `disparo` se descartó por legibilidad fuera del gremio: se entiende en un
    contexto de software, no en uno de transporte.
24. **El paper ya trazaba la línea entre los dos objetos**, en dos frases seguidas
    de V-C: «La regla… aplicada a lo observado, **marcó** 15 245 eventos… Aplicada
    a lo predicho…, emitió catorce **triggers**». Doce usos la habían cruzado y
    volvieron. La frase que peor la cruzaba era la definición misma de la tasa, en
    IV-D, que decía «la fracción de posiciones que **marca** como evento». Ahora
    las dos se definen sobre su ecuación: `marcar` en la (5) y `trigger` en la (6),
    en lugar de deducirse doscientas líneas después.

25. **Retirados el 2026-09-24 por no ser términos del campo.** `punto de
    operación` (*operating point*) solo aparece en Fawcett 2006, el tutorial de
    ROC; ningún paper de bunching lo usa, y todos dicen `threshold` —Yu, Jiao,
    Santos, Sun `bunching threshold`—. `veredicto` (*verdict*) da **cero** usos
    en las 22 referencias: el campo no nombra ese objeto, dice qué le pasa a la
    comparación —Kim «overestimate», Wu «illusion of progress»—. En el paper
    quedó `qué método gana` o `la comparación sin umbral`, y en el Apéndice B,
    donde se definía, `comparación pareada`. En el código sobreviven `verdict()`
    y comentarios con «operating point»; no se traducen al paper.
26. **`baseline` se presta sin traducir, en masculino** (`el baseline`), porque
    se entiende mejor que `línea base`. Nombra solo las dos varas de medir,
    nunca a la persistencia, que conserva `error de referencia`. `detector`
    queda reservado para la regla de la Ecuación (5): llamar «detector» a la
    vara de medir juntaba dos objetos de papel opuesto. El nombre viene de Flach
    & Kull, ya citados: «the **baseline** to beat is the **always-positive
    classifier**». `classifier` no se toma, porque el LSTM no es un clasificador
    y sumaría un tercer término.
27. **`historical average` es el término estándar** del baseline en predicción
    de transporte: Rodrigues 2022, «the average weekly pattern for each
    location… commonly referred in literature as the “Historical Average” (HA)»,
    citado en el Apéndice A-B; Jiao 2023, «historical average models». **Pero el
    HA estándar agrupa por hora del día y día de la semana**, y el nuestro por
    posición del vector (`PROFILE_KEY` en `build_positional_null.py:88`:
    corredor, sentido, horizonte, `pair_rank`). Por eso `por posición` no se
    omite nunca: sin él, el revisor supone el agrupamiento por hora.
    `posicional` se retiró porque sonaba a posición geográfica (ver nota 5).

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
