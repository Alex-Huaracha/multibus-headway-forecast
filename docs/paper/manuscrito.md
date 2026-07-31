<!--
  ANDAMIAJE DEL MANUSCRITO — borrador en español.
  Flujo: se escribe y pule TODO en español → traducción al inglés al final (IJACSA es en inglés).
  Las notas entre bloques ">" con la etiqueta [ANDAMIAJE] son guías para escribir; BORRARLAS antes de traducir.
  Orden de escritura sugerido: Related Work → Métodos → Resultados (ya existen) → Discusión → Conclusión → Introducción → Abstract.

  REESTRUCTURADO 2026-07-29. El encuadre anterior ("la métrica decide el ganador":
  el MAE escalar y la fidelidad vectorial nombran ganadores opuestos) se RETIRÓ.
  Era falso: dependía por completo de un umbral mal transportado. Ver la nota de
  retractación en docs/resultados/documento-resultados.md §5 y el inventario de
  literatura en docs/paper/fuentes-verificadas.md.

  Insumos:
    docs/resultados/documento-resultados.md   — resultados y cifras
    docs/paper/fuentes-verificadas.md         — ~55 referencias con estado de verificación
-->

# Título

## El umbral, no el modelo: compresión de dispersión y recalibración del corte en la detección de *bunching* de buses a partir de datos GPS reales

> [ANDAMIAJE — ELEGIDO 2026-07-30] Nombra las tres cosas que el título debía
> nombrar: el objeto (*detección de bunching*), el ángulo (*el umbral no es
> transportable*, en la cabeza y sin adorno) y el dato real. **Es un cambio de una
> línea**, así que quedan registradas las dos alternativas anteriores por si se
> prefiere otro tono:
>
> - Más polémica: *"Un pronóstico aplanado parece ciego sin serlo: transportabilidad del umbral en la detección de bunching de buses a partir de datos GPS reales"*. Se descartó porque la afirmación va en la cabeza sin el mecanismo al lado, y un revisor puede leerla como reclamo antes de llegar a la evidencia.
> - Más conservadora: *"Compresión de dispersión y recalibración de umbrales en la detección de bunching a partir de pronósticos de headway"*. Se descartó porque nombra el mecanismo pero no la consecuencia, y en un corpus donde la norma es una fila en negrita ganando, no distingue.

**Autores** · Afiliación · Contacto

---

## Abstract

> [ANDAMIAJE — ESCRITO 2026-07-30] 235 palabras, un párrafo, dentro del rango
> 150–250. Sin citas y sin abreviaturas indefinidas: no aparecen las siglas del
> coeficiente de variación ni del área bajo la curva, y "correlación de Matthews"
> va desarrollada. Es el **único** lugar donde el lector ve cifras antes de la
> Sección IV — la Conclusión deliberadamente no las repite.
> **Al traducir, verificar el conteo de palabras otra vez:** el inglés suele
> comprimir y puede caer por debajo de 200.

El paradigma dominante para anticipar el *bunching* de buses predice el *headway*
con un regresor y compara la predicción contra un umbral definido sobre
observaciones. Está documentado que ese esquema rinde mal en detección aunque su
error de regresión se degrade suavemente, y la respuesta habitual ha sido cambiar
de clase de modelo. Este trabajo muestra que la causa es de unidades y no de
información. Sobre 152 días de datos de posicionamiento vehicular de tres
corredores de alta frecuencia de Arequipa, Perú, medimos la compresión de
dispersión del pronóstico como razón de coeficientes de variación del vector de
*headways*: el aprendiz describe un corredor con coeficiente 0.16 cuando el real
es 0.79, con sesgo negativo en las 36 celdas evaluadas. Un umbral relativo
calibrado sobre observaciones, trasplantado a ese vector comprimido, fabrica una
degradación aparente de hasta 253 veces. Al reajustar el corte por correlación de
Matthews sobre una ventana anterior y disjunta, la persistencia recupera el
umbral publicado en 11 de 12 celdas mientras el aprendiz requiere cortes más
laxos; puntuado sin umbral, el veredicto se invierte y el aprendiz discrimina
mejor a diez minutos en los tres corredores y las tres ventanas. Un detector sin
contenido supera al ganador declarado en 5 de 12 celdas, y bajo un corte absoluto
el colapso empeora en dos órdenes de magnitud. Un veredicto de detección obtenido
trasplantando un umbral mide el umbral, no el modelo.

**Keywords** — bus bunching detection; headway forecasting; forecast
under-dispersion; decision threshold calibration; event detection metrics;
evaluation methodology; AVL/GPS data

---

## I. Introducción

> [ANDAMIAJE] SE ESCRIBE CASI AL FINAL. Cuatro movimientos, en este orden.

### A. Contexto y motivación
> [ANDAMIAJE — ESCRITO 2026-07-30] Dos párrafos. El segundo cierra sobre el
> paradigma operativo, que es lo que la Sección I-B va a atacar.

En un corredor de buses de alta frecuencia, lo que determina la espera de un
pasajero no es la velocidad del vehículo sino el **intervalo entre vehículos
sucesivos** —el *headway*—. Cuando el servicio es lo bastante frecuente, nadie
consulta un horario: se llega a la parada y se espera, y esa espera depende
directamente de la regularidad de los intervalos. El *bunching* es el colapso de
esa regularidad: un vehículo se retrasa, recoge más pasajeros de los previstos
porque acumuló más tiempo de espera, se retrasa todavía más, y el que lo sigue
encuentra paradas vacías y lo alcanza. El resultado es un par de vehículos
viajando juntos seguido de un hueco largo, con una capacidad total sin cambios y
una espera media considerablemente peor.

Para el operador, la utilidad de detectar *bunching* depende por completo del
momento en que se detecta. Constatarlo cuando ya ocurrió sirve para el reporte
mensual, no para el servicio: las acciones disponibles —retener un vehículo en
parada, ajustar la velocidad de crucero, inyectar una unidad de reserva—
necesitan minutos de anticipación para tener efecto. Esa es la razón por la que
la línea de trabajo dominante no intenta clasificar el estado presente sino
**pronosticar el intervalo futuro**, y sobre ese pronóstico define la alarma. El
esquema operativo resultante tiene tres pasos: entrenar un modelo que prediga
*headways*, aplicar a la predicción una regla de umbral que declare el evento, y
despachar en función de la alarma. Este artículo se ocupa del segundo paso, que
es el que ha recibido menos escrutinio.

### B. El problema
> [ANDAMIAJE — ESCRITO 2026-07-30] Dos reglas vinculantes que se respetaron:
> **NO** decir "el subcampo no usa *baselines naive* ni tests de significancia"
> —es plomería y además es un encuadre débil, y va en I-D como contribución
> metodológica, no acá como queja—; y **NO** decir "nadie se dio cuenta", porque
> Sun et al. (2021) diagnosticaron el síntoma y están citados en las dos primeras
> oraciones.

Ese segundo paso arrastra un supuesto que rara vez se enuncia: que la regla de
umbral con la que se etiquetan las observaciones significa lo mismo aplicada a un
pronóstico. Sun, Schmöcker y Nakamura (2021) mostraron que el esquema falla —los
regresores rinden mal en detección aunque su error de regresión se degrade de
forma suave, y fallan justamente cuando el intervalo real se acorta— y
atribuyeron la causa a que un pronóstico determinista entrega **un único punto de
operación no ajustable**. Su remedio fue coherente con ese diagnóstico: cambiar
de clase de modelo, pasar a predicción probabilística y construir curvas ROC.

Lo que quedó sin examinar es el paso anterior a esa decisión: **por qué falla el
umbral, y si basta con recalibrarlo.** Un pronóstico puntual está sub-disperso,
porque minimizar una pérdida puntual apunta a un funcional central de la
distribución condicional y ningún funcional central tiene por qué reproducir la
dispersión. El vector predicho resulta entonces más parejo que el real, y un
corte calibrado sobre observaciones cae, sobre ese vector comprimido, mucho más
adentro de la cola de lo que caía sobre el original. Cuando además el corte es
**relativo al propio vector** —la forma que impone la ausencia de horario
programado— el efecto se duplica, porque el denominador se contrae junto con el
numerador. La alarma deja de sonar, y el diagnóstico inmediato, que el modelo no
ve el *bunching*, es falso.

Las consecuencias se reparten entre dos audiencias. Para quien evalúa: **un
veredicto de detección obtenido trasplantando un umbral entre espacios de
distinta dispersión mide el umbral y no el modelo**, de modo que la comparación
publicada puede ordenar los competidores al revés. Para quien despliega: una
línea de trabajo se cierra por un motivo que nada tiene que ver con lo que el
modelo sabe, cuando la reparación cuesta un escalar recalibrado sobre datos que
la operación ya tiene.

### C. Contribuciones
> [ANDAMIAJE] Rankeadas por defensibilidad, con la evidencia al lado. Cada una nombra explícitamente qué NO reclama, porque la mitad del argumento anterior estaba tomado y conviene que el revisor vea que lo sabemos.

- **C1 (mecanismo, medido).** Cuantificamos la causa de la falla: la **compresión de dispersión** del pronóstico, medida como razón de coeficientes de variación del vector de *headways*. El LSTM predice un corredor con CV ≈ 0.16 cuando el real es ≈ 0.79, con sesgo negativo en **las 36 celdas** de corredor × horizonte × ventana de prueba y deterioro monótono con el horizonte. Y la convertimos en profundidad de cola usando `Z = 0.5/cv`, **la propia relación que el TCQSM establece** para ligar el CV del *headway* a la probabilidad de desvío: sobre las observaciones da Z ≈ 0.63, sobre el pronóstico Z ≈ 3.1. La escala de nivel de servicio del manual califica al mismo corredor como **LOS A, "service provided like clockwork"** por el pronóstico y **LOS F, "most vehicles bunched"** por lo observado.
  *No reclamamos* que la sub-dispersión de pronósticos puntuales sea un hallazgo nuevo, y hay que decirlo de frente porque **está publicado como enunciado general**: Mayer y Yang (2022) escriben que *"MSE-optimized forecasts are always underdispersed"*, cuantifican el déficit como razón de varianza (< 75 % de la varianza observada), establecen que los pronósticos optimizados en MAE quedan **menos** sub-dispersos que los de MSE, y atribuyen el agravamiento con el horizonte a Vannitsem y Hagedorn (2011). Patton y Timmermann (2012, Cor. 2) prueban esa monotonía como teorema. Tampoco reclamamos haber sido los primeros en atar la compresión a una métrica categórica: Petetin et al. (2022) miden el déficit de variabilidad de pronósticos de ozono, lo aparean con métricas de umbral y con AUC, y documentan su agravamiento con el horizonte. Reclamamos tres cosas más angostas: (a) **medirlo en el vector de *headways***, como dispersión **transversal** entre componentes en un mismo instante —Mayer y Yang trabajan sobre la varianza **temporal de una serie escalar** de irradiancia, que es la cantidad que los teoremas acotan y **no** es la nuestra—; (b) **darle la vuelta a la fórmula del propio manual sobre el pronóstico**; y (c) atarlo a una regla de evento **relativa y auto-referencial**, donde la compresión mueve numerador y denominador a la vez. Esto último no está en ninguno de los dos precedentes, y en el caso de Petetin et al. no por omisión sino por construcción: sus umbrales son regulatorios (60 y 90 ppbv por normativa) y no admiten recalibración.

- **C2 (reparación por recalibración, no por cambio de modelo).** Donde la literatura responde a la falla cambiando de clase de modelo, mostramos que **el punto de operación alcanza**. Ajustando el corte por MCC sobre una ventana anterior y disjunta y aplicándolo hacia adelante, la **persistencia recupera 0.5× en 11 de las 12 celdas** —o sea, el umbral publicado *era* su óptimo— mientras que el aprendiz necesita **0.58×–0.91×**. Con el corte en sus propias unidades, o puntuando sin umbral, **el veredicto se invierte**: a h=10 el aprendiz discrimina mejor que la persistencia en **3 de 3 corredores y 3 de 3 ventanas**, exactamente donde el corte fijo le daba 253× en contra. Y a h=1 la persistencia gana en 3 de 3 y 3 de 3, así que el error escalar y la detección **coinciden** una vez removido el artefacto.

- **C3 (validez de la métrica).** El veredicto original descansaba en F1, y F1 es el resumen equivocado a estas tasas base (17–30 %, por encima del rango de 0.15–17 % que reporta el campo). Marcar **todas** las celdas —una regla sin contenido— supera al ganador declarado en **5 de las 12 celdas, incluidas las tres de h=10**. Reportamos MCC, ROC-AUC y **precisión media, que no aparece en la literatura de *bunching***, junto con el piso trivial explícito en cada tabla. La ausencia no es nuestra impresión: la propia tabla comparativa de Santos et al. (2022) lista las métricas de cada trabajo previo del subcampo —RMSE, *accuracy*, precisión, *recall*, MAPE, especificidad, sensibilidad, F-measure— y **ninguno reporta AUC ni precisión media**. Y medimos algo que nadie publicó: la **estabilidad comparada** del corte ajustado por F1 contra el ajustado por MCC entre ventanas. El resultado es mixto y se reporta mixto — el F1 es más ajustado en la mediana, pero tiene un modo de falla degenerado que el MCC no tiene, y el costo fuera de muestra favorece al MCC por un factor de 3 a 6.

- **C4 (el hallazgo no es artefacto de nuestra regla).** Nuestro umbral normaliza por la media del propio vector porque estos datos son GPS crudo sin horario — la misma sustitución que hace Yu et al. (2016), que usan el *headway* observado en la primera parada por el mismo motivo. La diferencia es el punto de referencia, y **es el mecanismo**: la suya es observada y fija, la nuestra es predicha y se mueve. Para probar que el colapso no depende de esa elección, repetimos toda la detección con un **corte absoluto en minutos**, calibrado fuera de muestra e idéntico para observado y pronóstico. **No se atenúa: empeora**, y bajo la convención dominante del campo (un cuarto del programado) es **110 veces peor** que bajo la nuestra. Nuestra elección resultó ser la conservadora. Rezazada et al. (2024) confirman además que en este campo *"no existe un único valor de umbral"*, con los publicados entre 20 s y un cuarto del programado.

> [ANDAMIAJE — ESCRITO 2026-07-30] El párrafo de "lo que NO afirma", que en este
> venue es diferenciador y no debilidad. Versión corta acá; el desarrollo con las
> cuatro delimitaciones está en V-B, y no debe duplicarse.

Conviene acotar el alcance desde el principio. Este trabajo **no** afirma que los
modelos evaluados estén listos para operar una alarma de despacho: la capacidad
de ordenamiento que medimos es información real y no equivale a un sistema
operativo, y falta la función de costo que la traduzca en decisiones. **No**
afirma que la sub-dispersión de los pronósticos puntuales sea un hallazgo nuevo;
es un enunciado publicado y, en su dependencia del horizonte, un teorema. **No**
afirma que el cruce por horizonte entre persistencia y aprendiz sea novedoso, que
es conocido en pronóstico de tráfico. Y **no** afirma que el resultado nulo sobre
las arquitecturas espaciales sea una contribución: está publicado y aquí se
reporta como confirmación. La Sección V-B delimita cada uno de estos puntos
frente a su antecedente.

### D. Aporte metodológico, declarado al frente
> [ANDAMIAJE — ESCRITO 2026-07-30] Deliberadamente al frente y no escondido en
> Métodos: en el relevamiento del venue (ocho artículos, Vol. 15–17), **0 de 8**
> reportan un test de significancia pareado, **1 de 8** declara un corte temporal
> real y **0 de 8** tienen declaración de disponibilidad de datos o código. Lo
> que en un venue más exigente sería higiene, acá es diferenciador — pero solo si
> se enuncia. **No** escribir la comparación con el venue en el artículo: es
> insumo de decisión editorial, no contenido publicable.

Una parte del aporte de este trabajo es de protocolo, y se declara aquí en lugar
de quedar sepultada en la Sección III porque condiciona la lectura de todos los
resultados. Las comparaciones entre modelos se hacen mediante **auditoría pareada
sobre muestras idénticas**: cuando dos modelos se puntúan sobre conjuntos de
filas distintos, el estadístico de Diebold-Mariano no queda sesgado sino
**indefinido**, dado que se construye sobre el diferencial de pérdida fila a
fila. Sobre esa base se reportan Diebold-Mariano con varianza agrupada por día de
servicio y corrección de muestra pequeña, y Wilcoxon pareado, siempre con el
tamaño de efecto por delante y el valor *p* como piso y no como veredicto.

El resto del protocolo apunta a que los resultados sean reproducibles y a que las
decisiones de preprocesamiento no filtren información del futuro. La
winsorización se calcula **exclusivamente sobre el conjunto de entrenamiento** y
se aplica a los tres cortes, en lugar de recalcularse en cada uno. Los terciles
de volatilidad se congelan sobre entrenamiento y validación y se aplican a prueba,
nunca se calibran sobre prueba. Cada insumo de entrenamiento está fijado por su
*hash* SHA-256 y el procedimiento falla de forma cerrada antes de entrenar si los
bytes no coinciden. La calibración del umbral se ajusta sobre una ventana
anterior y disjunta y se aplica hacia adelante, que es la única dirección
disponible para un operador. Y los veredictos se validan sobre **tres orígenes de
*rolling*** con ventanas de prueba disjuntas, declarando que los conjuntos de
entrenamiento correspondientes están anidados y no son independientes. El código
de análisis y los residuos por muestra que sostienen cada tabla se publican junto
al artículo.

### E. Estructura del artículo
> [ANDAMIAJE — ESCRITO 2026-07-30] Un párrafo. Al traducir, verificar que las
> letras de subsección sigan coincidiendo si el maquetado fusiona secciones.

El resto del artículo se organiza como sigue. La Sección II revisa la familia de
trabajos que predice el *headway* y después umbraliza, la teoría de la
sub-dispersión de los pronósticos puntuales, los precedentes de recalibración de
umbrales fuera del transporte y los criterios para elegir una métrica de
detección, y cierra delimitando el vacío. La Sección III describe los datos AVL,
el preprocesamiento, los modelos comparados, la definición del evento de
*bunching* y el protocolo de evaluación. La Sección IV presenta los resultados,
empezando por el contexto escalar y siguiendo con la compresión de dispersión
medida, el artefacto de umbral, su reparación, los análisis de robustez y la
comparación entre criterios de calibración. La Sección V discute la
interpretación operativa, delimita el aporte frente a sus antecedentes, sitúa el
resultado nulo espacial y enumera las amenazas a la validez. La Sección VI
concluye y ordena el trabajo futuro.

---

## II. Trabajos Relacionados (Related Work)

> [ANDAMIAJE — REESTRUCTURADO 2026-07-29] La versión anterior organizaba esta sección alrededor de un *gap* que ya no es el nuestro ("el subcampo reporta ganancias del DL sin *baseline naive* ni significancia"). Las FUENTES se conservan casi todas; el ARGUMENTO se rehace. El orden nuevo va de lo más cercano a lo más general, para que el lector vea primero a quién le estamos hablando.
>
> **Regla de citación para esta sección:** cada referencia debe llevar identificador verificado. `docs/paper/fuentes-verificadas.md` tiene el estado entrada por entrada y una lista explícita de lo que NO se puede citar porque no se pudo recuperar. **No citar afirmaciones específicas de fuentes marcadas `[SNIPPET]` o `[ABSTRACT]` sin leerlas primero.**

### A. Predecir el *headway* y después umbralizar: la familia y su falla documentada
> [ANDAMIAJE — ESCRITO 2026-07-30] Citación en autor-año; la conversión a `[n]`
> se hace al maquetar. Toda cita textual de esta subsección proviene de fuentes
> marcadas `[TEXTO COMPLETO]` en `docs/paper/fuentes-verificadas.md`. Dos
> restricciones que se respetaron: (a) a Moreira-Matias et al. (2016) se lo cita
> **solo** por la convención de un cuarto del *headway* programado, que es lo
> verificado — no por el mecanismo de alarma; (b) el >95 % de Yu et al. no se
> cita como tal: se cita el par de cifras reconciliado vía Sun et al. (2021).
> Pendiente que toca esta subsección: Jiao, Shen y Zhang (2023) reclama 89 % con
> el mismo esquema y sigue en `[ABSTRACT]` con umbral no verificado. **No se cita
> hasta leerlo**; si se confirma que su horizonte es corto, entra en el párrafo
> de la reconciliación junto a Yu et al.

La vía dominante para anticipar el *bunching* no lo predice: predice el
*headway* y después lo compara contra un umbral. Yu et al. (2016) fijan la
formulación de forma explícita —*"the occurrence of bus bunching can be detected
by thresholding the predicted headway with the planned bus schedule"*— y la
instancian sobre datos AVL de Pekín con máquinas de vectores de soporte y sus
variantes. El esquema se repite con pocas variaciones: un regresor entrega un
valor de *headway*, una regla binaria lo convierte en alarma, y las métricas que
se reportan son las de la alarma.

El umbral de esa regla es, casi siempre, una fracción del *headway* programado.
Un cuarto del programado es la convención más difundida (Yu et al. 2016;
Moreira-Matias et al. 2016; Santos et al. 2022), aunque no la única: la revisión
de Rezazada et al. (2024) advierte que *"there is no single threshold value to
define bunching events, as it depends on the type of the service, time of the
day, location, and service frequency"*, y sitúa los valores publicados entre 20 s
y un cuarto del programado. La elección no es inocua, y la literatura lo sabe:
Santos et al. (2022) declaran que cuando el GTFS no permite calcular el *headway*
programado recurren a un corte definido por el usuario, y usan cinco minutos
absolutos.

Ese hueco —qué se hace cuando no hay horario— es más común de lo que sugiere la
convención, y la solución establecida es sustituir el horario ausente por una
referencia **observada dentro del propio dato**. Yu et al. (2016) lo hacen de
frente: como en Pekín *"most route schedules change over time"* y el operador no
publica una tabla fija, *"this study uses the headway at the first stop as the
'scheduled' headway"*, de modo que su regla marca *bunching* cuando el *headway*
en la parada i cae por debajo de un cuarto del *headway* observado en la primera
parada de la misma corrida. La Sección III-D retoma este punto, porque nuestra
propia regla es una sustitución de la misma clase con otro punto de referencia, y
la diferencia entre ambas referencias resulta ser el mecanismo que este trabajo
mide.

El trabajo al que este artículo responde es Sun, Schmöcker y Nakamura (2021).
Describen la familia con precisión —*"for bunching prediction then an additional
step is required judging whether the predicted headway is below a prior defined
bunching threshold or not"*— y muestran que rinde mal en detección aunque su
error de regresión no lo anticipe: a diez paradas de anticipación la sensibilidad
cae al rango de 34.9–52.7 % mientras, en sus palabras, *"in terms of MAPE and
RMSE… evaluation metrics deteriorate gradually"*. Observan además dónde falla el
regresor, y es exactamente donde importa: *"neither in 1- nor 10-stop-ahead
prediction can these two methods perform favorably under the circumstance that
the actual headway becomes extremely short and bunching is going to happen"*. El
diagnóstico que proponen es el determinismo del pronóstico: como *"headway
prediction produces an exact value for each headway"*, del regresor *"only one
combination of sensitivity and specificity is derived"*. Su remedio es coherente
con ese diagnóstico y consiste en **cambiar de clase de modelo** —pasar a
clasificación probabilística y construir curvas ROC—, con un aparato de
evaluación notablemente más cuidado que el del resto del subcampo: AUC,
corrección de King–Zeng para eventos raros, elección de corte ponderada por costo
y matrices de confusión completas. El giro probabilístico no es aislado: Chen et
al. (2022) construyen un pronóstico bayesiano de tiempo de viaje que concatena
los vectores de tiempo de recorrido por tramo **y el vector de *headways*** de un
par de buses adyacentes en una variable aumentada, modelada como mezcla gaussiana
multivariada, y Yu, Wu, Chen y Ma (2017) predicen el *headway* con máquinas de
vectores de relevancia.

Lo que queda abierto es el paso anterior a esa decisión. Sun et al. aplican a los
pronósticos de sus regresores el mismo corte absoluto de un minuto derivado de
las observaciones, sin recalibrarlo, en los quince horizontes; no reportan
ninguna métrica libre de umbral para los detectores basados en regresión —las
curvas ROC son del clasificador—; no comparan contra un detector trivial; y
atribuyen el déficit al determinismo y no a la compresión de dispersión, que es
visible en sus propias tablas, donde el R² ajustado del modelo de *headway* cae de
0.968 a 0.635 mientras el coeficiente del *headway* rezagado se mantiene en
torno a 1.00. Su propio trabajo futuro propone calcular probabilidades de
excedencia; hasta donde alcanza este relevamiento, nadie lo escribió.

La misma ausencia, en una forma más aguda, aparece en el vecino arquitectónico
más cercano a este trabajo. Usama y Koutsopoulos (2025) pronostican el campo
espacio-temporal completo de *headways* de una línea de metro con ConvLSTM —el
mismo objeto vectorial que modelamos aquí— y reportan únicamente MAE, MSE y
RMSE. En su texto no hay análisis de dispersión, ni detección de eventos, ni
umbral, ni *baseline* de persistencia, ni ablación arquitectónica. Esa lista de
ausencias es, casi renglón por renglón, la lista de contribuciones de este
artículo.

### B. Por qué el umbral se mueve: sub-dispersión de los pronósticos puntuales
> [ANDAMIAJE — ESCRITO 2026-07-30] Restricciones respetadas al redactar:
> (a) Gneiting (2011) se cita **solo** por el funcional que elicita cada pérdida
> — el preprint completo no contiene ninguna afirmación de sub-dispersión;
> (b) Vannitsem y Hagedorn (2011) entra **acreditado a través de** Mayer y Yang,
> porque no lo leímos: no se le atribuye texto ni cifras;
> (c) Wernli et al. (2009), von Storch (1999), Huth (2002) y Maraun (2013) están
> en `[CROSSREF]`/`[SNIPPET]`, así que aparecen parafraseados y **sin comillas**;
> (d) el párrafo de la distinción transversal/temporal es obligatorio y no se
> puede ablandar: es lo que separa nuestro resultado del teorema.

Que el umbral se corra no es un accidente del entrenamiento, sino una
consecuencia de qué cantidad estima un pronóstico puntual. Gneiting (2011)
caracteriza esa correspondencia: cada función de pérdida elicita un funcional
determinado de la distribución condicional —las pérdidas de Bregman, la media;
las lineales por tramos, un cuantil, y su caso simétrico la mediana—. Un
regresor entrenado con error cuadrático apunta entonces a la media condicional y
uno entrenado con error absoluto a la mediana condicional. Ninguno de los dos
apunta a la dispersión, y ninguno tiene incentivo para reproducirla.

La consecuencia sobre la varianza es cuantificable. Patton y Timmermann (2012)
establecen que para un pronóstico óptimo se cumple `V[Y] = V[Ŷ*] + E[e*²]`, de
modo que la varianza del pronóstico está acotada por la de la observación, y su
Corolario 2 hace de la monotonía en el horizonte un teorema:
`V[Ŷ*_{t|t−hS}] ≥ V[Ŷ*_{t|t−hL}]` para `hS < hL`. El deterioro que reportamos en
la Sección IV-B es esa cota volviéndose visible en datos de buses, no un
descubrimiento sobre la cota.

Conviene ser explícito sobre el alcance de esos resultados, porque la diferencia
decide qué parte de nuestro hallazgo es empírica. Los teoremas acotan la varianza
**temporal de una serie escalar**: la dispersión de un mismo pronóstico a lo
largo del tiempo. La cantidad que este trabajo mide es otra — la dispersión
**transversal entre las componentes del vector de *headways* en un mismo
instante**, que es la que gobierna una regla de evento definida sobre el vector.
La descomposición de la varianza se aplica componente a componente, de manera que
la dirección del efecto es la esperable; pero no implica el resultado
transversal, y en particular no dice nada sobre el coeficiente de variación del
corte transversal. Que el sesgo aparezca en la totalidad de las celdas evaluadas
es, por lo tanto, un resultado empírico y se reporta como tal.

Como enunciado general, la sub-dispersión de los pronósticos puntuales está
publicada, y este artículo no la reclama. Mayer y Yang (2022) la formulan sin
matices —*"as MSE-optimized forecasts are always underdispersed, the common
practice of using RMSE skill score for evaluation overrates the forecasts with
lower dispersion"*—, la cuantifican como razón de varianzas sobre pronósticos de
irradiancia solar, donde los pronósticos optimizados en MSE *"only capture <75 %
of the observed variance"*, y establecen la dependencia de la pérdida que
podríamos haber creído propia: los pronósticos optimizados en MAE quedan también
sub-dispersos, pero menos, con menor sesgo condicional de tipo 2 y mayor
discriminación que los de MSE. El agravamiento con el horizonte lo acreditan a
Vannitsem y Hagedorn (2011), que constituye así un segundo precedente del
comportamiento monótono, junto al teorema de Patton y Timmermann. Nuestra
posición frente a ese trabajo es la que corresponde: lo citamos como enunciado
previo, y delimitamos lo que agregamos. Su objeto es una serie escalar de
irradiancia —varianza temporal, la cantidad que los teoremas acotan— y su
evaluación es de calidad de pronóstico: en su texto no aparecen umbrales,
detección de eventos ni excedencias. El puente entre la compresión de dispersión
y una **regla de evento** es lo que queda sin construir, y es lo que este
artículo construye.

Fuera del pronóstico solar, el mismo fenómeno aparece asociado justamente al tipo
de evento que nos ocupa. Ravuri et al. (2021) reportan que una pérdida puntual
produce *nowcasts* de precipitación borrosos a mayor horizonte, con desempeño
pobre sobre los eventos de lluvia media a intensa —es decir, los eventos que se
definen por un umbral—; su explicación es la falta de restricciones sobre la
salida, no la elicitabilidad. Subich et al. (2025) atribuyen el suavizado a la
doble penalización del MSE y lo corrigen cambiando la función de pérdida.
Bonavita (2024) va un paso más lejos y observa que la ventaja de los modelos de
aprendizaje automático en métricas deterministas es parcialmente atribuible al
suavizado, o sea que la métrica premia el defecto.

El fenómeno tiene nombre propio en al menos dos literaturas más. En meteorología
se lo discute como doble penalización (Ebert 2008; Wernli, Hofmann y Zimmer
2009): un campo pronosticado con estructura correcta pero desplazada es
penalizado dos veces, y suavizar lo evita. En *downscaling* climático se lo
discute como inflación de varianza, y sin acuerdo — von Storch (1999) argumenta
en contra de inflar y a favor de aleatorizar, Huth (2002) llega a la conclusión
opuesta, y Maraun (2013) reabre el punto más de una década después. Que el debate
siga abierto es relevante para este trabajo y conviene decirlo: **cómo** corregir
la sub-dispersión no está resuelto, y por eso una contribución que no la corrige
sino que ajusta la regla de decisión aguas abajo tiene lugar legítimo.

Queda por distinguir esta línea de un diagnóstico vecino con el que es fácil
confundirla. La *over-stationarization* (Liu et al. 2022), llevada por Li, Yang y
Wang (2025) a la predicción de arribos de buses, describe un síntoma muy
parecido: en sus palabras, *"the model tends to produce overly stable and
indistinguishable outputs"*. Pero el agente causal que identifican es la
**normalización del preprocesamiento**, y por eso su propuesta es arquitectónica
— la curabilidad por diseño es la premisa de su propio método. El agente que
identificamos aquí es la **pérdida**, y en esa medida es estructural: sobrevive a
cualquier arquitectura entrenada con el mismo objetivo, que es exactamente lo que
observamos al comparar tres arquitecturas distintas. Son afirmaciones opuestas
sobre tratabilidad, y ninguna de las dos fuentes cuantifica el aplanamiento ni lo
atribuye a la función de pérdida.

### C. Recalibrar el umbral: precedentes fuera del transporte
> [ANDAMIAJE — ESCRITO 2026-07-30] Subsección obligatoria: sin ella, un revisor
> de clima o meteorología nos liquida con una cita de 2018. Restricciones que se
> respetaron: (a) a Hoffmann et al. (2018) se lo parafrasea, porque tenemos el
> texto completo pero no dejamos registrada ninguna cita textual; (b) Petetin et
> al. (2022) es la cita central y se le acredita **todo** lo que tiene, incluido
> lo que creíamos nuestro — ver §0.5 de `fuentes-verificadas.md`; (c) Alfieri et
> al. (2019), Zsoter et al. (2020) y Lalaurette (2003) quedaron **fuera**: siguen
> en `[POR VERIFICAR ANTES DE CITAR]` y Petetin ya cubre su papel. Si se
> consiguen, entran como refuerzo, no cambian el argumento.

Que un umbral calibrado sobre observaciones deje de funcionar cuando se lo aplica
a un pronóstico no es un descubrimiento de este trabajo, ni siquiera un problema
del transporte. Está documentado en clima y en calidad del aire, y allí ya se
ensayaron dos respuestas distintas: **corregir el pronóstico** para que su
distribución se parezca a la observada, o **mover el umbral** hacia la
distribución del pronóstico. Este artículo transfiere la segunda al transporte, y
conviene decir con precisión qué se toma prestado y qué no.

La primera familia es la más transitada. Mayer y Yang (2022) proponen calibrar
explícitamente la razón de varianzas del pronóstico solar; Petetin et al. (2022)
aplican *quantile mapping*, cuyo objetivo declarado es *"adjusting the
distribution of the forecast concentrations to the distribution of observed
concentrations"*; Subich et al. (2025) cambian directamente la función de pérdida.
Todas comparten un requisito operativo que no siempre está disponible: para
corregir la distribución del pronóstico hace falta una distribución de referencia
de observaciones —o un reentrenamiento—, y en el caso del *quantile mapping* hace
falta además que esa referencia siga siendo válida cuando el sistema se despliega.

La segunda familia es la que este trabajo continúa, y su precedente más limpio es
Hoffmann, Menz y Spekat (2018). Frente al mismo problema en *downscaling*
climático, identifican a qué percentil de los datos de referencia corresponde el
umbral de interés y **lo recalculan modelo por modelo**, precisamente porque
trasplantar un corte fijo del espacio de las observaciones subestima de forma
severa el conteo de eventos. Es el mismo mecanismo y la misma clase de reparación
que aplicamos aquí, ocho años antes y en otro dominio. Lo que su marco no
contiene es un horizonte de pronóstico, métricas de detección, ni un contexto de
transporte.

El vecino más cercano en el tiempo y en la forma del argumento es Petetin et al.
(2022), y hay que acreditarles sin regateo bastante más de lo que resulta cómodo.
Sobre pronósticos de ozono del servicio Copernicus, documentan que *"there is a
clear trade-off between the continuous and categorical skill scores"* y que *"the
quality of a MOS-corrected forecast assessed solely based on metrics like RMSE or
PCC thus tells little about the forecast value"*, con un caso explícito en el que
un método *"can give the best RMSE and PCC, yet the poorest high O₃ detection
skills"*. Diagnostican y cuantifican la sub-dispersión —el pronóstico crudo
subestima la variabilidad en torno al 30 % y los métodos más sofisticados quedan
demasiado suaves, con dificultad para capturar los valores extremos—, reportan
AUC junto a las métricas de umbral, y muestran que *"all categorical metrics show
a similarly strong sensitivity to the lead time"*. La disociación entre error
continuo y desempeño categórico, la compresión de dispersión que la explica y su
agravamiento con el horizonte están, las tres, publicadas allí.

Lo que no está allí es recalibrar el umbral, y la razón es estructural más que
una omisión: **sus cortes son regulatorios**. Sesenta partes por mil millones
para el máximo diario de ocho horas y noventa para el máximo horario son valores
fijados por normativa, y un umbral legal no se reajusta contra la distribución de
un modelo — dentro de su marco, la pregunta no puede formularse. Lo mismo vale,
por motivos distintos, para el resto de los precedentes: todos operan sobre
cortes **absolutos**, sean regulatorios o convencionales.

Ahí queda delimitado el caso de este artículo, y la diferencia no es de grado.
Bajo un corte absoluto, la compresión de dispersión mueve un solo lado de la
comparación: el valor pronosticado se acerca a su centro mientras el corte
permanece donde estaba. Bajo un corte **relativo y auto-referencial** —una
fracción de la media del propio vector pronosticado, que es la forma que impone
la ausencia de horario— se mueven los dos lados a la vez, porque el denominador
es una función del mismo pronóstico comprimido. El resultado deja de estar
gobernado por el nivel del pronóstico y pasa a estarlo por su **coeficiente de
variación**, que es exactamente la cantidad que la Sección IV-B mide y que
ninguno de estos precedentes necesitó considerar.

### D. Qué métrica puede decidir una detección
> [ANDAMIAJE — ESCRITO 2026-07-30] Restricciones respetadas al redactar:
> (a) **2π/(1+π) se presenta como derivación en un paso**, no como cita — la
> fórmula no está impresa en ninguna fuente, y Lipton et al. solo la instancian
> numéricamente; (b) la **invariancia del AUC ante transformaciones monótonas**
> se deriva de la identidad de rangos, tampoco se atribuye: no figura como
> teorema etiquetado en ninguna de las fuentes; (c) para la regla siempre-activa
> el MCC es **0/0, indeterminado**, y el cero es extensión por continuidad —
> **nunca escribir "cero por construcción"**; (d) Zhu (2020) está en `[ABSTRACT]`
> y solo se le atribuye lo que dice el resumen; (e) el DOI de proceedings de
> McDermott et al. no está verificado: citar por NeurIPS 2024 / arXiv.
> **Pendiente que NO bloquea:** Boughorbel et al. (2017) está en `[CROSSREF]` con
> la instrucción de leer el cuerpo antes de citar, así que el argumento de que
> ajustar un umbral por métrica es principiado se apoya en Koyejo et al. (2014).
> Si se lee Boughorbel, entra como refuerzo. Ídem Itaya et al. (2025), que da
> intervalos de confianza para **diferencias pareadas** de MCC y encaja mejor en
> la Sección III-E que acá.

La familia de trabajos descrita en la subsección A decide sus veredictos con
*accuracy*, precisión, *recall* y F-measure. Esa elección no es neutral, y deja
de serlo especialmente en el régimen de tasa base de este trabajo: entre 17 % y
30 % de celdas marcadas, por encima del rango de 0.15 % a 17 % que reporta el
subcampo. Conviene entonces establecer qué puede y qué no puede decidir cada
resumen antes de usarlos.

El primer problema del F-measure es lo que omite. Powers (2011) señala que esta
familia de medidas ignora el desempeño sobre los ejemplos negativos y no descuenta
el nivel de azar, de modo que *"a system that performs worse in the objective
sense of Informedness, can appear to perform better under any of these commonly
used measures"*. Chicco y Jurman (2020) lo hacen explícito para el caso binario:
el F1 es independiente de los verdaderos negativos y no es simétrico al
intercambiar las etiquetas de clase, mientras que el coeficiente de correlación de
Matthews (MCC) sí lo es. Y Fawcett (2006) agrega la consecuencia práctica: el
F-score se mueve con la distribución de clases aunque el clasificador no cambie,
por lo que comparar desempeños a un umbral común entre escalas distintas *"will
be meaningless"*.

El segundo problema es que el F1 no tiene un cero informativo. Flach y Kull
(2015) fijan la referencia correcta: *"the baseline to beat is the always-positive
classifier rather than any random classifier. This baseline has prec = π and
rec = 1"*. De ahí se sigue en un paso que el F1 de la regla que marca todas las
instancias vale `2π/(1+π)` —una derivación inmediata, no un resultado citable—, lo
que a nuestras tasas base produce un piso de entre 0.29 y 0.46. Un F1 de 0.40 no
significa nada hasta saber de qué lado de ese piso cae, y esa comparación no
aparece reportada en la literatura de *bunching*.

El tercer problema es el que convierte lo anterior en un modo de falla concreto
al calibrar. Lipton, Elkan y Naryanaswamy (2014) demuestran que *"given an
uninformative classifier, optimal thresholding to maximize F1 predicts all
instances positive regardless of the base rate"*, y su Teorema 1 sitúa el umbral
óptimo en la mitad del F1 máximo alcanzable. Sobre un modelo débil con tasa base
del 30 %, ese máximo está cerca del piso trivial, de modo que el corte óptimo cae
por debajo de casi toda predicción y la regla degenera en marcar todo. Los mismos
autores observan además que la selección de umbral por F1 es de alta varianza,
con cortes que *"may be set erroneously"*. La degeneración que reportamos en la
Sección IV-F no es entonces una peculiaridad de nuestros datos: es el
comportamiento que este teorema predice.

El MCC evita las tres cosas —descuenta el azar, usa las cuatro celdas de la
matriz y es simétrico ante el intercambio de clases— pero exige una precisión que
se omite con frecuencia. Chicco, Tötsch y Jurman (2021) señalan que el
coeficiente *"is undefined whenever the confusion matrix has a whole row or a
whole column filled with zeros"*, aunque *"by simple mathematical considerations
it is possible to cover such cases"*. Para la regla siempre-activa —fila de
negativos predichos vacía— el cociente es literalmente indeterminado, y el valor
cero que se le asigna es una extensión por continuidad y una convención, no una
identidad. Este trabajo la adopta y lo declara. Que ajustar un umbral maximizando
una métrica sea un procedimiento principiado y no una conveniencia tiene también
respaldo: Koyejo et al. (2014) establecen que para las métricas expresables como
razones de combinaciones lineales de la matriz de confusión, el clasificador
óptimo es precisamente un umbral que depende de la métrica elegida.

El cuadro sobre el MCC no es unánime y no tiene sentido presentarlo como si lo
fuera. Luque et al. (2019) muestran que no es completamente invariante a la
prevalencia; Zhu (2020) sostiene que su comportamiento se deteriora sobre
conjuntos desbalanceados; y Chicco y Jurman (2023) llegan a argumentar que el MCC
debería **reemplazar** al ROC-AUC. La respuesta que adopta este trabajo es que
las dos medidas responden preguntas distintas y por eso se reportan juntas: el
AUC y la precisión media miden **discriminación sin comprometerse con un punto de
operación**, mientras que el MCC resume la calidad de **un punto de operación
elegido**. Un artículo cuyo objeto es precisamente la transportabilidad del punto
de operación no puede permitirse reportar solo uno de los dos.

Queda una objeción previsible que conviene desactivar antes de que llegue: que a
clases desbalanceadas corresponde la curva de precisión-*recall* y no la ROC. La
objeción existe, pero su respaldo no cubre este régimen. El brazo desbalanceado
de Saito y Rehmsmeier (2015) es de 1:10, es decir una prevalencia cercana al 9 %,
más extrema que la nuestra; Davis y Goadrich (2006) hablan de *"large skew"*, y
su Teorema 3.2 es una **equivalencia** —dominancia en ROC si y solo si dominancia
en PR—, de modo que el AUC no puede invertir una conclusión de dominancia.
McDermott et al. (2024) van más lejos y refutan la premisa de frente: *"AUPRC is
not generally superior in cases of class imbalance"*, y observan que la creencia
contraria *"is often made without citation, misattributed to papers that do not
argue this point"*. Boyd et al. (2012) terminan de dar vuelta el argumento: el
área bajo la curva PR tiene un piso libre `AUCPR_MIN = 1 + (1−π)·ln(1−π)/π`, que
**crece** con la prevalencia y vale 0.168 al 30 % de tasa base. La alternativa
exclusivamente PR padece, en nuestro régimen, la misma enfermedad que le
reprochamos al F1, y peor que en el régimen de eventos raros para el que se la
recomienda. Seguimos su recomendación explícita y graficamos la curva PR mínima
junto a cada curva empírica.

Esta discusión no es abstracta: Li (2024) documenta un caso publicado en el que un
modelo sin capacidad predictiva real alcanza F1 = 0.475 mientras su MCC es 0.042 y
su AUC 0.524. Es exactamente el fallo que las tablas de la subsección A no pueden
detectar. Y la ausencia no es una impresión nuestra: la tabla comparativa con la
que Santos et al. (2022) resumen la literatura previa del subcampo enumera las
métricas de cada trabajo —RMSE, *accuracy*, precisión, *recall*, MAPE,
especificidad, sensibilidad y F-measure— y ninguno reporta AUC ni precisión
media. Tampoco, hasta donde alcanza este relevamiento, ningún trabajo del
subcampo publica el piso del detector trivial junto a sus resultados.

### E. Síntesis del vacío
> [ANDAMIAJE — ESCRITO 2026-07-30] **La formulación del andamiaje anterior se
> quedó corta y hubo que angostarla.** Decía "nadie mide la compresión de
> dispersión que causa la falla", y eso **ya no es cierto fuera del transporte**:
> Petetin et al. (2022) la miden, la cuantifican y la aparean con métricas
> categóricas. El vacío hay que enunciarlo con dos ejes —dominio y tipo de
> umbral—, no con uno. Dos reglas que siguen vigentes: **NO escribir "el
> subcampo no se dio cuenta"** (no sobrevive a un revisor que conozca a Sun et
> al.), y no ensanchar el reclamo más allá del caso auto-referencial.

El estado de la cuestión se ordena en dos ejes, y el aporte de este trabajo queda
donde los dos se cruzan.

Por el eje del **dominio**, el síntoma está diagnosticado. Sun, Schmöcker y
Nakamura (2021) mostraron que el paradigma de predecir el *headway* y umbralizar
falla en detección aunque el error de regresión no lo anuncie, y lo atribuyeron
al punto de operación único del pronóstico determinista. Lo que no se hizo dentro
del transporte es medir la causa ni ensayar la reparación más barata: la
respuesta ha sido invariablemente **cambiar de clase de modelo** —hacia
clasificación, hacia predicción probabilística— y no reajustar el punto de
operación de los modelos que ya se tienen.

Por el eje del **mecanismo**, en cambio, hay bastante más publicado de lo que
sugeriría la literatura de transporte, y este artículo no lo reclama. La
sub-dispersión de los pronósticos puntuales es un enunciado general (Mayer y Yang
2022) con respaldo teórico (Patton y Timmermann 2012). Su daño sobre eventos
definidos por umbral está documentado (Ravuri et al. 2021). Y en calidad del aire
está medida, cuantificada y apareada con métricas categóricas y con AUC, junto
con su agravamiento con el horizonte (Petetin et al. 2022). Recalibrar el corte
contra la distribución del propio modelo también tiene precedente, en clima
(Hoffmann et al. 2018).

Lo que no existe es la intersección, y admite una formulación precisa. **Ningún
trabajo mide la compresión de dispersión sobre un vector de *headways* como
causa de una falla de detección de *bunching*; ninguno recalibra el umbral de
*bunching* contra la distribución del propio pronóstico; y ninguno —dentro ni
fuera del transporte— trata el caso del umbral relativo y auto-referencial**, que
es el que impone la ausencia de horario programado y en el que la compresión
mueve numerador y denominador a la vez, de modo que el resultado queda gobernado
por el coeficiente de variación y no por el nivel. A eso se suma una ausencia
metodológica acotada al subcampo, respaldada por la propia tabla comparativa de
Santos et al. (2022): en detección de *bunching* no se reporta ROC-AUC ni
precisión media, ni se publica el piso del detector trivial contra el que
cualquier F1 debería leerse.

### Referencias de la Sección II
> [ANDAMIAJE] Tomarlas de `docs/paper/fuentes-verificadas.md`, respetando el estado de verificación. Presupuesto observado en IJACSA: 26–46, mediana ≈41.

---

## III. Materiales y Métodos

> [ANDAMIAJE] Casi todo redactado en documento-resultados.md §2 y en el código (`src/evaluation/`). Reordenar y completar el preprocesamiento que el doc de resultados omite a propósito.

### A. Datos: SIT Arequipa (AVL/GPS)
> [ANDAMIAJE — TRASLADADO 2026-07-30] Fuente: `documento-resultados.md` §2 y
> `docs/dataset-manifest.md`.

Los datos provienen del sistema de localización automática de vehículos (AVL) del
Sistema Integrado de Transporte de Arequipa, Perú, y consisten en trazas GPS
crudas de la flota en operación. El período cubre **152 días**, del 1 de octubre
de 2023 al 29 de febrero de 2024. Se estudian tres corredores de alta frecuencia,
identificados aquí como E2, E59 y E4. El más pequeño, E4, opera con **19
vehículos** y aporta validez externa acotada a la escala de flota: es otra línea,
no otra ciudad.

Cada vehículo se identifica por la **clave compuesta** de empresa y unidad. Esto
no es un detalle de implementación: los identificadores de unidad se reutilizan
entre empresas —una fracción sustantiva de ellos aparece en tres o más—, de modo
que indexar solo por unidad mezcla vehículos de operadores distintos y corrompe
el cálculo de intervalos.

Un rasgo de esta fuente condiciona todo lo que sigue y conviene declararlo aquí:
**no existe horario programado**. No hay GTFS ni tabla de servicio publicada
contra la cual medir desvíos. De ahí se deriva la definición de evento de la
subsección D, que es donde la metodología de este trabajo se aparta de la
convención del campo.

### B. Preprocesamiento, población de muestras y definición del *headway*
> [ANDAMIAJE — TRASLADADO 2026-07-30] Fuente: `documento-resultados.md` §2 y
> `docs/decisiones-headway-fase2.md` §2.1.
> **Los tres contratos se renombraron a propósito.** En el documento de
> resultados se llaman C1/C2/C3, y acá eso chocaría con las contribuciones C1–C4
> de la Sección I. Van por nombre y no por número.
> Y al describir el *headway*: usar **la ecuación**, no los nombres de columna del
> parquet — `front`/`back` están invertidos respecto del movimiento físico y la
> aritmética es correcta, pero los nombres confunden a quien lea el código.

El *headway* de un vehículo se define como el tiempo transcurrido desde que el
vehículo precedente ocupó la posición en la que el vehículo actual se encuentra
ahora, interpolado sobre la trayectoria del precedente. Es una definición de
cruce por posición y no por parada, lo que la hace robusta a la ausencia de una
tabla de paradas confiable.

Los datos se dividen **temporalmente**, sin mezcla aleatoria: entrenamiento hasta
el 15 de enero de 2024 (107 días), validación hasta el 7 de febrero (23 días) y
prueba del 8 al 29 de febrero (**22 días**). Todos los resultados reportados
corresponden al conjunto de prueba. La winsorización de los *headways* se aplica
en el percentil 99, y el contrato es explícito: **el umbral se calcula
exclusivamente sobre el conjunto de entrenamiento y se aplica a los tres cortes**,
en lugar de recalcularse dentro de cada uno, lo que filtraría información de las
particiones futuras. El recorte afecta entre el 0.78 % y el 1.11 % de los
objetivos, y la subsección IV-E documenta su efecto nulo sobre los veredictos.

La población de muestras se construye bajo tres contratos, verificados en cada
corrida:

| Contrato | Qué garantiza | Cómo se verifica |
|---|---|---|
| **Identidad de muestra** | Una muestra por (empresa, sentido, instante de inicio, horizonte). Todos los modelos puntúan exactamente las mismas celdas. | Cada ejecución recomputa el índice y compara su SHA-256 contra un manifiesto congelado. |
| **Contigüidad temporal** | Los `12 + h` instantes de una ventana son minutos **consecutivos**, de modo que el horizonte es tiempo y no posición de fila. | Se verifica al materializar; una violación aborta la corrida. |
| **Frontera de información** | Ninguna variable de entrada usa información posterior al instante de predicción. | El portón de entrada falla de forma cerrada si una variable prohibida reaparece. |

Exigir contigüidad tiene un costo medido: sobreviven entre el **81.9 %** y el
**90.2 %** de las ventanas candidatas. Se pierde menos de una quinta parte de los
datos y se gana que el horizonte signifique lo que dice.

El contrato de identidad de muestra se verifica de forma directa: cada modelo se
puntúa dos veces, sobre sus propias filas y sobre la intersección de todos, y si
el contrato se cumple restringir a la intersección no debe mover nada. El sesgo
de encuadre medido es de **0.001 minutos como máximo** sobre 36 filas. Bajo un
pipeline anterior que no imponía estos contratos, el mismo sesgo iba de **0.28 a
0.53 minutos** — mayor que la mayoría de los márgenes que se reclamaban por
encima de él. La comparación que aquí se reporta es atribuible; la anterior no lo
era.

### C. Modelos comparados
> [ANDAMIAJE — TRASLADADO 2026-07-30] Fuente: `documento-resultados.md` §2.
> **La asimetría de tuning va en el cuerpo, no en una nota al pie.** Es lo que
> impide atribuir a la clase de modelo la victoria del XGBoost en E2.

La comparación central enfrenta tres modelos:

| Modelo | Qué hace |
|---|---|
| **Persistencia** | Repite el último *headway* observado. Es el rival serio: sobre series cortas resulta notablemente difícil de superar, y es el *baseline* que la literatura del subcampo rara vez incluye. |
| **XGBoost** | *Gradient boosting* entrenado sobre la **misma ventana de 12 pasos** que la red, más hora, día y sentido. Es el competidor aprendido no profundo. |
| **LSTM** | Red recurrente sobre la secuencia de *headways* del vector. |

Se evaluaron además dos arquitecturas con estructura espacial explícita
—convolucional recurrente y de atención— que **no superan al LSTM plano** sobre
estos datos. Ese resultado se estableció sobre un conjunto de experimentos
anterior y no se rehízo bajo el protocolo de contigüidad descrito arriba, de modo
que se reporta como antecedente y no como parte de esta evidencia; la
Sección V-C lo sitúa frente a la literatura.

**El presupuesto de ajuste no está nivelado, y la asimetría corre en contra de la
red.** El XGBoost recibió **24 configuraciones por celda**, elegidas en
validación; el LSTM recibió **una** en E2 y E59, y **tres** en E4, heredadas de
una fase previa. La consecuencia para la lectura de los resultados es directa y
se aplica en la Sección IV-A: donde el LSTM gana, la conclusión es segura porque
gana con menos presupuesto; donde gana el XGBoost, **la ventaja no es atribuible
a la clase de modelo**. Nivelar el contraste requiere aproximadamente catorce
horas de cómputo en GPU y no se realizó.

### D. Definición del evento de *bunching*, y por qué esta
> [ANDAMIAJE — ESCRITO 2026-07-30] Es la subsección más delicada del artículo:
> declara que nuestra regla **no** es la del campo. Restricciones respetadas:
> (a) la forma "fracción de la media observada" **no se encontró como definición
> de evento publicada** —solo en descripciones de proveedores CAD/AVL, cuya única
> fuente localizada es un post de LinkedIn, no citable— así que se declara como
> **sustitución nuestra**, con el precedente de Yu et al. para la *clase* de
> sustitución, no para la forma; (b) **no** se cita la Ec. 3-7 del TCQSM como
> definición de lo que medimos: su `cvh` se normaliza por el *headway*
> **programado** y el nuestro por la media observada, así que no son la misma
> cantidad (ver §C2 de `fuentes-verificadas.md`); (c) no se afirma nada sobre la
> procedencia del cociente ¼ más allá de quién lo usa; (d) la implementación
> literal está en `bunching_flags` de `src/evaluation/vector_metrics.py`, y la
> prosa tiene que coincidir con ella: la media se toma **sobre la misma columna
> que se marca**.

El objeto que este trabajo pronostica es el vector de *headways* de un corredor
en un instante, y el evento se define sobre ese vector. Una celda se marca como
*bunching* cuando su *headway* cae por debajo de la mitad de la media del vector
al que pertenece. Un punto de esa definición merece énfasis porque es donde se
concentra todo el argumento del artículo: **la media se calcula siempre sobre el
mismo vector que se está marcando**. Al evaluar observaciones, es la media
observada; al evaluar un pronóstico, es la media del **vector predicho**. No es
una simplificación de implementación sino la única lectura operativamente
honesta, porque en el momento de decidir si suena una alarma el operador dispone
del vector que su modelo predijo y no del que va a ocurrir.

Esta no es la convención del campo, y conviene decirlo antes que un revisor. La
forma relativa dominante normaliza por el *headway* **programado**: un cuarto del
programado en Yu et al. (2016) y en Moreira-Matias et al. (2016), la mitad del
programado en el *Transit Capacity and Quality of Service Manual*. La alternativa
es un corte absoluto, como el minuto que emplean Sun et al. (2021). Las dos
opciones exigen algo que estos datos no tienen: un horario publicado, o un valor
en minutos fijado de antemano. El corredor estudiado aquí se reconstruye desde
GPS crudo, sin GTFS y sin tabla de servicio, de modo que ninguna de las dos
formas es aplicable tal cual.

La media del propio vector es el sustituto disponible, y se declara como
**sustitución nuestra y no como herencia**: la forma "fracción de la media
observada" no se encontró como definición de evento en la literatura publicada.
Lo que sí tiene precedente directo es la *clase* de sustitución. Yu et al. (2016)
enfrentan el mismo vacío en Pekín —el operador no publica una tabla fija— y lo
resuelven tomando como *headway* "programado" el **observado en la primera parada**
de la misma corrida. Sustituir un horario ausente por una referencia extraída del
propio dato es, entonces, práctica establecida en el trabajo más citado del
subcampo. Nuestra elección difiere en el punto de referencia, y **esa diferencia
es exactamente el mecanismo que este artículo estudia**: la referencia de Yu et
al. es observada y permanece fija a lo largo de la corrida, mientras que la
nuestra se recalcula sobre el vector evaluado y, por lo tanto, se comprime junto
con el pronóstico. Bajo un corte absoluto o anclado a un horario, la compresión
de dispersión mueve un solo lado de la desigualdad; bajo esta regla mueve los
dos.

El cociente de un medio proviene del TCQSM, que lo emplea para caracterizar
cuándo un vehículo está fuera de intervalo, y aparece en uso reciente en Zhang et
al. (2022). Aquí se aplica ese mismo cociente a un denominador distinto —la media
observada en lugar del *headway* programado—, lo cual es una elección propia y no
una lectura del manual. Por la misma razón, el coeficiente de variación que
reportamos en la Sección IV-B **no** es el `cvh` del manual, que se normaliza por
el programado y se calcula sobre desvíos respecto de él; son cantidades distintas
y no se citan como equivalentes.

La regla produce una tasa base de entre 17 % y 30 % según corredor y horizonte.
Es alta para el subcampo, cuyos trabajos reportan prevalencias de entre 0.15 % y
17 %. La consecuencia se declara aquí y se explota en las Secciones IV-C y IV-F: a
mayor prevalencia, **más alto** es el piso que un F1 regala a un detector sin
contenido, de modo que el régimen de este trabajo vuelve el argumento del piso
trivial más exigente y no más laxo.

Finalmente, y puesto que toda la contribución depende de esta elección, la
elección se somete a prueba en lugar de defenderse por argumento. La Sección IV-E
repite la detección completa con un corte **absoluto en minutos**, calibrado
fuera de muestra e idéntico para el vector observado y el predicho, incluida la
convención de un cuarto que domina la literatura. El resultado se anticipa aquí
porque contradice lo que esperábamos: el colapso del detector no se atenúa bajo
un corte absoluto, sino que **empeora**, y bajo la convención del campo empeora
en dos órdenes de magnitud. La regla auto-referencial resultó ser la conservadora
de las dos.

### E. Protocolo de evaluación
> [ANDAMIAJE — TRASLADADO 2026-07-30] Fuente: `documento-resultados.md` §4 y
> §5.5. Es lo que diferencia al artículo en este venue, así que va desarrollado.

Se evalúan cuatro horizontes de predicción: **1, 3, 5 y 10 minutos**. Todas las
comparaciones son **pareadas sobre muestras idénticas**. Esto no es una precaución
cosmética: cuando dos modelos se puntúan sobre conjuntos de filas distintos, el
estadístico de Diebold-Mariano queda **indefinido** y no meramente sesgado, porque
se construye sobre el diferencial de pérdida fila a fila. El contrato de identidad
de muestra descrito en la subsección B es lo que hace que el estadístico exista.

**Significancia.** Las muestras del mismo día de servicio comparten clima,
incidentes y demanda —un accidente a las 08:00 moldea toda la mañana—, de modo
que tratarlas como independientes infla la significancia. La varianza se agrupa
por **día de servicio**, y el conjunto de prueba contiene 22 días: ese es el
tamaño de muestra efectivo, no las decenas de miles de filas. Se aplica corrección
de muestra pequeña, con estimadores robustos a heterocedasticidad y
autocorrelación como contraste, y se reporta además el Wilcoxon pareado. El
tamaño de efecto se presenta primero y el valor *p* actúa como piso, no como
veredicto.

**Estratificación ex-ante.** Cada predicción se estratifica por la dispersión de
su **propia ventana de entrada** —cuánto se movió el *headway* en los doce minutos
que el modelo efectivamente observó—. Es una variable conocida en el momento de
predecir, y sus terciles se **congelan sobre entrenamiento y validación** y se
aplican a prueba, nunca se calibran sobre prueba. Condicionar sobre ella y luego
contrastar es por lo tanto legítimo.

**Validación en múltiples orígenes.** El protocolo completo —winsorización,
portón de población, entrenamiento y exportación— se re-ejecutó sobre dos
orígenes anteriores con ventanas de prueba disjuntas de la publicada. No es una
re-partición de los mismos residuos sino entrenamientos nuevos. Se declara que
los conjuntos de **entrenamiento** correspondientes están **anidados y no son
independientes**: lo que la comparación establece es estabilidad frente a la
elección de ventana de prueba, no replicación independiente.

| Origen | Entrena | Prueba |
|---|---|---|
| `r1` | 61 días | 2023-12-23 → 2024-01-13 |
| `r2` | 83 días | 2024-01-14 → 2024-02-04 |
| `main` | 107 días | 2024-02-08 → 2024-02-29 (la publicada) |

**Métricas.** En el eje escalar se reportan el error absoluto medio y el error
cuadrático, y se los reporta juntos por una razón que la Sección IV-A desarrolla:
a horizonte corto nombran ganadores opuestos. En el eje de detección se reportan
**MCC, ROC-AUC y precisión media**, con el **piso del detector trivial explícito
en cada tabla**. No se usa F1 como métrica de decisión, por los motivos
establecidos en la Sección II-D.

**Calibración del umbral.** Cuando una métrica requiere un punto de operación,
el corte se ajusta maximizando MCC sobre una **ventana anterior y disjunta**
(`r2`) y se aplica hacia adelante (`main`). Es la única dirección en la que un
operador podría calibrar en producción, y la Sección IV-F contrasta empíricamente
esa elección de objetivo contra la alternativa basada en F1.

---

## IV. Resultados

> [ANDAMIAJE — REESTRUCTURADO 2026-07-29] El orden cambió. Antes iba: cruce → significancia → volatilidad → enrutador, o sea el resultado escalar como titular. Ahora el escalar es **contexto** y el titular es el artefacto de umbral y su reparación.
>
> Figuras: usar `contiguo-artefacto-umbral.png` y `contiguo-deteccion-sin-umbral.png` (**solo funcionan como par**), más `contiguo-degradacion.png` y `contiguo-volatilidad.png`. **NO usar** `curva-degradacion.png` ni `volatilidad-crossover.png`: son de las familias congeladas y no de este pipeline. La figura `contiguo-disociacion.png` fue eliminada — graficaba F1 con corte fijo como si midiera a los modelos.

### A. Contexto: el resultado escalar y su frontera
> [ANDAMIAJE — TRASLADADO 2026-07-30] Doc §3 y §4, comprimido. **Presentar como
> contexto establecido, no como aporte** — el cruce por horizonte es folclore
> conocido en pronóstico de tráfico. Figuras `contiguo-degradacion.png` y
> `contiguo-volatilidad.png`.

El eje escalar se reporta como contexto y no como contribución: el cruce por
horizonte entre persistencia y modelo aprendido es un comportamiento conocido en
pronóstico de tráfico. Medido sobre la población pareada, la diferencia de error
absoluto medio contra la persistencia (negativo indica ventaja del aprendiz) es:

| Corredor | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| E2 | +0.067 | −0.851 | −1.109 | **−1.473** |
| E59 | +0.334 | −0.186 | −0.491 | **−1.173** |
| E4 | +0.464 | −0.064 | −0.536 | **−1.381** |

**El XGBoost reproduce el patrón completo** —a h=10, −1.585 en E2, −0.787 en E59
y −1.085 en E4—, de modo que el cruce no es una propiedad del aprendizaje
profundo sino del problema: existe un umbral de anticipación a partir del cual el
último valor observado deja de bastar, y cualquier aprendiz razonable lo cruza.
La comparación entre el LSTM y el XGBoost se parte por corredor y no admite un
ganador global; conforme a lo declarado en la Sección III-C, la ventaja del
XGBoost en E2 **no es atribuible a la clase de modelo** porque recibió
veinticuatro veces más presupuesto de ajuste.

Con la varianza agrupada por día de servicio, el titular defendible se acota: a
**h=1 gana la persistencia**, con firmeza en E4 y E59 y al borde en E2
(*p* = 0.062); **h=3 es zona de transición sin victoria declarable**, algo en lo
que convergen cuatro diagnósticos independientes —media contra mediana, terciles
de volatilidad, desagregación por sentido y el enrutador de la subsección G—; y a
**h≥5 el aprendiz gana en media y en mediana, con significancia amplia, en los
tres corredores**.

**La frontera real no es el horizonte sino la volatilidad.** Estratificando cada
predicción por la dispersión de su propia ventana de entrada, en **11 de las 12
celdas** la ventaja del aprendiz crece de forma ordenada del tercil calmo al
volátil, y dentro de cada tercil crece con el horizonte:

| Celda | Ventana calma | Ventana media | Ventana volátil |
|---|---|---|---|
| E2 h=1 | +0.218 | +0.086 | −0.080 |
| E4 h=3 | **+0.370** | +0.006 | **−0.451** |
| E59 h=3 | +0.159 | −0.157 | −0.559 |
| E4 h=10 | −0.659 | −1.116 | −2.172 |

El cruce es, entonces, un umbral de volatilidad que el horizonte va cruzando:
alargar el horizonte empuja la ventaja del aprendiz hacia terciles cada vez más
calmos hasta cubrirlos todos. Eso explica agregados engañosos como el de E4 a
h=3, cuyos −0.064 minutos son la mezcla de perder claramente en dos tercios de la
masa y ganar claramente en el tercio restante.

**Una inversión de signo a h=1 anticipa el mecanismo de la subsección siguiente.**
Contra la persistencia y sobre las mismas filas:

| Corredor | Δ MAE | Gana | *p* | Δ error cuadrático | Gana | *p* |
|---|---|---|---|---|---|---|
| E2 | +0.067 min | persistencia | 0.062 | −15.36 min² | **LSTM** | 7.3e−18 |
| E4 | +0.464 min | persistencia | 1.0e−13 | −6.49 min² | **LSTM** | 2.7e−14 |
| E59 | +0.334 min | persistencia | 2.6e−13 | −8.35 min² | **LSTM** | 8.4e−16 |

El vector aplanado **pierde** el error absoluto y **gana** el cuadrático, en los
tres corredores. Es el comportamiento esperable de un pronóstico contraído:
evita los errores grandes, que el cuadrático castiga desproporcionadamente, al
costo de fallar más seguido por poco, que es lo único que el absoluto cuenta. El
par de signos también descarta una explicación tentadora: si la contracción
viniera de haber elegido el error absoluto como pérdida, el mismo vector no
podría perder en absoluto y ganar en cuadrático a la vez. La causa no es cuál de
los dos funcionales centrales se elige, sino **emitir un único número por celda**.

### B. La compresión de dispersión, medida
> [ANDAMIAJE — TRASLADADO 2026-07-30] Doc §5.2 y §5.5. **La distinción
> transversal contra temporal es obligatoria y va como bloque destacado**: es lo
> que separa nuestro resultado empírico del teorema de Patton y Timmermann.
> Y **no** citar la Ec. 3-7 del TCQSM como definición del CV (§C2 de
> `fuentes-verificadas.md`); sí su escala de nivel de servicio.

El coeficiente de variación del vector de *headways* mide cuán desparejo está el
corredor, y es la cantidad que el promedio no distingue. Un corredor con
intervalos de 9, 10, 11 y 10 minutos y otro con 1, 19, 2 y 18 tienen el mismo
promedio de diez minutos y describen servicios opuestos: en el primero el
pasajero espera diez minutos siempre; en el segundo los vehículos pasan de a
pares y dejan huecos de veinte minutos, que es precisamente el *bunching*. Sus
coeficientes de variación son 0.07 y 0.85. Se divide por la media porque dos
minutos de desvío son un desastre en un corredor de cinco minutos y son
irrelevantes en uno de quince.

Medido así, **el modelo describe el primer corredor cuando la realidad es el
segundo**:

| | CV real | CV predicho por el LSTM | Sesgo |
|---|---|---|---|
| E2 h=1 | 0.777 | 0.362 | −0.415 |
| E2 h=10 | 0.787 | **0.161** | **−0.626** |
| E4 h=10 | 0.577 | 0.213 | −0.365 |
| E59 h=10 | 0.614 | 0.260 | −0.354 |

El sesgo es negativo en **las 36 combinaciones** de corredor, horizonte y ventana
de prueba, empeora monótonamente con el horizonte y es notablemente estable entre
ventanas: en E2 h=10 vale −0.664, −0.647 y −0.626 en los tres orígenes. La
persistencia, que propaga el vector observado y por lo tanto conserva su forma
por construcción, mantiene un sesgo de a lo sumo **0.022** en valor absoluto. No
se trata de un defecto de esta arquitectura ni de esta pérdida: toda pérdida
puntual apunta a un funcional central, y un pronóstico que devuelve un único
número por celda devuelve una medida de centro.

> **Qué acotan los teoremas, y qué no.** La descomposición de la varianza y el
> Corolario 2 de Patton y Timmermann acotan la varianza **temporal de una serie
> escalar**: cuánto se mueve el pronóstico de una posición a lo largo del tiempo.
> La cantidad medida aquí es distinta —la dispersión **transversal entre las
> componentes del vector en un mismo instante**, promediada luego sobre
> instantes—. La ley de varianza total se aplica componente a componente, de modo
> que la dirección del efecto es la esperable, pero **no implica** el resultado
> transversal: un pronóstico podría tener un patrón posicional fuerte que le
> inflara la dispersión transversal aun siendo temporalmente plano. Las 36 celdas
> son por lo tanto un resultado **empírico y no un corolario**, y eso es lo que
> las vuelve reportables: miden la cantidad que le importa al operador —cuán
> desparejo se ve el corredor ahora— que ningún teorema existente cubre.

La consecuencia se aprecia mejor traduciendo el coeficiente a profundidad de
cola con la relación que el propio manual de referencia establece, `Z = 0.5/cv`.
Sobre las observaciones, un coeficiente de 0.79 da `Z ≈ 0.63`; sobre el
pronóstico, un coeficiente de 0.16 da `Z ≈ 3.1`, es decir una probabilidad de
excedencia del orden del 0.1 %. Aplicando la escala de nivel de servicio del
mismo manual, **el pronóstico califica al corredor como LOS A, "service provided
like clockwork", mientras que las observaciones lo califican como LOS F, "most
vehicles bunched"**. Mismo corredor, mismo instrumento, veredictos opuestos. La
aritmética es del manual; lo que este trabajo aporta es aplicarla al pronóstico.

### C. El artefacto: la alarma no suena
> [ANDAMIAJE — TRASLADADO 2026-07-30] Doc §1 y §5.3. Figura
> `contiguo-artefacto-umbral.png`. **Va emparejada con la figura de la subsección
> D: solo funcionan como par**, y compararlas es el aporte del trabajo.
> Precisión obligatoria sobre el MCC de "marcar todo": es **0/0 indeterminado**,
> y el cero es extensión por continuidad. Nunca "cero por construcción".

Aplicada la regla de evento de la Sección III-D, el cuadro con el umbral fijo es
inequívoco y, leído de forma inmediata, demoledor:

| E2, a 10 minutos de anticipación | LSTM | Persistencia |
|---|---|---|
| MAE (menor es mejor) | **5.32 min** | 6.79 min |
| F1 de *bunching* con el umbral fijo | 0.0013 | **0.332** |
| Veces que disparó, en 50 356 oportunidades | **14** | 15 084 |
| Eventos reales a detectar | 15 245 | 15 245 |

Catorce alarmas donde había quince mil eventos, un factor de **253** contra la
persistencia. Con ese corte la persistencia gana las doce celdas, por márgenes
que crecen con el horizonte: 2.8×, 10.9×, 35.6× y 253.4× en E2 a h=1, 3, 5 y 10.
La lectura inmediata es que el modelo se volvió ciego a la irregularidad.

**Tres hechos vacían esa tabla de contenido, y los tres salen del mismo conjunto
de datos que la produjo.**

*Primero, el corte publicado es el óptimo de la persistencia.* Reajustándolo
libremente por MCC sobre una ventana anterior y disjunta, **la persistencia
recupera 0.5× en 11 de las 12 celdas** (rango 0.46×–0.60×; la excepción es E2
h=10), mientras que el LSTM aterriza entre **0.58× y 0.91×**, siempre más laxo,
porque su vector está comprimido. La regla estaba escrita en las unidades de uno
de los dos competidores.

*Segundo, el ganador declarado pierde contra una regla sin contenido.* Marcar
todas las celdas produce un F1 de `2π/(1+π)`, y ese piso supera al F1 de la
persistencia en **5 de las 12 celdas, incluidas las tres de h=10**:

| Celda | F1 persistencia | F1 de "marcar todo" | ¿Supera el piso? |
|---|---|---|---|
| E2 h=10 | 0.332 | **0.465** | no |
| E4 h=10 | 0.268 | **0.304** | no |
| E59 h=10 | 0.303 | **0.344** | no |
| E2 h=1 | **0.581** | 0.460 | sí |

Conviene ser preciso sobre el MCC de esa misma regla, porque la formulación
descuidada es atacable: con ella los falsos negativos y los verdaderos negativos
son ambos cero, de modo que numerador y denominador se anulan y **el cociente
queda indeterminado**. El valor cero es la extensión por continuidad, la
convención estándar para matrices degeneradas, y coincide con el valor esperado
de un clasificador al azar. El F1, en cambio, vale `2π/(1+π) > 0` para esa misma
regla. Una métrica que sitúa una regla sin contenido por encima de los dos
modelos no puede decidir cuál de los dos detecta *bunching*.

*Tercero, cuando el aprendiz habla, acierta más.* En E2 a h=10 el LSTM dispara
catorce veces y acierta diez: **71 % de precisión contra una tasa base del 30 %**.
La persistencia dispara 15 084 veces y acierta 5 036, un **33 %**, tres puntos por
encima del azar. El *recall* del aprendiz colapsa; su precisión, no. Esa
proporción tiene un intervalo de confianza ancho (aproximadamente 42 %–92 % al
95 %) y el argumento no descansa en ella: en E59 a h=10 el LSTM dispara **1 573**
veces y acierta **777**, un 49 % contra una tasa base del 21 %, y en E4 acierta 75
de 150 contra el 18 %. El patrón es el mismo en los tres corredores y el volumen
lo sostiene en dos. **El modelo no se equivoca: está callado**, que es la firma de
un corte mal puesto y no de información ausente.

### D. La reparación: sin umbral, el veredicto se invierte
> [ANDAMIAJE — TRASLADADO 2026-07-30] Doc §5.4. Figura
> `contiguo-deteccion-sin-umbral.png`, que **forma par con la de la subsección C**.
> La celda partida (E4 h=5, una milésima) va **nombrada**, no omitida.

Dos instrumentos no pueden ser movidos por un corte: el ROC-AUC y la precisión
media son invariantes ante cualquier reescalado monótono del puntaje, que es
exactamente la transformación ante la cual un corte relativo fijo **no** es
invariante. La invariancia se sigue en un paso de que el AUC es una función de
los rangos del puntaje. Se añade un tercer instrumento que sí emplea un corte,
pero calibrado fuera de muestra sobre `r2` y aplicado hacia adelante a `main`.

| Celda | AUC LSTM | AUC persist. | MCC cal. LSTM | MCC cal. persist. | Ganador |
|---|---|---|---|---|---|
| E2 h=1 | 0.714 | **0.723** | 0.310 | **0.401** | persistencia |
| E2 h=3 | **0.629** | 0.598 | **0.178** | 0.160 | LSTM |
| E2 h=5 | **0.604** | 0.567 | **0.139** | 0.102 | LSTM |
| E2 h=10 | **0.565** | 0.528 | **0.085** | 0.027 | LSTM |
| E4 h=1 | 0.811 | **0.833** | 0.476 | **0.615** | persistencia |
| E4 h=10 | **0.604** | 0.558 | **0.126** | 0.111 | LSTM |
| E59 h=1 | 0.760 | **0.781** | 0.363 | **0.517** | persistencia |
| E59 h=10 | **0.632** | 0.571 | **0.161** | 0.119 | LSTM |

La tabla establece cuatro cosas. **El aprendiz no es ciego en ninguna celda**: su
AUC va de 0.565 a 0.811 y su elevación de precisión media sobre el piso trivial
de 1.19 a 3.16, contra valores de azar de 0.5 y 1.0 respectivamente; un modelo sin
información sobre el evento no puede producir esos números. **A h=10 el aprendiz
gana la detección en los tres corredores**, con umbral calibrado y sin umbral,
exactamente donde el corte fijo le atribuía un factor de 253× en contra. **A h=1
gana la persistencia en los tres**, que es también donde gana el error absoluto:
las dos métricas **coinciden** una vez removido el artefacto, de modo que no hay
disociación entre error escalar y detección. Y **en el agregado**, 6 de las 12
celdas van al aprendiz por AUC y 5 por MCC calibrado, contra 0 de 12 con el corte
fijo; un veredicto que pasa de unánime a repartido según el punto de operación es,
por definición, un veredicto sobre el punto de operación.

Esto retira una afirmación intermedia que conviene nombrar. La idea de que
alargar el horizonte mejora la ventaja escalar y destruye la fidelidad vectorial
era correcta solo a medias: la fidelidad de **forma** del vector sí se destruye
—las 36 celdas de la subsección B—, pero la capacidad de **discriminar el evento**
no se destruye, se reordena, y a favor del aprendiz en el horizonte largo.

### E. Robustez: no es de febrero, ni de nuestro umbral
> [ANDAMIAJE — TRASLADADO 2026-07-30] Doc §5.5, §5.6 y §6 (winsorización).
> **La salvedad del AUC absoluto va completa**, incluida la celda en 0.4934: es
> lo único que corre en contra y omitirla invalidaría la subsección.

Tres objeciones previsibles se responden con medición y no con argumento.

**No depende de la ventana de prueba.** El aplanamiento es negativo en las **36
de 36** combinaciones de corredor, horizonte y origen. El veredicto sin umbral
coincide en **11 de las 12** celdas entre los tres orígenes, y en los extremos es
unánime: a h=10 el aprendiz gana el AUC en los tres corredores y los tres
orígenes —nueve de nueve— y a h=1 la persistencia gana también nueve de nueve. La
única celda que se parte es **E4 h=5**, donde en la ventana publicada los dos AUC
son 0.6476 y 0.6486, **una milésima**: esa celda está sobre el cruce y no hay
victoria que reclamar, de modo que el desacuerdo confirma el mecanismo en lugar
de contradecirlo.

El contraste con el artefacto es el que más dice. Con el corte fijo la
persistencia gana las 36 celdas, pero el **tamaño** de la ventaja se mueve entre
ventanas de un modo que ninguna propiedad del modelo explicaría:

| Celda | `r1` | `r2` | `main` |
|---|---|---|---|
| E2 h=5 | 125.9× | 57.6× | 35.6× |
| **E2 h=10** | **2299×** | **817×** | **253×** |
| E4 h=10 | 21.3× | 46.4× | 17.7× |
| E59 h=10 | 11.1× | 9.9× | 8.8× |

Un factor que va de 253 a 2299 según el mes en la misma celda no mide una
capacidad: mide cuán lejos cayó el corte en la cola del pronóstico esa ventana en
particular, y la divergencia se agranda justo donde el denominador se hace
pequeño, que es la firma aritmética de una razón sin sentido. **En 15 de las 36
celdas la persistencia ni siquiera supera al detector trivial.**

**No depende de la forma auto-referencial del umbral.** Es la objeción más fuerte
que admite este trabajo, porque la regla la introdujimos nosotros. Se midió con
un corte **absoluto en minutos**, `K = ρ × mediana(headway observado en r2)` por
corredor y sentido, calibrado fuera de muestra e idéntico para el vector observado
y el predicho, con ρ = 0.5 para quedar comparable con nuestra regla y ρ = 0.25
para igualar la convención dominante del campo:

| Regla | Sub-disparo del LSTM (mediana de 12 celdas) | Peor celda |
|---|---|---|
| Auto-referencial, 0.5× la media del vector | 0.079 | — |
| **Absoluto, 0.5× la mediana de `r2`** | **0.040** | 0.00028 |
| **Absoluto, 0.25× la mediana de `r2`** (convención del campo) | **0.0007** | 0.000000 |

**El resultado contradice lo que esperábamos y refuerza el argumento.** El
colapso no se atenúa bajo un corte absoluto: **empeora**, y bajo la convención
del campo es **110 veces peor** que bajo la nuestra. La razón es geométrica: un
corte absoluto de 1.4 a 2.4 minutos vive en la cola lejana, que es exactamente
donde la compresión muerde más fuerte, mientras que la regla auto-referencial al
menos mueve su denominador con el nivel del vector. La objeción queda no solo
respondida sino invertida: de haber usado la convención dominante, el colapso
aparente habría sido dos órdenes de magnitud mayor.

**Y una salvedad que corre en contra, que se reporta porque corre en contra.** El
aprendiz carga **menos** información sobre el evento absoluto que sobre el
relativo. Con ρ = 0.25 el AUC mediano baja a **0.599**, contra el rango de
0.63–0.81 del evento relativo, y en E2 h=10 llega a **0.4934**, indistinguible del
azar. La afirmación de que el aprendiz no es ciego en ninguna celda **se sostiene
para el evento relativo y no se sostiene para el absoluto en esa celda**. Con
ρ = 0.5 el cuadro mejora: mediana 0.655, mínimo 0.518, una sola celda en 0.55 o
por debajo.

**No depende del techo de winsorización.** La objeción evidente al contrato de la
Sección III-B es que el uno por ciento recortado es la cola extrema, justo donde
se juega el argumento. Repuntuando todo contra objetivos crudos, con una
persistencia también recalculada sin techo, **ningún signo cambia** en las doce
celdas, el margen se mueve menos de 0.01 minutos en todas y el F1 de *bunching*
cambia menos de 0.005. La razón es estructural: lo que el techo recorta es la
cola **alta**, es decir huecos de servicio, mientras que el *bunching* es un
*headway* que colapsa hacia cero y ningún techo superior puede tocarlo.

### F. Por qué MCC y no F1, medido
> [ANDAMIAJE — TRASLADADO 2026-07-30] Doc §5.7. **Reportar el resultado mixto
> como mixto.** La formulación "el MCC es más estable" sería falsa en la mediana
> y un revisor lo verificaría en la primera fila de la tabla.

La Sección II-D justifica calibrar por MCC con un teorema. Lo que no está
publicado es la comparación **empírica** de estabilidad entre los dos objetivos,
y con tres ventanas disjuntas es medible:

| | Objetivo MCC | Objetivo F1 |
|---|---|---|
| Rango del corte entre los tres orígenes, mediana | 0.0357 | **0.0226** |
| Rango del corte, peor celda | **0.864** | 3.688 |
| Celdas con rango > 0.5 | **1 de 24** | 4 de 24 |
| MCC logrado en `main` según la ventana de calibración, mediana del rango | **0.00071** | 0.00242 |
| Ídem, peor celda | **0.018** | 0.098 |

**El resultado es mixto y no conviene maquillarlo.** En la **mediana**, el corte
ajustado por F1 es *más* estable, no menos. Lo que distingue al MCC son las
colas: el ajuste por F1 tiene cuatro celdas con rango superior a 0.5 y tres por
encima de 1.0 —todas de persistencia, es decir los colapsos degenerados de la
subsección C—, mientras que al MCC le ocurre en una sola.

Lo que decide es la última fila, que mide cuánto se mueve el desempeño realmente
desplegado según qué ventana tocó calibrar: allí el MCC es **3.4× más estable en
la mediana y 5.6× en el peor caso**. Un operador no elige un umbral, elige un
procedimiento, y el procedimiento basado en F1 funciona casi siempre y falla
catastróficamente a veces. La formulación defendible no es "el MCC es más
estable" —sería falsa en la mediana— sino que **el ajuste por F1 tiene un modo de
falla degenerado que el de MCC no tiene, y el costo fuera de muestra favorece al
MCC por un factor de 3 a 6**.

### G. El enrutador ex-ante
> [ANDAMIAJE — TRASLADADO 2026-07-30] Doc §6. Dos párrafos, que es lo que
> amerita. **Decir explícitamente que vale como demostración de ejecutabilidad y
> no como contribución** — inflarlo sería el tipo de reclamo que este artículo
> critica en otros.

Si cada modelo domina un régimen distinto, una política que conmute entre ellos
usando la volatilidad de la ventana —conocida al momento de predecir— debería
capturar algo. Se construyó con tres candidatos, aprendiendo la política sobre los
primeros trece días de servicio del conjunto de prueba y puntuándola sobre los
nueve restantes, que es el único corte que imita el despliegue. Un barrido de
veinte semillas mide cuánto se mueve la ganancia con la partición.

La política supera el ruido de partición en **2 de 12 celdas, ambas a h=3** (E4
−0.073 min, E59 −0.042 min) y en ninguna a h≥5, donde el aprendiz ya domina los
tres terciles y no queda nada que conmutar. **7 de las 12 políticas son
degeneradas**: eligen el mismo modelo en los tres terciles, de modo que el
enrutador *es* un modelo puro disfrazado. Y en E2 h=1 la política ayuda bajo
partición aleatoria (−0.028 de mediana) pero **perjudica** bajo corte temporal
(+0.037), es decir que no generaliza hacia adelante en el tiempo, que es la única
dirección que importa. La conclusión es acotada: conmutar paga solo donde ningún
modelo puro domina, o sea en la zona de transición, y allí paga unos pocos
segundos de error absoluto. **Vale como demostración de ejecutabilidad, no como
contribución.**

---

## V. Discusión

### A. Interpretación operativa
> [ANDAMIAJE — ESCRITO 2026-07-30] Es la subsección que le habla al operador.
> El eje es la diferencia entre dos diagnósticos que se parecen y tienen
> consecuencias opuestas. **El límite honesto va en el mismo párrafo que la
> lectura optimista, no en una sección aparte** — si el AUC de 0.60 aparece solo
> en Limitaciones, la sección se lee como venta.

Para un operador, las dos lecturas posibles de la Sección IV-C no se parecen en
nada. *"El modelo no anticipa el bunching"* cierra una línea de trabajo: si el
pronóstico no contiene la información, no hay ajuste que la haga aparecer y la
inversión en modelado fue a pérdida. *"El modelo anticipa el bunching, pero la
alarma está mal seteada"* dice lo contrario, y su costo de reparación es de otro
orden: un escalar recalibrado sobre datos que la operación ya tiene.

La evidencia sostiene la segunda lectura, y lo hace de una forma que conviene
enunciar en términos operativos y no estadísticos. Con el umbral fijo, en E2 a
diez minutos, el aprendiz emitió catorce alarmas donde había quince mil eventos;
la lectura inmediata es ceguera. Pero de esas catorce, diez fueron correctas —un
71 % de precisión contra una tasa base del 30 %—, mientras que la persistencia
disparó quince mil veces para acertar el 33 %, apenas tres puntos por encima del
azar. La proporción de aciertos de esa celda tiene un intervalo ancho, y por eso
el argumento no descansa en ella: en E59 a diez minutos el aprendiz emite 1 573
alarmas y acierta 777, un 49 % contra una tasa base del 21 %, y en E4 acierta 75
de 150 contra el 18 %. El patrón es el mismo en los tres corredores y el volumen
lo sostiene en dos. **El modelo no se equivoca cuando habla: está callado.** Y
estar callado es precisamente lo que corrige un umbral.

Ese diagnóstico tiene una consecuencia práctica inmediata. Una organización que
haya evaluado un pronóstico de *headways* trasplantándole el corte con el que
etiqueta sus observaciones históricas puede estar descartando un modelo que sí
ordena correctamente el riesgo. El procedimiento que este trabajo propone no
requiere reentrenar, ni cambiar de arquitectura, ni pasar a pronóstico
probabilístico: requiere ajustar el corte sobre una ventana anterior y disjunta y
aplicarlo hacia adelante, que además es la única dirección en la que un operador
podría calibrarlo en producción. Y sugiere una práctica de evaluación mínima
—reportar al menos una métrica sin umbral junto a la métrica de alarma— que
habría bastado para no emitir el veredicto equivocado.

Hasta ahí llega lo que estos resultados autorizan, y el límite es importante. Un
AUC en torno a 0.60, que es el orden de magnitud de las celdas donde el aprendiz
gana, es información real y está lejos de ser un sistema de despacho. Este
trabajo no modela la función de costo que traduciría esa capacidad de
ordenamiento en una decisión concreta —cuántos vehículos retener, durante cuánto
tiempo, con qué penalización sobre el resto del servicio—, ni calibra el umbral
contra incidentes registrados por la operación. Lo que se sostiene, entonces, es
más modesto y más útil de lo que sugeriría el titular: la línea de trabajo **no
está cerrada**, y la evidencia que parecía cerrarla medía el umbral y no el
modelo.

### B. Qué queda del aporte, y qué no
> [ANDAMIAJE — ESCRITO 2026-07-30] Cuatro delimitaciones, de la más cercana a la
> más lejana, dos oraciones cada una. Vale más que reclamar de más. **No ablandar
> el párrafo de cierre**: nombra lo que se retiró durante la propia investigación,
> y en este venue eso es diferenciador, no debilidad.

Conviene delimitar el aporte contra sus vecinos más cercanos, en orden de
cercanía, porque buena parte del argumento que parecía nuestro no lo es.

Frente a **Petetin et al. (2022)**, que es el vecino más próximo, no reclamamos
la disociación entre error continuo y desempeño categórico, ni la medición de la
compresión de dispersión, ni su agravamiento con el horizonte: las tres están
publicadas allí, sobre pronósticos de ozono. Lo que agregamos es la recalibración
del umbral, que en su marco no es una omisión sino una imposibilidad —sus cortes
son regulatorios— y el régimen del umbral **relativo y auto-referencial**, donde
la compresión mueve numerador y denominador a la vez y el resultado queda
gobernado por el coeficiente de variación en lugar del nivel.

Frente a **Sun, Schmöcker y Nakamura (2021)**, que diagnosticaron el síntoma
dentro del transporte, no reclamamos haber descubierto que el paradigma falla, ni
la crítica del punto de operación único, ni la reversión del veredicto al puntuar
sin umbral. La diferencia está en la respuesta: ellos concluyen que hace falta
cambiar de clase de modelo, y nosotros medimos que basta con recalibrar el punto
de operación del modelo que ya se tiene.

Frente a **Hoffmann, Menz y Spekat (2018)**, no reclamamos la idea de recalcular
un umbral contra la distribución de cada modelo: es exactamente su procedimiento,
ocho años antes, en *downscaling* climático. Lo que agregamos es el traslado al
transporte, con un horizonte de pronóstico como eje y con métricas de detección
que su marco no contempla.

Frente a **Li, Yang y Wang (2025)**, que describen un aplanamiento muy similar en
predicción de arribos de buses, la diferencia es de agente causal y por lo tanto
de tratabilidad: su villano es la normalización del preprocesamiento y se cura
por arquitectura, mientras que el que identificamos es la función de pérdida y
sobrevive a la arquitectura. Ninguna de las dos afirmaciones invalida a la otra,
pero no son la misma y no deben leerse como tales.

Queda por decir qué **no** afirma este trabajo, y una parte de esa lista es
producto de haber retirado conclusiones propias durante la investigación. No
afirmamos que la sub-dispersión de los pronósticos puntuales sea un hallazgo
nuevo: es un enunciado publicado y, en su dependencia del horizonte, un teorema.
No afirmamos que el cruce por horizonte entre persistencia y aprendiz sea
novedoso, que es folclore conocido en pronóstico de tráfico. No afirmamos que el
resultado nulo espacial sea una contribución, por los motivos de la subsección
siguiente. Y no afirmamos —como sostuvo una versión anterior de este trabajo— que
la métrica escalar y la fidelidad vectorial nombren ganadores opuestos: esa
lectura dependía por completo del umbral mal transportado, y las mediciones que
la sostenían son las mismas que ahora la refutan.

### C. El resultado nulo espacial, en contexto
> [ANDAMIAJE — ESCRITO 2026-07-30] Corto y enmarcado como **confirmación de
> resultado publicado**, nunca como hallazgo. Y con la salvedad de procedencia
> que las limitaciones 4 y 5 del documento de resultados obligan a declarar: el
> nulo se estableció sobre las familias congeladas, no se rehízo bajo el pipeline
> contiguo, y solo el LSTM tiene barrido de semillas.

Las dos arquitecturas con estructura espacial explícita —convolucional y de
atención— no superan al LSTM plano sobre estos corredores. El resultado se
presenta aquí como **confirmación de trabajo publicado y no como contribución**,
porque llegar segundo a una conclusión no la vuelve propia. Boudabbous et al.
(2026) reportan, sobre datos de tránsito de Montreal, que un LSTM supera a
arquitecturas de tipo *transformer* entre un 18 % y un 52 % con setenta y siete
veces menos parámetros —con la salvedad de que su objetivo de predicción es el
**retraso** y no el *headway*—, y Rodrigues (2022), en un trabajo dedicado
precisamente a la importancia de los *baselines* fuertes en problemas de
predicción de transporte, muestra que un patrón semanal combinado con regresión
lineal iguala al estado del arte espacio-temporal. Nuestro resultado es coherente
con ambos.

Dos advertencias impiden leerlo como algo más general. La primera es de alcance:
es un enunciado sobre estos corredores, este horizonte y esta escala de flota, no
una ley sobre arquitecturas espaciales. La segunda es de procedencia, y pesa
más: el nulo se estableció sobre las familias de experimentos congeladas y no se
rehízo bajo el pipeline contiguo con el que se produjeron los resultados de la
Sección IV, y solo el LSTM cuenta con barrido de semillas. No puede descartarse
que el desempeño de las arquitecturas espaciales esté afectado por la
inicialización. Por eso el resultado se reporta como antecedente propio, con su
respaldo en la literatura, y no como evidencia de esta comparación.

### D. Amenazas a la validez y limitaciones
> [ANDAMIAJE] YA REDACTADO — las 11 limitaciones del doc §8. Mantenerla sustantiva: en el relevamiento del venue, 3 de 8 papers tienen algo parecido y ninguno tiene disponibilidad de datos. Las que más pesan: el valor operativo no está modelado (lim. 8), el umbral no está calibrado contra incidentes registrados (lim. 9), y la calibración usa dos ventanas con entrenamientos anidados en lugar de validación cruzada (lim. 10).

---

## VI. Conclusión y Trabajo Futuro

> [ANDAMIAJE — ESCRITO 2026-07-30] Reescrita desde cero: la versión anterior
> cerraba sobre "la métrica decide el ganador", retirado el 2026-07-29. Cierra
> sobre el **vacío**, no sobre los resultados — los números están en la Sección IV
> y repetirlos acá los devalúa. Se agregó un quinto ítem de trabajo futuro que no
> estaba en el guion y que salió de leer a Yu et al.: la **tercera forma de
> umbral** (referencia observada y fija) es la que nadie midió, ni ellos ni
> nosotros.

Un pronóstico puntual describe un corredor más regular de lo que es. No es un
defecto de entrenamiento sino la consecuencia de qué cantidad estima: una pérdida
puntual elicita un funcional central de la distribución condicional, y ninguno de
los funcionales centrales tiene por qué reproducir la dispersión. En el vector de
*headways* que estudiamos, ese déficit aparece en la totalidad de las celdas
evaluadas y se agrava con el horizonte, tal como la teoría del pronóstico
anticipa y como se ha documentado en otros dominios.

La contribución de este trabajo no es ese déficit, sino lo que se sigue de él
cuando la salida del modelo se convierte en una alarma. **La compresión de
dispersión no le quita al pronóstico información sobre el evento: le cambia las
unidades en las que esa información está escrita.** Una regla de decisión
calibrada sobre observaciones deja de significar lo mismo al aplicarse sobre un
vector comprimido, y bajo un umbral relativo y auto-referencial deja de
significarlo dos veces, porque también el denominador se contrae. Trasplantar esa
regla fabrica entonces una degradación aparente de dos órdenes de magnitud que no
existe en la información, y que se disuelve al reajustar un solo escalar sobre
una ventana anterior y disjunta o al puntuar sin umbral. La verificación más
importante es la que corrió en contra de nuestra hipótesis: con un corte absoluto
—incluida la convención de un cuarto que domina la literatura— el colapso no se
atenúa sino que empeora, de modo que la forma auto-referencial resultó ser la
conservadora y no la que maximiza el artefacto.

De ahí se desprende una recomendación de evaluación más general que el caso de
estudio, y es lo que este artículo querría dejar instalado. **Un veredicto de
detección obtenido trasplantando un umbral entre espacios de distinta dispersión
mide el umbral, no el modelo.** Sostenerlo exige tres cosas que hoy no son
práctica en el subcampo: reportar al menos una métrica sin umbral junto a la
métrica de alarma, publicar el piso del detector trivial contra el cual leerla, y
declarar en qué espacio fue calibrado el corte. Ninguna de las tres requiere
datos adicionales ni capacidad de cómputo; las tres habrían bastado para no
emitir el veredicto que este trabajo tuvo que retirar sobre sus propios
resultados.

El trabajo futuro se ordena por lo que más cambiaría las conclusiones. Primero, el
pronóstico probabilístico del vector de *headways* con probabilidades de
excedencia explícitas, que es la ruta que Sun et al. (2021) propusieron como
continuación de su propio trabajo y que, hasta donde alcanza nuestro relevamiento,
nadie escribió; sustituiría la recalibración del punto de operación por una
distribución predictiva de la que el punto de operación se deriva. Segundo, la
**tercera forma de umbral**: entre el corte absoluto y el auto-referencial queda
la referencia observada y fija que emplean Yu et al. (2016), que ni ellos ni
nosotros evaluamos frente a las otras dos y que la Sección IV-E no cubre.
Tercero, validación cruzada rotativa de la calibración del umbral en lugar de las
dos ventanas con entrenamientos anidados que usamos aquí. Cuarto, la función de
costo operativo que traduzca capacidad de ordenamiento en decisiones de despacho,
sin la cual la lectura optimista de la Sección V-A no puede llevarse a operación.
Y quinto, replicación en otras ciudades y escalas de flota, que es lo único que
distinguiría un mecanismo general de una propiedad de estos tres corredores.

---

## Referencias

> [ANDAMIAJE — ESCRITO 2026-07-30] **44 entradas**, dentro del presupuesto
> observado del venue (26–46, mediana ≈41). Ordenadas por **primera aparición en
> el texto**, que es la convención IEEE; si la traducción reordena secciones, la
> renumeración es mecánica. Solo entra lo que la prosa cita: la lista de "no
> citar" de `fuentes-verificadas.md` es vinculante y ninguna de esas fuentes
> aparece acá.
>
> **PROCEDENCIA DE LOS METADATOS.** Títulos, bylines completos, volúmenes,
> números y páginas se **transcribieron** desde la API de Crossref (29 entradas
> con DOI) y desde las páginas de arXiv (10 entradas), no desde memoria. Los
> scripts que hicieron las consultas quedaron en el *scratchpad* de la sesión.
> El primer borrador de esta lista tenía **títulos inventados** —el de Sun et al.
> y el de Mayer y Yang no se parecían al real, y el de Rodrigues tampoco—, así que
> **ninguna entrada de aquí debe reescribirse de memoria**. Faltan solo [3], [6],
> [10], [30] y [33], verificadas por Crossref o por la fuente primaria pero sin
> pasar por este lote.
>
> Cuatro avisos que sobreviven a la verificación:
> - **[2] Mayer y Yang.** Crossref fecha el número impreso en **2023** (vol. 39,
>   n.º 2); la disponibilidad en línea es de 2022, que es el año que usa el
>   borrador en castellano. En el formato final IEEE la cita en texto es `[2]`, así
>   que la discrepancia desaparece — **pero la entrada debe decir 2023**.
> - **[13] Yu, Wu, Chen y Ma.** Mismo caso: el DOI es de 2016 y el número impreso
>   es **2017**, vol. 18 n.º 7. El texto en castellano ya dice (2017).
> - **[24] Liu et al.** El venue NeurIPS 2022 **sigue sin confirmar**: la página
>   de arXiv no trae `journal-ref`. Se cita como preprint.
> - **[39] McDermott et al.** El DOI de proceedings no está verificado, pero la
>   página de arXiv confirma NeurIPS 2024 con enlace a OpenReview.
>
> **Resuelto de paso: V5.** El orden de autores de Boyd et al. era el pendiente
> más molesto y quedó fijado desde arXiv: Boyd, Santos Costa, Davis, Page.
>
> Y una nota de honestidad sobre **[3] Vannitsem y Hagedorn**: se cita de segunda
> mano, como atribución de Mayer y Yang. El §II-B no le atribuye texto ni cifras.
> Si se lee el original antes de someter, se puede citar de primera mano.

[1] W. Sun, J. D. Schmöcker, and T. Nakamura, "On the tradeoff between sensitivity and specificity in bus bunching prediction," *Journal of Intelligent Transportation Systems*, vol. 25, no. 4, pp. 384–400, 2021. doi: 10.1080/15472450.2020.1725887
[2] M. J. Mayer and D. Yang, "Calibration of deterministic NWP forecasts and its impact on verification," *International Journal of Forecasting*, vol. 39, no. 2, pp. 981–991, 2023 (en línea 2022). doi: 10.1016/j.ijforecast.2022.03.008
[3] S. Vannitsem and R. Hagedorn, "Ensemble forecast post-processing over Belgium: comparison of deterministic-like and ensemble regression methods," *Meteorological Applications*, vol. 18, no. 1, pp. 94–104, 2011. doi: 10.1002/met.217
[4] A. J. Patton and A. Timmermann, "Forecast rationality tests based on multi-horizon bounds," *Journal of Business & Economic Statistics*, vol. 30, no. 1, pp. 1–17, 2012. doi: 10.1080/07350015.2012.634337
[5] H. Petetin, D. Bowdalo, P.-A. Bretonnière, M. Guevara, O. Jorba, J. Mateu Armengol, M. Samso Cabre, K. Serradell, A. Soret, and C. Pérez García-Pando, "Model output statistics (MOS) applied to Copernicus Atmospheric Monitoring Service (CAMS) O₃ forecasts: trade-offs between continuous and categorical skill scores," *Atmospheric Chemistry and Physics*, vol. 22, no. 17, pp. 11603–11630, 2022. doi: 10.5194/acp-22-11603-2022
[6] Transportation Research Board, *Transit Capacity and Quality of Service Manual*, 2nd ed. (TCRP Report 100), Part 3, Ch. 3, Exhibit 3-30, p. 3-48. Washington, DC: TRB, 2003.
[7] V. Borges Santos, C. E. S. Pires, D. Cassimiro Nascimento, and A. R. M. de Queiroz, "A decision tree ensemble model for predicting bus bunching," *The Computer Journal*, vol. 65, no. 8, pp. 2044–2062, 2022. doi: 10.1093/comjnl/bxab045
[8] H. Yu, D. Chen, Z. Wu, X. Ma, and Y. Wang, "Headway-based bus bunching prediction using transit smart card data," *Transportation Research Part C: Emerging Technologies*, vol. 72, pp. 45–59, 2016. doi: 10.1016/j.trc.2016.09.007
[9] M. Rezazada, N. Nassir, E. Tanin, and A. Ceder, "Bus bunching: a comprehensive review from demand, supply, and decision-making perspectives," *Transport Reviews*, vol. 44, no. 4, pp. 766–790, 2024. doi: 10.1080/01441647.2024.2313969
[10] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995. doi: 10.1080/07350015.1995.10524599
[11] L. Moreira-Matias, O. Cats, J. Gama, J. Mendes-Moreira, and J. F. de Sousa, "An online learning approach to eliminate bus bunching in real-time," *Applied Soft Computing*, vol. 47, pp. 460–482, 2016. doi: 10.1016/j.asoc.2016.06.031
[12] X. Chen, Z. Cheng, J. G. Jin, M. Trépanier, and L. Sun, "Probabilistic forecasting of bus travel time with a Bayesian Gaussian mixture model," arXiv:2206.06915, 2022.
[13] H. Yu, Z. Wu, D. Chen, and X. Ma, "Probabilistic prediction of bus headway using relevance vector machine regression," *IEEE Transactions on Intelligent Transportation Systems*, vol. 18, no. 7, pp. 1772–1781, 2017. doi: 10.1109/tits.2016.2620483
[14] M. Usama and H. Koutsopoulos, "Real time headway predictions in urban rail systems and implications for service control: a deep learning approach," arXiv:2510.03121, 2025.
[15] T. Gneiting, "Making and evaluating point forecasts," *Journal of the American Statistical Association*, vol. 106, no. 494, pp. 746–762, 2011. doi: 10.1198/jasa.2011.r10138
[16] S. Ravuri, K. Lenc, M. Willson, D. Kangin, R. Lam, P. Mirowski, M. Fitzsimons, M. Athanassiadou, S. Kashem, S. Madge, R. Prudden, A. Mandhane, A. Clark, A. Brock, K. Simonyan, R. Hadsell, N. Robinson, E. Clancy, A. Arribas, and S. Mohamed, "Skilful precipitation nowcasting using deep generative models of radar," *Nature*, vol. 597, no. 7878, pp. 672–677, 2021. doi: 10.1038/s41586-021-03854-z
[17] C. Subich, S. Z. Husain, L. Separovic, and J. Yang, "Fixing the double penalty in data-driven weather forecasting through a modified spherical harmonic loss function," *ICML*, 2025. arXiv:2501.19374
[18] M. Bonavita, "On some limitations of current machine learning weather prediction models," *Geophysical Research Letters*, vol. 51, no. 12, 2024. doi: 10.1029/2023GL107377
[19] E. E. Ebert, "Fuzzy verification of high-resolution gridded forecasts: a review and proposed framework," *Meteorological Applications*, vol. 15, no. 1, pp. 51–64, 2008. doi: 10.1002/met.25
[20] H. Wernli, C. Hofmann, and M. Zimmer, "Spatial forecast verification methods intercomparison project: application of the SAL technique," *Weather and Forecasting*, vol. 24, no. 6, pp. 1472–1484, 2009. doi: 10.1175/2009WAF2222271.1
[21] H. von Storch, "On the use of 'inflation' in statistical downscaling," *Journal of Climate*, vol. 12, no. 12, pp. 3505–3506, 1999. doi: 10.1175/1520-0442(1999)012<3505:OTUOII>2.0.CO;2
[22] R. Huth, "Statistical downscaling of daily temperature in central Europe," *Journal of Climate*, vol. 15, no. 13, pp. 1731–1742, 2002. doi: 10.1175/1520-0442(2002)015<1731:SDODTI>2.0.CO;2
[23] D. Maraun, "Bias correction, quantile mapping, and downscaling: revisiting the inflation issue," *Journal of Climate*, vol. 26, no. 6, pp. 2137–2143, 2013. doi: 10.1175/JCLI-D-12-00821.1
[24] Y. Liu, H. Wu, J. Wang, and M. Long, "Non-stationary Transformers: exploring the stationarity in time series forecasting," arXiv:2205.14415, 2022. **[venue no verificado — el registro de arXiv no trae journal-ref]**
[25] Z. Li, B. Yang, and M. Wang, "Exploring over-stationarization in deep learning-based bus/tram arrival time prediction: analysis and non-stationary effect recovery," arXiv:2509.06979, 2025. **[preprint sin venue]**
[26] P. Hoffmann, C. Menz, and A. Spekat, "Bias adjustment for threshold-based climate indicators," *Advances in Science and Research*, vol. 15, pp. 107–116, 2018. doi: 10.5194/asr-15-107-2018
[27] D. M. W. Powers, "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation," *International Journal of Machine Learning Technology*, vol. 2, no. 1, pp. 37–63, 2011. arXiv:2010.16061
[28] D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation," *BMC Genomics*, vol. 21, no. 1, art. 6, 2020. doi: 10.1186/s12864-019-6413-7
[29] T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006. doi: 10.1016/j.patrec.2005.10.010
[30] P. Flach and M. Kull, "Precision-recall-gain curves: PR analysis done right," *Advances in Neural Information Processing Systems 28*, pp. 838–846, 2015. ACM DL 10.5555/2969239.2969333
[31] Z. C. Lipton, C. Elkan, and B. Naryanaswamy, "Optimal thresholding of classifiers to maximize F1 measure," in *Machine Learning and Knowledge Discovery in Databases (ECML PKDD)*, LNCS 8725, pp. 225–239, 2014. doi: 10.1007/978-3-662-44851-9_15
[32] D. Chicco, N. Tötsch, and G. Jurman, "The Matthews correlation coefficient (MCC) is more reliable than balanced accuracy, bookmaker informedness, and markedness in two-class confusion matrix evaluation," *BioData Mining*, vol. 14, no. 1, art. 13, 2021. doi: 10.1186/s13040-021-00244-z
[33] O. Koyejo, N. Natarajan, P. Ravikumar, and I. S. Dhillon, "Consistent binary classification with generalized performance metrics," *Advances in Neural Information Processing Systems 27*, pp. 2744–2752, 2014.
[34] A. Luque, A. Carrasco, A. Martín, and A. de las Heras, "The impact of class imbalance in classification performance metrics based on the binary confusion matrix," *Pattern Recognition*, vol. 91, pp. 216–231, 2019. doi: 10.1016/j.patcog.2019.02.023
[35] Q. Zhu, "On the performance of Matthews correlation coefficient (MCC) for imbalanced dataset," *Pattern Recognition Letters*, vol. 136, pp. 71–80, 2020. doi: 10.1016/j.patrec.2020.03.030
[36] D. Chicco and G. Jurman, "The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric for assessing binary classification," *BioData Mining*, vol. 16, no. 1, art. 4, 2023. doi: 10.1186/s13040-023-00322-4
[37] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, e0118432, 2015. doi: 10.1371/journal.pone.0118432
[38] J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. 23rd Int. Conf. on Machine Learning (ICML)*, pp. 233–240, 2006. doi: 10.1145/1143844.1143874
[39] M. B. A. McDermott, H. Zhang, L. H. Hansen, G. Angelotti, and J. Gallifant, "A closer look at AUROC and AUPRC under class imbalance," *NeurIPS*, 2024. arXiv:2401.06091 **[DOI de proceedings no verificado; confirmado en OpenReview]**
[40] K. Boyd, V. Santos Costa, J. Davis, and D. Page, "Unachievable region in precision-recall space and its effect on empirical evaluation," *ICML*, 2012. arXiv:1206.4667
[41] J. Li, "Area under the ROC curve has the most consistent evaluation for binary classification," *PLOS ONE*, vol. 19, no. 12, e0316019, 2024. doi: 10.1371/journal.pone.0316019
[42] Y. Zhang, H. Xu, Q. C. Lu, and X. Fan, "Travel time reliability analysis considering bus bunching: a case study in Xi'an, China," *Sustainability*, vol. 14, no. 23, art. 15583, 2022. doi: 10.3390/su142315583
[43] E. Boudabbous, M. Karaa, L. Sboui, J. Montecinos, and O. Alam, "Scalable transit delay prediction at city scale: a systematic approach with multi-resolution feature engineering and deep learning," arXiv:2601.18521, 2026.
[44] F. Rodrigues, "On the importance of stationarity, strong baselines and benchmarks in transport prediction problems," arXiv:2203.02954, 2022.

<!--
  MAPA DE PROGRESO (borrar antes de enviar):

  [ESCRITO 2026-07-30]       ✅ SECCIÓN II COMPLETA — A, B, C, D y E
                             ✅ III.D Definición del evento
                             ✅ V.A Interpretación · V.B Aporte · V.C Nulo espacial
                             ✅ VI Conclusión y Trabajo Futuro
                             ✅ SECCIÓN I COMPLETA — A, B, C, D y E
                             ✅ Abstract (235 palabras) · Título · Keywords
                             ✅ Referencias (44, transcritas de Crossref/arXiv)
                             ✅ III.A-C y III.E trasladadas · IV.A-G trasladadas
  [POR TRASLADAR]            V.D Limitaciones (las 11 del doc §8)

  ESTADO: el manuscrito está completo salvo V.D. Lo que queda:
    (a) trasladar las 11 limitaciones del documento de resultados §8;
    (b) borrar TODOS los bloques [ANDAMIAJE] y este mapa;
    (c) traducir al inglés, recontando el Abstract — el inglés comprime;
    (d) completar la plantilla IJACSA a dos columnas (≤10 pp de cuerpo).

  VERIFICAR ANTES DE SOMETER, y no es opcional:
    - Cada cifra de la Sección IV contra documento-resultados.md. Se trasladaron
      a mano y un dígito cambiado invalida un argumento.
    - Las figuras: contiguo-artefacto-umbral.png y contiguo-deteccion-sin-umbral.png
      SOLO funcionan como par (IV.C e IV.D), más contiguo-degradacion.png y
      contiguo-volatilidad.png en IV.A. NO usar curva-degradacion.png ni
      volatilidad-crossover.png: son de las familias congeladas, no de este pipeline.

  NOTA para el Abstract: la Conclusión NO repite cifras a propósito. El Abstract
  SÍ tiene que traerlas (los candidatos están en el andamiaje del Abstract), y es
  el único lugar donde el lector las ve antes de la Sección IV.

  Las dos subsecciones que decidían si el paper se lee como honesto o como ingenuo
  —II-A y II-C— acreditan de frente lo que no es nuestro. II-E enuncia el vacío en
  DOS ejes (dominio y tipo de umbral), no en uno: la formulación de un solo eje se
  volvió falsa cuando se leyó Petetin.

  SIGUIENTE: V.A / V.B / V.C (Discusión), después VI Conclusión, después I e
  Introducción y Abstract al final. Las Referencias pueden ir en paralelo.
  Ya no quedan bloqueantes de literatura para nada de eso.

  BLOQUEANTES — RESUELTOS el 2026-07-29, los cuatro papers leídos (ver fuentes-verificadas.md §0):
    V1  Mayer y Yang 2022  → CONFIRMADO. "MSE-optimized forecasts are always
                             underdispersed" es literal. C1 reformulado: queda el CV
                             transversal sobre el vector, la inversión del TCQSM y la
                             consecuencia sobre la regla de evento (ellos: 0 umbrales).
    V2  Rezazada et al.    → VERIFICADO. "20 s a ¼ del programado", y "no existe un
                             único valor de umbral". Pendiente menor: revisar Gong et
                             al. (2020), umbral variable.
    V3  Yu et al. >95 %    → RECONCILIADO. Es a 2 PARADAS; su propia sensibilidad cae
                             a 73 % a 5 paradas (vía Sun et al. 2021). Métricas sin
                             precisión, sin AUC, sin detector trivial. Y su umbral usa
                             el headway OBSERVADO de la primera parada como sustituto
                             del horario ausente — mismo problema nuestro, misma clase
                             de solución, referencia distinta.
    V8  Santos et al. 2022 → SIN BARRIDO DE UMBRAL. C2 sobrevive. Y su tabla de la
                             literatura previa confirma que nadie reporta AUC ni AP.

  PENDIENTE MENOR: Jiao et al. 2023 (89 % con LSTM → umbral), doi 10.1109/icite59717.2023.10733869.
  No conseguido. Probablemente se reconcilia igual que Yu et al. (horizonte corto), pero
  hay que verificarlo antes de someter.
-->
