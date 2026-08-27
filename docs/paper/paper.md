# El umbral, no el modelo: por qué un pronóstico de headways parece ciego al bunching, y cómo se repara

## Resumen

_(pendiente — se escribe al final)_

---

## I. Introducción

_(pendiente)_

---

## II. Trabajos relacionados

_(A–C pendientes: la receta estándar, por qué el umbral se mueve,
y el precedente de recalibración fuera del transporte.)_

### D. Qué es previo y qué no

Buena parte del mecanismo que este trabajo mide ya está publicado. Delimitar qué
es previo es lo que deja a la vista la contribución, que es más angosta de lo que
el mecanismo completo sugiere.

Que un pronóstico optimizado en error cuadrático salga más parejo que la realidad
está enunciado por Mayer y Yang y **demostrado como teorema** por Patton y
Timmermann. Nada de eso se reclama acá. Tampoco se reclama haber sido los primeros
en atar esa compresión a una métrica categórica ni en observar que empeora con el
horizonte: las dos cosas están en Petetin y colaboradores. Que el paradigma de
predecir-y-umbralizar falla, y que el veredicto se revierte al puntuar sin punto de
operación, lo diagnosticaron Sun, Schmöcker y Nakamura. Y recalcular un umbral
contra la distribución de cada modelo es, exactamente, el procedimiento de
Hoffmann, Menz y Spekat en reducción de escala climática, ocho años antes. El cruce
entre persistencia y modelo aprendido al alargar el horizonte es folclore conocido
en pronóstico de tráfico, y el resultado nulo de las variantes espaciales confirma
trabajo publicado: llegar segundo a una conclusión no la vuelve propia.

Lo que este trabajo reclama son tres cosas más angostas. **Primera**, medir la
compresión sobre el vector de headways, como dispersión entre buses en un instante —
los precedentes trabajan sobre la variabilidad temporal de una serie escalar, que
no es lo mismo. **Segunda**, dar vuelta la fórmula de calidad de servicio del
manual del oficio y aplicarla al pronóstico en lugar de a lo observado. **Tercera**,
y es la que no tiene precedente dentro ni fuera del transporte, atarlo a una regla
de evento **relativa y auto-referencial**, donde la compresión mueve el numerador y
el denominador a la vez. En Petetin eso no falta por descuido sino por
construcción: sus umbrales son regulatorios y no admiten recalibración.

---

## III. Método propuesto

### A. Del GPS al headway

El headway es el tiempo que separa a dos buses consecutivos: si uno pasa por una
esquina y el siguiente llega cinco minutos después, el headway en esa esquina es de
cinco minutos. Es la cantidad que dice si un corredor va parejo o si sus buses
viajan en pelotón, y es lo que este trabajo pronostica.

La forma habitual de medirlo es en una parada, usando la lista de paradas de la ruta
y los horarios de paso. Acá no existe ninguna de las dos cosas: el dato disponible
son coordenadas GPS crudas.

Entonces, para llegar al headway desde esas coordenadas, se aplicó una serie de
procesos en el siguiente orden.

**1) El eje.** Lo primero es saber por dónde va el corredor, y eso sale de los
propios buses: se ajusta una línea central a las posiciones de los que están en
movimiento y después se suaviza, lo que da una curva principal a lo largo del
recorrido.

**2) La proyección a una dimensión.** Con el eje ya trazado, cada posición *p* se
reduce a dos números: cuánto ha avanzado el bus a lo largo del corredor y a qué
distancia quedó del eje. Es la operación estándar de referenciación lineal,
proyectar un punto sobre una polilínea:

$$s(p) = \text{arco del punto del eje } C \text{ más cercano a } p,
\qquad \ell(p) = \lVert\, p - C(s(p)) \,\rVert \tag{1}$$

La posición se conserva solo si $\ell(p) \le 300$ m; lo que cae más lejos no
pertenece al corredor.

**3) El sentido de marcha.** Sobre ese mismo eje circulan los buses de ida y los de
vuelta, y el dato no dice cuál es cuál. La definición adoptada es el signo del
desplazamiento promediado sobre cinco posiciones, de modo que un error aislado no
invierta la dirección:

$$d = \operatorname{sign}\!\big(\overline{\Delta s}_{5}\big) \tag{2}$$

Se deriva porque no había alternativa: uno de los corredores no reporta rumbo en
absoluto.

**4) Los viajes.** Ya con sentido, el recorrido de cada bus se corta en viajes: un
salto de más de treinta minutos sin señal, una inversión de sentido o una espera
prolongada en terminal cierran el viaje en curso.

**5) La rejilla común.** Los buses no emiten sincronizados entre sí, de modo que
hace falta un instante compartido. Todo se lleva a una rejilla de sesenta segundos,
y así cada minuto queda descrito por una **foto** del corredor: la posición de todos
sus buses en ese momento.

**6) El headway.** Sobre esa foto, para un par de buses consecutivos en el mismo
sentido —el de adelante *L*, el de atrás *F*— en el instante *T*:

$$t_{c} = \max\{\, t \le T \;:\; s_{L}(t) = s_{F}(T) \,\},
\qquad h = T - t_{c} \tag{3}$$

Es decir: **hace cuánto tiempo el bus de adelante pasó por el punto donde el de
atrás está ahora.** Es un cruce por posición y no por parada, y eso es exactamente
lo que permite prescindir de la tabla de paradas. Si no existe tal $t_c$, o si
$h$ supera los treinta minutos, se emite «sin dato» en lugar de arrastrar un paso
de horas antes. Con *N* buses circulando, el corredor queda descrito en cada minuto
por un vector de *N* − 1 números.

![Definición del headway](figuras/headway/headway.png)

**Fig. 1.** El headway, medido en un punto fijo del corredor. El bus de adelante
—Bus 1, el *L* de la Ecuación (3)— pasó por el punto p₂ a las 12:30; el de atrás
—Bus 2, el *F*— llega a ese mismo punto a las 12:35. El headway en p₂ es la
diferencia entre esas dos horas: cinco minutos. La separación espacial entre los dos
buses no interviene. Esquema ilustrativo, no datos reales.

Esta forma de medir el headway no se eligió por comodidad. Se compararon las
siguientes cuatro formulaciones, sobre criterios de cobertura, variabilidad,
autocorrelación, información compartida entre buses vecinos y estabilidad de la
distribución.

**1) Tiempo entre pasadas por puntos virtuales del eje** — sembrar puntos
artificiales a lo largo del corredor y medir el tiempo entre buses sucesivos por
cada uno. Quedó afuera por autocorrelación demasiado baja: el valor de ahora casi
no informaba sobre el de cinco minutos más tarde, que es uno de los horizontes que
hay que predecir.

**2) Tiempo proyectado hacia adelante** — la separación entre los dos buses dividida
por la velocidad del de atrás. Quedó afuera por lo mismo, y arrastra además una
debilidad de forma: dividir por la velocidad actual supone que esa velocidad se
mantiene, de modo que introduce una estimación dentro de la cantidad que después se
quiere estimar.

**3) Distancia en metros entre buses consecutivos** — iguala a la adoptada en
calidad de señal, y se descartó por el objeto de estudio y no por su desempeño:
mide separación espacial y no tiempo entre pasadas, que es la cantidad que el
operador necesita y la que define el bunching.

**4) Tiempo desde el cruce hacia atrás** — la definición de la Ecuación (3), y la
adoptada.

### B. Qué cuenta como bunching

Hay que decidir cuándo un headway cuenta como bunching. La convención del
campo es una fracción del headway programado —normalmente un cuarto—, pero aquí
no hay programación contra la cual comparar. Se sustituye por el análogo directo:
**un headway cuenta como bunching si cae por debajo de la mitad del
promedio de su propio vector en ese instante.** Se exige que el vector tenga al
menos tres headways —o sea cuatro buses en circulación— y los vectores más cortos
se descartan. Con dos
headways hay un solo hueco: cualquier medida de qué tan desparejo está
el corredor se reduce a esa única diferencia, que no describe una forma. Con tres
ya hay patrón —uno colapsado, uno estirado, uno normal—, y por eso el mínimo está
ahí y no en dos.

El promedio del propio vector cumple la función que cumplía la programación: fijar
cuál es la separación normal en ese corredor en ese instante. Un corte fijo en
minutos no la cumple, porque no es comparable entre corredores que corren a
frecuencias distintas. La elección del valor no es neutral y conviene decirlo: los
umbrales publicados van desde veinte segundos hasta un cuarto del headway
programado, y no existe un único valor aceptado.

En notación: sea $h(t) = (h_1, \dots, h_m)$ el vector de headways del corredor en el
instante $t$, con $m = N - 1$ posiciones. Su promedio y el corte del evento son

$$\bar{h}(t) \;=\; \frac{1}{m}\sum_{j=1}^{m} h_j(t),
\qquad \tau(t) \;=\; \rho\,\bar{h}(t), \qquad \rho = \tfrac{1}{2} \tag{4}$$

y la posición $i$ cuenta como bunching cuando cae por debajo de ese corte:

$$b_i(t) \;=\; \mathbb{1}\!\left[\, h_i(t) < \tau(t) \,\right],
\qquad \text{definido solo si } m \ge 3. \tag{5}$$

Nótese que $\tau$ no es un número fijo de minutos: es función del propio vector que
se evalúa. Aplicar la misma regla a lo observado y a un pronóstico $\hat{h}(t)$
produce por eso dos cortes distintos:

$$\tau(\hat{h}) \;=\; \rho\,\bar{\hat{h}}
\;\neq\; \rho\,\bar{h} \;=\; \tau(h)
\qquad \text{siempre que } \bar{\hat{h}} \neq \bar{h}. \tag{6}$$

Escribir «la mitad del promedio» en los dos casos no los vuelve el mismo corte. Los
dos ejemplos siguientes lo muestran con el mismo headway de dos minutos.

![Corredor disparejo](figuras/bunching/with_bunching.png)

**Fig. 2.** Corredor disparejo. El vector es [9,5 · 1,2 · 11,0 · 2,0], su promedio
5,9 min y el corte 3,0 min. Los headways de 2,0 y 1,2 quedan debajo del corte:
**los dos son bunching.**

![Corredor parejo](figuras/bunching/without_bunching.png)

**Fig. 3.** Corredor parejo. El vector es [3,5 · 2,0 · 4,0 · 3,0], su promedio
3,1 min y el corte 1,6 min. El mismo headway de 2,0 min queda ahora encima del
corte: **no es bunching.**

Dos minutos entre buses es el mismo hecho físico en las dos figuras, y la regla lo
clasifica al revés. No cambió el corredor ni cambió la medición: cambió el corte,
porque bajó de 3,0 a 1,6 cuando el vector se volvió más parejo.

De ahí sale la consecuencia que ocupa el resto del trabajo. Aplicado a lo
observado, el corte se calibra sobre la dispersión observada; aplicado a un
pronóstico, se mide contra la dispersión del pronóstico. Si esas dos dispersiones
difieren, no es el mismo corte aunque se escriba igual. La Sección V mide qué
ocurre cuando se ignora esa diferencia.

---

## IV. Diseño experimental

### A. Los datos

El trabajo usa los registros de posición de la flota del Sistema Integrado de
Transporte de Arequipa. Cada unidad emite su coordenada **cada 20 segundos**, y la
cadencia es regular: la mediana y el percentil 95 del tiempo entre emisiones
coinciden en los tres corredores, de modo que el dato no llega a ráfagas. Pero cada
bus emite por su cuenta y sin sincronizarse con los demás, así que dos posiciones
del mismo corredor casi nunca corresponden al mismo instante. De ahí la rejilla
común de la Sección III-A, y de ahí que sea de sesenta segundos: con emisiones cada
veinte, promedia tres por bus sin perder granularidad operativa.

Se cubren tres corredores —identificados aquí como E2, E4 y E59, uno por empresa
operadora— durante 152 días seguidos, del 1 de octubre de 2023 al 29 de febrero de
2024, sin huecos de calendario. Son 90 unidades en total y 43,4 millones de
posiciones crudas.

**El vector es corto, y conviene fijarlo antes de leer cualquier medida de
dispersión.** En promedio, un corredor queda descrito en cada minuto por 3,2
headways en E4, 3,9 en E2 y 6,3 en E59. El mínimo que exige la Sección III-B son
tres, de modo que la restricción no es una salvaguarda ocasional: el vector medio
de E4 está apenas por encima del corte y la regla está actuando casi siempre. Y
toda la dispersión que mide la Sección V se calcula sobre listas de esa longitud,
lo que la vuelve un estadístico ruidoso en los dos corredores cortos y bastante
más firme en E59.

Importa tanto lo que el dato tiene como lo que no. **No hay horario publicado, no
hay archivo GTFS y no hay tabla de paradas.** Eso obliga a construir todo desde la
posición cruda, que es trabajo extra, pero también es lo que vuelve el método
aplicable: la mayoría de las ciudades donde el bunching es un problema
cotidiano son exactamente las que no tienen ese dato ordenado. Un método que exija
GTFS no sirve donde más falta hace.

Aplicado a estos datos, el procedimiento de la Sección III-A deja alrededor de tres
millones de headways válidos en E2 y E59, con una cobertura que va del 57,9 % al
79,7 % según corredor y sentido. Una posición del vector sin headway válido queda
como «sin dato»: no se imputa ni se convierte en cero, y el error se computa solo
sobre las posiciones observadas. La consecuencia hay que decirla: los resultados
describen el corredor **donde el dato existe**, y esa cobertura no es uniforme.

### B. Los métodos comparados

Se comparan cuatro. Una red recurrente con memoria de largo y corto plazo
(**LSTM**), que recibe el vector de headways reciente junto a variables de
calendario y emite el vector completo para el horizonte pedido. Un conjunto de
árboles con refuerzo de gradiente (**XGBoost**), que recibe la misma información en
forma de rezagos y variables de calendario. La **persistencia**, que repite el
último valor observado y es la línea base obligada de todo pronóstico de series de
tiempo. Y el **promedio histórico por franja horaria**, que responde con lo que
suele pasar a esa hora del día.

Los dos últimos no son adorno. Repetir el último valor es una vara exigente a
horizonte corto y se vuelve débil al alargarlo, de modo que por sí sola dejaría al
LSTM compitiendo contra nadie a diez minutos. El promedio histórico cubre
justamente ese flanco: como no depende del horizonte, su error es plano, y a diez
minutos se convierte en el competidor real.

La configuración completa del LSTM es la siguiente.

| | |
| :--- | :--- |
| Entrada | vector de headways de los últimos 12 minutos, rellenado hasta una longitud fija —el percentil 99 de la cantidad de pares de buses en entrenamiento— y estandarizado con estadísticos calculados solo sobre entrenamiento, por corredor y sentido |
| Contexto | seno y coseno de la hora del día y del día de la semana |
| Red | 32 unidades ocultas; una capa en E2, dos capas con 20 % de apagado aleatorio de unidades en E59; en E4 se eligió entre tres configuraciones en validación |
| Ajuste | Adam con paso 5 × 10⁻⁴, lotes de 128, hasta 50 pasadas por los datos, corte temprano tras 10 sin mejora, semilla fija en 42 |
| Objetivo | error cuadrático medio, calculado solo sobre las posiciones donde hay bus |

La última fila gobierna el resto del trabajo. **El objetivo que se minimiza es el
error cuadrático**, y un pronóstico que minimiza error cuadrático tiende a la media
condicional, que es más pareja que la realidad. La compresión que documenta la
Sección V-B no es entonces una falla del ajuste: es lo que este objetivo pide.

Las cuatro variables de contexto son de calendario y nada más: en el momento de
predecir, ninguna depende de lo que va a pasar.

Corresponde declarar una asimetría del procedimiento. El XGBoost se seleccionó
sobre veinticuatro configuraciones por celda; el LSTM heredó una única
configuración en dos de los tres corredores. **Donde el LSTM pierde contra el
XGBoost, ese resultado no es atribuible a la clase de modelo**, y el trabajo no lo
usa como si lo fuera.

El trabajo probó más métodos de los que compara, y ninguno mejoró al LSTM. Del lado
estadístico quedaron afuera la media del período de entrenamiento, una media móvil
causal en ventanas de cinco, diez y quince minutos, y un suavizado exponencial
simple con factor 0,3: el LSTM le gana a las tres en las doce combinaciones de
corredor y horizonte.

Del lado del aprendizaje profundo se probaron dos arquitecturas que modelan
explícitamente la relación entre buses vecinos: una convolución a lo largo del eje
de los buses (**SpatialConvLSTM**) y atención entre las posiciones del vector
(**SpatialTransformer**). Ninguna de las dos supera al LSTM plano en ninguna celda.
Sobre las veinticuatro comparaciones —dos arquitecturas, tres corredores, cuatro
horizontes— **no hay una sola victoria espacial**, y las cuatro celdas donde la
convolución sale nominalmente mejor lo hacen por menos de 0,005 minutos, frente a
un ruido de semilla de ±0,009 medido sobre cinco semillas. El vecino inmediato en
el vector no aporta información que la red plana no tenga ya.

### C. Cómo se evalúa

La partición es **por fecha y nunca al azar**, porque un operador solo dispone del
pasado: 107 días de entrenamiento, 23 de validación y 22 de prueba. Para comprobar
que el resultado no depende del mes elegido, todo se repite sobre tres orígenes que
arrancan el mismo día y alargan el entrenamiento —61, 83 y 107 días—, con períodos
de prueba que no se solapan entre sí. Como los entrenamientos están anidados, esto
establece estabilidad frente a la elección del período de prueba, y no réplica
independiente; se declara así.

![Partición temporal y los tres orígenes](figuras/esquema-particion-temporal.es.png)

**Fig. 4.** La partición por tiempo y los tres orígenes de evaluación. Los tres
arrancan el mismo día y alargan el entrenamiento; sus períodos de prueba no se
solapan.

Cuatro reglas más gobiernan la comparación, y las cuatro existen para cerrar un
camino por el que un número podría entrar sin merecerlo.

**Continuidad estricta.** Una muestra solo es válida si los minutos que la componen
son consecutivos de verdad. Sin esa exigencia, una ventana puede saltar un hueco de
señal y un «horizonte de diez minutos» aterrizar horas después. Cumplirla cuesta
datos —sobrevive entre el 81 % y el 91 % de las fotos— y ese es el precio de que el
horizonte signifique lo que dice.

**Población compartida.** Los métodos se puntúan sobre exactamente las mismas
filas. No se declara: se verifica. El trabajo de entrenamiento recalcula la lista
de muestras, compara su huella criptográfica contra la registrada y **aborta antes
de tocar la GPU** si no coincide. Cuando dos métodos se puntúan sobre conjuntos de
filas distintos, la comparación no queda sesgada sino indefinida.

**Tope al 1 % más alto.** El umbral se calcula **solo sobre el entrenamiento** y se
aplica a las tres particiones por igual. Calcularlo sobre cada partición dejaría
entrar información del período de prueba. Afecta entre el 0,78 % y el 1,11 % de los
objetivos.

**Varianza agrupada por día de servicio.** Dos minutos del mismo día no son
observaciones independientes. Agrupar por día lleva el tamaño efectivo de muestra
de decenas de miles de filas a 22 días, que es la cifra honesta. Tres veredictos
que parecían significativos no sobreviven a ese cambio, y se reportan como no
significativos.


---

## V. Resultados y discusión

### A. El resultado escalar y su frontera

A diez minutos de anticipación, el modelo aprendido predice el headway entre
buses mejor que repetir el último valor observado. El error absoluto medio baja
1,47 minutos en E2, 1,38 en E4 y 1,17 en E59: entre 21 % y 22 % en los tres
corredores. A un minuto la relación se invierte y repetir el último valor gana,
por 0,46 minutos en E4 y 0,33 en E59; en E2 la diferencia es de 0,07 minutos y no
resiste la prueba estadística una vez que se agrupan las observaciones por día de
servicio.

Tres precisiones acotan ese resultado. La primera es que el cruce no es una
propiedad del aprendizaje profundo: el modelo de árboles lo reproduce entero, y a
diez minutos aventaja a la repetición por 1,59 minutos en E2, 1,09 en E4 y 0,79
en E59. La segunda es que la frontera real no es el horizonte sino qué tan movida
viene la ventana de entrada. Separando cada celda en tercios según la dispersión
de los headways que el modelo recibe —y fijando los cortes sobre los datos de
entrenamiento, nunca sobre los de prueba—, la ventaja del aprendiz crece de forma
ordenada del tercio calmo al volátil en 11 de las 12 celdas. Alargar el horizonte
no cambia quién gana: mueve la ventaja hacia tercios cada vez más tranquilos.

La tercera es que el promedio histórico por franja horaria cumple el papel que la
Sección IV-B le asignaba. Su error no se mueve con el horizonte —se queda entre 4,7
y 5,7 minutos en los tres corredores—, así que la ventaja del aprendiz sobre él se
estrecha a medida que el horizonte crece: en E2 el aprendiz le gana por 0,99
minutos a un horizonte de un minuto y le pierde por 0,07 a diez. Esa es la única
de las doce celdas donde el promedio histórico gana, y es la razón de que a
horizonte largo el competidor exigente sea él y no la repetición.

Este eje se reporta como contexto y no como contribución. El cruce entre
persistencia y modelo aprendido al alargar el horizonte es conocido en pronóstico
de tráfico. Lo que sigue es el objeto del trabajo.

### B. El pronóstico sale más parejo que la realidad

Un corredor de buses puede describirse, en cada instante, por lo disparejos que
están sus headways: la desviación de los headways dividida por su promedio.
Cero significa buses perfectamente espaciados; valores altos significan pelotones
y huecos.

Medida sobre lo observado, esa cifra vale 0,79 en E2. Medida sobre lo que el
modelo predice para el mismo instante y el mismo corredor a diez minutos, vale
0,16. El pronóstico describe un corredor casi cinco veces más ordenado que el
real.

No es un caso aislado. El sesgo es negativo —el pronóstico siempre más parejo que
la realidad— en **las 36 celdas** que resultan de cruzar tres corredores, cuatro
horizontes y tres ventanas de prueba, y se profundiza de forma estrictamente
ordenada a medida que se alarga el horizonte: en E2 pasa de −0,42 a un minuto a
−0,63 a diez, sin una sola excepción en las seis series de corredor y modelo.

Dos comparaciones acotan de qué depende el efecto. La primera identifica la causa
por descarte: repetir el último valor observado no aplana nada. Su sesgo se queda
dentro de ±0,02 en las 36 celdas, porque propaga el vector observado y hereda su
dispersión tal cual. Es el control del experimento, y sitúa el efecto en el acto
de **emitir un número por celda**, no en los datos ni en el corredor. La segunda
descarta la arquitectura: el modelo de árboles aplana igual que la red en E2
—las dos curvas se superponen— y **más** que ella en los otros dos corredores, con
un sesgo de −0,46 contra −0,35 en E59 a diez minutos. Un fenómeno que aparece
igual en una red recurrente y en un conjunto de árboles no es una propiedad de
ninguna de las dos.

La consecuencia práctica se ve mejor traduciendo esas cifras a la escala de
calidad de servicio que usa el manual del oficio. El mismo corredor, en el mismo
instante, califica como *servicio de reloj* según el pronóstico y como *casi todos
los buses apelotonados* según lo observado.

Conviene ser preciso sobre qué es nuevo acá. El teorema que la Sección II-D
reconoce como previo cubre la variabilidad de una serie a lo largo del tiempo. Lo
que estas 36 celdas miden es otra cosa: cuán disparejos están los buses **entre sí
en un mismo instante**. Son por lo tanto un resultado empírico y no un corolario.

![Dispersión observada frente a predicha](figuras/compresion-dispersion.es.png)

**Fig. 5.** Dispersión observada frente a dispersión predicha, horizonte de diez
minutos. La barra de la persistencia iguala a la observada: hereda el vector real
y sirve de control. Los dos aprendices lo aplanan.

![Sesgo de dispersión contra horizonte](figuras/compresion-vs-horizonte.es.png)

**Fig. 6.** El mismo sesgo contra el horizonte. La persistencia no se despega de
cero; los dos aprendices se hunden de forma monótona. La compresión escala con la
distancia que se pide anticipar.

### C. La alarma no suena

La regla de la Sección III-B, aplicada a lo observado, marca 15 245 eventos en E2 a
diez minutos. Aplicada al pronóstico del modelo aprendido, con el mismo corte, se
dispara **catorce veces**.

Repetir el último valor observado dispara 15 083 veces. Puntuada con la medida
habitual de detección, la repetición aparece 253 veces mejor que el modelo. En las
otras celdas el factor va de 1,5 a 36. En los tres casos donde el modelo de árboles
se evalúa así, su puntaje es exactamente cero: no dispara nunca.

Leído sin más contexto, ese resultado dice que el aprendiz es incapaz de ver el
fenómeno que se le pidió anticipar. Hay tres motivos para desconfiar de esa
lectura.

El primero es que el ganador declarado tampoco es bueno. Una regla sin ningún
contenido —marcar todas las celdas como bunching— supera a la repetición en
5 de las 12 celdas, y en 15 de las 36 al considerar las tres ventanas. Un
procedimiento de evaluación en el que una regla vacía vence al ganador no está
ordenando modelos.

El segundo es el mecanismo de la sección anterior. El corte se mide contra el
promedio del propio vector evaluado. Si el pronóstico es más parejo que la realidad,
sus headways se apartan menos de su propio promedio, y el corte deja de alcanzarse
casi siempre. Lo que la regla registra no es que el modelo no vea el evento: es que
el modelo no produce la dispersión necesaria para cruzar un umbral calibrado sobre
otra distribución.

El tercero es que, en esas pocas ocasiones en que sí dispara, el modelo acierta.
De los catorce disparos de E2, diez corresponden a eventos de bunching reales: 71 %
de precisión contra una tasa base de 30 %. La muestra es pequeña y el intervalo de
confianza va aproximadamente de 42 % a 92 %, de modo que la cifra señala un
régimen y no un valor. Las celdas con más disparos lo confirman con menos
incertidumbre: en E59 a diez minutos, 776 aciertos en 1 572 disparos contra una
tasa base de 21 %; en E4, 75 en 150 contra 18 %. El modelo no se equivoca: está
callado. Su cobertura colapsa; su precisión, no. Una medida que castiga por igual
al que calla y al que se equivoca no distingue esos dos casos.

![Tasa de disparo contra tasa real del evento](figuras/artefacto-umbral.es.png)

**Fig. 7.** Fracción de celdas que cada método marca como bunching, contra
la tasa real del evento (punteada). La persistencia propaga el vector observado,
hereda su dispersión y el corte cae donde fue diseñado: marca casi tan seguido
como el evento ocurre. El pronóstico puntual emite un vector comprimido, y el
mismo corte relativo le queda en la cola.

**Tabla 1.** Detección con el corte trasplantado, con el piso del detector trivial
al lado.

| Corredor | h | Tasa base | Piso trivial | F1 persistencia | F1 aprendiz | Factor |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2 | 1 | 0,299 | 0,460 | 0,581 | 0,207 | 2,8× |
| E2 | 3 | 0,301 | 0,462 | 0,414&nbsp;† | 0,038 | 11× |
| E2 | 5 | 0,300 | 0,462 | 0,375&nbsp;† | 0,011 | 36× |
| E2 | 10 | 0,303 | 0,465 | 0,332&nbsp;† | 0,001 | 253× |
| E4 | 1 | 0,183 | 0,310 | 0,686 | 0,466 | 1,5× |
| E4 | 3 | 0,173 | 0,295 | 0,486 | 0,177 | 2,7× |
| E4 | 5 | 0,172 | 0,294 | 0,381 | 0,066 | 5,8× |
| E4 | 10 | 0,179 | 0,304 | 0,268&nbsp;† | 0,015 | 18× |
| E59 | 1 | 0,212 | 0,350 | 0,620 | 0,308 | 2,0× |
| E59 | 3 | 0,209 | 0,345 | 0,469 | 0,130 | 3,6× |
| E59 | 5 | 0,208 | 0,344 | 0,405 | 0,083 | 4,9× |
| E59 | 10 | 0,208 | 0,344 | 0,303&nbsp;† | 0,034 | 8,8× |

† La regla vacía —marcar todas las celdas— supera al ganador declarado en estas celdas.

### D. Ese factor no mide al modelo

Si el factor de 253 de la sección anterior midiera una capacidad del modelo,
debería ser aproximadamente estable al cambiar la ventana de prueba. No lo es.

En la misma celda —E2, diez minutos, el mismo modelo, la misma regla— el factor
vale **2 299** en la primera ventana, **817** en la segunda y **253** en la tercera.
En E2 a cinco minutos va de 126 a 58 a 36. La magnitud del supuesto fracaso cambia
un orden de magnitud según en qué mes se lo mida.

Ninguna propiedad de un modelo se comporta así. Un número que se mueve un orden de
magnitud entre ventanas contiguas está midiendo la interacción entre el corte y la
distribución sobre la que cayó, no una capacidad del sistema evaluado. Esta es la
observación central del trabajo, y no depende de qué modelo se use ni de qué datos:
depende de que el corte se haya trasladado entre dos distribuciones con dispersión
distinta.

### E. La reparación: mover la regla, no el modelo

Si el problema es el punto de operación, entonces debería bastar con recalibrarlo.

El corte se ajusta sobre una ventana temporal y se aplica a la siguiente, sin mirar
nunca los datos con los que se lo puntúa y sin tocar el modelo. Se fija maximizando
la correlación de Matthews. La alternativa habitual —maximizar la
medida de detección usual— tiene un modo de falla que la descarta: sobre la
repetición en E2, de tres minutos en adelante, el corte que la optimiza dispara
entre el 99,9 % y el 100 % de las veces. Es decir, reencuentra la regla vacía.

Con el corte recalibrado, el veredicto se invierte. Puntuado sin umbral, mediante
el área bajo la curva, **el modelo aprendido gana en las nueve combinaciones de
corredor y ventana a diez minutos**, y en 6 de 12 celdas en la ventana principal.
La repetición conserva la ventaja en el horizonte de un minuto, donde también
ganaba el error escalar.

Esa coincidencia es el resultado, y merece decirse aparte. Puestos en el mismo
eje, el cruce del error escalar y el cruce de la detección van en el mismo sentido
y ocurren en la misma zona de horizontes. Las dos métricas —una continua, la otra
categórica— coinciden en quién gana y desde dónde. La disociación que las Secciones
V-A y V-C parecían mostrar —el aprendiz ganando en error y perdiendo en detección—
no existía. La fabricaba el umbral.

Frente a una falla de detección, el campo cambia de modelo: otra arquitectura,
más capas, más datos. Acá no hizo falta ninguna de esas cosas. Las predicciones
que puntúa la Fig. 8 son, una por una, las mismas que puntúa la Fig. 7.
No se reentrenó, no se agregó información y no se tocó una línea del modelo. Se
movió un número —dónde se traza la raya entre alarma y silencio— y el ganador
cambió de bando.

De ahí salen las dos consecuencias del trabajo. Para quien evalúa: como ninguna
otra cosa varió, ninguna otra cosa puede explicar la inversión, y el corte queda
identificado como la variable que producía el veredicto. Para quien opera:
reparar esto no cuesta una GPU ni un rediseño. Cuesta recalibrar un umbral con
datos que ya se tienen.

![Ventaja escalar y AUC de detección](figuras/deteccion-sin-umbral.es.png)

**Fig. 8.** Las mismas predicciones puntuadas sin umbral. Eje izquierdo: cuánto
error absoluto le gana el aprendiz a la persistencia. Eje derecho: área bajo la
curva de detección, invariante a cualquier reescalado monótono del pronóstico y
por lo tanto inmune al artefacto. Los dos cruces van en el mismo sentido y en la
misma zona, y ninguna serie se acerca al azar.

**Tabla 2.** Veredicto sin umbral y con el corte recalibrado fuera de muestra.

| Corredor | h | AUC aprendiz | AUC persist. | MCC recal. aprendiz | MCC recal. persist. | Gana AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| E2 | 1 | 0,714 | **0,723** | 0,310 | **0,401** | persistencia |
| E2 | 3 | **0,629** | 0,598 | **0,178** | 0,160 | aprendiz |
| E2 | 5 | **0,604** | 0,567 | **0,139** | 0,102 | aprendiz |
| E2 | 10 | **0,565** | 0,528 | **0,085** | 0,027 | aprendiz |
| E4 | 1 | 0,811 | **0,833** | 0,476 | **0,615** | persistencia |
| E4 | 3 | 0,702 | **0,719** | 0,269 | **0,375** | persistencia |
| E4 | 5 | 0,648 | **0,649** | 0,190 | **0,254** | persistencia |
| E4 | 10 | **0,604** | 0,558 | **0,126** | 0,111 | aprendiz |
| E59 | 1 | 0,760 | **0,781** | 0,363 | **0,517** | persistencia |
| E59 | 3 | 0,688 | **0,689** | 0,237 | **0,328** | persistencia |
| E59 | 5 | **0,665** | 0,648 | 0,205 | **0,249** | aprendiz |
| E59 | 10 | **0,632** | 0,571 | **0,161** | 0,119 | aprendiz |

### F. Robustez, incluido el ataque más duro encontrado

El hallazgo no depende del mes: las tres ventanas temporales coinciden en el
veredicto sin umbral en 11 de 12 celdas, y a diez minutos coinciden en las nueve.

Tampoco depende de la definición del evento adoptada acá, y conviene enfrentar la
objeción de frente: el corte relativo —media del propio vector— es una elección de
este trabajo, y un corte absoluto en minutos, como el que usa la mayor parte de la
literatura, podría disolver el efecto. Se probó. **No se atenúa: empeora.** Bajo la
convención dominante del campo, la fracción de eventos que el modelo efectivamente
marca es unas 115 veces menor que bajo la regla relativa. La elección adoptada
resultó ser la conservadora.

Ese mismo ensayo impone un límite que corresponde declarar. Bajo esa convención más
exigente, la capacidad de discriminación del modelo cae: la mediana del área bajo la
curva baja a 0,60, y en E2 a diez minutos llega a 0,49, indistinguible del azar. La
afirmación *el aprendiz no es ciego* se sostiene para el evento definido en términos
relativos y falla para el evento absoluto en esa celda. El alcance se enuncia
completo o no se enuncia.

**Tabla 3.** Robustez: las tres ventanas temporales y el ensayo con el umbral
absoluto de la convención dominante.

| Corredor | h | Ventana 1 | Ventana 2 | Ventana 3 | Coinciden | AUC, corte absoluto |
| :--- | ---: | :--- | :--- | :--- | :---: | ---: |
| E2 | 1 | persist. | persist. | persist. | sí | 0,645 |
| E2 | 3 | aprendiz | aprendiz | aprendiz | sí | 0,582 |
| E2 | 5 | aprendiz | aprendiz | aprendiz | sí | 0,550 |
| E2 | 10 | aprendiz | aprendiz | aprendiz | sí | 0,493&nbsp;‡ |
| E4 | 1 | persist. | persist. | persist. | sí | 0,728 |
| E4 | 3 | persist. | persist. | persist. | sí | 0,576 |
| E4 | 5 | persist. | aprendiz | persist. | **no** | 0,566 |
| E4 | 10 | aprendiz | aprendiz | aprendiz | sí | 0,551 |
| E59 | 1 | persist. | persist. | persist. | sí | 0,731 |
| E59 | 3 | persist. | persist. | persist. | sí | 0,654 |
| E59 | 5 | aprendiz | aprendiz | aprendiz | sí | 0,637 |
| E59 | 10 | aprendiz | aprendiz | aprendiz | sí | 0,616 |

‡ Indistinguible del azar. Es el único punto donde la afirmación no se sostiene bajo la convención del campo, y se declara como tal.

### G. Qué significa para quien opera un corredor

El resultado operativo no es que el modelo detecte mejor. Es que **el modelo calla
mucho y acierta cuando habla**, y esas son dos cosas distintas que la métrica
habitual suma en un solo número.

Con el corte trasladado, el aprendiz marca el 0,1 % de las celdas en E2 a diez
minutos y acierta el 71 % de las veces que marca, contra una tasa base del 30 %.
En E59 marca más y acierta la mitad, contra un 21 % de base. En los tres corredores
la señal, cuando aparece, es entre dos y tres veces más informativa que el azar.

Eso no es una alarma y no conviene venderlo como tal. Una alarma tiene que sonar
cuando ocurre el evento, y ésta se queda callada la mayoría de las veces. Lo que sí
es, es un **filtro de prioridad**: un despachador que vigila tres corredores no
puede mirar todo a la vez, y un aviso que acierta la mitad de las veces que habla
merece que se lo mire, siempre que se acepte de antemano que no va a hablar en la
mayoría de los casos.

Y hay una consecuencia inmediata para cualquiera que hoy esté evaluando un
pronóstico de este tipo: **el punto de operación se recalibra contra la
distribución del propio pronóstico, no se hereda de las observaciones.** Es una
línea de código y no requiere reentrenar nada.

---

## VI. Amenazas a la validez

El alcance de cada afirmación se enuncia completo.

**El hallazgo del umbral vale para el evento relativo.** Bajo un corte absoluto en
minutos —la convención dominante— el efecto se agrava, pero la capacidad de
discriminación del aprendiz cae, y en E2 a diez minutos llega a 0,49: azar. Ahí la
afirmación *el aprendiz no es ciego* no se sostiene.

**Las dos formas de puntuar la detección coinciden en once de doce celdas.** La
excepción es E59 a cinco minutos, donde el aprendiz gana el área bajo la curva y
pierde la correlación recalibrada. Ordenar bien y operar bien en un punto fijo son
capacidades distintas, y esa celda las separa.

**El eje escalar tiene un competidor que le gana en una celda.** Frente al promedio
histórico por franja horaria, el aprendiz gana en once de doce; pierde en E2 a diez
minutos por 0,07 minutos. Es cuatro segundos y está en el eje que este trabajo
reporta como contexto, pero el número existe y se declara.

**La comparación entre los dos aprendices no está nivelada.** Como se dijo en la
Sección IV, el conjunto de árboles recibió veinticuatro configuraciones por celda
y la red una sola en dos corredores. Donde la red pierde, la causa no es
atribuible a la clase de modelo.

**El umbral del evento no está calibrado contra incidentes registrados.** Se eligió
por analogía con la convención del campo, no contra un registro operativo de
eventos de bunching. Validarlo así exige un dato que estos corredores no producen.

**La dispersión se mide sobre vectores cortos.** Un corredor queda descrito por
entre 3,2 y 6,3 headways por minuto según el corredor, así que la dispersión
transversal es un estadístico de pocas observaciones, y el corte del evento se
compara contra un promedio que incluye al propio elemento evaluado. El efecto es el
mismo en los tres corredores y en las tres ventanas, lo que hace poco probable que
lo produzca la longitud del vector; pero la precisión de cada cifra individual es
menor en E4 y E2, donde el vector es más corto, que en E59.

**La evidencia es de tres corredores de una ciudad y cinco meses**, y el período de
prueba contiene los días de Carnaval, cuya composición no se caracterizó. Los tres
orígenes comparten día de inicio, de modo que establecen estabilidad frente a la
elección del período de prueba y no réplica independiente.

**Y lo que este trabajo no afirma:** que estos modelos estén listos para operar una
alarma de bunching. Un área bajo la curva de 0,60 es información real y está
muy lejos de un sistema de despacho. Ninguna función de costo liga aquí un error de
1,47 minutos, ni un área de 0,60, a una decisión de intervención concreta. Cerrar
esa distancia es trabajo por hacer, no resultado logrado.

---

## VII. Conclusión

_(pendiente)_

---

## VIII. Declaraciones

_(pendiente — disponibilidad de datos y código)_

---

## Referencias

_(pendiente)_
