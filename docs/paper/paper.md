# Compresión de dispersión en la predicción del vector de headways: el punto de operación, y no el modelo, determina la detección de bunching

## Resumen

_(pendiente — se escribe al final)_

---

## I. Introducción

_(pendiente)_

---

## II. Trabajos relacionados

_(A–C pendientes: la receta estándar, por qué el umbral se mueve,
y el precedente de recalibración fuera del transporte.)_

### D. Delimitación de lo previo y de la contribución

Buena parte del mecanismo que este trabajo mide ya está publicada, y delimitar
qué es previo deja a la vista una contribución más angosta que ese mecanismo
completo. Que una predicción optimizada en error cuadrático resulte menos dispersa
que la realidad está enunciado por Mayer y Yang y demostrado como teorema por
Patton y Timmermann. Nada de eso lo reclamamos. Tampoco reclamamos haber sido los
primeros en atar esa compresión a una métrica categórica ni en observar que
empeora con el horizonte: las dos cosas están en Petetin y colaboradores. Que el
paradigma de predecir y umbralizar falle, y que el veredicto se revierta al
puntuar sin punto de operación, lo diagnosticaron Sun, Schmöcker y Nakamura.
Recalcular un umbral contra la distribución de cada modelo es el procedimiento de
Hoffmann, Menz y Spekat en reducción de escala climática, ocho años antes. El
cruce entre la persistencia y un método entrenado al alargar el horizonte ya se
reporta en predicción de tráfico [CITA_REQUERIDA]. Y el resultado nulo de las
variantes espaciales coincide con trabajo previo [CITA_REQUERIDA]. Llegar segundo a una
conclusión no la vuelve propia.

Reclamamos tres contribuciones más angostas. **Primera**, medimos la compresión
sobre el vector de headways, como dispersión entre buses en un mismo instante;
los precedentes trabajan sobre la variabilidad temporal de una serie escalar, que
no es la misma cantidad. **Segunda**, invertimos la fórmula de calidad de
servicio del *Transit Capacity and Quality of Service Manual* (TCQSM) y la
aplicamos a lo predicho en lugar de a lo observado. **Tercera**, y es la que no
encontramos con precedente dentro ni fuera del transporte, la atamos a una regla
de evento **relativa y auto-referencial**, donde la compresión mueve el numerador
y el denominador a la vez. En Petetin y colaboradores esa pieza no falta por
descuido sino por construcción: sus umbrales son regulatorios y no admiten
recalibración.

---

## III. Método propuesto

### A. Construcción del headway a partir de posiciones GPS

El headway es el tiempo que separa el paso de dos buses consecutivos por un mismo
punto: si uno pasa por una esquina y el siguiente llega cinco minutos después, el
headway en esa esquina es de cinco minutos. Es la cantidad que revela si un
corredor mantiene sus buses espaciados o si circulan apelotonados, el fenómeno que
la Sección III-C define como **bunching**. El headway es la variable que
predecimos.

La forma habitual de medir el headway es en una parada, con la lista de paradas de
la ruta y los horarios de paso. Ninguna de las dos existe en este caso: el dato
disponible son coordenadas GPS crudas. Para llegar al headway desde esas
coordenadas se aplicó la secuencia de seis pasos que sigue.

**1) El eje.** El trazado del corredor se estima de los propios buses: se ajusta
una línea central a las posiciones de las unidades en movimiento y después se
suaviza, lo que entrega una curva principal a lo largo del recorrido.

**2) La proyección a una dimensión.** Con el eje ya trazado, cada posición se
reduce a dos números: cuánto ha avanzado el bus a lo largo del corredor y a qué
distancia quedó del eje. Es la operación estándar de referenciación lineal,
proyectar un punto sobre una polilínea:

$$s(p) = \text{arco del punto del eje } C \text{ más cercano a } p,
\qquad \ell(p) = \lVert\, p - C(s(p)) \,\rVert \tag{1}$$

donde $C$ es el eje del corredor parametrizado por su longitud de arco y $p$ es la
posición reportada por el bus. La coordenada $s(p)$ es el arco del punto de $C$
más cercano a $p$, y $\ell(p)$ es la distancia de $p$ a ese punto. La posición se
conserva solo si $\ell(p) \le 300$ m; lo que cae más lejos no pertenece al
corredor.

**3) El sentido de marcha.** Sobre ese mismo eje circulan los buses de ida y los
de vuelta, y el dato no distingue unos de otros. El sentido se define como el
signo del desplazamiento promediado sobre cinco posiciones, de modo que un error
aislado no invierta la dirección:

$$d = \operatorname{sign}\!\big(\overline{\Delta s}_{5}\big) \tag{2}$$

donde $\overline{\Delta s}_{5}$ es el promedio del avance sobre el eje en las
últimas cinco posiciones y $d \in \{-1, 0, +1\}$ es el sentido asignado. La
derivación es forzosa: uno de los corredores no reporta rumbo en absoluto.

**4) Los viajes.** Ya con sentido, el recorrido de cada bus se corta en viajes: un
salto de más de treinta minutos sin señal, una inversión de sentido o una espera
prolongada en terminal cierran el viaje en curso.

**5) La rejilla común.** Los buses no emiten sincronizados entre sí, de modo que
hace falta un instante compartido. Todo se lleva a una rejilla de sesenta
segundos, y así cada minuto queda descrito por una **instantánea** del corredor:
la posición de todos sus buses en ese momento.

**6) El headway.** Sobre esa instantánea, para un par de buses consecutivos en el
mismo sentido —el de adelante *L*, el de atrás *F*— en el instante *T*:

$$t_{c} = \max\{\, t \le T \;:\; s_{L}(t) = s_{F}(T) \,\},
\qquad h = T - t_{c} \tag{3}$$

donde $T$ es el instante evaluado, y $s_{L}$ y $s_{F}$ son las coordenadas de arco
del bus de adelante y del de atrás. El instante $t_{c}$ es el último en que el de
adelante ocupó la posición que el de atrás ocupa en $T$, y $h$ es el headway
resultante. La Ecuación (3) mide **hace cuánto tiempo el bus de adelante pasó por
el punto donde el de atrás está ahora.** Es un cruce por posición y no por parada,
y eso es exactamente lo que permite prescindir de la tabla de paradas. Si no
existe tal $t_{c}$, o si $h$ supera los treinta minutos, se emite «sin dato» en
lugar de arrastrar un paso de horas antes. Con $N$ buses circulando, el corredor
queda descrito en cada minuto por un vector de $N-1$ números, y la Figura 1
ilustra la medición sobre un punto fijo.

![Definición del headway](figuras/headway/headway.png)

**Fig. 1.** El headway, medido en un punto fijo del corredor. El bus de adelante
—Bus 1, el *L* de la Ecuación (3)— pasó por el punto p₂ a las 12:30; el de atrás
—Bus 2, el *F*— llega a ese mismo punto a las 12:35. El headway en p₂ es la
diferencia entre esas dos horas: cinco minutos. La separación espacial entre los
dos buses no interviene. Esquema ilustrativo, no datos reales.

Esta forma de medir el headway no se eligió por comodidad. Se compararon las
cuatro formulaciones siguientes, sobre criterios de cobertura, variabilidad,
autocorrelación, información compartida entre buses vecinos y estabilidad de la
distribución.

**1) Tiempo entre pasadas por puntos virtuales del eje** — sembrar puntos
artificiales a lo largo del corredor y medir el tiempo entre buses sucesivos por
cada uno. Quedó afuera por autocorrelación demasiado baja: el valor de ahora casi
no informaba sobre el de cinco minutos más tarde, que es uno de los horizontes que
hay que predecir.

**2) Tiempo proyectado hacia adelante** — la separación entre los dos buses
dividida por la velocidad del de atrás. Quedó afuera por lo mismo, y arrastra
además una debilidad de forma: dividir por la velocidad actual supone que esa
velocidad se mantiene, de modo que introduce una estimación dentro de la cantidad
que después se quiere estimar.

**3) Distancia en metros entre buses consecutivos** — iguala a la adoptada en
calidad de señal, y se descartó por el objeto de estudio y no por su desempeño.
Mide separación espacial y no tiempo entre pasadas, que es la cantidad que el
operador necesita y la que define el bunching.

**4) Tiempo desde el cruce hacia atrás** — la definición de la Ecuación (3), y la
adoptada.

### B. Formulación de la tarea de predicción

La Sección III-A deja el corredor descrito, minuto a minuto, por un vector de
headways. Lo que predecimos es ese vector completo: no un headway suelto ni un
promedio del corredor, sino todas sus posiciones a la vez. Dado el historial de
los últimos $T$ minutos y un contexto de calendario, se busca el vector del
corredor $H$ minutos más adelante:

$$\hat{\mathbf{h}}(t+H) \;=\; f\big(\mathbf{h}(t-T+1), \dots, \mathbf{h}(t);\; c(t)\big),
\qquad T = 12 \tag{4}$$

donde $\mathbf{h}(t)$ es el vector de headways del corredor en el minuto $t$ y
$\hat{\mathbf{h}}(t+H)$ es el vector predicho para $H$ minutos más adelante.
El término $c(t)$ reúne las variables de calendario disponibles en $t$, $f$ es el
modelo ajustado y $T$ es la cantidad de minutos de historia que recibe.

Se predice a cuatro horizontes —uno, tres, cinco y diez minutos— con un modelo
ajustado por separado para cada uno: no hay recursión, cada horizonte se predice
de forma directa. El vector no tiene longitud fija, porque cuántos headways hay en
un minuto depende de cuántos buses estén circulando. El modelo emite entonces una
salida de longitud fija y el error se computa solo sobre las posiciones donde hay
bus. **El objetivo que se minimiza es el error cuadrático**, promediado sobre esas
posiciones válidas:

$$\mathcal{L} \;=\; \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}
\big(\hat{h}_i - h_i\big)^{2} \tag{5}$$

donde $\mathcal{V}$ es el conjunto de posiciones del vector con bus asignado en el
instante objetivo, $|\mathcal{V}|$ es su cardinal, y $\hat{h}_i$ y $h_i$ son el
valor predicho y el observado en la posición $i$.

Esa elección gobierna el resto del trabajo. Una predicción que minimiza error
cuadrático tiende a la media condicional, que es menos dispersa que la realidad.
La compresión de dispersión que documenta la Sección V-B no es entonces una falla
del ajuste: es lo que este objetivo pide. El efecto de esa compresión sobre la
regla del evento es el asunto de la Sección III-C.

### C. Definición del evento de bunching

El bunching es el fenómeno en que dos o más buses que deberían circular espaciados
terminan viajando casi juntos y dejan un intervalo largo detrás de ellos. Su costo
recae sobre quien espera en ese intervalo: la espera que enfrenta es la que el
intervalo mide, y no el headway promedio del corredor. Sus causas son
heterogéneas, entre ellas la congestión, un día de demanda atípica, la acumulación
de pasajeros en el bus adelantado o las restricciones horarias del conductor
[CITA_REQUERIDA]. Este trabajo no observa ninguna de ellas: el registro disponible
trae identificador, instante y coordenada, y no pasajeros, ocupación ni estado del
tránsito. Por eso el evento se define sobre la geometría del vector de headways,
que sí es observable, y no sobre lo que la produjo.

Dos rasgos del fenómeno gobiernan cómo se lo define aquí. Es una propiedad del
patrón colectivo y no de una unidad: cada bus puede estar donde le corresponde y
el corredor estar apelotonado igual. Y se manifiesta en posiciones del vector de
la Sección III-A, de modo que un mismo instante puede llevar varias posiciones
afectadas a la vez.

Resta decidir cuándo un headway cuenta como bunching. La convención del campo es
una fracción del headway programado, normalmente un cuarto, pero en estos
corredores no hay programación contra la cual comparar. Se sustituye por el
análogo directo: **un headway cuenta como bunching si cae por debajo de la mitad
del promedio de su propio vector en ese instante.** El promedio del propio vector
cumple así la función que cumplía la programación: fijar cuál es la separación
normal en ese corredor en ese instante. Un corte fijo en minutos no la cumple,
porque no es comparable entre corredores que operan a frecuencias distintas. La
elección del valor tampoco es neutral, y conviene declararlo: los umbrales
publicados van desde veinte segundos hasta el cuarto ya mencionado, y no existe un
único valor aceptado.

En notación, sea $\mathbf{h}(t) = (h_1, \dots, h_m)$ el vector de headways del
corredor en el instante $t$. Su promedio y el corte del evento son

$$\bar{h}(t) \;=\; \frac{1}{m}\sum_{j=1}^{m} h_j(t),
\qquad \tau(t) \;=\; \rho\,\bar{h}(t), \qquad \rho = \tfrac{1}{2} \tag{6}$$

donde $m = N - 1$ es la cantidad de posiciones del vector, $N$ es la cantidad de
buses en circulación y $h_j(t)$ es el headway de la posición $j$. El promedio del
vector es $\bar{h}(t)$, el corte del evento es $\tau(t)$ y $\rho$ es la fracción
que lo fija. La posición $i$ cuenta como bunching cuando cae por debajo de ese
corte:

$$b_i(t) \;=\; \mathbb{1}\!\left[\, h_i(t) < \tau(t) \,\right],
\qquad \text{definido solo si } m \ge 3, \tag{7}$$

donde $b_i(t)$ vale 1 si la posición $i$ cuenta como bunching y 0 si no, y
$\mathbb{1}[\cdot]$ es la función indicadora. La condición $m \ge 3$ descarta los
vectores más cortos y exige al menos cuatro buses en circulación. Su razón es que
por debajo de tres posiciones no hay forma que describir: con dos headways hay un
solo intervalo intermedio, así que cualquier medida de irregularidad se reduce a
esa única diferencia. Con tres ya hay patrón —uno colapsado, uno estirado, uno
normal—, y por eso el mínimo se fija ahí y no en dos.

El detector que este trabajo evalúa es esa misma regla aplicada al vector predicho
de la Ecuación (4), con el promedio de ese mismo vector fijando el corte:

$$\hat{b}_i(t) \;=\; \mathbb{1}\!\left[\, \hat{h}_i(t) < \rho\,\bar{\hat{h}}(t)
\,\right] \tag{8}$$

donde $\hat{b}_i(t)$ es la detección emitida sobre la posición $i$ del vector
predicho y $\bar{\hat{h}}(t)$ es el promedio de ese mismo vector predicho. Que el
corte salga del vector predicho y no del observado no es un detalle de
implementación: quien opera un corredor no dispone del observado. Al momento de
decidir solo cuenta con lo predicho, así que puntuar contra el promedio real
mediría algo que nadie puede desplegar.

Como $\tau$ es función del propio vector que se evalúa, y no un número fijo de
minutos, las Ecuaciones (7) y (8) no comparan contra el mismo corte:

$$\tau(\hat{\mathbf{h}}) \;=\; \rho\,\bar{\hat{h}}
\;\neq\; \rho\,\bar{h} \;=\; \tau(\mathbf{h})
\qquad \text{siempre que } \bar{\hat{h}} \neq \bar{h}, \tag{9}$$

donde $\tau(\mathbf{h})$ y $\tau(\hat{\mathbf{h}})$ son los cortes que resultan de
aplicar $\rho$ al vector observado y al vector predicho. Escribir «la mitad del
promedio» en los dos casos no los vuelve el mismo corte. Las Figuras 2 y 3 lo
muestran con el mismo headway de dos minutos.

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
porque bajó de 3,0 a 1,6 minutos cuando el vector se volvió más regular. La
Sección V mide qué ocurre cuando esa diferencia se ignora sobre datos reales.

---

## IV. Diseño experimental

### A. Datos

El trabajo usa los registros de posición de la flota del Sistema Integrado de
Transporte de Arequipa. Cada unidad emite su coordenada **cada 20 segundos**, y la
cadencia es regular: la mediana y el percentil 95 del tiempo entre emisiones
coinciden, de modo que el dato no llega a ráfagas. Esa regularidad es la que
sostiene la rejilla de sesenta segundos de la Sección III-A, porque cada minuto
reúne unas tres emisiones por bus.

Se cubren tres corredores —identificados aquí como E2, E4 y E59, uno por empresa
operadora— durante 152 días seguidos, del 1 de octubre de 2023 al 29 de febrero de
2024, sin huecos de calendario. Son 90 unidades en total. Importa tanto lo que el
dato tiene como lo que no: **no hay horario publicado, no hay archivo GTFS y no
hay tabla de paradas.** Eso obliga a construir todo desde la posición cruda, que
es trabajo extra, pero también es lo que vuelve el método aplicable. La mayoría de
las ciudades donde el bunching es un problema cotidiano son exactamente las que no
tienen ese dato ordenado. Un método que exija GTFS no se puede desplegar donde el
problema es más frecuente.

Construir el headway desde la posición cruda no siempre resulta. Un par de buses
queda sin valor cuando la Ecuación (3) no encuentra el cruce, o cuando el headway
supera los treinta minutos. Medido sobre todos los pares evaluados, dieron headway
válido el 63,5 % en E2, el 64,8 % en E4 y el 77,1 % en E59: casi cuatro millones en
total. El resto quedó **sin dato**. Una posición del vector sin headway válido no 
se rellena con un valor estimado ni con un cero: se enmascara, y el error se computa 
solo sobre las posiciones observadas del vector.

Los huecos que deja ese enmascaramiento no se distribuyen al azar: ninguna de las
dos condiciones es neutral, y ambas recortan por el extremo alto de la
distribución. Los descartados son entonces **los intervalos más largos**, de modo 
que el corredor evaluado excluye parte de su peor servicio. Las cifras de la 
Sección V describen esa población recortada y no la operación completa. La 
cobertura tampoco es pareja: entre el corredor mejor y el peor medido hay casi 
catorce puntos porcentuales. Las comparaciones **entre** corredores no parten, por 
lo tanto, de bases igual de completas.

### B. Métodos comparados

Se comparan cuatro métodos, evaluados todos sobre las mismas muestras. Dos ajustan
parámetros al entrenamiento: una red recurrente con memoria de largo y corto plazo
(**LSTM**) y un conjunto de árboles con refuerzo de gradiente (**XGBoost**). Los
otros dos no ajustan ninguno y sirven de **método de referencia**: fijan el error
que un método de predicción debe bajar para ser útil.

La **persistencia** repite el último vector observado, así que su error crece con
el horizonte: cuanto más lejos se pregunta, más envejecida está la copia. El
**promedio histórico por franja horaria** responde con el valor típico de esa hora
del día, calculado sobre entrenamiento por corredor y sentido. No lee la ventana de
entrada, de modo que su error no depende del horizonte. La persistencia es entonces
el rival a batir a horizonte corto, y el promedio histórico lo es a horizonte
largo. Con los dos dentro, ningún horizonte queda sin rival: un método de
predicción que solo superara a la persistencia a diez minutos podría estar
perdiendo contra una tabla de promedios. La Sección V-A compara el desempeño de los cuatro.

Esos cuatro métodos no se eligieron de una vez. La comparación corrió en dos fases.
La primera evaluó nueve métodos y descartó cinco: tres estadísticos y dos
arquitecturas de aprendizaje profundo. Los cuatro que quedaron son los que esta
sección compara, y la segunda fase los midió bajo los contratos de la Sección
IV-C. El margen de un método es su error absoluto medio menos el del otro con que
se compara, mediano sobre las doce celdas que resultan de cruzar tres corredores y
cuatro horizontes.

Ninguno de los tres estadísticos combina las dos fuentes de la Ecuación (4) —el
historial reciente y el calendario—, y ahí está su límite. La media del período de
entrenamiento no usa ninguna de las dos, y el LSTM le gana en las doce celdas por
una mediana de 0,791 minutos. La media móvil causal en tres ventanas y el suavizado
exponencial simple de factor 0,3 promedian el historial, lo que descarta el valor
más fresco, y no leen el calendario. Quedan apretados por los dos métodos de
referencia: a diez minutos el promedio histórico les gana por 0,597 y 0,658
minutos, y a un minuto la persistencia le gana a la media móvil por 0,962.

Las otras dos descartadas modelan la relación entre posiciones vecinas del vector,
con una convolución sobre el eje de los buses (**SpatialConvLSTM**) y con atención
entre posiciones (**SpatialTransformer**). Sus márgenes contra el LSTM tienen
medianas de 0,004 y 0,027 minutos. Eso queda por debajo de los 0,44 que el error de
un mismo modelo podía moverse en esa fase según cuáles filas le tocara puntuar, así
que el margen no decide. Lo decide que la búsqueda de hiperparámetros, pudiendo
dimensionar el componente espacial, lo dejó en el mínimo de la grilla en tres de las
cuatro combinaciones de arquitectura y corredor: el vecino inmediato no aporta
información que la red plana no tenga ya. Los dos descartes quedaron establecidos
en la primera fase y no se rehicieron en la segunda.

En la segunda fase, el LSTM se ajusta por corredor y por horizonte, lo que da doce
ajustes. Los dos sentidos comparten el modelo de su corredor y entran juntos al
entrenamiento. Lo que se separa por sentido son los estadísticos de
estandarización, de modo que lo predicho se devuelve a minutos con los del
sentido que le corresponde. El XGBoost lee los mismos doce valores,
sobre exactamente las mismas muestras.

| | LSTM | XGBoost |
| :--- | :--- | :--- |
| Entrada | vector de headways de los últimos 12 minutos, rellenado hasta una longitud fija —el percentil 99 de la cantidad de pares de buses en entrenamiento— y estandarizado con estadísticos calculados solo sobre entrenamiento | los mismos 12 valores, como rezagos del headway de la posición evaluada, leídos sobre la rejilla de minutos de la Sección III-A: sin relleno hacia adelante y sin desplazamiento por posición de fila |
| Contexto | seno y coseno de la hora del día y del día de la semana; las cuatro son de calendario, así que en el momento de predecir ninguna depende de lo que va a ocurrir | hora del día, día de la semana, sentido e índice de la posición dentro del vector |
| Capacidad | 32 unidades ocultas; una capa en E2, dos capas con 20 % de apagado aleatorio de unidades en E59 | hasta 400 rondas de refuerzo, con corte tras 30 sin mejora |
| Ajuste | Adam con paso 5 × 10⁻⁴, lotes de 128, hasta 50 pasadas por los datos, corte temprano tras 10 sin mejora, semilla fija en 42 | árboles por histograma, semilla fija en 42 |
| Búsqueda | una configuración heredada en E2 y E59, ganadora de la búsqueda de la primera fase a un minuto de horizonte; tres en E4, elegidas sobre validación | veinticuatro por celda, sorteadas con semilla fija de un espacio de 22 500 combinaciones y elegidas solo con el error de validación |
| Objetivo | el error cuadrático de la Ecuación (5), calculado solo sobre las posiciones donde hay bus | el error cuadrático sobre el headway de la posición evaluada |

El presupuesto de búsqueda no quedó nivelado entre los dos, y la diferencia corre
en contra de la red: su configuración se heredó de la primera fase, mientras que la
del XGBoost se eligió en la segunda. Nivelarlo exigía repetir la
búsqueda completa de la red, y no se hizo. La Sección VI acota qué afirmaciones
sobreviven a esa diferencia y cuáles no.

### C. Protocolo de evaluación

La partición es **por fecha y nunca al azar**, porque un operador solo dispone del
pasado: 107 días de entrenamiento, 23 de validación y 22 de prueba. Para
comprobar que el resultado no depende del mes elegido, todo se repite sobre tres
orígenes que arrancan el mismo día y alargan el entrenamiento —61, 83 y 107
días—, con períodos de prueba que no se solapan entre sí. Como los entrenamientos
están anidados, esto establece estabilidad frente a la elección del período de
prueba, y no réplica independiente; se declara así. La Figura 4 muestra el
esquema.

![Partición temporal y los tres orígenes](figuras/esquema-particion-temporal.es.png)

**Fig. 4.** La partición por tiempo y los tres orígenes de evaluación. Los tres
arrancan el mismo día y alargan el entrenamiento; sus períodos de prueba no se
solapan.

Cuatro reglas más gobiernan la comparación. Cada una descarta un mecanismo
concreto por el cual una ventaja aparente podría no corresponder al modelo que la
reclama.

**Continuidad estricta.** Una muestra solo es válida si los minutos que la
componen son consecutivos de verdad. Sin esa exigencia, una ventana puede saltar
un hueco de señal y un «horizonte de diez minutos» aterrizar horas después.
Cumplirla cuesta datos —sobrevive entre el 81,9 % y el 90,2 % de las
instantáneas— y ese es el precio de que el horizonte signifique lo que dice.

**Población compartida.** Los métodos se puntúan sobre exactamente las mismas
filas. No se declara: se verifica. El trabajo de entrenamiento recalcula la lista
de muestras, compara su huella criptográfica contra la registrada y **aborta antes
de tocar la GPU** si no coincide. Cuando dos métodos se puntúan sobre conjuntos de
filas distintos, la comparación no queda sesgada sino indefinida.

**Tope al 1 % más alto.** El umbral se calcula **solo sobre el entrenamiento** y
se aplica a las tres particiones por igual. Calcularlo sobre cada partición
dejaría entrar información del período de prueba. Afecta entre el 0,78 % y el
1,11 % de los objetivos.

**Varianza agrupada por día de servicio.** Dos minutos del mismo día no son
observaciones independientes. Agrupar por día lleva el tamaño efectivo de muestra
de decenas de miles de filas a 22 días, que es la cifra honesta. Tres veredictos
que parecían significativos no sobreviven a ese cambio, y se reportan como no
significativos.

---

## V. Resultados y discusión

### A. Error escalar y su frontera de régimen

A diez minutos de anticipación, el LSTM predice el headway entre buses
mejor que la persistencia. El error absoluto medio baja 1,47 minutos en E2, 1,38
en E4 y 1,17 en E59: entre 21 % y 22 % en los tres corredores. A un minuto la
relación se invierte y la persistencia gana, por 0,46 minutos en E4 y 0,33 en E59;
en E2 la diferencia es de 0,07 minutos y no resiste la prueba estadística una vez
que se agrupan las observaciones por día de servicio.

Tres precisiones acotan ese resultado. La primera es que el cruce no es una
propiedad del aprendizaje profundo: el XGBoost lo reproduce entero, y a
diez minutos aventaja a la persistencia por 1,59 minutos en E2, 1,09 en E4 y 0,79
en E59. La segunda es que la frontera real no es el horizonte sino la dispersión
de la ventana de entrada. Cada celda se separó en tercios según la dispersión de
los headways que el modelo recibe, con los cortes fijados sobre los datos de
entrenamiento y nunca sobre los de prueba. Así medida, la ventaja del LSTM
crece de forma ordenada del tercio calmo al volátil en 11 de las 12 celdas.
Alargar el horizonte no cambia quién gana: mueve la ventaja hacia tercios cada vez
más tranquilos.

La tercera es que el promedio histórico por franja horaria cumple el papel que la
Sección IV-B le asignaba. Su error no se mueve con el horizonte: se queda entre
4,7 y 5,7 minutos en los tres corredores. La ventaja del LSTM sobre él se
estrecha entonces a medida que el horizonte crece. En E2 el LSTM le gana por
0,99 minutos a un horizonte de un minuto y le pierde por 0,07 a diez. Esa es la
única
de las doce celdas donde el promedio histórico gana, y es la razón de que a
horizonte largo el competidor exigente sea él y no la persistencia. Este eje se
reporta como contexto y no como contribución, porque el cruce entre la persistencia
y un método entrenado al alargar el horizonte ya se reporta en predicción de
tráfico [CITA_REQUERIDA].

### B. Compresión de la dispersión transversal

Un corredor de buses puede describirse, en cada instante, por la irregularidad de
sus headways: la desviación estándar del vector dividida por su promedio. Cero
significa buses perfectamente espaciados; valores altos significan grupos y
huecos. Medida sobre lo observado, esa cifra vale 0,79 en E2. Medida sobre lo que
el modelo predice para el mismo instante y el mismo corredor a diez minutos, vale
0,16. El vector predicho describe un corredor casi cinco veces más regular que el real.

No es un caso aislado. El sesgo, definido como la dispersión predicha menos la
observada, es negativo —lo predicho siempre más regular que la realidad— en
**las 36 celdas** que resultan de cruzar tres corredores, cuatro horizontes y tres
ventanas de prueba. Y se profundiza de forma estrictamente
ordenada a medida que se alarga el horizonte: en E2 pasa de −0,42 a un minuto a
−0,63 a diez. No hay una sola excepción en las seis series de corredor y modelo.

Dos comparaciones acotan de qué depende el efecto. La primera identifica la causa
por descarte: la persistencia no comprime nada. Su sesgo se mantiene dentro de
±0,022 en las 36 celdas, porque propaga el vector observado y hereda su dispersión
sin traducción. Es el control del experimento, y sitúa el efecto en el acto de
**emitir un número por celda**, no en los datos ni en el corredor. La segunda
descarta la arquitectura: el XGBoost comprime igual que la red en E2, y
las dos curvas se superponen. En los otros dos corredores comprime **más** que
ella, con un sesgo de −0,46 contra −0,35 en E59 a diez minutos. Un fenómeno que
aparece
igual en una red recurrente y en un conjunto de árboles no es una propiedad de
ninguna de las dos.

La consecuencia práctica se aprecia al traducir esas cifras a la escala de nivel
de servicio del TCQSM. El mismo corredor, en el mismo instante, califica como
nivel A —«service provided like clockwork»— según lo predicho y como nivel F
—«most vehicles bunched»— según lo observado. Conviene ser preciso sobre qué es
nuevo. El teorema que la Sección II-D reconoce como previo cubre la variabilidad
de una serie a lo largo del tiempo. Lo que estas 36 celdas miden es otra cantidad:
cuán desparejos están los buses **entre sí en un mismo instante**. Son por lo
tanto un resultado empírico y no un corolario. Las Figuras 5 y 6 muestran el
efecto y su dependencia del horizonte.

![Dispersión observada frente a predicha](figuras/compresion-dispersion.es.png)

**Fig. 5.** Dispersión observada frente a dispersión predicha, horizonte de diez
minutos. La barra de la persistencia iguala a la observada: hereda el vector real
y sirve de control. Los dos modelos ajustados la comprimen.

![Sesgo de dispersión contra horizonte](figuras/compresion-vs-horizonte.es.png)

**Fig. 6.** El mismo sesgo contra el horizonte. La persistencia no se despega de
cero; los dos modelos ajustados descienden de forma monótona. La compresión escala con la
distancia que se pide anticipar.

### C. Colapso de la detección al trasladar el umbral

La regla de la Sección III-C, aplicada a lo observado, marca 15 245 eventos en E2
a diez minutos. Aplicada a lo predicho por el LSTM, con el mismo corte,
se dispara **catorce veces**. La persistencia dispara 15 083 veces. Puntuada con
la medida habitual de detección —F1, la media armónica entre precisión y
cobertura—, la persistencia aparece 253 veces mejor que el LSTM. En
las otras celdas el factor va de 1,5 a 36. El XGBoost obtiene un F1
exactamente cero en tres de las doce celdas: ahí no dispara nunca. La Tabla 1
recoge las doce celdas.

Leído sin más contexto, ese resultado dice que el LSTM es incapaz de ver el
fenómeno que se le pidió anticipar. Hay tres motivos para desconfiar de esa
lectura. El primero es que el ganador declarado tampoco detecta bien. Una regla
sin ningún contenido —marcar todas las celdas como bunching— supera a la
persistencia en 5 de las 12 celdas, y en 15 de las 36 al considerar las tres
ventanas. Un procedimiento de evaluación en el que una regla vacía vence al
ganador declarado no ordena modelos.

El segundo es el mecanismo de la Sección V-B. El corte se mide contra el promedio
del propio vector evaluado. Si el vector predicho es más regular que la realidad, sus
headways se apartan menos de su propio promedio, y el corte deja de alcanzarse
casi siempre. Lo que la regla registra no es que el modelo no vea el evento: es
que el modelo no produce la dispersión necesaria para cruzar un umbral calibrado
sobre otra distribución.

El tercero es que, en las pocas ocasiones en que dispara, el modelo acierta. De
los catorce disparos de E2, diez corresponden a eventos de bunching reales: 71 %
de precisión contra una tasa base de 30 %. La muestra es pequeña y el intervalo de
confianza va aproximadamente de 42 % a 92 %, de modo que la cifra señala un
régimen y no un valor. Las celdas con más disparos lo confirman con menos
incertidumbre: en E59 a diez minutos, 776 aciertos en 1 572 disparos contra una
tasa base de 21 %; en E4, 75 en 150 contra 18 %. La cobertura del modelo colapsa;
su precisión, no. Una medida que castiga por igual al que no marca y al que marca
mal no distingue esos dos casos.

![Tasa de disparo contra tasa real del evento](figuras/artefacto-umbral.es.png)

**Fig. 7.** Fracción de celdas que cada método marca como bunching, contra la tasa
real del evento (punteada). La persistencia propaga el vector observado, hereda su
dispersión y el corte cae donde fue diseñado: marca casi tan seguido como el
evento ocurre. La predicción puntual es un vector comprimido, y el mismo corte
relativo le queda en la cola.

**Tabla 1.** Detección con el corte del evento observado aplicado sin cambios a lo predicho, con el piso del detector trivial al lado.

| Corredor | h | Tasa base | Piso trivial | F1 persistencia | F1 LSTM | Factor |
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

### D. Inestabilidad del factor de degradación entre ventanas

Si el factor de 253 de la Sección V-C midiera una capacidad del modelo, debería
ser aproximadamente estable al cambiar la ventana de prueba. No lo es. En la misma
celda —E2, diez minutos, el mismo modelo, la misma regla— el factor vale **2 299**
en la primera ventana, **817** en la segunda y **253** en la tercera. En E2 a cinco
minutos va de 126 a 58 a 36. La magnitud del supuesto fracaso cambia un orden de
magnitud según en qué mes se lo mida.

Ninguna propiedad de un modelo se comporta así. Un número que se mueve un orden de
magnitud entre ventanas contiguas está midiendo la interacción entre el corte y la
distribución sobre la que cayó, no una capacidad del sistema evaluado. Esta es la
observación central del trabajo, y no depende de qué modelo se use ni de qué datos:
depende de que el corte se haya trasladado entre dos distribuciones con dispersión
distinta.

### E. Recalibración del punto de operación

Si el problema es el punto de operación, recalibrarlo debería bastar. El corte se
ajusta sobre una ventana temporal y se aplica a la siguiente, sin mirar nunca los
datos con los que se lo puntúa y sin tocar el modelo. Se fija maximizando la
correlación de Matthews (MCC). La alternativa habitual, maximizar el F1, tiene un
modo de falla que la descarta. Sobre la persistencia en E2, de tres minutos en
adelante, el corte que optimiza el F1 dispara entre el 99,9 % y el 100 % de las
veces. Es decir, reencuentra la regla vacía.

Con el corte recalibrado, el veredicto se invierte. Puntuado sin umbral, mediante
el área bajo la curva ROC (AUC), **el LSTM gana en las nueve
combinaciones de corredor y ventana a diez minutos**, y en 6 de 12 celdas en la
ventana principal. La persistencia conserva la ventaja en el horizonte de un
minuto, donde también ganaba el error escalar. La Tabla 2 reúne los dos
instrumentos.

Esa coincidencia es el resultado, y merece decirse aparte. Puestos en el mismo
eje, el cruce del error escalar y el cruce de la detección van en el mismo sentido
y ocurren en la misma zona de horizontes. Las dos métricas —una continua, la otra
categórica— coinciden en quién gana y desde dónde. La disociación que las
Secciones V-A y V-C parecían mostrar, con el LSTM ganando en error y perdiendo
en detección, no existía: la producía el umbral.

Frente a una falla de detección, la respuesta habitual del campo es cambiar de
modelo: otra arquitectura, más capas, más datos. Ninguna de esas cosas hizo falta.
Las predicciones que puntúa la Figura 8 son, una por una, las mismas que puntúa la
Figura 7. No se reentrenó, no se agregó información y no se modificó el modelo. Se
movió un solo número —dónde se traza el límite entre alarma y silencio— y el
ganador cambió de lado.

De ahí salen las dos consecuencias del trabajo. Para quien evalúa: como ninguna
otra cosa varió, ninguna otra cosa puede explicar la inversión, y el corte queda
identificado como la variable que producía el veredicto. Para quien opera:
reparar esto no cuesta una GPU ni un rediseño, sino recalibrar un umbral con datos
que ya se tienen.

![Ventaja escalar y AUC de detección](figuras/deteccion-sin-umbral.es.png)

**Fig. 8.** Las mismas predicciones puntuadas sin umbral. Eje izquierdo: cuánto
error absoluto le gana el LSTM a la persistencia. Eje derecho: área bajo la
curva de detección, invariante a cualquier reescalado monótono de lo predicho y
por lo tanto inmune al artefacto. Los dos cruces van en el mismo sentido y en la
misma zona, y ninguna serie se acerca al azar.

**Tabla 2.** Veredicto sin umbral y con el corte recalibrado fuera de muestra.

| Corredor | h | AUC LSTM | AUC persist. | MCC recal. LSTM | MCC recal. persist. | Gana AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| E2 | 1 | 0,714 | **0,723** | 0,310 | **0,401** | persistencia |
| E2 | 3 | **0,629** | 0,598 | **0,178** | 0,160 | LSTM |
| E2 | 5 | **0,604** | 0,567 | **0,139** | 0,102 | LSTM |
| E2 | 10 | **0,565** | 0,528 | **0,085** | 0,027 | LSTM |
| E4 | 1 | 0,811 | **0,833** | 0,476 | **0,615** | persistencia |
| E4 | 3 | 0,702 | **0,719** | 0,269 | **0,375** | persistencia |
| E4 | 5 | 0,648 | **0,649** | 0,190 | **0,254** | persistencia |
| E4 | 10 | **0,604** | 0,558 | **0,126** | 0,111 | LSTM |
| E59 | 1 | 0,760 | **0,781** | 0,363 | **0,517** | persistencia |
| E59 | 3 | 0,688 | **0,689** | 0,237 | **0,328** | persistencia |
| E59 | 5 | **0,665** | 0,648 | 0,205 | **0,249** | LSTM |
| E59 | 10 | **0,632** | 0,571 | **0,161** | 0,119 | LSTM |

### F. Robustez frente a la ventana y a la definición del evento

El hallazgo no depende del mes: las tres ventanas temporales coinciden en el
veredicto sin umbral en 11 de 12 celdas, y a diez minutos coinciden en las nueve.
Tampoco depende de la definición del evento adoptada aquí, y la objeción conviene
enfrentarla de frente. El corte relativo a la media del propio vector es una
elección de este trabajo. Un corte absoluto en minutos, como el que usa la mayor
parte de la literatura, podría disolver el efecto. Se probó. **No se atenúa:
empeora.** Bajo la convención dominante del campo, la fracción de eventos que el
modelo efectivamente marca es unas 115 veces menor que bajo la regla relativa. La
elección adoptada resultó ser la conservadora.

Ese mismo ensayo impone un límite que corresponde declarar. Bajo esa convención
más exigente, la capacidad de discriminación del modelo cae: la mediana del área
bajo la curva baja a 0,60, y en E2 a diez minutos llega a 0,49, indistinguible del
azar. La afirmación de que el LSTM no es ciego se sostiene para el evento
definido en términos relativos y falla para el evento absoluto en esa celda. La
Tabla 3 recoge las tres ventanas y el ensayo con el umbral absoluto.

**Tabla 3.** Robustez: las tres ventanas temporales y el ensayo con el umbral
absoluto de la convención dominante.

| Corredor | h | Ventana 1 | Ventana 2 | Ventana 3 | Coinciden | AUC, corte absoluto |
| :--- | ---: | :--- | :--- | :--- | :---: | ---: |
| E2 | 1 | persist. | persist. | persist. | sí | 0,645 |
| E2 | 3 | LSTM | LSTM | LSTM | sí | 0,582 |
| E2 | 5 | LSTM | LSTM | LSTM | sí | 0,550 |
| E2 | 10 | LSTM | LSTM | LSTM | sí | 0,493&nbsp;‡ |
| E4 | 1 | persist. | persist. | persist. | sí | 0,728 |
| E4 | 3 | persist. | persist. | persist. | sí | 0,576 |
| E4 | 5 | persist. | LSTM | persist. | **no** | 0,566 |
| E4 | 10 | LSTM | LSTM | LSTM | sí | 0,551 |
| E59 | 1 | persist. | persist. | persist. | sí | 0,731 |
| E59 | 3 | persist. | persist. | persist. | sí | 0,654 |
| E59 | 5 | LSTM | LSTM | LSTM | sí | 0,637 |
| E59 | 10 | LSTM | LSTM | LSTM | sí | 0,616 |

‡ Indistinguible del azar. Es el único punto donde la afirmación no se sostiene bajo la convención del campo, y se declara como tal.

### G. Implicaciones operativas

El resultado operativo no es que el modelo detecte mejor. Es que **el modelo marca
poco y acierta cuando marca**, y esas son dos propiedades distintas que la métrica
habitual suma en un solo número. Con el corte trasladado, el LSTM marca 14 de
las 50 353 celdas de E2 a diez minutos y acierta el 71 % de las veces que marca,
contra una tasa base del 30 %. En E59 marca más y acierta la mitad, contra un
21 % de base. En los tres corredores la señal, cuando aparece, es entre dos y tres
veces más informativa que el azar.

Eso no es una alarma y no conviene presentarlo como tal. Una alarma tiene que
sonar cuando ocurre el evento, y ésta se queda callada la mayoría de las veces. Lo
que sí constituye es un **filtro de prioridad**: un aviso poco frecuente pero más
informativo que el azar, útil para ordenar la atención de un despachador que
vigila tres corredores y no puede mirar todo a la vez. Y hay una consecuencia
inmediata para cualquiera que hoy esté evaluando una predicción de este tipo: **el
punto de operación se recalibra contra la distribución de lo predicho, no
se hereda de las observaciones.** Requiere recalcular un escalar y no reentrenar
nada.

---

## VI. Amenazas a la validez

Esta sección enuncia el alcance de cada afirmación y los puntos donde no se
sostiene.

**El hallazgo del umbral vale para el evento relativo.** Bajo un corte absoluto en
minutos —la convención dominante— el efecto se agrava, pero la capacidad de
discriminación del LSTM cae, y en E2 a diez minutos llega a 0,49: azar. Ahí la
afirmación de que el LSTM no es ciego no se sostiene.

**Las dos formas de puntuar la detección coinciden en once de doce celdas.** La
excepción es E59 a cinco minutos, donde el LSTM gana el área bajo la curva y
pierde la correlación recalibrada. Ordenar bien y operar bien en un punto fijo son
capacidades distintas, y esa celda las separa.

**El eje escalar tiene un competidor que le gana en una celda.** Frente al
promedio histórico por franja horaria, el LSTM gana en once de doce; pierde en
E2 a diez minutos por 0,07 minutos. Es cuatro segundos y está en el eje que este
trabajo reporta como contexto, pero el número existe y se declara.

**La comparación entre los dos modelos ajustados no está nivelada.** Como se dijo
en la Sección IV-B, el XGBoost recibió veinticuatro configuraciones por celda y el
LSTM una sola en dos corredores. Donde el LSTM pierde, la causa no es atribuible a
la clase de modelo.

**El umbral del evento no está calibrado contra incidentes registrados.** Se
eligió por analogía con la convención del campo, no contra un registro operativo
de eventos de bunching. Validarlo así exige un dato que estos corredores no
producen.

**La dispersión se mide sobre vectores cortos.** Un corredor queda descrito por
entre 3,8 y 5,9 headways por minuto, así que la dispersión transversal es un
estadístico de pocas observaciones. El corte del evento se compara además contra
un promedio que incluye al propio elemento evaluado. El efecto es el mismo en los
tres corredores y en las tres ventanas, lo que hace poco probable que lo produzca
la longitud del vector; pero la precisión de cada cifra individual es menor en E4
y E2, donde el vector es más corto, que en E59.

**La evidencia es de tres corredores de una ciudad y cinco meses**, y el período de
prueba contiene los días de Carnaval, cuya composición no se caracterizó. Los tres
orígenes comparten día de inicio, de modo que establecen estabilidad frente a la
elección del período de prueba y no réplica independiente.

**Lo que este trabajo no afirma** es que estos modelos estén listos para operar una
alarma de bunching. Un área bajo la curva de 0,60 es información real y está muy
lejos de un sistema de despacho. Ninguna función de costo liga aquí un error de
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
