# Compresión de dispersión en la predicción del vector de headways: el punto de operación, y no el modelo, determina la detección de bunching

## Resumen

_(pendiente — se escribe al final)_

---

## I. Introducción

_(pendiente)_

Reclamamos tres contribuciones. Este trabajo predice el vector completo de
headways de un corredor con un LSTM. La regla de la Sección III-C convierte lo
predicho en un indicador de bunching, y la evaluación puntúa esa detección con y
sin umbral. La Sección II-D delimita cuánto del mecanismo que este trabajo mide ya
estaba publicado.

- **Primera.** Medimos la compresión sobre el vector de headways, como dispersión
  entre buses en un mismo instante. Los precedentes trabajan sobre la variabilidad
  temporal de una serie escalar, que no es la misma cantidad.
- **Segunda.** Invertimos la fórmula de calidad de servicio del *Transit Capacity
  and Quality of Service Manual* (TCQSM) y la aplicamos a lo predicho en lugar de
  a lo observado.
- **Tercera.** La atamos a una regla de evento **relativa y auto-referencial**,
  donde la compresión mueve el numerador y el denominador a la vez. Es la que no
  encontramos con precedente dentro ni fuera del transporte: en Petetin y
  colaboradores esa pieza no falta por descuido sino por construcción, porque sus
  umbrales son regulatorios y no admiten recalibración.

---

## II. Trabajos relacionados

Esta sección recorre la literatura en cuatro pasos. El primero describe el método
con que el campo predice el bunching y las medidas con que lo evalúa. El segundo
reúne lo que ya se estableció sobre la compresión de la dispersión de un
pronóstico. El tercero recoge los dos remedios publicados para esa compresión,
ambos fuera del transporte. El cuarto delimita cuánto del mecanismo que este
documento mide ya estaba publicado.

### A. Predicción del headway y detección por umbral

El bunching se predice en dos etapas. La primera estima el headway que separará a
dos buses en un instante futuro. La segunda compara ese valor contra una
referencia y emite un indicador binario del evento. Yu y colaboradores dan la
formulación canónica de esa secuencia sobre datos de tarjeta inteligente de dos
rutas de Pekín: la ocurrencia del bunching se detecta umbralizando el headway
predicho contra el horario programado [@yu2016]. Las dos etapas optimizan
objetivos distintos, porque la primera minimiza un error en minutos y la segunda
decide una clase.

Jiao, Shen y Zhang repiten esa secuencia sobre una ruta de Xiangyang y fijan el
umbral en un cuarto del headway observado en la primera parada [@jiao2023]. Su
modelo no minimiza solo el error en minutos: la pérdida suma al error cuadrático
un término de clasificación, y el entrenamiento sobremuestrea la clase del
evento, que reúne el 6,1 % de las muestras. Justifican ese diseño advirtiendo que
una pérdida atenta solo al error de regresión lleva al modelo a tratar como ruido
los casos que la regla marca como evento.

La segunda etapa se evalúa en un punto de operación único. Yu y colaboradores
reportan exactitud, sensibilidad y especificidad [@yu2016]. Santos y colaboradores
resumen siete trabajos previos del subcampo en una tabla y agregan el suyo
[@santos2022]. Las medidas que esa tabla registra son de dos clases: errores
continuos, como el error cuadrático medio, y conteos sobre la clasificación, como
la exactitud, la precisión y el recall. Ninguna de sus ocho filas registra una
medida que puntúe el ordenamiento del pronóstico sin fijar antes un umbral.

La primera etapa tiene además un margen angosto donde más importa. Manibardo, Laña
y Del Ser equiparan la persistencia con repetir el último valor observado, y
reportan que su desempeño a horizontes cortos deja poco espacio de mejora a los
modelos entrenados [@manibardo2022]. Su afirmación sobre el horizonte es que todos
los modelos se degradan al alargarlo, y no que la relación entre ellos se
invierta.

### B. Compresión de la dispersión del pronóstico

La primera etapa de esa receta arrastra una propiedad conocida. Un pronóstico
ajustado para minimizar el error cuadrático sale menos disperso que la cantidad
que predice. Mayer y Yang lo miden sobre irradiancia solar: sus pronósticos
optimizados de ese modo capturan menos del 75 % de la varianza observada
[@mayer2023]. Y señalan la consecuencia sobre la comparación entre métodos: como
la raíz del error cuadrático medio premia justamente al pronóstico de menor
dispersión, evaluar con ella sobrevalora al más comprimido.

La propiedad es un teorema y no una regularidad empírica. Patton y Timmermann
descomponen la varianza del objetivo en la del pronóstico óptimo más el error
cuadrático esperado, y su Corolario 2 ordena esas varianzas por horizonte: la del
pronóstico a horizonte corto es mayor o igual que la del pronóstico a horizonte
largo [@patton2012]. La compresión crece entonces al alargar el horizonte, por
construcción y no por una falla del ajuste. Ese resultado recae sobre la varianza
temporal de una serie escalar, y no sobre la dispersión entre unidades medidas en
un mismo instante.

El daño de esa compresión sobre una regla de umbral ya se documentó fuera del
transporte. Petetin y colaboradores corrigen pronósticos de ozono y encuentran
que el método con mejor error cuadrático y mejor correlación es el que peor
detecta los episodios altos, porque subestima la variabilidad [@petetin2022].
Todas sus métricas categóricas se degradan además al alargar el horizonte, de
modo que el efecto sobre la decisión sigue al efecto sobre la dispersión.

### C. Recalibrar el umbral: precedente fuera del transporte

El efecto de la compresión sobre una regla de umbral tiene dos remedios
publicados fuera del transporte, y se distinguen por qué objeto tocan. El primero
mueve el umbral. Hoffmann, Menz y Spekat trabajan con indicadores climáticos
definidos por un valor fijo, como los días con temperatura máxima sobre 30 °C.
Cada modelo climático reproduce ese indicador con un sesgo propio. Su
procedimiento localiza el percentil que ese valor ocupa
en los datos de referencia, calcula el valor de ese mismo percentil en cada
simulación y recalcula el indicador con el umbral así ajustado, sin tocar los
datos del modelo [@hoffmann2018].

El segundo mueve el pronóstico. Petetin y colaboradores corrigen pronósticos de
ozono cuyos umbrales están fijados por normativa y no admiten ajuste. Su mapeo de
cuantiles lleva la distribución de lo predicho a la de lo observado
[@petetin2022]. Los dos remedios piden insumos distintos. El mapeo de cuantiles
necesita una distribución de observaciones de referencia. Recalibrar el umbral
necesita solo una ventana anterior del propio pronóstico.

Ninguno de los dos se enfrenta a un umbral que se mueva con lo que evalúa. El de
Hoffmann y colaboradores es un valor fijo, y el de Petetin y colaboradores es
regulatorio. Ellos mismos observan que un indicador definido sobre un cuantil de
la distribución de referencia queda libre de sesgo por construcción
[@hoffmann2018]. Un umbral que es una fracción del promedio de lo predicho no
tiene esa propiedad, porque la compresión mueve el promedio y la separación entre
posiciones a la vez.

### D. Delimitación de lo previo

Cinco trabajos llegan cerca del mecanismo que este documento mide, y ninguno cubre
el caso que lo define. Los tres de la Sección II-B establecen la compresión y su
daño sobre una regla de umbral, pero ninguno la mide entre unidades de un mismo
instante. Los dos remedios de la Sección II-C llegan ocho años antes que este
trabajo, y ninguno se aplica sobre un umbral que se mueva con lo que evalúa.

Dentro del transporte el precedente más cercano es Sun, Schmöcker y Nakamura
[@sun2021]. Diagnostican que el paradigma de predecir y umbralizar falla, y que el veredicto
se revierte al puntuar sin punto de operación. Su etiqueta es un umbral absoluto de
un minuto y no una regla relativa al propio vector, y su remedio es cambiar de
clase de modelo, no recalibrar el umbral.

Ninguno de los cinco mide un umbral relativo y auto-referencial, donde la
compresión de lo predicho mueve el umbral y el valor comparado a la vez. El
umbral de Jiao y colaboradores es relativo pero no auto-referencial, porque se
ancla en una observación fija y la compresión alcanza solo al valor comparado; y
su reparación agrega un término de clasificación a la pérdida, es decir, cambia
el objetivo que el modelo optimiza [@jiao2023]. Ese es el caso que la Ecuación
(7) hace explícito, y es donde este documento interviene: recalibra ese umbral
sobre una ventana anterior disjunta, sin reentrenar ni cambiar el objetivo.

---

## III. Método propuesto

Esta sección construye el headway a partir de posiciones GPS, formula la tarea de
predicción sobre el vector de headways y define la regla que convierte ese vector
en un evento de bunching.

### A. Construcción del headway a partir de posiciones GPS

El headway es el tiempo que separa el paso de dos buses consecutivos por un mismo
punto, y la Figura 1 lo ilustra. Es la cantidad que revela si un corredor mantiene
sus buses espaciados o si dos de ellos terminan viajando casi juntos y dejan un
intervalo largo detrás. Ese segundo caso es el evento que define la Sección III-C.
El headway es la variable que predecimos.

La forma habitual de medir el headway es en una parada, con la lista de paradas de
la ruta y los horarios de paso. Ninguna de las dos existe en este caso: el dato
disponible son coordenadas GPS crudas. Para llegar al headway desde esas
coordenadas se aplica la secuencia de seis pasos que sigue.

**1) El eje.** El trazado del corredor se estima de los propios buses: se ajusta
una línea central a las posiciones de las unidades que superan los 10 km/h y
después se suaviza, lo que entrega una curva principal a lo largo del recorrido.

**2) La proyección a una dimensión.** Con el eje ya trazado, cada posición se
reduce a dos números: cuánto ha avanzado el bus a lo largo del corredor y a qué
distancia quedó del eje. Es la operación que la norma ISO 19148 [@iso19148] especifica
para referenciar posiciones contra un objeto unidimensional. La posición se
conserva solo si su desvío lateral no pasa de 300 m; lo que cae más lejos no
pertenece al corredor.

**3) El sentido de marcha.** Sobre ese mismo eje circulan los buses de ida y los
de vuelta, y el dato no distingue unos de otros. El sentido se asigna como el
signo del avance sobre el eje promediado en las últimas cinco posiciones, de modo
que un error aislado no invierta la dirección. La derivación es forzosa: uno de
los corredores no reporta rumbo en absoluto.

**4) Los viajes.** Ya con sentido, el recorrido de cada bus se corta en viajes: un
salto de más de treinta minutos sin señal, una inversión de sentido o una espera
de más de cinco minutos en terminal cierran el viaje en curso.

**5) La rejilla común.** Los buses no emiten sincronizados entre sí, de modo que
hace falta un instante compartido. Todo se lleva a una rejilla de sesenta
segundos, y así cada minuto queda descrito por una **instantánea** del corredor:
la posición de todos sus buses en ese momento.

**6) El headway.** Sobre esa instantánea, para un par de buses consecutivos en el
mismo sentido —el de adelante $L$, el de atrás $F$— en el instante $T$:

$$t_{c} = \max\{\, t \le T \;:\; s_{L}(t) = s_{F}(T) \,\},
\qquad h = T - t_{c}, \tag{1}$$

donde $T$ es el instante evaluado, y $s_{L}$ y $s_{F}$ son las coordenadas de arco
del bus de adelante y del de atrás. El instante $t_{c}$ es el último en que el de
adelante ocupó la posición que el de atrás ocupa en $T$, y $h$ es el headway
resultante. Es un cruce por posición y no por parada, lo que permite prescindir de
la tabla de paradas. Si no existe tal $t_{c}$, o si $h$ supera los treinta
minutos, se emite «sin dato» en lugar de arrastrar un paso de horas antes.

![Definición del headway](figuras/headway/headway.png)

**Fig. 1.** El headway en un punto fijo del corredor: el bus de adelante —el $L$ de
la Ecuación (1)— pasa por p₂ a las 12:30 y el de atrás —el $F$— a las 12:35, de
modo que el headway en p₂ es de cinco minutos. La separación espacial entre los dos
buses no interviene. Esquema ilustrativo, no datos reales.

La Ecuación (1) entrega tiempo entre pasadas. La distancia en metros entre dos
buses consecutivos es la alternativa inmediata, y queda fuera porque mide
separación espacial. Tampoco se proyecta ese tiempo hacia adelante dividiendo la
separación por la velocidad del bus de atrás: esa división supone que la
velocidad actual se mantiene, e introduce una estimación dentro de la cantidad
que se busca estimar.

### B. Formulación de la tarea de predicción

La Sección III-A deja el corredor descrito, minuto a minuto, por un vector de
headways de $N-1$ posiciones, con $N$ el número de buses circulando en ese minuto.
Lo que predecimos es ese vector completo: no un headway suelto ni un
promedio del corredor, sino todas sus posiciones a la vez. Dado el historial de
los últimos $T$ minutos y un contexto de calendario, se busca el vector del
corredor $H$ minutos más adelante:

$$\hat{\mathbf{h}}(t+H) \;=\; f\big(\mathbf{h}(t-T+1), \dots, \mathbf{h}(t);\; c(t)\big),
\qquad T = 12, \tag{2}$$

donde $\mathbf{h}(t)$ es el vector de headways del corredor en el minuto $t$ y
$\hat{\mathbf{h}}(t+H)$ es el vector predicho para $H$ minutos más adelante.
El término $c(t)$ reúne cuatro variables de calendario disponibles en $t$ —el seno
y el coseno de la hora, y el seno y el coseno del día de la semana—, $f$ es el
modelo ajustado y $T$ es la cantidad de minutos de historia que recibe.

Se predice a cuatro horizontes —uno, tres, cinco y diez minutos— con un modelo
ajustado por separado para cada uno: no hay recursión, cada horizonte se predice
de forma directa. El vector no tiene longitud fija, porque $N$ varía minuto a
minuto. El modelo emite entonces una
salida de longitud fija y el error se computa solo sobre las posiciones donde hay
bus. **El objetivo que se minimiza es el error cuadrático**, promediado sobre esas
posiciones válidas:

$$\mathcal{L} \;=\; \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}
\big(\hat{h}_i - h_i\big)^{2}, \tag{3}$$

donde $\mathcal{L}$ es la pérdida que el ajuste minimiza, $\mathcal{V}$ es el
conjunto de posiciones del vector con bus asignado en el instante objetivo,
$|\mathcal{V}|$ es su cardinal, y $\hat{h}_i$ y $h_i$ son el valor predicho y el
observado en la posición $i$.

Esa elección gobierna el resto del trabajo. Una predicción que minimiza error
cuadrático tiende a la media condicional, y la Sección II-B recoge por qué esa
media es menos dispersa que la realidad. La compresión de
dispersión que documenta la Sección V-B no es entonces una falla del ajuste. El
efecto de esa compresión sobre la regla del evento es el asunto de la
Sección III-C.

### C. Definición del evento de bunching

El bunching es el fenómeno en que dos o más buses que deberían circular espaciados
terminan viajando casi juntos y dejan un intervalo largo detrás de ellos. Su costo
recae sobre quien espera en ese intervalo: la espera que enfrenta es la que el
intervalo mide, y no el headway promedio del corredor. Sus causas son
heterogéneas, entre ellas la congestión, un día de demanda atípica, la acumulación
de pasajeros en el bus adelantado o el comportamiento del conductor
[@rezazada2024]. Este trabajo no observa ninguna de ellas: el registro disponible
trae identificador, instante y coordenada, y no pasajeros, ocupación ni estado del
tránsito. Por eso el evento se define sobre la geometría del vector de headways,
que sí es observable, y no sobre lo que la produjo.

Dos rasgos del fenómeno gobiernan cómo se lo define aquí. Es una propiedad del
patrón colectivo y no de una unidad: cada bus puede estar donde le corresponde y
el corredor estar apelotonado igual. Y se manifiesta en posiciones del vector de
la Sección III-A, de modo que un mismo instante puede llevar varias posiciones
afectadas a la vez.

Resta decidir cuándo un headway cuenta como bunching. La convención del campo es
una fracción del headway programado: un cuarto en las formulaciones más citadas
[@moreiramatias2016], y la mitad en el TCQSM [@tcqsm2003]. Estos corredores no
tienen programación contra la cual comparar. Sustituir esa referencia por una
observada del propio dato es práctica establecida: Yu y colaboradores reemplazan
el horario ausente de su corredor por el headway observado en la primera parada de
la misma corrida [@yu2016], y Jiao y colaboradores fijan su umbral en un cuarto de
ese mismo headway de la primera parada [@jiao2023]. Aquí el denominador se sustituye por el promedio del
propio vector en ese instante. **Un headway cuenta como bunching si cae por debajo de la
mitad de ese promedio.** Ese valor es el umbral relativo del evento: se lo llama
relativo porque es una fracción del promedio vigente y no un número fijo de
minutos, de modo que se mueve con cada vector.

La sustitución del denominador es
nuestra y no una herencia: la fracción de la media observada no aparece como
definición de evento en la literatura consultada. La fracción sí es heredada, y es
la del TCQSM. El promedio del vector cumple la función de la programación: fijar
la separación normal en ese corredor en ese instante. Un umbral absoluto, fijo en
minutos, no la cumple, porque no es comparable entre corredores que operan a
frecuencias distintas. La elección del valor tampoco es neutral: los umbrales
publicados van desde veinte segundos hasta un cuarto del headway programado
[@rezazada2024], y no existe un único valor aceptado.

El vector de la Sección III-B se escribe por componentes como
$\mathbf{h}(t) = (h_1, \dots, h_m)$. Su promedio y el umbral del evento son

$$\bar{h}(t) \;=\; \frac{1}{m}\sum_{j=1}^{m} h_j(t),
\qquad \tau(t) \;=\; \rho\,\bar{h}(t), \qquad \rho = \tfrac{1}{2}, \tag{4}$$

donde $m = N - 1$ es la cantidad de posiciones del vector, $N$ es la cantidad de
buses en circulación y $h_j(t)$ es el headway de la posición $j$. El promedio del
vector es $\bar{h}(t)$, el umbral relativo del evento es $\tau(t)$ y $\rho$ es la
fracción del promedio que lo fija. La posición $i$ cuenta como bunching cuando cae
por debajo de ese umbral:

$$b_i(t) \;=\; \mathbb{1}\!\left[\, h_i(t) < \tau(t) \,\right],
\qquad \text{definido solo si } m \ge 3, \tag{5}$$

donde $b_i(t)$ vale 1 si la posición $i$ cuenta como bunching y 0 si no, y
$\mathbb{1}[\cdot]$ es la función indicadora. La condición $m \ge 3$ descarta los
vectores más cortos y exige al menos cuatro buses en circulación. Por debajo de
tres posiciones no hay patrón que describir. Con dos headways hay un solo
intervalo intermedio, así que cualquier medida de irregularidad se reduce a esa
única diferencia. Con tres ya hay patrón: uno colapsado, uno estirado, uno normal.

El detector que este trabajo evalúa es esa misma regla aplicada al vector predicho
de la Ecuación (2), con el promedio de ese mismo vector fijando el umbral:

$$\hat{b}_i(t) \;=\; \mathbb{1}\!\left[\, \hat{h}_i(t) < \rho\,\bar{\hat{h}}(t)
\,\right], \tag{6}$$

donde $\hat{b}_i(t)$ es la detección emitida sobre la posición $i$ del vector
predicho y $\bar{\hat{h}}(t)$ es el promedio de ese mismo vector predicho. El
umbral sale del vector predicho y no del observado porque quien opera un corredor
no dispone del observado al momento de decidir.

Como $\tau$ es función del propio vector que se evalúa, y no un número fijo de
minutos, las Ecuaciones (5) y (6) no comparan contra el mismo umbral:

$$\tau(\hat{\mathbf{h}}) \;=\; \rho\,\bar{\hat{h}}
\;\neq\; \rho\,\bar{h} \;=\; \tau(\mathbf{h})
\qquad \text{siempre que } \bar{\hat{h}} \neq \bar{h}, \tag{7}$$

donde $\tau(\mathbf{h})$ y $\tau(\hat{\mathbf{h}})$ son los umbrales que resultan
de aplicar $\rho$ al vector observado y al vector predicho. La referencia de Yu y
colaboradores no tiene esa propiedad: es observada, de modo que no se mueve con el
pronóstico. Las Figuras 2 y 3 lo muestran con el mismo headway de dos minutos.

![Corredor disparejo](figuras/bunching/with_bunching.png)

**Fig. 2.** Corredor disparejo. El vector es [9,5 · 1,2 · 11,0 · 2,0], su promedio
5,9 min y el umbral 3,0 min. Los headways de 2,0 y 1,2 quedan debajo del umbral:
**los dos son bunching.** Esquema ilustrativo, no datos reales.

![Corredor parejo](figuras/bunching/without_bunching.png)

**Fig. 3.** Corredor parejo. El vector es [3,5 · 2,0 · 4,0 · 3,0], su promedio
3,1 min y el umbral 1,6 min. El mismo headway de 2,0 min queda ahora encima del
umbral: **no es bunching.** Esquema ilustrativo, no datos reales.

Dos minutos entre buses es el mismo hecho físico en las dos figuras, y la regla lo
clasifica al revés porque el umbral se movió con el vector. La Sección V mide qué
ocurre cuando esa diferencia se ignora sobre datos reales.

---

## IV. Diseño experimental

Esta sección describe los datos, los métodos comparados, el protocolo de
partición, las métricas y las pruebas estadísticas con los que se evalúa la
predicción del vector de headways.

### A. Datos

El trabajo usa los registros de posición de la flota del Sistema Integrado de
Transporte de Arequipa. Cada unidad emite su coordenada **cada 20 segundos**, y la
cadencia es regular: la mediana y el percentil 95 del tiempo entre emisiones
coinciden, de modo que el dato no llega a ráfagas. Esa regularidad sostiene la
rejilla de sesenta segundos de la Sección III-A, porque cada minuto reúne tres
emisiones por bus.

Se cubren tres corredores —identificados aquí como E2, E4 y E59, uno por empresa
operadora— durante 152 días seguidos, del 1 de octubre de 2023 al 29 de febrero de
2024, sin huecos de calendario. Son 90 unidades en total. El registro no incluye
horario publicado, archivo GTFS ni tabla de paradas, de modo que el corredor, el
sentido y el headway se construyen desde la posición cruda, como describe la
Sección III-A.

La construcción del headway desde la posición cruda no siempre produce un valor.
Dos condiciones dejan un par de buses sin headway: que la Ecuación (1) no
encuentre el cruce, o que el headway supere los treinta minutos. La cobertura —la
fracción de pares evaluados con headway válido— es del 63,5 % en E2, del 64,8 % en
E4 y del 77,1 % en E59: 3 938 174 pares en total. Una posición del vector sin
headway válido se enmascara.

Los huecos que deja ese enmascaramiento no se distribuyen al azar. Ninguna de las
dos condiciones es neutral: ambas recortan por el extremo alto de la distribución,
de modo que los descartados son los intervalos más largos. La cobertura tampoco es
pareja entre corredores: entre el mejor y el peor medido hay 13,6 puntos
porcentuales.

### B. Métodos comparados

Se comparan cuatro métodos sobre las mismas muestras, y cada uno cumple un papel
distinto. El método bajo estudio es una red recurrente (**LSTM**); la Sección V-G
contrasta esa elección contra dos arquitecturas que modelan la relación entre
posiciones vecinas del vector. Un conjunto de árboles con refuerzo de gradiente
(**XGBoost**) [@chen2016] actúa como **control de arquitectura**: si reproduce el patrón del LSTM, ese patrón
no proviene del aprendizaje profundo sino del objetivo de la Ecuación (3). Los dos
restantes no ajustan parámetros y fijan el error de referencia. La **persistencia**
repite el último vector observado, así que su error crece con el horizonte. El
**promedio histórico por franja horaria** responde con el valor típico de esa hora
del día, calculado sobre entrenamiento por corredor y sentido; no lee la ventana de
entrada, de modo que su error no depende del horizonte.

El conjunto excluye tres métodos estadísticos. La media del período de
entrenamiento, la media móvil causal en tres ventanas y el suavizado exponencial
simple de factor 0,3 no combinan las dos entradas de la Ecuación (2), el historial
reciente y el calendario. Los tres repiten información que la persistencia o el
promedio histórico ya aportan.

De los cuatro métodos retenidos, solo el LSTM y el XGBoost ajustan parámetros.
Ambos se ajustan por corredor y por horizonte, y cada par de corredor y horizonte
se denomina aquí **celda**: hay doce. Los dos sentidos comparten el modelo de su
corredor y entran juntos al entrenamiento. Lo que se separa por sentido son los
estadísticos de estandarización, de modo que lo predicho se devuelve a minutos con
los del sentido que le corresponde. El LSTM usa 32 unidades ocultas, una o dos
capas según la celda, paso 5 × 10⁻⁴, lotes de 128 y semilla fija en 42. El XGBoost
usa hasta 400 rondas con parada temprana tras 30 sin mejora, y la misma semilla. Los presupuestos de
búsqueda no son iguales: el XGBoost eligió veinticuatro configuraciones por celda
sobre las muestras definitivas, mientras que el LSTM heredó la suya de una
búsqueda previa que no se rehízo sobre esas muestras, en dos de los tres
corredores. La Sección VI acota qué afirmaciones no se sostienen con esa
diferencia.

### C. Protocolo de evaluación

La partición es **por fecha y nunca al azar**, porque un operador solo dispone del
pasado. El período se divide en 107 días de entrenamiento, 23 de validación y 22
de prueba. Todo el protocolo se repite después sobre tres orígenes de evaluación,
identificados aquí como ventanas 1, 2 y 3. Los tres arrancan el mismo día y
alargan el entrenamiento a 61, 83 y 107 días. Sus períodos de prueba no se solapan
entre sí, y el de la ventana 3 es el que se publica. Como los entrenamientos están
anidados, esto establece estabilidad frente a la elección del período de prueba y
no réplica independiente. La Figura 4 muestra el esquema.

![Partición temporal y los tres orígenes](figuras/esquema-particion-temporal.es.png)

**Fig. 4.** La partición por tiempo y los tres orígenes de evaluación. Los tres
arrancan el mismo día y alargan el entrenamiento; sus períodos de prueba no se
solapan.

La comparación exige además tres condiciones, cada una sobre una fuente distinta
de fuga: el tiempo, la población evaluada y los valores extremos. La primera es la
continuidad estricta: una muestra es válida solo si sus minutos son consecutivos.
Sin ella la ventana puede atravesar un hueco de señal, y el horizonte mediría un
intervalo mayor que el declarado. La regla retiene entre el 81,9 % y el 90,2 % de
las instantáneas del período de prueba.

La segunda es la población compartida: los cuatro métodos se puntúan sobre
exactamente las mismas filas. El trabajo de entrenamiento recalcula la lista de
muestras, compara su resumen SHA-256 contra el registrado y aborta antes de usar
la GPU si no coincide. La verificación evita comparar métodos puntuados sobre
poblaciones distintas. La tercera es el tope al percentil 99 del headway de
entrenamiento, aplicado como techo a las tres particiones. Calcularlo por
partición dejaría entrar información del período de prueba. El techo afecta entre
el 0,78 % y el 1,11 % de los objetivos, y las posiciones sin headway válido siguen
enmascaradas.

Sobre esa misma población, los resultados se desglosan además por régimen de
dispersión. La dispersión se mide sobre cada posición del vector por separado: es
la desviación estándar muestral de los headways que esa posición registró a lo
largo de la ventana de entrada, en minutos. Cada combinación de corredor y
horizonte se parte en tercios por esa cantidad, con los dos umbrales fijados sobre
entrenamiento y validación y aplicados sin cambios a prueba. Calibrarlos sobre
prueba dejaría que la estratificación conociera el período que evalúa.

### D. Métricas

El modelo entrega un vector de headways que la regla de la Sección III-C convierte
en un indicador binario de bunching, y la evaluación mide esos dos objetos en
cadena. El error del vector es el error absoluto medio (MAE) sobre las posiciones
válidas que define la Ecuación (3):

$$\mathrm{MAE} \;=\; \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}
\big|\hat{h}_i - h_i\big|, \tag{8}$$

donde $\mathcal{V}$, $|\mathcal{V}|$, $\hat{h}_i$ y $h_i$ conservan el
significado de la Ecuación (3). Se reporta el MAE y no el error cuadrático porque
expresa el resultado en minutos de headway.

El MAE no describe la forma del vector. El coeficiente de variación (CV) es su
desviación estándar muestral dividida por su promedio:

$$\mathrm{CV}(\mathbf{h}) \;=\; \frac{1}{\bar{h}}
\sqrt{\frac{1}{m-1}\sum_{j=1}^{m}\big(h_j - \bar{h}\big)^{2}}, \tag{9}$$

donde $m$, $h_j$ y $\bar{h}$ conservan el significado de la Ecuación (4). Se
calcula sobre los vectores de tres posiciones o más que exige la Ecuación (5). Se
reporta porque es adimensional, de modo que corredores de frecuencias distintas
quedan sobre la misma escala. Su sesgo es el CV de lo predicho menos el de lo
observado, y un valor negativo dice que lo predicho es más regular que la realidad.

El indicador derivado se puntúa con tres cantidades, ordenadas por cuánto dependen
del umbral, sobre el cruce entre el indicador observado de la Ecuación (5) y el
detector de la Ecuación (6). Sean TP las posiciones con $b_i = \hat{b}_i = 1$, FP
las que tienen $\hat{b}_i = 1$ y $b_i = 0$, FN las que tienen $b_i = 1$ y
$\hat{b}_i = 0$, y TN las restantes. La precisión, el recall y el F1 son entonces

$$\mathrm{F}_1 \;=\; \frac{2PR}{P+R}, \qquad
P \;=\; \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \qquad
R \;=\; \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \tag{10}$$

donde $P$ es la precisión y $R$ el recall.

El F1 no usa TN [@chicco2020], y premia por eso al detector que marca toda
posición como evento: maximizar el F1 sobre un pronóstico sin información conduce
a ese detector con independencia de la tasa base [@lipton2014]. La tasa base de
una celda es la fracción de sus posiciones donde el indicador observado vale 1.
Ese detector alcanza recall 1 y precisión igual a la tasa base [@flach2015], así
que su F1 queda fijado por ella y acompaña como piso a todo F1 reportado. El
coeficiente de correlación de Matthews (MCC) usa los cuatro conteos. Para ese
detector su cociente queda indeterminado, porque numerador y denominador se
anulan a la vez, y se le asigna cero por extensión por continuidad [@chicco2020].
El área bajo la curva ROC (AUC) prescinde del umbral —el punto de operación del
detector— y puntúa el ordenamiento del puntaje continuo
$-\hat{h}_i/\bar{\hat{h}}$, del cual la Ecuación (6) es el umbral en $-\rho$. Es
la probabilidad de que una posición de bunching reciba un puntaje mayor que una
sin bunching [@handtill2001], y vale 0,5 cuando el pronóstico no ordena.

Sobre esas cantidades se construyen tres cocientes. La tasa de disparo de un
método es la fracción de posiciones que marca como evento. El factor entre dos
métodos es el cociente de sus F1, y mide cuántas veces mejor aparece uno de ellos
bajo el mismo umbral. El tercero exige una cantidad más. La precisión promedio
también prescinde del umbral: recorre el ordenamiento que el AUC puntúa, de mayor
a menor, y promedia la precisión de la Ecuación (10) sobre las posiciones de
bunching. Un pronóstico que no ordena alcanza una precisión promedio igual a la
tasa base. El lift es entonces la precisión promedio dividida por la tasa base, y
vale 1 cuando el pronóstico no ordena mejor que el azar.

El umbral no se hereda de lo observado. Se ajusta maximizando el MCC sobre el
período de prueba de la ventana 2 y se aplica sin cambios al de la ventana 3. Los
dos períodos son disjuntos y provienen de modelos entrenados por separado, de modo
que el período publicado no informa su propio umbral.

### E. Pruebas estadísticas

La Sección IV-D define las métricas, y compararlas entre dos métodos exige
declarar qué cuenta como resultado de esa comparación. Un veredicto es la
comparación de dos métodos sobre las mismas muestras bajo una métrica declarada,
y consta de tres partes: cuál de los dos gana, por cuánto y si la diferencia
sobrevive su prueba. Exigir muestras idénticas es lo que lo distingue de la resta
de dos métricas agregadas, que pueden haberse calculado sobre poblaciones
distintas. Este trabajo emite veredictos sobre el MAE de la Ecuación (8) y sobre
las cantidades de detección de la Ecuación (10), el MCC y el AUC. Un veredicto
sin umbral es el que usa el AUC, que no depende del punto de operación.

Una diferencia de MAE entre dos métodos puede ser ruido del período de prueba. Se
contrasta con la prueba de Diebold–Mariano [@diebold1995] sobre el diferencial de pérdida por
muestra, con la corrección de muestra pequeña de Harvey–Leybourne–Newbold [@harvey1997]. La
varianza se estima agrupando por día de servicio, porque las
muestras de un mismo día comparten clima, incidentes y demanda. El agrupamiento
lleva el tamaño efectivo de muestra de decenas de miles de filas a los 22 días del
período de prueba.

La precisión de la Ecuación (10) admite su propia acotación, porque puede
descansar sobre muy pocas posiciones marcadas. Se acota con el intervalo exacto de
Clopper–Pearson [@clopper1934] al 95 %, calculado sobre los conteos de TP y de
FP de cada celda. Se prefiere el intervalo exacto a la aproximación normal. Los
conteos que necesitan acotarse aquí son los pequeños, y en ellos la aproximación
deja parte de su intervalo fuera del rango válido de una proporción. Una celda
donde el detector nunca marca no recibe intervalo: no hay precisión que acotar.

---

## V. Resultados y discusión

Esta sección reporta el error escalar del vector y la frontera de régimen que lo
acota. Mide después la dispersión transversal de lo predicho, la detección con el
umbral del evento observado y el comportamiento del factor entre las tres ventanas.
Cierra con la detección puntuada sin umbral y con el umbral recalibrado, los
ensayos de robustez frente a la ventana y a la definición del evento, el
contraste entre arquitecturas y las implicaciones operativas.

### A. Error escalar y su frontera de régimen

A diez minutos de anticipación, el LSTM predijo el headway entre buses
mejor que la persistencia. El error absoluto medio bajó 1,47 minutos en E2, 1,38
en E4 y 1,17 en E59: entre 21 % y 22 % en los tres corredores. A un minuto la
relación se invirtió y la persistencia ganó, por 0,46 minutos en E4 y 0,33 en E59.
En E2 la diferencia fue de 0,07 minutos y no resistió la prueba estadística al
agrupar las observaciones por día de servicio.

Tres precisiones acotan ese resultado. La primera es que el cruce no es una
propiedad del aprendizaje profundo: el XGBoost lo reprodujo entero, y a
diez minutos aventajó a la persistencia por 1,59 minutos en E2, 1,09 en E4 y 0,79
en E59. La segunda es que la frontera real no es el horizonte sino la dispersión
de la ventana de entrada. Medida con los tercios de dispersión de la
Sección IV-C, la ventaja del LSTM creció de forma ordenada del tercio calmo al
volátil en 11 de las 12 celdas.
Alargar el horizonte no cambió quién ganaba: movió la ventaja hacia tercios cada
vez más tranquilos.

La tercera es que el promedio histórico por franja horaria cumplió el papel que la
Sección IV-B le asignaba. Su error no se movió con el horizonte: se quedó entre
4,7 y 5,7 minutos en los tres corredores. La ventaja del LSTM sobre él se
estrechó entonces a medida que el horizonte crecía. En E2 el LSTM le ganó por
0,99 minutos a un horizonte de un minuto y le perdió por 0,07 a diez. Esa fue la
única
de las doce celdas donde el promedio histórico ganó, y es la razón de que a
horizonte largo el competidor exigente sea él y no la persistencia.

### B. Compresión de la dispersión transversal

El error escalar de la Sección V-A no dice nada sobre la forma del vector. El
coeficiente de variación de la Ecuación (9) sí. Medido sobre lo observado, fue de
0,79 en E2. Medido sobre lo que el modelo predijo para el mismo instante y el mismo
corredor a diez minutos, fue de
0,16. El vector predicho describió un corredor casi cinco veces más regular que el real.

Esa brecha no fue un caso aislado. El sesgo del coeficiente de variación resultó
negativo —lo predicho siempre más regular que la realidad— en
**las doce celdas y las tres ventanas de prueba**. Y se profundizó de forma estrictamente
ordenada a medida que se alarga el horizonte: en E2 pasó de −0,42 a un minuto a
−0,63 a diez. No hubo una sola excepción en los tres corredores.

Dos comparaciones acotan de qué depende el efecto. La primera identifica la causa
por descarte: la persistencia no comprimió nada. Su sesgo se mantuvo dentro de
±0,022 en las doce celdas y las tres ventanas, porque propaga el vector observado y
hereda su dispersión sin traducción. Es el control del experimento, y sitúa el
efecto en el acto de **emitir una predicción puntual**, no en los datos ni en el
corredor. La segunda
descarta la arquitectura: el XGBoost comprimió igual que la red en E2, y
las dos curvas se superponen. En los otros dos corredores comprimió **más** que
ella, con un sesgo de −0,46 contra −0,35 en E59 a diez minutos. Un fenómeno que
aparece
igual en una red recurrente y en un conjunto de árboles no es una propiedad de
ninguna de las dos.

La consecuencia práctica se aprecia al leer esas cifras contra la escala de nivel
de servicio del TCQSM [@tcqsm2003]. El manual indexa sus bandas
por la dispersión del headway respecto del programado. Estos corredores no tienen
programación, así que la escala se lee con el coeficiente de variación de la
Ecuación (9). Con esa sustitución, el mismo corredor en el mismo instante calificó
como nivel A —«service provided like clockwork»— según lo predicho y como nivel F
—«most vehicles bunched»— según lo observado. La cantidad que estas medidas
capturan es la dispersión **entre buses en un mismo instante**, y no la
variabilidad de una serie a lo largo del tiempo que la Sección II-D delimita como
previa. Las Figuras 5 y 6 muestran el efecto y su dependencia del horizonte.

![Dispersión observada frente a predicha](figuras/compresion-dispersion.es.png)

**Fig. 5.** Dispersión observada frente a dispersión predicha, horizonte de diez
minutos. La barra de la persistencia iguala a la observada: hereda el vector real
y sirve de control. Los dos modelos ajustados la comprimen.

![Sesgo de dispersión contra horizonte](figuras/compresion-vs-horizonte.es.png)

**Fig. 6.** El mismo sesgo contra el horizonte. La persistencia no se despega de
cero; los dos modelos ajustados descienden de forma monótona. La compresión escala con la
distancia que se pide anticipar.

### C. Colapso de la detección al trasladar el umbral

La regla de la Sección III-C, aplicada a lo observado, marcó 15 245 eventos en E2
a diez minutos. Aplicada a lo predicho por el LSTM, con el mismo umbral,
se disparó **catorce veces**. La persistencia disparó 15 083 veces. Puntuada con
el F1 de la Sección IV-D, la persistencia apareció 253 veces mejor que el LSTM. En
las otras celdas el factor va de 1,5 a 36. El XGBoost obtuvo un F1
exactamente cero en tres de las doce celdas: ahí no disparó nunca. La Tabla 1
recoge las doce celdas.

Leído sin más contexto, ese resultado dice que el LSTM es incapaz de ver el
fenómeno que se le pidió anticipar. Hay tres motivos para desconfiar de esa
lectura. El primero es que el ganador declarado tampoco detectó bien. El detector
trivial de la Sección IV-D superó a la
persistencia en 5 de las doce celdas, y en 15 de las 36 combinaciones de celda y
ventana. Un procedimiento de evaluación en el que una regla vacía vence al
ganador declarado no ordena modelos.

El segundo es el mecanismo de la Sección V-B. El umbral se mide contra el promedio
del propio vector evaluado. Si el vector predicho es más regular que la realidad, sus
headways se apartan menos de su propio promedio, y el umbral deja de alcanzarse
casi siempre. Lo que la regla registra no es que el modelo no vea el evento: es
que el modelo no produce la dispersión necesaria para cruzar un umbral calibrado
sobre otra distribución.

El tercero es el acierto del modelo en las pocas ocasiones en que disparó. De
los catorce disparos de E2, diez correspondieron a eventos de bunching reales: 71 %
de precisión contra una tasa base de 30 %. El intervalo de la Sección IV-E va de
42 % a 92 % sobre esos catorce disparos, de modo que la cifra señala un régimen y
no un valor. Las celdas con más disparos lo estrechan. A diez minutos el modelo
acertó 776 de 1 572 disparos en E59, con precisión entre 47 % y 52 % contra una
tasa base de 21 %. En E4 acertó 75 de 150, entre 42 % y 58 % contra 18 %. Una
medida que castiga por igual al que no marca y al que marca mal no distingue esos
dos casos.

![Tasa de disparo contra tasa real del evento](figuras/artefacto-umbral.es.png)

**Fig. 7.** Fracción de posiciones que cada método marca como bunching, contra la tasa
real del evento (punteada). La persistencia propaga el vector observado, hereda su
dispersión y el umbral cae donde fue diseñado: marca casi tan seguido como el
evento ocurre. La predicción puntual es un vector comprimido, y el mismo umbral
relativo le queda en la cola.

**Tabla 1.** Detección con el umbral del evento observado aplicado sin cambios a lo predicho, con el piso del detector trivial al lado.

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

† La regla vacía —marcar toda posición— supera al ganador declarado en estas celdas.

### D. Estabilidad del factor entre ventanas y sus dos excepciones

Si el factor de 253 de la Sección V-C midiera una capacidad del modelo, debería
ser aproximadamente estable al cambiar la ventana de prueba. En diez de las doce
celdas lo es: entre la primera ventana y la tercera varía entre 0,90 y 1,58. Las
dos excepciones están en E2. A cinco minutos el factor valió **126** en la primera
ventana, **58** en la segunda y **36** en la tercera. A diez minutos valió
**2 299**, **817** y **253**.

Esas dos son las celdas donde el umbral trasplantado dejó al detector casi sin
disparos: su F1 cayó a 0,011 y 0,001 en la Tabla 1. Un cociente cuyo denominador
se acerca a cero no mide una capacidad del sistema evaluado, sino la interacción
entre el umbral y la distribución sobre la que cayó. La observación no depende de
qué modelo se use ni de cuál de las tres ventanas se mida. Depende de que el umbral
se haya trasladado entre dos distribuciones con dispersión distinta.

### E. La detección sin umbral y con el umbral recalibrado

Si el problema es el umbral, recalibrarlo debería bastar. Se aplicó
entonces la recalibración de la Sección IV-D, sin tocar el modelo. Elegir el MCC y
no el F1 como objetivo responde a que en este corpus el F1 degenera. Sobre la
persistencia en E2, de tres minutos en adelante, el umbral que optimiza el F1
disparó entre el 99,9 % y el 100 % de las posiciones, esto es, la regla vacía de
la Tabla 1.

El umbral trasplantado de la Tabla 1 dejaba a la persistencia por delante en las
doce celdas. Los dos instrumentos que la Tabla 2 reúne mueven ese conteo en
distinta medida. Puntuado sin umbral, mediante el AUC, **el LSTM ganó en las nueve
combinaciones de corredor y ventana a diez minutos**, y en 6 de las 12 celdas de
la ventana 3. Recalibrar el umbral en lugar de eliminarlo lo mueve menos: con el
MCC recalibrado el LSTM ganó en 5 de las 12 celdas, entre ellas las tres de diez
minutos. La persistencia conservó la ventaja en el horizonte de un minuto, donde
el error escalar también la favorecía en E4 y E59.

El AUC no es la única forma de puntuar sin umbral. El lift de la Sección IV-D
recorre el mismo ordenamiento, pero pesa más su cabeza, donde caen las posiciones
que un detector marcaría primero. Los dos coincidieron en las doce celdas: en cada
una ganó el mismo método. A diez minutos el lift del LSTM valió 1,19 en E2, 1,45
en E4 y 1,48 en E59, contra 1,08, 1,24 y 1,26 de la persistencia. El veredicto sin
umbral no depende entonces de cuál de los dos puntajes se use. Con el MCC
recalibrado el acuerdo baja a once de doce. La excepción es E59 a cinco minutos,
donde el LSTM gana el AUC y pierde la correlación recalibrada.

Los dos cruces van en el mismo sentido. Medido por el signo de la diferencia, el
error escalar pasó a favor del LSTM entre uno y tres minutos en los tres
corredores. El AUC pasó a su favor entre uno y tres minutos en E2, entre tres y
cinco en E59, y entre cinco y diez en E4. La detección cruzó entonces uno o dos
escalones de horizonte más tarde que el error en dos de los tres corredores.
Ninguna de las dos métricas cruzó en sentido contrario. La disociación que las
Secciones V-A y V-C parecían mostrar, con el LSTM ganando en error y perdiendo en
detección, la producía el umbral.

El cambio de veredicto no requirió tocar el modelo. El AUC de la Figura 8 se
calculó sobre las mismas predicciones que la Figura 7 puntúa con el umbral
trasplantado. No se reentrenó, no se agregó información y no se modificó ninguna
arquitectura. Entre las dos figuras cambió el umbral de la Ecuación (6). La
Figura 7 lo hereda de lo observado y la Figura 8 lo elimina; la columna del MCC
recalibrado de la Tabla 2 lo reajusta contra lo predicho. Como ninguna otra cosa
varió, ninguna otra cosa explica el cambio de conteo, y el umbral queda
identificado como la variable que producía el veredicto. La Sección V-H recoge lo
que sigue de esto para quien opera.

![Ventaja escalar y AUC de detección](figuras/deteccion-sin-umbral.es.png)

**Fig. 8.** Las mismas predicciones puntuadas sin umbral. Eje izquierdo: cuánto
error absoluto le gana el LSTM a la persistencia. Eje derecho: área bajo la
curva de detección, invariante a cualquier reescalado monótono de lo predicho y
por lo tanto inmune al artefacto. Los dos cruces van en el mismo sentido, y
ninguna serie se acerca al azar.

**Tabla 2.** Veredicto sin umbral y con el umbral recalibrado fuera de muestra.

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

El veredicto sin umbral de la Sección V-E no depende de la ventana calendaria. Las
tres ventanas coincidieron en 11 de las 12 celdas, y a diez minutos coincidieron
en las nueve combinaciones de corredor y ventana. La primera de las tres cubre del
23 de diciembre al 13 de enero. Ese acuerdo incluye entonces el período de
fiestas, cuando la frecuencia del servicio y la demanda no se parecen a las de un
mes ordinario.

Tampoco depende de la definición del evento. El umbral relativo de la Sección
III-C podría estar produciendo el efecto por sí solo, y un umbral absoluto en
minutos —como el de un minuto de Sun, Schmöcker y Nakamura [@sun2021]— podría
disolverlo. Se probó con uno fijo en la cuarta parte del headway mediano
observado de cada corredor y dirección. Queda entre 1,4 y 2,4 minutos, se calibró
sobre la ventana 2 y se aplicó sin cambios a la ventana 3. **No se atenuó:
empeoró.** La tasa de disparo del modelo cayó por un factor de mediana 138 en diez
de las doce celdas, y en las otras dos no marcó ninguna posición.

El mismo ensayo acota una afirmación anterior. Bajo el umbral absoluto la
capacidad de discriminación del modelo cayó: la mediana del AUC bajó a 0,60, y en
E2 a diez minutos llegó a 0,49, indistinguible del azar. La afirmación de que el
LSTM no es ciego se sostiene para el evento relativo y falla para el evento
absoluto en esa celda. La Tabla 3 recoge las tres ventanas y ese ensayo.

**Tabla 3.** Robustez: las tres ventanas temporales y el ensayo con un umbral
absoluto en minutos.

| Corredor | h | Ventana 1 | Ventana 2 | Ventana 3 | Coinciden | AUC, umbral absoluto |
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

### G. Selección de la arquitectura

Antes de fijar el protocolo de la Sección IV, tres arquitecturas se contrastaron
entre sí sobre los mismos datos. Por eso sus cifras se leen unas contra otras y no
contra las del resto de la Sección V. La primera es el LSTM que este trabajo
lleva, que recibe el vector aplanado. Las otras dos modelan la relación entre
posiciones vecinas: una convolución sobre el eje de los buses y atención entre las
posiciones del vector. Esa relación es la estructura que un pronóstico vectorial
podría aprovechar.

Las tres quedaron dentro de un rango de 0,017 a 0,074 minutos en las doce celdas,
y ninguna quedó primera en las doce. La Tabla 4 las recoge. Modelar la relación
entre posiciones vecinas no movió el error escalar, de modo que el trabajo
continuó con la más simple de las tres.

**Tabla 4.** Error absoluto medio de las tres arquitecturas contrastadas antes de
fijar el protocolo de la Sección IV. La última columna es la diferencia entre la
mayor y la menor de cada fila.

| Corredor | h | LSTM | SpatialConvLSTM | SpatialTransformer | Rango |
| :--- | ---: | ---: | ---: | ---: | ---: |
| E2 | 1 | 4,464 | 4,464 | 4,482 | 0,018 |
| E2 | 3 | 4,916 | 4,916 | 4,936 | 0,020 |
| E2 | 5 | 5,040 | 5,037 | 5,075 | 0,038 |
| E2 | 10 | 5,128 | 5,123 | 5,142 | 0,019 |
| E4 | 1 | 3,774 | 3,811 | 3,833 | 0,059 |
| E4 | 3 | 4,679 | 4,698 | 4,754 | 0,074 |
| E4 | 5 | 5,014 | 5,054 | 5,086 | 0,072 |
| E4 | 10 | 5,348 | 5,367 | 5,380 | 0,032 |
| E59 | 1 | 3,334 | 3,329 | 3,350 | 0,021 |
| E59 | 3 | 3,847 | 3,847 | 3,883 | 0,036 |
| E59 | 5 | 4,029 | 4,037 | 4,051 | 0,022 |
| E59 | 10 | 4,224 | 4,239 | 4,222 | 0,017 |

### H. Implicaciones operativas

El resultado operativo no es que el modelo detecte mejor. Es que **marcó poco y
acertó cuando marcó**, y el F1 de la Ecuación (10) combina esas dos propiedades en
un solo número. La Sección V-C reporta los conteos: catorce disparos en E2 a diez
minutos, y en ese horizonte la precisión quedó por encima de la tasa base en los
tres corredores, con su intervalo al lado. Esa lectura describe el umbral
trasplantado, y recalibrarlo deshace su primera mitad: el detector recalibrado
marcó el 26,98 % de las posiciones de E2 a diez minutos, contra el 0,03 % del
trasplantado.

Eso no es una alarma. Una alarma tiene que sonar cuando ocurre el evento, y con el
umbral trasplantado el detector se queda callado la mayoría de las veces. Lo que
queda es un **filtro de prioridad**: un aviso poco frecuente y más informativo que
el azar, que sirve para ordenar la atención de un despachador y no para
dispararla. La consecuencia para quien evalúa un pronóstico de este tipo es
distinta. **El punto de operación se recalibra contra la distribución de lo
predicho, no se hereda de las observaciones.** Requiere recalcular un escalar y no
reentrenar nada.

---

## VI. Amenazas a la validez

El umbral del evento es la fracción del promedio que usa la convención del campo,
y no proviene de un registro de eventos observados. Esa elección lo hace
comparable con los trabajos que la Sección III-C cita, y deja sin verificar que la
fracción marque lo que un operador llamaría bunching. Validarla exigiría un
registro de incidentes que estos corredores no producen. El alcance de todo
reclamo de detección es entonces el evento así definido.

El corpus acota dos cosas más. Un vector reúne entre 3,8 y 5,9 headways en
promedio, de modo que la dispersión transversal reposa sobre pocas observaciones.
Que el efecto se repita en los tres corredores y en las tres ventanas lo hace poco
atribuible a esa longitud. Cada cifra individual es menos estable en E2 y en E4,
que tienen el vector más corto, que en E59. El período de prueba contiene además
los días de Carnaval, cuya composición no se caracterizó, de modo que la
comparación incluye días atípicos sin identificarlos.

Los dos métodos que ajustan parámetros no reciben el mismo presupuesto de
búsqueda. Como declara la Sección IV-B, el XGBoost elige veinticuatro
configuraciones por celda sobre las muestras definitivas, mientras que el LSTM
hereda la suya en dos de los tres corredores. Eso acota una comparación y solo
una: donde el LSTM queda por detrás del XGBoost, la diferencia no es atribuible a
la clase de modelo. Los otros dos métodos no ajustan nada, de modo que el error de
referencia que fijan no depende de esa asimetría. El contraste de arquitecturas
de la Sección V-G tampoco está nivelado con el resto, porque precede al protocolo
de la Sección IV y no se rehízo después.

Las métricas de la Sección IV-D son genéricas y comparables entre corredores, y
ninguna liga un error de predicción a una decisión de intervención. Un despacho
necesitaría una función de costo que pondere el aviso perdido contra el aviso
falso, y esa función depende de la operación de cada empresa. Construirla y
evaluar el modelo contra ella queda para trabajo futuro.

---

## VII. Conclusión

_(pendiente)_

---

## VIII. Declaraciones

_(pendiente — disponibilidad de datos y código)_

---

## Referencias

_(lista en construcción: solo las fuentes ya verificadas en
`fuentes-verificadas.md` y ya llamadas desde el texto. Las llamadas usan claves
con arroba y no números, de modo que insertar una fuente no obliga a renumerar ni
a corregir llamadas. La numeración por orden de primera aparición se resuelve al
convertir al formato IJACSA, sustituyendo cada clave por su número; el orden de
esta lista no es todavía el definitivo.)_

`[@chen2016]` T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System,"
in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 785–794,
doi: 10.1145/2939672.2939785.

`[@chicco2020]` D. Chicco and G. Jurman, "The advantages of the Matthews
correlation coefficient (MCC) over F1 score and accuracy in binary classification
evaluation," *BMC Genomics*, vol. 21, no. 1, art. 6, 2020,
doi: 10.1186/s12864-019-6413-7.

`[@clopper1934]` C. J. Clopper and E. S. Pearson, "The use of confidence or
fiducial limits illustrated in the case of the binomial," *Biometrika*, vol. 26,
no. 4, pp. 404–413, 1934, doi: 10.1093/biomet/26.4.404.

`[@diebold1995]` F. X. Diebold and R. S. Mariano, "Comparing Predictive Accuracy,"
*Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995,
doi: 10.1080/07350015.1995.10524599.

`[@flach2015]` P. A. Flach and M. Kull, "Precision-Recall-Gain Curves: PR
Analysis Done Right," in *Advances in Neural Information Processing Systems 28*,
2015, pp. 838–846.

`[@handtill2001]` D. J. Hand and R. J. Till, "A Simple Generalisation of the Area
Under the ROC Curve for Multiple Class Classification Problems," *Machine
Learning*, vol. 45, no. 2, pp. 171–186, 2001, doi: 10.1023/A:1010920819831.

`[@hoffmann2018]` P. Hoffmann, C. Menz, and A. Spekat, "Bias adjustment for
threshold-based climate indicators," *Advances in Science and Research*, vol. 15,
pp. 107–116, 2018, doi: 10.5194/asr-15-107-2018.

`[@harvey1997]` D. Harvey, S. Leybourne, and P. Newbold, "Testing the equality of
prediction mean squared errors," *International Journal of Forecasting*, vol. 13,
no. 2, pp. 281–291, 1997, doi: 10.1016/S0169-2070(96)00719-4.

`[@iso19148]` Geographic information — Linear referencing, ISO 19148:2021, 2nd ed.,
International Organization for Standardization, Geneva, Switzerland, 2021.

`[@jiao2023]` J. Jiao, P. Shen, and Y. Zhang, "Headway-based Bus Bunching
Prediction Using LSTM with Attention," in *2023 IEEE 8th International Conference
on Intelligent Transportation Engineering (ICITE)*, 2023, pp. 451–458,
doi: 10.1109/ICITE59717.2023.10733869.

`[@lipton2014]` Z. C. Lipton, C. Elkan, and B. Naryanaswamy, "Optimal
Thresholding of Classifiers to Maximize F1 Measure," in *ECML PKDD 2014*, Lecture
Notes in Computer Science, vol. 8725, 2014, pp. 225–239,
doi: 10.1007/978-3-662-44851-9_15.

`[@manibardo2022]` E. L. Manibardo, I. Laña, and J. Del Ser, "Deep Learning for
Road Traffic Forecasting: Does it Make a Difference?," *IEEE Transactions on
Intelligent Transportation Systems*, vol. 23, no. 7, pp. 6164–6188, 2022,
doi: 10.1109/TITS.2021.3083957.

`[@mayer2023]` M. J. Mayer and D. Yang, "Calibration of deterministic NWP
forecasts and its impact on verification," *International Journal of
Forecasting*, vol. 39, no. 2, pp. 981–991, 2023,
doi: 10.1016/j.ijforecast.2022.03.008.

`[@moreiramatias2016]` L. Moreira-Matias, O. Cats, J. Gama, J. Mendes-Moreira, and
J. Freire de Sousa, "An online learning approach to eliminate Bus Bunching in
real-time," *Applied Soft Computing*, vol. 47, pp. 460–482, 2016,
doi: 10.1016/j.asoc.2016.06.031.

`[@petetin2022]` H. Petetin, D. Bowdalo, P.-A. Bretonnière, M. Guevara, O. Jorba,
J. Mateu Armengol, M. Samso Cabre, K. Serradell, A. Soret, and C. Pérez
Garcia-Pando, "Model output statistics (MOS) applied to Copernicus Atmospheric
Monitoring Service (CAMS) O₃ forecasts: trade-offs between continuous and
categorical skill scores," *Atmospheric Chemistry and Physics*, vol. 22,
pp. 11603–11630, 2022, doi: 10.5194/acp-22-11603-2022.

`[@patton2012]` A. J. Patton and A. Timmermann, "Forecast Rationality Tests Based
on Multi-Horizon Bounds," *Journal of Business & Economic Statistics*, vol. 30,
no. 1, pp. 1–17, 2012, doi: 10.1080/07350015.2012.634337.

`[@rezazada2024]` M. Rezazada, N. Nassir, E. Tanin, and A. Ceder, "Bus bunching: a
comprehensive review from demand, supply, and decision-making perspectives,"
*Transport Reviews*, vol. 44, no. 4, pp. 766–790, 2024,
doi: 10.1080/01441647.2024.2313969.

`[@santos2022]` V. B. Santos, C. E. S. Pires, D. C. Nascimento, and A. R. M. de
Queiroz, "A Decision Tree Ensemble Model for Predicting Bus Bunching," *The
Computer Journal*, vol. 65, no. 8, pp. 2044–2062, 2022,
doi: 10.1093/comjnl/bxab045.

`[@sun2021]` W. Sun, J.-D. Schmöcker, and T. Nakamura, "On the tradeoff between
sensitivity and specificity in bus bunching prediction," *Journal of Intelligent
Transportation Systems*, vol. 25, no. 4, pp. 384–400, 2021,
doi: 10.1080/15472450.2020.1725887.

`[@tcqsm2003]` *Transit Capacity and Quality of Service Manual*, 2nd ed., TCRP
Report 100, Transportation Research Board, 2003, Part 3, ch. 3, p. 3-48,
Exhibit 3-30.

`[@yu2016]` H. Yu, D. Chen, Z. Wu, X. Ma, and Y. Wang, "Headway-based bus bunching
prediction using transit smart card data," *Transportation Research Part C:
Emerging Technologies*, vol. 72, pp. 45–59, 2016,
doi: 10.1016/j.trc.2016.09.007.
