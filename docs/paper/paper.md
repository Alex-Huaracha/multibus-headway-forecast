# Un threshold diseñado para el headway observado hace que la persistencia gane la detección de bunching

## Resumen

Para anticipar el bunching, un modelo predice el headway y lo compara contra un threshold, el valor por debajo del cual un headway cuenta como demasiado corto. Se predijo con una red LSTM el vector de headways de tres corredores de Arequipa, a partir de 152 días de registros GPS, y se comparó contra la persistencia, que repite el último vector observado. A diez minutos, en el corredor E59, el LSTM tuvo un error absoluto medio 1.17 minutos menor, pero el F1 de la persistencia fue 8.8 veces el suyo: el LSTM emitió alarma en el 0.75 % de las posiciones y la persistencia en el 20.8 %. El entrenamiento por error cuadrático deja el headway predicho menos disperso que el headway observado: el mismo corredor queda en nivel A del TCQSM según el headway predicho y en nivel F según el headway observado. Con un threshold pensado para el headway observado, sea la mitad del promedio del vector o un valor fijo en minutos, la persistencia ganó en las doce combinaciones de corredor y horizonte. El área bajo la curva ROC (AUC), que no usa threshold, dio como ganador al LSTM en seis de ellas. Al ajustar el threshold sobre el headway predicho, el ganador volvió a coincidir con el del AUC en casi todos los casos. Por eso, la detección de bunching sobre predicciones debe evaluarse con un threshold ajustado al headway predicho.

---

## I. Introducción

El headway es el tiempo que separa el paso de dos buses consecutivos por un
mismo punto de una ruta, y el bunching es la circulación conjunta de dos buses
que ese tiempo debería mantener separados, que desiguala la espera entre los
pasajeros. Para anticiparlo se siguen dos etapas [@yu2016] [@jiao2023]: un
modelo estima el headway de dentro de unos minutos, el **headway predicho**, y
lo compara contra un **threshold**: el valor por debajo del cual un headway
cuenta como demasiado corto. Si queda por debajo, se anuncia bunching. El GPS
registra después el headway que ocurrió, el **headway observado**, y con él se
comprueba la predicción.

El threshold no tiene un valor acordado [@rezazada2024]: unos trabajos lo fijan
en tiempo, entre veinte segundos y tres minutos, y otros como una fracción del
headway programado, de hasta un cuarto. El procedimiento supone que la segunda
etapa hereda la mejora de la primera, y Yu y colaboradores lo reportan así: al
predecir cinco paradas adelante en lugar de dos, su error subió de dos a seis
minutos y la fracción de eventos detectados bajó del 100 % al 74 % [@yu2016].

En nuestros datos esa suposición no se cumplió. Se predijo con una red LSTM
(*long short-term memory*) el vector de headways de cada corredor: un headway
por cada par de buses consecutivos de una empresa sobre una misma ruta. A diez
minutos, en el corredor E59, el LSTM tuvo un error absoluto medio 1.17 minutos
menor que el de la persistencia, que repite el último vector observado. El
evento se define con un *event threshold*, la mitad del promedio del vector, que
marcó bunching en el 20.8 % de las posiciones del headway observado. Aplicado
sin ajuste al vector predicho, como *unadjusted threshold*, la persistencia
emitió alarma en el 20.8 % de las posiciones y el LSTM en el 0.75 %. Con el F1,
que vale 1 cuando las alarmas coinciden con los eventos y cae hacia 0 si sobran
o faltan, el de la persistencia fue 8.8 veces el del LSTM. Así, el F1 premió al
método que más alarmas emitió y no al que menos se equivocó en minutos.

La causa está en la primera etapa. Un modelo entrenado con error cuadrático
medio, cuando no sabe si un headway será corto o largo, predice un valor
intermedio [@gneiting2011], y la predicción queda **subdispersa**
(*underdispersion*), con menos dispersión que el headway observado
[@mayer2023]. Casi no contiene headways mucho más cortos que los demás, que es
lo que el bunching es, y casi ninguno queda bajo un threshold pensado para la
dispersión del headway observado, aunque el promedio salga del propio vector
predicho.

Ese efecto no se ha medido. Usama y Koutsopoulos predicen el vector completo de
headways de una línea de metro y reportan solo el error en minutos
[@usama2025]; Sun, Schmöcker y Nakamura llegan a la detección, pero puntúan
sin threshold solo a su clasificador probabilístico, y evalúan los métodos que
predicen el headway con un threshold fijo [@sun2021].

Este trabajo lo mide puntuando la misma detección con y sin threshold. Sin
threshold, los headways predichos se ordenan de más corto a más largo, en
proporción al promedio de su vector, y el área bajo la curva ROC (*receiver
operating characteristic*), el AUC, mide si los eventos reales quedan al
principio. Las dos puntuaciones se contradicen: sin threshold, el LSTM ordenó
mejor que la persistencia a diez minutos en los tres corredores y en tres
períodos de prueba distintos.

Mostramos tres cosas. Primero, el headway predicho describe un corredor más
regular que el real: en la escala del *Transit Capacity and Quality of Service
Manual* (TCQSM), el headway predicho queda en nivel A y el headway observado en
nivel F. Segundo, por esa subdispersión, un threshold pensado para el headway
observado, sea la mitad del promedio del vector o un valor fijo en minutos, hace
que el LSTM casi no emita alarmas y que la persistencia gane la evaluación.
Tercero, ajustar el threshold sobre el headway predicho corrige el defecto: el
ganador vuelve a coincidir con el que da el AUC en casi todos los casos.

---

## II. Antecedentes

### A. Predicción de bunching en dos etapas

El procedimiento de dos etapas de la Sección I tiene una formulación canónica:
un headway cuenta como bunching si cae por debajo de la cuarta parte del headway
observado en la primera parada del mismo viaje [@yu2016] [@jiao2023]. El
threshold se fija así sobre la escala del headway observado y se compara contra
el headway predicho. Los trabajos previos evalúan la segunda etapa solo con un
threshold, sin ninguna medida que puntúe sin él [@santos2022].

### B. Subdispersión bajo error cuadrático medio

La subdispersión de la Sección I es un teorema. Un modelo entrenado con error
cuadrático predice la media condicional [@gneiting2011]. Por eso la varianza del
headway observado es la del headway predicho más el error cuadrático esperado,
de modo que el headway predicho sale menos disperso que el headway observado, y
la brecha crece con el horizonte [@patton2012].

### C. Precedentes y delimitación

La subdispersión ya se midió en series escalares [@mayer2023], en seis dominios,
entre ellos el tráfico [@green2026], y en campos espaciales [@bonavita2024].
Petetin y colaboradores muestran su daño sobre una regla de threshold: el método
con mejor error cuadrático es el que peor detecta los episodios altos de ozono
[@petetin2022].

Fuera del transporte, las correcciones publicadas actúan después de la
predicción y no sobre el modelo. Hoffmann y colaboradores recalculan el
threshold como el percentil equivalente dentro de las predicciones
[@hoffmann2018], Petetin y colaboradores ajustan la distribución de las
predicciones a la de las mediciones [@petetin2022], y Roberts y Lean calculan el
percentil dentro de cada campo de lluvia [@roberts2008].

En transporte, Sun, Schmöcker y Nakamura diagnostican que predecir y después
aplicar un threshold falla, con un threshold fijo de un minuto, y reportan el
área bajo la curva solo para su clasificador probabilístico [@sun2021]. Jiao y
colaboradores corrigen el modelo, con un término de clasificación en la pérdida
[@jiao2023]. Este trabajo, en cambio, corrige la regla y no reentrena el modelo.

---

## III. Defecto de la evaluación con el *unadjusted threshold*

### A. Formulación del problema

Lo que se predice es el **vector de headways** del corredor: un headway por cada
par de buses consecutivos que circulan en el mismo sentido, con todas sus
posiciones a la vez. El Apéndice A lo construye desde los registros GPS. Dado el
historial de los últimos $L$ minutos y un contexto de calendario, se busca el
vector del corredor $H$ minutos más adelante:

$$\hat{\mathbf{h}}(t+H) \;=\; f\big(\mathbf{h}(t-L+1), \dots, \mathbf{h}(t);\;
c(t-L+1), \dots, c(t)\big), \qquad L = 12, \tag{1}$$

donde $\mathbf{h}(t)$ es el vector de headways del corredor en el minuto $t$ y
$\hat{\mathbf{h}}(t+H)$ es el vector predicho para $H$ minutos más adelante. El
término $c(t)$ reúne cuatro variables de calendario del minuto $t$: el seno y el
coseno de la hora, y el seno y el coseno del día de la semana. Aquí $f$ es el
modelo ajustado.

Se predice a uno, tres, cinco y diez minutos, con un modelo ajustado por
separado para cada horizonte. El vector no tiene longitud fija, porque la
cantidad de buses varía minuto a minuto, de modo que el error se computa solo
sobre las posiciones con headway válido. **El objetivo que se minimiza es el error
cuadrático**, promediado sobre esas posiciones:

$$\mathcal{L} \;=\; \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}
\big(\hat{h}_i - h_i\big)^{2}, \tag{2}$$

donde $\mathcal{V}$ es el conjunto de posiciones del vector con headway válido en
el instante objetivo, y $|\mathcal{V}|$ es su cardinal. Los términos $\hat{h}_i$
y $h_i$ son el headway predicho y el headway observado en la posición $i$, en la
escala estandarizada por sentido que fija el Apéndice A.2.

Los registros GPS no traen pasajeros ni estado del tránsito, de modo que el
evento se define sobre la geometría del vector. La convención del campo marca el
evento con una fracción de un headway de referencia, programado u observado
(Secciones I y II-A); el TCQSM, en cambio, usa la mitad del programado [@tcqsm2003].
Estos corredores no tienen programación, y aquí la referencia es el promedio del
propio vector en ese instante. Ese promedio refleja la separación normal de cada
corredor, cosa que un threshold fijo en minutos no hace entre corredores de
frecuencias distintas. La fracción de un medio se hereda del TCQSM y la
sustitución es nuestra. El resultado es el ***event threshold***.

Con $\mathcal{V}(t)$ el conjunto de posiciones con headway válido del vector
$\mathbf{h}(t)$, su promedio y el *event threshold* son

$$\bar{h}(t) \;=\; \frac{1}{m}\sum_{j \in \mathcal{V}(t)} h_j(t),
\qquad \tau(t) \;=\; \rho\,\bar{h}(t), \qquad \rho = \tfrac{1}{2}, \tag{3}$$

donde $m = |\mathcal{V}(t)|$ es la cantidad de posiciones con headway válido,
$h_j(t)$ es el headway de la posición $j$ y $\rho$ es la fracción que fija el
threshold $\tau(t)$; los pares sin headway válido del Apéndice A no entran en el
cómputo.
La posición $i$ cuenta como bunching cuando cae por debajo de ese threshold:

$$b_i(t) \;=\; \mathbb{1}\!\left[\, h_i(t) < \tau(t) \,\right],
\qquad \text{definido solo si } m \ge 3, \tag{4}$$

donde $\mathbb{1}[\cdot]$ es la función indicadora. Cada posición con
$b_i(t) = 1$ es un evento, y se dice que la regla la **marca**. La condición
$m \ge 3$ existe porque con dos headways cualquier medida de irregularidad se
reduce a la diferencia entre ellos y no describe un patrón.

El detector aplica esa misma regla al vector predicho de la Ecuación (1), con el
promedio de ese mismo vector fijando el threshold:

$$\hat{b}_i(t) \;=\; \mathbb{1}\!\left[\, \hat{h}_i(t) < \rho\,\bar{\hat{h}}(t)
\,\right], \tag{5}$$

donde $\hat{b}_i(t)$ es la detección emitida sobre la posición $i$ y
$\bar{\hat{h}}(t)$ es el promedio del vector predicho. Cada posición con
$\hat{b}_i(t) = 1$ es una **alarma** (*alarm*) [@moreiramatias2016]. El
threshold sale del vector predicho porque quien opera un corredor no dispone del
headway observado al momento de decidir. La fracción $\rho = \tfrac{1}{2}$ pasa
así sin cambios del headway observado al headway predicho, y a ese threshold,
aplicado sin ajustarlo a la predicción, se le llama el ***unadjusted
threshold*** [@provost2000].

La evaluación compara el detector de la Ecuación (5) contra el indicador de la
Ecuación (4). Sean TP las posiciones con $b_i = \hat{b}_i = 1$, FP las que
tienen $\hat{b}_i = 1$ y $b_i = 0$, y FN las que tienen $b_i = 1$ y
$\hat{b}_i = 0$. La precisión, el recall y el F1 son entonces

$$\mathrm{F}_1 \;=\; \frac{2PR}{P+R}, \qquad
P \;=\; \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \qquad
R \;=\; \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \tag{6}$$

donde la precisión $P$ es la fracción de alarmas que caen sobre un evento, y el
recall $R$ es la fracción de eventos que reciben alarma.

### B. El F1 bajo el *unadjusted threshold*

El F1 de la Ecuación (6) mezcla dos cantidades de naturaleza distinta. Como TP
es la precisión multiplicada por la cantidad de alarmas, al dividir por $n$ el
F1 se reescribe como

$$\mathrm{F}_1 \;=\;
\frac{2\,\mathrm{TP}}{(\mathrm{TP}+\mathrm{FP}) + (\mathrm{TP}+\mathrm{FN})}
\;=\; P \cdot \frac{2q}{q+\pi}, \tag{7}$$

donde $n$ es la cantidad de posiciones evaluadas,
$q = (\mathrm{TP}+\mathrm{FP})/n$ es la **tasa de alarma**, la fracción de
posiciones con alarma, y $\pi = (\mathrm{TP}+\mathrm{FN})/n$ es la **tasa
base**, la fracción de posiciones con evento.

La precisión $P$ mide si las alarmas aciertan. El término $2q/(q+\pi)$
depende solo de cuántas se emiten, y vale 1 cuando la tasa de alarma iguala a la
tasa base. Si la tasa de alarma cae muy por debajo de la tasa base, ese término
se acerca a $2q/\pi$ y arrastra el F1 hacia cero, aunque todas las
alarmas acierten. El **baseline siempre positivo**, que marca bunching en toda
posición, tiene $q = 1$ y $P = \pi$ [@flach2015]. Alcanza así
$\mathrm{F}_1 = 2\pi/(1+\pi)$ sin ningún modelo, y acompaña como baseline a
todo F1 reportado.

Queda por ver qué tasa de alarma recibe cada método. La persistencia predice
$\hat{\mathbf{h}}(t+H) = \mathbf{h}(t)$, de modo que su detector aplica la regla
del evento a un vector observado $H$ minutos antes. Su tasa de alarma es la tasa
base de ese instante anterior. Mientras la tasa base no cambie en $H$ minutos,
el término $2q/(q+\pi)$ vale casi 1 y el F1 de la persistencia queda igual a su
precisión.

El LSTM, en cambio, se ajusta con la Ecuación (2), se aproxima a la media
condicional y queda subdisperso (Sección II-B). Para recibir alarma, un headway
predicho tiene que alejarse de su promedio, hacia abajo, en más de la mitad de
ese promedio. Un vector subdisperso casi no tiene headways tan alejados, de modo
que la tasa de alarma del LSTM cae por debajo de la tasa base y su F1 cae con
ella.

Con el *unadjusted threshold*, el F1 ordena entonces a los métodos por la
dispersión de su vector predicho antes que por su acierto. Un método que predice
con más error puede superar en F1 a uno que predice con menos, aun con menor
precisión, porque copia la dispersión del headway observado.

---

## IV. Puntuación de la detección bajo subdispersión

### A. Puntuación sin threshold

La Ecuación (7) deja el F1 atado a la tasa de alarma, y la tasa de alarma del
*unadjusted threshold*, a la dispersión del vector predicho. La forma más
directa de que esa dispersión deje de decidir el resultado es puntuar sin
threshold, de modo que no quede tasa de alarma que medir. El detector de la
Ecuación (5) compara el cociente $\hat{h}_i/\bar{\hat{h}}$ contra la fracción
$\rho$. Sin fijar $\rho$, ese cociente, calculado dentro de cada vector, ordena
las posiciones de la más corta a la más larga respecto del promedio de su
vector. El AUC [@handtill2001] puntúa ese ordenamiento. Vale 1 si todos los
eventos quedan antes que todas las posiciones sin evento, y 0.5 si el
ordenamiento no informa. Se calcula por **celda**, una combinación de corredor y
horizonte, juntando en un solo ordenamiento las posiciones de todos sus vectores
y de los dos sentidos.

Un AUC de 0.5 es el piso de una predicción sin ninguna información, pero no el
de una predicción sin información temporal. Las posiciones del vector no son
intercambiables, y algunas llevan headways más cortos que otras sin que haga
falta leer la ventana de entrada. El **baseline de promedio histórico por
posición** del Apéndice A.2, repite para cada posición su headway promedio de un
período anterior y fija ese segundo piso. Cumple para el AUC la función que el
baseline siempre positivo cumple para el F1, y acompaña a todo AUC reportado.

### B. *Adjusted threshold*

La otra forma conserva un threshold, pero lo ajusta sobre el headway predicho en
lugar de heredarlo del headway observado, para que la tasa de alarma deje de
depender de su dispersión: es el ***adjusted threshold***, que se prueba de dos
maneras. El ***optimized threshold*** reemplaza la fracción $\rho$ de la
Ecuación (5) por un valor ajustado para cada método y cada celda. El ajuste usa
un **origen de evaluación** anterior: una fecha de fin de entrenamiento con su
propio período de prueba. La evaluación usa tres orígenes y reporta el tercero
como resultado principal. El threshold se ajusta sobre el período de prueba del
origen 2 y se aplica sin cambios al del origen 3. Los dos períodos son disjuntos
y provienen de modelos entrenados por separado, de modo que el período reportado
no informa su propio threshold.

El objetivo del ajuste es el coeficiente de correlación de Matthews (MCC) y no
el F1. Por la Ecuación (7), el F1 premia subir la tasa de alarma, y maximizarlo
sobre una predicción sin información conduce al baseline siempre positivo
[@lipton2014]. El MCC es la correlación entre las alarmas y los eventos: vale 1
cuando coinciden y 0 cuando las alarmas no informan sobre los eventos, como en
el baseline siempre positivo [@chicco2020]. Emitir más alarmas no basta para
subirlo, porque también cuenta las posiciones sin evento ni alarma, los
verdaderos negativos (TN).

La segunda es el ***percentile threshold*** [@roberts2008]. No compara el
headway contra un valor: marca en cada vector las posiciones más cortas hasta
cubrir una fracción fija. Esa fracción es la que la regla de la Ecuación (4)
marcó en promedio en la celda, sobre el origen 2. Aplicado al headway observado,
el percentil define su propio evento, y el detector se puntúa contra él. Como
los dos marcan la misma cantidad de posiciones en cada vector, la tasa de alarma
iguala a la tasa base, y el término $2q/(q+\pi)$ de la Ecuación (7) vale 1 para
todo método.

Ese evento no es el de la Ecuación (4): marca las posiciones más cortas de cada
vector, estén o no por debajo de la mitad de su promedio. El **índice de
Jaccard** mide cuánto coinciden los dos eventos: divide las posiciones que
marcan ambos entre las que marca al menos uno. Vale 0.58 en la mediana de las
doce celdas (Tabla 3), de modo que los dos eventos coinciden solo en parte.

---

## V. Resultados

### A. Datos y métodos comparados

Los datos son 152 días seguidos de registros GPS de tres corredores de Arequipa,
identificados como E2, E4 y E59. Se comparan tres métodos sobre las mismas
muestras. El LSTM es el método bajo estudio. Un conjunto de árboles con refuerzo
de gradiente (**XGBoost**) [@chen2016] sirve de control: se ajusta con el mismo
error cuadrático y no comparte la arquitectura del LSTM, de modo que lo que los
dos tengan en común se debe al ajuste. La persistencia no se ajusta. Salvo
aviso, las cifras son del origen 3, el único sobre el que se ajustó el XGBoost;
el Apéndice A da el detalle. Un método **gana** solo si la diferencia pasa su
prueba del Apéndice B: un intervalo del 95 % que no incluye el cero o, en el
MAE, una prueba de Diebold–Mariano con p < 0.05. Si no la pasa, los dos
**empatan**.

Con cuatro horizontes por corredor hay doce celdas, y se reportan todas. A un
minuto no hay margen para intervenir y la persistencia es más difícil de superar
[@manibardo2022].

### B. Error escalar y subdispersión

El error de la predicción se mide con el error absoluto medio (MAE) sobre las
posiciones de la Ecuación (2). Se reporta en lugar del error cuadrático
porque queda en minutos de headway. A diez minutos, el LSTM bajó el MAE de la
persistencia entre 21 % y 22 %: 1.47 minutos en E2, 1.38 en E4 y 1.17 en E59.
A un minuto ganó la persistencia en E4 y E59, y en E2 empataron, con una
diferencia de 0.07 minutos.

El MAE no describe la forma del vector. El coeficiente de variación (CV) sí: es
la desviación estándar del vector dividida por su promedio,

$$\mathrm{CV}(\mathbf{h}) \;=\; \frac{1}{\bar{h}}
\sqrt{\frac{1}{m-1}\sum_{j \in \mathcal{V}}\big(h_j - \bar{h}\big)^{2}}, \tag{8}$$

donde $\mathcal{V}$, $m$, $h_j$ y $\bar{h}$ son los de la Ecuación (3), y se calcula sobre los
vectores de tres posiciones o más de la Ecuación (4). El CV vale 0 cuando todos
los buses del corredor van igual de separados, y crece a medida que los
headways se desigualan. No tiene unidades, de modo que compara corredores de
frecuencias distintas. Su **sesgo** es el CV del vector predicho menos el del
vector observado en el mismo instante, y un valor negativo indica un vector
predicho más regular que el observado.

En E2 a diez minutos, el CV valió 0.79 sobre el headway observado y 0.16 sobre
el headway predicho por el LSTM. El TCQSM califica la regularidad de un
servicio en niveles de A a F según la dispersión del headway respecto del
programado [@tcqsm2003], y en un corredor sin programación esa dispersión es la
de la Ecuación (8). Leídas en esa escala, las dos cifras ponen al mismo
corredor en nivel A —«service provided like clockwork»— según el headway
predicho, y en nivel F —«most vehicles bunched»— según el headway observado.

El sesgo del CV del LSTM fue negativo en **las doce celdas y los tres
orígenes**, y creció sin excepción con el horizonte (Figura 1). El de la
persistencia, que no se ajusta, quedó dentro de ±0.022: el efecto está en el
ajuste, no en los datos ni en el corredor.

![Sesgo de dispersión contra el horizonte](figuras/subdispersion-vs-horizonte.es.png)

**Fig. 1.** Sesgo del CV, el CV del vector predicho menos el del vector
observado, por método y horizonte. Un panel por corredor, origen 3; un valor negativo es un
vector predicho más regular que el observado.

La Sección II-B da la causa: la varianza del headway observado es la del
headway predicho más la del error. Medida entre las posiciones de cada vector:

$$V_h \;=\; V_{\hat h} + V_e + 2\,C_{\hat h e}, \qquad
r \;=\; \frac{V_{\hat h}}{V_h}, \qquad r_0 \;=\; 1 - \frac{V_e}{V_h}, \tag{9}$$

donde $V_h$, $V_{\hat h}$ y $V_e$ son las varianzas entre posiciones del headway
observado, del headway predicho y del error $e_i = h_i - \hat{h}_i$, promediadas
sobre los vectores de la celda, y $C_{\hat h e}$ es la covarianza entre el
headway predicho y el error. La fracción $r$ es la dispersión del headway
observado que conserva el headway predicho, y $r_0$ la que predice el error;
coinciden si esa covarianza es nula. Sobre las doce celdas, $r$ siguió a $r_0$
con una correlación de 0.993 en el LSTM y de 0.996 en el XGBoost, y fue del
4.5 % (E2 a diez minutos) al 55.2 % (E4 a un minuto) en el LSTM, y del 4.0 % al
54.7 % en el XGBoost.

### C. Tasa de alarma y precisión bajo el *unadjusted threshold*

En E59 a diez minutos, el caso de la Sección I, el LSTM emitió alarma en el 0.75
% de las posiciones, contra una tasa base del 20.8 %. Acertó el 49 % de sus
alarmas y la persistencia el 30 %, pero el F1 invirtió ese orden: 0.303 para la
persistencia contra 0.034 para el LSTM. La Ecuación (7) da la causa: el término
$2q/(q+\pi)$ valió 1.00 para la persistencia y 0.07 para el LSTM, y ninguna
diferencia de precisión compensa esa caída.

El patrón se repitió en las doce celdas de la Tabla 1: el LSTM tuvo más
precisión que la persistencia y una tasa de alarma por debajo de la tasa base, y
el F1 de la persistencia fue de 1.5 a 253 veces el suyo. La Figura 2 muestra cómo esa tasa cae
al alargar el horizonte.

E2 a diez minutos es el caso extremo: sobre 15 245 eventos, la persistencia
emitió 15 083 alarmas y el LSTM, **catorce**. Diez de las catorce acertaron, una
precisión del 71 % (intervalo del Apéndice B, 42 % a 92 %) contra una tasa base
del 30 %, y aun así el F1 de la persistencia fue 253 veces el suyo.

El F1 tampoco premia a la persistencia por detectar bien. El baseline siempre
positivo, que no usa ningún modelo, tuvo un F1 mayor que el de ella en 5 de las
doce celdas, marcadas con † en la Tabla 1, y en 15 de las 36 combinaciones de celda y origen.

![Tasa de alarma contra tasa real del evento](figuras/artefacto-threshold.es.png)

**Fig. 2.** Tasa de alarma de la persistencia y del LSTM con el *unadjusted
threshold*, contra la tasa base (punteada), por horizonte. Un panel por
corredor, origen 3.

**Tabla 1.** Detección con el *unadjusted threshold*, con el F1 del baseline
siempre positivo al lado. Factor es el F1 de la persistencia dividido por el del
LSTM.

| Corredor | h | Tasa base | F1 baseline | F1 persistencia | F1 LSTM | Factor |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2 | 1 | 0.299 | 0.460 | 0.581 | 0.207 | 2.8× |
| E2 | 3 | 0.301 | 0.462 | 0.414&nbsp;† | 0.038 | 11× |
| E2 | 5 | 0.300 | 0.462 | 0.375&nbsp;† | 0.011 | 36× |
| E2 | 10 | 0.303 | 0.465 | 0.332&nbsp;† | 0.001 | 253× |
| E4 | 1 | 0.183 | 0.310 | 0.686 | 0.466 | 1.5× |
| E4 | 3 | 0.173 | 0.295 | 0.486 | 0.177 | 2.7× |
| E4 | 5 | 0.172 | 0.294 | 0.381 | 0.066 | 5.8× |
| E4 | 10 | 0.179 | 0.304 | 0.268&nbsp;† | 0.015 | 18× |
| E59 | 1 | 0.212 | 0.350 | 0.620 | 0.308 | 2.0× |
| E59 | 3 | 0.209 | 0.345 | 0.469 | 0.130 | 3.6× |
| E59 | 5 | 0.208 | 0.344 | 0.405 | 0.083 | 4.9× |
| E59 | 10 | 0.208 | 0.344 | 0.303&nbsp;† | 0.034 | 8.8× |

† El F1 del baseline siempre positivo es mayor que el de la persistencia en estas celdas.

### D. Detección sin threshold, acotada por el promedio histórico por posición

Sin threshold, el orden de la Tabla 1 se invierte a diez minutos (Tabla 2 y
Figura 3). Puntuado mediante el AUC de la Sección IV-A, **el LSTM ganó en las
nueve combinaciones de corredor y origen a diez minutos**, con diferencias de
0.033 a 0.061. En el origen 3 ganó en 6 de las 12 celdas, y los tres orígenes
coincidieron en el resultado de 10 de las 12. A un
minuto ganó la persistencia en los tres corredores y los tres orígenes, como
en el MAE de E4 y E59.

El promedio histórico por posición de la Sección IV-A, que no lee la ventana de
entrada, acota esa ventaja. En E4 y E59 el baseline queda cerca del azar, entre
0.49 y 0.52, y el LSTM le ganó en las ocho celdas. En E2 el baseline queda por
encima del azar en todos los horizontes, y **a diez minutos el LSTM pierde
contra él, 0.565 contra 0.579**. Es la única celda donde ocurre, la misma del
caso extremo de la Sección V-C.

![AUC de detección contra el promedio histórico por posición](figuras/deteccion-contra-baseline.es.png)

**Fig. 3.** AUC de detección del LSTM y de la persistencia contra el promedio
histórico por posición (punteado), por horizonte. Un panel por corredor, origen
3; la línea en 0.5 es el azar.

**Tabla 2.** Diferencias del LSTM, sin threshold y con el *optimized threshold*,
con su intervalo de confianza del 95 %. Un signo positivo favorece al LSTM, y un
intervalo que contiene el cero es un empate (Sección V-A).

| Corredor | h | Δ AUC frente a la persistencia | Δ AUC frente al promedio histórico por posición | Δ MCC *optimized* frente a la persistencia |
| :--- | ---: | :---: | :---: | :---: |
| E2 | 1 | -0.009 [-0.015, -0.004] | +0.126 [+0.117, +0.138] | -0.091 [-0.106, -0.078] |
| E2 | 3 | +0.031 [+0.025, +0.036] | +0.050 [+0.040, +0.061] | +0.018 [+0.005, +0.028] |
| E2 | 5 | +0.037 [+0.031, +0.042] | +0.021 [+0.013, +0.031] | +0.037 [+0.026, +0.046] |
| E2 | 10 | +0.037 [+0.026, +0.047] | -0.013 [-0.023, -0.003] | +0.058 [+0.039, +0.073] |
| E4 | 1 | -0.022 [-0.027, -0.016] | +0.288 [+0.273, +0.305] | -0.140 [-0.150, -0.130] |
| E4 | 3 | -0.017 [-0.025, -0.009] | +0.182 [+0.167, +0.199] | -0.106 [-0.122, -0.086] |
| E4 | 5 | -0.001 [-0.010, +0.008] | +0.128 [+0.113, +0.144] | -0.064 [-0.083, -0.043] |
| E4 | 10 | +0.047 [+0.030, +0.063] | +0.083 [+0.068, +0.097] | +0.015 [-0.005, +0.033] |
| E59 | 1 | -0.021 [-0.025, -0.016] | +0.262 [+0.253, +0.271] | -0.154 [-0.164, -0.144] |
| E59 | 3 | 0.000 [-0.005, +0.005] | +0.195 [+0.186, +0.204] | -0.091 [-0.100, -0.081] |
| E59 | 5 | +0.017 [+0.012, +0.022] | +0.173 [+0.162, +0.184] | -0.044 [-0.053, -0.036] |
| E59 | 10 | +0.061 [+0.054, +0.067] | +0.146 [+0.133, +0.159] | +0.042 [+0.033, +0.052] |

### E. Detección con el *adjusted threshold*

El *optimized threshold* de la Sección IV-B, que no toca el modelo, llevó al
LSTM a ganar en MCC 4 de las 12 celdas, entre ellas E2 y E59 a diez minutos, y
a empatar en E4 a diez minutos (última columna de la Tabla 2).

La Tabla 3 compara las cinco reglas sobre la misma población. Las tres que
aplican al headway predicho un threshold diseñado para el headway observado
colapsaron. Una de ellas es un threshold fijo en minutos. Moreira-Matias y
colaboradores lo fijan en la cuarta parte del headway programado
[@moreiramatias2016]. Estos corredores no tienen horario, así que aquí se usó la
cuarta parte del headway mediano observado de cada corredor y sentido. Con ese
threshold, el LSTM tuvo un A/E de 0.001 y no emitió ninguna alarma en dos
celdas. Los dos *adjusted thresholds* no colapsaron, y la
persistencia no colapsó bajo ninguna. La Figura 4 muestra por qué: con el
*unadjusted threshold*, el threshold sobre el headway predicho quedó a menos de
0.35 minutos del threshold sobre el headway observado, mientras que el
*percentile threshold* lo sube entre 1.4 y 4.1 minutos por encima, hasta donde
caen los headways predichos.

![Threshold en minutos de cada regla](figuras/threshold-en-minutos.es.png)

**Fig. 4.** Threshold en minutos que aplica cada regla, promediado sobre las
posiciones, sobre el headway observado (discontinua) y sobre el headway predicho
por el LSTM (continua), por horizonte. Un panel por corredor, origen 3.

Contando solo las celdas sin empate, el
*optimized threshold* reproduce el ganador de la Sección V-D en 8 de 9 y el
*percentile threshold* en 9 de 10, con E59 a cinco minutos como única
discrepancia; el *unadjusted threshold* y su variante en minutos, en 4 de 10, y
su variante con promedio observado, en 5 de 7 (Tabla 3). Aun así, el percentil
no convierte al LSTM en mejor detector: pierde contra la persistencia en
siete de las doce celdas, entre ellas las tres de un minuto.

**Tabla 3.** Las cinco reglas, sobre la misma población y el mismo origen. A/E
es el sesgo de frecuencia (*frequency bias*) [@ferro2011]: la tasa de alarma
dividida por la tasa base, el cociente $q/\pi$ de la Ecuación (7). Vale uno
cuando el detector avisa tan seguido como el evento ocurre; pers. es la
persistencia. La variante con promedio observado fija el threshold del evento y
el de la alarma con el promedio del último vector observado, de modo que cambia
también el evento. La variante en minutos marca el evento y la alarma por debajo
de la cuarta parte del headway mediano observado en el origen 2, por corredor y
sentido. Jaccard es el índice de la Sección IV-B contra el evento de
la Ecuación (4). Coincide cuenta, sobre las celdas sin empate, las que tienen el
mismo ganador que sin threshold. Cada valor
es la mediana de las doce combinaciones de corredor y horizonte, salvo Coincide.

| Regla | A/E LSTM | A/E pers. | MCC LSTM | Jaccard | Coincide |
| :--- | ---: | ---: | ---: | ---: | :---: |
| *Unadjusted* | 0.079 | 1.011 | 0.100 | 1.000 | 4/10 |
| *Unadjusted*, promedio observado | 0.153 | 0.980 | 0.143 | 0.710 | 5/7 |
| *Unadjusted*, en minutos | 0.001 | 1.062 | 0.004 | 0.351 | 4/10 |
| *Optimized* | 1.125 | 1.140 | 0.198 | 1.000 | 8/9 |
| *Percentile* | 1.000&nbsp;‡ | 1.000&nbsp;‡ | 0.210 | 0.580 | 9/10 |

‡ Vale uno por construcción: la alarma y el evento marcan la misma cantidad de
posiciones en cada vector (Sección IV-B).

---

## VI. Discusión

Con el *unadjusted threshold*, el F1 ordenó a los métodos por la dispersión de
su vector predicho y no por su acierto. Para evaluar la detección de bunching
sobre predicciones sugerimos cuatro prácticas: reportar el AUC junto a todo F1
(Sección V-D); acompañar cada puntuación con un baseline que no use la
predicción (Secciones V-C y V-D); ajustar sobre el headway predicho todo
threshold que la operación necesite (Sección V-E); y reportar la tasa de alarma
junto a la tasa base, porque su cociente dice cuánto del F1 mide la cantidad de
alarmas (Ecuación 7).

El evento no se validó contra un registro de incidentes, que estos corredores no
producen, y las cifras de detección valen para el evento así definido. La
tercera práctica no depende de esa definición: el threshold diseñado para el
headway observado deja de avisar tanto si es una fracción del promedio como si
es un valor en minutos.

La subdispersión admite dos lecturas alternativas al ajuste por error
cuadrático. El ruido de medición del eje del corredor y del sentido de marcha la
agranda, pero la compresión siguió al término del error de la Ecuación (9) en
las doce celdas (Sección V-B), de modo que el dataset fija el tamaño del efecto
y no su existencia. El azar tampoco la explica: cada modelo se entrenó con una
sola semilla, y el efecto se repitió en los tres corredores y los tres orígenes,
cada uno con su propio entrenamiento. El dataset cubre tres corredores de una
sola ciudad durante 152 días, y el promedio histórico por posición se ajustó
sobre un solo origen anterior.

---

## VII. Conclusión

Este trabajo mostró que las predicciones entrenadas por error cuadrático
describen un corredor más regular que el real: el mismo corredor queda en nivel
A del TCQSM según el headway predicho y en nivel F según el headway observado.
Mostró también que, por esa subdispersión, el *unadjusted threshold* hace que el
F1 ordene a los métodos por cuántas alarmas emiten y no por cuántas aciertan, y
que a diez minutos la puntuación sin threshold invierte ese orden. Para
corregirlo, ajustamos el threshold sobre el headway predicho, optimizado en un
origen anterior o como un percentil de cada vector, y con ambas el ganador
vuelve a coincidir con el que da el AUC en casi todos los casos. Esperamos que la detección de bunching sobre
predicciones se evalúe en adelante con un *adjusted threshold*.

---

## VIII. Declaraciones

Se usaron herramientas asistidas por inteligencia artificial generativa para la
redacción del texto y para la verificación de las citas contra sus fuentes. El
diseño experimental, la implementación, las cifras reportadas y su
interpretación fueron revisados y verificados por los autores, que asumen la
responsabilidad del contenido final.

---

## Apéndice A. Datos, métodos comparados y protocolo

### A.1. Datos y construcción del headway

El trabajo usa los registros GPS de tres corredores del Sistema Integrado de
Transporte de Arequipa —E2, E4 y E59, uno por empresa operadora—, con 90 buses
en total. Cada bus emite su coordenada **cada 20 segundos** durante 152 días
seguidos, del 1 de octubre de 2023 al 29 de febrero de 2024.

Estos registros no traen la lista de paradas ni los horarios de paso con que se
mide habitualmente el headway: cada bus emite su identificador, el instante y su
coordenada. El headway se construye entonces desde la posición de cada bus a lo
largo del corredor, como en Andres y Nair [@andres2017], con una diferencia: su
ciudad publica la geometría de la ruta en GTFS y la nuestra no, de modo que el
eje del corredor —la línea que los buses siguen— se ajusta de los propios
registros [@quek2020] [@biagioni2012]. Cada coordenada se proyecta sobre ese
eje, la referenciación lineal de la norma ISO 19148 [@iso19148], y queda su
**coordenada de arco** $s$: los metros recorridos sobre el eje. El sentido de
marcha es el signo del cambio de $s$. En E2 y E59 la ida y la vuelta circulan
por calles paralelas, la dificultad que Andres y Nair señalan para asignar el
bus de adelante usando solo GPS [@andres2017], y ahí el eje se ajusta una vez
por sentido. La posición de cada bus se interpola a cada minuto, y cada minuto
queda descrito por un **snapshot**: la coordenada de todos los buses del
corredor.

Sobre ese snapshot, para un par de buses consecutivos en el mismo sentido —el de
adelante $i-1$, el de atrás $i$— en el minuto $t$:

$$t_{c} = \max\{\, t' \le t \;:\; s_{i-1}(t') = s_{i}(t) \,\},
\qquad h_i(t) = t - t_{c}, \tag{10}$$

donde $s_{i-1}$ y $s_{i}$ son las coordenadas de arco del bus de adelante y del
de atrás. El instante $t_{c}$ es el último en que el de adelante pasó por la
coordenada que el de atrás ocupa en $t$, y $h_i(t)$ es el headway resultante. La
definición es la de Pilachowski [@pilachowski2009], que Andres y Nair evalúan en
la coordenada del bus de atrás [@andres2017], y la Figura 5 la ilustra. Si no
existe tal $t_{c}$, o si $h_i(t)$ supera los treinta minutos, el par queda sin valor.

![El headway como cruce hacia atrás](figuras/esquema-headway.es.png)

**Fig. 5.** El headway de la Ecuación (10) sobre dos trayectorias ilustrativas:
el tiempo entre el paso del bus de adelante por la coordenada $s_i(t)$ y la
llegada del de atrás a ella.

Ese headway describe un solo par. En cada snapshot, los buses de un mismo
sentido se ordenan por su coordenada de arco, y con $N$ buses quedan $N-1$
pares: el vector de headways ordenado desde el frente. Ese orden numera las
posiciones del vector —la primera es la del par que va más adelante—, y un par
sin headway válido conserva su posición con «sin valor», para que el orden no
dependa de cuántos pares resolvieron.

El máximo de treinta minutos existe porque, cuando la trayectoria del bus de
adelante tiene huecos, la búsqueda del cruce retrocede hasta una vuelta
anterior, horas antes. La cobertura —la
fracción de pares evaluados con headway válido— es del 63.5 % en E2, del 64.8 %
en E4 y del 77.1 % en E59: 3 938 174 pares sobre 5 601 738 evaluados. Una
posición sin headway válido se enmascara.

### A.2. Métodos comparados

El LSTM y el XGBoost se ajustan por separado en cada celda, con un modelo por
corredor para los dos sentidos. Los estadísticos de estandarización son por
sentido, y con ellos el headway predicho vuelve a minutos.

- **LSTM:** 32 unidades ocultas, una o dos capas según la celda, tasa de
  aprendizaje de 5 × 10⁻⁴, lotes de 128 y semilla 42. En dos de los tres
  corredores, su configuración viene de una búsqueda previa que no se rehízo
  sobre las muestras definitivas.
- **XGBoost:** solo en el origen 3, hasta 800 rondas, parada temprana tras 40
  sin mejora y semilla 42, con la mejor de veinticuatro configuraciones por celda
  elegida en validación.
- **Persistencia:** repite el último vector observado y no tiene parámetros.
- **Promedio histórico por posición** [@rodrigues2022]: repite en cada minuto
  del período de prueba el headway promedio que cada posición registró en el
  período de prueba del origen 2, sin leer la ventana de entrada.

### A.3. Protocolo de evaluación

La partición es por fecha y nunca al azar. Los tres orígenes comienzan el 1 de
octubre de 2023, y sus períodos de prueba no se solapan:

| Origen | Entrenamiento | Validación | Prueba |
| :---: | :--- | :--- | :--- |
| 1 | hasta el 30 nov 2023 (61 días) | 1–22 dic | 23 dic – 13 ene |
| 2 | hasta el 22 dic 2023 (83 días) | 23 dic – 13 ene | 14 ene – 4 feb |
| 3 | hasta el 15 ene 2024 (107 días) | 16 ene – 7 feb | 8–29 feb |

Como los entrenamientos están anidados, los tres orígenes miden estabilidad
frente al período de prueba y no son réplicas independientes. Tres condiciones
evitan fugas de información:

- **Continuidad estricta:** una muestra entra solo si sus minutos son
  consecutivos, para que la ventana no atraviese un hueco de señal. Retiene
  entre el 81.9 % y el 90.2 % de los snapshots de prueba.
- **Población compartida:** todos los métodos se puntúan sobre las mismas filas,
  y el entrenamiento aborta si el resumen SHA-256 de la lista de muestras no
  coincide con el registrado.
- **Tope:** el percentil 99 del headway de entrenamiento se aplica como techo a
  las tres particiones, y afecta entre el 0.78 % y el 1.11 % de los objetivos.

---

## Apéndice B. Pruebas estadísticas

Toda comparación entre métodos es pareada, sobre las mismas muestras, y agrupa
por día de servicio: las muestras de un mismo día comparten clima, incidentes y
demanda, y eso lleva el tamaño efectivo de entre 75 747 y 240 907 filas a los 22
días de prueba.

- **MAE:** prueba de Diebold–Mariano [@diebold1995] sobre el diferencial de
  pérdida por muestra, con la corrección de Harvey–Leybourne–Newbold
  [@harvey1997].
- **AUC y MCC:** como no se descomponen en una pérdida por muestra, intervalo
  percentil al 95 % sobre dos mil remuestreos de días con reemplazo.
- **Precisión:** intervalo exacto de Clopper–Pearson [@clopper1934] al 95 %,
  porque puede descansar sobre muy pocas alarmas. Una celda sin alarmas no
  recibe intervalo.

---

## Referencias

`[@andres2017]` M. Andres and R. Nair, "A predictive-control framework to address
bus bunching," *Transportation Research Part B: Methodological*, vol. 104,
pp. 123–148, 2017, doi: 10.1016/j.trb.2017.06.013.

`[@biagioni2012]` J. Biagioni and J. Eriksson, "Inferring Road Maps from Global
Positioning System Traces: Survey and Comparative Evaluation," *Transportation
Research Record*, vol. 2291, no. 1, pp. 61–71, 2012, doi: 10.3141/2291-08.

`[@bonavita2024]` M. Bonavita, "On some limitations of data-driven weather
forecasting models," arXiv:2309.08473, 2023.

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

`[@ferro2011]` C. A. T. Ferro and D. B. Stephenson, "Extremal Dependence
Indices: Improved Verification Measures for Deterministic Forecasts of Rare
Binary Events," *Weather and Forecasting*, vol. 26, no. 5, pp. 699–713, 2011,
doi: 10.1175/WAF-D-10-05030.1.

`[@flach2015]` P. A. Flach and M. Kull, "Precision-Recall-Gain Curves: PR
Analysis Done Right," in *Advances in Neural Information Processing Systems 28*,
2015, pp. 838–846.

`[@gneiting2011]` T. Gneiting, "Making and Evaluating Point Forecasts," *Journal
of the American Statistical Association*, vol. 106, no. 494, pp. 746–762, 2011,
doi: 10.1198/jasa.2011.r10138.

`[@green2026]` S. Green, Z. Abdallah, and T. Silva Filho, "Expectations vs.
Realities: The Cost of MSE-Optimal Forecasting Under Conditional Uncertainty,"
arXiv:2606.04342, 2026.

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

`[@pilachowski2009]` J. M. Pilachowski, "An Approach to Reducing Bus Bunching,"
Ph.D. dissertation, Univ. of California, Berkeley, CA, USA, 2009. [Online].
Available: https://escholarship.org/uc/item/6zc5j8xg

`[@provost2000]` F. Provost, "Machine Learning from Imbalanced Data Sets 101,"
in *Proc. AAAI Workshop on Learning from Imbalanced Data Sets*, 2000, pp. 1–3.

`[@quek2020]` W. L. Quek, N. N. Chung, V.-L. Saw, and L. Y. Chew, "Analysis and
simulation of intervention strategies against bus bunching by means of an
empirical agent-based model," arXiv:2004.13022, 2020.

`[@rezazada2024]` M. Rezazada, N. Nassir, E. Tanin, and A. Ceder, "Bus bunching: a
comprehensive review from demand, supply, and decision-making perspectives,"
*Transport Reviews*, vol. 44, no. 4, pp. 766–790, 2024,
doi: 10.1080/01441647.2024.2313969.

`[@roberts2008]` N. M. Roberts and H. W. Lean, "Scale-selective verification of
rainfall accumulations from high-resolution forecasts of convective events,"
*Monthly Weather Review*, vol. 136, no. 1, pp. 78–97, 2008, doi:
10.1175/2007MWR2123.1.

`[@rodrigues2022]` F. Rodrigues, "On the importance of stationarity, strong
baselines and benchmarks in transport prediction problems," arXiv:2203.02954,
2022.

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

`[@usama2025]` M. Usama and H. Koutsopoulos, "Real Time Headway Predictions in
Urban Rail Systems and Implications for Service Control: A Deep Learning
Approach," arXiv:2510.03121, 2025.

`[@yu2016]` H. Yu, D. Chen, Z. Wu, X. Ma, and Y. Wang, "Headway-based bus bunching
prediction using transit smart card data," *Transportation Research Part C:
Emerging Technologies*, vol. 72, pp. 45–59, 2016,
doi: 10.1016/j.trc.2016.09.007.
