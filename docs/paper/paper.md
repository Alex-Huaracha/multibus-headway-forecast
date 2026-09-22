# Compresión de dispersión en la predicción del vector de headways: el punto de operación, y no el modelo, determina la detección de bunching

## Resumen

_(pendiente — se escribe al final)_

---

## I. Introducción

El headway es el tiempo que separa el paso de dos buses consecutivos por un
mismo punto de una ruta. El bunching es la circulación conjunta de dos buses
que ese tiempo debería mantener separados, y desiguala la espera entre los
pasajeros de esa ruta. Rezazada y colaboradores lo atribuyen a la congestión,
a la demanda atípica, a la acumulación de pasajeros y al comportamiento del
conductor [@rezazada2024]. Trompet, Liu y Graham comparan doce empresas de bus
urbano, y las que publican un indicador de servicio lo definen sobre la
regularidad agregada del recorrido y no sobre un headway aislado [@trompet2011].

La predicción de ese evento sigue un procedimiento de dos etapas: primero se
estima el headway futuro, y después se lo compara contra un umbral que decide
si hay evento. Yu y colaboradores fijan su formulación canónica [@yu2016], y la
literatura la repite sobre corredores y modelos distintos [@jiao2023]. Su
segunda etapa no tiene un valor acordado: los umbrales publicados van desde
veinte segundos hasta un cuarto del headway programado [@rezazada2024].

Ese procedimiento deja dos huecos. Usama y Koutsopoulos predicen el vector
completo de headways de una línea de metro con una red profunda, y reportan
solo el error en minutos, sin convertir lo predicho en un indicador de evento
[@usama2025]. Sun, Schmöcker y Nakamura sí llegan a la detección, y dejan
pendiente construir la curva que compararía a los métodos basados en headway
sin fijar un punto de operación [@sun2021]. Queda sin medir qué le hace el
error de la primera etapa a la decisión de la segunda.

La primera etapa introduce el defecto que la segunda hereda. Un modelo
entrenado con error cuadrático medio tiende al promedio condicional, y la
dispersión entre los buses de un mismo instante —la cantidad que el bunching
altera— sale comprimida. Un umbral trasladado sin cambios a esa distribución
encogida encuentra pocos valores que lo crucen: el detector casi deja de
emitir avisos aunque el error en minutos mejore. La regla de este trabajo
divide por el promedio del propio vector predicho: es relativa, y aun así la
tasa del evento cae.

Este trabajo mide ese efecto y separa lo que aporta el modelo de lo que aporta
el punto de operación. Predice el vector completo de headways de un corredor
—los buses de una empresa que circulan sobre una misma ruta— con una red
recurrente. La regla de la Sección III-B convierte lo predicho en un indicador
de bunching, y la evaluación puntúa esa detección con y sin umbral. Las dos
puntuaciones se contradicen: con el umbral del evento observado trasladado sin
cambios, la persistencia llega a superar a la red por un factor de 253 en el
F1. Sin umbral, la misma predicción ordena mejor que la persistencia en las
nueve combinaciones de corredor y origen a diez minutos. La Sección II-C
delimita cuánto del mecanismo que este trabajo mide ya estaba publicado.
Nuestras contribuciones son cuatro:

- Medimos esa compresión sobre el vector de headways y la aislamos con la
  persistencia como control de compresión nula.
- Invertimos la fórmula de calidad de servicio del *Transit Capacity and Quality
  of Service Manual* (TCQSM) y la aplicamos a lo predicho en lugar de a lo
  observado.
- Atamos esa fórmula a una regla de evento **relativa pero no preservadora de
  tasa**, que se recalcula con cada vector que evalúa.
- Mostramos que el daño alcanza a toda regla que nombre una cantidad de
  minutos, y proponemos una regla de cuota sobre las posiciones más cortas que
  reproduce el veredicto sin umbral sin ajustar ningún parámetro.

---

## II. Antecedentes

### A. Predicción de bunching en dos etapas

El bunching se predice en dos etapas: la primera estima el headway que separará
a dos buses en un instante futuro y minimiza un error en minutos, y la segunda
lo compara contra un umbral y decide una clase. Yu y colaboradores dan la
formulación canónica, con el headway de la primera parada del mismo viaje como
denominador y un cuarto como fracción [@yu2016]. Jiao, Shen y Zhang heredan esa
misma regla, y su pérdida suma un término de clasificación, porque una
pérdida atenta solo al error de regresión trata como ruido los casos que la
regla marca [@jiao2023]. La segunda etapa se evalúa en un punto de operación
único: ninguna de las ocho filas con que Santos y colaboradores resumen el
subcampo registra una medida que puntúe el ordenamiento sin fijar antes un
umbral [@santos2022]. La excepción es el clasificador probabilístico de Sun,
Schmöcker y Nakamura [@sun2021], que la Sección II-C delimita; este trabajo
puntúa las mismas predicciones con el umbral y sin él, para aislar el punto de
operación como la variable bajo prueba.

### B. Compresión de la dispersión bajo error cuadrático medio

La primera etapa de ese procedimiento arrastra una propiedad conocida, y es un
teorema y no una regularidad empírica: lo ajustado para minimizar el error
cuadrático sale menos disperso que la cantidad que predice. La predicción
óptima es la media condicional [@gneiting2011], y la varianza del objetivo se
descompone en la de esa predicción más el error cuadrático esperado, con una
compresión que crece al alargar el horizonte [@patton2012]. Esa compresión
está medida sobre la varianza temporal de una serie escalar [@mayer2023],
sobre conjuntos de instancias en seis dominios, entre ellos el tráfico
[@green2026], y sobre la dispersión transversal de un campo espacial
[@bonavita2024].

El daño sobre una regla de umbral también está documentado: el método con mejor
error cuadrático es el que peor detecta los episodios altos de ozono, porque
subestima la variabilidad [@petetin2022]. Lo que no encontramos es esa medición
sobre el vector de headways de un corredor, ni un control que separe la
compresión del resto del procedimiento; la Sección V-B usa la persistencia
para eso.

### C. Correcciones del umbral y delimitación

El efecto de la compresión sobre una regla de umbral tiene dos correcciones
publicadas fuera del transporte, y ambas alinean cuantiles entre lo predicho y
lo observado: uno reubica el umbral en el valor que su percentil ocupa dentro de
lo predicho [@hoffmann2018], y el otro lleva la distribución de lo predicho a la
de lo observado con el umbral quieto [@petetin2022]. El *Extreme Forecast Index*
declara extremo un pronóstico comparándolo contra la climatología **del propio
modelo** [@ecmwffug]: referir el umbral a lo que el modelo produce no es
entonces nuevo.

Lo que ninguna de las tres prácticas enfrenta es un umbral que se mueva **dentro
de la instancia que evalúa**. Esa alineación tiene además una propiedad que
separa a esas correcciones de la nuestra: un umbral definido sobre un cuantil
queda libre de sesgo por construcción [@hoffmann2018], porque un cuantil
**conserva la frecuencia del evento** bajo cualquier transformación monótona
de lo predicho. Una fracción del promedio no la conserva —la Sección III-B
formaliza por qué—, y ese es el caso que este trabajo mide: relativo sin ser
preservador de tasa.

Dentro del transporte el precedente más cercano es Sun, Schmöcker y Nakamura:
diagnostican que el paradigma de predecir y umbralizar falla, y reportan el área
bajo la curva para su clasificador probabilístico [@sun2021]. Dos rasgos separan
ese trabajo del nuestro. Su etiqueta es un umbral absoluto de un minuto y no una
regla relativa al propio vector, de modo que la compresión alcanza al valor
comparado y no al umbral. Y su diseño no declara ninguna ventana anterior
disjunta sobre la cual su corte se ajuste. El umbral de Jiao y colaboradores es
relativo pero se ancla en una observación fija, y su reparación cambia el
objetivo que el modelo optimiza [@jiao2023]. Esa distinción entre el umbral
anclado y el que se mueve con el vector es la que la Sección III-B formaliza.
Ahí interviene este documento: recalibra el punto de operación sobre un período
anterior disjunto, sin reentrenar ni cambiar el objetivo.

---

## III. Formulación y definición del evento

Los datos, los métodos comparados y el protocolo de partición quedan en el
Apéndice A.

### A. Formulación de la tarea de predicción

Lo que predecimos es el **vector de headways** del corredor: un headway por cada
par de buses consecutivos que circulan en el mismo sentido, todas sus posiciones
a la vez y no un promedio. El Apéndice A lo construye desde los registros GPS,
que son la única entrada disponible. Dado el historial de los últimos $T$
minutos y un contexto de calendario, se busca el vector del corredor $H$ minutos
más adelante:

$$\hat{\mathbf{h}}(t+H) \;=\; f\big(\mathbf{h}(t-T+1), \dots, \mathbf{h}(t);\;
c(t-T+1), \dots, c(t)\big), \qquad T = 12, \tag{1}$$

donde $\mathbf{h}(t)$ es el vector de headways del corredor en el minuto $t$ y
$\hat{\mathbf{h}}(t+H)$ es el vector predicho para $H$ minutos más adelante. El
término $c(t)$ reúne cuatro variables de calendario del minuto $t$: el seno y el
coseno de la hora, y el seno y el coseno del día de la semana. Aquí $f$ es el
modelo ajustado.

Se predice a cuatro horizontes —uno, tres, cinco y diez minutos— con un modelo
ajustado por separado para cada uno, sin recursión. A un minuto la predicción no
deja margen de intervención, y es el régimen donde repetir el último vector
observado es difícil de superar [@manibardo2022]; ese horizonte queda como
referencia, y las afirmaciones operativas de la Sección V-G se leen sobre los de
cinco y diez. El vector no tiene longitud fija, porque $N$ varía minuto a
minuto. El modelo emite entonces una salida de longitud fija y el error se
computa solo sobre las posiciones donde hay bus. **El objetivo que se minimiza
es el error cuadrático**, promediado sobre esas posiciones válidas:

$$\mathcal{L} \;=\; \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}
\big(\hat{h}_i - h_i\big)^{2}, \tag{2}$$

donde $\mathcal{L}$ es la pérdida que el ajuste minimiza, $\mathcal{V}$ es el
conjunto de posiciones del vector con bus asignado en el instante objetivo, y
$|\mathcal{V}|$ es su cardinal. Los términos $\hat{h}_i$ y $h_i$ son el valor
predicho y el observado en la posición $i$, expresados en la escala tipificada
por sentido que fija el Apéndice A, sección B y no en minutos.

### B. Definición del evento de bunching

Las causas del bunching no son observables en estos registros GPS, que no
traen pasajeros, ocupación ni estado del tránsito, de modo que el evento se
define sobre la geometría del vector de headways y no sobre lo que la produjo.
Es una propiedad del patrón colectivo y no de un bus, y se manifiesta en
posiciones del vector de la Sección III-A: un mismo instante puede llevar
varias posiciones afectadas a la vez.

La convención del campo marca el evento con una fracción del headway programado:
un cuarto en las formulaciones más citadas [@moreiramatias2016], y la mitad en
el TCQSM [@tcqsm2003]. Estos corredores no tienen programación, y sustituir ese
denominador por uno observado en el propio corredor es la práctica que la
Sección II-A recoge. Aquí se sustituye por el promedio del propio vector en ese
instante: la fracción es heredada del TCQSM y la sustitución del denominador es
nuestra. El promedio vigente cumple la función de la programación —fijar la
separación normal del corredor—, que un umbral fijo en minutos no cumple entre
corredores de frecuencias distintas. **Un headway cuenta como bunching si cae
por debajo de la mitad de ese promedio.** Ese valor es el umbral relativo del
evento, y se mueve con cada vector.

El vector de la Sección III-A se escribe por componentes como $\mathbf{h}(t) =
(h_1, \dots, h_m)$. Su promedio y el umbral del evento son

$$\bar{h}(t) \;=\; \frac{1}{m}\sum_{j=1}^{m} h_j(t),
\qquad \tau(t) \;=\; \rho\,\bar{h}(t), \qquad \rho = \tfrac{1}{2}, \tag{3}$$

donde $m$ es la cantidad de posiciones con headway resuelto, $h_j(t)$ es el
headway de la posición $j$ y $\rho$ es la fracción que fija el umbral
$\tau(t)$; las posiciones «sin valor» del Apéndice A no entran en el cómputo.
La posición $i$ cuenta como bunching cuando cae por debajo de ese umbral:

$$b_i(t) \;=\; \mathbb{1}\!\left[\, h_i(t) < \tau(t) \,\right],
\qquad \text{definido solo si } m \ge 3, \tag{4}$$

donde $\mathbb{1}[\cdot]$ es la función indicadora. Cada posición con
$b_i(t) = 1$ es un evento, y se dice que la regla la **marca**. La condición
$m \ge 3$ existe porque con dos headways cualquier medida de irregularidad se
reduce a la diferencia entre ellos y no describe un patrón.

El detector que este trabajo evalúa es esa misma regla aplicada al vector
predicho de la Ecuación (1), con el promedio de ese mismo vector fijando el
umbral:

$$\hat{b}_i(t) \;=\; \mathbb{1}\!\left[\, \hat{h}_i(t) < \rho\,\bar{\hat{h}}(t)
\,\right], \tag{5}$$

donde $\hat{b}_i(t)$ es la detección emitida sobre la posición $i$ y
$\bar{\hat{h}}(t)$ es el promedio del vector predicho. Cada posición con
$\hat{b}_i(t) = 1$ es un **trigger**: la señal que el detector emite, y lo
único que un operador vería. El umbral sale del vector predicho porque quien
opera un corredor no dispone del observado al momento de decidir.

Como $\tau$ es función del propio vector que se evalúa, y no un número fijo de
minutos, las Ecuaciones (4) y (5) no comparan contra el mismo umbral:

$$\tau(\hat{\mathbf{h}}) \;=\; \rho\,\bar{\hat{h}}
\;\neq\; \rho\,\bar{h} \;=\; \tau(\mathbf{h})
\qquad \text{siempre que } \bar{\hat{h}} \neq \bar{h}, \tag{6}$$

donde $\tau(\mathbf{h})$ y $\tau(\hat{\mathbf{h}})$ resultan de aplicar $\rho$
al vector observado y al predicho. Un denominador observado, como el de la
Sección II-A, no tiene esa propiedad: no se mueve con la predicción. Eso no lo
pone a salvo, y la Sección V-F mide cuánto de la diferencia le corresponde. La
Figura 1 lo muestra con el mismo headway de dos minutos, y la Sección V-C mide
qué ocurre sobre datos reales cuando los dos umbrales de la Ecuación (6) se
tratan como uno.

![El mismo headway bajo dos umbrales](figuras/bunching-umbral.es.png)

**Fig. 1.** El mismo headway de 2.0 min bajo el umbral relativo. Cada barra es
una posición del vector y la línea discontinua es el umbral τ = promedio/2 de ese
panel. (a) En el corredor irregular el promedio es 5.9 min y el umbral 3.0 min:
los headways de 2.0 y 1.2 quedan debajo y **los dos son bunching**. (b) En el
corredor regular el promedio es 3.1 min y el umbral 1.6 min: el mismo headway de
2.0 min queda encima y **no es bunching**. Valores ilustrativos, no datos reales.

### C. Métricas

El modelo entrega un vector de headways que la regla de la Sección III-B
convierte en un indicador binario de bunching, y la evaluación mide esos dos
objetos en cadena. El error del vector es el error absoluto medio (MAE) sobre
las posiciones válidas de la Ecuación (2), y se reporta en lugar del error
cuadrático porque expresa el resultado en minutos de headway.

El MAE no describe la forma del vector. El coeficiente de variación (CV) es su
desviación estándar muestral dividida por su promedio:

$$\mathrm{CV}(\mathbf{h}) \;=\; \frac{1}{\bar{h}}
\sqrt{\frac{1}{m-1}\sum_{j=1}^{m}\big(h_j - \bar{h}\big)^{2}}, \tag{7}$$

donde $m$, $h_j$ y $\bar{h}$ conservan el significado de la Ecuación (3). Se
calcula sobre los vectores de tres posiciones o más que exige la Ecuación (4).
Se reporta porque es adimensional, de modo que corredores de frecuencias
distintas quedan sobre la misma escala, y porque es la cantidad con que se lee
la escala de nivel de servicio del TCQSM [@tcqsm2003]: el manual indexa sus
bandas por la dispersión del headway respecto del programado, y en un corredor
sin programación esa dispersión es la Ecuación (7). Su sesgo es el CV de lo
predicho menos el de lo observado, y un valor negativo dice que lo predicho es
más regular que la realidad.

El CV compara la forma de lo predicho con la de lo observado, pero no dice
cuánto de esa diferencia se debe al error. La varianza entre las posiciones de
un mismo vector la reparte:

$$V_h \;=\; V_{\hat h} + V_e + 2\,C_{\hat h e}, \qquad
r \;=\; \frac{V_{\hat h}}{V_h}, \qquad r_0 \;=\; 1 - \frac{V_e}{V_h}, \tag{8}$$

donde $V_h$, $V_{\hat h}$ y $V_e$ son las varianzas entre posiciones del
headway observado, del predicho y del error $e_i = h_i - \hat{h}_i$,
promediadas sobre los vectores de la celda, y $C_{\hat h e}$ es la covarianza
entre lo predicho y el error. La fracción $r$ es la parte de la dispersión
observada que sobrevive en lo predicho. La fracción $r_0$ es la que el error
deja explicada, y coincide con $r$ cuando el error no covaría con lo predicho,
como ocurre con la media condicional de la Sección II-B.

El indicador derivado se puntúa con tres cantidades, ordenadas por cuánto
dependen del umbral, sobre la matriz de confusión entre el indicador observado
de la Ecuación (4) y el detector de la Ecuación (5). Sean TP las posiciones con
$b_i = \hat{b}_i = 1$, FP las que tienen $\hat{b}_i = 1$ y $b_i = 0$, FN las que
tienen $b_i = 1$ y $\hat{b}_i = 0$, y TN las restantes. La precisión, el recall
y el F1 son entonces

$$\mathrm{F}_1 \;=\; \frac{2PR}{P+R}, \qquad
P \;=\; \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \qquad
R \;=\; \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \tag{9}$$

El F1 no usa TN [@chicco2020], y premia por eso al **detector trivial**: el que
emite un trigger en toda posición. Maximizar el F1 sobre una predicción sin
información conduce a ese detector con independencia de la tasa base
[@lipton2014]. La tasa base de
una celda es la fracción de sus posiciones donde el indicador observado vale 1.
Ese detector alcanza recall 1 y precisión igual a la tasa base [@flach2015], así
que su F1 queda fijado por ella y acompaña como piso a todo F1 reportado. El
coeficiente de correlación de Matthews (MCC) usa los cuatro conteos. Para ese
detector su cociente queda indeterminado, porque numerador y denominador se
anulan a la vez, y se le asigna cero por extensión por continuidad
[@chicco2020]. El área bajo la curva ROC (AUC) [@handtill2001] prescinde del
umbral —el punto de operación del detector— y puntúa el ordenamiento del puntaje
continuo $-\hat{h}_i/\bar{\hat{h}}$, del cual la Ecuación (5) es el umbral en
$-\rho$. Vale 0.5 cuando la predicción no ordena.

Ese 0.5 es el piso de una predicción sin ninguna información, y no el de una
predicción sin información **temporal**. El perfil posicional del Apéndice A,
sección B fija el segundo: es lo que alcanza el AUC cuando solo se conoce qué
posición del vector suele llevar el headway más corto. Cumple para el AUC la
misma función que el detector trivial cumple para el F1, y por eso acompaña a
todo AUC reportado.

Sobre esas cantidades se construyen dos cocientes. La tasa de trigger de un
método es la fracción de sus posiciones con $\hat{b}_i = 1$, y el factor entre
dos métodos es el cociente de sus F1 bajo el mismo umbral.

---

## IV. Reparaciones del punto de operación y de la regla

### A. Recalibración fuera de muestra

La primera reparación deja intacta la regla del evento y mueve el umbral del
detector de la Ecuación (5), su punto de operación. Toda la evaluación se
repite sobre tres **orígenes de evaluación** (*rolling origin*): tres fechas
de fin de entrenamiento, cada una con su propio período de prueba, que el
Apéndice A, sección C detalla. Ese umbral no se hereda de lo observado: se
ajusta maximizando el MCC sobre el período de prueba del origen 2 y se aplica
sin cambios al del origen 3. Los dos períodos son disjuntos y provienen de
modelos entrenados por separado, de modo que el período publicado no informa
su propio umbral.

### B. La regla de denominador observado y la regla de cuota

La regla de la Sección III-B divide por el promedio del vector predicho, que se
mueve con la predicción. Se la contrastó con otras dos sobre la misma población
y el mismo origen. La primera divide por el promedio del último vector
observado: se recalcula en cada instante y sigue al corredor, pero es el mismo
número para lo observado y para lo predicho. Se la llama aquí **la regla de
denominador observado**. La segunda no divide por nada. Marca las posiciones más
cortas de cada vector, y cuántas marca queda fijado por la longitud del vector
antes de leer los valores. Se la llama aquí **la regla de cuota**. Esa cantidad
es un entero y los vectores llevan entre tres y seis posiciones, de modo que el
redondeo levanta la frecuencia del evento hasta 8.9 puntos en E4, donde los
vectores son más cortos. La regla de cuota no marca entonces exactamente el
mismo evento que las otras dos.

---

## V. Resultados

### A. El error escalar del vector

Con la arquitectura del Apéndice A, sección B ya fijada, el error escalar del
vector sitúa a ese modelo contra la persistencia. A diez minutos de
anticipación, el LSTM predijo el headway entre buses mejor que la persistencia.
El error absoluto medio bajó 1.47 minutos en E2, 1.38 en E4 y 1.17 en E59: entre
21 % y 22 % en los tres corredores. A un minuto la relación se invirtió y la
persistencia ganó, por 0.46 minutos en E4 y 0.33 en E59. En E2 la diferencia fue
de 0.07 minutos y no resistió la prueba estadística al agrupar las observaciones
por día de servicio.

### B. Compresión de la dispersión transversal

Ese error escalar no dice nada sobre la forma del vector. El coeficiente de
variación de la Ecuación (7) sí. Medido sobre lo observado, fue de 0.79 en E2.
Medido sobre lo que el modelo predijo para el mismo instante y el mismo corredor
a diez minutos, fue de 0.16. El vector predicho describió un corredor casi cinco
veces más regular que el real.

Esa brecha no fue un caso aislado. El sesgo del coeficiente de variación resultó
negativo —lo predicho siempre más regular que la realidad— en **las doce celdas
y los tres orígenes de evaluación**. Y se profundizó de forma estrictamente
ordenada a medida que se alarga el horizonte: en E2 pasó de −0.42 a un minuto a
−0.63 a diez. No hubo una sola excepción en los tres corredores.

La persistencia acota la causa por descarte: no comprimió nada, con el sesgo
dentro de ±0.022 en las doce celdas y los tres orígenes. El control sitúa el
efecto en el **ajuste por error cuadrático**, que la persistencia no hace, y no
en los datos ni en el corredor.

La Ecuación (8) ata ese efecto a una sola cantidad. Sobre las doce celdas, $r$
siguió a $r_0$ con una correlación de 0.993 en el LSTM, con $r$ entre 0.045 y
0.552. El XGBoost quedó sobre la misma relación y sobre los mismos vectores,
con una correlación de 0.996 y $r$ entre 0.040 y 0.547. Las dos arquitecturas
no comparten sesgo inductivo y sí el objetivo de ajuste. La fracción $r_0$ del
LSTM cae de 49.5 % en E4 a un minuto hasta 1.3 % en E2 a diez.

Leídas contra la escala de nivel de servicio del TCQSM, según la convención de
la Sección III-C, esas cifras dicen que el mismo corredor en el mismo instante
calificó como nivel A —«service provided like clockwork»— según lo predicho y
como nivel F —«most vehicles bunched»— según lo observado. La Figura 2 muestra
el efecto.

![Dispersión observada frente a predicha](figuras/compresion-dispersion.es.png)

**Fig. 2.** Dispersión observada frente a dispersión predicha por método,
horizonte de diez minutos.

### C. Colapso de la detección al trasladar el umbral

La regla de la Sección III-B, aplicada a lo observado, marcó 15 245 eventos en
E2 a diez minutos. Aplicada a lo predicho por el LSTM, con el mismo umbral,
emitió **catorce triggers**. La persistencia emitió 15 083. Puntuada con el F1
de la Sección III-C, la persistencia apareció 253 veces mejor que el LSTM. En
las otras celdas el factor va de 1.5 a 36. El XGBoost obtuvo un F1 exactamente
cero en tres de las doce celdas: ahí no emitió ninguno. La Tabla 1 recoge las
doce celdas.

Leído sin más contexto, ese resultado dice que el LSTM es incapaz de ver el
fenómeno que se le pidió anticipar. Esa lectura no sobrevive a cuatro
observaciones. El ganador declarado tampoco detectó bien: el detector trivial de
la Sección III-C superó a la persistencia en 5 de las doce celdas, y en 15 de
las 36 combinaciones de celda y origen. El mecanismo es la compresión de la
Sección V-B: el umbral calibrado sobre lo observado deja de alcanzarse en el
vector comprimido. Y el modelo acertó en las pocas ocasiones en que emitió: de
los catorce triggers de E2, diez correspondieron a eventos reales, 71 % de
precisión contra una tasa base de 30 %, con el intervalo del Apéndice B entre 42
% y 92 %. Las celdas con más triggers estrechan ese intervalo y mantienen la
precisión por encima de su tasa base.

La cuarta observación es que el factor no se sostiene al cambiar el origen. Si
midiera una capacidad del modelo debería ser aproximadamente estable entre
ellos, y en diez de las doce celdas lo es: entre el primer origen y el tercero
varía entre 0.90 y 1.58. Las dos excepciones están en E2: a cinco minutos el
factor valió **126**, **58** y **36** en los tres orígenes, y a diez minutos **2
299**, **817** y **253**. Esas dos son las celdas donde el umbral trasplantado
dejó al detector casi sin triggers, con el F1 en 0.011 y 0.001 en la Tabla 1. Un
cociente cuyo denominador se acerca a cero no mide una capacidad del sistema
evaluado, sino la interacción entre el umbral y la distribución sobre la que
cayó.

![Tasa de trigger contra tasa real del evento](figuras/artefacto-umbral.es.png)

**Fig. 3.** Fracción de posiciones con trigger de cada método, contra la tasa
real del evento (punteada).

**Tabla 1.** Detección con el umbral del evento observado aplicado sin cambios a
lo predicho, con el piso del detector trivial al lado.

| Corredor | h | Tasa base | Piso trivial | F1 persistencia | F1 LSTM | Factor |
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

† El detector trivial supera al ganador declarado en estas celdas.

### D. Detección sin umbral

Si el problema es el umbral, recalibrarlo debería bastar. Se aplicó entonces la
recalibración de la Sección IV-A sin tocar el modelo, con el MCC como objetivo
porque el F1 degenera en este corpus: sobre la persistencia en E2, de tres
minutos en adelante, el umbral que optimiza el F1 emitió un trigger entre el
99.9 % y el 100 % de las posiciones, esto es, el detector trivial de la Tabla 1.

El umbral trasplantado de la Tabla 1 dejaba a la persistencia por delante en las
doce celdas. Los dos instrumentos que la Tabla 2 reúne mueven ese conteo en
distinta medida. Puntuado sin umbral, mediante el AUC, **el LSTM ganó en las
nueve combinaciones de corredor y origen a diez minutos**, y en 6 de las 12
celdas del origen 3. Las nueve diferencias de diez minutos sobrevivieron su
intervalo, y van de 0.033 a 0.061. Recalibrar el umbral en lugar de eliminarlo
lo mueve menos: el LSTM ganó en 5 de las 12 celdas, entre ellas las tres de diez
minutos, si bien la de E4 no resiste su propio intervalo. La persistencia
conservó la ventaja a un minuto, donde el error escalar también la favorecía, en
los tres corredores y los tres orígenes. La inversión del AUC llegó en el mismo
escalón que la del error escalar en E2, y uno o dos escalones más tarde en E59 y
E4. En ninguna de las dos métricas volvió la ventaja a la persistencia
al alargar el horizonte.

Entre la Figura 3 y la Figura 4 cambió solo el umbral de la Ecuación (5): la
primera lo hereda de lo observado, la segunda lo elimina y la columna del MCC
recalibrado de la Tabla 2 lo reajusta contra lo predicho.

![Ventaja escalar y AUC de detección](figuras/deteccion-sin-umbral.es.png)

**Fig. 4.** Ventaja de error absoluto del LSTM sobre la persistencia (escala
izquierda) y AUC de detección de cada método (escala derecha), un panel por
corredor.

**Tabla 2.** Veredicto sin umbral y con el umbral recalibrado fuera de muestra,
con el piso del perfil posicional al lado del AUC que acota.

| Corredor | h | AUC LSTM | AUC persist. | Piso posicional | Δ AUC [IC 95 %] | MCC recal. LSTM | MCC recal. persist. | Δ MCC [IC 95 %] |
| :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: |
| E2 | 1 | 0.714 | **0.723** | 0.587 | -0.009 [-0.015, -0.004] | 0.310 | **0.401** | -0.091 [-0.106, -0.078] |
| E2 | 3 | **0.629** | 0.598 | 0.580 | +0.031 [+0.025, +0.036] | **0.178** | 0.160 | +0.018 [+0.005, +0.028] |
| E2 | 5 | **0.604** | 0.567 | 0.582&nbsp;§ | +0.037 [+0.031, +0.042] | **0.139** | 0.102 | +0.037 [+0.026, +0.046] |
| E2 | 10 | **0.565** | 0.528 | 0.579&nbsp;§ | +0.037 [+0.026, +0.047] | **0.085** | 0.027 | +0.058 [+0.039, +0.073] |
| E4 | 1 | 0.811 | **0.833** | 0.523 | -0.022 [-0.027, -0.016] | 0.476 | **0.615** | -0.140 [-0.150, -0.130] |
| E4 | 3 | 0.702 | **0.719** | 0.520 | -0.017 [-0.025, -0.009] | 0.269 | **0.375** | -0.106 [-0.122, -0.086] |
| E4 | 5 | 0.648 | 0.649 | 0.519 | -0.001 [-0.010, +0.008] | 0.190 | **0.254** | -0.064 [-0.083, -0.043] |
| E4 | 10 | **0.604** | 0.558 | 0.521 | +0.047 [+0.030, +0.063] | 0.126 | 0.111 | +0.015 [-0.005, +0.033] |
| E59 | 1 | 0.760 | **0.781** | 0.498 | -0.021 [-0.025, -0.016] | 0.363 | **0.517** | -0.154 [-0.164, -0.144] |
| E59 | 3 | 0.688 | 0.689 | 0.493 | 0.000 [-0.005, +0.005] | 0.237 | **0.328** | -0.091 [-0.100, -0.081] |
| E59 | 5 | **0.665** | 0.648 | 0.492 | +0.017 [+0.012, +0.022] | 0.205 | **0.249** | -0.044 [-0.053, -0.036] |
| E59 | 10 | **0.632** | 0.571 | 0.486 | +0.061 [+0.054, +0.067] | **0.161** | 0.119 | +0.042 [+0.033, +0.052] |

La negrita marca al ganador de cada par, y se omite donde el intervalo de esa
diferencia contiene al cero: ahí los dos métodos son indistinguibles. Ocurre en
tres celdas, y en E4 a diez minutos afecta solo a la correlación recalibrada,
donde la ventaja del LSTM no resiste su intervalo aunque sí resista la del AUC.

§ El piso posicional supera a la persistencia en estas celdas, y al LSTM en la
de diez minutos. La negrita compara los dos métodos entre sí y no contra el
piso.

### E. El piso posicional acota el ordenamiento

Ese AUC no basta por sí solo para atribuirle el ordenamiento a la anticipación,
y el perfil posicional del Apéndice A, sección B lo acota. En E4 y E59 el piso
queda indistinguible del azar, entre 0.486 y 0.523 en las ocho celdas, de modo
que el ordenamiento del LSTM ahí no proviene de la posición: lo supera por entre
0.08 y 0.29, y las ocho diferencias sobreviven su intervalo. En E2 el piso sube
a 0.58 y no se mueve con el horizonte, señal de una estructura posicional
estable.

**A diez minutos en E2 el LSTM queda por debajo de ese piso, 0.565 contra
0.579.** La diferencia vale -0.013 [-0.023, -0.003] y sobrevive su intervalo, de
modo que ahí la ventaja sin umbral no se sostiene contra un método que no lee la
ventana de entrada. Es la única de las doce celdas donde ocurre, y es la que la
Sección V-C usa para exhibir el artefacto del umbral. El piso también supera a
la persistencia en E2 a cinco y a diez minutos, de modo que acota a los dos
métodos comparados.

### F. El umbral en minutos es lo que la compresión alcanza

Las dos reglas con denominador colapsaron; la de cuota no podía hacerlo. La
razón entre la tasa de trigger del LSTM y la tasa real del evento tuvo mediana
**0.079** bajo la regla de la Sección III-B y **0.153** bajo la regla de
denominador observado. Quitar la auto-referencia duplicó el disparo y lo dejó un
orden de magnitud por debajo de la frecuencia del evento. Bajo la regla de cuota
esa razón valió **1.000** en las doce celdas por construcción: la cuota marca la
misma cantidad en lo predicho y en lo observado. La persistencia no colapsó bajo
ninguna de las tres, con medianas de 1.011, 0.980 y 1.000.

El mecanismo se lee en el umbral que cada regla termina aplicando, medido en
minutos. En E2 a diez minutos, bajo la regla de la Sección III-B, ese umbral
valió 3.89 minutos sobre lo observado y 3.92 sobre lo predicho: se quedó donde
estaba. Bajo la regla de cuota el mismo par valió 2.76 y 6.82 minutos. La regla
sin denominador sube su propio umbral hasta donde quedó la distribución
comprimida, y la distancia que sube crece con el horizonte en los tres
corredores. Un umbral en minutos no puede seguirla, porque su valor no depende
de la escala de lo que evalúa. La compresión de la Sección V-B alcanza
entonces a toda regla que nombre una cantidad de minutos, y no queda contenida
en la que divide por lo predicho.

### G. Qué recupera la regla de cuota

La consecuencia está en el veredicto. Las mismas residuales dan un veredicto sin
umbral en la Sección V-D, y las tres reglas se contrastan contra él. La regla
de cuota lo reproduce en **once** de las doce celdas; la de denominador
observado en siete y la de la Sección III-B en seis. Bajo esta última la
persistencia ganó las doce, que es lo que hizo leer el colapso como ceguera del
modelo. La única celda donde la regla de cuota discrepa es E59 a cinco minutos.
La Tabla 3 recoge las tres reglas con sus medianas y ese conteo de
coincidencias.

La regla de cuota no convierte al modelo en mejor detector. Su MCC tuvo mediana
**0.199** contra **0.100** bajo la regla de la Sección III-B, y la superó en las
doce celdas. Esa mediana iguala a la del umbral recalibrado de la Sección V-D,
que vale 0.198, y la regla de cuota no ajusta ningún parámetro sobre una ventana
anterior. Aun así **sigue por debajo de la persistencia** en siete de las doce.
Las cinco que gana son las tres de E2 desde los tres minutos, y las de diez
minutos en E4 y E59. Reparar la regla recupera discriminación y no cambia de
dueño el veredicto a un minuto en ninguno de los tres corredores.

Las tres reglas tampoco marcan el mismo evento sobre lo observado. El índice de
Jaccard, las posiciones que dos reglas marcan a la vez entre las que marca al
menos una, tuvo mediana 1.000, 0.710 y 0.580 contra la regla de la Sección
III-B. El solape no explica entonces la diferencia: la regla de denominador
observado es la que más se le parece, y es la que colapsa con ella.

**Tabla 3.** Las tres reglas del evento sobre la misma población y el mismo
origen. Cada celda es la mediana de las doce combinaciones de corredor y
horizonte.

| Regla | Denominador | Trigger/evento, LSTM | Trigger/evento, persistencia | MCC del LSTM | Solape con la regla de la Sección III-B | Coincide con el veredicto sin umbral |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| Sección III-B | promedio del vector predicho | 0.079 | 1.011 | 0.100 | 1.000 | 6 de 12 |
| Denominador observado | promedio del último vector observado | 0.153 | 0.980 | 0.143 | 0.710 | 7 de 12 |
| Cuota | ninguno | 1.000&nbsp;‡ | 1.000&nbsp;‡ | 0.199 | 0.580 | **11 de 12** |

‡ Vale uno por construcción y no por medición: la cantidad de posiciones
marcadas queda fijada antes de leer los valores.

---

## VI. Discusión y amenazas a la validez

### A. Robustez frente al origen y a la definición del evento

El veredicto sin umbral de la Sección V-D no depende del origen calendario.
Los tres orígenes coincidieron en 11 de las 12 celdas —la única discrepancia
es E4 a cinco minutos, donde solo el origen 2 favoreció al LSTM— y a diez
minutos coincidieron en las nueve combinaciones de corredor y origen. El
primero de los tres cubre del 23 de diciembre al 13 de enero. Ese acuerdo
incluye entonces el
período de fiestas, cuando la frecuencia del servicio y la demanda no se parecen
a las de un mes ordinario.

Tampoco depende de la definición del evento. El umbral relativo de la Sección
III-B podría estar produciendo el efecto por sí solo, y un umbral absoluto en
minutos —como el de un minuto de Sun, Schmöcker y Nakamura [@sun2021]— podría
disolverlo. Se probó con uno fijo en la cuarta parte del headway mediano
observado de cada corredor y sentido. Queda entre 1.4 y 2.4 minutos, se calibró
sobre el origen 2 y se aplicó sin cambios al origen 3. **No se atenuó:
empeoró.** La tasa de trigger del modelo cayó por un factor de mediana 138 en
diez de las doce celdas, y en las otras dos no emitió ninguno.

El mismo ensayo acota una afirmación anterior. Bajo el umbral absoluto la
capacidad de discriminación del modelo cayó: la mediana del AUC bajó a 0.60, y
en E2 a diez minutos llegó a 0.493, indistinguible del azar. Esa celda ya había
fallado bajo el evento relativo, contra el perfil posicional de la Sección
V-E. Las dos definiciones del evento coinciden entonces en ella. La afirmación
de que el LSTM no es ciego se sostiene fuera de esa celda y no dentro.

### B. Amenazas a la validez

El umbral del evento es la fracción del promedio que usa la convención del
campo, y no proviene de un registro de eventos observados. Queda sin verificar
que la fracción marque lo que un operador llamaría bunching, y validarla
exigiría un registro de incidentes que estos corredores no producen. El
alcance de toda afirmación de detección es entonces el evento así definido, y
sobre las posiciones que resolvieron. El tope de los treinta minutos del
Apéndice A recorta los headways más largos, de modo que el promedio del vector
—y con él el umbral— queda por debajo del que daría el registro completo. La
tasa base que se reporta no admite entonces comparación directa con tasas de
bunching medidas sobre registros sin enmascarar.

La compresión de la Sección V-B admite una lectura que apunta al ruido de
medición y no a la predicción. El eje del corredor se estima de los registros y
el sentido de marcha se infiere del signo del arco. Un error de medición entra
entonces en el error de predicción y agranda la compresión, sin decir nada sobre
la predicción misma. La descomposición de la Ecuación (8) acota esa lectura sin
eliminarla: $r$ sigue a $r_0$ con una correlación de 0.993. Del corpus depende el tamaño del efecto, no su existencia: un corredor
con geometría publicada y sentido declarado tendría un error menor y una
compresión menor, en la proporción que esa descomposición fija.

El corpus acota dos cosas más. Un vector reúne entre 3.8 y 5.9 headways en
promedio, de modo que la dispersión transversal reposa sobre pocas
observaciones. Que el efecto se repita en los tres corredores y en los tres
orígenes lo hace poco atribuible a esa longitud. Cada cifra individual es menos
estable en E2 y en E4, que tienen el vector más corto, que en E59. El período de
prueba contiene además los días de Carnaval, cuya composición no se caracterizó,
de modo que la comparación incluye días atípicos sin identificarlos.

Los dos métodos que ajustan parámetros no reciben el mismo presupuesto de
búsqueda: el XGBoost elige veinticuatro configuraciones por celda sobre las
muestras definitivas, mientras que el LSTM hereda la suya en dos de los tres
corredores. Eso acota una comparación y solo
una: donde el LSTM queda por detrás del XGBoost, la diferencia no es atribuible
a la clase de modelo. Los otros dos métodos no ajustan nada, de modo que el
error de referencia que fijan no depende de esa asimetría. El contraste de
arquitecturas del Apéndice A, sección B tampoco está nivelado con el resto,
porque precede al protocolo del Apéndice A, sección C y no se rehízo después.

El piso posicional que acota el AUC tiene dos límites propios. Se ajusta sobre
un solo origen anterior, mientras que los veredictos entre métodos se replican
sobre tres. Y no carece de información: el piso lee qué posiciones ocupó el
corredor en ese minuto, aunque no la ventana de entrada; los dos métodos que
acota leen esa misma composición, de modo que la comparación no le concede nada
a ninguno.

Las métricas de la Sección III-C son genéricas y comparables entre corredores, y
ninguna liga un error de predicción a una decisión de intervención. Los dos
puntos de operación de la Sección V ofrecen además avisos distintos: el
trasplantado emitió un trigger en el 0.03 % de las posiciones de E2 a diez
minutos, con la precisión por encima de la tasa base según la Sección V-C —un
filtro de prioridad—, y el recalibrado en el 26.98 %. Elegir entre esos dos
regímenes de aviso exige una función de costo que pondere el aviso perdido
contra el aviso falso, y esa función depende de la operación de cada empresa.

---

## VII. Conclusión

Este trabajo predice el vector de headways de tres corredores de Arequipa con un
LSTM, y convierte lo predicho en un indicador de bunching mediante una regla
relativa al promedio del propio vector. La dispersión transversal de lo predicho
queda por debajo de la observada en las doce celdas y los tres orígenes de
evaluación, y la brecha se profundiza al alargar el horizonte. Con el umbral del
evento observado trasladado sin cambios, el detector emite catorce triggers
sobre los 15 245 eventos que la regla marca en E2 a diez minutos. La
persistencia lo supera ahí por un factor de 253 en el F1.

Ese colapso no mide la capacidad del modelo sino el punto de operación en el que
se lo evalúa. Puntuada sin fijar un umbral, mediante el AUC, la predicción del
LSTM ordena mejor que la persistencia en las nueve combinaciones de corredor y
origen a diez minutos. Esa ventaja se sostiene contra un perfil que solo conoce
la posición en E4 y en E59. Recalibrar el umbral sobre un período anterior
disjunto recupera parte de esa ventaja sin reentrenar. El punto de operación se
calcula entonces contra la distribución de lo predicho, y no se hereda de las
observaciones. Definir el evento por una cuota de posiciones, y no por una
cantidad de minutos, alcanza esa misma discriminación sin ajustar ningún
parámetro, y reproduce el veredicto sin umbral en once de las doce celdas.

El mismo perfil acota hasta dónde llega la afirmación. En E2 a diez minutos
ordena mejor que el modelo, y esa es la celda de la que sale el factor de 253:
allí queda medido el colapso que el umbral trasplantado produce, y no la
capacidad de anticipación que quedaría al retirarlo.

Tres extensiones quedan abiertas. La primera liga la detección a una función de
costo que pondere el aviso perdido contra el aviso falso, que la Sección VI-B
declara ausente. La segunda emite una predicción probabilística en lugar de
puntual, de modo que el ajuste no reduzca lo predicho a la media condicional. La
tercera valida la regla del evento contra un registro de incidentes, que estos
corredores todavía no producen.

---

## VIII. Declaraciones

Los datos primarios son registros GPS del Sistema Integrado de Transporte de
Arequipa, cuya fuente es la Municipalidad Provincial de Arequipa. El conjunto
crudo y el procesado están disponibles en Kaggle, en
`kaggle.com/datasets/alexhuaracha/multibus-headway-forecast-raw` y
`kaggle.com/datasets/alexhuaracha/multibus-headway-forecast-clean`. El código de
preprocesamiento, entrenamiento y análisis, junto con los guiones que generan
cada tabla y cada figura de este documento, está disponible en
`github.com/Alex-Huaracha/multibus-headway-forecast`.

Se usaron herramientas asistidas por inteligencia artificial generativa para la
redacción del texto y para la verificación de las citas contra sus fuentes. El
diseño experimental, la implementación, las cifras reportadas y su interpretación
fueron revisados y verificados por los autores, que asumen la responsabilidad del
contenido final.

---

## Apéndice A. Datos, métodos comparados y protocolo

### A. Datos y construcción del headway

El trabajo usa los registros GPS de empresas del Sistema Integrado de
Transporte de Arequipa. Cada bus emite su coordenada **cada 20 segundos**.
Se cubren tres corredores —identificados aquí como E2, E4 y E59, uno por empresa
operadora— durante 152 días seguidos, del 1 de octubre de 2023 al 29 de febrero de
2024, sin huecos de calendario. Son 90 buses en total. Una empresa entra como
corredor bajo dos condiciones. La primera es que el desplazamiento de sus buses
esté dominado por una sola dirección: la varianza de las coordenadas a lo largo de
esa dirección supera cuatro veces la lateral. La segunda es que circulen al menos
cinco buses a la vez, sin lo cual un vector de headways no describe nada.

El headway de la Sección I se construye aquí desde la coordenada de los buses.
La forma habitual lo mide en una parada, con la lista de
paradas y los horarios de paso; estos registros no traen ninguna de las dos:
cada bus emite su identificador, el instante y su coordenada, y ese **registro
GPS** es la única entrada. Andres y Nair resuelven la misma construcción —de
registros GPS a headways— con una secuencia de pasos, cada uno con su umbral
[@andres2017]. Esta sección sigue esa forma con una diferencia: ellos
proyectan contra la geometría GTFS que su ciudad publica, y aquí el eje se
ajusta de los propios registros. La secuencia tiene seis pasos.

1. **El eje.** No existe una geometría publicada de la ruta, de modo que el eje
   —la línea que los buses siguen a lo largo del corredor— se ajusta de los
   propios registros [@quek2020] [@biagioni2012]: componentes principales para
   la orientación, mediana de la coordenada transversal en 50 tramos y promedio
   móvil de cinco tramos como suavizado. Solo entran registros a más de
   10 km/h.

2. **La proyección al eje.** Cada coordenada se lleva a metros con una
   aproximación plana local y se proyecta al punto más cercano del eje, la
   referenciación lineal de la norma ISO 19148 [@iso19148]. Quedan la
   **coordenada de arco** $s$ —los metros recorridos sobre el eje— y el
   **desvío lateral**, la distancia al eje; el registro se descarta si el
   desvío pasa de 300 m.

3. **El sentido de marcha.** El sentido es el signo del cambio de la
   coordenada de arco, promediado sobre los cinco últimos registros; con
   promedio nulo, el bus queda fuera de los pares de ese minuto.

4. **El eje por sentido.** En dos de los tres corredores la ida y la vuelta
   circulan por calles paralelas, la dificultad que Andres y Nair señalan para
   asignar el antecesor usando solo GPS [@andres2017]. Los pasos 1 y 2 se
   repiten una vez por sentido, ya con el sentido asignado.

5. **La rejilla común.** La coordenada de arco se interpola entre los dos
   registros vecinos sobre una **rejilla** de sesenta segundos —tres registros
   por bus y minuto—, y cada minuto queda descrito por un **snapshot**: la
   coordenada de todos los buses del corredor en ese minuto.

6. **El headway.** Sobre ese snapshot, para un par de buses consecutivos en el
   mismo sentido —el de adelante $L$, el de atrás $F$— en el instante $T$:

$$t_{c} = \max\{\, t \le T \;:\; s_{L}(t) = s_{F}(T) \,\},
\qquad h = T - t_{c}, \tag{10}$$

donde $s_{L}$ y $s_{F}$ son las coordenadas de arco del bus de adelante y del de
atrás. El instante $t_{c}$ es el último en que el de adelante pasó por la
coordenada que el de atrás ocupa en $T$, y $h$ es el headway resultante. La
definición es la de Pilachowski [@pilachowski2009], que Andres y Nair evalúan
en la coordenada del bus de atrás [@andres2017]. El cruce se resuelve sobre los
registros originales del bus de adelante; la rejilla solo fija el instante $T$
y el orden de los buses. Si no existe tal $t_{c}$, o si $h$ supera los treinta
minutos, se emite «sin valor».

Ese headway describe un solo par. En cada snapshot, los buses de un mismo
sentido se ordenan por su coordenada de arco, y con $N$ buses quedan $N-1$
pares: el vector de headways ordenado desde el frente. Ese orden numera las
posiciones del vector —la primera es la del par que va más adelante—, y un par
sin headway válido conserva su posición con «sin valor», para que el orden no
dependa de cuántos pares resolvieron.

El signo del paso 3 es la única fuente del sentido de marcha. El tope de
treinta minutos del paso 6 existe porque, sin él, dos calles paralelas
proyectadas sobre un mismo eje producen cruces de horas antes. La
cobertura —la fracción de pares evaluados con headway
válido— es del 63.5 % en E2, del 64.8 % en E4 y del 77.1 % en E59:
3 938 174 pares con headway válido sobre 5 601 738 evaluados, y una posición sin
headway válido se enmascara. Los huecos no se distribuyen al azar: casi todo el
faltante viene del tope de treinta minutos —el cruce existe, pero quedó atrás—,
que recorta los intervalos más largos, y la condición sin cruce explica menos de
un punto porcentual en cada corredor.

### B. Métodos comparados

Se comparan tres métodos sobre las mismas muestras, y cada uno cumple un papel
distinto. El método bajo estudio es una red recurrente (**LSTM**). Un conjunto
de árboles con refuerzo de gradiente (**XGBoost**) [@chen2016] actúa como
**control de arquitectura**: si reproduce el patrón del LSTM, ese patrón no
proviene del aprendizaje profundo sino del objetivo de la Ecuación (2). El
tercero no ajusta parámetros y fija el error de referencia: la
**persistencia** repite el último vector observado, así que su error crece con
el horizonte.

De los tres métodos retenidos, solo el LSTM y el XGBoost ajustan parámetros.
Ambos se ajustan por corredor y por horizonte, y cada combinación de corredor y
horizonte se denomina aquí **celda**: hay doce. Los dos sentidos comparten el
modelo de su corredor y entran juntos al entrenamiento. Lo que se separa por
sentido son los estadísticos de estandarización, de modo que lo predicho se
devuelve a minutos con los del sentido que le corresponde. El LSTM usa 32
unidades ocultas, una o dos capas según la celda, tasa de aprendizaje de
5 × 10⁻⁴, lotes de 128 y semilla fija en 42. El XGBoost usa hasta 400 rondas con
parada temprana tras 30 sin mejora, y la misma semilla. Los presupuestos de
búsqueda no son iguales: el XGBoost eligió veinticuatro configuraciones por
celda sobre las muestras definitivas, mientras que el LSTM heredó la suya de una
búsqueda previa que no se rehízo sobre esas muestras, en dos de los tres
corredores. La Sección VI-B acota qué afirmaciones no se sostienen con esa
diferencia.

La elección del LSTM se resolvió antes de fijar el protocolo del Apéndice A,
sección C, contra dos arquitecturas que modelan la relación entre posiciones
vecinas del vector: una convolución sobre las posiciones contiguas y una
atención entre todas. Las tres quedaron dentro de un rango de 0.017 a 0.074
minutos de MAE en las doce celdas, y ninguna quedó primera en las doce, de modo
que el trabajo continuó con la más simple. Ese contraste precede al protocolo y
no se rehízo, y la Sección VI-B declara esa limitación.

A los tres se agrega un cuarto método que no compite con ellos y cumple otra
función: fijar un piso. El **perfil posicional** responde con el headway
promedio que cada posición del vector registró en un período anterior, y lo
repite sin cambios en cada minuto del período de prueba. No lee la ventana de
entrada, de modo que no puede anticipar nada. Existe porque las posiciones del
vector no son intercambiables: las de más adelante llevan headways
sistemáticamente más cortos, así que algunas caen por debajo de la mitad del
promedio de su vector por la posición que ocupan y no por lo que ocurrió ese
minuto. Lo que ese perfil ordena es la parte del ordenamiento que la posición
explica por sí sola. Se ajusta sobre el período de prueba del origen 2 y se
aplica al del origen 3, los mismos dos períodos que la Sección IV-A usa para el
umbral y por la misma razón.

### C. Protocolo de evaluación

La partición es **por fecha y nunca al azar**, porque un operador solo dispone
del pasado. El protocolo completo se evalúa sobre tres orígenes, numerados aquí
1, 2 y 3: los tres comienzan el mismo día, alargan el entrenamiento a 61, 83 y
107 días, y sus períodos de prueba no se solapan entre sí. El primero entrena
hasta el 30 de noviembre de 2023 y prueba del 23 de diciembre al 13 de enero. El
segundo entrena hasta el 22 de diciembre y prueba del 14 de enero al 4 de
febrero. El tercero —el que se publica, con 107 días de entrenamiento, 23 de
validación y 22 de prueba— entrena hasta el 15 de enero, valida hasta el 7 de
febrero y prueba del 8 al 29 de febrero de 2024. Como los entrenamientos están
anidados, esto establece estabilidad frente a la elección del período de prueba
y no réplica independiente.

La comparación exige además tres condiciones, cada una sobre una fuente distinta
de fuga: el tiempo, la población evaluada y los valores extremos. La primera es
la continuidad estricta: una muestra es válida solo si sus minutos son
consecutivos. Sin ella la ventana de entrada puede atravesar un hueco de señal,
y el horizonte mediría un intervalo mayor que el declarado. La regla retiene
entre el 81.9 % y el 90.2 % de los snapshots del período de prueba.

La segunda es la población compartida: los métodos comparados se puntúan sobre
exactamente las mismas filas. El trabajo de entrenamiento recalcula la lista de
muestras, compara su resumen SHA-256 contra el registrado y aborta antes de usar
la GPU si no coincide. La verificación evita comparar métodos puntuados sobre
poblaciones distintas. La tercera es el tope al percentil 99 del headway de
entrenamiento, aplicado como techo a las tres particiones. Calcularlo por
partición dejaría entrar información del período de prueba. El techo afecta
entre el 0.78 % y el 1.11 % de los objetivos, y las posiciones sin headway
válido siguen enmascaradas.

---

## Apéndice B. Pruebas estadísticas

La Sección III-C define las métricas, y compararlas entre dos métodos exige
declarar qué cuenta como resultado de esa comparación. Un veredicto es la
comparación de dos métodos sobre las mismas muestras bajo una métrica declarada,
y consta de tres partes: cuál de los dos gana, por cuánto y si la diferencia
sobrevive su prueba. Exigir muestras idénticas es lo que lo distingue de la resta
de dos métricas agregadas, que pueden haberse calculado sobre poblaciones
distintas. Este trabajo emite veredictos sobre el MAE de la Sección III-C y
sobre las cantidades de detección de la Ecuación (9), el MCC y el AUC. Un
veredicto sin umbral es el que usa el AUC, que no depende del punto de
operación.

Una diferencia de MAE entre dos métodos puede ser ruido del período de prueba. Se
contrasta con la prueba de Diebold–Mariano [@diebold1995] sobre el diferencial de pérdida por
muestra, con la corrección de muestra pequeña de Harvey–Leybourne–Newbold [@harvey1997]. La
varianza se estima agrupando por día de servicio, porque las
muestras de un mismo día comparten clima, incidentes y demanda. El agrupamiento
lleva el tamaño efectivo de muestra de entre 75 747 y 240 907 filas, según la
celda, a los 22 días del período de prueba.

El AUC y el MCC no admiten ninguna de esas dos pruebas: la primera contrasta un
diferencial de pérdida por muestra y la segunda acota una proporción. La
diferencia de cualquiera de los dos entre dos métodos se acota remuestreando días
de servicio con reemplazo. Se recalculan ambas cantidades sobre cada remuestreo y
se toma el intervalo percentil al 95 %, con dos mil remuestreos y semilla fija.
El agrupamiento es el mismo del diferencial de pérdida y responde a la misma
razón.

La precisión de la Ecuación (9) admite su propia acotación, porque puede
descansar sobre muy pocos triggers. Se acota con el intervalo exacto de
Clopper–Pearson [@clopper1934] al 95 %, calculado sobre los conteos de TP y de
FP de cada celda. Los conteos que necesitan acotarse aquí son los pequeños, y en
ellos la aproximación normal deja parte de su intervalo fuera del rango válido de
una proporción. Una celda
sin ningún trigger no recibe intervalo: no hay precisión que acotar.

---

## Referencias

_(lista en construcción: solo las fuentes ya verificadas en
`fuentes-verificadas.md` y ya llamadas desde el texto. Las llamadas usan claves
con arroba y no números, de modo que insertar una fuente no obliga a renumerar ni
a corregir llamadas. La numeración por orden de primera aparición se resuelve al
convertir al formato IJACSA, sustituyendo cada clave por su número; el orden de
esta lista no es todavía el definitivo.)_

`[@andres2017]` M. Andres and R. Nair, "A predictive-control framework to address
bus bunching," *Transportation Research Part B: Methodological*, vol. 104,
pp. 123–148, 2017, doi: 10.1016/j.trb.2017.06.013.

`[@biagioni2012]` J. Biagioni and J. Eriksson, "Inferring Road Maps from Global
Positioning System Traces: Survey and Comparative Evaluation," *Transportation
Research Record*, vol. 2291, no. 1, pp. 61–71, 2012, doi: 10.3141/2291-08.

`[@bonavita2024]` M. Bonavita, "On some limitations of data-driven weather
forecasting models," arXiv:2309.08473, 2023. La afirmación de la Sección
II-A proviene de este preprint, que examina un solo modelo; la versión publicada
—"On Some Limitations of Current Machine Learning Weather Prediction Models,"
*Geophysical Research Letters*, vol. 51, no. 12, art. e2023GL107377, 2024,
doi: 10.1029/2023GL107377— lleva otro título y examina tres, de modo que no se le
atribuye texto.

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

`[@ecmwffug]` *Forecast User Guide*, European Centre for Medium-Range Weather
Forecasts, Reading, U.K., §5.3.1 "M-Climate, the Medium Range Model Climate" and
§8.1.9.2 "Extreme Forecast Index — EFI", accessed Sep. 16, 2026. [Online].
Available: https://confluence.ecmwf.int/display/FUG/

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

`[@quek2020]` W. L. Quek, N. N. Chung, V.-L. Saw, and L. Y. Chew, "Analysis and
simulation of intervention strategies against bus bunching by means of an
empirical agent-based model," arXiv:2004.13022, 2020.

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

`[@trompet2011]` M. Trompet, X. Liu, and D. J. Graham, "Development of Key
Performance Indicator to Compare Regularity of Service between Urban Bus
Operators," *Transportation Research Record: Journal of the Transportation
Research Board*, vol. 2216, no. 1, pp. 33–41, 2011, doi: 10.3141/2216-04.

`[@usama2025]` M. Usama and H. Koutsopoulos, "Real Time Headway Predictions in
Urban Rail Systems and Implications for Service Control: A Deep Learning
Approach," arXiv:2510.03121, 2025.

`[@yu2016]` H. Yu, D. Chen, Z. Wu, X. Ma, and Y. Wang, "Headway-based bus bunching
prediction using transit smart card data," *Transportation Research Part C:
Emerging Technologies*, vol. 72, pp. 45–59, 2016,
doi: 10.1016/j.trc.2016.09.007.
