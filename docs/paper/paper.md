# El umbral, no el modelo: por qué un pronóstico de intervalos parece ciego al apelotonamiento, y cómo se repara

## Resumen

_(pendiente — se escribe al final)_

---

## I. Introducción

_(pendiente)_

---

## II. Trabajos relacionados

_(pendiente)_

---

## III. Datos y método

### A. Los datos

El trabajo usa los registros de posición de la flota del Sistema Integrado de
Transporte de Arequipa: cada unidad emite su coordenada cada pocos segundos.
Se cubren tres corredores —identificados aquí como E2, E4 y E59— durante 152 días
seguidos, del 1 de octubre de 2023 al 29 de febrero de 2024, sin huecos de
calendario.

Importa tanto lo que el dato tiene como lo que no. **No hay horario publicado, no
hay archivo GTFS y no hay tabla de paradas.** Eso obliga a construir todo desde la
posición cruda, que es trabajo extra, pero también es lo que vuelve el método
aplicable: la mayoría de las ciudades donde el apelotonamiento es un problema
cotidiano son exactamente las que no tienen ese dato ordenado. Un método que exija
GTFS no sirve donde más falta hace.

Una advertencia de identidad, porque condiciona todo lo demás: el identificador de
unidad se repite entre empresas, así que un bus solo queda determinado por el par
empresa-unidad. Tratarlo de otro modo mezcla vehículos de corredores distintos.

### B. Del GPS al intervalo entre buses

Sin tabla de paradas no se puede medir el intervalo en una parada. Lo que sigue
reconstruye la geometría del corredor desde los propios datos.

Primero se ajusta el **eje del corredor**: se toman las posiciones de los buses en
movimiento y se les ajusta una línea central, que después se suaviza. El eje no
viene de un archivo de ruta; sale de por dónde circularon los buses. Cada posición
se proyecta sobre ese eje y queda reducida a un solo número: cuánto ha avanzado el
bus a lo largo del corredor. Las posiciones que caen a más de 300 metros del eje se
descartan por no pertenecer al corredor.

El **sentido de marcha** se deduce del signo de ese avance, suavizado sobre varias
posiciones para que un error de GPS aislado no invierta la dirección. Después el
recorrido de cada bus se corta en viajes: un salto de más de treinta minutos sin
señal, una inversión de sentido o una espera prolongada en terminal cierran el
viaje en curso. Por último todo se lleva a una **rejilla de sesenta segundos**, de
modo que en cada minuto exista una foto del corredor completo.

Sobre esa foto se define el intervalo. Para cada par de buses consecutivos en el
mismo sentido, el intervalo es **hace cuánto tiempo el bus de adelante pasó por el
punto donde el de atrás está ahora**. Es un cruce por posición y no por parada, que
es precisamente lo que permite prescindir de la tabla de paradas. Con *N* buses
circulando, el corredor queda descrito en cada minuto por un vector de *N* − 1
números. Si el cruce hallado tiene más de treinta minutos de antigüedad se emite
«sin dato», para no arrastrar un paso de horas antes.

Esta forma no se eligió por comodidad. Se compararon cuatro definiciones
alternativas del intervalo sobre siete criterios de viabilidad —entre ellos la
cobertura, la plausibilidad física de la velocidad implícita y la información
compartida entre buses vecinos—, y ésta ganó en seis de los siete.

![Definición del intervalo entre buses](figuras/esquema-headway.png)

**Fig. 1.** El intervalo entre dos buses consecutivos. Trayectorias ilustrativas,
no datos reales.

### C. Los métodos comparados

Se comparan cuatro. Una **red recurrente** que recibe el vector de intervalos de
los últimos doce minutos junto a cuatro variables de contexto, y emite el vector
completo para el horizonte pedido. Un **conjunto de árboles con refuerzo de
gradiente**, que recibe la misma información en forma de rezagos y variables de
calendario. **Repetir el último valor observado**, que es la línea base obligada de
todo pronóstico de series de tiempo. Y el **promedio histórico por franja horaria**,
que responde con lo que suele pasar a esa hora del día.

Los dos últimos no son adorno. Repetir el último valor es una vara exigente a
horizonte corto y se vuelve débil al alargarlo, de modo que por sí sola dejaría al
aprendiz compitiendo contra nadie a diez minutos. El promedio histórico cubre
justamente ese flanco: como no depende del horizonte, su error es plano, y a diez
minutos se convierte en el competidor real.

Corresponde declarar una asimetría del procedimiento. El conjunto de árboles se
seleccionó sobre veinticuatro configuraciones por celda; la red heredó una única
configuración en dos de los tres corredores. **Donde la red pierde contra los
árboles, ese resultado no es atribuible a la clase de modelo**, y el trabajo no lo
usa como si lo fuera.

### D. Qué cuenta como apelotonamiento

Hay que decidir cuándo un intervalo cuenta como apelotonamiento. La convención del
campo es una fracción del intervalo programado —normalmente un cuarto—, pero aquí
no hay programación contra la cual comparar. Se sustituye por el análogo directo:
**un intervalo cuenta como apelotonamiento si cae por debajo de la mitad del
promedio de su propio vector en ese instante.** Se exige que el vector tenga al
menos tres buses, porque con dos el promedio es poco informativo.

La sustitución es nuestra y se declara como tal: esta forma —fracción del promedio
observado— no se encontró como definición de evento en la literatura publicada. La
elección del valor tampoco es neutral, y el campo lo sabe: los umbrales publicados
van desde veinte segundos hasta un cuarto del intervalo programado, y no existe un
único valor aceptado.

Conviene hacer explícita una propiedad de esta regla, porque es la bisagra de todo
el trabajo. **El corte se mide contra el promedio del propio vector que se está
evaluando.** No es un número fijo en minutos: se mueve con el vector. Aplicado a lo
observado se calibra sobre la dispersión observada; aplicado a un pronóstico se
mide contra la dispersión del pronóstico. Si esas dos dispersiones difieren, no es
el mismo corte aunque se escriba igual. La Sección IV mide qué ocurre cuando se
ignora esa diferencia.

### E. Cómo se evalúa

La partición es **por fecha y nunca al azar**, porque un operador solo dispone del
pasado: 107 días de entrenamiento, 23 de validación y 22 de prueba. Para comprobar
que el resultado no depende del mes elegido, todo se repite sobre tres orígenes que
arrancan el mismo día y alargan el entrenamiento —61, 83 y 107 días—, con períodos
de prueba que no se solapan entre sí. Como los entrenamientos están anidados, esto
establece estabilidad frente a la elección del período de prueba, y no réplica
independiente; se declara así.

![Partición temporal y los tres orígenes](figuras/esquema-particion-temporal.png)

**Fig. 2.** La partición por tiempo y los tres orígenes de evaluación.

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

## IV. Resultados

### A. El resultado escalar y su frontera

A diez minutos de anticipación, el modelo aprendido predice el intervalo entre
buses mejor que repetir el último valor observado. El error absoluto medio baja
1,47 minutos en E2, 1,38 en E4 y 1,17 en E59: entre 21 % y 22 % en los tres
corredores. A un minuto la relación se invierte y repetir el último valor gana,
por 0,46 minutos en E4 y 0,33 en E59; en E2 la diferencia es de 0,07 minutos y no
resiste la prueba estadística una vez que se agrupan las observaciones por día de
servicio.

Dos precisiones acotan ese resultado. La primera es que el cruce no es una
propiedad del aprendizaje profundo: el modelo de árboles lo reproduce entero, y a
diez minutos aventaja a la repetición por 1,59 minutos en E2, 1,09 en E4 y 0,79
en E59. La segunda es que la frontera real no es el horizonte sino qué tan movida
viene la ventana de entrada. Separando cada celda en tercios según la dispersión
de los intervalos que el modelo recibe —y fijando los cortes sobre los datos de
entrenamiento, nunca sobre los de prueba—, la ventaja del aprendiz crece de forma
ordenada del tercio calmo al volátil en 11 de las 12 celdas. Alargar el horizonte
no cambia quién gana: mueve la ventaja hacia tercios cada vez más tranquilos.

Este eje se reporta como contexto y no como contribución. El cruce entre
persistencia y modelo aprendido al alargar el horizonte es conocido en pronóstico
de tráfico. Lo que sigue es el objeto del trabajo.

### B. El pronóstico sale más parejo que la realidad

Un corredor de buses puede describirse, en cada instante, por lo disparejos que
están sus intervalos: la desviación de los intervalos dividida por su promedio.
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

Conviene ser preciso sobre qué es nuevo acá. Que un pronóstico optimizado en error
cuadrático salga más parejo que la realidad está publicado y demostrado como
teorema, pero ese teorema cubre la variabilidad de una serie a lo largo del tiempo.
Lo que medimos es otra cosa: cuán disparejos están los buses **entre sí en un mismo
instante**. Las 36 celdas son por lo tanto un resultado empírico, no un corolario.

![Dispersión observada frente a predicha](figuras/compresion-dispersion.es.png)

**Fig. 3.** Dispersión observada frente a dispersión predicha, horizonte de diez
minutos. La barra de la persistencia iguala a la observada: hereda el vector real
y sirve de control. Los dos aprendices lo aplanan.

![Sesgo de dispersión contra horizonte](figuras/compresion-vs-horizonte.es.png)

**Fig. 4.** El mismo sesgo contra el horizonte. La persistencia no se despega de
cero; los dos aprendices se hunden de forma monótona. La compresión escala con la
distancia que se pide anticipar.

### C. La alarma no suena

El apelotonamiento se define aquí como un intervalo que cae por debajo de la mitad
del promedio de su propio corredor en ese instante. Aplicada a lo observado, esa
regla marca 15 245 eventos en E2 a diez minutos. Aplicada al pronóstico del modelo
aprendido, con el mismo corte, se dispara **catorce veces**.

Repetir el último valor observado dispara 15 083 veces. Puntuada con la medida
habitual de detección, la repetición aparece 253 veces mejor que el modelo. En las
otras celdas el factor va de 1,5 a 36. En los tres casos donde el modelo de árboles
se evalúa así, su puntaje es exactamente cero: no dispara nunca.

Leído sin más contexto, ese resultado dice que el aprendiz es incapaz de ver el
fenómeno que se le pidió anticipar. Hay tres motivos para desconfiar de esa
lectura.

El primero es que el ganador declarado tampoco es bueno. Una regla sin ningún
contenido —marcar todas las celdas como apelotonamiento— supera a la repetición en
5 de las 12 celdas, y en 15 de las 36 al considerar las tres ventanas. Un
procedimiento de evaluación en el que una regla vacía vence al ganador no está
ordenando modelos.

El segundo es el mecanismo de la sección anterior. El corte se mide contra el
promedio del propio vector evaluado. Si el pronóstico es más parejo que la realidad,
sus intervalos se apartan menos de su propio promedio, y el corte deja de alcanzarse
casi siempre. Lo que la regla registra no es que el modelo no vea el evento: es que
el modelo no produce la dispersión necesaria para cruzar un umbral calibrado sobre
otra distribución.

El tercero es que, en esas pocas ocasiones en que sí dispara, el modelo acierta.
De los catorce disparos de E2, diez corresponden a apelotonamientos reales: 71 %
de precisión contra una tasa base de 30 %. La muestra es pequeña y el intervalo de
confianza va aproximadamente de 42 % a 92 %, de modo que la cifra señala un
régimen y no un valor. Las celdas con más disparos lo confirman con menos
incertidumbre: en E59 a diez minutos, 776 aciertos en 1 572 disparos contra una
tasa base de 21 %; en E4, 75 en 150 contra 18 %. El modelo no se equivoca: está
callado. Su cobertura colapsa; su precisión, no. Una medida que castiga por igual
al que calla y al que se equivoca no distingue esos dos casos.

![Tasa de disparo contra tasa real del evento](figuras/artefacto-umbral.es.png)

**Fig. 5.** Fracción de celdas que cada método marca como apelotonamiento, contra
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

Si el 253 midiera una capacidad del modelo, debería ser aproximadamente estable al
cambiar la ventana de prueba. No lo es.

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

Ajustamos el corte sobre una ventana temporal y lo aplicamos a la siguiente, sin
mirar nunca los datos con los que se lo puntúa, y sin tocar el modelo. Fijamos el
corte maximizando la correlación de Matthews. La alternativa habitual —maximizar la
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
categórica— coinciden en quién gana y desde dónde. La disociación entre ellas, que
era el punto de partida de todo este análisis, no existía. La fabricaba el umbral.

Frente a una falla de detección, el campo cambia de modelo: otra arquitectura,
más capas, más datos. Acá no hizo falta ninguna de esas cosas. Los pronósticos
que produce la Fig. 6 son, uno por uno, los mismos que produce la Fig. 5.
No se reentrenó, no se agregó información y no se tocó una línea del modelo. Se
movió un número —dónde se traza la raya entre alarma y silencio— y el ganador
cambió de bando.

De ahí salen las dos consecuencias del trabajo. Para quien evalúa: como ninguna
otra cosa varió, ninguna otra cosa puede explicar la inversión, y el corte queda
identificado como la variable que producía el veredicto. Para quien opera:
reparar esto no cuesta una GPU ni un rediseño. Cuesta recalibrar un umbral con
datos que ya se tienen.

![Ventaja escalar y AUC de detección](figuras/deteccion-sin-umbral.es.png)

**Fig. 6.** Las mismas predicciones puntuadas sin umbral. Eje izquierdo: cuánto
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

### F. Robustez, incluido el ataque más duro que encontramos

El hallazgo no depende del mes: las tres ventanas temporales coinciden en el
veredicto sin umbral en 11 de 12 celdas, y a diez minutos coinciden en las nueve.

Tampoco depende de nuestra definición del evento, y esto es lo que más nos costó
aceptar. La objeción evidente es que el corte relativo —media del propio vector— es
una elección nuestra, y que un corte absoluto en minutos, como el que usa la mayor
parte de la literatura, disolvería el efecto. Lo probamos. **No se atenúa: empeora.**
Bajo la convención dominante del campo, la fracción de eventos que el modelo
efectivamente marca es unas 115 veces menor que bajo nuestra propia regla.
Nuestra elección resultó ser la conservadora.

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

---

## V. Discusión y limitaciones

_(pendiente)_

---

## VI. Conclusión

_(pendiente)_

---

## Referencias

_(pendiente)_
