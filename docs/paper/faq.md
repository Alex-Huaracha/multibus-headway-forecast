# Preguntas frecuentes

Respuestas cortas a las preguntas que se repiten. Cada una apunta a la parte
del paper que ya la responde.

## ¿Por qué no usaste la distancia de Haversine?

Porque el método nunca mide la distancia entre dos buses. El headway es un
tiempo: cuánto hace que el bus de adelante pasó por el punto donde hoy está el
de atrás (Ecuación 10). Para eso solo hace falta saber dónde está cada bus a lo
largo de la ruta, y eso lo da la coordenada de arco $s$, los metros recorridos
sobre el eje. Haversine da la línea recta entre dos puntos, que corta las curvas
y no dice cuál bus va adelante.

Paper: Apéndice A.1, Ecuación (10) y Figura 5.

## ¿Por qué construiste el headway así y no con paradas y horarios?

Porque los registros no los traen. Cada bus emite solo su identificador, el
instante y su coordenada, y Arequipa no publica la geometría de las rutas en
GTFS. Por eso el eje de la ruta se ajusta de los propios registros y el headway
se mide sobre ese eje. La definición no es nuestra: es la de Pilachowski, que
Andres y Nair ya evaluaron con GPS.

Paper: Apéndice A.1.

## ¿Cómo se arma el headway de una ruta si tiene muchos buses?

Las rutas nunca se mezclan. Cada ruta es una empresa distinta, y cada una se
trata por separado de principio a fin.

Dentro de una ruta circulan varios buses a la vez. Cada minuto se toma la
posición de todos ellos y se separan por sentido, ida y vuelta. En cada
sentido, los buses se ordenan del que va más adelante al que va más atrás, y
cada par de buses seguidos da un headway. Si en ese minuto hay 6 buses de ida,
salen 5 headways. Esa lista es el vector de headways de la ruta en ese sentido y
en ese minuto.

El LSTM se entrena por separado para cada ruta: la ruta A tiene su propio
modelo, la B el suyo y la C el suyo. Cada modelo ve solo los vectores de su
ruta, de ida y de vuelta. Además hay un modelo distinto por horizonte (1, 3, 5
y 10 minutos), así que son doce modelos por cada origen de evaluación.

La cantidad de buses cambia minuto a minuto, así que el largo del vector
también. Un par sin headway conserva su lugar en el vector como «sin valor».

Paper: Sección III-A y Apéndices A.1 y A.2.

## ¿Por qué solo entre 63 % y 77 % de los pares tiene headway?

Porque un par queda sin valor cuando no se encuentra el paso del bus de
adelante o cuando el headway supera los treinta minutos. Ese límite existe
porque, si el GPS del bus de adelante tiene huecos, la búsqueda retrocede hasta
una vuelta anterior, horas antes. El par sin valor no se rellena: su posición
se enmascara y no entra ni en el entrenamiento ni en las métricas.

Paper: Apéndice A.1 y Ecuación (2).

## ¿Para qué sirve cada ecuación?

Las ecuaciones siguen el orden del argumento. Las de la (1) a la (5) definen qué
se predice y cuándo hay bunching. La (6) y la (7) muestran por qué el F1
engaña. La (8) y la (9) miden que la predicción es demasiado pareja. La (10)
construye el headway.

**Ecuación (1). Qué se predice.** Con los headways de los últimos 12 minutos y
la hora y el día, el modelo predice el vector completo de headways $H$ minutos
después. Sirve para fijar la tarea antes de hablar de detección.

**Ecuación (2). Qué se minimiza al entrenar.** Es el error cuadrático medio,
calculado solo sobre las posiciones que tienen headway. Sirve para mostrar el
origen del problema. Un modelo que minimiza este error predice valores
intermedios, y por eso su vector sale más parejo que el real.

**Ecuación (3). El threshold del evento.** Es la mitad del promedio del vector
en ese minuto. Sirve para que el threshold se adapte a cada ruta. Una ruta con
buses cada 10 minutos y otra con buses cada 4 no pueden usar el mismo valor en
minutos.

**Ecuación (4). Cuándo hay bunching de verdad.** Una posición del vector
observado es bunching si su headway queda por debajo del threshold de la (3).
Solo se define con 3 headways o más. Sirve como la respuesta correcta contra la
que se puntúa toda alarma.

**Ecuación (5). Cuándo el modelo da alarma.** Es la misma regla de la (4), pero
aplicada al vector predicho y con el promedio del vector predicho. Sirve para
definir el *unadjusted threshold*, la regla que el paper critica. Pasa el mismo
$\rho = 1/2$ del headway observado al predicho sin ajustarlo.

**Ecuación (6). Precisión, recall y F1.** Son las medidas habituales de
detección. Sirven para puntuar las alarmas de la (5) contra los eventos de la
(4), como lo hace el campo.

**Ecuación (7). El F1 descompuesto.** Reescribe el F1 como la precisión
multiplicada por un término que solo depende de cuántas alarmas se emiten. Es
la ecuación central del paper. Explica por qué el LSTM pierde en F1 aunque
acierte más. Emite muy pocas alarmas, ese término se acerca a cero y arrastra
el F1.

**Ecuación (8). El coeficiente de variación (CV).** Mide qué tan desiguales son
los headways de un vector. Vale 0 si todos los buses van igual de separados.
Sirve para mostrar que el vector predicho es mucho más parejo que el real, por
ejemplo 0.18 contra 0.74 en la ruta A a diez minutos.

**Ecuación (9). Cuánta variación conserva la predicción.** Reparte la variación
del headway observado entre la del predicho y la del error. La fracción $r$ es
la parte que conserva la predicción, y $r_0$ es la que el tamaño del error
anticipa. Sirve para probar que la predicción pierde justo la variación que su
error anticipa, y que la causa es el entrenamiento y no los datos.

**Ecuación (10). El headway.** Es el tiempo desde que el bus de adelante pasó
por el punto donde hoy está el de atrás. Sirve para construir el headway desde
el GPS, sin paradas ni horarios.

Paper: Secciones III, IV y V y Apéndice A.1.
