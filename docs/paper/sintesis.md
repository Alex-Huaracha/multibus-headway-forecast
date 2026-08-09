# Qué se hizo, y qué se encontró

Síntesis del trabajo, para leerse de una sentada. No hay ecuaciones ni nombres de
archivo. El detalle completo —cada decisión, cada número, cada límite— está en
[`metodologia.md`](metodologia.md), que es unas seis veces más largo que esto.

---

## El problema

Cuando dos buses de la misma línea terminan circulando pegados, seguido de un hueco
largo sin servicio, el fenómeno se llama ***bunching***. Es el modo de falla
característico de los corredores de alta frecuencia: degrada el servicio aunque la
cantidad de buses sea la correcta.

La cantidad que importa se llama ***headway***: el tiempo que pasa entre un bus y
el siguiente por un mismo punto. Un corredor sano los tiene parecidos entre sí. Uno
con *bunching* los tiene muy desparejos.

El objetivo del trabajo fue **anticiparlo**: predecir, con varios minutos de
antelación, cómo van a quedar esos intervalos, para que un operador pueda intervenir
antes de que el problema ocurra.

Los datos son de **tres corredores de Arequipa**, del sistema de GPS a bordo de los
buses: 152 días seguidos, sin huecos.

---

## El hecho del que cuelga todo lo demás

Los datos de Arequipa **no traen horario programado**. No existe una tabla que diga
"este bus debe pasar cada 8 minutos". Tampoco hay tabla de paradas confiable.

Esa única carencia encadena todo el trabajo:

```
Sin horario
   └─> tampoco hay paradas confiables
          └─> hubo que reconstruir el recorrido desde las posiciones del GPS
                 └─> eso permitió definir el headway por cruce de posición
                        └─> con el headway definido se armó el experimento
                               └─> pero medir detección exige definir el evento,
                                   y la definición estándar necesita el horario
                                   que no existe
                                      └─> hubo que fabricar una alternativa
                                             └─> y al fabricarla apareció el hallazgo
```

Cada paso resuelve el problema que dejó abierto el anterior. Es la razón de que el
trabajo tenga la forma que tiene.

---

## Lo que hubo que fabricar

Sin horario y sin paradas, el *headway* no viene en el dato. Hay que construirlo
desde cero, usando solamente las posiciones que reporta el GPS.

**Primero, el recorrido.** Se ajusta una línea al montón de posiciones de los buses
en movimiento. Lo que queda lejos de esa línea —calles paralelas, depósitos, otras
líneas— se descarta.

![El eje del corredor](figuras/esquema-eje-corredor.png)

**Segundo, se pasa de dos dimensiones a una.** Cada posición se lleva
perpendicularmente sobre esa línea y se convierte en un solo número: cuántos metros
lleva recorridos el bus a lo largo del corredor.

![De dos dimensiones a una](figuras/esquema-proyeccion.png)

Esto es clave y conviene detenerse un segundo. Con latitud y longitud no se puede
decidir sin ambigüedad cuál bus va adelante. Con un solo número sí: el que lo tiene
más grande va más adelantado. Todo el cálculo de intervalos depende de poder ordenar
los buses.

**Tercero, la definición del intervalo:**

> El *headway* de un bus es **hace cuánto tiempo el bus de adelante pasó por el punto
> donde el bus de atrás está ahora.**

![La definición de headway](figuras/esquema-headway.png)

Es una definición por cruce de posición, no por parada. Y eso importa: la vuelve
inmune a la ausencia de una tabla de paradas, que es exactamente lo que falta acá.

No se eligió por intuición. Se compararon cuatro definiciones posibles sobre siete
criterios de calidad, y se adoptó la única que mide tiempo entre pasadas —que es la
cantidad que el operador necesita— sin depender de datos que no existen.

---

## Cómo se armó el experimento

El modelo mira **12 minutos de historia reciente** y predice cómo va a estar el
corredor completo a 1, 3, 5 y 10 minutos hacia adelante.

![Partición temporal](figuras/esquema-particion-temporal.png)

Los 152 días se parten **por tiempo, no al azar**: el modelo aprende con los
primeros, se afina con los del medio, y se mide con los últimos, que no vio nunca.
Si se mezclara al azar, el modelo entrenaría con minutos del futuro y el resultado
sería optimista de mentira, porque en la operación real solo se tiene el pasado.

Se lo compara contra tres rivales, y el orden importa:

- **La persistencia** — predecir que dentro de N minutos todo estará igual que
  ahora. Es la vara mínima. Si un modelo no le gana a esto, no sirve.
- **Un método de árboles de decisión**, el competidor no profundo.
- **El promedio histórico por hora** — un almanaque, ciego al presente, que solo
  mira el reloj.

Todo se evalúa **sobre exactamente las mismas filas**, verificado automáticamente.
Sin eso, parte de la diferencia entre dos modelos viene de qué filas le tocaron a
cada uno y la comparación deja de significar algo.

---

## Lo que se encontró

### 1. El modelo pronostica bien

A 10 minutos de anticipación mejora el error frente a la persistencia **entre un
21 % y un 22 %**, igual en los tres corredores. Y la ventaja se concentra donde más
importa: en los momentos en que el corredor viene más irregular.

Con dos fronteras que conviene dejar dichas: **a 1 y a 3 minutos no hay ventaja** —
a un minuto gana la persistencia. La afirmación sólida empieza a los 5 minutos.

### 2. Pero la alarma no sonaba

Acá aparece lo interesante. Al aplicar el procedimiento de detección estándar del
campo, el mismo modelo que pronostica mejor **tocó la alarma 14 veces donde había
15 245 eventos reales**.

Leído de frente: parecía haberse vuelto completamente ciego a la irregularidad.

Pero había una pista incómoda en la misma tabla. Un detector que no sabe nada —que
marca absolutamente todas las celdas como *bunching*— saca mejor puntaje que el
"ganador" de esa comparación. Los dos métodos perdían contra una regla vacía. Lo
único que los distinguía era cuánto perdían.

**Ninguna de las dos cifras estaba midiendo qué sabe el modelo. Estaban midiendo
dónde había quedado el corte.**

### 3. Era el instrumento, no el modelo

La pregunta que desarma el veredicto es una sola: ese número, ¿mide al modelo, o
mide a la regla con la que se lo está midiendo?

Es contestable, porque las dos respuestas predicen cosas distintas. Si al modelo le
falta la información, mover el corte no la va a recuperar. Si la información está y
lo que falla es dónde quedó la raya, entonces correrla la hace reaparecer.

Se hicieron las dos pruebas que separan esas hipótesis. Las dos dieron lo mismo:
**la información estaba presente. El modelo no es ciego.**

### 4. Por qué pasaba

Todo modelo que predice un solo número **aplana la realidad**: describe un corredor
más parejo de lo que es.

Un ejemplo con buses que pasan cada 10 minutos en promedio. Un corredor ordenado da
intervalos como 9, 10, 11, 10. Uno con *bunching*, como 2, 18, 3, 17. **El promedio
de los dos es el mismo.** Lo que los distingue es qué tan desparejos son.

Medido con esa idea, la realidad de un corredor da **0.79 y la predicción del modelo
da 0.16**. El pronóstico describe un corredor casi cinco veces más parejo de lo que
realmente es.

![El aplanamiento del pronóstico](../resultados/contiguo-compresion-dispersion.png)

Y no es un defecto de las redes neuronales: el método de árboles aplana igual o más.
Es lo que hace cualquier pronóstico que devuelve un único número, porque predecir el
valor esperado es exactamente promediar los futuros posibles. En estadística está
demostrado como teorema desde hace más de una década.

**Ahí está el mecanismo completo.** La regla que define el *bunching* está calibrada
sobre la realidad, que es despareja. Aplicada sobre una predicción aplanada,
simplemente nunca se dispara.

### 5. Se repara moviendo el corte, no el modelo

Reajustando dónde va la raya —con datos de un período anterior, nunca con los del
período donde después se mide— **el veredicto se da vuelta**: el modelo detecta
mejor que el método trivial, en los tres corredores.

Y midiendo sin ninguna raya, lo mismo.

![El veredicto sin umbral](../resultados/contiguo-deteccion-sin-umbral.png)

---

## El aporte, en una frase

> **La degradación de detección que este campo viene atribuyendo a los modelos es,
> al menos en parte, un artefacto de trasplantar un umbral calibrado sobre la
> realidad a un pronóstico que está aplanado por construcción. Y se corrige moviendo
> el corte, no cambiando de modelo.**

Lo que le da peso es que no es una opinión sobre el campo: el procedimiento que
produce el problema está enunciado, con esas palabras, por el trabajo más citado del
subcampo. Y la consecuencia no es un accidente de implementación — es necesaria,
porque el aplanamiento del pronóstico es un teorema, no una casualidad.

**Aporte secundario, y no menor:** todo el método funciona sin horario publicado, sin
GTFS y sin tabla de paradas. Eso lo vuelve aplicable a la mayoría de las ciudades
donde el *bunching* es un problema real y el dato ordenado no existe.

---

## Lo que este trabajo NO hace

Conviene ser explícito, porque marca el borde de lo que se puede afirmar.

- **No construye un sistema de despacho.** Lo que se establece es que la información
  está presente y es explotable, no que alcance para operar un servicio real.
- **Las magnitudes absolutas son modestas.** El error de pronóstico sigue siendo
  grande frente al tamaño típico de un intervalo, y la calidad de detección, aunque
  mejor que el azar, recorre una fracción chica del camino.
- **Contra una vara más exigente que la persistencia** —el almanaque por franja
  horaria— el modelo gana en dos de los tres corredores, no en los tres.
- **Tres corredores de una ciudad, cinco meses.** El alcance geográfico y temporal es
  el que es.
- **Hay defectos declarados**, identificados en revisión interna y documentados con
  su efecto: entre ellos, una variable de entrada que no respeta del todo la regla de
  no usar información del futuro. Está medido que el hallazgo no depende de ella.

Los límites completos, con su magnitud medida, están en la Parte VI de
[`metodologia.md`](metodologia.md).

---

## Dónde está cada cosa

| | |
|---|---|
| Este documento | La síntesis. Qué se hizo y qué se encontró |
| [`metodologia.md`](metodologia.md) | El registro completo: cada decisión, con su justificación y su número |
| [`fuentes-verificadas.md`](fuentes-verificadas.md) | Las fuentes citadas, verificadas una por una contra el texto original |
