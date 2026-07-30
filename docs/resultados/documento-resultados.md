# El umbral, no el modelo, decide quién ve el *bunching*: un pronóstico aplanado parece ciego sin serlo[^headway]

**Documento de resultados** · Corredores E2, E59 y E4 · Horizontes de 1 a 10 minutos · Pipeline contiguo

> Este documento presenta **qué encontramos**. El preprocesamiento se resume al mínimo necesario para que los resultados sean reproducibles.
>
> **Todas las cifras provienen del pipeline contiguo** (familias `21-lstm-contiguous` y `22-xgb-contiguous`), que cumple los tres contratos de la Sección 2. Los resultados de las familias 11/12/13 y 17/18/19 **no son comparables** con estos y se conservan solo para el ranking entre arquitecturas, cuya validez descansa en que las tres comparten el mismo sesgo.

---

## 1. El hallazgo

> **Un pronóstico puntual predice un corredor más regular que el real. Si le montás encima una alarma calibrada sobre datos reales, la alarma no suena — y el diagnóstico natural, "el modelo no ve el *bunching*", es falso.**

### El problema

Una empresa de transporte quiere que le avisen cuándo dos buses se le van a juntar. Eso es el *bunching*[^bunching]: un *headway* que colapsa hacia cero mientras el de al lado se abre. Es la falla que le arruina el servicio y ocurre en el **17 % al 30 %** de las celdas de estos corredores — no es un evento raro.

La receta obvia es: entrenar un modelo que prediga los *headways*, definir la regla de alarma sobre esa predicción, y despachar. Nosotros la ejecutamos. Entrenamos un LSTM[^lstm] y un XGBoost[^xgboost] nivelado sobre tres corredores reales, los medimos contra la persistencia[^persistencia] sobre muestras idénticas, y aplicamos una regla de umbral relativo: **marcar toda celda cuyo *headway* caiga por debajo de la mitad de la media de su propio vector**.

> **Sobre esa regla, con precisión, porque una versión anterior de este documento la llamaba "estándar" y no lo es.** La forma relativa que domina la literatura es una fracción del *headway* **programado** —un cuarto, en Yu et al. (2016) y Moreira-Matias et al. (2016); un medio, en el TCQSM—, no una fracción de la media del propio vector. Nosotros **no tenemos horario programado**: los datos son GPS crudos sin GTFS, así que la media del vector observado es el sustituto disponible del *headway* programado. Es una elección defendible y es **nuestra**, no heredada.
>
> Y la pregunta obvia —si el colapso no es más que un artefacto de esa forma auto-referencial— **está medida**, no argumentada. La Sección 5.6 repite toda la detección con un corte **absoluto en minutos** calibrado fuera de muestra. El resultado va en contra de lo que esperábamos: **el colapso empeora**, y con la convención dominante del campo (un cuarto del *headway* programado) es **110 veces peor** que con nuestra regla. El alcance del hallazgo es entonces más ancho que los umbrales auto-referenciales — nuestra elección de umbral resultó ser la **conservadora**.

El resultado escalar salió como se esperaba: a partir de 5 minutos de anticipación los aprendices le ganan a la persistencia con holgura creciente, hasta **1.47 min de MAE**[^mae] a 10 minutos.

La alarma, en cambio, no sonó nunca.

| E2, a 10 minutos de anticipación | LSTM | Persistencia |
|---|---|---|
| MAE (menor es mejor) | **5.32 min** | 6.79 min |
| *Bunching* — F1[^f1] con el umbral fijo | 0.0013 | **0.332** |
| Veces que disparó, en 50 356 oportunidades | **14** | 15 084 |
| Eventos reales que había que agarrar | 15 245 | 15 245 |

Catorce disparos donde había quince mil eventos. Un factor de **253** contra la persistencia. La lectura inmediata —y la que este documento sostuvo en versiones anteriores— es que el modelo se volvió ciego a la irregularidad.

### Por qué esa lectura es falsa

Dos cosas la desarman, y las dos salen de los mismos datos.

**Primera: la persistencia no estaba detectando nada.** Disparó 15 084 veces y acertó 5 036 — un **33 % de precisión contra una tasa base del 30 %**. Estaba disparando a la frecuencia correcta con acierto de casi-azar, y el F1 premia exactamente eso. La prueba: *marcar todas las celdas* saca F1 = 0.465 en esa misma celda, **más** que los 0.332 de la persistencia. El ganador declarado perdía contra una regla constante, en los tres corredores a h=10.

**Segunda: el umbral publicado era el umbral óptimo de la persistencia.** La persistencia propaga el vector observado, así que hereda su dispersión real (CV[^cv] ≈ 0.79) y el corte del 0.5 cae donde fue diseñado para caer. Un pronóstico puntual emite un vector comprimido (CV ≈ 0.16), así que ese mismo corte *relativo* le queda a tres desviaciones estándar dentro de la cola izquierda. Y esto no es una conjetura: ajustando el corte libremente sobre una ventana anterior, **la persistencia recupera 0.5× en 11 de las 12 celdas**, mientras que el LSTM necesita entre 0.58× y 0.91×. La regla estaba escrita en las unidades de un competidor.

![El artefacto de umbral](contiguo-artefacto-umbral.png)

*Figura 1 — El artefacto en una imagen. La línea gris de la persistencia se apoya sobre la punteada de la tasa real del evento: dispara casi exactamente tan seguido como el bunching ocurre, porque el corte fue calibrado en el espacio donde ella vive. La roja del LSTM se hunde a cero. **Ninguna de las dos curvas mide qué sabe el modelo; miden dónde quedó el corte.***

### La corrección

Sacale el umbral de encima y el veredicto se da vuelta.

| E2, a 10 minutos | LSTM | Persistencia |
|---|---|---|
| AUC[^auc] (sin umbral) | **0.565** | 0.528 |
| MCC[^mcc] con umbral calibrado fuera de muestra | **0.085** | 0.027 |
| Precisión cuando dispara, con el umbral fijo | **71 %** (10 de 14) | 33 % (5 036 de 15 084) |

Esa última fila es la que cierra el caso. Los catorce disparos del LSTM aciertan al **71 %** contra una tasa base del 30 %; los quince mil de la persistencia aciertan al 33 %. El modelo no está equivocado cuando habla: está callado. Y estar callado es lo que arregla un umbral.

*Con la salvedad obvia:* 10 de 14 tiene un intervalo de confianza ancho (≈ 42 %–92 % al 95 %). El punto no descansa en esa celda. En E59 h=10, donde el LSTM dispara **1 573** veces, acierta **777** — un 49 % contra una tasa base del 21 %, o sea **2.4× el azar** con muestra de sobra. En E4 h=10: 75 de 150, un 50 % contra el 18 %. El patrón es el mismo en los tres y el volumen lo sostiene en dos.

**A 10 minutos el LSTM discrimina el *bunching* mejor que la persistencia, en los tres corredores, con y sin umbral.** A 1 minuto la persistencia gana, también en los tres — y también gana el MAE ahí. Las dos métricas **coinciden** una vez removido el artefacto: la persistencia manda en el horizonte corto, el aprendiz en el largo. No hay disociación.

![El veredicto sin umbral](contiguo-deteccion-sin-umbral.png)

*Figura 2 — Los dos cruces, juntos. En azul, cuánto MAE le gana el LSTM a la persistencia. En rojo y gris, el AUC de detección de bunching, que es invariante a cualquier reescalado monótono del pronóstico y por lo tanto no puede ser movido por la compresión del vector. Los dos cruces van en el mismo sentido y en la misma zona. **Comparar esta figura con la anterior es el aporte del trabajo.***

### Qué queda en pie, y por qué importa

Lo que sobrevive intacto es el aplanamiento: **el sesgo del coeficiente de variación es negativo en las 36 celdas** de corredor × horizonte × ventana, y empeora monotónicamente con el horizonte. El LSTM predice un corredor con CV de 0.16 cuando el real es 0.79. Eso es real, es sistemático, y el MAE agregado no lo ve.

Pero su costo no es informativo, es **de unidades**. El aporte de este trabajo es esa distinción, medida:

1. **Todo pronóstico puntual está sub-disperso**, porque toda pérdida puntual apunta a un funcional central de la distribución condicional — la mediana si es MAE, la media si es error cuadrático. No es un defecto del LSTM ni del MAE en particular.
2. **Por eso una regla de evento calibrada en el espacio de las observaciones no es transportable al espacio del pronóstico.** Trasplantarla fabrica una degradación aparente de hasta 253× que no existe en la información.
3. **Y por eso el despliegue ingenuo falla por un motivo que no tiene nada que ver con lo que el modelo sabe** — con un arreglo concreto y barato: calibrar el corte sobre una ventana anterior, o puntuar sin umbral.

Para la empresa la diferencia es todo. "El modelo no sirve para anticipar *bunching*" cierra la línea de trabajo. "El modelo sirve, pero la alarma está mal seteada" es un ajuste de un escalar sobre datos que ya tenés.

**La advertencia metodológica es el punto:** cualquier trabajo que evalúe detección de eventos trasplantando un umbral relativo sobre un pronóstico puntual está midiendo su propio umbral, no su modelo. Y si además reporta solo error escalar agregado, tampoco ve el aplanamiento que causa el problema.

Lo demostramos en cuatro pasos: **(1)** el resultado escalar, y que su frontera no es el horizonte sino la volatilidad → **(2)** cuánto de eso resiste una prueba estadística honesta → **(3)** el aplanamiento, el artefacto de umbral y el veredicto corregido → **(4)** qué sobrevive a los ataques obvios.

---

## 2. El terreno de juego

### Quién compite

| Modelo | Qué hace |
|---|---|
| **Persistencia (B1)**[^persistencia] | Repite el último *headway* observado. El rival serio: en series cortas es sorprendentemente difícil de superar. |
| **XGBoost**[^xgboost] | *Gradient boosting* entrenado que ve la **misma ventana de 12 pasos** que la red, más hora, día y dirección. El competidor aprendido. |
| **LSTM**[^lstm] | Red recurrente sobre la secuencia de *headways*. |

Las arquitecturas espaciales (SpatialConvLSTM, SpatialTransformer) **no superan al LSTM plano** en estos datos. Ese nulo espacial se estableció sobre las familias congeladas y no se rehizo bajo el pipeline contiguo; se reporta como resultado previo, no como parte de esta evidencia.

### El terreno

- **3 corredores:** E2, E59 y E4. E4 es el más chico (19 buses) y aporta **validez externa acotada a la escala de flota** — otra línea, no otra ciudad.
- **4 horizontes:** 1, 3, 5 y 10 minutos.
- **Datos:** 152 días (2023-10-01 → 2024-02-29), divididos temporalmente[^split]: entrenamiento hasta 2024-01-15, validación hasta 2024-02-07, **prueba 2024-02-08 → 2024-02-29** (22 días). Todo lo reportado es sobre prueba.

### Los tres contratos

Una auditoría adversarial previa encontró que la comparación anterior no era atribuible: los modelos puntuaban sobre poblaciones distintas y las ventanas no eran contiguas en el tiempo. El pipeline se reconstruyó sobre tres contratos, y cada uno se verifica en cada corrida:

| Contrato | Qué garantiza | Cómo se verifica |
|---|---|---|
| **C1 — Identidad de muestra** | Una muestra por `(empresa, sentido, instante de inicio, horizonte)`. Los tres modelos puntúan las mismas celdas. | Cada kernel recomputa el índice y compara su SHA-256 contra un manifiesto congelado. |
| **C2 — Contigüidad temporal** | Los `12 + h` instantes de una ventana son minutos **consecutivos**. El horizonte es tiempo, no posición de fila. | Verificado al materializar; una violación aborta. |
| **C3 — Frontera de información** | Ninguna variable usa información posterior al instante de predicción. Se eliminó la bandera de día atípico, que era un agregado del día completo. | El gate de entrada falla cerrado si la variable reaparece. |

**Costo de exigir contigüidad:** entre el 81.9 % y el 90.2 % de los *snapshots*[^snapshot] sobreviven. Se perdió menos del 20 % de los datos y se ganó que el horizonte signifique lo que dice.

**Verificación del contrato C1.** Cada modelo se puntúa dos veces: sobre sus propias filas y sobre la intersección de los tres. Si C1 se cumple, restringir a la intersección no debe mover nada.

> Sesgo de encuadre medido: **0.001 min** como máximo, sobre 36 filas. En el pipeline anterior era de **0.28 a 0.53 min** — más grande que la mayoría de los márgenes que se reclamaban encima. La comparación ahora es atribuible; antes no lo era.

---

## 3. El resultado escalar, y dónde está su frontera real

![Curva de degradación](contiguo-degradacion.png)

*Figura 3 — MAE frente al horizonte, por corredor. Cuanto más bajo, mejor. La persistencia parte abajo a h = 1 y termina arriba a h = 10: ese es el cruce, y el XGBoost lo recorre igual que el LSTM.*

Medido sobre la población pareada a tres bandas (entre 75 747 y 240 907 predicciones escalares por celda):

**Δ MAE contra la persistencia** (negativo = el aprendiz gana):

| Corredor | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| E2 | +0.067 | −0.851 | −1.109 | **−1.473** |
| E59 | +0.334 | −0.186 | −0.491 | **−1.173** |
| E4 | +0.464 | −0.064 | −0.536 | **−1.381** |

Dos lecturas que cambian el titular respecto de versiones anteriores de este documento:

**El XGBoost reproduce el patrón completo.** Contra persistencia, a h=10: −1.585 en E2, −0.787 en E59, −1.085 en E4. El cruce **no es una propiedad del Deep Learning**, es una propiedad del problema: existe un umbral de anticipación a partir del cual el último valor observado deja de ser suficiente, y cualquier aprendiz razonable lo cruza.

**El LSTM contra el XGBoost se parte por corredor**, con signos opuestos que el horizonte amplifica: a h=10 el XGBoost gana en E2 (+0.113) y el LSTM gana en E59 (−0.385) y E4 (−0.295). No hay un ganador global.

> ⚠️ **Y ese contraste no está nivelado.** El XGBoost recibió **24 configuraciones por celda** elegidas en validación; el LSTM recibió **1** en E2/E59 y **3** en E4, heredadas de una fase previa. La asimetría corre **en contra** de la red. Consecuencia directa: *"el LSTM gana en E59"* es seguro, porque gana con menos presupuesto; *"el XGBoost gana en E2"* **no es atribuible a la clase de modelo**. Nivelar cuesta unas 14 horas de GPU y no se hizo.

### A h=1 el MAE y el error cuadrático nombran ganadores opuestos

Hay una inversión de signo en la tabla de significancia que las versiones anteriores de este documento no narraban, y que importa porque desarma la explicación mecánica que sostenían. A **h=1**, contra la persistencia y sobre las mismas filas:

| Corredor | Δ MAE | Gana | *p* | Δ error cuadrático[^rmse] | Gana | *p* |
|---|---|---|---|---|---|---|
| E2 | +0.067 min | persistencia | 0.062 | −15.36 min² | **LSTM** | 7.3e−18 |
| E4 | +0.464 min | persistencia | 1.0e−13 | −6.49 min² | **LSTM** | 2.7e−14 |
| E59 | +0.334 min | persistencia | 2.6e−13 | −8.35 min² | **LSTM** | 8.4e−16 |

*(Varianza agrupada por día de servicio, G = 22.)*

El LSTM aplanado **pierde** el error absoluto y **gana** el cuadrático, en los tres corredores, con significancia holgada en cinco de las seis celdas. Es el comportamiento esperable de un pronóstico contraído: la contracción evita los errores grandes —que el cuadrático castiga desproporcionadamente— al costo de fallar más seguido por poco, que es lo único que el absoluto cuenta.

**Y desmiente una afirmación que este documento hacía.** Versiones anteriores explicaban el aplanamiento diciendo que "el MAE premia contraer". Si eso fuera cierto, el vector aplanado no podría perder el MAE y ganar el cuadrático a la vez. El pronóstico que minimiza el error absoluto es la **mediana** condicional y el que minimiza el cuadrático es la **media**: las dos son medidas de centro, así que la sub-dispersión no viene de elegir una pérdida sobre la otra — viene de emitir **un solo número por celda**. La afirmación se corrige acá y en el glosario, y su consecuencia se desarrolla en la Sección 5.2.

### La frontera no es el horizonte, es la volatilidad

Estratificamos cada predicción por la **dispersión de su propia ventana de entrada** — cuánto se movía el *headway* en los 12 minutos que el modelo efectivamente vio. Es una variable conocida al momento de predecir, con umbrales congelados en train+val y aplicados a test, así que condicionar sobre ella y después testear es legítimo.

![Frontera de volatilidad](contiguo-volatilidad.png)

*Figura 4 — Δ MAE contra la persistencia según la volatilidad de la ventana de entrada. Cada línea es un horizonte. Todas descienden de izquierda a derecha: dentro de cualquier horizonte, el aprendiz gana más cuanto más se movía el corredor. Y alargar el horizonte baja la línea entera, hasta que incluso el tercil calmo queda por debajo de cero.*

**Δ MAE contra la persistencia, por tercil de volatilidad de la ventana:**

| Celda | Ventana calma | Ventana media | Ventana volátil |
|---|---|---|---|
| E2 h=1 | +0.218 | +0.086 | −0.080 |
| E4 h=3 | **+0.370** | +0.006 | **−0.451** |
| E59 h=3 | +0.159 | −0.157 | −0.559 |
| E4 h=10 | −0.659 | −1.116 | −2.172 |

En **11 de las 12 celdas** la ventaja crece de forma ordenada del tercil calmo al volátil, y dentro de cada tercil crece con el horizonte. La excepción es E59 h=1, donde el tercil medio y el volátil quedan empatados (+0.29502 contra +0.29531): la diferencia es de **tres diezmilésimas de minuto**, dos órdenes de magnitud por debajo del ruido de semilla (±0.024 min), así que es un empate y no una inversión. El gradiente calmo → medio sí se cumple en las 12.

> **El cruce no es un umbral de horizonte: es un umbral de volatilidad que el horizonte va cruzando.** El aprendiz gana donde el corredor está inestable; la persistencia gana donde está calmo. Alargar el horizonte empuja la ventaja del aprendiz hacia los terciles cada vez más calmos, hasta cubrirlos todos.

Eso explica de dónde sale el agregado engañoso de E4 a h=3: **−0.064 min** es la mezcla de perder claramente en dos tercios de la masa y ganar claramente en el otro tercio.

---

## 4. ¿Cuánto de esto resiste una prueba honesta?

### El *n* efectivo son 22 días, no 90 000 filas

Las muestras del mismo día de servicio comparten clima, incidentes y demanda: un accidente a las 08:00 moldea toda la mañana. Tratarlas como independientes infla la significancia. La varianza correcta se agrupa por **día de servicio**, y el conjunto de prueba tiene **22 días**. Ese es el tamaño de muestra real.

Al corregirlo, tres verdictos se caen:

| Celda | *p* sin agrupar | *p* agrupado por día |
|---|---|---|
| E2 h=1, LSTM vs persistencia | 0.000129 | **0.0619** |
| E4 h=3, LSTM vs persistencia | 0.00084 | **0.1849** |
| E2 h=1, XGBoost vs persistencia | 0.0285 | **0.2320** |

A h≥5 todo sigue significativo con márgenes amplios (*p* < 1e-9 en las nueve celdas). El daño está concentrado en h=1 y h=3.

### A h=3 no hay victoria declarable

Cuatro métodos independientes convergen en lo mismo:

| Método | Qué dice de h=3 |
|---|---|
| **Media contra mediana** | En E4 y E59 el LSTM gana el MAE promedio pero **pierde en la mayoría de las muestras individuales**: gana el 46.0 % y el 47.3 % de las veces, con medianas de +0.185 y +0.155 min. El Wilcoxon[^wilcoxon] unilateral en la dirección que afirma la media da *p* = 1.000 y *p* = 0.952. |
| **Terciles de volatilidad** | Pierde en el tercil calmo, empata en el medio, gana en el volátil (Sección 3). |
| **Direcciones** | En E4 los dos sentidos se contradicen: +0.078 en uno, −0.170 en el otro. En E59 un sentido prácticamente empata (−0.019). |
| **Enrutador** | Es el **único** horizonte donde conmutar entre modelos paga (Sección 6). |

La lectura honesta —el aprendiz cambia muchas pérdidas chicas por pocas ganancias grandes— es más informativa que "el DL gana", y encaja con todo lo demás del documento.

### El titular defendible

| Horizonte | Qué se puede afirmar |
|---|---|
| **h=1** | Gana la persistencia. Firme en E4 y E59; **al borde en E2** (*p* = 0.062). |
| **h=3** | **Zona de transición.** Sin victoria declarable. |
| **h≥5** | **El aprendiz gana en media y en mediana, con significancia amplia, en los tres corredores.** Esta es la afirmación sólida. |

### ¿Y si el mes fuera otro?

Todo lo anterior sale de **una** ventana de prueba de 22 días. La objeción inmediata es que el titular sea una propiedad de febrero de 2024 y no del problema. Para responderla se re-corrió el protocolo completo —winsorización, portón de población, entrenamiento, exportación— en dos orígenes anteriores. No es una re-partición de los mismos residuos: son 16 entrenamientos nuevos sobre ventanas que no se solapan con la publicada.

| Origen | Entrena | Prueba |
|---|---|---|
| `r1` | 61 días | 2023-12-23 → 2024-01-13 |
| `r2` | 83 días | 2024-01-14 → 2024-02-04 |
| `main` | 107 días | 2024-02-08 → 2024-02-29 (la publicada) |

**11 de las 12 celdas ponen la victoria del mismo lado en los tres orígenes.** El signo de Δ MAE, donde negativo es victoria del aprendiz:

| Celda | `r1` | `r2` | `main` | ¿Coincide? |
|---|---|---|---|---|
| E2 h=1 | +0.041 | +0.054 | +0.066 | sí |
| E2 h=3 | −0.734 | −0.794 | −0.851 | sí |
| E2 h=5 | −0.993 | −1.074 | −1.109 | sí |
| E2 h=10 | −1.398 | −1.413 | −1.473 | sí |
| E59 h=1 | +0.374 | +0.409 | +0.334 | sí |
| E59 h=3 | −0.134 | −0.120 | −0.186 | sí |
| E59 h=5 | −0.429 | −0.405 | −0.491 | sí |
| E59 h=10 | −1.046 | −1.073 | −1.173 | sí |
| E4 h=1 | +0.459 | +0.424 | +0.464 | sí |
| **E4 h=3** | **+0.167** | **−0.017** | **−0.064** | **no** |
| E4 h=5 | −0.286 | −0.520 | −0.536 | sí |
| E4 h=10 | −1.215 | −1.375 | −1.381 | sí |

**La afirmación sólida se sostiene entera.** A h≥5 las **18 celdas** —tres corredores por dos horizontes por tres orígenes— dan ventaja al aprendiz, y las 18 son significativas con la varianza agrupada por día. Ninguna depende del mes.

**La única que se da vuelta es la que nunca fue una afirmación.** E4 h=3 es la celda que esta misma sección ya declaraba no significativa en la ventana publicada. Fuera de ella se comporta igual: *p* = 0.183 en `main` y 0.720 en `r2`, y solo en `r1` alcanza significancia, del lado de la persistencia. El desacuerdo no tumba un resultado — confirma que ahí, para ese corredor, el cruce está justo en el medio y no hay victoria que reclamar. Coincide con lo que ya decían los otros cuatro métodos.

**Y el borde de E2 h=1 tampoco era del mes.** No alcanza significancia en ninguno de los tres orígenes (*p* = 0.299 en `r1`, 0.068 en `r2`, 0.064 en `main`). La ventaja de la persistencia ahí es de cuatro segundos: la dirección es estable, el tamaño no se distingue de cero. La salvedad del titular pasa de "al borde en esta ventana" a **"al borde en las tres"**, que es una afirmación más fuerte, no más débil.

Hay algo más que la tabla de signos no muestra: **en los nueve pares (corredor, origen), Δ MAE cae monótonamente con el horizonte.** No solo aparece el cruce en las tres ventanas — aparece con la misma forma. Es el horizonte el que mueve la ventaja, y lo hace igual en diciembre, en enero y en febrero.

> Una advertencia de lectura. Esta tabla puntúa sobre la población completa del LSTM, mientras que las tablas de significancia de esta sección puntúan sobre la población LSTM∩XGBoost, porque el XGBoost no se re-corrió en los orígenes de rolling. La diferencia es de unas 11 filas en 90 000 y mueve el tercer decimal de *p* (E2 h=1: 0.0619 publicado contra 0.0638 acá). Se eligió comparabilidad **entre** ventanas antes que con la tabla publicada: restringir un origen y no los otros dos habría vaciado de sentido la comparación.

---

## 5. El aporte: el aplanamiento es real, la ceguera no

La afirmación de predecir "el vector completo de *headways*" no puede sostenerse con MAE agregado, porque el MAE agregado no distingue un pronóstico vectorial de N pronósticos escalares sueltos. Medimos tres cosas que sí lo distinguen — y una de las tres nos hizo retirar el titular anterior.

> **Nota de retractación.** Las versiones previas de este documento titulaban "la métrica decide el ganador" y sostenían que el aprendiz pierde la detección de *bunching* en las 12 celdas por factores de hasta 253×. Ese número es reproducible y está acá abajo, pero **la lectura era incorrecta**: dependía por completo de un umbral calibrado en el espacio de las observaciones. Las secciones 5.3 y 5.4 se reescribieron enteras; 5.1 y 5.2 se sostienen.

### 5.1 La posición dentro del vector sí importa

El MAE por posición no es plano: la dispersión relativa entre la mejor y la peor posición va de 0.14 a 1.35 según la celda. Hay estructura posicional que el promedio estaba borrando.

**Pero este resultado no soporta el peso que se le puso, y conviene desarmarlo acá antes que un revisor lo haga.** Dos objeciones, las dos válidas:

- **El techo lo fija la cola.** El 1.35 sale de posiciones con casi ningún dato: en E2 h=10 la posición de peor MAE es la 14, con **n = 2** (MAE 13.28) contra 7.09 en la posición 12, que tiene n = 78. El bin más chico de todo el CSV tiene **n = 1**. Exigiendo n ≥ 100 el rango se desploma a **0.14–0.53**, y el perfil que queda es una **U** —mínimo en el medio del vector— y no un gradiente.
- **No es una propiedad del aprendiz.** La persistencia dispersa **más** que el LSTM (0.17–1.84 contra 0.14–1.35; con n ≥ 100, 0.55 contra 0.53), y el XGBoost más todavía. Si el modelo aprendido no dispersa más que copiar el último valor, la estructura posicional es del **dato**, no de lo que el modelo aprendió sobre las posiciones.

Lo que queda en pie es acotado: el MAE agregado borra estructura posicional real, y por eso reportarlo solo es insuficiente. Lo que **no** queda en pie es leerlo como evidencia de que los modelos aprendieron algo específico del vector. La afirmación de versiones anteriores —que este era el resultado que apoyaba el encuadre original— **se retira**.

### 5.2 Los aprendices aplanan el servicio

El coeficiente de variación[^cv] del vector de *headways* mide **cuán desparejo está el corredor**. Antes de usarlo conviene ver por qué hace falta, porque el promedio solo no alcanza:

| | Huecos entre buses | Promedio | CV |
|---|---|---|---|
| Corredor A | 9, 10, 11, 10 min | **10 min** | 0.07 |
| Corredor B | 1, 19, 2, 18 min | **10 min** | 0.85 |

Mismo promedio, servicios opuestos. En A el pasajero espera diez minutos siempre. En B los buses pasan de a pares y dejan veinte minutos de hueco — eso *es* el *bunching*. **El promedio no los distingue; el coeficiente de variación sí.** Y se divide por la media en lugar de usar la desviación estándar pelada porque dos minutos de desvío son un desastre en un corredor de 5 minutos y son irrelevantes en uno de 15: dividir lo vuelve comparable entre corredores.

Dicho en esos términos, lo que sigue es que **el modelo predice el corredor A cuando la realidad es el corredor B.**

Es una propiedad **del vector como un todo**: un modelo puede acertar razonablemente cada *headway* individual y aun así destruir la forma.

> **Dos precisiones que una versión anterior de esta sección omitía, y las dos importan.**
>
> **Uno: nuestro CV no es el `cvh` del TCQSM.** El manual define `cvh` = desviación estándar de las *desviaciones respecto del horario*, dividida por la media del *headway* **programado** (Ec. 3-7). El nuestro es σ(*h*)/media(*h*) sobre el vector observado. Coinciden solo si el horario es constante y la media real lo iguala. No tenemos horario, así que no podemos citar la Ec. 3-7 como definición de lo que medimos — pero sí podemos citar su escala de nivel de servicio, que es donde está la fuerza del argumento (ver §5.3).
>
> **Dos: no es "la métrica estándar en operación".** El TCQSM la prescribe como medida de fiabilidad para servicio de alta frecuencia (≤10 min), con una escala de nivel de servicio cuyas bandas altas están definidas literalmente en términos de *bunching*. Pero el relevamiento de Trompet, Liu y Graham (2011) sobre doce operadores del *International Bus Benchmarking Group* muestra que **ninguno** usa coeficiente de variación: usan *Wait Assessment*, *Excess Wait Time* o *Service Regularity*. La formulación correcta es "la medida que el TCQSM prescribe", no "la que usa la industria".

| | CV real | CV que predice el LSTM | Sesgo |
|---|---|---|---|
| E2 h=1 | 0.777 | 0.362 | −0.415 |
| E2 h=10 | 0.787 | **0.161** | **−0.626** |
| E4 h=10 | 0.577 | 0.213 | −0.365 |
| E59 h=10 | 0.614 | 0.260 | −0.354 |

El sesgo es negativo en las 12 celdas y **empeora monotónicamente con el horizonte**. La persistencia tiene sesgo ≈ 0 — propaga el vector observado, así que conserva su forma por construcción. Eso no es un truco: es exactamente la propiedad que los aprendices pierden.

**Y no es un defecto de este LSTM ni de esta pérdida.** Toda pérdida puntual apunta a un funcional central de la distribución condicional: el error absoluto a la mediana, el cuadrático a la media. Un pronóstico que devuelve un único número por celda devuelve una medida de centro, y por lo tanto está sub-disperso por construcción.

Para el caso de la media hay una **identidad** que lo respalda: la descomposición de la varianza da Var(*Y*) = Var(E[*Y*|*X*]) + E[Var(*Y*|*X*)], así que la varianza del pronóstico óptimo no puede superar la del observable, y la iguala solo si *Y* es determinístico dado *X*. Patton y Timmermann (2012, Corolario 2) llevan eso un paso más: para un pronóstico racional óptimo en error cuadrático, la varianza es **débilmente decreciente en el horizonte**. O sea que nuestro deterioro monótono con el horizonte no es un descubrimiento — es esa cota teórica volviéndose visible sobre datos reales, y conviene presentarlo así.

> **Pero cuidado con qué acota esa identidad, porque no es lo que medimos.** El teorema acota la varianza **temporal de una serie escalar**: cuánto se mueve el pronóstico de una posición a lo largo del tiempo, comparado con cuánto se mueve el observable. Nuestro CV es otra cosa: la dispersión **transversal entre las componentes del vector en un mismo instante**, promediada después sobre instantes. La ley de varianza total se aplica componente por componente y por eso la dirección es la esperable, pero **no implica** el resultado transversal: un pronóstico podría tener un patrón posicional fuerte que le infle la dispersión transversal aun siendo temporalmente plano.
>
> Así que las **36 de 36 celdas son un resultado empírico, no un corolario.** Eso es precisamente lo que las hace reportables: es la cantidad que le importa al operador —qué tan desparejo se ve el corredor *ahora*— y es la que ningún teorema existente cubre. Las versiones anteriores de esta sección presentaban la identidad como si cubriera el CV transversal. No lo hace, y la distinción se declara acá para que un revisor no tenga que encontrarla.

Es el mismo fenómeno que en meteorología incentiva el suavizado a través del problema de la *doble penalización*, y que en *downscaling* climático se corrige explícitamente con inflación de varianza — un debate, además, que sigue abierto entre reescalar la varianza e inyectar ruido. La consecuencia práctica es la sección que sigue.

### 5.3 El umbral fijo no mide al modelo, se mide a sí mismo

Marcamos como *bunching* toda celda cuyo *headway* cae por debajo de la mitad de la media de su propio vector. El umbral es relativo al estado del corredor, no un valor absoluto en minutos, y para una predicción se calcula contra la media del **vector predicho** — un operador no tiene acceso a la media real. Ocurre en el **17 % al 30 %** de las celdas.

Con ese corte, la persistencia gana las 12 celdas por márgenes que crecen con el horizonte:

| Corredor | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| E2 | 2.8× | 10.9× | 35.6× | **253.4×** |
| E4 | 1.5× | 2.8× | 5.8× | 17.7× |
| E59 | 2.0× | 3.6× | 4.9× | 8.8× |

*(Cuántas veces mejor es el F1 de la persistencia que el del LSTM, con el corte de 0.5×.)*

**Tres hechos vacían esa tabla de contenido**, y los tres salen del mismo CSV que la produjo.

**Uno. El corte de 0.5× *es* el óptimo de la persistencia.** Ajustándolo libremente por MCC[^mcc] sobre una ventana anterior y disjunta —`r2`, con un modelo entrenado por separado—, la persistencia vuelve a **0.5× en 11 de las 12 celdas** (rango 0.46×–0.60×; la excepción es E2 h=10). El LSTM aterriza entre **0.58× y 0.91×**, siempre más laxo, porque su vector está comprimido. La regla publicada estaba escrita en las unidades de uno de los dos competidores.

**Dos. El ganador declarado pierde contra una regla constante.** Marcar *todas* las celdas da F1 = 2*b*/(1+*b*), y ese piso supera al F1 de la persistencia en **5 de las 12 celdas — incluidas las tres de h=10**:

| Celda | F1 persistencia | F1 de "marcar todo" | ¿La persistencia supera el piso? |
|---|---|---|---|
| E2 h=10 | 0.332 | **0.465** | no |
| E4 h=10 | 0.268 | **0.304** | no |
| E59 h=10 | 0.303 | **0.344** | no |
| E2 h=1 | **0.581** | 0.460 | sí |

El MCC de "marcar todo" es **0 por convención**, y conviene ser preciso porque la formulación descuidada es atacable: con esa regla FN = TN = 0, así que el numerador *y* el denominador del MCC valen cero y el cociente queda indeterminado. Cero es el valor de la extensión por continuidad, el que adopta la convención estándar para matrices de confusión degeneradas, y coincide con el valor esperado del MCC para un clasificador al azar. El F1, en cambio, vale 2*b*/(1+*b*) > 0 para esa misma regla. Una métrica que pone una regla sin contenido por encima de los dos modelos no puede ser la métrica que decida cuál de los dos detecta *bunching*. **El F1 era el resumen equivocado**, y lo era porque ignora los verdaderos negativos: premia disparar a la frecuencia correcta, no acertar.

**Tres. Cuando el aprendiz habla, acierta más.** En E2 h=10 el LSTM dispara 14 veces y acierta 10 — **71 % de precisión** contra una tasa base del 30 %. La persistencia dispara 15 084 veces y acierta 5 036: **33 %**, tres puntos por encima del azar. El *recall* del LSTM colapsa; su precisión, no. **El modelo no se equivoca: está callado.** Eso es la firma de un corte mal puesto, no de información ausente.

> **Por qué el ajuste se hace por MCC y no por F1.** En E2 la tasa base es del 30 %, así que "marcar todo" ya saca F1 = 0.46 y el corte que maximiza F1 **colapsa a esa regla para los dos modelos** (dispara el 99.99 % de las veces en la persistencia, el 97.6 % en el LSTM). Un umbral sin contenido discriminativo que igual reporta un F1 presentable. El MCC vale 0 para esa regla, así que maximizarlo no puede elegirla. Las dos variantes quedan en el CSV (`fire_rate_f1fit` contra `fire_rate_calibrated`) para que la elección sea auditable y no una afirmación.

### 5.4 Sin umbral, el veredicto se da vuelta

Dos instrumentos que un corte no puede mover: el **AUC**[^auc] y la **precisión media**, ambos invariantes a cualquier reescalado monótono del puntaje — exactamente la transformación a la que un corte relativo fijo *no* es invariante. Más un tercero que sí usa un corte, pero calibrado fuera de muestra sobre `r2` y aplicado hacia adelante a `main`, que es la única dirección en que un operador podría calibrar.

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

Cuatro cosas que esta tabla establece:

- **El aprendiz no es ciego en ninguna celda.** Su AUC va de 0.565 a 0.811 y su *ap_lift* de 1.19 a 3.16. El azar es 0.5 y 1.0. Un modelo sin información sobre el evento no puede dar esos números.
- **A h=10 el LSTM gana la detección en los tres corredores**, con y sin umbral, exactamente donde la tabla anterior le daba 253× en contra.
- **A h=1 la persistencia gana la detección en los tres corredores** — y también gana el MAE ahí. Las dos métricas **coinciden**.
- **En el agregado: 6 de las 12** celdas van al LSTM por AUC y 5 por MCC calibrado, contra 0 de 12 con el corte fijo. Un veredicto que pasa de unánime a repartido según el punto de operación es, por definición, un veredicto sobre el punto de operación.

**Lo que esto retira.** La afirmación de que "alargar el horizonte mejora la ventaja escalar y destruye la fidelidad vectorial" era mitad cierta. La primera mitad se sostiene. La segunda confundía la fidelidad de *forma* del vector —que sí se destruye, Sección 5.2, 36 de 36 celdas— con la capacidad de *discriminar el evento*, que no se destruye: se reordena, y a favor del aprendiz en el horizonte largo.

**Lo que esto deja en pie, y es más útil.** Una regla de evento calibrada sobre observaciones no es transportable a un pronóstico puntual, y trasplantarla fabrica una degradación aparente de hasta 253× que no existe en la información. Eso es un resultado sobre cómo se evalúa, no sobre qué modelo gana — y tiene un arreglo de un solo escalar.

### 5.5 Nada de esto es de febrero

La Sección 4 mostró que el resultado **escalar** aguanta en tres ventanas. Las secciones 5.2 a 5.4 se midieron en una, y una afirmación de una sola ventana es exactamente lo que un revisor ataca primero. Así que las recalculamos en los tres orígenes, sobre residuos que ya estaban en disco — sin GPU y sin volver a entrenar, porque la exportación ya traía todo lo que hacía falta. Esto vale tanto para el artefacto como para su corrección.

**El aplanamiento: 36 de 36.** El sesgo del coeficiente de variación es negativo en las 36 combinaciones de corredor × horizonte × origen. El LSTM predice un corredor más regular que el real en todas, siempre, y el sesgo es notablemente estable entre ventanas (en E2 h=10: −0.664, −0.647, −0.626). La persistencia se mantiene en un sesgo de a lo sumo **0.022** en valor absoluto. Esta es la parte del hallazgo que no depende de ninguna elección de umbral, y es la que sobrevive entera.

**El artefacto: 12 de 12, y de tamaño arbitrario.** Con el corte fijo la persistencia gana las 36 celdas. Pero el *tamaño* de la ventaja se mueve entre ventanas de una forma que ninguna propiedad del modelo explicaría:

| Celda | `r1` | `r2` | `main` |
|---|---|---|---|
| E2 h=5 | 125.9× | 57.6× | 35.6× |
| **E2 h=10** | **2299×** | **817×** | **253×** |
| E4 h=10 | 21.3× | 46.4× | 17.7× |
| E59 h=10 | 11.1× | 9.9× | 8.8× |

*(Cuántas veces mejor es el F1 de la persistencia que el del LSTM, con el corte de 0.5×.)*

Un factor que va de 253 a 2299 según el mes en la misma celda no es la medida de una capacidad: es la medida de cuán lejos cayó el corte en la cola del pronóstico esa ventana en particular. La divergencia se agranda justo donde el denominador se hace chico, que es la firma aritmética de una razón sin sentido. **En 15 de las 36 celdas la persistencia ni siquiera supera al detector trivial.**

**La corrección: 11 de 12, y unánime donde importa.** El mismo cálculo sin umbral, en los tres orígenes:

| Celda | AUC `r1` | AUC `r2` | AUC `main` | ¿Coinciden? |
|---|---|---|---|---|
| E2 h=1 | persist. | persist. | persist. | sí |
| E2 h=3 | **LSTM** | **LSTM** | **LSTM** | sí |
| E2 h=10 | **LSTM** | **LSTM** | **LSTM** | sí |
| E4 h=5 | persist. | LSTM | persist. | **no** |
| E4 h=10 | **LSTM** | **LSTM** | **LSTM** | sí |
| E59 h=10 | **LSTM** | **LSTM** | **LSTM** | sí |

- **A h=10 el LSTM gana el AUC en los tres corredores y en los tres orígenes.** Nueve de nueve. La inversión del veredicto no es de febrero.
- **A h=1 la persistencia gana en los tres corredores y en los tres orígenes.** También nueve de nueve. El cruce es real en las dos direcciones.
- **La única celda que se parte es E4 h=5**, y es la esperable: en `main` los dos AUC son 0.6476 contra 0.6486 — **una milésima**. Esa celda está sobre el cruce y no hay victoria que reclamar, igual que E4 h=3 en la Sección 4. Un desacuerdo ahí confirma el mecanismo en vez de contradecirlo.

*Advertencia de lectura, la misma que la Sección 4.* Esta tabla puntúa sobre la población completa del LSTM, no sobre la intersección con el XGBoost, porque el XGBoost no se re-corrió en los orígenes de rolling. Se eligió comparabilidad **entre** ventanas.

**Lo que esto cierra.** Las tres piezas están medidas en tres ventanas: el resultado escalar (Sección 4), el aplanamiento (36/36) y el cruce de detección sin umbral (11/12, 9/9 en los extremos). Lo que sigue apoyado en una sola ventana es el XGBoost, que no se re-corrió, y el umbral calibrado fuera de muestra de la Sección 5.4, que por construcción necesita dos ventanas y usa `r2` → `main`.

### 5.6 Tampoco es de nuestro umbral

La Sección 5.5 mostró que nada de esto es de febrero. Queda la otra objeción, y es la más fuerte que se le puede hacer a este documento: **el umbral relativo a la media del propio vector lo introdujimos nosotros** (§1). Si el colapso fuera un artefacto de esa forma auto-referencial, el hallazgo no diría nada sobre la práctica del campo, que usa una fracción del *headway* **programado**.

Así que lo medimos con un corte **absoluto en minutos**, calibrado en `r2` y aplicado a `main`, idéntico para el observado y para el pronóstico:

```
K = ρ × mediana(headway observado en r2),  por (corredor, dirección)
```

con ρ = 0.5 para quedar comparable con nuestra regla, y ρ = 0.25 para igualar la convención dominante del campo. Un corte absoluto **no es auto-referencial**: su denominador no se mueve con el pronóstico.

**Cuánto sub-dispara el LSTM** (1.0 = dispara tan seguido como ocurre el evento; mediana de las 12 celdas):

| Regla | Sub-disparo | Peor celda |
|---|---|---|
| Auto-referencial, 0.5× la media del vector | 0.079 | — |
| **Absoluto, 0.5× la mediana de `r2`** | **0.040** | 0.00028 |
| **Absoluto, 0.25× la mediana de `r2`** (convención del campo) | **0.0007** | 0.000000 |

**El resultado va en contra de lo que esperábamos, y refuerza el argumento.** El colapso no se atenúa con un corte absoluto: **empeora**. Con la convención del campo es **110 veces peor** que con la nuestra. La razón es geométrica: un corte absoluto en 1.4–2.4 minutos vive en la cola lejana, y es exactamente ahí donde la compresión muerde más fuerte; nuestra regla auto-referencial al menos mueve su denominador con el nivel del vector, así que algo agarra.

> **Lo que esto cierra.** La objeción "el umbral es invención suya, así que el hallazgo no aplica al campo" queda no solo respondida sino **invertida**: de haber usado la convención dominante, el colapso aparente habría sido dos órdenes de magnitud mayor. El alcance del hallazgo es más ancho que umbrales auto-referenciales, y la afirmación de la §1 sobre ese alcance queda corregida hacia arriba.

**Y una salvedad que corre en contra, y hay que decirla.** El aprendiz carga **menos** información sobre el evento absoluto que sobre el relativo. Con ρ = 0.25 el AUC mediano baja a **0.599** (contra 0.63–0.81 del evento relativo) y en E2 h=10 llega a **0.4934** — indistinguible del azar. Así que la afirmación "el aprendiz no es ciego en ninguna celda" **se sostiene para el evento relativo y no se sostiene para el absoluto en esa celda**. Con ρ = 0.5 el cuadro es mejor: mediana 0.655, mínimo 0.518, una sola celda en o por debajo de 0.55.

### 5.7 Por qué el umbral se ajusta por MCC, medido en vez de citado

La §5.3 justifica calibrar por MCC con un teorema: Lipton et al. (2014) prueban que maximizar F1 sobre un clasificador sin información degenera a "marcar todo". Lo que nadie publicó es la comparación **empírica** de estabilidad entre los dos objetivos. Tenemos tres ventanas disjuntas y los dos objetivos implementados, así que la medimos.

| | Objetivo MCC | Objetivo F1 |
|---|---|---|
| Rango del corte entre los tres orígenes, mediana | 0.0357 | **0.0226** |
| Rango del corte, peor celda | **0.864** | 3.688 |
| Celdas con rango > 0.5 | **1 de 24** | 4 de 24 |
| MCC logrado en `main` según la ventana de calibración, mediana del rango | **0.00071** | 0.00242 |
| Ídem, peor celda | **0.018** | 0.098 |

**El resultado es mixto y conviene no maquillarlo.** En la **mediana**, el corte ajustado por F1 es *más* estable, no menos. Lo que distingue al MCC son las **colas**: el F1 tiene cuatro celdas con rango mayor a 0.5 y tres por encima de 1.0 —E2 h=3, E2 h=5 y E59 h=10, todas de persistencia, o sea los colapsos degenerados de la §5.3—, mientras que al MCC le pasa en una sola.

Y lo que decide es la última fila: **cuánto se mueve el desempeño realmente desplegado según qué ventana te tocó calibrar.** Ahí el MCC es **3.4× más estable en la mediana y 5.6× en el peor caso**. Un operador no elige un umbral, elige un procedimiento; el procedimiento por F1 funciona casi siempre y falla catastróficamente a veces, el de MCC es apenas más laxo y no tiene ese modo de falla.

La formulación defendible, entonces, no es "el MCC es más estable" —sería falso en la mediana— sino: **el ajuste por F1 tiene un modo de falla degenerado que el de MCC no tiene, y el costo fuera de muestra favorece al MCC por un factor de 3 a 6.**

---

## 6. Qué sobrevive a los ataques obvios

### El techo de winsorización no sostiene nada

El contrato de entrenamiento recorta los *headways* en el percentil 99 de train. La objeción evidente: el 1 % recortado es la cola extrema, justo donde se juega el argumento. Repuntuamos todo contra objetivos **crudos** recuperados del parquet, y con una persistencia también recalculada sin techo, que es la competidora justa.

- Recorta entre **0.78 % y 1.11 %** de los objetivos.
- **Ningún signo cambia** en las 12 celdas; el margen se mueve **menos de 0.01 min** en todas.
- E2 h=1 y E4 h=3 siguen sin ser significativas (*p* = 0.078 y 0.170).
- El F1 de *bunching* cambia **menos de 0.005**.

Hay una razón para esto último y conviene decirla: **lo que el techo recorta es la cola alta, o sea huecos de servicio, no *bunching*.** El *bunching* es un *headway* que colapsa hacia cero y ningún techo puede tocarlo.

> **Un hallazgo lateral, y una corrección de mecanismo.** El techo **ayuda** a la persistencia en vez de perjudicarla: propaga la última observación, y recortar un extremo de 35 min a 28.5 acerca esa predicción al grueso de los objetivos, así que **baja** el MAE. Es contracción por preprocesamiento en vez de por función de pérdida, y el efecto sobre el error agregado es el mismo. **Pero no es cierto que "el MAE premia contraer"**, y versiones anteriores de este documento lo afirmaban: el pronóstico que minimiza el MAE es la mediana condicional, no un valor contraído hacia el promedio global. La prueba está en nuestros propios datos: a h=1 el LSTM aplanado **pierde** el MAE contra la persistencia en los tres corredores (*p* = 0.062 / 1e−13 / 2.6e−13) y **gana** el error cuadrático (*p* = 7.3e−18 / 2.7e−14 / 8.4e−16). Si el MAE premiara contraer, ese par de signos sería imposible. Lo que aplana el vector no es la elección de MAE sobre RMSE: es emitir un solo número por celda, que es siempre una medida de centro.

### El enrutador ex-ante: dos párrafos, que es lo que amerita

Si cada modelo domina un régimen distinto, una política que conmute entre ellos usando la volatilidad de la ventana —conocida al predecir— debería ganar algo. Lo construimos con tres candidatos, aprendiendo la política sobre los primeros 13 días de servicio del test y puntuándola sobre los 9 restantes, que es el único corte que imita el despliegue. Un barrido de 20 semillas mide cuánto se mueve la ganancia con la partición.

Supera el ruido de partición en **2 de 12 celdas, ambas a h=3** (E4 −0.073 min, E59 −0.042 min), y en ninguna a h≥5, donde el aprendiz ya domina los tres terciles y no queda nada que conmutar. **7 de 12 políticas son degeneradas** — eligen el mismo modelo en los tres terciles, o sea que el "enrutador" **es** un modelo puro disfrazado. Y en E2 h=1 la política ayuda bajo partición aleatoria (−0.028 de mediana) pero **perjudica** bajo corte temporal (+0.037): no generaliza hacia adelante en el tiempo, que es la única dirección que importa. La conclusión es acotada y honesta: **conmutar paga solo donde ningún modelo puro domina**, o sea en la zona de transición, y ahí paga unos pocos segundos de MAE. Vale como demostración de ejecutabilidad, no como contribución.

---

## 7. Conclusión

> **Un pronóstico puntual predice un corredor más regular que el real, siempre, en las 36 celdas medidas. Eso no lo vuelve ciego al *bunching*: le cambia las unidades. Una regla de alarma calibrada sobre observaciones y trasplantada a ese pronóstico fabrica una degradación aparente de hasta 253× que no existe en la información — y el arreglo es un escalar.**

Cuatro afirmaciones sostenidas por la evidencia de este documento:

1. **El cruce existe, no es del Deep Learning, y su frontera real es la volatilidad.** El XGBoost lo reproduce entero. Dentro de cada horizonte, la ventaja del aprendiz crece de forma ordenada del tercil calmo al volátil, en 11 de las 12 celdas y con un empate en la doceava.

2. **El aplanamiento es real, universal y estructural.** El sesgo del coeficiente de variación es negativo en las 36 celdas de corredor × horizonte × ventana y empeora monotónicamente con el horizonte. No es un vicio del MAE —a h=1 el MAE es justamente la métrica que **castiga** al vector aplanado, mientras el error cuadrático lo premia— sino la consecuencia de emitir un solo número por celda, que es siempre una medida de centro.

3. **El costo del aplanamiento es de unidades, no de información.** Con el corte relativo fijo la persistencia gana la detección en las 36 celdas por factores de hasta 2299×, y en 15 de esas 36 ni siquiera supera a marcar todas las celdas. Sin umbral, el veredicto se da vuelta: a h=10 el LSTM discrimina mejor en los tres corredores y en los tres orígenes, y a h=1 la persistencia gana en los tres y en los tres. El AUC del aprendiz nunca baja de 0.565 — nada cerca del azar.

4. **Por lo tanto: la calibración del umbral, no el modelo, decide quién parece ver el evento.** Un veredicto que pasa de unánime a repartido según el punto de operación es un veredicto sobre el punto de operación. Cualquier trabajo que evalúe detección de eventos trasplantando un corte relativo sobre un pronóstico puntual está midiendo su propio corte.

Lo que este trabajo **no** afirma: que estos modelos estén listos para operar una alarma de *bunching*. Un AUC de 0.60 es información real y muy lejos de un sistema de despacho; falta la función de costo que traduzca eso a una decisión (limitación 8). Lo que sí afirma es que la línea de trabajo **no está cerrada**, y la versión anterior de este documento la cerraba por un artefacto de medición.

**Y una retractación explícita.** Este documento sostuvo que "la métrica decide el ganador" en el sentido de que el escalar y el vector nombran ganadores opuestos. Eso era falso: nombran el mismo ganador una vez removido el artefacto. También sostuvo que "el MAE premia contraer", que contradice nuestros propios datos a h=1. Las dos afirmaciones se retiran, con las mediciones que las desmienten al lado.

---

## 8. Alcance y limitaciones

**Lo que está establecido y contra qué.** El cruce está establecido contra la persistencia y replicado por el XGBoost. La comparación **LSTM contra XGBoost no está nivelada** (24 configuraciones contra 1 o 3): donde el LSTM gana, gana con menos presupuesto y la conclusión es segura; donde pierde, no es atribuible a la clase de modelo.

**Limitaciones reales.**

1. **Alcance geográfico y temporal.** Tres corredores de una ciudad, una ventana de 5 meses. E4 aporta validez externa de escala de flota, no geográfica.
2. **La estabilidad temporal está medida para el LSTM y la persistencia, no para el XGBoost.** El resultado escalar se confirmó en tres ventanas (Sección 4), el aplanamiento en las 36 celdas y el cruce de detección sin umbral en 11 de 12 (Sección 5.5). Lo que **no** se re-corrió en los orígenes anteriores es el XGBoost, así que todo lo que involucra al árbol —la réplica del cruce, su colapso vectorial, el contraste LSTM contra XGBoost— sigue apoyado en **una sola** ventana de 22 días. Cerrarlo sí requiere GPU y Kaggle, a diferencia de todo lo vectorial, que se recalculó sobre bytes que ya estaban en disco.
3. **Confusor en el período de prueba.** Febrero 2024 en Arequipa incluye Carnaval (12–13 feb). La composición del test no está caracterizada.
4. **Cobertura de semillas.** Solo el LSTM tiene barrido de semillas, y sobre las familias congeladas. ConvLSTM y Transformer no lo tienen.
5. **El nulo espacial es previo.** Se estableció sobre las familias congeladas, que arrastran el sesgo de encuadre. No se rehízo bajo el pipeline contiguo.
6. **La política del enrutador se calibra sobre una porción del test**, no sobre train+val, porque los kernels solo exportaron predicciones del split de prueba. Política y evaluación son disjuntas, así que la ganancia no está contaminada, pero los niveles de MAE del enrutador no son comparables con los del test completo.
7. **Sin estratificar por magnitud del *headway*.** Un error de 1 min sobre un *headway* de 3 min y sobre uno de 15 min no pesan igual, y esa heterogeneidad queda en el promedio.
8. **Valor operativo argumentado, no modelado.** No hay función de costo que muestre que 1.47 min de MAE, o un AUC de detección de 0.60, cambien una decisión concreta de despacho. Es la limitación que más pesa sobre la lectura optimista de la Sección 5.4: mostramos que la información está ahí, no que alcance para operar.
9. **El umbral de *bunching* es nuestro, no heredado, y no está calibrado contra incidentes registrados.** La forma relativa que domina la literatura es una fracción del *headway* **programado**; la nuestra normaliza por la media del propio vector porque no tenemos horario. Eso acota el alcance del hallazgo a **umbrales relativos y auto-referenciales** — un corte en minutos absolutos sub-dispararía también, pero por otro mecanismo, y no lo medimos. El documento depende de esa elección menos que antes, no más: los veredictos se apoyan en el AUC y la precisión media, que **no usan umbral**, y el corte calibrado se ajusta fuera de muestra. Lo que **no** se puede sostener es la lectura anterior: los factores de F1 (253×, 2299×) son artefactos de esa elección y se reportan como tales.
10. **La calibración fuera de muestra usa dos ventanas, no validación cruzada.** El corte se ajusta en `r2` y se aplica a `main`. Son disjuntas y `r2` es anterior, que es la dirección correcta, pero los conjuntos de **entrenamiento** de los dos modelos están anidados (Sección 4), así que no son independientes en sentido estricto. Un esquema de *k* ventanas rotativas sería más fuerte y no se hizo.
11. **Un desajuste de ancho de vector, declarado.** El LSTM se dimensiona con un `max_N` global por corredor y el XGBoost con el de cada dirección, así que la red predice unas pocas posiciones de cola que el XGBoost no emite. Afecta al 0.05 % de las filas en el peor caso, quedan fuera de la intersección y de todo verdicto, y el sesgo de encuadre medido (0.001 min) confirma que no mueven nada.

**Trazabilidad.** Las cuatro figuras de este documento se generan desde los CSV versionados con `uv run python -m src.build_contiguous_figures`, no desde los residuos crudos: así una figura no puede discrepar de la tabla que ilustra. La figura `contiguo-disociacion.png` de versiones anteriores **fue eliminada**: graficaba el F1 con umbral fijo como si midiera a los modelos, o sea el artefacto que la Sección 5.3 desarma. Sus sucesoras son `contiguo-artefacto-umbral.png` y `contiguo-deteccion-sin-umbral.png`, y solo funcionan como par. Las figuras `curva-degradacion.png` y `volatilidad-crossover.png` que quedan en este directorio corresponden a las **familias congeladas** y no a este pipeline; se conservan solo como registro de esa comparación.

---

## Glosario de términos técnicos

[^headway]: **Headway** — el intervalo de tiempo entre el paso de un bus y el siguiente en un mismo corredor. Es la variable que predecimos, en minutos. Un *headway* estable significa buses espaciados regularmente; uno irregular indica *bunching* o huecos en el servicio.

[^bunching]: **Bunching** — fenómeno en que dos o más buses que deberían ir espaciados terminan circulando casi juntos, dejando un hueco largo detrás. Es el principal síntoma de un servicio desestabilizado, y es una anomalía **del patrón colectivo**: cada bus por separado puede estar donde corresponde.

[^persistencia]: **Persistencia (modelo naive)** — predice que el valor futuro será igual al último observado. En series temporales cortas es difícil de superar, por eso es el rival serio. Tiene además una propiedad que este documento explota: al copiar el vector observado conserva su forma y su dispersión, así que una regla de evento definida sobre observaciones se le aplica sin traducción. Eso la vuelve el patrón de referencia natural para medir el artefacto de umbral — y no, como sostenían versiones anteriores, el mejor detector de *bunching*: a h=10 discrimina peor que el LSTM en los tres corredores.

[^lstm]: **LSTM (Long Short-Term Memory)** — red neuronal recurrente diseñada para aprender de secuencias, capaz de retener información relevante a lo largo del tiempo.

[^xgboost]: **XGBoost** — biblioteca de *gradient boosting*: construye árboles de decisión donde cada uno corrige el error del anterior. Acá es el competidor aprendido, nivelado con la misma ventana de entrada que la red.

[^mae]: **MAE (Error Absoluto Medio)** — promedio de la diferencia absoluta entre lo predicho y lo real, en minutos. Trata todos los errores por igual. El pronóstico que lo minimiza es la **mediana** condicional; el que minimiza el error cuadrático es la **media** condicional. Las dos son medidas de centro, así que la sub-dispersión del pronóstico no es un vicio del MAE en particular: es propiedad de reportar un solo número por celda. En estos datos, además, el MAE es la métrica que **castiga** al vector aplanado a h=1, mientras el error cuadrático lo premia (ver §3).

[^rmse]: **RMSE (Raíz del Error Cuadrático Medio)** — como el MAE pero elevando los errores al cuadrado antes de promediar, así que penaliza más los errores grandes. Por eso favorece al pronóstico contraído cuando la alternativa arriesga: a h=1 el LSTM aplanado **pierde** el MAE y **gana** el error cuadrático contra la persistencia, en los tres corredores.

[^auc]: **AUC (área bajo la curva ROC)** — probabilidad de que el modelo le asigne más "riesgo de *bunching*" a una celda donde el evento realmente ocurrió que a una donde no. 0.5 es azar, 1.0 es perfecto. Su propiedad clave acá es que **no usa umbral** y es invariante a cualquier reescalado monótono del puntaje, así que comprimir un pronóstico hacia su media no puede moverla — a diferencia de un corte relativo fijo, que se rompe.

[^mcc]: **MCC (coeficiente de correlación de Matthews)** — resume una matriz de confusión en un número de −1 a 1 usando las cuatro celdas, incluidos los **verdaderos negativos** que el F1 ignora. Para la regla degenerada "marcar todo" vale **0 por convención** (el cociente es 0/0; cero es la extensión por continuidad y el valor esperado de un clasificador al azar), mientras que el F1 de esa misma regla es 2*b*/(1+*b*), o sea 0.30 a 0.46 en estos corredores. Por eso acá reemplaza al F1 como resumen y como objetivo de calibración.

[^cv]: **Coeficiente de variación (CV)** — desviación estándar dividida por la media. En castellano: **un número que dice cuán desparejo está el corredor**. Cero significa buses perfectamente espaciados; alto significa que hay huecos largos y buses pegados. Dos corredores con el mismo *headway* promedio de 10 min pueden tener CV de 0.07 (huecos de 9-10-11-10) o de 0.85 (huecos de 1-19-2-18): el promedio no los distingue y el CV sí. Se divide por la media para que sea comparable entre corredores de distinta frecuencia. Es una propiedad del vector como un todo, no de cada *headway* por separado. El TCQSM la prescribe como medida de fiabilidad para servicio de alta frecuencia (≤10 min) y le asigna una escala de nivel de servicio; **no** es, en cambio, la métrica que usan los operadores en la práctica (ver §5.2). Y su definición en el manual normaliza por el *headway* **programado**, que nosotros no tenemos: lo nuestro es σ(*h*)/media(*h*) sobre el vector observado.

[^f1]: **F1** — media armónica entre precisión (de lo que el modelo marcó, cuánto era cierto) y *recall* (de lo que era cierto, cuánto marcó el modelo). Resume la detección en un número entre 0 y 1. Un F1 bajo con precisión alta, como el del LSTM acá, indica un modelo que acierta cuando habla pero que casi no habla. **Su defecto en este documento:** ignora los verdaderos negativos, así que premia disparar a la frecuencia del evento aunque el acierto sea de casi-azar. Con una tasa base del 30 %, "marcar todo" saca F1 = 0.46 y supera al ganador declarado (§5.3). Se reporta por continuidad con las versiones anteriores, pero los veredictos de este documento descansan en el AUC[^auc] y el MCC[^mcc].

[^split]: **División train / validación / prueba** — los datos se separan en tres bloques **temporales**: entrenamiento, validación (ajuste de hiperparámetros) y prueba (evaluación final sobre datos nunca vistos, y posteriores en el tiempo).

[^wilcoxon]: **Test de Wilcoxon (pareado, de rangos con signo)** — prueba no paramétrica que compara dos modelos por las **medianas** de sus errores. Complementa al Diebold-Mariano, que compara medias, y acá los dos se contradicen a h=3 — que es precisamente el hallazgo.

[^snapshot]: **Snapshot** — una "foto" del estado de todos los buses del corredor en un mismo instante: el vector de *headways* completo en ese momento.
