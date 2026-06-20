# Deep Learning para el pronóstico de *headway*[^headway] de buses: ventaja condicionada al horizonte y a la volatilidad del servicio

**Documento de resultados** · Corredores E2 y E59 · Horizontes de 1 a 10 minutos

> Este documento presenta **qué encontramos**, no cómo se limpiaron o normalizaron los datos. El preprocesamiento se resume al mínimo necesario para que los resultados sean reproducibles; el foco está en la evidencia.

---

## 1. La pregunta

> **¿Cuándo vale la pena usar un modelo de Deep Learning[^dl] para predecir el *headway* de buses, en lugar de un método estadístico clásico?**

La respuesta corta —y el aporte de este trabajo— es que **no siempre conviene**. El Deep Learning (DL) gana, pero solo bajo condiciones concretas: cuando se predice con suficiente anticipación (horizonte[^horizonte] ≥ 3 min) y cuando el servicio está **desestabilizándose**. En servicio estable, un método clásico simple es igual de bueno o mejor.

Este documento demuestra esa afirmación en tres pasos: **(1)** el DL gana → **(2)** la diferencia es estadísticamente real → **(3)** explicamos *por qué* gana.

---

## 2. La comparativa: quién compite contra quién

Enfrentamos dos familias de modelos sobre exactamente los mismos datos y la misma métrica.

### Modelos estadísticos clásicos (*baselines*[^baseline])

| ID | Modelo | Qué hace |
|----|--------|----------|
| B0 | Media global | Promedio constante del *headway* histórico |
| **B1** | **Persistencia[^persistencia]** | **Repite el último valor observado — el rival a batir** |
| B2 | Media móvil | Promedio de las últimas *w* observaciones (*w* = 5, 10, 15) |
| B3 | Suavizado exponencial | Promedio que pondera más lo reciente (α = 0.3) |
| B4 | Media histórica horaria | Promedio típico para esa hora del día |

### Baseline ajustado (*fitted ML*)

| ID | Modelo | Qué hace |
|----|--------|----------|
| **B5_XGB** | **Gradient boosting (XGBoost[^xgboost])** | Modelo **entrenado** que ve la **misma ventana de 12 pasos** que el LSTM (su primer *lag* es exactamente la persistencia) más hora, día y dirección — el competidor *aprendido* a batir |

> **Por qué sumamos un baseline ajustado.** Ganarle solo a fórmulas naive (B0–B4) podría descartarse como "vencer rivales débiles". B5_XGB es un aprendiz de verdad: se entrena, se ajusta sobre validación y recibe **exactamente la misma información que la red neuronal**. Si el DL le gana también a esto, la ventaja deja de ser un artefacto de *baselines* pobres.

### Modelos de Deep Learning

| Modelo | Idea de la arquitectura |
|--------|-------------------------|
| **LSTM**[^lstm] | Red recurrente que aprende patrones en la secuencia temporal de *headways* |
| **SpatialConvLSTM**[^convlstm] | Añade una convolución que mira la relación entre buses cercanos antes del LSTM |
| **SpatialTransformer**[^transformer] | Añade auto-atención espacial entre buses antes del LSTM |

### El terreno de juego

- **2 corredores:** E2 y E59 (dos líneas de bus reales).
- **4 horizontes:** predecir a 1, 3, 5 y 10 minutos vista.
- **2 métricas:** MAE[^mae] y RMSE[^rmse], ambas en minutos (cuanto más bajas, mejor). El texto reporta MAE; el RMSE confirma el mismo patrón y se ve en la Figura 1.
- **Datos:** 5 meses contiguos (2023-10-01 → 2024-02-29, 152 días) divididos temporalmente[^split]:
  - **Entrenamiento:** 2023-10-01 → 2024-01-15 (107 días, ≈3.5 meses) — el modelo aprende.
  - **Validación:** 2024-01-16 → 2024-02-07 (23 días, ≈3 semanas) — se ajustan los hiperparámetros[^hiperparametros].
  - **Prueba:** 2024-02-08 → 2024-02-29 (22 días, ≈3 semanas) — evaluación final; **todo lo reportado es sobre este conjunto, que el modelo nunca vio.**

> **Por qué la persistencia (B1) es el rival serio.** Es tentador comparar el DL contra un promedio tonto y "ganar" fácil. Pero en series temporales cortas, *repetir el último valor* es sorprendentemente difícil de superar: si el bus venía con 4 minutos de *headway*, lo más probable es que dentro de 1 minuto siga cerca de 4. Batir a la persistencia es la verdadera prueba. Por eso B1 es la referencia central de las comparaciones; los demás *baselines* (B0, B2–B4) se incluyen como contexto en la Figura 1.

---

## 3. Resultado central: la curva de degradación

![Curva de degradación](curva-degradacion.png)

*Figura 1 — MAE y RMSE frente al horizonte de predicción, para E2 y E59. Cuanto más bajo, mejor. Se grafican la persistencia (B1), el mejor baseline formulaico (B3), el baseline ajustado XGBoost (B5) y los tres modelos profundos. El símbolo ⊘ marca comparaciones no significativas.*

**El mensaje:** a 1 minuto la persistencia es imbatible pero **operativamente inútil** (nadie puede reaccionar con 1 minuto de aviso). El valor del DL **emerge al anticipar a 3, 5 y 10 minutos** — justo el margen que un operador necesita para intervenir.

Mirá el cruce (*crossover*[^crossover]) en el extremo izquierdo de cada panel (✓ = gana el DL; ❌ = gana la persistencia):

| Corredor | h = 1 min | h = 3 min | h = 5 min | h = 10 min |
|----------|-----------|-----------|-----------|------------|
| **E59** (MAE) | B1 **3.10** vs LSTM 3.34 ❌ *gana persistencia* | 4.18 vs **3.85** ✓ | 4.70 vs **4.03** ✓ | 5.59 vs **4.23** ✓ |
| **E2** (MAE) | B1 4.76 vs **4.47** ✓ | 6.08 vs **4.94** ✓ | 6.49 vs **5.05** ✓ | 7.03 vs **5.15** ✓ |

**Lo más importante — la brecha crece con el horizonte.** A medida que predecimos más lejos, la persistencia se degrada rápido y el DL aguanta:

- **E2 a 10 min:** persistencia 7.026 vs LSTM **5.153** → **−1.87 min de error (−26.7 %)**.
- **E59 a 10 min:** persistencia 5.593 vs LSTM **4.225** → **−1.37 min de error (−24.5 %)**.

> Los valores de la tabla anterior están redondeados a 2 decimales; los porcentajes se calculan sobre los valores con 3 decimales para que la aritmética sea exacta.

> **Nota honesta sobre los modelos espaciales.** El mejor DL terminó siendo el **LSTM plano** en ambos corredores; las variantes con convolución y atención no aportaron mejora clara. Esto se reporta tal cual: añadir complejidad espacial **no** mejoró el pronóstico en estos datos.

### El DL también le gana al competidor ajustado (no solo a los naive)

La objeción natural a "el DL le gana a la persistencia" es: *¿y si los baselines son demasiado débiles?* Por eso entrenamos **B5_XGB** (gradient boosting) con la misma ventana de 12 pasos que ve el LSTM. Es un competidor **creíble**: le gana a la persistencia y al mejor baseline formulaico en casi todas las celdas (p. ej. E59 a h=10: mejor clásico 4.81 → XGBoost **4.64**). Y aun así:

| Corredor | h = 1 | h = 3 | h = 5 | h = 10 |
|----------|-------|-------|-------|--------|
| **E2** — Δ MAE (XGBoost − LSTM) | +0.07 | +0.08 | +0.09 | +0.09 |
| **E59** — Δ MAE (XGBoost − LSTM) | +0.05 | +0.19 | +0.28 | **+0.42** |

**El LSTM le gana al XGBoost en las 8 celdas**, y —de nuevo— **la brecha crece con el horizonte** (E59: de +0.05 a +0.42 min). Es la misma curva de degradación, ahora contra un **aprendiz fuerte** en vez de una fórmula naive. La ventaja del DL no es un artefacto de comparar contra rivales pobres.

> En E59 a h=1 la persistencia (3.10) sigue siendo la mejor de todos —incluido el XGBoost (3.39)—, coherente con que a 1 minuto el pronóstico es trivial.

---

## 4. ¿Es real o es casualidad? — Significancia estadística

Que un número sea más bajo no basta: podría ser ruido. Para descartarlo aplicamos dos tests pareados a cada comparación DL-vs-persistencia.

- **Tests usados:** Diebold-Mariano[^dm] (compara el error medio) y Wilcoxon[^wilcoxon] (compara las medianas por rangos).
- **Resultado: 35 de 36 comparaciones son significativas** con p-valor[^pvalor] < 0.001. Las 36 comparaciones surgen de 3 modelos DL × 2 corredores × 3 horizontes (h = 3, 5, 10; se excluye h=1, donde la persistencia gana) × 2 métricas.

| | Resultado |
|---|---|
| Comparaciones significativas | **35 / 36** (p < 0.001) |
| Única excepción | LSTM · E59 · h=3 · Wilcoxon p = 0.277 |

> **La única excepción no contradice nada.** En ese caso, Diebold-Mariano sí detecta la ventaja (p ≈ 0) pero Wilcoxon no: el DL gana **en promedio** pero el efecto es débil en términos de mediana. Es un único caso límite a 3 minutos; en el resto de la grilla la ventaja es contundente.

**Conclusión de esta sección:** la ventaja del DL a h ≥ 3 **no es ruido**. Es un efecto sistemático y medible.

### ¿O es casualidad del hiperparámetro?

Queda un último flanco: ¿el resultado del LSTM depende de haber acertado un ajuste fino y frágil? Para descartarlo corrimos un **mini-grid de sensibilidad** a h = 10. Para cada corredor se entrenaron 4 configuraciones —la **ganadora congelada** del grid de Fase 5 más **3 vecinas**, cada una alterando **un solo** hiperparámetro (capacidad oculta, *dropout*[^dropout] o tasa de aprendizaje)— con el **mismo seed, pipeline y *splits***. La comparación es interna: no depende de números históricos.

| Corredor | Vecindario (4 configs) | MAE de la ganadora | Rango de MAE del vecindario | Dispersión |
|---|---|---|---|---|
| **E2** | hidden∈{32,64}, dropout∈{0,0.2}, lr∈{5e-4,1e-3} | 5.163 | 5.138 – 5.165 | **0.5 %** |
| **E59** | hidden∈{32,64}, dropout∈{0,0.2}, lr∈{5e-4,1e-3} | 4.225 | 4.225 – 4.264 | **0.9 %** |

*Datos: [`csv-multihorizon/lstm_minigrid_h10.csv`](csv-multihorizon/lstm_minigrid_h10.csv) (8 filas: 2 corredores × 4 configs).*

**El rendimiento es estable: mover cualquier perilla mueve el MAE menos del 1 %.** En E59 la configuración elegida es además la **mejor** del vecindario; en E2 queda en el pelotón —una vecina la supera por ~0.5 %, distancia indistinguible del ruido de re-entrenamiento—. En ningún caso el rendimiento se desploma al perturbar el ajuste. La ventaja del DL **no es un artefacto de un hiperparámetro afortunado**: es robusta a su propia configuración.

---

## 5. ¿Por qué gana el DL? — El mecanismo de la volatilidad

Esta es la parte más interesante y la que convierte un número promedio en una **historia con sentido**.

Partimos las predicciones en tres **regímenes de volatilidad**[^volatilidad] según cuánto cambia realmente el *headway*:

![Crossover de volatilidad](volatilidad-crossover.png)

*Figura 2 — Diferencia de MAE (DL − persistencia) en cada régimen de volatilidad. Por debajo de cero, gana el DL.*

| Régimen | Cambio real del *headway* | ¿Quién gana? | Δ MAE (DL − persistencia) |
|---------|-----------------------------|--------------|----------------------------|
| **Estable** | < 1 min | Persistencia | +2.4 a +3.4 (DL peor) |
| **Moderado** | 1–3 min | Persistencia (justo) | +0.85 a +1.6 |
| **Alto** | ≥ 3 min | **DL, decisivo** | **−3.2 a −3.7 (DL mejor)** |

*Los rangos de Δ MAE abarcan los 2 corredores y los 3 horizontes (h = 3, 5, 10); el patrón es idéntico en todas las celdas. Los cortes de régimen son fijos en minutos (1 y 3 min), no cuantiles.*

**El hallazgo clave:** el DL **no mejora el promedio mejorando todo un poco**. El promedio "el DL le gana" es en realidad la **suma de dos regímenes opuestos**:

- En servicio **estable**, predecir es trivial (el *headway* casi no cambia) → la persistencia gana, pero es una victoria **sin valor operativo**.
- En servicio que se **desestabiliza** —el inicio del *bunching*[^bunching]— (saltos ≥ 3 min) → el DL gana de forma decisiva, **justo cuando importa** para intervenir.

**Y esto explica por qué la brecha crece con el horizonte** (Sección 3): a mayor horizonte, más muestras caen en el régimen de alto cambio.

- En E59, las ventanas de alto cambio pasan de **38.6 %** (h=3) a **54.4 %** (h=10).
- A 10 minutos, **más de la mitad de los casos** caen en el terreno donde el DL domina.

---

## 6. Conclusión

> **El Deep Learning conviene para predecir el *headway* a horizontes operativos (≥ 3 min) y, sobre todo, en los tramos donde el servicio se está desestabilizando. En condiciones estables, un método clásico como la persistencia es igual de bueno o mejor.**

La conclusión madura **no** es "el DL reemplaza a la persistencia": cada modelo domina un régimen distinto. Esto abre la puerta a combinarlos (ver trabajo futuro).

### Alcance y limitaciones

**Análisis deliberadamente no realizados (no por carencia, por redundancia).** La comparación pico vs. valle quedó subsumida en el análisis de volatilidad (Sección 5): las franjas pico coinciden con el régimen de alto cambio del *headway*, y la volatilidad explica el *mecanismo* causal —cuánto se mueve el *headway*— mejor que la mera hora del día. Un análisis pico/valle separado mediría el mismo fenómeno con peor resolución.

**Limitaciones reales del estudio.**
- Evaluado sobre 2 corredores (E2, E59) y una ventana de 5 meses; la generalización a otras líneas o ciudades queda por validar.
- Los modelos espaciales (Conv, Transformer) no superaron al LSTM plano en estos datos; queda abierto si lo harían con más buses por *snapshot*[^snapshot].

**Trabajo futuro: sistema híbrido.** Los resultados sugieren un enrutador que use persistencia en régimen estable y DL en régimen de alta volatilidad. Construirlo, sin embargo, requiere **predecir el régimen de volatilidad por adelantado**, lo cual es un problema abierto en sí mismo: en este estudio la volatilidad se midió de forma **retrospectiva** (con el cambio real del *headway*, conocido solo a posteriori). Por ello el sistema híbrido se plantea como dirección futura y no se evaluó en operación real.

---

## Glosario de términos técnicos

[^headway]: **Headway** — el intervalo de tiempo entre el paso de un bus y el siguiente en una misma parada/corredor. Es la variable que predecimos, medida en minutos. Un *headway* estable significa buses espaciados regularmente; uno irregular indica bunching (buses amontonados) o huecos en el servicio.

[^dl]: **Deep Learning (DL)** — familia de modelos de redes neuronales con varias capas que aprenden patrones complejos directamente de los datos, sin que un humano defina las reglas. Aquí se contrasta con métodos "clásicos" que aplican una fórmula fija.

[^horizonte]: **Horizonte de predicción (h)** — cuánto tiempo hacia el futuro se predice. *h = 10 min* significa "predecir el *headway* que habrá dentro de 10 minutos". Cuanto mayor el horizonte, más difícil la predicción, pero más útil operativamente.

[^baseline]: **Baseline (línea base)** — un modelo de referencia simple contra el cual se compara el modelo nuevo. Si un modelo complejo no le gana a un *baseline* simple, no aporta valor. Son la vara de medir.

[^persistencia]: **Persistencia (modelo naive)** — el *baseline* más básico: predice que el valor futuro será igual al último valor observado ("mañana lloverá lo mismo que hoy"). En series temporales es difícil de superar a corto plazo, por eso es el rival serio.

[^lstm]: **LSTM (Long Short-Term Memory)** — un tipo de red neuronal recurrente diseñada para aprender de secuencias, capaz de "recordar" información relevante a lo largo del tiempo. Procesa la historia reciente de *headways* para anticipar el siguiente.

[^convlstm]: **SpatialConvLSTM** — variante que, antes del LSTM, aplica una convolución 1D sobre los buses cercanos. La idea es capturar relaciones *espaciales* entre buses (qué le pasa al de adelante afecta al de atrás) además de las temporales.

[^transformer]: **SpatialTransformer / auto-atención** — variante que usa un mecanismo de *atención* (el del modelo Transformer) para que cada bus "mire" a los demás y pondere cuáles son relevantes, antes de pasar la secuencia al LSTM.

[^mae]: **MAE (Error Absoluto Medio)** — promedio de la diferencia absoluta entre lo predicho y lo real, en minutos. Trata todos los errores por igual. Un MAE de 4 significa que, en promedio, la predicción se equivoca por 4 minutos.

[^rmse]: **RMSE (Raíz del Error Cuadrático Medio)** — similar al MAE pero eleva al cuadrado los errores antes de promediar, por lo que **penaliza más los errores grandes**. Si el RMSE es mucho mayor que el MAE, hay errores grandes ocasionales.

[^split]: **División train / validación / prueba** — los datos se separan en tres bloques temporales: *entrenamiento* (el modelo aprende), *validación* (se ajustan los hiperparámetros) y *prueba* (evaluación final sobre datos nunca vistos). Reportar sobre prueba evita engañarse con resultados inflados.

[^dm]: **Test de Diebold-Mariano (DM)** — prueba estadística específica para comparar la precisión de dos modelos de pronóstico. Responde: ¿la diferencia de error medio entre los dos modelos es real o producto del azar?

[^wilcoxon]: **Test de Wilcoxon (pareado, de rangos con signo)** — prueba no paramétrica que compara dos modelos mirando las *medianas* de sus errores por rangos, sin asumir que los datos siguen una distribución normal. Complementa a Diebold-Mariano.

[^pvalor]: **p-valor** — probabilidad de observar una diferencia así de grande si en realidad **no hubiera** diferencia entre los modelos. Un p-valor < 0.001 significa "menos de 1 en 1000 de que sea casualidad" → la diferencia se considera estadísticamente significativa.

[^volatilidad]: **Régimen de volatilidad** — clasificación de cada predicción según cuánto cambia realmente el *headway* en ese momento, con cortes fijos en minutos: *estable* (cambio < 1 min), *moderado* (1–3 min) y *alto* (≥ 3 min). Los cortes son fijos (no cuantiles) a propósito: así se ve que la *proporción* de muestras en régimen alto crece con el horizonte. Permite ver dónde gana cada modelo en lugar de mirar solo el promedio global.

[^crossover]: **Crossover (cruce)** — el punto donde dos curvas se cruzan e invierten quién va ganando. Aquí marca el horizonte a partir del cual el DL pasa a superar a la persistencia (h ≥ 3 min en E59).

[^hiperparametros]: **Hiperparámetros** — los ajustes de configuración de un modelo que NO se aprenden de los datos sino que se fijan antes de entrenar (p. ej. cuántas capas, tamaño de la red, tasa de aprendizaje). Se eligen probando sobre el conjunto de validación.

[^dropout]: **Dropout** — técnica de regularización que, durante el entrenamiento, "apaga" al azar una fracción de las neuronas en cada paso para evitar que la red memorice el conjunto de entrenamiento (*overfitting*). Es uno de los hiperparámetros del mini-grid de sensibilidad (Sección 4); un valor de 0.2 apaga el 20 % de las unidades, 0.0 desactiva la técnica.

[^xgboost]: **XGBoost** — biblioteca de *gradient boosting*: construye un conjunto de árboles de decisión donde cada árbol corrige el error del anterior. Es un modelo de ML *ajustado* (se entrena con datos), a diferencia de los baselines B0–B4 que son fórmulas fijas. Aquí se usa como baseline B5_XGB con la misma ventana de 12 pasos que ve el LSTM, para que la comparación sea justa.

[^snapshot]: **Snapshot** — una "foto" del estado de todos los buses del corredor en un mismo instante: el vector de *headways* de cada bus en ese momento. Los modelos espaciales miran cada *snapshot* para relacionar buses entre sí.

[^bunching]: **Bunching** — fenómeno en que dos o más buses que deberían ir espaciados terminan circulando casi juntos (uno "alcanza" al de adelante), dejando un hueco largo detrás. Es el principal síntoma de un servicio desestabilizado.
