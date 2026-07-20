<!--
  ANDAMIAJE DEL MANUSCRITO — borrador en español.
  Flujo: se escribe y pule TODO en español → traducción al inglés al final (IJACSA es en inglés).
  Las notas entre bloques ">" con la etiqueta [ANDAMIAJE] son guías para escribir; BORRARLAS antes de traducir.
  Orden de escritura sugerido: Related Work → Métodos → Resultados (ya existen) → Discusión → Conclusión → Introducción → Abstract.
  Contenido reutilizable: docs/resultados/documento-resultados.md (Secciones 3, 4, 5, 6 y amenazas a la validez).
-->

# Título

> [ANDAMIAJE] Título provisional. Debe nombrar: (1) el objeto = pronóstico de headway de buses, (2) el ángulo = ventaja del DL condicionada al horizonte y la volatilidad, (3) opcional: dato real / Arequipa. Se afina al final.
>
> Provisional: **"¿Cuándo conviene el Deep Learning para pronosticar el headway de buses? Una evaluación pareada y condicional al régimen sobre datos GPS reales de Arequipa"**

**Autores** · Afiliación · Contacto

---

## Abstract

> [ANDAMIAJE] SE ESCRIBE AL FINAL. 150–250 palabras, un solo párrafo. Estructura: (1) contexto y problema, (2) qué hicimos, (3) datos, (4) 2–3 resultados con números, (5) el aporte. No citas, no abreviaturas sin definir.

*(pendiente)*

**Keywords** — *(pendiente — 5–7: bus headway forecasting, bus bunching, deep learning, LSTM, persistence baseline, Diebold-Mariano, AVL/GPS data)*

---

## I. Introducción

> [ANDAMIAJE] SE ESCRIBE CASI AL FINAL (después de saber qué prometiste). Cuatro movimientos:

### Contexto y motivación
> [ANDAMIAJE] Qué es el headway, qué es el bunching, por qué importa operativamente (predecir a tiempo para intervenir). Material ya discutido — bajar a 1–2 párrafos.

### El problema / gap
> [ANDAMIAJE] El subcampo de headway/bunching reporta ganancias del DL SIN baseline naive y SIN tests de significancia, y no caracteriza BAJO QUÉ condiciones el DL supera a métodos simples. (No usar el marco de "sesgo de publicación" — afirmación refutada.)

### Contribuciones
> [ANDAMIAJE] Bullets. Rankeadas por defensibilidad (de la relevación de literatura):

- **C1 (rigor).** Evaluación pareada con tests de significancia (Diebold-Mariano y Wilcoxon sobre muestras idénticas) frente a un baseline de **persistencia** y a un **XGBoost nivelado** — prácticamente ausente en el subcampo de headway/bunching.
- **C2 (condicional).** Caracterización explícita de **cuándo** el DL le gana a la persistencia, en función del **horizonte** (crossover a h ≥ 3 min) y del **régimen de volatilidad** (ex-ante), con una regla de conmutación ejecutable.
- **C3 (datos).** Evidencia sobre datos AVL/GPS reales de una ciudad del **Sur Global** (Arequipa, Perú), subrepresentada en la literatura. *(Soporte, no titular.)*
- **C4 (hallazgo honesto).** Resultado nulo de la complejidad espacial (ConvLSTM/Transformer no superan al LSTM plano) **en estos datos**, y ventaja del DL sobre XGBoost dependiente de la escala del corredor.

### Estructura del artículo
> [ANDAMIAJE] Un párrafo: "La Sección II revisa... la III describe... etc."

---

## II. Trabajos Relacionados (Related Work)

> [ANDAMIAJE — BORRAR ANTES DE ENVIAR] Versión PRELIMINAR. Todas las referencias son REALES y fueron verificadas en la relevación de fuentes; cada una lleva su identificador (arXiv / DOI / PII) para comprobarla. Los campos entre corchetes `[por confirmar]` NO se pudieron verificar a texto completo: complételos contra la fuente original, no los invente. Verificar SIEMPRE cada cita contra el documento original antes de someter.

El pronóstico de corto plazo del *headway* de buses se ubica en la intersección de tres líneas de trabajo: los modelos de predicción de intervalos y tiempos de arribo en transporte público, la discusión metodológica sobre el rigor de las comparaciones en pronóstico de series de tiempo, y la evidencia —mayormente ajena al dominio del transporte— acerca de los límites del aprendizaje profundo frente a métodos simples.

### A. Modelos de pronóstico de headway y de tiempos de arribo

El *bunching* de buses se ha formalizado como la predicción de la irregularidad del *headway* a nivel de parada. Yu et al. [1] modelan la fluctuación del intervalo en las paradas siguientes a partir del *headway* histórico, la demanda de pasajeros y el tiempo de viaje, mediante una regresión por máquinas de vectores de soporte por mínimos cuadrados (LS-SVM) sobre datos de tarjeta inteligente. Este trabajo establece la formulación canónica —predecir el patrón de *headway* como vía para anticipar el *bunching*— que la presente investigación adopta.

Con la incorporación del aprendizaje profundo, la línea dominante emplea arquitecturas que modelan explícitamente la correlación espacio-temporal de la red. Petersen, Rodrigues y Pereira [2] proponen una red convolucional-recurrente (ConvLSTM) multi-salida para predecir tiempos de viaje de buses en Copenhague y reportan que su modelo supera significativamente a todos los demás métodos comparados, un resultado positivo sin matices. En el trabajo más reciente, Li, Yang y Wang [3] abordan la predicción de tiempos de arribo de buses y tranvías en Dresde con arquitecturas de tipo Transformer no estacionario; su conjunto de comparación se limita a otros modelos profundos (TCN, Transformer, ArrivalNet) y sus mejoras se informan únicamente como reducciones porcentuales de RMSE, MAE y MAPE, sin pruebas de significancia estadística ni un modelo de referencia *naive*.

### B. Rigor de las comparaciones: el baseline naive y la significancia estadística

Una línea metodológica transversal advierte que buena parte de la literatura de pronóstico compara los modelos nuevos solo contra otros modelos complejos, omitiendo referencias simples. Beck, Dovern y Vogl [4] y Ughi, Lomurno y Matteucci [5] sostienen que un modelo de referencia simple es indispensable para verificar la efectividad real de un método propuesto. La Competencia M4 [6] mostró empíricamente que la mayoría de los métodos de aprendizaje automático puro no lograron superar a los referentes estadísticos y *naive*. En la misma dirección, Elsayed et al. [7] reportan que un árbol de regresión potenciado por gradiente (GBRT), con una transformación de ventana adecuada, supera a los modelos profundos evaluados sobre múltiples conjuntos de datos.

Esta discusión es directamente pertinente al dominio del transporte. Manibardo, Laña y Del Ser [8], en pronóstico de tráfico vial, muestran que el aprendizaje profundo no supera de manera consistente a enfoques menos complejos y que la referencia de persistencia (*last value*) es competitiva a horizontes cortos, lo que deja un margen estrecho de mejora. La prueba de Diebold-Mariano [9], concebida para comparar la precisión de dos pronósticos sobre las mismas observaciones y robusta a errores autocorrelacionados, constituye el instrumento estándar para dirimir estas comparaciones; su ausencia en el subcampo del *headway* es una de las carencias que este trabajo atiende.

### C. Resultados negativos y dependientes del régimen (evidencia de dominios vecinos)

Existe evidencia consolidada de que la complejidad arquitectónica no siempre mejora el desempeño, aunque —y esto debe subrayarse— proviene mayormente de fuera del dominio del *headway*. Rodrigues [10] argumenta que la literatura de aprendizaje profundo en transporte tiende a sobre-enfatizar el modelado de correlaciones espaciales, y que una referencia basada en el patrón semanal promedio más una regresión lineal alcanza resultados comparables a numerosos modelos profundos del estado del arte. En pronóstico de series volátiles, Beck, Dovern y Vogl [4] hallan que ningún método supera de forma consistente al pronóstico *naive* cuando la serie es altamente volátil, mientras que en series más predecibles muchos métodos sí lo logran. Es importante señalar que este resultado apunta en dirección **opuesta** al hallazgo del presente trabajo —donde la ventaja del aprendizaje profundo se concentra, y no se anula, en el régimen de alta volatilidad—, lo que refuerza que la relación entre volatilidad y desempeño relativo es dependiente del dominio y permanece sin resolver para el *headway* de buses.

### D. Síntesis del vacío (gap)

En conjunto, la evidencia crítica —competitividad de las referencias *naive*, paridad de los árboles potenciados frente al aprendizaje profundo, marginalidad de la complejidad espacial y dependencia del régimen— está bien establecida, pero casi enteramente **fuera** del subcampo del pronóstico de *headway* y *bunching*: en tráfico vial, series económico-financieras y bancos de prueba genéricos de pronóstico. Dentro del subcampo específico, el estado del arte continúa reportando ventajas del aprendizaje profundo sin una referencia de persistencia y sin pruebas de significancia. El presente trabajo atiende ese vacío mediante (i) una evaluación pareada con pruebas de significancia (Diebold-Mariano y Wilcoxon) sobre muestras idénticas, frente a la persistencia y a un XGBoost nivelado; (ii) una caracterización explícita de las condiciones —horizonte y volatilidad— bajo las cuales el aprendizaje profundo supera a la persistencia; y (iii) el uso de datos AVL/GPS reales de una ciudad del Sur Global (Arequipa, Perú), subrepresentada en la literatura.

> [ANDAMIAJE — POR VERIFICAR / LEER A TEXTO COMPLETO antes de citar afirmaciones específicas de estas fuentes]
> - Singh & Sahu, "A review of bus arrival time prediction using artificial intelligence", *WIREs Data Mining & Knowledge Discovery*, 2022 (DOI 10.1002/widm.1457) — survey útil para el panorama de modelos (Kalman, ARIMA, SVR, LSTM…); no se pudo leer a texto completo en la relevación.
> - Regime-switching bayesiano (Markov) para tiempo de viaje/ocupación de buses, arXiv:2401.17387 — MUY pertinente para el ángulo régimen-condicional (Sección C); leer y ubicar.
> - Corroboración de persistencia competitiva a horizonte corto: Deep Echo State Networks, arXiv:2004.08170 (verificada; opcional como cita de apoyo).

**Referencias provisionales de la Sección II** — todas reales; completar iniciales, título y datos faltantes contra la fuente (NO inventar):

- [1] Yu et al., "Headway-based bus bunching prediction using transit smart card data", *Transportation Research Part C*, 2016. (Elsevier PII: S0968090X16301747)
- [2] Petersen, Rodrigues, Pereira, "Multi-output bus travel time prediction with Convolutional LSTM", *Expert Systems with Applications*, 2019. (arXiv:1903.02791)
- [3] Li, Yang, Wang, "Exploring Over-stationarization in Deep Learning-based Bus/Tram Arrival Time Prediction: Analysis and Non-stationary Effect Recovery", 2025. (arXiv:2509.06979)
- [4] Beck, Dovern, Vogl, "Mind the naive forecast! A rigorous evaluation of forecasting models for time series with low predictability", *Applied Intelligence*, 2025. (DOI: 10.1007/s10489-025-06268-w)
- [5] Ughi, Lomurno, Matteucci, [título por confirmar], 2023. (arXiv:2304.04553)
- [6] Makridakis, Spiliotis, Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods" [confirmar título], *International Journal of Forecasting*, 2020. (Elsevier PII: S0169207019301128)
- [7] Elsayed et al., "Do We Really Need Deep Learning Models for Time Series Forecasting?", ECML/PKDD [confirmar venue], 2021. (arXiv:2101.02118)
- [8] Manibardo, Laña, Del Ser, "Deep Learning for Road Traffic Forecasting: Does it Make a Difference?", *IEEE Transactions on Intelligent Transportation Systems*, 2021. (arXiv:2012.02260)
- [9] Diebold, Mariano, "Comparing Predictive Accuracy" [confirmar datos], *Journal of Business & Economic Statistics*, 1995.
- [10] Rodrigues, [título por confirmar], 2022. (arXiv:2203.02954)

---

## III. Materiales y Métodos

> [ANDAMIAJE] YA REDACTADO casi todo en documento-resultados.md (Sección 2) y en el código (src/evaluation/). Reordenar y completar el preprocesamiento que el doc de resultados omite a propósito.

### A. Datos: SIT Arequipa (AVL/GPS)
> [ANDAMIAJE] Fuente, período (2023-10-01 → 2024-02-29, 152 días), corredores E2/E59/E4, clave compuesta (empresaid, unidadid), escala de flota. Ver docs/dataset-manifest.md.

### B. Preprocesamiento
> [ANDAMIAJE] Split temporal (train 107d / val 23d / test 22d), winsorización p99 en train aplicada a todos los splits. Declarar el contrato.

### C. Modelos comparados
> [ANDAMIAJE] Tabla de baselines B0–B4 (persistencia como rival central), B5_XGB nivelado, y los tres DL (LSTM, SpatialConvLSTM, SpatialTransformer). Reusar tablas de Sección 2 del doc de resultados.

### D. Protocolo de evaluación
> [ANDAMIAJE] Lo que te diferencia — desarrollarlo bien: comparación pareada sobre muestras idénticas (paired audit), Diebold-Mariano con HAC/Newey-West, Wilcoxon, test de signos para DL-vs-XGBoost, terciles de volatilidad ex-ante congelados en train+val. Horizontes h ∈ {1,3,5,10}. Métricas MAE/RMSE.

---

## IV. Resultados

> [ANDAMIAJE] YA REDACTADO — Secciones 3, 4 y 5 de documento-resultados.md. Trasladar figuras (curva-degradacion.png, volatilidad-crossover.png, volatilidad-exante.png) y tablas. Mantener la base pareada como canónica.

### A. Curva de degradación y crossover horizonte-dependiente
> [ANDAMIAJE] Sección 3 del doc. El crossover MAE (persistencia gana h=1, DL gana h≥3), la brecha que crece con el horizonte, chequeo contra mejor baseline simple, y el matiz XGBoost/escala (E4).

### B. Significancia estadística
> [ANDAMIAJE] Sección 4. DM + Wilcoxon, tamaño de efecto primero, p-valor como piso. Robustez por seed y por hiperparámetro.

### C. Dónde gana el DL: régimen de volatilidad
> [ANDAMIAJE] Sección 5. Estratificación retrospectiva (con su advertencia de circularidad) y confirmación ex-ante.

### D. El enrutador ex-ante
> [ANDAMIAJE] Sección 5. La política que empata/gana en las 12 celdas, con el matiz honesto de que la señal de volatilidad aporta solo ~1 s sobre la regla trivial.

---

## V. Discusión

> [ANDAMIAJE] MITAD nueva, MITAD reutilizada.

### A. Interpretación operativa
> [ANDAMIAJE] POR ESCRIBIR. Qué significa el crossover para un operador: a 1 min el pronóstico es trivial e inútil; a ≥3 min el DL da el margen para intervenir; la ventaja se concentra donde el bunching arranca. Conectar con el valor operativo.

### B. El resultado nulo espacial, en contexto
> [ANDAMIAJE] POR ESCRIBIR (corto). Enmarcarlo como específico de estos datos/corredores, NO como ley universal (modelos post-2023 tipo PatchTST/iTransformer recuperaron SOTA con atención).

### C. Amenazas a la validez
> [ANDAMIAJE] YA REDACTADO — las 8 de documento-resultados.md (objetivo censurado, sin rolling-origin, confusor Carnaval, n efectivo, agregación de direcciones, sin estratificar por magnitud, valor operativo afirmado, sin corrección múltiple).

### D. Limitaciones y alcance
> [ANDAMIAJE] YA REDACTADO — bloque de alcance del doc. 3 corredores, una ciudad, 5 meses; validez externa geográfica/temporal abierta.

---

## VI. Conclusión y Trabajo Futuro

> [ANDAMIAJE] YA REDACTADO (Sección 6 del doc) — REESCRIBIR para cerrar sobre el GAP, no solo sobre resultados. El aporte no es "el DL gana": es dónde/cuándo gana cada uno, medido con rigor, sobre datos reales. Trabajo futuro: enrutador de producción, rolling-origin, más ciudades.

---

## Referencias

> [ANDAMIAJE] POR ESCRIBIR. Formato de la plantilla IJACSA (IEEE-like). Empezar por las fuentes del Related Work ya relevadas. Verificar cada cita antes de incluirla.

<!--
  MAPA DE PROGRESO (borrar antes de enviar):
  [YA REDACTADO] III Métodos · IV Resultados · V.C Amenazas · V.D Limitaciones · VI Conclusión (reescribir)
  [POR ESCRIBIR] Abstract · I Introducción · II Related Work · V.A Interpretación · V.B Nulo espacial · Referencias
  Cuello de botella real: II Related Work (empezar por aquí).
-->
