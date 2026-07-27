# Deep Learning para el pronóstico de *headway*[^headway] de buses: ventaja condicionada al horizonte y a la volatilidad del servicio

**Documento de resultados** · Corredores E2, E59 y E4 · Horizontes de 1 a 10 minutos

> Este documento presenta **qué encontramos**, no cómo se limpiaron o normalizaron los datos. El preprocesamiento se resume al mínimo necesario para que los resultados sean reproducibles; el foco está en la evidencia.

---

## 1. La pregunta

> **¿Cuándo vale la pena usar un modelo de Deep Learning[^dl] para predecir el *headway* de buses, en lugar de un método estadístico clásico?**

La respuesta corta —y el aporte de este trabajo— es que **no siempre conviene**. El Deep Learning (DL) gana cuando se predice con suficiente anticipación (horizonte[^horizonte] **≥ 3 min**); a 1 minuto **la persistencia gana en los tres corredores, sin excepción** (medido sobre muestras idénticas). El cruce es una afirmación sobre el **MAE**: bajo RMSE el DL gana en todos los horizontes, por razones que se explican en la Sección 2. Esa condición —el horizonte— se conoce de antemano, así que la recomendación es directamente accionable. Además mostramos *de dónde* viene la ventaja: se concentra en los tramos de **alta volatilidad**[^volatilidad] del servicio —cuando el *headway* da saltos grandes—, que es donde un pronóstico preciso más valdría.

Y la respuesta no se queda en el diagnóstico: mostramos que la elección puede **automatizarse**. Una regla construida solo con información disponible al momento de predecir —el horizonte y la volatilidad reciente del servicio— nunca queda por detrás de ninguno de los dos modelos puros en las 12 combinaciones de corredor y horizonte. La honestidad obliga a acotar el tamaño: casi toda esa ganancia proviene de conocer el horizonte, y la estratificación por volatilidad agrega apenas **−0.018 min**. El aporte no es error más bajo, sino que el mapa de regímenes resulta **ejecutable**.

Este documento lo demuestra en cuatro pasos: **(1)** el DL gana → **(2)** la diferencia es estadísticamente real → **(3)** explicamos *por qué* gana → **(4)** convertimos ese "por qué" en una regla de decisión operativa.

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
| **B5_XGB** | **Gradient boosting (XGBoost[^xgboost])** | Modelo **entrenado** que ve la **misma ventana de 12 pasos** que el LSTM (su primer *lag* es exactamente la persistencia) más hora, día, dirección y la **bandera de día atípico** — nivelado con la misma información que la red y con hiperparámetros elegidos por búsqueda en validación; el competidor *aprendido* a batir |

> **Por qué sumamos un baseline ajustado.** Ganarle solo a fórmulas naive (B0–B4) podría descartarse como "vencer rivales débiles". B5_XGB es un aprendiz de verdad y está **nivelado con la red**: recibe la **misma bandera de día atípico** (construida con el mismo *helper*, para que la semántica no difiera entre familias) y sus **hiperparámetros salen de una búsqueda de 24 configuraciones seleccionada estrictamente sobre validación** —la configuración ganadora por (corredor, horizonte) queda registrada en `xgb_search_config_multih.csv` para que sea auditable—. Ninguna de las ventajas que la red tenía sobre él sigue en pie. Medido sobre **muestras idénticas** —la única base válida, ver la subsección correspondiente de la Sección 3— el resultado es de tres niveles: el LSTM le gana con holgura creciente en **E59**, **empata dentro del ruido de semilla en E2**, y **pierde en E4** salvo al horizonte más largo. Ganarle a un competidor con las mismas armas sería evidencia mucho más fuerte que ganarle a un rival de paja; el punto de este documento es que, en el caso general, **el DL no le gana**. La ventaja frente al aprendiz fuerte existe solo en el corredor de más tráfico y crece con el horizonte.

### Modelos de Deep Learning

| Modelo | Idea de la arquitectura |
|--------|-------------------------|
| **LSTM**[^lstm] | Red recurrente que aprende patrones en la secuencia temporal de *headways* |
| **SpatialConvLSTM**[^convlstm] | Añade una convolución que mira la relación entre buses cercanos antes del LSTM |
| **SpatialTransformer**[^transformer] | Añade auto-atención espacial entre buses antes del LSTM |

### El terreno de juego

- **3 corredores:** E2, E59 y E4 (tres líneas de bus reales de distinta escala de flota; E4 es el más chico, 19 buses, e ingresa como **validez externa acotada a la escala de flota** —una línea independiente y mucho más pequeña, no otra ciudad: la generalización geográfica queda fuera de alcance, ver Sección 6).
- **4 horizontes:** predecir a 1, 3, 5 y 10 minutos vista.
- **2 métricas:** MAE[^mae] y RMSE[^rmse], ambas en minutos (cuanto más bajas, mejor). El texto reporta **MAE**, y esa elección es sustantiva: **el cruce de la Sección 3 es específico del MAE**. Bajo RMSE el DL le gana a la persistencia en *todos* los horizontes, incluido h = 1: en las 9 celdas corredor×modelo de h = 1 el punto estimado favorece al DL, así que no hay cruce que reportar. **Con una salvedad que hay que declarar: una de esas nueve no es significativa** — `SpatialTransformer · E2 · h=1`, con un *p* de Wilcoxon de **0.409** (es la única marcada `⊘ ns` en la Figura 1). Las otras ocho sí lo son. Es decir: el "no hay cruce bajo RMSE" se apoya en 8 celdas significativas más una no concluyente, no en 9 victorias establecidas. La razón es conocida: el RMSE penaliza cuadráticamente los errores grandes, y la persistencia —que a h = 1 acierta casi siempre pero se rompe fuerte en los saltos— es castigada de forma desproporcionada por esa cola. El MAE, que trata todos los minutos de error por igual, es la métrica operativamente correcta acá: para un despachador, equivocarse 2 minutos es el doble de malo que equivocarse 1, no cuatro veces peor. **Todas las afirmaciones de cruce de este documento deben leerse como afirmaciones sobre el MAE**; las cifras de RMSE están en `csv-multihorizon/significance_multihorizon.csv` y en la Figura 1.
- **Datos:** 5 meses contiguos (2023-10-01 → 2024-02-29, 152 días) divididos temporalmente[^split]:
  - **Entrenamiento:** 2023-10-01 → 2024-01-15 (107 días, ≈3.5 meses) — el modelo aprende.
  - **Validación:** 2024-01-16 → 2024-02-07 (23 días, ≈3 semanas) — se ajustan los hiperparámetros[^hiperparametros].
  - **Prueba:** 2024-02-08 → 2024-02-29 (22 días, ≈3 semanas) — evaluación final; **todo lo reportado es sobre este conjunto, que el modelo nunca vio.**

> **Por qué la persistencia (B1) es el rival serio.** Es tentador comparar el DL contra un promedio tonto y "ganar" fácil. Pero en series temporales cortas, *repetir el último valor* es sorprendentemente difícil de superar: si el bus venía con 4 minutos de *headway*, lo más probable es que dentro de 1 minuto siga cerca de 4. Batir a la persistencia es la verdadera prueba. Por eso B1 es la referencia central de las comparaciones; los demás *baselines* (B0, B2–B4) se incluyen como contexto en la Figura 1.

---

## 3. Resultado central: la curva de degradación

![Curva de degradación](curva-degradacion.png)

*Figura 1 — MAE y RMSE frente al horizonte de predicción, para E2, E59 y E4. Cuanto más bajo, mejor. Se grafican la persistencia (B1), los baselines formulaicos B0/B3/B4_HA como contexto, el baseline ajustado XGBoost (B5) y los tres modelos profundos. Las barras de error del LSTM son el IC 95 % sobre 5 seeds (ver Sección 4, "¿O es casualidad del seed?"); su ancho sub-marcador refleja la estabilidad frente al seed.*
>
> ⚠️ **Cómo NO leer esta figura.** Los paneles son el MAE/RMSE **agregado** sobre el test completo, porque ese es el único encuadre en el que los baselines formulaicos y el ajustado existen. **La comparación canónica del trabajo es la pareada sobre muestras idénticas**, y el encuadre agregado favorece sistemáticamente al DL en 0.28–0.53 min (ver el recuadro "Pareado vs. agregado" más abajo). En consecuencia:
>
> - **⊘ ns** marca una diferencia DL-vs-persistencia no significativa.
> - **▪ B1** marca las celdas donde la persistencia es significativamente **mejor** que el modelo profundo. Son las nueve celdas de h = 1 en MAE (tres modelos × tres corredores) más el SpatialTransformer en E4·h=3.
> - **▪ B1≠** marca, además, las **4 celdas donde el signo que se ve en este panel se invierte en la base pareada** — los tres modelos en E2·h=1·MAE y el Transformer en E4·h=3·MAE. Ahí el panel agregado muestra al DL por debajo de la persistencia, pero sobre muestras idénticas gana la persistencia.
>
> **El cruce (*crossover*) debe leerse de la tabla pareada de esta sección, no de esta figura.** El pie de la figura reporta los conteos de significancia calculados desde `significance_multihorizon.csv`, no afirmados a mano.

**El mensaje:** a 1 minuto la persistencia es imbatible pero **operativamente inútil** (nadie puede reaccionar con 1 minuto de aviso). El valor del DL **emerge al anticipar a 3, 5 y 10 minutos** — justo el margen que un operador necesita para intervenir.

Mirá el cruce (*crossover*[^crossover]) en el extremo izquierdo de cada panel (✓ = gana el DL; ❌ = gana la persistencia):

| Corredor | h = 1 min | h = 3 min | h = 5 min | h = 10 min |
|----------|-----------|-----------|-----------|------------|
| **E59** (MAE) | B1 **2.82** vs LSTM 3.16 ❌ *gana persistencia* | 3.90 vs **3.72** ✓ | 4.40 vs **3.93** ✓ | 5.28 vs **4.19** ✓ |
| **E2** (MAE) | B1 **4.24** vs LSTM 4.27 ❌ *gana persistencia* | 5.76 vs **4.86** ✓ | 6.22 vs **5.02** ✓ | 6.73 vs **5.16** ✓ |
| **E4** (MAE) | B1 **2.84** vs LSTM 3.37 ❌ *gana persistencia* | 4.44 vs **4.40** ✓ | 5.38 vs **4.83** ✓ | 6.78 vs **5.36** ✓ |

*Toda la tabla usa la base **pareada** sobre muestras idénticas —la comparación canónica del trabajo—, no métricas agregadas sobre conjuntos de muestras distintos. El patrón es limpio y sin excepciones: **la persistencia gana a h = 1 en los tres corredores; el LSTM gana a h ≥ 3 en los tres**. Dos matices honestos: el margen de la persistencia en E2·h=1 es mínimo (0.037 min, un empate a efectos prácticos aunque estadísticamente significativo), y el del LSTM en E4·h=3 también lo es (0.040 min) — el cruce en ese corredor es esencialmente un empate que recién se despega a h ≥ 5.*

**Lo más importante — la brecha crece con el horizonte frente a la persistencia.** A medida que predecimos más lejos, la persistencia se degrada rápido y el DL aguanta. Esta comparación es la **canónica** del trabajo: se mide sobre **muestras idénticas** (pareadas) —cada predicción del LSTM se enfrenta a la persistencia sobre exactamente la misma observación—, no sobre agregados de distinto tamaño:

- **E2 a 10 min:** persistencia 6.734 vs LSTM **5.163** → **−1.57 min de error (−23.3 %)**.
- **E59 a 10 min:** persistencia 5.282 vs LSTM **4.188** → **−1.09 min de error (−20.7 %)**.
- **E4 a 10 min:** persistencia 6.776 vs LSTM **5.360** → **−1.42 min de error (−20.9 %)**.

> **Pareado vs. agregado — y por qué la distinción importa.** Todo lo reportado arriba usa la base **pareada** sobre muestras idénticas (`csv-multihorizon/paired_dl_persistence_metrics.csv`). La alternativa —métricas **agregadas** sobre el test completo— no es equivalente: el DL descarta las filas de arranque en frío que no completan la ventana de entrada, de modo que ambos encuadres promedian sobre conjuntos de muestras distintos. Ese sesgo **no es despreciable**: en la auditoría celda a celda (`paired_vs_reported_audit.csv`, 72 celdas = 3 modelos × 3 corredores × 4 horizontes × 2 métricas) la persistencia agregada aparece entre **0.28 y 0.53 min peor** que la pareada, siempre en la misma dirección —es decir, el encuadre agregado **favorece sistemáticamente al DL**—. El signo de la ventaja coincide en **68 de 72 celdas**; las 4 discrepancias son precisamente donde el margen real es menor que ese sesgo: los tres modelos en **E2·h=1** (el encuadre agregado daba la victoria al DL; en pareado gana la persistencia) y el SpatialTransformer en E4·h=3. Por eso las afirmaciones fuertes de este documento se apoyan solo en la base pareada. Los porcentajes se calculan sobre los valores con 3 decimales para que la aritmética sea exacta.

**Chequeo contra el mejor baseline simple (h = 3, 5, 10).** La persistencia es la referencia operativa central, pero no siempre es el baseline simple con menor MAE: el rival más duro **cambia con el horizonte** —persistencia (B1) en el muy corto plazo, suavizado exponencial (B3) en el medio y media horaria (B4_HA) en el largo—. Contra ese mejor baseline simple **por celda**, el LSTM gana en los tres corredores en todos los horizontes operativos (h ≥ 3), aunque con márgenes más chicos que frente a la sola persistencia:

| Corredor | Horizonte | Mejor baseline simple | LSTM | Mejora LSTM |
|---|---|---:|---:|---:|
| **E2** | h = 3 | B4_HA 5.259 | **4.916** | −0.34 min (−6.5 %) |
| **E2** | h = 5 | B4_HA 5.259 | **5.040** | −0.22 min (−4.2 %) |
| **E2** | h = 10 | B4_HA 5.259 | **5.128** | −0.13 min (−2.5 %) |
| **E59** | h = 3 | B3 (SES) 4.068 | **3.847** | −0.22 min (−5.4 %) |
| **E59** | h = 5 | B3 (SES) 4.438 | **4.029** | −0.41 min (−9.2 %) |
| **E59** | h = 10 | B4_HA 4.805 | **4.224** | −0.58 min (−12.1 %) |
| **E4** | h = 3 | B1 (persist.) 4.783 | **4.679** | −0.10 min (−2.2 %) |
| **E4** | h = 5 | B3 (SES) 5.492 | **5.014** | −0.48 min (−8.7 %) |
| **E4** | h = 10 | B4_HA 5.746 | **5.348** | −0.40 min (−6.9 %) |

El margen más ajustado es E4 a h = 3 (−2.2 %) y E2 a h = 10 (−2.5 %); aun así el signo favorece al DL en las nueve celdas. Esto no cambia la tesis operativa contra persistencia, pero evita sobredimensionar la magnitud del margen cuando el rival es el mejor baseline simple por celda. (A h = 1 el mejor baseline simple aún supera al DL en E59 y E4 —el *crossover* ya discutido—; por eso este chequeo se acota a los horizontes operativamente accionables.)

> **Nota honesta sobre los modelos espaciales: el resultado es un nulo, no un ranking.** Las tres arquitecturas profundas son **estadísticamente indistinguibles entre sí en estos datos**. El spread entre ellas es de 0.017 a 0.038 min según la celda, mientras que reentrenar *la misma* arquitectura cambiando solo la semilla mueve el MAE entre 0.001 y 0.037 min (Sección 4). **La diferencia entre arquitecturas cae dentro del ruido de entrenamiento**, así que este documento **no declara** un ganador entre LSTM, ConvLSTM y SpatialTransformer: hacerlo sería adjudicar celdas con márgenes un orden de magnitud por debajo de la incertidumbre que él mismo mide.
>
> Lo que sí se sostiene —y es el hallazgo— es el **nulo**: añadir complejidad espacial (convolución sobre buses vecinos, auto-atención espacial) **no mejoró el pronóstico**. Un modelo recurrente plano alcanza el mismo desempeño que las dos variantes espaciales, y el nulo se replica en los tres corredores, incluido el chico. Queda abierto si se sostendría con más buses por *snapshot*[^snapshot]. Advertencia de cobertura: solo el LSTM se reentrenó con múltiples semillas, así que el ruido citado arriba se midió sobre él (ver la Sección 4).

### El contraste con el competidor ajustado, medido sobre muestras idénticas

La objeción natural a "el DL le gana a la persistencia" es: *¿y si los baselines son demasiado débiles?* Por eso entrenamos **B5_XGB** (gradient boosting[^xgboost]) con la misma ventana de 12 pasos que ve el LSTM. Es un competidor **creíble**: le gana a la persistencia y al mejor baseline formulaico en casi todas las celdas (p. ej. E59 a h=10: mejor clásico 4.81 → XGBoost **4.64**).

> **Por qué esta subsección se recalculó, y qué estaba mal antes.** Una versión previa de este documento reportaba que el LSTM le ganaba al XGBoost en las 8 celdas de E2 y E59, con un test de signos a nivel de celda (8 de 8, *p* = 0.004). **Ese resultado era inválido, y lo era por la misma razón que la Sección 3 ya había identificado un nivel más abajo.** Las métricas agregadas del XGBoost se calculan sobre **todas** las filas de test con predicción no nula; las del LSTM, sobre la **población de ventanas** (sin filas de arranque en frío, con cada objetivo replicado una vez por *slot* de anclaje). Son conjuntos de muestras distintos. Y el sesgo de ese desalineamiento —que este documento mide en **0.28–0.53 min**, siempre a favor del DL— es **más grande que 7 de los 8 márgenes que se reclamaban** (+0.05 a +0.41 min). El test de signos no lo corregía: eran 8 tiradas de moneda sobre 8 mediciones sesgadas, así que heredaba el sesgo íntegro.
>
> La corrección **no requirió reentrenar nada**. El obstáculo era que los residuos por muestra del XGBoost no llevaban la clave necesaria: se exportaban con `t`, que **no es única** (hay ~4.49 filas por `(t, direction)`, una por *pair_rank*). Un kernel dedicado (`20-xgb-paired-export`) reexportó las predicciones por muestra con la clave completa `(corredor, direction, horizonte, t, pair_rank)` —única, verificada en tiempo de ejecución— y su reajuste **reprodujo exactamente** la búsqueda de hiperparámetros congelada (las 12 celdas coinciden con `xgb_search_config_multih.csv` a precisión de punto flotante completa), de modo que los números nuevos son directamente comparables con los viejos. Después se recalificó el XGBoost restringido a **exactamente las filas que evaluó el LSTM**: cobertura del **100.000 %** en las 12 celdas, y el portón de alineación muestra a muestra cierra a **2.3e-06 – 2.7e-06** contra una tolerancia de 1e-2.

Con la comparación así nivelada, el resultado es **más matizado y más informativo** que el que reemplaza:

| Corredor | h = 1 | h = 3 | h = 5 | h = 10 |
|----------|-------|-------|-------|--------|
| **E2** — LSTM / XGBoost | 4.274 / 4.310 | 4.857 / 4.862 | 5.025 / **5.020** | 5.163 / **5.163** |
| **E2** — Δ MAE (XGBoost − LSTM) | +0.036 | +0.005 | −0.005 | −0.000 |
| **E59** — LSTM / XGBoost | **3.163** / 3.169 | **3.721** / 3.783 | **3.930** / 4.060 | **4.188** / 4.430 |
| **E59** — Δ MAE (XGBoost − LSTM) | +0.006 | +0.061 | +0.130 | **+0.242** |
| **E4** — LSTM / XGBoost | 3.367 / **3.092** | 4.398 / **4.192** | 4.832 / **4.781** | **5.360** / 5.499 |
| **E4** — Δ MAE (XGBoost − LSTM) | −0.275 | −0.206 | −0.051 | +0.139 |

*Δ positivo = gana el LSTM. Datos: [`csv-multihorizon/xgb_paired_dl_metrics.csv`](csv-multihorizon/xgb_paired_dl_metrics.csv). Los MAE de esta tabla son sobre la población de ventanas del LSTM, así que **no** son comparables con los MAE agregados de la Figura 1; la reconciliación celda a celda entre ambos encuadres está en [`csv-multihorizon/xgb_paired_vs_reported_audit.csv`](csv-multihorizon/xgb_paired_vs_reported_audit.csv).*

> **Las dos poblaciones posibles, y por qué se reportan ambas.** La tabla de arriba usa la población **multiplicidad-emparejada**: una fila del XGBoost por cada fila del LSTM, replicando el objetivo tantas veces como el LSTM lo emite (≈ 4.5×). Es la única cuyo MAE es directamente comparable con el MAE del LSTM que este documento ya reportaba, y por eso encabeza. Pero existe una segunda población defendible —**objetivos distintos**, una fila por `(direction, t, pair_rank)`— que es la base correcta para un test por muestra, porque no cuenta cada objetivo varias veces. **Da un resultado peor para el LSTM, así que se reporta explícitamente:**
>
> | Corredor | h = 1 | h = 3 | h = 5 | h = 10 |
> |---|---|---|---|---|
> | **E2** — Δ MAE, objetivos distintos | −0.007 | −0.041 | −0.056 | −0.058 |
> | **E59** — Δ MAE, objetivos distintos | +0.010 | +0.059 | +0.128 | **+0.238** |
> | **E4** — Δ MAE, objetivos distintos | −0.197 | −0.123 | +0.015 | +0.178 |
>
> Diferencias respecto de la población emparejada: en **E2 el LSTM pasa a perder las cuatro celdas** (aunque por márgenes de ≤ 0.058 min, todos dentro o al borde del ruido de semilla); en **E4 el LSTM gana h = 5 además de h = 10**; en **E59 el patrón es idéntico**, con el mismo gradiente monótono. Reportar solo la población emparejada habría mostrado la más favorable al DL de las dos, y por eso van las dos.

**La vara para leer estos números es el ruido de semilla, no el cero.** La Sección 4 mide que reentrenar la misma configuración del LSTM con distintas semillas mueve el MAE entre 0.001 y 0.037 min, con un semi-ancho de IC 95 % de a lo sumo 0.046 min. Un margen por debajo de esa banda no es una victoria de nadie. Con ese criterio, el resultado se separa en **tres regímenes, uno por corredor**:

- **E59 (corredor grande) — el LSTM gana, y la brecha crece con el horizonte.** Gana las cuatro celdas, y el margen es monótono creciente: +0.006 → +0.061 → +0.130 → **+0.242 min**. A h = 5 y h = 10 está claramente fuera del ruido de semilla. **Es el único hallazgo del contraste que se replica idéntico en las dos poblaciones** (+0.010 → +0.238 en objetivos distintos), y es el hallazgo positivo: **la misma curva de degradación de la Sección 3, ahora contra un aprendiz fuerte y nivelado en vez de una fórmula naive.**
- **E2 (corredor grande) — indistinguible.** Los ocho márgenes de este corredor (cuatro por población) están entre −0.058 y +0.036 min, es decir **dentro o al borde del ruido de semilla** (±0.046). Las dos familias no se distinguen acá. Con la salvedad de que el signo **no** es simétrico: en la población emparejada el LSTM gana dos celdas, pero en objetivos distintos **pierde las cuatro**. La lectura honesta es "empate", no "gana el LSTM".
- **E4 (corredor chico, 19 buses) — el XGBoost gana en horizontes cortos.** Gana h = 1 y h = 3 en ambas poblaciones, por márgenes muy por fuera del ruido (−0.275/−0.197 y −0.206/−0.123): son victorias reales del XGBoost. h = 5 depende de la población (−0.051 emparejada, +0.015 distintos), y a h = 10 gana el LSTM en las dos. La dirección sí es consistente: **el margen se mueve monótonamente hacia el LSTM al alargar el horizonte**, igual que en E59.

**El test de signos, recalculado, en las dos poblaciones y sin recortes.**

| Grupo | Población emparejada | Objetivos distintos |
|---|---|---|
| E2 + E59 | 6 de 8 (*p* = 0.145) | **4 de 8 (*p* = 0.637)** |
| Las 12 celdas | 7 de 12 (*p* = 0.387) | **6 de 12 (*p* = 0.613)** |
| E4 aislado | 1 de 4 (*p* = 0.938) | 2 de 4 (*p* = 0.688) |

**Ninguno de los seis es significativo**, y lo reportamos así. Conviene además declarar que separar E4 del resto es una partición **post-hoc** —se aisló después de observar que fallaba—, así que la cifra honesta de referencia es la agrupada de 12 celdas, en cualquiera de las dos poblaciones. Los tests pareados por muestra (Diebold-Mariano con HAC y Wilcoxon **unilateral**, con su dirección registrada) sobre la población de objetivos distintos están en [`csv-multihorizon/xgb_paired_significance.csv`](csv-multihorizon/xgb_paired_significance.csv).

**Qué queda en pie, entonces.** Conviene separar **dos tesis de fuerza muy distinta**:

- **Tesis robusta (se replica en los tres corredores, incluido el chico):** frente a la **persistencia**, el DL gana a h ≥ 3 en E2, E59 y E4. Este es el hallazgo central del trabajo y es el que sobrevive al cambio de escala de flota, a la corrección de población y al test pareado por muestra.
- **Tesis condicional (depende de la escala, y es débil):** frente a un **aprendiz fuerte** como el XGBoost, la ventaja del DL es **clara solo en E59**, es **inexistente en E2** (empate dentro del ruido) y **se invierte en E4** salvo al horizonte más largo. Un test de signos sobre las 12 celdas **no** la respalda.

La lectura final es que **el DL no aporta nada sobre un gradient boosting bien nivelado en el caso general**: aporta en el corredor de más tráfico y al anticipar más lejos. Lo que sí es consistente en dos de los tres corredores —y merece ser el titular de esta subsección— es que **la ventaja relativa del DL sobre el aprendiz fuerte crece con el horizonte**, exactamente como crece frente a la persistencia. El mecanismo de la Sección 5 (el DL es robusto a la volatilidad, los métodos que extrapolan el estado reciente se rompen) explica ambos gradientes con el mismo argumento.

> En E59 a h=1 la persistencia (3.10) sigue siendo la mejor de todos —incluido el XGBoost (3.38)—, coherente con que a 1 minuto el pronóstico es trivial.

> **Nota sobre el nivelado del XGBoost.** Darle la bandera de día atípico y buscarle los hiperparámetros sobre validación movió su MAE ≤ 0.04 min, de modo que la comparación ya era representativa antes de nivelarlo. Hay, sin embargo, una asimetría que corre **en contra** del DL y que conviene declarar: los hiperparámetros del LSTM se eligieron una sola vez, prediciendo a 1 paso, sobre E2 y E59 (ver `configuraciones-ganadoras.md`), y se reutilizan en todos los horizontes y en E4; el XGBoost, en cambio, recibe una búsqueda de 24 configuraciones **por cada (corredor, horizonte)**. El competidor ajustado está, en ese sentido, mejor sintonizado que la red.

---

## 4. ¿Es real o es casualidad? — Significancia estadística

Que un número sea más bajo no basta: podría ser ruido. Para descartarlo aplicamos dos tests pareados a cada comparación DL-vs-persistencia.

- **Tests usados:** Diebold-Mariano[^dm] (compara el error medio) y Wilcoxon[^wilcoxon] (compara las medianas por rangos). El estadístico DM se calcula con una **varianza de largo plazo Newey-West (HAC)** —lag de truncación data-driven ⌊n^{1/3}⌋, kernel de Bartlett—, de modo que la **autocorrelación serial** que introducen las ventanas de entrada solapadas **no infla la significancia**: está corregida por construcción.
- **Advertencia sobre el p-valor.** Aun con esa corrección HAC, con n = 0.5–2.2 M muestras pareadas por celda cualquier diferencia mínima da p ≈ 0: el p-valor confirma que el **signo** del efecto no es ruido, pero **no mide su importancia práctica**. Por eso esta sección lidera con el **tamaño del efecto** (Δ MAE en minutos, Sección 3) y trata la significancia como un **piso de sanidad**, no como la evidencia principal.
- **Resultado (tamaño del efecto):** el DL tiene el menor error en **53 de las 54** comparaciones (3 modelos DL × 3 corredores × 3 horizontes h ∈ {3,5,10} × 2 métricas; se excluye h=1, donde la persistencia gana). Cada comparación se juzga sobre **su propia métrica**: la victoria/derrota en RMSE se decide por el diferencial de error cuadrático, no por el de MAE.
- **Resultado (significancia como piso):** de esas 53, **51 son significativas a p[^pvalor] < 0.001** en *ambos* tests, y **las 53 a p < 0.05**.

Las tres desviaciones, declaradas en su totalidad:

| Desviación | Caso | Lectura |
|---|---|---|
| Gana la persistencia (1 celda) | Transformer · E4 · h=3 (solo MAE) | El Transformer queda apenas detrás en MAE (Δ MAE +0.04); en RMSE ese mismo Transformer **sí** gana, aunque solo a p<0.05 (Wilcoxon p = 0.003); el LSTM y el ConvLSTM ganan ese RMSE a p<0.001 |
| DL gana, significativo solo a p<0.05 | ConvLSTM · E4 · h=3 (MAE) | DM p = 0.039 (pasa 0.05, no 0.001); efecto chico pero a favor del DL |
| DL gana, significativo solo a p<0.05 | Transformer · E4 · h=3 (RMSE) | Wilcoxon p = 0.003 (pasa 0.05, no 0.001); DM sí a p ≈ 0 |

> **Las desviaciones no contradicen el patrón.** Las tres se concentran en **h=3** (el horizonte más corto de los testeados) y todas en el corredor chico **E4**; a h ≥ 5 los tres modelos ganan con holgura en los tres corredores. Son casos límite, no contraejemplos.

**Conclusión de esta sección:** la ventaja del DL a h ≥ 3, **medida en minutos de error**, es sistemática y consistente; la significancia estadística la respalda como piso, no como su justificación.

### ¿O es casualidad del hiperparámetro?

Queda un último flanco: ¿el resultado del LSTM depende de haber acertado un ajuste fino y frágil? Para descartarlo corrimos un **mini-grid de sensibilidad** a h = 10. Para cada corredor se entrenaron 4 configuraciones —la **ganadora congelada** del grid de Fase 5 más **3 vecinas**, cada una alterando **un solo** hiperparámetro (capacidad oculta, *dropout*[^dropout] o tasa de aprendizaje)— con el **mismo seed, pipeline y *splits***. La comparación es interna: no depende de números históricos.

| Corredor | Vecindario (4 configs) | MAE de la ganadora | Rango de MAE del vecindario | Dispersión |
|---|---|---|---|---|
| **E2** | hidden∈{32,64}, dropout∈{0,0.2}, lr∈{5e-4,1e-3} | 5.128 | 5.128 – 5.152 | **0.5 %** |
| **E59** | hidden∈{32,64}, dropout∈{0,0.2}, lr∈{5e-4,1e-3} | 4.224 | 4.224 – 4.244 | **0.5 %** |

*Datos: [`csv-multihorizon/lstm_minigrid_h10.csv`](csv-multihorizon/lstm_minigrid_h10.csv) (8 filas: 2 corredores × 4 configs; en E2 son 3 distintas — el* dropout *es inerte con 1 capa, ver abajo).*

**El rendimiento es estable: mover cualquier perilla mueve el MAE menos del 1 %.** En ambos corredores la configuración elegida es la **mejor** de su vecindario y ninguna vecina la supera. Una salvedad de honestidad sobre E2: su "empate exacto" con la vecina de *dropout* **no es una corroboración independiente sino un duplicado**. Con una sola capa recurrente el *dropout* es inerte —`src/models/lstm.py` lo fuerza a 0 cuando `num_layers = 1`—, así que esa vecina es **bit a bit el mismo modelo** que la ganadora (MAE idéntico, 5.128151366538658). El vecindario efectivo de E2 son **3 configuraciones distintas** (varía capacidad oculta y tasa de aprendizaje), no 4; el de E59 (2 capas) sí son 4 genuinas. Aun con esa salvedad, ninguna perturbación **real** desploma el rendimiento: la ventaja del DL **no es un artefacto de un hiperparámetro afortunado**.

### ¿O es casualidad del seed?

El reclamo más automático contra cualquier resultado de Deep Learning: *todos los números salen de una sola corrida con un único seed[^seed] — ¿y si esa inicialización tuvo suerte?* Para cerrarlo re-entrenamos la **misma configuración ganadora congelada del LSTM** con **5 seeds** `[42, 123, 456, 789, 999]`, en cada horizonte (h ∈ {1, 3, 5, 10}) y ambos corredores, y reportamos **media ± intervalo de confianza del 95 %** (t de Student, n = 5).

| Corredor (MAE agregado, h=10) | Media de 5 seeds | IC 95 % | CV entre seeds |
|---|---|---|---|
| **E2** | 5.130 | [5.123, 5.136] | 0.10 % |
| **E59** | 4.224 | [4.218, 4.230] | 0.12 % |

*Datos: [`csv-multihorizon/multiseed_ci_multihorizon.csv`](csv-multihorizon/multiseed_ci_multihorizon.csv) (48 celdas: 2 corredores × 3 direcciones × 2 métricas × 4 horizontes, 5 seeds c/u). Las barras de error de la Figura 1 son justamente estos intervalos.*

**Los intervalos son diminutos: en las 48 celdas el coeficiente de variación entre seeds es de a lo sumo 0.476 %, y el IC 95 % más ancho es de ±0.05 min** — más angosto que el grosor del marcador en la curva. El valor canónico de la sección 3 proviene de una corrida **independiente** del lote de 5 seeds; difiere de la media multi-seed en **a lo sumo 0.03 min** en las 48 celdas (cae dentro del IC —angostísimo— en 38 de 48, y a ≤ 0.01 min del borde en el resto). A escala operativa es indistinguible de la media: el resultado canónico es **representativo**, no un golpe de suerte.

¿Por qué un IC tan angosto? No por un entrenamiento casi determinista, sino por dos razones legítimas: **(a)** el conjunto de test es enorme (0.5–2.2 M observaciones por celda), así que el estimador del MAE/RMSE es muy estable; y **(b)** el *early stopping* sobre validación lleva a todos los seeds a óptimos muy parecidos. Los seeds están realmente cableados (init de pesos y *dropout* vía `torch`/`cuda`/`numpy`, ver `src/train.py:set_seed`) y producen modelos **distintos** — el desvío entre seeds es **no nulo** (0.001–0.037 min). La ventaja del LSTM sobre la persistencia **no depende de un seed afortunado**: es estable frente al azar del entrenamiento.

> **Límite de cobertura de este experimento, declarado.** El barrido de semillas se corrió **solo sobre el LSTM, y solo en E2 y E59**. El ConvLSTM y el SpatialTransformer tienen **una sola corrida cada uno**, igual que las tres arquitecturas en E4. Una versión previa de este documento argumentaba que, al compartir curva las tres arquitecturas, la varianza por semilla del LSTM "acota a toda la familia DL"; **ese argumento no se sostiene** —que dos modelos coincidan en su estimación puntual no dice nada sobre cuánto varía cada uno entre semillas— y se retira. Lo que este experimento establece es que **el LSTM en E2 y E59 es estable frente a la semilla**; la estabilidad de las otras dos arquitecturas y del corredor E4 queda sin medir. Esto es también la razón por la que este documento no declara un ganador entre arquitecturas (ver la nota de la Sección 3).

---

## 5. ¿Dónde gana el DL? — El patrón de la volatilidad

Esta sección descompone el promedio global en regímenes para mostrar **dónde** se concentra la ventaja del DL. Convierte un número agregado en un patrón interpretable.

> **Advertencia metodológica (leer primero).** El régimen de volatilidad se define **de forma retrospectiva**, como la magnitud del cambio real del *headway* (`|y_real − persistencia|`), conocida solo *después* del hecho. Esto introduce una **dependencia parcial con el error de la persistencia**: el régimen "alto" es, por construcción, donde la persistencia más se equivoca. Por eso este análisis es **descriptivo, no causal**: muestra que la ventaja del DL se concentra en las ventanas que *resultaron* volátiles, no que el DL "detecte" la volatilidad por adelantado. La parte informativa —que sobrevive a la circularidad— es que el **error absoluto del DL se mantiene acotado** mientras el de la persistencia se dispara con la magnitud del cambio (ver más abajo). Y para cerrar la objeción de raíz, repetimos el análisis con un estratificador **ex-ante** —volatilidad conocida *antes* de predecir— en la última parte de esta sección: la ventaja del DL no solo se sostiene, sino que se acentúa.

Partimos las predicciones en tres **regímenes de volatilidad**[^volatilidad] según cuánto cambió realmente el *headway*:

![Crossover de volatilidad](volatilidad-crossover.png)

*Figura 2 — Diferencia de MAE (DL − persistencia) en cada régimen de volatilidad. Por debajo de cero, gana el DL.*

| Régimen | Cambio real del *headway* | ¿Quién gana? | Δ MAE (DL − persistencia) |
|---------|-----------------------------|--------------|----------------------------|
| **Estable** | < 1 min | Persistencia | +2.3 a +3.4 (DL peor) |
| **Moderado** | 1–3 min | Persistencia (justo) | +0.85 a +1.9 |
| **Alto** | ≥ 3 min | **DL, decisivo** | **−2.6 a −3.7 (DL mejor)** |

*Los rangos de Δ MAE abarcan los 3 corredores (E2, E59, E4) y los 3 horizontes (h = 3, 5, 10); el patrón —persistencia en estable/moderado, DL en alto— es idéntico en todas las celdas, con la magnitud variando algo según el corredor. Los cortes de régimen son fijos en minutos (1 y 3 min), no cuantiles.*

**El hallazgo clave:** el DL **no mejora el promedio mejorando todo un poco**. El promedio "el DL le gana" es en realidad la **suma de dos regímenes opuestos**:

- En servicio **estable**, predecir es trivial (el *headway* casi no cambia) → la persistencia gana, pero es una victoria **sin valor operativo**.
- En servicio que se **desestabiliza** —el inicio del *bunching*[^bunching]— (saltos ≥ 3 min) → el DL gana de forma decisiva: **es el régimen donde un pronóstico preciso tendría más valor operativo** (siempre que pueda anticiparse el régimen — ver Sección 6).

> **El punto que NO depende de la circularidad.** Aunque el régimen se defina por el error de la persistencia, hay un hecho genuino: el **error absoluto del DL se mantiene acotado** a través de los regímenes, mientras el de la persistencia *es* la volatilidad. En E59 a h=10, el MAE de la persistencia salta 0.49 → 1.88 → **8.65** min (estable → moderado → alto), pero el del LSTM apenas se mueve: 3.34 → 3.23 → **4.95**. En E4 a h=10, ídem: persistencia 0.48 → 1.88 → **10.10** vs LSTM 3.55 → 3.79 → **6.37**. El DL no "gana porque definimos el régimen a su favor": gana porque **es robusto a la volatilidad** justo donde la persistencia se rompe.

**Y esto explica por qué la brecha crece con el horizonte** (Sección 3): a mayor horizonte, más muestras caen en el régimen de alto cambio.

- En E59, las ventanas de alto cambio pasan de **38.5 %** (h=3) a **54.4 %** (h=10).
- A 10 minutos, **más de la mitad de los casos** caen en el terreno donde el DL domina.

### El test ex-ante: la ventaja se confirma con información disponible al predecir

El análisis anterior agrupa por el cambio *realizado* del *headway* (conocido a posteriori), lo que lo vuelve **descriptivo**. Para convertirlo en una afirmación **operativa** —ejecutable en el momento de decidir— repetimos el corte usando un estratificador **ex-ante**: la **volatilidad de la ventana de entrada** (el desvío estándar de los últimos 12 minutos de *headway* observados), que se conoce *antes* de predecir. Partimos el test en terciles por esa volatilidad reciente y medimos quién gana en cada uno.

![Estratificación ex-ante de volatilidad](volatilidad-exante.png)

*Figura 3 — Δ MAE (DL − persistencia) por tercil de volatilidad **ex-ante** (el desvío de la ventana de entrada, conocido al momento de predecir), por corredor, una línea por horizonte. Por debajo de cero gana el DL. La pendiente descendente es el hallazgo: la ventaja del DL se acentúa cuanto más errático venía el servicio. A diferencia de la Figura 2 (régimen retrospectivo), aquí el régimen se asigna con información disponible a priori, así que el patrón es operativamente accionable.*

**El patrón clave se sostiene en los tres corredores y los tres horizontes (h = 3, 5, 10):** la ventaja del DL **crece monótonamente** con la volatilidad reciente —es máxima justo cuando el servicio venía más errático—, y en el tercil de **alta** volatilidad ex-ante el DL le gana a la persistencia en los nueve casos. En ese tercil alto a h = 10:

| Corredor | Persistencia | LSTM | Δ MAE |
|----------|--------------|------|-------|
| **E2** | 8.57 | 5.76 | **−2.81** |
| **E59** | 6.81 | 4.84 | **−1.97** |
| **E4** | 8.41 | 6.20 | **−2.21** |

**Matiz honesto por horizonte.** A h = 5 y h = 10 el DL gana en los **tres** terciles (incluido el calmo). A h = 3 la ventaja queda **concentrada** en el régimen volátil: el DL gana el tercil alto en los tres corredores, pero en el tercil calmo la persistencia todavía iguala o supera (E59 +0.16, E4 +0.35 de Δ MAE). Esto refuerza —no debilita— la lectura operativa: cuanto más corto el horizonte, más se confina la ventaja del DL a los momentos que importan (servicio errático), que es precisamente cuando la regla ex-ante lo activa.

*Datos: [`csv-multihorizon/exante_volatility_multihorizon.csv`](csv-multihorizon/exante_volatility_multihorizon.csv) (27 filas: 3 corredores × 3 horizontes × 3 terciles). La estratificación corre sobre las muestras con desvío de ventana computable —se descarta ~1 % con datos de entrada insuficientes—. La alineación con los residuos se verificó muestra a muestra: la persistencia y el objetivo reconstruidos desde los datos crudos coinciden con los residuos a precisión de punto flotante (Δ máx **observado** = 2.7e-6 en los nueve corredor×horizonte, muy por debajo de la tolerancia de 1e-2 del chequeo de alineación). Los valores por celda quedan registrados en [`csv-multihorizon/exante_alignment_multihorizon.csv`](csv-multihorizon/exante_alignment_multihorizon.csv) (9 filas: 3 corredores × 3 horizontes).*

**¿La señal ex-ante no es el régimen retrospectivo disfrazado?** La objeción natural es que la volatilidad reciente esté tan correlacionada con el cambio realizado del *headway* que el corte ex-ante reprodujera, encubierto, el régimen retrospectivo de la Figura 2 —y con él, su circularidad—. Lo medimos de frente. La correlación entre la σ de la ventana de entrada y el cambio realizado |y_real − persistencia| (la variable que *define* el régimen retrospectivo) es **moderada**: Pearson r ≈ 0.25 y Spearman ρ ≈ 0.21 (r² ≈ 0.06) en los nueve corredor×horizonte —la señal ex-ante explica **menos del 8 %** de la varianza del régimen retrospectivo—. La tabla de contingencia lo confirma: el tercil de **alta** volatilidad ex-ante es apenas **1.1–1.3× más propenso** a caer en el régimen alto retrospectivo que la media, y entre el **29 % y el 54 %** de ese tercil corresponde a ventanas que *no* resultaron volátiles —régimen estable o moderado—. En ese subconjunto la persistencia gana, **necesariamente**: el régimen estable está *definido* como un error de persistencia < 1 min, así que ningún modelo puede superarla ahí por construcción. Y eso es justamente lo que prueba el punto: si el corte ex-ante fuera un proxy encubierto del retrospectivo, su tercil alto no contendría entre un tercio y la mitad de casos donde la persistencia es imbatible. Los dos estratificadores identifican poblaciones distintas —y aun conteniendo esa fracción sustancial de casos favorables a la persistencia, el DL gana el tercil alto ex-ante **en agregado** en los nueve corredor×horizonte. La ventaja del DL bajo el corte ex-ante **no puede atribuirse a la circularidad** del estratificador a posteriori.

*Datos: [`csv-multihorizon/exante_correlation_multihorizon.csv`](csv-multihorizon/exante_correlation_multihorizon.csv) (9 filas: 3 corredores × 3 horizontes).*

**Por qué esto importa:** la regla pasa a ser **ejecutable en vivo**. Un operador, al momento de decidir, mira qué tan errático vino el servicio en los últimos minutos y —si venía movido— confía en el DL. La ventaja del DL **no** es un artefacto de definir el régimen a posteriori: se confirma con información disponible *antes* de predecir, y se **acentúa** (no se desvanece) al asignar el régimen ex-ante.

### El enrutador ex-ante: una sola política que empata o gana contra ambos modelos puros

La consecuencia natural de lo anterior es dejar de elegir entre persistencia y DL y **conmutar entre ellos** según el régimen ex-ante. Lo construimos y lo medimos: para cada corredor y horizonte, los terciles de volatilidad de entrada se **congelan en train+val**, la política —qué modelo usar en cada tercil— se **aprende sobre una porción retenida del test (60 %)** y el enrutador se **puntúa sobre el 40 % restante, disjunto**. El MAE reportado nunca informó la política.

| Corredor | h | Política (bajo·medio·alto) | Persistencia | LSTM | **Enrutador** | Δ vs. LSTM |
|---|---|---|---|---|---|---|
| **E2** | 1 | P·P·D | 4.242 | 4.270 | **4.176** | **−0.094** |
| **E59** | 1 | P·P·P | 2.822 | 3.162 | **2.822** | **−0.340** |
| **E4** | 1 | P·P·P | 2.858 | 3.372 | **2.858** | **−0.514** |
| **E59** | 3 | P·D·D | 3.901 | 3.723 | **3.668** | **−0.055** |
| **E4** | 3 | P·P·D | 4.460 | 4.419 | **4.280** | **−0.139** |
| **E2** | 3 | D·D·D | 5.748 | 4.855 | **4.855** | 0.000 |
| *E2/E59/E4* | *5 y 10* | *D·D·D (6 celdas)* | — | — | *= LSTM* | *0.000* |

**El resultado:** en las **12 celdas** (3 corredores × 4 horizontes) el enrutador nunca es peor que el LSTM ni que la persistencia. Ponderado por muestras, mejora **−0.10 min** sobre usar siempre el LSTM.

**La comparación que de verdad importa — y que rebaja el resultado.** "Mejor que usar siempre el LSTM" es una vara baja, y "mejor que usar siempre la persistencia" directamente es un hombre de paja: nadie propondría eso después de la Sección 3. La vara honesta es la **regla trivial que este documento ya derivó** —persistencia a h = 1, LSTM a h ≥ 3— que no usa volatilidad, no requiere calibración y se lee de la tabla de cruce. Contra esa vara:

| Política | vs. usar siempre el LSTM |
|---|---|
| Regla trivial (solo horizonte) | −0.082 min |
| Enrutador de volatilidad | −0.100 min |
| **Aporte real de la señal de volatilidad** | **−0.018 min** |

Es decir: **el 82 % de la ganancia la da conocer el horizonte**, que ya sabíamos. La estratificación por volatilidad ex-ante agrega **1.1 segundos**, y solo en **3 de las 12 celdas** (E2·h=1, E59·h=3, E4·h=3); en las otras 9 es bit a bit idéntica a la regla trivial. Ese incremento está por debajo del intervalo entre semillas de entrenamiento que reporta la Sección 4, así que **no debe leerse como una mejora de rendimiento**. Lo que el enrutador aporta no es error más bajo: es la demostración de que el mapa de regímenes de la Sección 5 es **ejecutable** —una política construible solo con información ex-ante nunca queda por detrás de ninguno de los modelos puros—.

**Los cuatro matices honestos, sin los cuales el número engaña:**

- **En 9 de las 12 celdas la política aprendida ES un modelo puro** (`D·D·D` ×7, `P·P·P` ×2). Decir que ahí "empata con ambos modelos puros" es decir que *es* uno de ellos. Solo **3 celdas** conmutan de verdad entre regímenes: E2·h=1, E59·h=3 y E4·h=3 — exactamente las mismas 3 donde la volatilidad le gana a la regla trivial.
- **A h ≥ 5 la ganancia es exactamente cero.** La política es `D·D·D` en las 6 celdas: el DL gana los tres terciles, así que el enrutador **se reduce a usar siempre el DL**. Visto en positivo, es una simplificación de despliegue: por encima de 5 minutos no hace falta ninguna lógica de conmutación.
- **Los niveles de MAE de esta tabla no son comparables con los del resto del documento**, porque se calculan sobre el ~40 % del test retenido para evaluación, no sobre el test completo.
- **La política se aprende sobre una sub-porción del test, no sobre train+val**, porque los kernels de Kaggle exportaron predicciones por muestra únicamente del split de test. Es una limitación real del pipeline: la disciplina anti-fuga se preserva (política y evaluación son disjuntas), pero un despliegue estricto calibraría la política sobre train+val.

**¿Y si la partición 60/40 fue una barajada afortunada?** Es la objeción natural, porque el corte es aleatorio. La repetimos con **seis semillas**: la política aprendida resulta **idéntica en las 12 celdas bajo las seis** —ninguna cambia de modelo ganador en ningún tercil—, el agregado ponderado se mueve entre **−0.099 y −0.100 min** frente a usar siempre el LSTM, y en **ninguna** de las 72 combinaciones el enrutador queda por detrás de un modelo puro (peor caso: 0.000000). La política es, por lo tanto, una propiedad de los regímenes de volatilidad y no del sorteo.

*Datos: [`csv-multihorizon/router_seed_sweep_multihorizon.csv`](csv-multihorizon/router_seed_sweep_multihorizon.csv) (72 filas: 3 corredores × 4 horizontes × 6 semillas). Generado por `src/build_router_seed_sweep.py`; verificado en `tests/test_router_seed_sweep.py`.*

**¿Y si el problema es el corte aleatorio en sí?** El barrido de semillas descarta una barajada afortunada, pero no la objeción más profunda: como las muestras son ventanas solapadas (comparten 11 de 12 pasos de entrada), un corte aleatorio deja ventanas casi **gemelas** a ambos lados, y la política podría estar "viendo" en calibración vecinas casi idénticas a las de evaluación. Para descartarlo repetimos el experimento con un **corte temporal por bloques**: el primer ~60 % del período de test calibra la política y el último ~40 %, disjunto en el tiempo, la evalúa —así las ventanas gemelas ya no cruzan la frontera salvo en una costura fina—. El resultado **se sostiene**: la política aprendida es **idéntica en las 12 celdas** a la del corte aleatorio, el enrutador vuelve a igualar al oráculo en las 12, y la señal de volatilidad aporta **−0.016 min** sobre la regla trivial (vs −0.018 del corte aleatorio), otra vez solo en 3 celdas. El 12/12 **no es un artefacto del solapamiento**: el mapa de regímenes generaliza del pasado al futuro dentro del test. La modestia también aguanta —la ganancia real sobre la regla trivial sigue siendo ~1 segundo—.

*Datos: [`csv-multihorizon/router_temporal_multihorizon.csv`](csv-multihorizon/router_temporal_multihorizon.csv) (12 filas: 3 corredores × 4 horizontes, `policy_frac = 0.6`, corte por marca temporal del objetivo). Generado por `src/build_router_temporal.py`; verificado en `tests/test_router_temporal.py`.*

Como control, comparamos el enrutador contra un **oráculo** que elige el mejor modelo por tercil *mirando* el conjunto de evaluación (cota superior no desplegable): coinciden **exactamente** en las 12 celdas (Δ = 0.0), y —como acaba de mostrar el corte temporal— esa coincidencia no depende del tipo de partición. La política retenida no deja nada sobre la mesa: el régimen ex-ante es lo bastante estable entre porciones del test como para que aprenderlo a ciegas equivalga a conocerlo.

*Datos: [`csv-multihorizon/router_multihorizon.csv`](csv-multihorizon/router_multihorizon.csv) (12 filas: 3 corredores × 4 horizontes; `policy_frac = 0.6`, `seed = 42`, 5 238 295 muestras de evaluación en total). Las predicciones del DL se emparejan **por posición** con el objetivo y la persistencia reconstruidos desde los datos crudos, así que cada celda pasa —como portón duro, antes de puntuar— la misma verificación muestra a muestra que la estratificación ex-ante: Δ máx observado = **2.7e-6** (tolerancia 1e-2), registrado por fila en las columnas `align_max_abs_diff` / `align_tolerance` del propio CSV. Generado por `src/build_router.py`; invariantes verificadas en `tests/test_router.py`, incluida la disyunción efectiva entre la porción de calibración y la de evaluación.*

---

## 6. Conclusión

> **El Deep Learning conviene para predecir el *headway* a horizontes operativos (≥ 3 min): a partir de ahí le gana a la persistencia, y la ventaja se concentra —y se acentúa— en los tramos de alta volatilidad del servicio. Esa ventaja se confirma con un estratificador ex-ante (la volatilidad reciente observada, conocida al momento de predecir), así que la recomendación es ejecutable en vivo, no solo a posteriori. A 1 minuto, o en servicio estable, la persistencia gana en los tres corredores. Todo esto medido en MAE sobre muestras idénticas; bajo RMSE no hay cruce.**

La conclusión madura **no** es "el DL reemplaza a la persistencia": cada modelo domina un régimen distinto. Y esa complementariedad es **explotable, no solo observable**: el enrutador ex-ante de la Sección 5 nunca queda por detrás de ninguno de los dos modelos puros en las 12 celdas corredor×horizonte, recuperando el terreno de la persistencia a h = 1 sin ceder la ventaja del DL a horizontes largos. Con una salvedad que conviene decir antes de que la diga un revisor: **el grueso de esa ganancia lo da conocer el horizonte**, no la volatilidad —la señal de volatilidad aporta −0.018 min sobre la regla trivial, y solo en 3 de las 12 celdas—.

Y una segunda salvedad, del mismo tenor y más incómoda: **todo lo anterior está establecido frente a la persistencia, no frente a un aprendiz fuerte.** Medido sobre muestras idénticas, el LSTM le gana al XGBoost nivelado con claridad solo en E59, empata dentro del ruido de semilla en E2, y pierde en E4 en los horizontes cortos; el test de signos sobre las 12 celdas no es significativo en ninguna de las dos poblaciones defendibles (7 de 12, *p* = 0.387 con multiplicidad emparejada; **6 de 12, *p* = 0.613 sobre objetivos distintos**). Lo que sí se replica en dos de los tres corredores es que **la ventaja relativa del DL sobre el aprendiz fuerte crece con el horizonte**, igual que crece frente a la persistencia —el mismo mecanismo de robustez a la volatilidad explica ambos gradientes—. Quien lea este trabajo buscando una justificación para desplegar una red neuronal en lugar de un *gradient boosting* no la va a encontrar acá, salvo en el corredor de más tráfico y al anticipar lejos.

El aporte del trabajo, entonces, no es "el DL gana" ni "nuestro enrutador mejora el error", sino **dónde y cuándo gana cada uno, y que esa frontera puede decidirse con información disponible al predecir**.

### Alcance y limitaciones

**Análisis deliberadamente no realizados (no por carencia, por redundancia).** La comparación pico vs. valle quedó subsumida en el análisis de volatilidad (Sección 5): las franjas pico coinciden con el régimen de alto cambio del *headway*, y la volatilidad caracteriza el régimen —cuánto se mueve el *headway*— mejor que la mera hora del día (y, validada ex-ante en la Sección 5, de forma operativa). Un análisis pico/valle separado mediría el mismo fenómeno con peor resolución.

**Limitaciones reales del estudio.**
- Evaluado sobre 3 corredores (E2, E59, E4) de una misma ciudad y una ventana de 5 meses; la generalización a otras ciudades queda por validar. E4 aporta **validez externa acotada a la escala de flota** —una línea independiente y mucho más chica (19 buses), no otra ciudad ni otro período— y replica los dos hallazgos centrales (ventaja del DL sobre la persistencia a h ≥ 3, y nulo aporte de la complejidad espacial). La validez externa **geográfica y temporal** sigue abierta. Los estudios de robustez por seed y por hiperparámetro (Sección 4) se realizaron sobre E2 y E59.
- La ventaja del DL **sobre el baseline aprendido (XGBoost)**, medida sobre muestras idénticas, **no es general**: es clara y creciente con el horizonte solo en E59; en E2 las dos familias son indistinguibles (los cuatro márgenes caen dentro del ruido de semilla); y en E4 el XGBoost gana en horizontes corto y medio, con el LSTM imponiéndose solo a h = 10. El test de signos sobre las 12 celdas es 7 de 12 (*p* = 0.387), **no significativo**. Frente a la persistencia, en cambio, el DL gana a h ≥ 3 en los tres corredores. Es decir: **la contribución del DL está establecida contra la persistencia, no contra un gradient boosting bien nivelado.**
- El contraste **DL vs XGBoost sí recibe ahora un test pareado por muestra**, sobre la población de objetivos distintos (`xgb_paired_significance.csv`). Requirió un kernel dedicado que reexportara las predicciones del XGBoost con la clave `(corredor, direction, horizonte, t, pair_rank)`: la exportación original llevaba solo `t`, que **no es única** (≈ 4.49 filas por `(t, direction)`), y sin `pair_rank` el emparejamiento muestra a muestra era imposible. No hizo falta reentrenar los kernels DL. Queda pendiente la limitación relacionada: la población multiplicidad-emparejada replica cada objetivo ≈ 4.5×, así que el *n* nominal sigue exagerando el *n* independiente y el test por muestra debe leerse con la advertencia sobre el *n* efectivo (ver más abajo).
- Los modelos espaciales (Conv, Transformer) no superaron de forma consistente al LSTM plano en estos datos (solo lo igualan o superan en celdas aisladas por márgenes < 0.01 min); queda abierto si lo harían con más buses por *snapshot*[^snapshot].
- El **enrutador** (Sección 5) se evalúa sobre el ~40 % del test retenido, y su política se calibra sobre el 60 % restante del test —no sobre train+val— porque los kernels de Kaggle exportaron predicciones por muestra solo del split de test. Política y evaluación son disjuntas, así que la ganancia reportada no está contaminada, pero los niveles de MAE del enrutador no son comparables con los del test completo.

**Amenazas a la validez, declaradas explícitamente.** Más allá del alcance de arriba, el diseño tiene supuestos que un lector crítico debe conocer:

1. **Objetivo censurado.** La winsorización recorta `delta_t_min` en el percentil 99 de *train* y aplica ese techo a **todos** los splits, incluido test. El MAE se mide entonces contra un objetivo truncado: el 1 % superior —los eventos de *bunching* más extremos, que la Sección 5 asocia al terreno del DL— queda comprimido contra ese techo. La métrica no refleja plenamente el desempeño en esos eventos extremos: mide contra un *headway* acotado, no contra el real sin recortar.
2. **Sin evaluación de origen rodante.** La prueba es una única ventana temporal fija de ~22 días. No hay *rolling-origin* ni múltiples ventanas de evaluación —la demanda estándar de un revisor de forecasting—, así que la estabilidad temporal del resultado queda sin cuantificar.
3. **Confusor en el período de prueba.** Febrero 2024 en Arequipa incluye Carnaval (12–13 feb), fechas de tráfico y demanda atípicos. El documento no caracteriza la composición del test ni declara si esas fechas están en `atypical_days.csv`; parte del error medido podría deberse a ese confusor y no a la dificultad intrínseca del pronóstico.
4. **n efectivo sobreestimado.** Las observaciones son ventanas solapadas que comparten 11 de 12 pasos de entrada y se agrupan por bus dentro del mismo *snapshot*; el residuo del DL además sobrecuenta cada objetivo ≈ 4.5× (ver la comparación DL-vs-XGBoost arriba). El n nominal exagera el n independiente, lo que **infla la significancia** de un test por muestra. Además, el IC por semilla (5 semillas, mismos datos, Sección 4) es un intervalo de **varianza de entrenamiento**, no de **muestreo**, y no debe leerse como lo segundo.
5. **Agregación de direcciones.** Las tablas con `direction="aggregate"` mezclan los sentidos −1 y +1 en un solo pozo. Donde el MAE difiere entre sentidos, el agregado es una mezcla con pesos distintos a los de la base pareada, así que **no** es el promedio de los dos sentidos.
6. **Sin estratificar por magnitud del *headway*.** Todo el error se reporta en minutos absolutos. Un error de 1 min sobre un *headway* de 3 min (33 %) y sobre uno de 15 min (7 %) no pesan igual operativamente, y esa heterogeneidad queda oculta en el promedio.
7. **Valor operativo afirmado, no modelado.** El trabajo argumenta que la ventaja del DL tiene valor operativo, pero no hay función de costo ni modelo de intervención que muestre que −1.5 min de MAE a h=10 cambie una decisión concreta de despacho. Es una hipótesis razonable, no un resultado medido.
8. **Comparaciones múltiples: corrección aplicada, con una consecuencia real.** El trabajo reporta **576 pruebas de hipótesis** (144 en `significance_multihorizon.csv` —72 filas × 2 tests— y 432 en `volatility_multihorizon.csv`). Una versión previa de este documento afirmaba que controlar el error de familia "sería cosmética"; **eso es falso y se corrige acá.** Aplicando **Holm-Bonferroni** sobre las 576: de 575 pruebas que pasaban α = 0.05, quedan 574, y **un verdicto se cae** — `SpatialConvLSTM · E4 · h=3 · MAE`, cuyo *p* de Diebold-Mariano pasa de 0.0392 a **0.0783**. Esa celda es precisamente una de las tres desviaciones declaradas en la Sección 4, así que deja de ser "significativa a p<0.05" y pasa a ser no concluyente. Lo que **sí** sobrevive: restringiendo la familia a las **54 celdas primarias** (h ∈ {3,5,10}, 108 pruebas), **ningún** verdicto se cae y las 54 siguen pasando α en ambos tests. El titular de la Sección 4 —el DL con menor error en 53 de 54 celdas, todas a p<0.05— es por lo tanto robusto a la corrección; lo que no lo es es la celda marginal de E4·h=3. Sigue pendiente el problema más profundo, declarado en el punto 4: el *n* efectivo está sobreestimado por las ventanas solapadas, y una varianza agrupada por día de servicio (en vez del HAC actual) es la corrección que falta.

**Trabajo futuro: enrutador de producción.** El enrutador de la Sección 5 demuestra que la complementariedad de regímenes es explotable con una política ex-ante: empata o gana contra ambos modelos puros en las 12 celdas. Lo que queda abierto es llevarlo a operación: reemplazar los terciles fijos por un **umbral de conmutación ajustado**, incorporar el **costo asimétrico de los errores de clasificación de régimen** (equivocarse hacia la persistencia en un tramo que se desestabiliza cuesta más que lo inverso), calibrar la política sobre train+val con predicciones DL de todos los splits, y validar la ganancia neta en una corrida en vivo.

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

[^dm]: **Test de Diebold-Mariano (DM)** — prueba estadística específica para comparar la precisión de dos modelos de pronóstico. Responde: ¿la diferencia de error medio entre los dos modelos es real o producto del azar? Aquí se usa su forma robusta (varianza Newey-West/HAC), que ajusta el cálculo para que la correlación entre los errores de minutos consecutivos no exagere la significancia.

[^wilcoxon]: **Test de Wilcoxon (pareado, de rangos con signo)** — prueba no paramétrica que compara dos modelos mirando las *medianas* de sus errores por rangos, sin asumir que los datos siguen una distribución normal. Complementa a Diebold-Mariano.

[^pvalor]: **p-valor** — probabilidad de observar una diferencia así de grande si en realidad **no hubiera** diferencia entre los modelos. Un p-valor < 0.001 significa "menos de 1 en 1000 de que sea casualidad" → la diferencia se considera estadísticamente significativa.

[^volatilidad]: **Régimen de volatilidad** — clasificación de cada predicción según cuánto cambia realmente el *headway* en ese momento, con cortes fijos en minutos: *estable* (cambio < 1 min), *moderado* (1–3 min) y *alto* (≥ 3 min). Los cortes son fijos (no cuantiles) a propósito: así se ve que la *proporción* de muestras en régimen alto crece con el horizonte. Permite ver dónde gana cada modelo en lugar de mirar solo el promedio global.

[^crossover]: **Crossover (cruce)** — el punto donde dos curvas se cruzan e invierten quién va ganando. Aquí marca el horizonte a partir del cual el DL pasa a superar a la persistencia (h ≥ 3 min en E59).

[^hiperparametros]: **Hiperparámetros** — los ajustes de configuración de un modelo que NO se aprenden de los datos sino que se fijan antes de entrenar (p. ej. cuántas capas, tamaño de la red, tasa de aprendizaje). Se eligen probando sobre el conjunto de validación.

[^dropout]: **Dropout** — técnica de regularización que, durante el entrenamiento, "apaga" al azar una fracción de las neuronas en cada paso para evitar que la red memorice el conjunto de entrenamiento (*overfitting*). Es uno de los hiperparámetros del mini-grid de sensibilidad (Sección 4); un valor de 0.2 apaga el 20 % de las unidades, 0.0 desactiva la técnica.

[^seed]: **Seed (semilla)** — el número que fija el azar del entrenamiento: pesos iniciales de la red, barajado de los datos y *dropout*. Mismo seed → entrenamiento idéntico y reproducible; distinto seed → un modelo ligeramente distinto. Entrenar con varios seeds y reportar media ± intervalo de confianza demuestra que el resultado es estable frente a ese azar, no producto de una inicialización afortunada.

[^xgboost]: **XGBoost** — biblioteca de *gradient boosting*: construye un conjunto de árboles de decisión donde cada árbol corrige el error del anterior. Es un modelo de ML *ajustado* (se entrena con datos), a diferencia de los baselines B0–B4 que son fórmulas fijas. Aquí se usa como baseline B5_XGB con la misma ventana de 12 pasos que ve el LSTM, para que la comparación sea justa.

[^snapshot]: **Snapshot** — una "foto" del estado de todos los buses del corredor en un mismo instante: el vector de *headways* de cada bus en ese momento. Los modelos espaciales miran cada *snapshot* para relacionar buses entre sí.

[^bunching]: **Bunching** — fenómeno en que dos o más buses que deberían ir espaciados terminan circulando casi juntos (uno "alcanza" al de adelante), dejando un hueco largo detrás. Es el principal síntoma de un servicio desestabilizado.
