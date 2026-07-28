# Auditoría adversarial: hallazgos, correcciones y pendientes

**Fecha:** 2026-07-26 · **Alcance:** metodología y resultados (el manuscrito queda fuera)

Este documento registra el resultado de una auditoría adversarial de cuatro lentes
independientes (rigor metodológico, validez estadística, trazabilidad
afirmación→evidencia, y encaje editorial en IJACSA) sobre la metodología y los
resultados del trabajo. Registra **qué se encontró, qué se corrigió, y qué queda
pendiente**, con la evidencia y el costo de cada pendiente.

Sirve como punto de retomada: cada pendiente indica qué necesita (datos locales,
Kaggle, o GPU) y cuánto cuesta.

> ### Estado al 2026-07-27: 9 de 13 pendientes cerrados
>
> La remediación completa está mergeada a `main` (commit `a98d508`). Este
> documento se conserva como **registro de lo que la auditoría encontró**, no
> como lista de trabajo — esa vive en [`pendientes.md`](./pendientes.md).
>
> | # | Estado | Dónde se cerró |
> |---|---|---|
> | 1 Dirección del Wilcoxon | ✅ | `significance_clustered.wilcoxon_directional`. Confirmado: a h=3 el LSTM gana la media y pierde la mediana en E4 y E59 |
> | 2 432 p-valores circulares | ✅ | Eliminados. `volatility_effect_table` ya no reporta inferencia |
> | 3 Sensibilidad de winsorización | ✅ | El techo es inerte: ningún signo cambia, margen < 0.01 min |
> | 4 Semillas ConvLSTM/Transformer | ⬜ | Requiere GPU. Sigue declarado como límite |
> | 5 Métrica vectorial | ✅ | **Refutó el claim central**: la persistencia gana la detección de *bunching* en las 12 celdas, por hasta 253× |
> | 6 Varianza cluster-robusta | ✅ | Agrupada por día de servicio. El *n* efectivo son 22 días; tres verdictos se caen |
> | 7 HLN y piso de lag | ✅ | `hln_scale`, piso `lag ≥ h−1`, referencia *t* |
> | 8 Contradicciones del repo | ✅ | GNN y E58 corregidos, más una tercera contradicción encontrada (ruta de credenciales) |
> | 9 Enrutador sobredimensionado | ✅ | Comprimido a dos párrafos. Re-medido: solo paga a h=3 |
> | 10 Composición del test / Carnaval | ⬜ | Abierto. Lo absorbe parcialmente el *rolling origin* (#12) |
> | 11 Bandera de día atípico | ✅ | **Eliminada**, no parcheada — contrato C3 |
> | 12 Origen rodante | 🔨 | Infraestructura construida (`ROLLING_FOLDS`, 16 kernels emitidos). Falta correr |
> | 13 Objetivos contados 2.4–5.4× | ✅ | Contrato C1: una muestra por `(empresa, sentido, instante, horizonte)` |
>
> **Lo que la auditoría no anticipó:** el pendiente 5 no era una métrica faltante
> sino el claim central del paper sin sostén. Su cierre reencuadró el aporte
> entero — ver [`resultados/documento-resultados.md`](./resultados/documento-resultados.md) §5.

---

## 1. Puntajes de entrada

| Lente | Puntaje | P(aceptación tal cual) |
|---|---|---|
| Rigor metodológico | 52/100 | 15 % |
| Validez estadística | 52/100 | 20 % |
| Resultados y contribución | 72/100 | 5 %* |
| Preparación editorial | 22/100 | <1 % |

\* El 5 % refleja que **no hay sección de resultados escrita**, no la calidad de la
evidencia. Condicionado a traducir fielmente el contenido con las figuras
corregidas, el mismo auditor estima ~70 %.

**Calibración del venue.** Medido sobre 8–9 artículos recientes de IJACSA: el
testing formal de significancia aparece en **2 de 8**, y una sección de amenazas a
la validez en **0 de 8**. El aparato de rigor de este trabajo está por encima de la
mediana de la revista, y ese es el activo de posicionamiento — conviene que el
abstract lidere con el **protocolo de evaluación**, no con el resultado DL vs
persistencia.

**Estimaciones de aceptación (IJACSA).** Tal como está hoy: **<1 %** (rechazo de
escritorio: el manuscrito no existe). Escrita solo la prosa faltante: **55–65 %**.
Con las cinco acciones pre-sometimiento más la declaración de IA generativa y el
PDF anonimizado: **80–85 %**.

---

## 2. Corregido

### 2.1 El claim DL vs XGBoost comparaba poblaciones distintas

**El defecto.** «El LSTM le gana al XGBoost en las 8 celdas de E2 y E59
(*p* = 0.004)» comparaba MAE calculados sobre conjuntos de muestras diferentes: el
XGBoost sobre **todas** las filas de test con predicción no nula
(`src/baselines/harness.py:184-219`), el LSTM sobre la **población de ventanas**
(sin arranque en frío, con cada objetivo replicado por *slot* de anclaje). El sesgo
de encuadre que el propio trabajo mide en **0.28–0.53 min** es más grande que 7 de
los 8 márgenes reclamados (+0.05 a +0.41), y el test de signos lo heredaba entero.

**El obstáculo técnico.** `harness.py:135` calcula los residuos correctamente y
después descarta la clave en `.select(XGB_RESIDUAL_COLUMNS)`: conserva `t`, que
**no es única** — hay **4.49 filas por `(t, direction)`**, una por `pair_rank`. El
docstring de `harness.py:71` que declara `t` clave de join es incorrecto.

**La corrección, sin reentrenar.** Kernel nuevo `20-xgb-paired-export` que
reexporta las predicciones de test por muestra con
`(corridor, direction, horizon, t, pair_rank)`, reconstruyendo los residuos desde
`B5FitResult.predictions` (que conserva `pair_rank`) para no tocar `harness.py`
—ese módulo se inlinea verbatim en NB10 y NB16—. El reajuste **reprodujo
exactamente** la búsqueda de hiperparámetros congelada (las 12 celdas coinciden
con `xgb_search_config_multih.csv` a precisión float completa), así que los números
nuevos son comparables con los viejos sin cláusula de reconciliación.

**El resultado.** Δ MAE (XGBoost − LSTM), Δ positivo = gana el LSTM:

| Corredor | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|
| E2 | +0.036 | +0.005 | −0.005 | −0.000 |
| E59 | +0.006 | +0.061 | +0.130 | +0.242 |
| E4 | −0.275 | −0.206 | −0.051 | +0.139 |

Cobertura del join **100.000 %** en las 12 celdas; portón de alineación en
2e-6…3e-6 contra `ALIGN_TOL = 1e-2`.

**Tests de signos, ambas poblaciones, ninguno significativo:**

| Grupo | Multiplicidad emparejada | Objetivos distintos |
|---|---|---|
| E2 + E59 | 6/8 (*p* = 0.145) | 4/8 (*p* = 0.637) |
| Las 12 celdas | 7/12 (*p* = 0.387) | 6/12 (*p* = 0.613) |
| E4 aislado | 1/4 (*p* = 0.938) | 2/4 (*p* = 0.688) |

**Tesis resultante, de tres niveles:** el LSTM gana con brecha creciente **solo en
E59** (se replica idéntico en ambas poblaciones); en **E2 es indistinguible** (los
ocho márgenes dentro o al borde del ruido de semilla, ±0.046); en **E4 el XGBoost
gana los horizontes cortos** y el LSTM solo a h=10.

**Nota de honestidad.** La población emparejada es la más favorable al DL de las
dos; reportar solo esa habría sido selección sesgada, por eso van ambas.

**Artefactos:** `xgb_paired_dl_metrics.csv`, `xgb_paired_vs_reported_audit.csv`,
`xgb_paired_significance.csv`, `xgb_vs_lstm_signtest.csv` (regenerado).

### 2.2 Cuatro defectos en la Figura 1

1. **XGBoost obsoleto.** `consolidated_multihorizon.csv` y `curva-degradacion.png`
   se regeneraron por última vez en `d3df5c6` (17-jul), **tres días antes** de que
   entrara el XGBoost nivelado en `4098d32`. Al regenerar cambian 72 filas y **solo
   del baseline `B5_XGB`**; el XGBoost obsoleto era peor en todas las celdas, es
   decir **favorecía al DL**.
2. **El pie sobreafirmaba la significancia.** Estaba hardcodeado como «todas
   significativas p<0.001», falso en tres formas: son 51 de 53 a p<0.001, 53 a
   p<0.05, y una celda va en dirección contraria. Ahora se **calcula** desde
   `significance_multihorizon.csv`.
3. **`_load_significance` ignoraba `dl_better`** (el defecto más grave, no
   reportado por los auditores): evaluaba solo `dm_p` y `wilcoxon_p`, así que las
   **10 celdas donde la persistencia gana significativamente** pasaban el filtro
   como «significativas» y quedaban sin anillar — la figura las mostraba como
   victorias del DL. Entre ellas `SpatialTransformer/E4/h3/MAE`, dentro del rango
   que el pie declaraba como victorias del DL.
4. **El encuadre contradecía el titular.** Los paneles son agregados (el único
   encuadre en el que existen los baselines formulaicos y el ajustado), pero la
   comparación canónica es pareada. Las **4 celdas** donde el signo se invierte se
   marcan `B1≠`, y el pie instruye no leer el cruce desde la figura.

Ahora hay tres verdictos distinguibles: `B1` (gana persistencia), `B1≠` (gana
persistencia *y* el agregado invierte el signo), `⊘ ns` (no significativa).

### 2.3 El almacén local de residuos estaba rancio

**Todo** el árbol bajo `docs/resultados/residuos-multihorizon/` era **previo al
arreglo de winsorización**: los objetivos de test venían **sin recortar**, topados
en el techo crudo de 30 min del corte de viaje en vez del p99 de train. Afectaba a
las **nueve** combinaciones modelo × corredor.

| Corredor | Techo p99 de train | Máximo en el CSV rancio | Filas sobre el techo (h=3) |
|---|---|---|---|
| E2 | 28.467923 | 30.000000 | 6 080 (1.02 %) |
| E59 | 27.996949 | 30.000000 | 14 322 (0.66 %) |
| E4 | 29.098441 | 30.000000 | 5 626 (1.05 %) |

**Consecuencia:** cualquier recálculo local (significancia, volatilidad, ex-ante,
router, auditoría pareada) producía números equivocados en silencio.

**Los números publicados en `documento-resultados.md` son correctos** — salieron de
la corrida buena. Lo que estaba rancio eran las copias locales.

**Corregido:** los 24 outputs de kernel se rebajaron de Kaggle (el árbol viejo se
apartó en `residuos-multihorizon.stale/`, no se borró). Verificación: **36 de 36**
combinaciones archivo × corredor limpias, cero filas sobre el techo y cero en 30.0.
Guardián de regresión en `tests/test_residual_freshness.py`.

> **Gotcha para el guardián:** usar tolerancia **1e-4**, no 1e-6. Los CSV hacen
> round-trip por float32, así que las filas que están *en* el techo se leen ~1e-6
> por encima; con 1e-6 aparecen ~900 falsos positivos por archivo de E4.

**Bonus:** aparecieron los residuos de **h=1**, que no existían localmente. La
afirmación titular del cruce es a h=1 y no se podía reproducir desde el repo.

**Descubrimiento colateral:** `04-preprocessing` publica un `headways_E4.parquet`
**byte-idéntico** al que pinean NB17/18/19, así que el dato de E4 se alcanza sin
involucrar NB16 nunca.

### 2.4 Cuatro afirmaciones sin sostén

1. **Ranking entre arquitecturas.** Se declaraba al LSTM plano «el mejor» en E59 y
   E4, adjudicando celdas con márgenes de 0.017–0.038 min cuando el desvío por
   semilla de la *misma* arquitectura es 0.001–0.037. **El ranking cae dentro del
   ruido de entrenamiento** y se retiró. El resultado **nulo** (la complejidad
   espacial no mejora el pronóstico) se mantiene: ese sí se sostiene y se replica
   en los tres corredores.
2. **«La corrección por comparaciones múltiples sería cosmética».** Falso.
   Holm-Bonferroni sobre las **576** pruebas (144 en `significance_multihorizon` +
   432 en `volatility_multihorizon`) tira **un verdicto**:
   `SpatialConvLSTM/E4/h3/MAE` pasa de *p* = 0.0392 a **0.0783**, y esa celda es una
   de las tres desviaciones declaradas. **Lo que sí sobrevive:** restringida la
   familia a las **54 celdas primarias** (h ∈ {3,5,10}, 108 pruebas), **ningún**
   verdicto se cae y las 54 siguen pasando α en ambos tests — el titular de «53 de
   54, todas a p<0.05» es robusto.
3. **«La varianza por semilla del LSTM acota a toda la familia DL».** No se sigue:
   coincidir en la estimación puntual no dice nada sobre la dispersión entre
   semillas de las otras dos arquitecturas, que corrieron con **una sola**. Se
   retiró y se declaró el límite de cobertura (solo LSTM, solo E2 y E59).
4. **«Bajo RMSE el DL gana en todos los horizontes (9 de 9)».** Cierto en
   estimación puntual, pero **una de las nueve no es significativa**
   (`SpatialTransformer/E2/h1`, Wilcoxon *p* = 0.409) y el texto lo omitía.

Cada retiro deja constancia de la afirmación previa en lugar de borrarla.

---

## 3. El hallazgo mayor: el horizonte no es un horizonte temporal

### 3.1 El defecto

`make_window_index` (`src/data/windowing.py:154-188`) corta las ventanas por
**índice posicional** y **nunca verifica que las posiciones consecutivas sean
minutos consecutivos**. La lista de marcas temporales de un *slot* es discontinua
en cada frontera de día, cada corte de viaje (`GAP_CUT_SECONDS = 30*60`) y cada
caída de flota. Por lo tanto **el horizonte nominal es un desplazamiento de filas,
no de tiempo.**

Ejemplo concreto:

| | |
|---|---|
| Ventana sana | 08:00, 08:01 … 08:11 → objetivo a las **08:21**. El horizonte es de verdad 10 min. |
| Ventana rota | 08:00 … 08:11, y la siguiente posición de la lista es **el día siguiente a las 07:00** → el «objetivo a 10 minutos» está **23 horas después**. |

No hay ningún test en el repo que verifique contigüidad.

### 3.2 Medición

`src/build_contiguity_audit.py` → `contiguity_audit_multihorizon.csv`.
Δ = MAE(LSTM) − MAE(persistencia); Δ negativo = gana el LSTM.

| Celda | % roto | h real medio | h real p99 | Δ todas | Δ solo sanas |
|---|---|---|---|---|---|
| E2 h=1 | 7.4 % | 3.97 | 18 | +0.037 | **+0.119** |
| E2 h=3 | 16.5 % | 12.4 | 96 | −0.905 | −0.809 |
| E2 h=5 | 21.9 % | 20.7 | 457 | −1.191 | −1.093 |
| E2 h=10 | **30.1 %** | **41.0** | 809 | −1.571 | −1.486 |
| E59 h=1 | 3.7 % | 2.51 | 6 | +0.343 | +0.371 |
| E59 h=3 | 7.6 % | 7.63 | 34 | −0.177 | −0.122 |
| E59 h=5 | 9.7 % | 12.8 | 110 | −0.466 | −0.387 |
| E59 h=10 | 13.4 % | 25.9 | 505 | −1.094 | −1.011 |
| E4 h=1 | 3.9 % | 2.49 | 7 | +0.522 | +0.558 |
| **E4 h=3** | 8.6 % | 7.79 | 54 | −0.040 | **+0.047** ⚠ |
| E4 h=5 | 11.6 % | 13.6 | 119 | −0.553 | −0.441 |
| E4 h=10 | 16.9 % | 28.0 | 575 | −1.416 | −1.309 |

### 3.3 Consecuencias

**Una afirmación del titular se debilita.** `documento-resultados.md` afirma que
«el patrón es limpio y sin excepciones: la persistencia gana a h = 1 en los tres
corredores; el LSTM gana a h ≥ 3 en los tres». Sobre ventanas donde el horizonte
es efectivamente el declarado, **E4 · h=3 se da vuelta** (−0.040 → **+0.047**): gana
la persistencia. **El cruce de E4 se corre de h=3 a h=5.**

Atenuante: el documento ya calificaba esa celda como «esencialmente un empate que
recién se despega a h ≥ 5», así que es un refinamiento de algo ya sospechado.

**Lo que aguanta.** A **h ≥ 5 las seis celdas mantienen la ventaja del DL**,
encogiéndose entre 7 % y 20 %: E2 h=10 pasa de −1.571 a −1.486; E59 de −1.094 a
−1.011; E4 de −1.416 a −1.309. Siguen siendo −1.0 a −1.5 min, operativamente
material.

**El h=1 se refuerza.** En E2 el margen de la persistencia **se triplica** (+0.037
→ +0.119). La afirmación de que la persistencia gana a 1 minuto queda más firme.

**Dos cosas peores de lo esperado.** (a) **E2 es el corredor más contaminado**, no
E4: a h=10 el **30 %** de sus muestras son ventanas rotas y el horizonte real medio
es de **41 minutos** para un nominal de 10. (b) **La contaminación crece con el
horizonte** en los tres corredores (7 % → 30 % en E2), así que **parte de la
pendiente de la curva de degradación es contaminación creciente**, no dificultad
creciente. Los dos efectos no son separables desde los números agregados.

### 3.4 Decisión tomada

Se descartó arreglar de raíz (agregar el predicado de contigüidad, regenerar los 6
*builders* y **reentrenar** las seis familias en GPU: semanas, y todos los números
se recalculan). Se adoptó **reportar el subconjunto de ventanas sanas como
resultado primario, con el conjunto completo como análisis de sensibilidad** —
offline, sin GPU, con la tabla de arriba publicada.

### 3.5 Pendiente de redacción

- [ ] La tabla de cruce pasa a reportar **ventanas sanas como primario**, con el
      conjunto completo al lado.
- [ ] Corregir «el LSTM gana a h≥3 en los tres corredores» → **gana a h≥3 en E2 y
      E59, y a h≥5 en E4**.
- [ ] Amenaza a la validez nueva y explícita sobre el horizonte posicional, con los
      porcentajes de contaminación por celda.
- [ ] Anotar o regenerar la curva de degradación sobre ventanas sanas.

---

## 4. Pendientes

### 4.1 Se hacen localmente, sin GPU

| # | Pendiente | Evidencia / detalle |
|---|---|---|
| 1 | **Dirección del Wilcoxon a h=3.** No es un arreglo: es un **hallazgo sin escribir**. En E59 y E4 el LSTM gana el MAE medio pero **pierde contra la persistencia en la mayoría de las muestras individuales** (mediana **+0.158** y **+0.183**, *win rate* 47.4 % y 46.6 %). El documento lo tabula como victoria a p<0.001 usando un Wilcoxon **bilateral**, que no distingue dirección. La lectura honesta —el DL cambia muchas pérdidas chicas por pocas ganancias grandes— encaja con la Sección 5 de volatilidad y es **más interesante** que «el DL gana». | `significance.py:296-307` fija `dl_better` desde la **media**; el Wilcoxon bilateral se reporta al lado como si corroborara |
| 2 | **432 p-valores circulares.** El análisis retrospectivo de volatilidad define el régimen como `\|y_real − persistencia\|` —el error de la persistencia— y después testea el diferencial de pérdida. **Condiciona sobre la variable dependiente**, así que el resultado está forzado aritméticamente. Borrar las columnas de p-valor; conservar los MAE por régimen, que son la parte informativa y no circular. | `volatility.py:48-64`, `:199-216` |
| 3 | **Sensibilidad de winsorización** (amenaza #1). El techo p99 recorta el 1 % superior, que son los eventos de *bunching* extremos donde se reclama la ventaja del DL. Medir contra objetivos sin recortar y reportar si la brecha se ensancha o se cierra. Es el punto más atacable y el más barato de blindar. | Residuos en disco; `dataset-manifest.md` ya andamia la nota |
| 4 | **Ranking entre arquitecturas: cobertura de semillas.** El nulo espacial es sólido, pero solo el LSTM tiene 5 semillas, y solo en E2/E59. Correr 5 semillas para ConvLSTM y Transformer cerraría el hueco (requiere GPU) o basta con dejar el límite declarado (ya hecho). | `multiseed_ci_multihorizon.csv` cubre solo LSTM |
| 5 | **Métrica vectorial ausente.** Se reclama «predicción del vector completo de headways» pero todas las métricas son MAE/RMSE **escalares agrupados**: no hay perfil de error por posición, ni índice de regularidad, ni detección conjunta de eventos de *bunching*. Tal como está evaluado es indistinguible de N pronósticos escalares. O se agrega una métrica vectorial, o se reformula el claim. | `evaluation/metrics.py` expone solo `mae()` y `rmse()` |
| 6 | **Varianza cluster-robusta.** El HAC de Newey-West se aplica sobre un orden **slot-major**, no temporal, así que las réplicas exactas del mismo objetivo quedan a decenas de miles de posiciones de distancia — fuera del lag de truncación ⌊n^{1/3}⌋. La corrección correcta agrupa por **día de servicio**. Con clustering por identidad de objetivo, los SE se inflan 1.5–4.1× y una victoria declarada se evapora (ConvLSTM·E4·h3: p 0.005 → 0.266). | `significance.py:186-188`, `:282-293` |
| 7 | **HLN y piso de lag.** Falta la corrección de muestra pequeña de Harvey-Leybourne-Newbold y el piso `lag ≥ h−1`, y se usa referencia normal en vez de *t*. No cambia conclusiones a estos *n*, pero el texto presenta ⌊n^{1/3}⌋ como si fuera el estándar DM, y no lo es. | `significance.py:186-187`, `:207` |
| 8 | **Contradicciones del repo.** `README.md:3` describe el proyecto como «usando GNN+LSTM», que **nunca se construyó**. Y `propuesta.md:21` / `README.md:10` / `CLAUDE.md` declaran **cuatro** corredores (2, 4, 58, 59) cuando **E58 no aparece en ningún resultado**. | `rg E58 documento-resultados.md` → cero coincidencias |
| 9 | **El enrutador está sobredimensionado.** Aporta **−0.018 min** (≈1.1 s) sobre la regla trivial, en 3 de 12 celdas, por debajo del intervalo entre semillas. Y 9 de las 12 políticas son degeneradas (`D·D·D` ×7, `P·P·P` ×2), así que «empata con ambos modelos puros» ahí significa que **es** uno de ellos. Comprimir a dos párrafos como demostración de ejecutabilidad. | `router_multihorizon.csv`; ya declarado en el texto |

### 4.2 Requiere Kaggle (descarga, sin GPU)

| # | Pendiente | Detalle |
|---|---|---|
| 10 | **Composición del test y Carnaval.** El test son 22 días de febrero 2024, que incluye Carnaval (12–13). No se caracterizó. Un revisor argumentará que el resultado es propiedad de esa ventana. Se puede caracterizar la composición desde los parquets, pero **para declarar si esas fechas están marcadas hace falta `atypical_days.csv`**, que no está local: `uv run kaggle kernels output alexhuaracha/02-eda-corridors`. SHA-256 esperado `2054245c…`. | Amenaza #3, declarada y abierta |

### 4.3 Requiere GPU (reentrenamiento)

| # | Pendiente | Detalle |
|---|---|---|
| 11 | **Bandera de día atípico: tres defectos.** (a) el umbral se calcula sobre los **152 días incluyendo test**; (b) es un agregado de **día completo**, así que a las 08:00 no se puede conocer el total de registros del día — es información del futuro dentro del test; (c) `context_features.py:166-190` **descarta el `empresaid`**, así que un día marcado para una empresa marca a **los tres corredores**. Está horneada en todas las corridas: es un input requerido y hash-pineado. | `build_notebook_02.py:595-627`, `context_features.py:109-131` |
| 12 | **Evaluación de origen rodante.** Una única ventana fija de ~22 días. Es la demanda estándar de un revisor de forecasting. Mínimo: 3–4 orígenes con ventana expansiva. | Amenaza #2, declarada y abierta |
| 13 | **Cada objetivo se cuenta 2.4–5.4×.** Las ventanas se anclan por `(empresaid, direction, pair_rank)` pero el objetivo es el *snapshot* completo, así que un objetivo se emite una vez por *slot* de anclaje. El MAE reportado es una media **ponderada por densidad de flota**, y la ponderación se concentra en los *snapshots* más cargados — justo el régimen de *bunching* donde se reclama la ventaja. Anclar una vez por `(empresaid, direction, start_ts)` lo arreglaría. | `windowing.py:158-188`, `build_notebook_11.py:492-512` |

---

## 5. Camino a IJACSA

Ordenado por impacto sobre la probabilidad de aceptación.

### A. De <1 % a 55–65 % — el manuscrito

~7 000 palabras en inglés. 3–4 semanas. El contenido existe en
`documento-resultados.md`; falta traducir y reformatear a estructura de *journal*
(abstract, introducción, métodos —la sección de preprocesamiento hay que
escribirla de cero, el documento la excluye a propósito—, resultados, discusión).

**Riesgo declarado:** `manuscrito.md:3` planea escribir todo en español y traducir
al final. La traducción tardía es donde los trabajos no-nativos acumulan los
problemas de lenguaje que disparan pedidos de revisión. **Decisión del autor:**
arreglar primero el contenido, escribir una sola vez al final, para no mantener dos
documentos en paralelo.

### B. De 65 % a 80–85 %

1. **Referencias: de 10 a 25–32**, ≥10 in-domain, ≥40 % de 2022–2026. La mediana de
   IJACSA es 28. Leer primero las **dos fuentes que el propio esqueleto marca como
   no leídas**: Singh & Sahu 2022 (*WIREs*, DOI 10.1002/widm.1457) y arXiv:2401.17387
   (regímenes markovianos para tiempo de viaje de buses) — el vecino más cercano a
   la contribución de regímenes. Agregar además PatchTST e iTransformer (ya se los
   invoca en el plan de la Discusión sin estar en la bibliografía), TMS-GNN
   (*TR-C* 2025), literatura de *bus holding*, y 1–2 artículos de transporte
   publicados en IJACSA. Suavizar la afirmación de brecha de `manuscrito.md:76`, que
   hoy sostiene un enunciado sobre todo un subcampo con **tres** citas in-domain.
2. **Los huecos de encuadre** — pendientes 5, 8 y 9 de la sección 4.1.
3. **Paquete de sometimiento.** Figuras en inglés (las tres están en español, ejes
   y leyendas incluidos); la de 6 paneles hay que partirla o hacerla *full-width*.
   Plantilla SAI, **PDF anonimizado** (es doble ciego y el manuscrito lleva
   *placeholder* de autores), ≤5 MB, ≤9 coautores. **Declaración obligatoria de IA
   generativa**: IJACSA trata el incumplimiento como mala conducta académica, con
   retiro retroactivo de trabajo publicado.
4. **Las dos costuras baratas** — pendientes 3 y 10.

### Caveats del venue, para decidir con los ojos abiertos

IJACSA **no está en DOAJ**; el editor (SAI) aparece en la lista de Beall (congelada
desde ~2017 y discutida, pero todavía consultada por comités). Volumen de
megajournal: ~1 539 artículos en 2024, ~1 346 en 2025. La tasa de aceptación
anunciada del 15 % es autodeclarada y difícil de reconciliar con ese volumen.
Scopus y WoS-ESCI activos, JIF 1.1, SJR 0.327, Q3, APC £800. Conviene confirmar que
el venue cuenta para quien deba contarlo antes de pagar.

---

## 6. Lo que no se movió en toda la auditoría

**La tesis central contra la persistencia.** A h = 10: E2 −1.57 min (−23.3 %),
E59 −1.09 (−20.7 %), E4 −1.42 (−20.9 %). Significativa en los tres corredores,
replicada, con gradiente monótono por horizonte, concentrada en alta volatilidad y
validada con un estratificador ex-ante. Cuatro auditores adversariales la atacaron
y no la movieron. Restringida a ventanas sanas se encoge 7–10 % y **sigue en pie**.

De 24 afirmaciones auditadas contra los CSV, **18 salieron plenamente soportadas**.
El auditor de resultados: «no pude romper la comparación central contra
persistencia, y lo intenté con ganas».

**Lo que la auditoría también reconoció como fortaleza real:** la auditoría pareada
sobre muestras idénticas que llevó a los autores a **retirar sus propias
afirmaciones agregadas**; el Diebold-Mariano con HAC más Wilcoxon; los IC sobre 5
semillas; el mini-grid con el duplicado degenerado detectado y declarado; el
XGBoost nivelado; la anti-circularidad ex-ante; el router validado bajo corte
aleatorio **y** temporal; el portón de hashes de entrada que falla cerrado; la
winsorización *train-only* aplicada a todos los *splits* con test dedicado; las
compilaciones deterministas; ~880 tests; y ocho amenazas a la validez declaradas.
Ese paquete está **por encima de la mediana de IJACSA**, y hoy está enterrado en un
documento en español escrito para lector no especialista.

---

## 7. Reproducir estos hallazgos

```bash
uv sync

# Refrescar los residuos por muestra (los locales pueden estar rancios)
#   24 outputs de kernel; ver la sección 2.3 para la verificación
uv run kaggle kernels output alexhuaracha/11-lstm-multihorizon-h3 -p <dir>

# Auditoría de contigüidad de ventanas (sección 3)
uv run python -m src.build_contiguity_audit

# Recalificación pareada DL vs XGBoost (sección 2.1)
uv run python -m src.build_xgb_paired_metrics
uv run python -m src.build_xgb_vs_lstm_signtest

# Figura 1 y tabla maestra (sección 2.2)
uv run python -m src.build_degradation_curve

# Guardián de rancidez + regresiones de la auditoría
uv run pytest tests/test_residual_freshness.py tests/test_xgb_paired_metrics.py -q
```

⚠️ Correr la suite completa reescribe las *notebooks* en su lugar. Después de
`uv run pytest`, verificar `git status` y revertir con `git checkout -- notebooks/`
lo que no se pretendía cambiar.

**Faltan localmente y solo viven en Kaggle:** `atypical_days.csv` y el GPS crudo
(`clean_gps.parquet`). Los tres parquets procesados **sí** están, y sus SHA-256
coinciden con los hashes congelados en las *notebooks* — las mediciones de este
documento corren sobre los mismos bytes que vio el entrenamiento.
