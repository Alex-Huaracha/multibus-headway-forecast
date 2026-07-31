# Fuentes verificadas para el manuscrito

Relevamiento del **2026-07-29**, cinco frentes en paralelo. Este archivo es el
insumo de P3 (referencias) y la evidencia de P2 (reencuadre de contribuciones).

> **Regla de uso.** Cada entrada lleva su estado de verificación. Las marcadas
> `[SNIPPET]` o `[ABSTRACT]` **no se citan afirmando contenido específico** sin
> leer la fuente primero. Nada en este archivo fue inventado; lo que no se pudo
> recuperar está declarado como tal.

| Estado | Significado |
|---|---|
| `[TEXTO COMPLETO]` | PDF extraído y citado literal |
| `[ABSTRACT]` | Solo resumen, verificado literal |
| `[CROSSREF]` | Campos bibliográficos verificados, contenido no leído |
| `[SNIPPET]` | Contenido solo de resultados de búsqueda — **verificar antes de citar** |
| `[NO RECUPERADO]` | Bloqueado. **No citar.** |

---

## 1. El veredicto de novedad, en una tabla

| Pieza de nuestro argumento | ¿Nueva? | Quién llegó antes |
|---|---|---|
| Los pronósticos puntuales están sub-dispersos | **No** | Patton & Timmermann 2012 (teorema); Mayer & Yang 2022 |
| Empeora monótonamente con el horizonte | **No** | Patton & Timmermann 2012, Corolario 2 (es teoría, no hallazgo) |
| La sub-dispersión rompe umbrales de evento | **No** | Ravuri et al. 2021 (*Nature*); Hoffmann et al. 2018; **Petetin et al. 2022** (§0.5 — el más cercano) |
| Aparear error continuo con métrica categórica, y que nombren ganadores distintos | **No** | **Petetin et al. 2022**, como titular. Ver §0.5 — por esto se retiró nuestro titular viejo |
| Reportar AUC junto a métricas de umbral | **No** | Petetin et al. 2022 (fuera de transporte); Sun et al. 2021 (dentro) |
| El paradigma "predecir headway → umbral → bunching" falla | **No** | **Sun, Schmöcker & Nakamura 2021** — en transporte |
| Disociación RMSE/recall en detección de bunching | **No** | **Sun et al. 2021** |
| Crítica del punto de operación único | **No** | **Sun et al. 2021** |
| Reversión al puntuar sin umbral (ROC) | **No** | **Sun et al. 2021** |
| Maximizar F1 degenera a "marcar todo" | **No** | Lipton et al. 2014 (teorema) |
| **Medir la causa: razón de CV pronóstico/observación** | **Sí** | Nadie, en transporte |
| **Invertir `Z = 0.5/cv` del TCQSM sobre el pronóstico** | **Sí** | Nadie |
| **Recalibrar el corte en una ventana anterior disjunta** | **Sí** | Nadie en transporte; y es el caso de umbral **relativo**, que la prior art de hidrología/meteorología no cubre |
| **Precisión media (AP) en detección de bunching** | **Sí** | Nadie |

**Síntesis.** Novedoso como contribución de *mecanismo y reparación* para
umbrales de evento **relativos y auto-referenciales**. **No** novedoso como
principio general. Solo **parcialmente** novedoso dentro de transporte.

---

## 0. Los cuatro papers de *paywall*, leídos — 2026-07-29

PDFs en `docs/paper/papers/`. Texto extraído con el extractor de solo-stdlib; las
tablas de Yu et al. quedaron con los dígitos pegados y **no se parsearon**, así
que sus tasas base no se citan desde ahí. Todo lo demás es literal.

### 0.1 Mayer & Yang (2022) — **la amenaza se CONFIRMA**
`[TEXTO COMPLETO]` — doi:`10.1016/j.ijforecast.2022.03.008`, *IJF* 39(2):981–991.

La frase existe, literal:

> *"As MSE-optimized forecasts are **always underdispersed**, the common practice
> of using RMSE skill score for evaluation overrates the forecasts with lower
> dispersion. Such underdispersed forecasts are less discriminatory, have a
> higher type-2 conditional bias, and are suboptimal in terms of MAE."*

Y también: *"MSE leads to the inevitable underdispersion of the forecast"*, más la
dependencia de la pérdida que creíamos nuestra: *"The MAE-optimized forecasts are
also slightly underdispersed, but they have a lower type-2 conditional bias and
higher discrimination as compared to the MSE-optimized ones"*, y *"The
MSE-optimized power forecasts were shown to be more underdispersed than
MAE-optimized ones, which seems to be the general case in solar forecasting."*

Además atribuyen la dependencia del horizonte a **Vannitsem y Hagedorn (2011)**:
el post-procesamiento que reduce el MSE produce pronósticos más sub-dispersos a
mayor *lead time*, porque convergen a la media climatológica cuando cae la
correlación. **Es un segundo precedente de nuestra monotonía**, además de Patton
y Timmermann.

Cuantifican: *"The MSE-optimized forecasts are much underdispersed, as they only
capture <75% of the observed variance"*, y proponen calibrar la **razón de
varianza** en el dominio σ² < F < 1.

**Qué NO hacen** (conteo sobre el texto completo): `threshold` **0**, `detect`
**0**, `exceed` **0**, `AUC` **0**, `coefficient of variation` **0**. Cero
detección de eventos, cero umbrales. Y su objeto es **irradiancia solar escalar**
en siete estaciones SURFRAD: varianza **temporal de una serie escalar**, no
dispersión **transversal** entre componentes de un vector.

**Consecuencia: C1 se reformula.** No podemos reclamar el enunciado estructural.
Queda: (a) primera cuantificación sobre un **vector de *headways***, como CV
transversal; (b) la inversión del `Z = 0.5/cv` del TCQSM sobre el pronóstico;
(c) la consecuencia **sobre una regla de evento**, que ellos no tocan. Citarlos
de frente en la §II-B.

### 0.2 Santos et al. (2022) — **C2 sobrevive**
`[TEXTO COMPLETO]` — doi:`10.1093/comjnl/bxab045`, *The Computer Journal* 65(8).

Era el lugar más probable donde ya existiera nuestro barrido de umbral. **No está.**
`sweep` **0**, `calibrat` **0**, `AUC` **0**, `MCC` **0**.

Su umbral, literal: `BB = 1 si h < τ`, con **τ = h_programado / 4**. Y cuando no
hay GTFS: *"when the scheduled headway cannot be calculated and are not
available, which usually occurs when GTFS is out of date, a user-defined
threshold is utilized"* — usaron **τ = 5 min** absolutos.

Predicen **la ocurrencia** directamente (clasificación), no el *headway*.
Métricas: **Precisión, Recall y F-measure**. *Baselines*: Regresión Lineal,
Logística, SVM y RVM.

**Y traen un regalo:** una tabla de la literatura previa con las métricas de cada
trabajo — RMSE, Accuracy, Precision, Recall, MAPE, Specificity, Sensitivity,
F-measure. **Ninguno reporta AUC ni precisión media.** Es confirmación publicada,
dentro del subcampo, de nuestra afirmación de la §II-D.

### 0.3 Rezazada et al. (2024) — **el rango se VERIFICA**
`[TEXTO COMPLETO]` — doi:`10.1080/01441647.2024.2313969`, *Transport Reviews* 44(4).

Literal:

> *"**There is no single threshold value to define bunching events**, as it
> depends on the type of the service, time of the day, location, and service
> frequency. Earlier studies suggest using a constant threshold, which is a fixed
> time interval that separates consecutive vehicles. Feng and Figliozzi (2011)
> considered three minutes and Iliopoulou et al. (2020b) employed one minute.
> **Typically, this threshold ranges from 20 s to ¼ of the planned headways.**
> However, some recent works suggest a variable threshold, which considers the
> service type, real-time information, and passenger demand."*

El rango "20 s a ¼" queda verificado. **La atribución disputada a
"Moreira-Matias et al. (2015)" no aparece en este pasaje** — no propagarla.

Y un pendiente nuevo: mencionan **Gong et al. (2020)** con umbral **variable**
según tipo de servicio e información en tiempo real. Hay que revisar que no sea
una amenaza a C2 (variable por contexto operativo ≠ recalibrado contra la
distribución del pronóstico, pero conviene verificarlo).

### 0.4 Yu et al. (2016) — **la reconciliación, y nos AYUDA**
`[TEXTO COMPLETO]` — doi:`10.1016/j.trc.2016.09.007`, *TR-C* 72:45–59.

**Confirmado que son la familia que criticamos**, literal: *"the occurrence of bus
bunching can be detected by thresholding the predicted headway with the planned
bus schedule."*

**Su regla (Ec. 13) resuelve nuestro mismo problema, y de la misma manera.**
Pekín no tiene horario fijo: *"It cannot provide a fixed timetable for
passengers because most route schedules change over time... **Therefore, this
study uses the headway at the first stop as the "scheduled" headway.**"* La regla
es `BB = 1 si h_i < h_1/4`, con `h_1` el *headway* **observado** en la primera
parada de la misma corrida.

> **Esto mejora nuestra posición.** Sustituir el horario ausente por una
> referencia **observada del propio dato** es práctica establecida, del paper más
> citado del subcampo. Nuestra media del vector es la misma clase de sustitución
> con otro punto de referencia. Deja de ser "invención nuestra sin precedente" y
> pasa a ser "la misma sustitución que Yu et al., con otra referencia" — y la
> diferencia que importa queda nítida: **su referencia es observada y no se mueve
> con el pronóstico; la nuestra es predicha y sí se mueve.** Ahí está el
> mecanismo, expresado como contraste con un precedente en lugar de como
> peculiaridad nuestra.

**La reconciliación del >95 %**, verificada en Sun et al. (2021), texto limpio:

> *"Yu, Chen, Wu, Ma, and Wang (2016) used several well-developed algorithms to
> predict headway first then convert the result to binary bunching occurrence.
> 2min RMSE is obtained for headway and **99% sensitivity is realized for
> bunching in 2-stop-ahead prediction, but the performance deteriorates to 6min
> RMSE and 73% sensitivity for 5-stop-ahead prediction.**"*

Su titular es de **2 paradas de anticipación**, y **su propia sensibilidad se
degrada a 73 % a 5 paradas**. O sea: cualitativamente, nuestro hallazgo. Sus
métricas son *accuracy*, sensibilidad y especificidad — **sin precisión, sin AUC,
sin detector trivial**.

Las diferencias que hay que declarar al reconciliar: horizonte en **paradas**, no
en minutos; objetivo un **par de buses en la próxima parada**, no el vector
completo en un instante futuro; y referencia del umbral **observada**, no
predicha.

### 0.5 Petetin et al. (2022) — **el precedente más cercano, y es de acceso abierto**
`[TEXTO COMPLETO]` — doi:`10.5194/acp-22-11603-2022`, *ACP* 22:11603–11630. CC-BY.
*"Model output statistics (MOS) applied to CAMS O₃ forecasts: trade-offs between
continuous and categorical skill scores."*

Estaba marcado `[POR VERIFICAR ANTES DE CITAR]` y era el candidato a *scoop* más
serio que quedaba vivo. **Leído el 2026-07-30. Se confirma casi todo lo que se
sospechaba — y aun así C2 sobrevive, por un motivo estructural.**

**Lo que tienen, y hay que acreditarles sin regatear:**

- **El apareamiento continuo ↔ categórico, como titular:** *"there is a clear
  trade-off between the continuous and categorical skills scores… The quality of
  a MOS-corrected forecast assessed solely based on metrics like RMSE or PCC thus
  tells little about the forecast value."* Y el caso concreto: *"a MOS method can
  give the best RMSE and PCC, yet the poorest high O₃ detection skills. This was
  the case of the unweighted GBM method."*
- **La sub-dispersión, diagnosticada y cuantificada:** el pronóstico crudo tiene
  *"underestimated variability (nMSDB around −30 %)"*, y los métodos sofisticados
  quedan *"too smooth"*, con *"more difficulty in capturing the lowest and highest
  O₃ concentrations"*.
- **AUC reportada junto a métricas de umbral** (H, F, FB, SR, CSI, PSS, AUC).
- **Sensibilidad de las métricas categóricas al horizonte**, explícita: *"all
  categorical metrics show a similarly strong sensitivity to the lead time"*, con
  CSI cayendo de 0.27 a 0.14 entre D+1 y D+4.

> **Esto valida a posteriori haber retirado el titular viejo.** *"La métrica
> decide el ganador"* es, esencialmente, su conclusión — publicada en 2022, en
> una revista de acceso abierto y muy citada. Si lo hubiéramos sometido, se caía.

**Lo que NO tienen, y es donde queda C2:**

- **No recalibran el umbral. No pueden.** Sus cortes son **regulatorios**: 60 ppbv
  para el máximo diario de 8 h y 90 ppbv para el máximo horario, fijados por
  normativa de la UE. Un umbral legal no se reajusta contra la distribución de un
  modelo — la pregunta no existe en su marco. **Ahí está la diferencia
  estructural con nuestro caso**, y no es un tecnicismo: nuestro corte es
  **relativo y auto-referencial**, así que el denominador se mueve con el
  pronóstico. El suyo no se mueve nunca.
- **Su remedio va del otro lado.** Corrigen el pronóstico para que su
  distribución se parezca a la observada —*"the quantile mapping (QM) method aims
  at adjusting the distribution of the forecast concentrations to the
  distribution of observed concentrations"*—, no la regla de decisión. Es la
  alternativa a lo nuestro, y hay que nombrarla como tal en la §II-C: QM exige
  una distribución de referencia de observaciones; recalibrar el corte exige solo
  una ventana anterior.
- Sin MCC, sin precisión media, y **sin detector trivial**: su referencia es
  persistencia, que en su tarea tiene destreza real y no es un piso de azar.
- Fuera de transporte, y su objeto es una serie escalar por estación.

**Consecuencias.** (a) **C2 sobrevive** —el precedente de recalibración de umbral
sigue siendo Hoffmann et al. (2018), y ninguno de los dos cubre el caso
auto-referencial—. (b) **C3 se acota**: el apareamiento de AUC con métricas de
umbral **ya existe**, así que C3 no puede reclamarlo; lo que queda es precisión
media, MCC y el piso trivial, y el alcance declarado sigue siendo *la literatura
de bunching*, que la tabla de Santos et al. respalda. (c) **Petetin pasa a ser la
cita central de la §II-C**, por encima de Hoffmann.

---

## 1-bis. Tres correcciones a **nuestra propia metodología**

Estas no son sobre el encuadre: son errores de hecho que teníamos escritos.

### C1. Nuestro umbral **no es el estándar del campo** — es nuestro
`[TEXTO COMPLETO]` de las fuentes primarias donde se pudo.

La forma relativa que domina la literatura es una fracción del *headway*
**programado**, no de la media del propio vector:

| Regla | Fuente | Forma |
|---|---|---|
| ¼ del *headway* programado | Yu et al. 2016, doi:`10.1016/j.trc.2016.09.007` | relativa al horario |
| ¼ del *headway* programado | Moreira-Matias et al. 2016, doi:`10.1016/j.asoc.2016.06.031` | relativa al horario |
| **1 minuto absoluto** | Sun et al. 2021, doi:`10.1080/15472450.2020.1725887` | absoluta, justificada empíricamente |
| ½ del *headway* programado | TCQSM 2ª ed., Exhibit 3-30 | relativa al horario |
| Parada servida antes que su líder (rebase) | Diab, Bertini & El-Geneidy 2016, TRB 95th | topológica |
| Clustering no supervisado, sin umbral | Degeler et al. 2020, doi:`10.1007/s12469-020-00251-z` | ninguna |
| CV del *headway*, sin umbral de evento | Drabicki, Kucharski & Cats 2022, doi:`10.1007/s11116-022-10270-3` | ninguna |

**La forma "fracción de la media observada" no se encontró como definición
publicada de evento.** Aparece solo en descripciones de implementaciones de
proveedores CAD/AVL, y la única fuente localizada es un post de LinkedIn — no
citable, y que además **critica** la práctica por medir "baja adherencia al
*headway*" en lugar de *bunching*.

**Consecuencia.** No tenemos horario programado (GPS crudo, sin GTFS), así que la
media del vector es el sustituto disponible. Es defendible, pero hay que
declararlo como **sustitución nuestra, no herencia**. Y acota el alcance del
hallazgo a **umbrales relativos y auto-referenciales**: la forma auto-referencial
es justamente la que hace que el CV gobierne el resultado vía `Z = 0.5/cv`, y por
lo tanto la que maximiza el artefacto. Un corte absoluto también sub-dispararía,
pero por otro mecanismo, y no lo medimos.

### C2. Nuestro CV **no es** el `cvh` del TCQSM
`[TEXTO COMPLETO]` — TCQSM 2ª ed., Parte 3 Cap. 3, Ec. 3-7, pp. 3-47/3-48.

`cvh` = σ(*headway* real − programado) / media(*headway* **programado**).
El nuestro = σ(*h*)/media(*h*) sobre el vector observado.

Coinciden solo si el horario es constante **y** la media real lo iguala. **No se
puede citar la Ec. 3-7 como definición de lo que medimos.** Sí se puede citar la
escala de nivel de servicio (Exhibit 3-30), que es donde está la fuerza.

### C3. El CV **no es** "la métrica estándar en operación"
`[TEXTO COMPLETO]` de la presentación del seminario IBBG; artículo `[META]`.

**Trompet, Liu & Graham (2011)**, *TRR* 2216(1):33–41, doi:`10.3141/2216-04`.
Relevamiento de doce operadores del *International Bus Benchmarking Group*:

| Indicador en uso | Operadores |
|---|---|
| *Wait Assessment* | Barcelona TMB, Bruselas STIB, Milán ATM, Nueva York NYCT, París RATP, Singapur SMRT |
| *Service Regularity* | Lisboa Carris |
| *Excess Wait Time* | London Buses |
| Ninguno basado en *headway* | Dublín, LA Metro, Montreal STM, Sydney Buses |

**Ninguno usa coeficiente de variación.** Y su conclusión: de los cuatro KPI
probados, *"the methodology for excess wait time is the best option when the key
performance indicator should reflect the customer experience."*

Base de Osuna–Newell para el EWT: `E(AWT) = 0.5·E(h) + V(h)/(2·E(h))`,
`EWT = AWT − SWT`.

**Redacción defendible:** "la medida de fiabilidad que el TCQSM prescribe para
servicio de alta frecuencia (≤10 min), con una escala de nivel de servicio cuyas
bandas altas están definidas en términos de *bunching*" — y reconocer que los KPI
de operador están dominados por *excess wait time* y bandas de *wait assessment*.
**No** escribir "la métrica estándar en operación".

### Dato colateral que nos favorece
Las tasas base del campo van de **0.15 % a 17 %** (Sun et al. reportan las de Yu
et al. y Moreira-Matias et al.). La nuestra es **17–30 %**, o sea en el extremo
alto o por encima. Eso vuelve el argumento del piso trivial **más** filoso, no
menos: a mayor prevalencia, más alto el piso que el F1 regala.

---

## 2. Las tres que obligan a reencuadrar

### 2.1 Sun, Schmöcker & Nakamura (2021) — el scoop del dominio
`[TEXTO COMPLETO]` — manuscrito de autor, Kyoto University KURENAI, handle `2433/269552`.

Ya publicaron, literal:
- El paradigma: *"existing approaches that predict headways and then utilize the headway prediction for bunching prediction."*
- La disociación: a 10 paradas, sensibilidad **34.9–52.7 %** mientras *"in terms of MAPE and RMSE… evaluation metrics deteriorate gradually."*
- La compresión, observada: *"neither in 1- nor 10-stop-ahead prediction can these two methods perform favorably under the circumstance that the actual headway becomes extremely short and bunching is going to happen."*
- El punto de operación único: *"only one combination of sensitivity and specificity is derived, as headway prediction produces an exact value for each headway"*; *"Deterministic methods can only produce one combination of prediction performance which greatly limits its contribution to the real application."*
- La reversión sin umbral: grafican los puntos de operación de los regresores contra la curva ROC del clasificador (AUC 0.9922 → 0.9279 de 1 a 15 paradas).

**Qué nos deja.** Su etiqueta es un corte **absoluto de 1 minuto**, no una regla
relativa al propio vector. No miden CV ni dispersión. No tienen el argumento de
profundidad en la cola. **Su remedio es cambiar de clase de modelo
(regresión → logística), no recalibrar el corte.** Sin ventana de calibración
disjunta, sin modelos profundos, sin persistencia, sin F1, sin AP. Una ruta,
cinco días de test. Ocho citas y ningún trabajo que lo extienda.

### 2.2 TCQSM, Exhibit 3-30 — la aritmética ya estaba en un manual de 2003
`[TEXTO COMPLETO]` — *Transit Capacity and Quality of Service Manual*, 2ª ed.
(TCRP Report 100), TRB 2003, Parte 3 Cap. 3, p. 3-48, Exhibit 3-30.
Fuente primaria: `https://onlinepubs.trb.org/onlinepubs/tcrp/docs/tcrp100/Part3.pdf`

Literal:

> *"the coefficient of variation of headways can be related to the probability P
> that a given transit vehicle's headway hᵢ will be off-headway by more than
> one-half the scheduled headway h. This probability is measured by the area to
> the right of Z on one tail of a normal distribution curve, where **Z in this
> case is 0.5 divided by cvh**."*

**Corta para los dos lados y hay que decidir cómo se presenta.**

*En contra:* nuestro mecanismo está aritméticamente implicado por un manual de
2003. `Z = 0.5/cv` con CV real 0.79 da Z ≈ 0.63; con CV del pronóstico 0.16 da
Z ≈ 3.1 y P ≈ 0.1 %. Un revisor hostil deriva nuestro titular en una línea.

*A favor:* la escala de nivel de servicio del propio manual califica al
pronóstico como **LOS A, "service provided like clockwork"**, y a las
observaciones como **LOS F, "most vehicles bunched"**. Mismo corredor, mismo
instrumento, veredictos opuestos. Nadie aplicó `Z = 0.5/cv` al CV de un
**pronóstico**.

**Decisión:** asumir la aritmética, citar la fuente primaria, y reclamar novedad
exactamente donde está — la inversión, no la fórmula. Es más fuerte
retóricamente que un argumento ad hoc, porque usa la fuente de la regla contra
el mal uso de la regla.

**Resuelto de paso:** la procedencia del 0.5× es el TCQSM. La variante 0.25×
**no tiene origen académico** — solo literatura gris (NYC Comptroller, *Behind
Schedule*, 2025-04-10, sin cita). Cita limpia del 0.5× en uso:
Zhang, Xu, Lu & Fan, *Sustainability* 14(23):15583 (2022),
doi:`10.3390/su142315583` `[TEXTO COMPLETO]`.

### 2.3 Mayer & Yang (2022) — ✅ **resuelto: leído, y la amenaza era real**
`[TEXTO COMPLETO]` — doi:`10.1016/j.ijforecast.2022.03.008`,
*International Journal of Forecasting* 39(2):981–991. CC-BY.

**Ver §0.1 para el detalle.** La atribución de los *snippets* era literal, así
que la afirmación estructural es suya y se cita como enunciado previo. C1 quedó
reformulado y la §II-B del manuscrito ya está escrita sobre esa base.

---

## 3. Teoría de la sub-dispersión

| Fuente | ID | Estado | Qué establece |
|---|---|---|---|
| **Patton & Timmermann (2012)**, *JBES* 30(1):1–17 | doi:`10.1080/07350015.2012.634337` | `[TEXTO COMPLETO]` | **Nuestro anclaje más fuerte.** `V[Y] = V[Ŷ*] + E[e*²]` y Corolario 2: `V[Ŷ*_{t|t−hS}] ≥ V[Ŷ*_{t|t−hL}]` para `hS < hL`. La monotonía en el horizonte es **teorema**, no hallazgo |
| **Gneiting (2011)**, *JASA* 106(494):746–762 | doi:`10.1198/jasa.2011.r10138`, arXiv:`0912.0902` | `[TEXTO COMPLETO]` | Elicitabilidad: Bregman ⟺ media; GPL de orden α ⟺ cuantil α. **NO contiene ninguna afirmación de sub-dispersión** — verificado por grep del preprint completo. Citar solo por el funcional |
| Gneiting, Balabdaoui & Raftery (2007), *JRSS-B* 69(2):243–268 | doi:`10.1111/j.1467-9868.2007.00587.x` | `[ABSTRACT]` | "Maximizar sharpness sujeto a calibración". El pivote natural hacia pronóstico probabilístico |
| **Ravuri et al. (2021)**, *Nature* 597:672–677 | doi:`10.1038/s41586-021-03854-z`, arXiv:`2104.00954` | `[ABSTRACT]` literal | *"blurry nowcasts at longer lead times, yielding poor performance on more rare medium-to-heavy rain events."* Sub-dispersión + horizonte + daño sobre eventos por umbral, las tres en *Nature*. Su explicación es "lack of constraints", no elicitabilidad |
| Subich et al. (2025), ICML 2025 | arXiv:`2501.19374` | `[ABSTRACT]` | MSE causa suavizado por doble penalización; lo arreglan con pérdida armónica esférica. GraphCast: resolución efectiva 1250 km → 160 km |
| Bonavita (2024), *GRL* 51 | doi:`10.1029/2023GL107377`, arXiv:`2309.08473` | `[ABSTRACT]` | Modelos ML de clima producen espectros de energía suavizados; su ventaja en métricas deterministas es **parcialmente atribuible** al suavizado |
| Hoffmann, Menz & Spekat (2018), *ASR* 15:107–116 | doi:`10.5194/asr-15-107-2018` | `[TEXTO COMPLETO]` | **Nuestro mecanismo y nuestro arreglo, en clima, en 2018.** Identifican el percentil del umbral en la referencia y lo recalculan por modelo. Sin horizonte, sin métricas de detección, sin transporte |

### Dos citas de la prosa que no estaban en este archivo — verificadas 2026-07-30

Ambas aparecen en el manuscrito y no tenían entrada. **Verificadas contra la API
de Crossref**, no de memoria.

| Fuente | ID | Estado | Por qué está citada |
|---|---|---|---|
| **Vannitsem & Hagedorn (2011)**, *Meteorological Applications* 18(1):94–104 | doi:`10.1002/met.217` | `[CROSSREF]` — título, byline, volumen y páginas confirmados | Mayer y Yang le acreditan el agravamiento de la sub-dispersión con el *lead time*. **Citada de segunda mano**: no leímos el cuerpo, así que en §II-B se la nombra como atribución de Mayer y Yang y **no se le atribuye texto ni cifras** |
| **Diebold & Mariano (1995)**, *JBES* 13(3):253–263 | doi:`10.1080/07350015.1995.10524599` | `[CROSSREF]` | El test pareado de §I-D y §III-E. Existe un reimpreso de 2002 (doi:`10.1198/073500102753410444`, *JBES* 20(1):134–144); **citar el original de 1995** |

### Doble penalización (meteorología)

| Fuente | ID | Estado |
|---|---|---|
| Ebert (2008), *Met. Apps* 15(1):51–64 | doi:`10.1002/met.25` | `[CROSSREF]` — verificación *fuzzy*; pasajes de doble penalización de fuentes secundarias |
| Gilleland et al. (2009), *Wea. Forecasting* 24(5):1416–1430 | doi:`10.1175/2009WAF2222269.1` | `[CROSSREF]` — AMS devuelve 403 |
| Wernli, Hofmann & Zimmer (2009), *Wea. Forecasting* 24(6):1472–1484 | doi:`10.1175/2009WAF2222271.1` | `[CROSSREF]` — tiene la definición más limpia del mecanismo, pero **la redacción exacta viene de snippet: verificar antes de citar textual** |
| Roberts & Lean (2008), *MWR* 136(1):78–97 | doi:`10.1175/2007MWR2123.1` | `[CROSSREF]` — Fractions Skill Score |

### Inflación de varianza (downscaling) — debate **abierto**

| Fuente | ID | Estado | Postura |
|---|---|---|---|
| von Storch (1999), *J. Climate* 12(12):3505–3506 | doi:`10.1175/1520-0442(1999)012<3505:OTUOII>2.0.CO;2` | `[CROSSREF]` | **Contra** la inflación; a favor de aleatorización. Es una nota de 2 páginas — no sobrevenderla |
| Huth (2002), *J. Climate* 15(13):1731–1742 | doi:`10.1175/1520-0442(2002)015<1731:SDODTI>2.0.CO;2` | `[SNIPPET]` | **A favor** de la inflación sobre la aleatorización. Conclusión opuesta a von Storch |
| Bürger (1996), *Climate Research* 7:111–128 | doi:`10.3354/CR007111` | `[SNIPPET]` | *Expanded downscaling*: construir la varianza correcta desde el diseño |
| Maraun (2013), *J. Climate* 26(6):2137–2143 | doi:`10.1175/JCLI-D-12-00821.1` | `[SNIPPET]` | Restatement moderno; el tema seguía litigándose 14 años después |

> Que el debate esté **abierto** nos sirve: "cómo arreglar la sub-dispersión"
> sigue sin resolverse, y ahí hay lugar legítimo para plantar una contribución.

---

## 4. Métricas de detección con desbalance moderado (nuestra tasa base: 17–30 %)

### 4.1 La cita que convierte nuestra degeneración en teorema
**Lipton, Elkan & Naryanaswamy (2014)**, *ECML PKDD*, LNCS 8725:225–239.
doi:`10.1007/978-3-662-44851-9_15`, arXiv:`1402.1892` `[TEXTO COMPLETO]`

Literal:
> *"we demonstrate that given an uninformative classifier, optimal thresholding
> to maximize F1 predicts all instances positive regardless of the base rate."*

Su Teorema 1: el umbral óptimo es **la mitad del F1 máximo alcanzable**. Con
tasa base 30 % y modelo débil, F1 máximo ≈ piso trivial 0.46 → corte ≈ 0.23,
debajo de casi toda predicción. **Eso es exactamente nuestro "dispara el
99.99 %", y es teorema.** Justifica calibrar por MCC.

También: *"That F1 is asymmetric in the positive and negative class is
well-known"*, y que la selección de umbral por F1 es de alta varianza — *"some
thresholds converge to their true error rates while others have higher variance
and may be set erroneously."*

### 4.2 El piso trivial
**Flach & Kull (2015)**, *NIPS 28*:838–846. Sin DOI; ACM DL `10.5555/2969239.2969333` `[TEXTO COMPLETO]`

> *"the baseline to beat is the always-positive classifier rather than any random
> classifier. This baseline has prec = π and rec = 1"*

**La fórmula 2b/(1+b) no está impresa literal en ninguna fuente.** Se deriva en
un paso de ahí. Y Lipton et al. la instancian numéricamente: 0.67 con b=0.5,
0.18 con b=0.1 — coincide exacto. **Atribuir así, no reclamarla de un paper.**

**Boyd et al. (2012)**, *ICML* — arXiv:`1206.4667` `[TEXTO COMPLETO]`
*(orden de autores CONFLICTIVO entre dblp y otras fuentes: verificar el byline del PDF)*

Teorema 1: `p ≥ πr/(1 − π + πr)`. Y `AUCPR_MIN = 1 + (1−π)ln(1−π)/π`.

**Calculado a nuestras tasas base:** AUCPR_MIN = 0.090 con b=0.17; 0.107 con
b=0.20; 0.137 con b=0.25; **0.168 con b=0.30**. El piso de AP al azar es π.

> **Esto da vuelta la objeción a nuestro favor.** La alternativa PR-exclusiva
> tiene el **mismo** problema de piso trivial que le criticamos al F1 — y **peor
> a nuestra prevalencia** que a la de eventos raros. Recomiendan explícitamente
> dibujar la curva PR mínima en cada gráfico: hacerlo.

### 4.3 MCC sobre F1

| Fuente | ID | Estado | Qué aporta |
|---|---|---|---|
| Chicco & Jurman (2020), *BMC Genomics* 21(1):6 | doi:`10.1186/s12864-019-6413-7` | `[CROSSREF]` + render | *"F1 score is independent from TN"*; F1 no es simétrico al intercambiar clases, MCC sí. MCC = 0 es el valor esperado de un clasificador al azar |
| **Chicco, Tötsch & Jurman (2021)**, *BioData Mining* 14(1):13 | doi:`10.1186/s13040-021-00244-z` | `[CROSSREF]` + render | **La cita para nuestro MCC = 0:** el MCC *"is undefined whenever the confusion matrix has a whole row or a whole column filled with zeros"*, pero *"by simple mathematical considerations it is possible to cover such cases"* |
| Powers (2011/2020), *JMLT* 2(1):37–63 | arXiv:`2010.16061` | `[TEXTO COMPLETO]` | *"they ignore performance in correctly handling negative examples… and… fail to take account the chance level performance."* Y: *"a system that performs worse in the objective sense of Informedness, can appear to perform better under any of these commonly used measures"* |
| Boughorbel, Jarray & El-Anbari (2017), *PLOS ONE* 12(6):e0177678 | doi:`10.1371/journal.pone.0177678` | `[CROSSREF]` | **Clasificador de Bayes óptimo para MCC**, con prueba de consistencia. La cita de que calibrar por MCC es principiado y no ad hoc. **Leer el cuerpo antes de citar** |
| Koyejo et al. (2014), *NIPS 27*:2744–2752 | dblp `conf/nips/KoyejoNRD14` | `[ABSTRACT]` | Para métricas que son razones de combinaciones lineales de la matriz de confusión, el clasificador óptimo es un **umbral dependiente de la métrica**. Legitima "ajustar un umbral maximizando una métrica" |
| Itaya et al. (2025), *Stat. in Medicine* 44(1–2):e10303 | doi:`10.1002/sim.10303`, arXiv:`2405.12622` | `[ABSTRACT]` | IC asintóticos para MCC y para **diferencias pareadas de MCC**. Con nuestro contrato pareado, es el instrumento correcto para reportar IC sobre ΔMCC |

**Críticas al MCC — el cuadro honesto:**
- Zhu (2020), *Pattern Recognition Letters* 136:71–80, doi:`10.1016/j.patrec.2020.03.030` `[ABSTRACT]`: *"MCC deteriorates seriously when the dataset in classification are imbalanced."* No se leyó el cuerpo, no se sabe qué niveles probó.
- Luque et al. (2019), *Pattern Recognition* 91:216–231, doi:`10.1016/j.patcog.2019.02.023` `[SNIPPET]`: el MCC no es totalmente invariante a la prevalencia.
- **Chicco & Jurman (2023)**, *BioData Mining* 16:4, doi:`10.1186/s13040-023-00322-4` `[CROSSREF]` + render: argumentan que el MCC debe **reemplazar** al ROC-AUC. **Es el ataque más filoso disponible y viene de nuestra propia cita.** Respuesta: responden preguntas distintas — AUC/AP son discriminación sin umbral, MCC es resumen del punto de operación.

### 4.4 ROC contra PR: la objeción del desbalance, desactivada

| Fuente | ID | Estado | Por qué nos sirve |
|---|---|---|---|
| Davis & Goadrich (2006), *ICML*:233–240 | doi:`10.1145/1143844.1143874` | `[TEXTO COMPLETO]` | Dicen *"large skew"*, no nuestro régimen. Y su **Teorema 3.2 es una equivalencia**: dominancia en ROC ⟺ dominancia en PR. El AUC no puede invertir una conclusión de dominancia |
| Saito & Rehmsmeier (2015), *PLOS ONE* 10(3):e0118432 | doi:`10.1371/journal.pone.0118432` | render | **Su brazo "desbalanceado" es 1:10 (π ≈ 0.09)** — más extremo que nuestro 17–30 %. No transfiere hacia abajo |
| **McDermott et al. (2024)**, NeurIPS 2024 | arXiv:`2401.06091` | `[ABSTRACT]` | *"AUPRC is not generally superior in cases of class imbalance"*; la creencia *"is often made without citation, misattributed to papers that do not argue this point."* **El escudo principal.** DOI de proceedings NO verificado |
| Li (2024), *PLOS ONE* 19(12):e0316019 | doi:`10.1371/journal.pone.0316019`, arXiv:`2408.10193` | render | 156 escenarios, prevalencia 0.08–0.83: el AUC tiene la menor varianza; el F1 tiene relación **monótona creciente con la prevalencia**. Y un caso publicado con **F1 = 0.475 contra MCC = 0.042 y AUC = 0.524** para un modelo al azar — nuestro argumento, en otra revista |
| Fawcett (2006), *PRL* 27(8):861–874 | doi:`10.1016/j.patrec.2005.10.010` | `[TEXTO COMPLETO]` | AUC = P(positivo al azar rankeado sobre negativo al azar) = Wilcoxon. Y: *"Comparing model performance at a common threshold will be meaningless"* entre escalas distintas. **La invariancia monótona NO figura como teorema etiquetado en ninguna fuente: derivarla en una línea del rank identity, no atribuirla** |
| Hanley & McNeil (1982), *Radiology* 143(1):29–36 | doi:`10.1148/radiology.143.1.7063747` | `[CROSSREF]` | Fuente original de AUC = Wilcoxon/Mann-Whitney |

---

## 5. Vecinos en el dominio

| Fuente | ID | Estado | Rol |
|---|---|---|---|
| **Usama & Koutsopoulos (2025)** | arXiv:`2510.03121` | `[TEXTO COMPLETO]` | **Nuestra mejor cita motivadora.** ConvLSTM sobre el vector de headways de una línea de metro. Grepeado: **cero** ocurrencias de "bunch", "threshold", "classif", "F1", "recall", "smooth", "underestimat", "variance". Solo RMSE/MAE. Grupo de primera línea. Su lista de ausencias es nuestra lista de contribuciones |
| **Jiao, Shen & Zhang (2023)**, IEEE ICITE | doi:`10.1109/icite59717.2023.10733869` | `[ABSTRACT]` | **LIABILIDAD.** LSTM → umbral, reclama *"accurately identify 89% of bus bunching events"*. **Su umbral es NO VERIFICADO.** Reconciliar antes de someter |
| **Yu et al. (2016)**, *TR-C* | doi:`10.1016/j.trc.2016.09.007` | `[TEXTO COMPLETO]` | ✅ **Reconciliado — ver §0.4.** El titular es a **2 paradas**; su propia sensibilidad cae a **73 % a 5 paradas** (vía Sun et al. 2021). Y su Ec. 13 usa el *headway* **observado** de la primera parada como sustituto del horario ausente: precedente directo de nuestra sustitución. Citado en §II-A |
| Li, Yang & Wang (2025) | arXiv:`2509.06979` | `[TEXTO COMPLETO]` | Zirui Li, Bin Yang, Meng Wang. 31 ago 2025. **Preprint sin venue.** Grepeado: "headway", "bunching", "dispersion", "coefficient of variation" **no aparecen**. Su villano es la **normalización** (curable por arquitectura); el nuestro es la **pérdida** (estructural). Afirmaciones opuestas sobre tratabilidad = nuestra novedad. Su borde expuesto: *"overly stable and indistinguishable outputs"* — citar y neutralizar |
| Liu, Wu, Wang & Long (2022) | arXiv:`2205.14415` | `[ABSTRACT]` | Origen del término *over-stationarization*. **Venue NeurIPS 2022 NO VERIFICADO** — el registro de arXiv no tiene journal-ref |
| Boudabbous et al. (2026) | arXiv:`2601.18521` | `[ABSTRACT]` | Montreal STM. LSTM le gana a transformers **18–52 % con 77× menos parámetros**. Target = **retraso**, no headway; sin ConvLSTM. Mata nuestro nulo espacial como contribución |
| Rodrigues (2022) | arXiv:`2203.02954` | `[ABSTRACT]` | Baseline de patrón semanal + regresión lineal iguala al SOTA espacio-temporal. La cita canónica que sepulta el nulo espacial. **Título verificado 2026-07-30** y es mejor de lo que suponíamos: *"On the importance of stationarity, strong baselines and benchmarks in transport prediction problems"* — el paper trata explícitamente el punto que nos interesa |
| Chen, Cheng, Jin, Trépanier & Sun (2022) | arXiv:`2206.06915` | `[ABSTRACT]` verbatim | ⚠️ **Corregido 2026-07-30.** La descripción anterior decía "mezcla gaussiana sobre vectores de headway" y era imprecisa: su objetivo es el **tiempo de viaje**. Literal: *"we concatenate the link travel time vectors and the headway vector from a pair of two adjacent buses as a new augmented variable and model it with a constrained Multivariate Gaussian mixture distributions."* El *headway* entra como parte de la variable aumentada, no como objetivo. Guangzhou, veinte tramos. Sigue sirviendo: el remedio probabilístico **ya existe** en pronóstico de buses |
| Yu, Wu, Chen & Ma (2016), *IEEE T-ITS* | doi:`10.1109/tits.2016.2620483` | `[ABSTRACT]` | Predicción **probabilística** de headway con RVM |
| Zhang, Xu, Lu & Fan (2022), *Sustainability* 14(23):15583 | doi:`10.3390/su142315583` | `[TEXTO COMPLETO]` | Cita limpia del 0.5× en uso |
| Manibardo, Laña & Del Ser (2021), *IEEE T-ITS* | arXiv:`2012.02260` | `[ABSTRACT]` | ⚠️ **El cruce por horizonte NO está confirmado en su abstract.** Podría ser arXiv:`2004.08170`. **No citar el cruce a este paper sin leerlo completo** |

---

## 6. Pendientes bloqueantes antes de someter

| # | Qué | Por qué |
|---|---|---|
| ~~V1~~ | ✅ **Cerrado 2026-07-29.** Mayer & Yang leído — §0.1 | La afirmación era literal. C1 reformulado, §II-B escrita sobre esa base |
| ~~V2~~ | ✅ **Cerrado 2026-07-29.** Rezazada et al. leído — §0.3 | Rango "20 s a ¼" verificado. Citado en §II-A |
| V3 | **Reconciliar Jiao et al. (89 %)** con el colapso de nuestro LSTM | La mitad de Yu et al. ya está cerrada (§0.4: su titular es a 2 paradas y cae a 73 % a 5). Falta Jiao, doi:`10.1109/icite59717.2023.10733869`, todavía `[ABSTRACT]` con umbral no verificado. **No se cita en §II-A hasta leerlo** |
| V4 | Confirmar el venue de arXiv:`2205.14415` en los proceedings de NeurIPS 2022 | El registro de arXiv no lo trae |
| ~~V5~~ | ✅ **Cerrado 2026-07-30.** Byline de Boyd et al. fijado desde arXiv: **K. Boyd, V. Santos Costa, J. Davis, D. Page** | El orden discrepaba entre dblp y otras fuentes; la página de arXiv lo resuelve |
| V6 | Verificar la redacción textual de Wernli et al. (2009) | La definición que tenemos viene de snippet |
| V7 | Pre-empt "¿por qué no probabilidades de excedencia?" | Sun et al. **proponen ese paper ellos mismos** como trabajo futuro. La pregunta ya está planteada en la literatura del dominio |
| ~~V8~~ | ✅ **Cerrado 2026-07-29.** Santos et al. leído — §0.2 | Sin barrido de umbral: C2 sobrevive. Y su tabla de la literatura previa sostiene la §II-D |
| ~~V9~~ | ✅ **Cerrado 2026-07-29.** Yu et al. a texto completo — §0.4 | Su Ec. 13 y la sustitución del horario están verificadas de primera mano. Citado en §II-A |
| V11 | **Gong et al. (2020)**, umbral **variable** — surgió de Rezazada | Verificar que "variable según contexto operativo" no colisione con C2 (recalibrado contra la distribución del pronóstico). Prioridad media |
| V10 | Kim, Moon & Kim (2024), *TRR* 2679(1):1662–1679, doi:`10.1177/03611981241255907` | `[NO RECUPERADO]` (SAGE 403). Probablemente tiene una crítica actualizada del CV como medida de regularidad — nos toca directo |

**No citar bajo ninguna circunstancia** (no recuperados): SSRN `abstract_id=6880258`;
la versión de journal de arXiv:2601.18521 (PII `S1383762126001293`);
arXiv:`2510.25045`, `2606.08587`, `2512.00916`, `2206.09821` — solo nivel de
listado. Powers (2015) arXiv:`1503.06410`: solo abstract, extracción del cuerpo
falló. Powers (2012) doi:`10.1109/ICIST.2012.6221710`: solo búsqueda.
Narasimhan et al. (2014): aparece solo en la lista de referencias de Flach & Kull.
Hurley, Mendeley Data doi:`10.17632/4dn7drrtgc`: dataset verificado, **el paper
no está indexado**.

---

## 7. Calibración del venue (IJACSA)

Ocho papers leídos completos, Vol. 15 (2024) a Vol. 17 (2026).

| | |
|---|---|
| Con test de significancia pareado | **0 de 8** |
| Con corte temporal real declarado | **1 de 8** (y es de demanda hotelera, no de tráfico) |
| Con declaración de disponibilidad de datos/código | **0 de 8** |
| Con sección de limitaciones sustantiva | 3 de 8 |
| Referencias | 26–46, mediana ≈41 |
| Extensión | 9–11 pp incluyendo referencias |

Guías: ≤10 pp de cuerpo, plantilla SAI a dos columnas estilo IEEE, revisión
doble ciego con ≥3 revisores. Tasa de aceptación declarada ~15 %. Indexado en
Scopus (CiteScore 3.4, Q3 2025), WoS ESCI. Cargo de publicación **£800**.
Ciclo del CFP: someter 25 ago 2026 → notificación 15 sep → publicación 30 sep.

> **Advertencia táctica.** El encuadre natural —"nuestro modelo no le gana
> uniformemente a la persistencia"— **se lee como resultado débil** en un corpus
> donde la norma es una fila en negrita ganando. Hay que invertirlo: el *paired
> audit*, la winsorización en train, los hashes congelados, los terciles ex-ante
> y el rolling origin van **al frente como contribuciones metodológicas
> explícitas**, porque acá son diferenciadores, no higiene. Y agregar la
> declaración de disponibilidad de datos/código que ninguno de los ocho tiene.
