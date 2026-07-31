<!--
  ANDAMIAJE DEL MANUSCRITO — borrador en español.
  Flujo: se escribe y pule TODO en español → traducción al inglés al final (IJACSA es en inglés).
  Las notas entre bloques ">" con la etiqueta [ANDAMIAJE] son guías para escribir; BORRARLAS antes de traducir.
  Orden de escritura sugerido: Related Work → Métodos → Resultados (ya existen) → Discusión → Conclusión → Introducción → Abstract.

  REESTRUCTURADO 2026-07-29. El encuadre anterior ("la métrica decide el ganador":
  el MAE escalar y la fidelidad vectorial nombran ganadores opuestos) se RETIRÓ.
  Era falso: dependía por completo de un umbral mal transportado. Ver la nota de
  retractación en docs/resultados/documento-resultados.md §5 y el inventario de
  literatura en docs/paper/fuentes-verificadas.md.

  Insumos:
    docs/resultados/documento-resultados.md   — resultados y cifras
    docs/paper/fuentes-verificadas.md         — ~55 referencias con estado de verificación
-->

# Título

> [ANDAMIAJE] Debe nombrar: (1) el objeto = detección de *bunching* a partir de pronóstico de *headway*, (2) el ángulo = el umbral no es transportable al espacio del pronóstico, (3) dato real. Se afina al final.
>
> Provisional: **"Un pronóstico aplanado parece ciego sin serlo: transportabilidad del umbral en la detección de *bunching* de buses a partir de datos GPS reales"**
>
> Alternativa más conservadora, si se quiere sonar menos polémico: **"Compresión de dispersión y recalibración de umbrales en la detección de *bunching* a partir de pronósticos de *headway*"**

**Autores** · Afiliación · Contacto

---

## Abstract

> [ANDAMIAJE] SE ESCRIBE AL FINAL. 150–250 palabras, un párrafo. Estructura: (1) el paradigma "predecir *headway* → umbralizar → alarma" y su falla documentada, (2) qué medimos que nadie midió, (3) datos, (4) 3–4 resultados con números, (5) el aporte. Sin citas ni abreviaturas indefinidas.
>
> Números candidatos para el abstract: CV 0.16 predicho contra 0.79 real (36/36 celdas); el corte de 0.5× recuperado por la persistencia en 11/12 celdas contra 0.58×–0.91× del aprendiz; reversión del veredicto sin umbral a h=10 en 3/3 corredores y 3/3 ventanas; el detector trivial supera al ganador declarado en 5/12 celdas; el colapso es 110× peor bajo la convención absoluta del campo.

*(pendiente)*

**Keywords** — *(pendiente — 5–7: bus bunching detection, headway forecasting, forecast under-dispersion, decision threshold calibration, evaluation methodology, AVL/GPS data)*

---

## I. Introducción

> [ANDAMIAJE] SE ESCRIBE CASI AL FINAL. Cuatro movimientos, en este orden.

### A. Contexto y motivación
> [ANDAMIAJE] 1–2 párrafos. Qué es el *headway*, qué es el *bunching*, y por qué al operador le importa que le avisen ANTES: una alarma sirve solo si deja tiempo de intervenir. Cerrar con el paradigma operativo: entrenar un modelo que prediga *headways*, definir la regla de alarma sobre la predicción, despachar.

### B. El problema
> [ANDAMIAJE] Este párrafo es el que cambió. NO decir "el subcampo no usa *baselines naive* ni tests de significancia" — eso es plomería y además es un encuadre débil. Y NO decir "nadie se dio cuenta": Sun, Schmöcker y Nakamura (2021) ya diagnosticaron el síntoma y hay que citarlos en las primeras dos oraciones.
>
> El problema, en la forma que sí se sostiene:
>
> La familia dominante de trabajos predice el *headway* con un regresor y después compara la predicción contra un umbral de *bunching* definido sobre observaciones. Sun et al. (2021) mostraron que esa familia rinde mal en detección aunque su error de regresión se degrade suavemente, y lo atribuyeron al hecho de que un pronóstico puntual entrega **un único punto de operación no ajustable**. Su remedio fue **cambiar de clase de modelo**: pasar a predicción probabilística y construir curvas ROC.
>
> Lo que quedó sin hacer es lo anterior a esa decisión: **nadie midió por qué el umbral falla, ni probó recalibrarlo.** Un pronóstico puntual está sub-disperso —minimizar una pérdida puntual apunta a un funcional central de la distribución condicional— así que el vector predicho es más parejo que el real. Un corte calibrado en el espacio de las observaciones cae, sobre ese vector comprimido, mucho más adentro de la cola. La alarma no suena, y el diagnóstico natural —"el modelo no ve el *bunching*"— es falso.
>
> La consecuencia para quien evalúa: **un veredicto de detección obtenido trasplantando un umbral relativo mide el umbral, no el modelo.** Y para quien despliega: la línea de trabajo se cierra por un motivo que no tiene nada que ver con lo que el modelo sabe, cuando el arreglo es un escalar.

### C. Contribuciones
> [ANDAMIAJE] Rankeadas por defensibilidad, con la evidencia al lado. Cada una nombra explícitamente qué NO reclama, porque la mitad del argumento anterior estaba tomado y conviene que el revisor vea que lo sabemos.

- **C1 (mecanismo, medido).** Cuantificamos la causa de la falla: la **compresión de dispersión** del pronóstico, medida como razón de coeficientes de variación del vector de *headways*. El LSTM predice un corredor con CV ≈ 0.16 cuando el real es ≈ 0.79, con sesgo negativo en **las 36 celdas** de corredor × horizonte × ventana de prueba y deterioro monótono con el horizonte. Y la convertimos en profundidad de cola usando `Z = 0.5/cv`, **la propia relación que el TCQSM establece** para ligar el CV del *headway* a la probabilidad de desvío: sobre las observaciones da Z ≈ 0.63, sobre el pronóstico Z ≈ 3.1. La escala de nivel de servicio del manual califica al mismo corredor como **LOS A, "service provided like clockwork"** por el pronóstico y **LOS F, "most vehicles bunched"** por lo observado.
  *No reclamamos* que la sub-dispersión de pronósticos puntuales sea un hallazgo nuevo, y hay que decirlo de frente porque **está publicado como enunciado general**: Mayer y Yang (2022) escriben que *"MSE-optimized forecasts are always underdispersed"*, cuantifican el déficit como razón de varianza (< 75 % de la varianza observada), establecen que los pronósticos optimizados en MAE quedan **menos** sub-dispersos que los de MSE, y atribuyen el agravamiento con el horizonte a Vannitsem y Hagedorn (2011). Patton y Timmermann (2012, Cor. 2) prueban esa monotonía como teorema. Tampoco reclamamos haber sido los primeros en atar la compresión a una métrica categórica: Petetin et al. (2022) miden el déficit de variabilidad de pronósticos de ozono, lo aparean con métricas de umbral y con AUC, y documentan su agravamiento con el horizonte. Reclamamos tres cosas más angostas: (a) **medirlo en el vector de *headways***, como dispersión **transversal** entre componentes en un mismo instante —Mayer y Yang trabajan sobre la varianza **temporal de una serie escalar** de irradiancia, que es la cantidad que los teoremas acotan y **no** es la nuestra—; (b) **darle la vuelta a la fórmula del propio manual sobre el pronóstico**; y (c) atarlo a una regla de evento **relativa y auto-referencial**, donde la compresión mueve numerador y denominador a la vez. Esto último no está en ninguno de los dos precedentes, y en el caso de Petetin et al. no por omisión sino por construcción: sus umbrales son regulatorios (60 y 90 ppbv por normativa) y no admiten recalibración.

- **C2 (reparación por recalibración, no por cambio de modelo).** Donde la literatura responde a la falla cambiando de clase de modelo, mostramos que **el punto de operación alcanza**. Ajustando el corte por MCC sobre una ventana anterior y disjunta y aplicándolo hacia adelante, la **persistencia recupera 0.5× en 11 de las 12 celdas** —o sea, el umbral publicado *era* su óptimo— mientras que el aprendiz necesita **0.58×–0.91×**. Con el corte en sus propias unidades, o puntuando sin umbral, **el veredicto se invierte**: a h=10 el aprendiz discrimina mejor que la persistencia en **3 de 3 corredores y 3 de 3 ventanas**, exactamente donde el corte fijo le daba 253× en contra. Y a h=1 la persistencia gana en 3 de 3 y 3 de 3, así que el error escalar y la detección **coinciden** una vez removido el artefacto.

- **C3 (validez de la métrica).** El veredicto original descansaba en F1, y F1 es el resumen equivocado a estas tasas base (17–30 %, por encima del rango de 0.15–17 % que reporta el campo). Marcar **todas** las celdas —una regla sin contenido— supera al ganador declarado en **5 de las 12 celdas, incluidas las tres de h=10**. Reportamos MCC, ROC-AUC y **precisión media, que no aparece en la literatura de *bunching***, junto con el piso trivial explícito en cada tabla. La ausencia no es nuestra impresión: la propia tabla comparativa de Santos et al. (2022) lista las métricas de cada trabajo previo del subcampo —RMSE, *accuracy*, precisión, *recall*, MAPE, especificidad, sensibilidad, F-measure— y **ninguno reporta AUC ni precisión media**. Y medimos algo que nadie publicó: la **estabilidad comparada** del corte ajustado por F1 contra el ajustado por MCC entre ventanas. El resultado es mixto y se reporta mixto — el F1 es más ajustado en la mediana, pero tiene un modo de falla degenerado que el MCC no tiene, y el costo fuera de muestra favorece al MCC por un factor de 3 a 6.

- **C4 (el hallazgo no es artefacto de nuestra regla).** Nuestro umbral normaliza por la media del propio vector porque estos datos son GPS crudo sin horario — la misma sustitución que hace Yu et al. (2016), que usan el *headway* observado en la primera parada por el mismo motivo. La diferencia es el punto de referencia, y **es el mecanismo**: la suya es observada y fija, la nuestra es predicha y se mueve. Para probar que el colapso no depende de esa elección, repetimos toda la detección con un **corte absoluto en minutos**, calibrado fuera de muestra e idéntico para observado y pronóstico. **No se atenúa: empeora**, y bajo la convención dominante del campo (un cuarto del programado) es **110 veces peor** que bajo la nuestra. Nuestra elección resultó ser la conservadora. Rezazada et al. (2024) confirman además que en este campo *"no existe un único valor de umbral"*, con los publicados entre 20 s y un cuarto del programado.

> [ANDAMIAJE] Y un párrafo corto de "lo que este trabajo NO afirma", que en este venue es un diferenciador y no una debilidad:
> - No afirma que estos modelos estén listos para operar una alarma. Un AUC de 0.60 es información real y está lejos de un sistema de despacho; falta la función de costo.
> - No afirma que el nulo espacial sea una contribución: está publicado (Boudabbous et al. 2026; Rodrigues 2022) y se reporta como confirmación.
> - No afirma que el cruce por horizonte sea nuevo: es conocido en pronóstico de tráfico.

### D. Aporte metodológico, declarado al frente
> [ANDAMIAJE] Sección corta y deliberada. En los ocho papers de IJACSA que relevamos (Vol. 15–17), CERO reportan un test de significancia pareado y UNO declara un corte temporal real. Lo que en un venue exigente sería higiene, acá es diferenciador — pero solo si se declara explícitamente en lugar de esconderlo en Métodos. Listar: auditoría pareada sobre muestras idénticas, Diebold-Mariano con varianza agrupada por día de servicio, Wilcoxon, winsorización calculada en train y aplicada a todos los *splits*, hashes SHA-256 congelados de cada insumo con falla cerrada antes de entrenar, terciles de volatilidad congelados en train+val, y validación en tres orígenes de *rolling*. Agregar la declaración de disponibilidad de datos y código, que ninguno de los ocho tiene.

### E. Estructura del artículo
> [ANDAMIAJE] Un párrafo: "La Sección II revisa… la III describe… etc."

---

## II. Trabajos Relacionados (Related Work)

> [ANDAMIAJE — REESTRUCTURADO 2026-07-29] La versión anterior organizaba esta sección alrededor de un *gap* que ya no es el nuestro ("el subcampo reporta ganancias del DL sin *baseline naive* ni significancia"). Las FUENTES se conservan casi todas; el ARGUMENTO se rehace. El orden nuevo va de lo más cercano a lo más general, para que el lector vea primero a quién le estamos hablando.
>
> **Regla de citación para esta sección:** cada referencia debe llevar identificador verificado. `docs/paper/fuentes-verificadas.md` tiene el estado entrada por entrada y una lista explícita de lo que NO se puede citar porque no se pudo recuperar. **No citar afirmaciones específicas de fuentes marcadas `[SNIPPET]` o `[ABSTRACT]` sin leerlas primero.**

### A. Predecir el *headway* y después umbralizar: la familia y su falla documentada
> [ANDAMIAJE — ESCRITO 2026-07-30] Citación en autor-año; la conversión a `[n]`
> se hace al maquetar. Toda cita textual de esta subsección proviene de fuentes
> marcadas `[TEXTO COMPLETO]` en `docs/paper/fuentes-verificadas.md`. Dos
> restricciones que se respetaron: (a) a Moreira-Matias et al. (2016) se lo cita
> **solo** por la convención de un cuarto del *headway* programado, que es lo
> verificado — no por el mecanismo de alarma; (b) el >95 % de Yu et al. no se
> cita como tal: se cita el par de cifras reconciliado vía Sun et al. (2021).
> Pendiente que toca esta subsección: Jiao, Shen y Zhang (2023) reclama 89 % con
> el mismo esquema y sigue en `[ABSTRACT]` con umbral no verificado. **No se cita
> hasta leerlo**; si se confirma que su horizonte es corto, entra en el párrafo
> de la reconciliación junto a Yu et al.

La vía dominante para anticipar el *bunching* no lo predice: predice el
*headway* y después lo compara contra un umbral. Yu et al. (2016) fijan la
formulación de forma explícita —*"the occurrence of bus bunching can be detected
by thresholding the predicted headway with the planned bus schedule"*— y la
instancian sobre datos AVL de Pekín con máquinas de vectores de soporte y sus
variantes. El esquema se repite con pocas variaciones: un regresor entrega un
valor de *headway*, una regla binaria lo convierte en alarma, y las métricas que
se reportan son las de la alarma.

El umbral de esa regla es, casi siempre, una fracción del *headway* programado.
Un cuarto del programado es la convención más difundida (Yu et al. 2016;
Moreira-Matias et al. 2016; Santos et al. 2022), aunque no la única: la revisión
de Rezazada et al. (2024) advierte que *"there is no single threshold value to
define bunching events, as it depends on the type of the service, time of the
day, location, and service frequency"*, y sitúa los valores publicados entre 20 s
y un cuarto del programado. La elección no es inocua, y la literatura lo sabe:
Santos et al. (2022) declaran que cuando el GTFS no permite calcular el *headway*
programado recurren a un corte definido por el usuario, y usan cinco minutos
absolutos.

Ese hueco —qué se hace cuando no hay horario— es más común de lo que sugiere la
convención, y la solución establecida es sustituir el horario ausente por una
referencia **observada dentro del propio dato**. Yu et al. (2016) lo hacen de
frente: como en Pekín *"most route schedules change over time"* y el operador no
publica una tabla fija, *"this study uses the headway at the first stop as the
'scheduled' headway"*, de modo que su regla marca *bunching* cuando el *headway*
en la parada i cae por debajo de un cuarto del *headway* observado en la primera
parada de la misma corrida. La Sección III-D retoma este punto, porque nuestra
propia regla es una sustitución de la misma clase con otro punto de referencia, y
la diferencia entre ambas referencias resulta ser el mecanismo que este trabajo
mide.

El trabajo al que este artículo responde es Sun, Schmöcker y Nakamura (2021).
Describen la familia con precisión —*"for bunching prediction then an additional
step is required judging whether the predicted headway is below a prior defined
bunching threshold or not"*— y muestran que rinde mal en detección aunque su
error de regresión no lo anticipe: a diez paradas de anticipación la sensibilidad
cae al rango de 34.9–52.7 % mientras, en sus palabras, *"in terms of MAPE and
RMSE… evaluation metrics deteriorate gradually"*. Observan además dónde falla el
regresor, y es exactamente donde importa: *"neither in 1- nor 10-stop-ahead
prediction can these two methods perform favorably under the circumstance that
the actual headway becomes extremely short and bunching is going to happen"*. El
diagnóstico que proponen es el determinismo del pronóstico: como *"headway
prediction produces an exact value for each headway"*, del regresor *"only one
combination of sensitivity and specificity is derived"*. Su remedio es coherente
con ese diagnóstico y consiste en **cambiar de clase de modelo** —pasar a
clasificación probabilística y construir curvas ROC—, con un aparato de
evaluación notablemente más cuidado que el del resto del subcampo: AUC,
corrección de King–Zeng para eventos raros, elección de corte ponderada por costo
y matrices de confusión completas. El giro probabilístico no es aislado: Chen et
al. (2022) modelan vectores de *headway* de pares adyacentes con una mezcla
gaussiana bayesiana, y Yu, Wu, Chen y Ma (2016) predicen el *headway* con
máquinas de vectores de relevancia.

Lo que queda abierto es el paso anterior a esa decisión. Sun et al. aplican a los
pronósticos de sus regresores el mismo corte absoluto de un minuto derivado de
las observaciones, sin recalibrarlo, en los quince horizontes; no reportan
ninguna métrica libre de umbral para los detectores basados en regresión —las
curvas ROC son del clasificador—; no comparan contra un detector trivial; y
atribuyen el déficit al determinismo y no a la compresión de dispersión, que es
visible en sus propias tablas, donde el R² ajustado del modelo de *headway* cae de
0.968 a 0.635 mientras el coeficiente del *headway* rezagado se mantiene en
torno a 1.00. Su propio trabajo futuro propone calcular probabilidades de
excedencia; hasta donde alcanza este relevamiento, nadie lo escribió.

La misma ausencia, en una forma más aguda, aparece en el vecino arquitectónico
más cercano a este trabajo. Usama y Koutsopoulos (2025) pronostican el campo
espacio-temporal completo de *headways* de una línea de metro con ConvLSTM —el
mismo objeto vectorial que modelamos aquí— y reportan únicamente MAE, MSE y
RMSE. En su texto no hay análisis de dispersión, ni detección de eventos, ni
umbral, ni *baseline* de persistencia, ni ablación arquitectónica. Esa lista de
ausencias es, casi renglón por renglón, la lista de contribuciones de este
artículo.

### B. Por qué el umbral se mueve: sub-dispersión de los pronósticos puntuales
> [ANDAMIAJE — ESCRITO 2026-07-30] Restricciones respetadas al redactar:
> (a) Gneiting (2011) se cita **solo** por el funcional que elicita cada pérdida
> — el preprint completo no contiene ninguna afirmación de sub-dispersión;
> (b) Vannitsem y Hagedorn (2011) entra **acreditado a través de** Mayer y Yang,
> porque no lo leímos: no se le atribuye texto ni cifras;
> (c) Wernli et al. (2009), von Storch (1999), Huth (2002) y Maraun (2013) están
> en `[CROSSREF]`/`[SNIPPET]`, así que aparecen parafraseados y **sin comillas**;
> (d) el párrafo de la distinción transversal/temporal es obligatorio y no se
> puede ablandar: es lo que separa nuestro resultado del teorema.

Que el umbral se corra no es un accidente del entrenamiento, sino una
consecuencia de qué cantidad estima un pronóstico puntual. Gneiting (2011)
caracteriza esa correspondencia: cada función de pérdida elicita un funcional
determinado de la distribución condicional —las pérdidas de Bregman, la media;
las lineales por tramos, un cuantil, y su caso simétrico la mediana—. Un
regresor entrenado con error cuadrático apunta entonces a la media condicional y
uno entrenado con error absoluto a la mediana condicional. Ninguno de los dos
apunta a la dispersión, y ninguno tiene incentivo para reproducirla.

La consecuencia sobre la varianza es cuantificable. Patton y Timmermann (2012)
establecen que para un pronóstico óptimo se cumple `V[Y] = V[Ŷ*] + E[e*²]`, de
modo que la varianza del pronóstico está acotada por la de la observación, y su
Corolario 2 hace de la monotonía en el horizonte un teorema:
`V[Ŷ*_{t|t−hS}] ≥ V[Ŷ*_{t|t−hL}]` para `hS < hL`. El deterioro que reportamos en
la Sección IV-B es esa cota volviéndose visible en datos de buses, no un
descubrimiento sobre la cota.

Conviene ser explícito sobre el alcance de esos resultados, porque la diferencia
decide qué parte de nuestro hallazgo es empírica. Los teoremas acotan la varianza
**temporal de una serie escalar**: la dispersión de un mismo pronóstico a lo
largo del tiempo. La cantidad que este trabajo mide es otra — la dispersión
**transversal entre las componentes del vector de *headways* en un mismo
instante**, que es la que gobierna una regla de evento definida sobre el vector.
La descomposición de la varianza se aplica componente a componente, de manera que
la dirección del efecto es la esperable; pero no implica el resultado
transversal, y en particular no dice nada sobre el coeficiente de variación del
corte transversal. Que el sesgo aparezca en la totalidad de las celdas evaluadas
es, por lo tanto, un resultado empírico y se reporta como tal.

Como enunciado general, la sub-dispersión de los pronósticos puntuales está
publicada, y este artículo no la reclama. Mayer y Yang (2022) la formulan sin
matices —*"as MSE-optimized forecasts are always underdispersed, the common
practice of using RMSE skill score for evaluation overrates the forecasts with
lower dispersion"*—, la cuantifican como razón de varianzas sobre pronósticos de
irradiancia solar, donde los pronósticos optimizados en MSE *"only capture <75 %
of the observed variance"*, y establecen la dependencia de la pérdida que
podríamos haber creído propia: los pronósticos optimizados en MAE quedan también
sub-dispersos, pero menos, con menor sesgo condicional de tipo 2 y mayor
discriminación que los de MSE. El agravamiento con el horizonte lo acreditan a
Vannitsem y Hagedorn (2011), que constituye así un segundo precedente del
comportamiento monótono, junto al teorema de Patton y Timmermann. Nuestra
posición frente a ese trabajo es la que corresponde: lo citamos como enunciado
previo, y delimitamos lo que agregamos. Su objeto es una serie escalar de
irradiancia —varianza temporal, la cantidad que los teoremas acotan— y su
evaluación es de calidad de pronóstico: en su texto no aparecen umbrales,
detección de eventos ni excedencias. El puente entre la compresión de dispersión
y una **regla de evento** es lo que queda sin construir, y es lo que este
artículo construye.

Fuera del pronóstico solar, el mismo fenómeno aparece asociado justamente al tipo
de evento que nos ocupa. Ravuri et al. (2021) reportan que una pérdida puntual
produce *nowcasts* de precipitación borrosos a mayor horizonte, con desempeño
pobre sobre los eventos de lluvia media a intensa —es decir, los eventos que se
definen por un umbral—; su explicación es la falta de restricciones sobre la
salida, no la elicitabilidad. Subich et al. (2025) atribuyen el suavizado a la
doble penalización del MSE y lo corrigen cambiando la función de pérdida.
Bonavita (2024) va un paso más lejos y observa que la ventaja de los modelos de
aprendizaje automático en métricas deterministas es parcialmente atribuible al
suavizado, o sea que la métrica premia el defecto.

El fenómeno tiene nombre propio en al menos dos literaturas más. En meteorología
se lo discute como doble penalización (Ebert 2008; Wernli, Hofmann y Zimmer
2009): un campo pronosticado con estructura correcta pero desplazada es
penalizado dos veces, y suavizar lo evita. En *downscaling* climático se lo
discute como inflación de varianza, y sin acuerdo — von Storch (1999) argumenta
en contra de inflar y a favor de aleatorizar, Huth (2002) llega a la conclusión
opuesta, y Maraun (2013) reabre el punto más de una década después. Que el debate
siga abierto es relevante para este trabajo y conviene decirlo: **cómo** corregir
la sub-dispersión no está resuelto, y por eso una contribución que no la corrige
sino que ajusta la regla de decisión aguas abajo tiene lugar legítimo.

Queda por distinguir esta línea de un diagnóstico vecino con el que es fácil
confundirla. La *over-stationarization* (Liu et al. 2022), llevada por Li, Yang y
Wang (2025) a la predicción de arribos de buses, describe un síntoma muy
parecido: en sus palabras, *"the model tends to produce overly stable and
indistinguishable outputs"*. Pero el agente causal que identifican es la
**normalización del preprocesamiento**, y por eso su propuesta es arquitectónica
— la curabilidad por diseño es la premisa de su propio método. El agente que
identificamos aquí es la **pérdida**, y en esa medida es estructural: sobrevive a
cualquier arquitectura entrenada con el mismo objetivo, que es exactamente lo que
observamos al comparar tres arquitecturas distintas. Son afirmaciones opuestas
sobre tratabilidad, y ninguna de las dos fuentes cuantifica el aplanamiento ni lo
atribuye a la función de pérdida.

### C. Recalibrar el umbral: precedentes fuera del transporte
> [ANDAMIAJE — ESCRITO 2026-07-30] Subsección obligatoria: sin ella, un revisor
> de clima o meteorología nos liquida con una cita de 2018. Restricciones que se
> respetaron: (a) a Hoffmann et al. (2018) se lo parafrasea, porque tenemos el
> texto completo pero no dejamos registrada ninguna cita textual; (b) Petetin et
> al. (2022) es la cita central y se le acredita **todo** lo que tiene, incluido
> lo que creíamos nuestro — ver §0.5 de `fuentes-verificadas.md`; (c) Alfieri et
> al. (2019), Zsoter et al. (2020) y Lalaurette (2003) quedaron **fuera**: siguen
> en `[POR VERIFICAR ANTES DE CITAR]` y Petetin ya cubre su papel. Si se
> consiguen, entran como refuerzo, no cambian el argumento.

Que un umbral calibrado sobre observaciones deje de funcionar cuando se lo aplica
a un pronóstico no es un descubrimiento de este trabajo, ni siquiera un problema
del transporte. Está documentado en clima y en calidad del aire, y allí ya se
ensayaron dos respuestas distintas: **corregir el pronóstico** para que su
distribución se parezca a la observada, o **mover el umbral** hacia la
distribución del pronóstico. Este artículo transfiere la segunda al transporte, y
conviene decir con precisión qué se toma prestado y qué no.

La primera familia es la más transitada. Mayer y Yang (2022) proponen calibrar
explícitamente la razón de varianzas del pronóstico solar; Petetin et al. (2022)
aplican *quantile mapping*, cuyo objetivo declarado es *"adjusting the
distribution of the forecast concentrations to the distribution of observed
concentrations"*; Subich et al. (2025) cambian directamente la función de pérdida.
Todas comparten un requisito operativo que no siempre está disponible: para
corregir la distribución del pronóstico hace falta una distribución de referencia
de observaciones —o un reentrenamiento—, y en el caso del *quantile mapping* hace
falta además que esa referencia siga siendo válida cuando el sistema se despliega.

La segunda familia es la que este trabajo continúa, y su precedente más limpio es
Hoffmann, Menz y Spekat (2018). Frente al mismo problema en *downscaling*
climático, identifican a qué percentil de los datos de referencia corresponde el
umbral de interés y **lo recalculan modelo por modelo**, precisamente porque
trasplantar un corte fijo del espacio de las observaciones subestima de forma
severa el conteo de eventos. Es el mismo mecanismo y la misma clase de reparación
que aplicamos aquí, ocho años antes y en otro dominio. Lo que su marco no
contiene es un horizonte de pronóstico, métricas de detección, ni un contexto de
transporte.

El vecino más cercano en el tiempo y en la forma del argumento es Petetin et al.
(2022), y hay que acreditarles sin regateo bastante más de lo que resulta cómodo.
Sobre pronósticos de ozono del servicio Copernicus, documentan que *"there is a
clear trade-off between the continuous and categorical skill scores"* y que *"the
quality of a MOS-corrected forecast assessed solely based on metrics like RMSE or
PCC thus tells little about the forecast value"*, con un caso explícito en el que
un método *"can give the best RMSE and PCC, yet the poorest high O₃ detection
skills"*. Diagnostican y cuantifican la sub-dispersión —el pronóstico crudo
subestima la variabilidad en torno al 30 % y los métodos más sofisticados quedan
demasiado suaves, con dificultad para capturar los valores extremos—, reportan
AUC junto a las métricas de umbral, y muestran que *"all categorical metrics show
a similarly strong sensitivity to the lead time"*. La disociación entre error
continuo y desempeño categórico, la compresión de dispersión que la explica y su
agravamiento con el horizonte están, las tres, publicadas allí.

Lo que no está allí es recalibrar el umbral, y la razón es estructural más que
una omisión: **sus cortes son regulatorios**. Sesenta partes por mil millones
para el máximo diario de ocho horas y noventa para el máximo horario son valores
fijados por normativa, y un umbral legal no se reajusta contra la distribución de
un modelo — dentro de su marco, la pregunta no puede formularse. Lo mismo vale,
por motivos distintos, para el resto de los precedentes: todos operan sobre
cortes **absolutos**, sean regulatorios o convencionales.

Ahí queda delimitado el caso de este artículo, y la diferencia no es de grado.
Bajo un corte absoluto, la compresión de dispersión mueve un solo lado de la
comparación: el valor pronosticado se acerca a su centro mientras el corte
permanece donde estaba. Bajo un corte **relativo y auto-referencial** —una
fracción de la media del propio vector pronosticado, que es la forma que impone
la ausencia de horario— se mueven los dos lados a la vez, porque el denominador
es una función del mismo pronóstico comprimido. El resultado deja de estar
gobernado por el nivel del pronóstico y pasa a estarlo por su **coeficiente de
variación**, que es exactamente la cantidad que la Sección IV-B mide y que
ninguno de estos precedentes necesitó considerar.

### D. Qué métrica puede decidir una detección
> [ANDAMIAJE — ESCRITO 2026-07-30] Restricciones respetadas al redactar:
> (a) **2π/(1+π) se presenta como derivación en un paso**, no como cita — la
> fórmula no está impresa en ninguna fuente, y Lipton et al. solo la instancian
> numéricamente; (b) la **invariancia del AUC ante transformaciones monótonas**
> se deriva de la identidad de rangos, tampoco se atribuye: no figura como
> teorema etiquetado en ninguna de las fuentes; (c) para la regla siempre-activa
> el MCC es **0/0, indeterminado**, y el cero es extensión por continuidad —
> **nunca escribir "cero por construcción"**; (d) Zhu (2020) está en `[ABSTRACT]`
> y solo se le atribuye lo que dice el resumen; (e) el DOI de proceedings de
> McDermott et al. no está verificado: citar por NeurIPS 2024 / arXiv.
> **Pendiente que NO bloquea:** Boughorbel et al. (2017) está en `[CROSSREF]` con
> la instrucción de leer el cuerpo antes de citar, así que el argumento de que
> ajustar un umbral por métrica es principiado se apoya en Koyejo et al. (2014).
> Si se lee Boughorbel, entra como refuerzo. Ídem Itaya et al. (2025), que da
> intervalos de confianza para **diferencias pareadas** de MCC y encaja mejor en
> la Sección III-E que acá.

La familia de trabajos descrita en la subsección A decide sus veredictos con
*accuracy*, precisión, *recall* y F-measure. Esa elección no es neutral, y deja
de serlo especialmente en el régimen de tasa base de este trabajo: entre 17 % y
30 % de celdas marcadas, por encima del rango de 0.15 % a 17 % que reporta el
subcampo. Conviene entonces establecer qué puede y qué no puede decidir cada
resumen antes de usarlos.

El primer problema del F-measure es lo que omite. Powers (2011) señala que esta
familia de medidas ignora el desempeño sobre los ejemplos negativos y no descuenta
el nivel de azar, de modo que *"a system that performs worse in the objective
sense of Informedness, can appear to perform better under any of these commonly
used measures"*. Chicco y Jurman (2020) lo hacen explícito para el caso binario:
el F1 es independiente de los verdaderos negativos y no es simétrico al
intercambiar las etiquetas de clase, mientras que el coeficiente de correlación de
Matthews (MCC) sí lo es. Y Fawcett (2006) agrega la consecuencia práctica: el
F-score se mueve con la distribución de clases aunque el clasificador no cambie,
por lo que comparar desempeños a un umbral común entre escalas distintas *"will
be meaningless"*.

El segundo problema es que el F1 no tiene un cero informativo. Flach y Kull
(2015) fijan la referencia correcta: *"the baseline to beat is the always-positive
classifier rather than any random classifier. This baseline has prec = π and
rec = 1"*. De ahí se sigue en un paso que el F1 de la regla que marca todas las
instancias vale `2π/(1+π)` —una derivación inmediata, no un resultado citable—, lo
que a nuestras tasas base produce un piso de entre 0.29 y 0.46. Un F1 de 0.40 no
significa nada hasta saber de qué lado de ese piso cae, y esa comparación no
aparece reportada en la literatura de *bunching*.

El tercer problema es el que convierte lo anterior en un modo de falla concreto
al calibrar. Lipton, Elkan y Naryanaswamy (2014) demuestran que *"given an
uninformative classifier, optimal thresholding to maximize F1 predicts all
instances positive regardless of the base rate"*, y su Teorema 1 sitúa el umbral
óptimo en la mitad del F1 máximo alcanzable. Sobre un modelo débil con tasa base
del 30 %, ese máximo está cerca del piso trivial, de modo que el corte óptimo cae
por debajo de casi toda predicción y la regla degenera en marcar todo. Los mismos
autores observan además que la selección de umbral por F1 es de alta varianza,
con cortes que *"may be set erroneously"*. La degeneración que reportamos en la
Sección IV-F no es entonces una peculiaridad de nuestros datos: es el
comportamiento que este teorema predice.

El MCC evita las tres cosas —descuenta el azar, usa las cuatro celdas de la
matriz y es simétrico ante el intercambio de clases— pero exige una precisión que
se omite con frecuencia. Chicco, Tötsch y Jurman (2021) señalan que el
coeficiente *"is undefined whenever the confusion matrix has a whole row or a
whole column filled with zeros"*, aunque *"by simple mathematical considerations
it is possible to cover such cases"*. Para la regla siempre-activa —fila de
negativos predichos vacía— el cociente es literalmente indeterminado, y el valor
cero que se le asigna es una extensión por continuidad y una convención, no una
identidad. Este trabajo la adopta y lo declara. Que ajustar un umbral maximizando
una métrica sea un procedimiento principiado y no una conveniencia tiene también
respaldo: Koyejo et al. (2014) establecen que para las métricas expresables como
razones de combinaciones lineales de la matriz de confusión, el clasificador
óptimo es precisamente un umbral que depende de la métrica elegida.

El cuadro sobre el MCC no es unánime y no tiene sentido presentarlo como si lo
fuera. Luque et al. (2019) muestran que no es completamente invariante a la
prevalencia; Zhu (2020) sostiene que su comportamiento se deteriora sobre
conjuntos desbalanceados; y Chicco y Jurman (2023) llegan a argumentar que el MCC
debería **reemplazar** al ROC-AUC. La respuesta que adopta este trabajo es que
las dos medidas responden preguntas distintas y por eso se reportan juntas: el
AUC y la precisión media miden **discriminación sin comprometerse con un punto de
operación**, mientras que el MCC resume la calidad de **un punto de operación
elegido**. Un artículo cuyo objeto es precisamente la transportabilidad del punto
de operación no puede permitirse reportar solo uno de los dos.

Queda una objeción previsible que conviene desactivar antes de que llegue: que a
clases desbalanceadas corresponde la curva de precisión-*recall* y no la ROC. La
objeción existe, pero su respaldo no cubre este régimen. El brazo desbalanceado
de Saito y Rehmsmeier (2015) es de 1:10, es decir una prevalencia cercana al 9 %,
más extrema que la nuestra; Davis y Goadrich (2006) hablan de *"large skew"*, y
su Teorema 3.2 es una **equivalencia** —dominancia en ROC si y solo si dominancia
en PR—, de modo que el AUC no puede invertir una conclusión de dominancia.
McDermott et al. (2024) van más lejos y refutan la premisa de frente: *"AUPRC is
not generally superior in cases of class imbalance"*, y observan que la creencia
contraria *"is often made without citation, misattributed to papers that do not
argue this point"*. Boyd et al. (2012) terminan de dar vuelta el argumento: el
área bajo la curva PR tiene un piso libre `AUCPR_MIN = 1 + (1−π)·ln(1−π)/π`, que
**crece** con la prevalencia y vale 0.168 al 30 % de tasa base. La alternativa
exclusivamente PR padece, en nuestro régimen, la misma enfermedad que le
reprochamos al F1, y peor que en el régimen de eventos raros para el que se la
recomienda. Seguimos su recomendación explícita y graficamos la curva PR mínima
junto a cada curva empírica.

Esta discusión no es abstracta: Li (2024) documenta un caso publicado en el que un
modelo sin capacidad predictiva real alcanza F1 = 0.475 mientras su MCC es 0.042 y
su AUC 0.524. Es exactamente el fallo que las tablas de la subsección A no pueden
detectar. Y la ausencia no es una impresión nuestra: la tabla comparativa con la
que Santos et al. (2022) resumen la literatura previa del subcampo enumera las
métricas de cada trabajo —RMSE, *accuracy*, precisión, *recall*, MAPE,
especificidad, sensibilidad y F-measure— y ninguno reporta AUC ni precisión
media. Tampoco, hasta donde alcanza este relevamiento, ningún trabajo del
subcampo publica el piso del detector trivial junto a sus resultados.

### E. Síntesis del vacío
> [ANDAMIAJE — ESCRITO 2026-07-30] **La formulación del andamiaje anterior se
> quedó corta y hubo que angostarla.** Decía "nadie mide la compresión de
> dispersión que causa la falla", y eso **ya no es cierto fuera del transporte**:
> Petetin et al. (2022) la miden, la cuantifican y la aparean con métricas
> categóricas. El vacío hay que enunciarlo con dos ejes —dominio y tipo de
> umbral—, no con uno. Dos reglas que siguen vigentes: **NO escribir "el
> subcampo no se dio cuenta"** (no sobrevive a un revisor que conozca a Sun et
> al.), y no ensanchar el reclamo más allá del caso auto-referencial.

El estado de la cuestión se ordena en dos ejes, y el aporte de este trabajo queda
donde los dos se cruzan.

Por el eje del **dominio**, el síntoma está diagnosticado. Sun, Schmöcker y
Nakamura (2021) mostraron que el paradigma de predecir el *headway* y umbralizar
falla en detección aunque el error de regresión no lo anuncie, y lo atribuyeron
al punto de operación único del pronóstico determinista. Lo que no se hizo dentro
del transporte es medir la causa ni ensayar la reparación más barata: la
respuesta ha sido invariablemente **cambiar de clase de modelo** —hacia
clasificación, hacia predicción probabilística— y no reajustar el punto de
operación de los modelos que ya se tienen.

Por el eje del **mecanismo**, en cambio, hay bastante más publicado de lo que
sugeriría la literatura de transporte, y este artículo no lo reclama. La
sub-dispersión de los pronósticos puntuales es un enunciado general (Mayer y Yang
2022) con respaldo teórico (Patton y Timmermann 2012). Su daño sobre eventos
definidos por umbral está documentado (Ravuri et al. 2021). Y en calidad del aire
está medida, cuantificada y apareada con métricas categóricas y con AUC, junto
con su agravamiento con el horizonte (Petetin et al. 2022). Recalibrar el corte
contra la distribución del propio modelo también tiene precedente, en clima
(Hoffmann et al. 2018).

Lo que no existe es la intersección, y admite una formulación precisa. **Ningún
trabajo mide la compresión de dispersión sobre un vector de *headways* como
causa de una falla de detección de *bunching*; ninguno recalibra el umbral de
*bunching* contra la distribución del propio pronóstico; y ninguno —dentro ni
fuera del transporte— trata el caso del umbral relativo y auto-referencial**, que
es el que impone la ausencia de horario programado y en el que la compresión
mueve numerador y denominador a la vez, de modo que el resultado queda gobernado
por el coeficiente de variación y no por el nivel. A eso se suma una ausencia
metodológica acotada al subcampo, respaldada por la propia tabla comparativa de
Santos et al. (2022): en detección de *bunching* no se reporta ROC-AUC ni
precisión media, ni se publica el piso del detector trivial contra el que
cualquier F1 debería leerse.

### Referencias de la Sección II
> [ANDAMIAJE] Tomarlas de `docs/paper/fuentes-verificadas.md`, respetando el estado de verificación. Presupuesto observado en IJACSA: 26–46, mediana ≈41.

---

## III. Materiales y Métodos

> [ANDAMIAJE] Casi todo redactado en documento-resultados.md §2 y en el código (`src/evaluation/`). Reordenar y completar el preprocesamiento que el doc de resultados omite a propósito.

### A. Datos: SIT Arequipa (AVL/GPS)
> [ANDAMIAJE] Fuente, período (2023-10-01 → 2024-02-29, 152 días), corredores E2/E59/E4, clave compuesta `(empresaid, unidadid)`, escala de flota. Ver `docs/dataset-manifest.md`.
>
> **Declarar que no hay horario programado.** Es GPS crudo sin GTFS, y de ahí sale la elección de umbral de la subsección D. No es un detalle menor: es la razón por la que nuestra regla difiere de la convención del campo.

### B. Preprocesamiento y definición del *headway*
> [ANDAMIAJE] Corte temporal (train 107 d / val 23 d / test 22 d) y winsorización p99 **calculada en train y aplicada a todos los *splits***. Declarar el contrato.
>
> Y la definición del *headway* (C.2, *trailing crossing*): hace cuánto tiempo el bus de adelante estuvo donde está ahora el que lo sigue, interpolado sobre su trayectoria. Ver `docs/decisiones-headway-fase2.md` §2.1 — y **cuidado**: las columnas del parquet tienen los nombres `front`/`back` invertidos respecto del movimiento físico. La aritmética es correcta; al escribir Métodos hay que usar la ecuación, no los nombres de columna.

### C. Modelos comparados
> [ANDAMIAJE] Tabla de *baselines* B0–B4 (persistencia como rival central), B5_XGB nivelado, y los tres DL (LSTM, SpatialConvLSTM, SpatialTransformer). Reusar tablas de §2 del doc de resultados. Declarar la asimetría de *tuning* (24 configuraciones contra 1 o 3) donde corresponde.

### D. Definición del evento de *bunching*, y por qué esta
> [ANDAMIAJE] POR ESCRIBIR, y es delicado. Contenido obligatorio:
>
> - Nuestra regla: celda marcada si su *headway* cae por debajo de 0.5× la media de su propio vector, calculada contra la media del **vector predicho** cuando se evalúa una predicción, porque un operador no tiene acceso a la media real.
> - **Declarar que no es la convención del campo.** La forma dominante es una fracción del *headway* programado (un cuarto en Yu et al. y Moreira-Matias et al.; un medio en el TCQSM). La forma "fracción de la media observada" no existe como definición publicada: la usamos porque no hay horario. Es sustitución nuestra.
> - Declarar la tasa base resultante: 17–30 %, por encima del rango 0.15–17 % que reporta el campo.
> - Y anunciar la verificación de la Sección IV-E: la elección se somete a prueba con un corte absoluto, y resulta ser la conservadora.

### E. Protocolo de evaluación
> [ANDAMIAJE] Es lo que nos diferencia en este venue — desarrollarlo bien y explícitamente.
>
> - **Comparación pareada sobre muestras idénticas** (*paired audit*). Señalar que Diebold-Mariano está **indefinido**, no solo sesgado, cuando los modelos se puntúan sobre filas distintas: se define sobre el diferencial de pérdida por fila.
> - Diebold-Mariano con varianza **agrupada por día de servicio** (G = 22, gl = 21) y corrección de muestra chica; HAC/Newey-West como contraste. Wilcoxon pareado. Tamaño de efecto primero, *p* como piso.
> - Terciles de volatilidad **ex-ante**, congelados en train+val y aplicados a test.
> - Tres orígenes de *rolling* con ventanas de test disjuntas. Declarar que los conjuntos de entrenamiento están **anidados**, no independientes.
> - Métricas escalares MAE y error cuadrático. **Métricas de detección: MCC, ROC-AUC y precisión media, con el piso del detector trivial en cada tabla.** Justificar por qué no F1 (§II-D).
> - Calibración del umbral: ajuste por MCC en una ventana anterior disjunta, aplicado hacia adelante. Declarar que es la única dirección en que un operador podría calibrar.
> - Horizontes h ∈ {1, 3, 5, 10} minutos.

---

## IV. Resultados

> [ANDAMIAJE — REESTRUCTURADO 2026-07-29] El orden cambió. Antes iba: cruce → significancia → volatilidad → enrutador, o sea el resultado escalar como titular. Ahora el escalar es **contexto** y el titular es el artefacto de umbral y su reparación.
>
> Figuras: usar `contiguo-artefacto-umbral.png` y `contiguo-deteccion-sin-umbral.png` (**solo funcionan como par**), más `contiguo-degradacion.png` y `contiguo-volatilidad.png`. **NO usar** `curva-degradacion.png` ni `volatilidad-crossover.png`: son de las familias congeladas y no de este pipeline. La figura `contiguo-disociacion.png` fue eliminada — graficaba F1 con corte fijo como si midiera a los modelos.

### A. Contexto: el resultado escalar y su frontera
> [ANDAMIAJE] Doc §3, comprimido. El cruce de MAE (persistencia gana h=1, el aprendiz gana h≥5), replicado por el XGBoost, y que la frontera real es la volatilidad y no el horizonte. Presentarlo como **contexto establecido**, no como aporte — es folclore conocido en pronóstico de tráfico.
>
> Incluir acá la inversión MAE / error cuadrático a h=1 (doc §3): el vector aplanado **pierde** el absoluto y **gana** el cuadrático en los tres corredores. Es la evidencia interna de que la sub-dispersión no viene de elegir MAE.

### B. La compresión de dispersión, medida
> [ANDAMIAJE] Doc §5.2. CV predicho contra real, las 36 celdas, monotonía con el horizonte, persistencia con sesgo ≈0. El argumento de `Z = 0.5/cv` del TCQSM y el contraste LOS A / LOS F. **Y la distinción transversal contra temporal**, explícita.

### C. El artefacto: la alarma no suena
> [ANDAMIAJE] Doc §1 y §5.3. Figura `contiguo-artefacto-umbral.png`. Los 14 disparos en 50 356 oportunidades con 15 245 eventos; el factor 253×; y los tres hechos que vacían esa tabla — el corte de 0.5× es el óptimo de la persistencia (11/12 al reajustar), el ganador declarado pierde contra la regla constante en 5/12, y la precisión del aprendiz cuando dispara es 71 % contra 30 % de tasa base (con el IC ancho declarado, y respaldado por E59 con 1 573 disparos).

### D. La reparación: sin umbral, el veredicto se invierte
> [ANDAMIAJE] Doc §5.4 y §5.5. Figura `contiguo-deteccion-sin-umbral.png`. AUC y MCC calibrado, celda por celda. La inversión a h=10 en 3/3 corredores y 3/3 ventanas; la persistencia ganando a h=1 en 3/3 y 3/3; el cruce de detección coincidiendo con el escalar. Y la celda partida (E4 h=5, diferencia de una milésima) **nombrada**.

### E. Robustez: no es de febrero, ni de nuestro umbral
> [ANDAMIAJE] Doc §5.5 y §5.6. Los tres orígenes para el aplanamiento (36/36) y para el AUC (11/12, 9/9 en los extremos). Y el corte absoluto: el colapso empeora, 110× peor bajo la convención del campo. **Con la salvedad que corre en contra**: el AUC del evento absoluto es más bajo, mediana 0.599, y una celda en 0.4934, o sea azar.

### F. Por qué MCC y no F1, medido
> [ANDAMIAJE] Doc §5.7. La tabla de estabilidad. **Reportar el resultado mixto como mixto**: el F1 es más ajustado en la mediana; lo que lo descalifica es el modo de falla degenerado y el costo fuera de muestra (3.4× en la mediana, 5.6× en el peor caso).

### G. El enrutador ex-ante
> [ANDAMIAJE] Doc §6, dos párrafos. Supera el ruido de partición en 2 de 12 celdas, 7 de 12 políticas son degeneradas, y en E2 h=1 ayuda bajo partición aleatoria pero perjudica bajo corte temporal. Vale como demostración de ejecutabilidad, no como contribución. **Decirlo así.**

---

## V. Discusión

### A. Interpretación operativa
> [ANDAMIAJE] POR ESCRIBIR, y es la sección que le habla a la empresa de transporte. El eje: la diferencia entre *"el modelo no sirve para anticipar bunching"* —que cierra la línea de trabajo— y *"el modelo sirve pero la alarma está mal seteada"* —que es ajustar un escalar sobre datos que ya tenés—. Con el límite honesto: un AUC de 0.60 no es un sistema de despacho, y no modelamos el costo.

### B. Qué queda del aporte, y qué no
> [ANDAMIAJE] POR ESCRIBIR, corto y franco. **Cuatro** delimitaciones, no tres, y el orden importa porque va de la más cercana a la más lejana:
> 1. **Petetin et al. (2022)** — es el vecino principal desde el 2026-07-30, por encima de Hoffmann. Ellos tienen el apareamiento continuo↔categórico, la sub-dispersión cuantificada y la sensibilidad al horizonte. Nosotros tenemos la recalibración del umbral, que en su marco es **inexpresable** porque sus cortes son regulatorios. Corte absoluto contra corte relativo y auto-referencial: en el primero se mueve un solo lado, en el segundo se mueven los dos.
> 2. **Sun et al. (2021)** — síntoma contra mecanismo; cambio de clase de modelo contra recalibración del punto de operación.
> 3. **Hoffmann et al. (2018)** — misma clase de reparación, pero en clima, sin horizonte y sin métricas de detección.
> 4. **Li, Yang y Wang (2025)** — normalización contra pérdida; curable por arquitectura contra estructural.
>
> Dos oraciones por delimitación alcanzan, y valen más que reclamar de más.

### C. El resultado nulo espacial, en contexto
> [ANDAMIAJE] POR ESCRIBIR, corto. Enmarcarlo como **confirmación de resultado publicado**, no como hallazgo: Boudabbous et al. (2026) sobre datos de Montreal reportan LSTM superando a transformers por 18–52 % con 77× menos parámetros, y Rodrigues (2022) muestra un *baseline* de patrón semanal igualando al SOTA espacio-temporal. Y decir que es específico de estos datos y corredores, no una ley.

### D. Amenazas a la validez y limitaciones
> [ANDAMIAJE] YA REDACTADO — las 11 limitaciones del doc §8. Mantenerla sustantiva: en el relevamiento del venue, 3 de 8 papers tienen algo parecido y ninguno tiene disponibilidad de datos. Las que más pesan: el valor operativo no está modelado (lim. 8), el umbral no está calibrado contra incidentes registrados (lim. 9), y la calibración usa dos ventanas con entrenamientos anidados en lugar de validación cruzada (lim. 10).

---

## VI. Conclusión y Trabajo Futuro

> [ANDAMIAJE] REESCRIBIR desde cero — la versión anterior cerraba sobre "la métrica decide el ganador", que se retiró.
>
> Cerrar sobre el *gap* y no sobre los resultados. La forma: un pronóstico puntual está sub-disperso; ese déficit no le quita información sobre el evento, le cambia las unidades; y por eso una regla de alarma calibrada sobre observaciones fabrica un fracaso aparente de hasta 253× que no existe. La reparación es un escalar, no una arquitectura nueva.
>
> Trabajo futuro, en orden de valor: (1) pronóstico probabilístico del vector de *headways* con probabilidades de excedencia —que es lo que Sun et al. propusieron y nadie escribió—, (2) validación cruzada rotativa de la calibración del umbral en lugar de dos ventanas, (3) función de costo operativo que traduzca AUC a decisiones de despacho, (4) más ciudades.

---

## Referencias

> [ANDAMIAJE] POR ESCRIBIR. Formato de la plantilla IJACSA (IEEE-like, numeradas `[n]`). Fuente única: `docs/paper/fuentes-verificadas.md`, respetando el estado de verificación de cada entrada. Presupuesto ≈40.
>
> **Verificar cada cita contra el documento original antes de someter.** La lista de "no citar" de ese archivo es vinculante.

<!--
  MAPA DE PROGRESO (borrar antes de enviar):

  [ESCRITO 2026-07-30]       ✅ SECCIÓN II COMPLETA — A, B, C, D y E
  [YA REDACTADO, trasladar]  III Métodos (parcial) · IV Resultados A-G · V.D Limitaciones
  [POR ESCRIBIR]             Abstract · I Introducción · III.D Definición del evento ·
                             V.A Interpretación · V.B Qué queda del aporte ·
                             V.C Nulo espacial · VI Conclusión · Referencias

  Las dos subsecciones que decidían si el paper se lee como honesto o como ingenuo
  —II-A y II-C— acreditan de frente lo que no es nuestro. II-E enuncia el vacío en
  DOS ejes (dominio y tipo de umbral), no en uno: la formulación de un solo eje se
  volvió falsa cuando se leyó Petetin.

  SIGUIENTE CUELLO DE BOTELLA: III.D (definición del evento). Es la que tiene que
  declarar que nuestro umbral NO es la convención del campo, y ahora puede apoyarse
  en el precedente de Yu et al. (referencia observada por horario ausente).

  BLOQUEANTES — RESUELTOS el 2026-07-29, los cuatro papers leídos (ver fuentes-verificadas.md §0):
    V1  Mayer y Yang 2022  → CONFIRMADO. "MSE-optimized forecasts are always
                             underdispersed" es literal. C1 reformulado: queda el CV
                             transversal sobre el vector, la inversión del TCQSM y la
                             consecuencia sobre la regla de evento (ellos: 0 umbrales).
    V2  Rezazada et al.    → VERIFICADO. "20 s a ¼ del programado", y "no existe un
                             único valor de umbral". Pendiente menor: revisar Gong et
                             al. (2020), umbral variable.
    V3  Yu et al. >95 %    → RECONCILIADO. Es a 2 PARADAS; su propia sensibilidad cae
                             a 73 % a 5 paradas (vía Sun et al. 2021). Métricas sin
                             precisión, sin AUC, sin detector trivial. Y su umbral usa
                             el headway OBSERVADO de la primera parada como sustituto
                             del horario ausente — mismo problema nuestro, misma clase
                             de solución, referencia distinta.
    V8  Santos et al. 2022 → SIN BARRIDO DE UMBRAL. C2 sobrevive. Y su tabla de la
                             literatura previa confirma que nadie reporta AUC ni AP.

  PENDIENTE MENOR: Jiao et al. 2023 (89 % con LSTM → umbral), doi 10.1109/icite59717.2023.10733869.
  No conseguido. Probablemente se reconcilia igual que Yu et al. (horizonte corto), pero
  hay que verificarlo antes de someter.
-->
