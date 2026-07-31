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
  *No reclamamos* que la sub-dispersión de pronósticos puntuales sea un hallazgo nuevo, y hay que decirlo de frente porque **está publicado como enunciado general**: Mayer y Yang (2022) escriben que *"MSE-optimized forecasts are always underdispersed"*, cuantifican el déficit como razón de varianza (< 75 % de la varianza observada), establecen que los pronósticos optimizados en MAE quedan **menos** sub-dispersos que los de MSE, y atribuyen el agravamiento con el horizonte a Vannitsem y Hagedorn (2011). Patton y Timmermann (2012, Cor. 2) prueban esa monotonía como teorema. Reclamamos tres cosas más angostas: (a) **medirlo en el vector de *headways***, como dispersión **transversal** entre componentes en un mismo instante —Mayer y Yang trabajan sobre la varianza **temporal de una serie escalar** de irradiancia, que es la cantidad que los teoremas acotan y **no** es la nuestra—; (b) **darle la vuelta a la fórmula del propio manual sobre el pronóstico**; y (c) atarlo a una **consecuencia sobre una regla de evento**, que en esa literatura no aparece: en Mayer y Yang las palabras *threshold*, *detect* y *exceed* no figuran ni una vez.

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
> [ANDAMIAJE] POR ESCRIBIR, y **es obligatorio**. Si presentamos el mecanismo como nuevo, un revisor de meteorología o clima nos termina con una cita de 2018.
>
> - Hoffmann, Menz y Spekat (2018): identifican el percentil del umbral en los datos de referencia y **lo recalculan por modelo**, porque un corte fijo del espacio de observaciones subestima severamente el conteo de eventos. Es nuestro mecanismo **y** nuestro arreglo, en clima, ocho años antes. Citarlo como el precedente que estamos transfiriendo.
> - Y decir qué NO cubre: sin horizonte de pronóstico, sin métricas de detección, sin transporte, y —crítico— **todos esos precedentes usan cortes absolutos o regulatorios, no relativos y auto-referenciales.** Ahí queda nuestro caso.
> - [POR VERIFICAR ANTES DE CITAR] La familia hidrología / calidad de aire que el relevamiento identificó (Alfieri et al. 2019; Zsoter et al. 2020; Petetin et al. 2022; Lalaurette 2003 y su EFI sobre climatología del modelo). Petetin et al. aparentemente combinan nuestro mecanismo, nuestro diagnóstico de dispersión y nuestro apareamiento AUC-con-métrica-de-umbral: **hay que leerlo antes de escribir esta subsección.**
> - **LEÍDO 2026-07-30, y la amenaza se confirmó.** Mayer y Yang (2022, *IJF*) enuncian literalmente *"As MSE-optimized forecasts are always underdispersed, the common practice of using RMSE skill score for evaluation overrates the forecasts with lower dispersion."* Se cita como **enunciado previo**, junto con Vannitsem y Hagedorn (2011) que ellos acreditan por el agravamiento con el horizonte. C1 ya está reformulado en consecuencia. Lo que NO tienen, verificado por conteo sobre el texto completo: `threshold` 0, `detect` 0, `exceed` 0, `AUC` 0, `coefficient of variation` 0 — y su objeto es irradiancia solar **escalar**, o sea varianza temporal, exactamente la distinción de la subsección B.

### D. Qué métrica puede decidir una detección
> [ANDAMIAJE] POR ESCRIBIR. Es la subsección que blinda C3.
>
> - Powers (2011): el F-measure ignora los verdaderos negativos y no descuenta el nivel de azar; un sistema peor en *informedness* puede parecer mejor. Fawcett (2006): F-score se mueve con la distribución de clases aunque el clasificador no cambie.
> - Flach y Kull (2015): el *baseline* a superar es el clasificador siempre-positivo, con precisión π y *recall* 1. **De ahí se deriva 2π/(1+π) en un paso — la fórmula no está impresa en ninguna fuente, así que atribuirla como derivación, no como cita.** Lipton et al. (2014) la instancian numéricamente (0.67 con π=0.5; 0.18 con π=0.1).
> - Lipton et al. (2014): **maximizar F1 sobre un clasificador sin información predice todo positivo, sea cual sea la tasa base**, y el umbral óptimo es la mitad del F1 máximo alcanzable. Es el teorema que explica nuestra degeneración medida (el corte por F1 dispara el 99.99 % en E2) y justifica calibrar por MCC.
> - Chicco y Jurman (2020) y Chicco, Tötsch y Jurman (2021) para MCC. **Precisión obligatoria:** para la regla siempre-positiva el MCC es 0/0, indeterminado; cero es la extensión por continuidad y la convención estándar. No escribir "0 por construcción". Boughorbel et al. (2017) para que calibrar por MCC sea principiado y no ad hoc. Y citar la crítica: Chicco y Jurman (2023) argumentan reemplazar el ROC-AUC por MCC — hay que responderla, no ignorarla.
> - **Anticipar la objeción del desbalance**, que va a venir: Saito y Rehmsmeier definen su brazo desbalanceado en 1:10 (π ≈ 0.09), *más* extremo que nuestro 17–30 %; Davis y Goadrich hablan de *"large skew"* y su teorema es una equivalencia de dominancia; McDermott et al. (NeurIPS 2024) refutan de frente la superioridad del AUPRC. Y Boyd et al. (2012) dan vuelta la objeción: el piso libre del AUPRC crece hacia π=0.5, o sea 0.168 a nuestra tasa base — la alternativa PR-exclusiva tiene la **misma** enfermedad, peor a nuestra prevalencia. Seguir su recomendación explícita y dibujar la curva PR mínima.

### E. Síntesis del vacío
> [ANDAMIAJE] POR ESCRIBIR. La formulación tiene que ser exactamente esta, ni más ancha ni más angosta:
>
> El síntoma está diagnosticado en el dominio (Sun et al. 2021) y el mecanismo tiene precedentes fuera de él (Hoffmann et al. 2018; Ravuri et al. 2021). Lo que no existe es lo del medio: **nadie mide la compresión de dispersión que causa la falla, nadie recalibra el corte contra la distribución del propio pronóstico —la respuesta del campo es siempre cambiar de clase de modelo—, nadie reporta un piso de detector trivial, y nadie usa precisión media a estas tasas base.** Y ninguno de los precedentes de recalibración trata el caso de umbral **relativo y auto-referencial**, que es donde el coeficiente de variación gobierna el resultado.
>
> **NO escribir "el subcampo no se dio cuenta".** Esa afirmación no sobrevive a un revisor que conozca a Sun et al.

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
> [ANDAMIAJE] POR ESCRIBIR, corto y franco. Delimitar contra Sun et al. (2021) —síntoma contra mecanismo; cambio de modelo contra recalibración—, contra Hoffmann et al. (2018) —clima con cortes absolutos contra transporte con cortes relativos y auto-referenciales— y contra Li, Yang y Wang (2025) —normalización contra pérdida—. Tres párrafos de dos oraciones cada uno alcanzan, y valen más que reclamar de más.

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

  [ESCRITO 2026-07-30]       II-A (la familia y Sun et al.) · II-B (sub-dispersión)
  [YA REDACTADO, trasladar]  III Métodos (parcial) · IV Resultados A-G · V.D Limitaciones
  [POR ESCRIBIR]             Abstract · I Introducción · II-C · II-D · II-E ·
                             III.D Definición del evento · V.A Interpretación ·
                             V.B Qué queda del aporte · V.C Nulo espacial · VI Conclusión ·
                             Referencias

  CUELLO DE BOTELLA: queda II-C (precedentes de recalibración). II-A ya está escrita.
  Son las dos que deciden si el paper se lee como honesto o como ingenuo.
  BLOQUEA A II-C: leer Petetin et al. (2022) — marcado [POR VERIFICAR ANTES DE CITAR],
  y aparentemente combina nuestro mecanismo con nuestro apareamiento de métricas.

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
