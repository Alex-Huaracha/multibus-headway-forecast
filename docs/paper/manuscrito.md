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
  *No reclamamos* que la sub-dispersión de pronósticos puntuales sea un hallazgo nuevo: es teoría establecida (la descomposición de la varianza; Patton y Timmermann 2012 prueban además la monotonía en el horizonte) y fue observada en otros dominios. Reclamamos **medirla en el vector de *headways*** —donde nadie la midió— y **darle la vuelta a la fórmula del manual sobre el pronóstico**, que es la inversión no publicada.

- **C2 (reparación por recalibración, no por cambio de modelo).** Donde la literatura responde a la falla cambiando de clase de modelo, mostramos que **el punto de operación alcanza**. Ajustando el corte por MCC sobre una ventana anterior y disjunta y aplicándolo hacia adelante, la **persistencia recupera 0.5× en 11 de las 12 celdas** —o sea, el umbral publicado *era* su óptimo— mientras que el aprendiz necesita **0.58×–0.91×**. Con el corte en sus propias unidades, o puntuando sin umbral, **el veredicto se invierte**: a h=10 el aprendiz discrimina mejor que la persistencia en **3 de 3 corredores y 3 de 3 ventanas**, exactamente donde el corte fijo le daba 253× en contra. Y a h=1 la persistencia gana en 3 de 3 y 3 de 3, así que el error escalar y la detección **coinciden** una vez removido el artefacto.

- **C3 (validez de la métrica).** El veredicto original descansaba en F1, y F1 es el resumen equivocado a estas tasas base (17–30 %, por encima del rango de 0.15–17 % que reporta el campo). Marcar **todas** las celdas —una regla sin contenido— supera al ganador declarado en **5 de las 12 celdas, incluidas las tres de h=10**. Reportamos MCC, ROC-AUC y **precisión media, que no aparece en la literatura de *bunching***, junto con el piso trivial explícito en cada tabla. Y medimos algo que nadie publicó: la **estabilidad comparada** del corte ajustado por F1 contra el ajustado por MCC entre ventanas. El resultado es mixto y se reporta mixto — el F1 es más ajustado en la mediana, pero tiene un modo de falla degenerado que el MCC no tiene, y el costo fuera de muestra favorece al MCC por un factor de 3 a 6.

- **C4 (el hallazgo no es artefacto de nuestra regla).** Nuestro umbral relativo a la media del propio vector **no es la convención del campo** —la forma dominante es una fracción del *headway* programado— porque estos datos son GPS crudo sin horario. Así que repetimos toda la detección con un **corte absoluto en minutos**, calibrado fuera de muestra e idéntico para observado y pronóstico. El colapso **no se atenúa: empeora**, y bajo la convención dominante del campo (un cuarto del programado) es **110 veces peor** que bajo la nuestra. Nuestra elección de umbral resultó ser la conservadora.

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
> [ANDAMIAJE] POR ESCRIBIR — es la subsección más importante y va primera. Contenido:
>
> - La formulación canónica: Yu et al. (2016) establecen predecir el patrón de *headway* como vía para anticipar el *bunching*, con LS-SVM y umbral en un cuarto del *headway* programado; reportan >95 % de eventos identificados. Moreira-Matias et al. (2016) umbralizan verosimilitudes de *headway* predicho para disparar alarmas de control.
> - Sun, Schmöcker y Nakamura (2021) son **el trabajo al que este paper responde**, y hay que citarlos temprano y generosamente. Describen la familia con precisión —*"For bunching prediction then an additional step is required judging whether the predicted headway is below a prior defined bunching threshold or not"*— muestran la disociación entre error de regresión y sensibilidad, observan que los regresores fallan justo cuando el *headway* real se acorta, y diagnostican la causa como el **punto de operación único** de un pronóstico determinista. Reportan curvas ROC, AUC, corrección de King–Zeng para eventos raros, elección de corte ponderada por costo y matrices de confusión completas.
> - **Y decir con claridad qué dejaron abierto**, porque ahí está nuestro lugar: aplican el corte de 1 minuto derivado de observaciones a los pronósticos de LR y SVM **sin recalibrarlo**, en los quince horizontes; no reportan métrica libre de umbral para los detectores basados en regresión; no comparan contra un detector trivial; y atribuyen el déficit al determinismo, no a la compresión de dispersión — que es visible en sus propias tablas, donde el R² ajustado cae de 0.968 a 0.635 mientras el coeficiente del *headway* se queda en ≈1.00. Su propio trabajo futuro propone calcular probabilidades de excedencia; nadie lo escribió.
> - Vecino arquitectónico, y la mejor cita motivadora: Usama y Koutsopoulos (2025) pronostican el campo espacio-temporal completo de *headways* de una línea de metro con ConvLSTM, y reportan solo MAE/MSE/RMSE — sin análisis de dispersión, sin detección de eventos, sin umbral, sin *baseline* de persistencia y sin ablación arquitectónica. Esa lista de ausencias es la lista de contribuciones de este paper, y conviene decirlo así.

### B. Por qué el umbral se mueve: sub-dispersión de los pronósticos puntuales
> [ANDAMIAJE] POR ESCRIBIR. Ordenar de la teoría al fenómeno:
>
> - Gneiting (2011): qué funcional elicita cada pérdida (Bregman ⟺ media; lineal por tramos ⟺ cuantil, y su caso α=½ ⟺ mediana). **Ojo: NO contiene ninguna afirmación de sub-dispersión** — verificado por grep del preprint completo. Citarlo solo por el funcional.
> - Patton y Timmermann (2012): `V[Y] = V[Ŷ*] + E[e*²]`, y el Corolario 2, que hace de la monotonía en el horizonte un **teorema**. Presentar nuestro 36/36 como esa cota volviéndose visible, no como descubrimiento.
> - **La distinción que hay que hacer explícita, o un revisor la usa en contra:** esos teoremas acotan la varianza **temporal de una serie escalar**. Nuestro CV es la dispersión **transversal entre las componentes del vector en un mismo instante**. La ley de varianza total se aplica componente a componente, así que la dirección es la esperable, pero **no implica** el resultado transversal. Las 36 celdas son resultado empírico, y eso es lo que las hace reportables.
> - Corroboración entre dominios: Ravuri et al. (2021, *Nature*) — pérdida puntual → *nowcasts* borrosos que empeoran con el horizonte y rinden mal justo en los eventos definidos por umbral; su explicación es "lack of constraints", no elicitabilidad. Subich et al. (ICML 2025) atribuyen el suavizado a la doble penalización del MSE y lo arreglan cambiando la pérdida. Bonavita (2024): el suavizado *mejora* las métricas deterministas.
> - El mismo fenómeno tiene nombre en meteorología (doble penalización: Ebert 2008; Wernli et al. 2009) y en *downscaling* climático (inflación de varianza: von Storch 1999 en contra, Huth 2002 a favor, Maraun 2013 revisitándolo). **Que el debate siga abierto nos sirve** y hay que decirlo: cómo corregir la sub-dispersión no está resuelto.
> - Distinguir de la *over-stationarization* (Liu et al. 2022; Li, Yang y Wang 2025, que la lleva a predicción de arribos de buses en Dresde). Su villano es la **normalización** del preprocesamiento, curable por arquitectura — es la premisa de su propio método. El nuestro es la **pérdida**, y es estructural. Son afirmaciones opuestas sobre tratabilidad. Su borde expuesto es la frase *"the model tends to produce overly stable and indistinguishable outputs"*: citarla, acreditarla, y señalar que no la cuantifican ni la atribuyen a la pérdida.

### C. Recalibrar el umbral: precedentes fuera del transporte
> [ANDAMIAJE] POR ESCRIBIR, y **es obligatorio**. Si presentamos el mecanismo como nuevo, un revisor de meteorología o clima nos termina con una cita de 2018.
>
> - Hoffmann, Menz y Spekat (2018): identifican el percentil del umbral en los datos de referencia y **lo recalculan por modelo**, porque un corte fijo del espacio de observaciones subestima severamente el conteo de eventos. Es nuestro mecanismo **y** nuestro arreglo, en clima, ocho años antes. Citarlo como el precedente que estamos transfiriendo.
> - Y decir qué NO cubre: sin horizonte de pronóstico, sin métricas de detección, sin transporte, y —crítico— **todos esos precedentes usan cortes absolutos o regulatorios, no relativos y auto-referenciales.** Ahí queda nuestro caso.
> - [POR VERIFICAR ANTES DE CITAR] La familia hidrología / calidad de aire que el relevamiento identificó (Alfieri et al. 2019; Zsoter et al. 2020; Petetin et al. 2022; Lalaurette 2003 y su EFI sobre climatología del modelo). Petetin et al. aparentemente combinan nuestro mecanismo, nuestro diagnóstico de dispersión y nuestro apareamiento AUC-con-métrica-de-umbral: **hay que leerlo antes de escribir esta subsección.**
> - **BLOQUEANTE:** Mayer y Yang (2022, *IJF*) — los *snippets* le atribuyen *"MSE-optimized forecasts are always underdispersed"* como enunciado general. Es CC-BY. **Leerlo antes de someter.** Si es literal, C1 se reformula como "primera cuantificación en detección de eventos de transporte" y se cita como enunciado previo.

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

  [YA REDACTADO, trasladar]  III Métodos (parcial) · IV Resultados A-G · V.D Limitaciones
  [POR ESCRIBIR]             Abstract · I Introducción · II Related Work (entera) ·
                             III.D Definición del evento · V.A Interpretación ·
                             V.B Qué queda del aporte · V.C Nulo espacial · VI Conclusión ·
                             Referencias

  CUELLO DE BOTELLA: II-A (la familia y Sun et al.) y II-C (precedentes de recalibración).
  Son las dos que deciden si el paper se lee como honesto o como ingenuo.

  BLOQUEANTES ANTES DE ESCRIBIR II (ver fuentes-verificadas.md §6):
    V1  Mayer y Yang 2022 — puede contener C1 como enunciado previo
    V2  Rezazada et al. 2024 — la review que un referí espera citada
    V3  Reconciliar Jiao et al. (89 %) y Yu et al. (>95 %) con nuestro colapso
    V8  Santos et al. 2022 — el lugar más probable donde ya exista un barrido de umbral
-->
