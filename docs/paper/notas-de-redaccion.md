# Notas de redacción del manuscrito

Extraído de `manuscrito.md` el 2026-07-30, antes de traducir. Cada bloque es el
andamiaje que guió la redacción de una sección: **restricciones de citación,
decisiones tomadas y su motivo, y lo que NO se puede escribir**.

No es documentación histórica. Sirve para dos cosas concretas:

1. **Al traducir**, verificar que la versión en inglés no reintroduzca una
   afirmación que acá está explícitamente prohibida (por ejemplo "el MCC es cero
   por construcción", o citar textual una fuente marcada `[SNIPPET]`).
2. **Al responder a revisores**, saber por qué una afirmación está redactada
   así y no de otro modo.

La fuente de verdad sobre el estado de verificación de cada referencia sigue
siendo `fuentes-verificadas.md`.

---

## Bloque de control: (preámbulo)

```
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
```

### El umbral, no el modelo: compresión de dispersión y recalibración del corte en la detección de *bunching* de buses a partir de datos GPS reales

[ANDAMIAJE — ELEGIDO 2026-07-30] Nombra las tres cosas que el título debía
nombrar: el objeto (*detección de bunching*), el ángulo (*el umbral no es
transportable*, en la cabeza y sin adorno) y el dato real. **Es un cambio de una
línea**, así que quedan registradas las dos alternativas anteriores por si se
prefiere otro tono:

- Más polémica: *"Un pronóstico aplanado parece ciego sin serlo: transportabilidad del umbral en la detección de bunching de buses a partir de datos GPS reales"*. Se descartó porque la afirmación va en la cabeza sin el mecanismo al lado, y un revisor puede leerla como reclamo antes de llegar a la evidencia.
- Más conservadora: *"Compresión de dispersión y recalibración de umbrales en la detección de bunching a partir de pronósticos de headway"*. Se descartó porque nombra el mecanismo pero no la consecuencia, y en un corpus donde la norma es una fila en negrita ganando, no distingue.

### Abstract

[ANDAMIAJE — ESCRITO 2026-07-30] 235 palabras, un párrafo, dentro del rango
150–250. Sin citas y sin abreviaturas indefinidas: no aparecen las siglas del
coeficiente de variación ni del área bajo la curva, y "correlación de Matthews"
va desarrollada. Es el **único** lugar donde el lector ve cifras antes de la
Sección IV — la Conclusión deliberadamente no las repite.
**Al traducir, verificar el conteo de palabras otra vez:** el inglés suele
comprimir y puede caer por debajo de 200.

### I. Introducción

[ANDAMIAJE] SE ESCRIBE CASI AL FINAL. Cuatro movimientos, en este orden.

### A. Contexto y motivación

[ANDAMIAJE — ESCRITO 2026-07-30] Dos párrafos. El segundo cierra sobre el
paradigma operativo, que es lo que la Sección I-B va a atacar.

### B. El problema

[ANDAMIAJE — ESCRITO 2026-07-30] Dos reglas vinculantes que se respetaron:
**NO** decir "el subcampo no usa *baselines naive* ni tests de significancia"
—es plomería y además es un encuadre débil, y va en I-D como contribución
metodológica, no acá como queja—; y **NO** decir "nadie se dio cuenta", porque
Sun et al. (2021) diagnosticaron el síntoma y están citados en las dos primeras
oraciones.

### C. Contribuciones

[ANDAMIAJE] Rankeadas por defensibilidad, con la evidencia al lado. Cada una nombra explícitamente qué NO reclama, porque la mitad del argumento anterior estaba tomado y conviene que el revisor vea que lo sabemos.

### C. Contribuciones

[ANDAMIAJE — ESCRITO 2026-07-30] El párrafo de "lo que NO afirma", que en este
venue es diferenciador y no debilidad. Versión corta acá; el desarrollo con las
cuatro delimitaciones está en V-B, y no debe duplicarse.

### D. Aporte metodológico, declarado al frente

[ANDAMIAJE — ESCRITO 2026-07-30] Deliberadamente al frente y no escondido en
Métodos: en el relevamiento del venue (ocho artículos, Vol. 15–17), **0 de 8**
reportan un test de significancia pareado, **1 de 8** declara un corte temporal
real y **0 de 8** tienen declaración de disponibilidad de datos o código. Lo
que en un venue más exigente sería higiene, acá es diferenciador — pero solo si
se enuncia. **No** escribir la comparación con el venue en el artículo: es
insumo de decisión editorial, no contenido publicable.

### E. Estructura del artículo

[ANDAMIAJE — ESCRITO 2026-07-30] Un párrafo. Al traducir, verificar que las
letras de subsección sigan coincidiendo si el maquetado fusiona secciones.

### II. Trabajos Relacionados (Related Work)

[ANDAMIAJE — REESTRUCTURADO 2026-07-29] La versión anterior organizaba esta sección alrededor de un *gap* que ya no es el nuestro ("el subcampo reporta ganancias del DL sin *baseline naive* ni significancia"). Las FUENTES se conservan casi todas; el ARGUMENTO se rehace. El orden nuevo va de lo más cercano a lo más general, para que el lector vea primero a quién le estamos hablando.

**Regla de citación para esta sección:** cada referencia debe llevar identificador verificado. `docs/paper/fuentes-verificadas.md` tiene el estado entrada por entrada y una lista explícita de lo que NO se puede citar porque no se pudo recuperar. **No citar afirmaciones específicas de fuentes marcadas `[SNIPPET]` o `[ABSTRACT]` sin leerlas primero.**

### A. Predecir el *headway* y después umbralizar: la familia y su falla documentada

[ANDAMIAJE — ESCRITO 2026-07-30] Citación en autor-año; la conversión a `[n]`
se hace al maquetar. Toda cita textual de esta subsección proviene de fuentes
marcadas `[TEXTO COMPLETO]` en `docs/paper/fuentes-verificadas.md`. Dos
restricciones que se respetaron: (a) a Moreira-Matias et al. (2016) se lo cita
**solo** por la convención de un cuarto del *headway* programado, que es lo
verificado — no por el mecanismo de alarma; (b) el >95 % de Yu et al. no se
cita como tal: se cita el par de cifras reconciliado vía Sun et al. (2021).
Pendiente que toca esta subsección: Jiao, Shen y Zhang (2023) reclama 89 % con
el mismo esquema y sigue en `[ABSTRACT]` con umbral no verificado. **No se cita
hasta leerlo**; si se confirma que su horizonte es corto, entra en el párrafo
de la reconciliación junto a Yu et al.

### B. Por qué el umbral se mueve: sub-dispersión de los pronósticos puntuales

[ANDAMIAJE — ESCRITO 2026-07-30] Restricciones respetadas al redactar:
(a) Gneiting (2011) se cita **solo** por el funcional que elicita cada pérdida
— el preprint completo no contiene ninguna afirmación de sub-dispersión;
(b) Vannitsem y Hagedorn (2011) entra **acreditado a través de** Mayer y Yang,
porque no lo leímos: no se le atribuye texto ni cifras;
(c) Wernli et al. (2009), von Storch (1999), Huth (2002) y Maraun (2013) están
en `[CROSSREF]`/`[SNIPPET]`, así que aparecen parafraseados y **sin comillas**;
(d) el párrafo de la distinción transversal/temporal es obligatorio y no se
puede ablandar: es lo que separa nuestro resultado del teorema.

### C. Recalibrar el umbral: precedentes fuera del transporte

[ANDAMIAJE — ESCRITO 2026-07-30] Subsección obligatoria: sin ella, un revisor
de clima o meteorología nos liquida con una cita de 2018. Restricciones que se
respetaron: (a) a Hoffmann et al. (2018) se lo parafrasea, porque tenemos el
texto completo pero no dejamos registrada ninguna cita textual; (b) Petetin et
al. (2022) es la cita central y se le acredita **todo** lo que tiene, incluido
lo que creíamos nuestro — ver §0.5 de `fuentes-verificadas.md`; (c) Alfieri et
al. (2019), Zsoter et al. (2020) y Lalaurette (2003) quedaron **fuera**: siguen
en `[POR VERIFICAR ANTES DE CITAR]` y Petetin ya cubre su papel. Si se
consiguen, entran como refuerzo, no cambian el argumento.

### D. Qué métrica puede decidir una detección

[ANDAMIAJE — ESCRITO 2026-07-30] Restricciones respetadas al redactar:
(a) **2π/(1+π) se presenta como derivación en un paso**, no como cita — la
fórmula no está impresa en ninguna fuente, y Lipton et al. solo la instancian
numéricamente; (b) la **invariancia del AUC ante transformaciones monótonas**
se deriva de la identidad de rangos, tampoco se atribuye: no figura como
teorema etiquetado en ninguna de las fuentes; (c) para la regla siempre-activa
el MCC es **0/0, indeterminado**, y el cero es extensión por continuidad —
**nunca escribir "cero por construcción"**; (d) Zhu (2020) está en `[ABSTRACT]`
y solo se le atribuye lo que dice el resumen; (e) el DOI de proceedings de
McDermott et al. no está verificado: citar por NeurIPS 2024 / arXiv.
**Pendiente que NO bloquea:** Boughorbel et al. (2017) está en `[CROSSREF]` con
la instrucción de leer el cuerpo antes de citar, así que el argumento de que
ajustar un umbral por métrica es principiado se apoya en Koyejo et al. (2014).
Si se lee Boughorbel, entra como refuerzo. Ídem Itaya et al. (2025), que da
intervalos de confianza para **diferencias pareadas** de MCC y encaja mejor en
la Sección III-E que acá.

### E. Síntesis del vacío

[ANDAMIAJE — ESCRITO 2026-07-30] **La formulación del andamiaje anterior se
quedó corta y hubo que angostarla.** Decía "nadie mide la compresión de
dispersión que causa la falla", y eso **ya no es cierto fuera del transporte**:
Petetin et al. (2022) la miden, la cuantifican y la aparean con métricas
categóricas. El vacío hay que enunciarlo con dos ejes —dominio y tipo de
umbral—, no con uno. Dos reglas que siguen vigentes: **NO escribir "el
subcampo no se dio cuenta"** (no sobrevive a un revisor que conozca a Sun et
al.), y no ensanchar el reclamo más allá del caso auto-referencial.

### Referencias de la Sección II

[ANDAMIAJE] Tomarlas de `docs/paper/fuentes-verificadas.md`, respetando el estado de verificación. Presupuesto observado en IJACSA: 26–46, mediana ≈41.

### III. Materiales y Métodos

[ANDAMIAJE] Casi todo redactado en documento-resultados.md §2 y en el código (`src/evaluation/`). Reordenar y completar el preprocesamiento que el doc de resultados omite a propósito.

### A. Datos: SIT Arequipa (AVL/GPS)

[ANDAMIAJE — TRASLADADO 2026-07-30] Fuente: `documento-resultados.md` §2 y
`docs/dataset-manifest.md`.

### B. Preprocesamiento, población de muestras y definición del *headway*

[ANDAMIAJE — TRASLADADO 2026-07-30] Fuente: `documento-resultados.md` §2 y
`docs/decisiones-headway-fase2.md` §2.1.
**Los tres contratos se renombraron a propósito.** En el documento de
resultados se llaman C1/C2/C3, y acá eso chocaría con las contribuciones C1–C4
de la Sección I. Van por nombre y no por número.
Y al describir el *headway*: usar **la ecuación**, no los nombres de columna del
parquet — `front`/`back` están invertidos respecto del movimiento físico y la
aritmética es correcta, pero los nombres confunden a quien lea el código.

### C. Modelos comparados

[ANDAMIAJE — TRASLADADO 2026-07-30] Fuente: `documento-resultados.md` §2.
**La asimetría de tuning va en el cuerpo, no en una nota al pie.** Es lo que
impide atribuir a la clase de modelo la victoria del XGBoost en E2.

### D. Definición del evento de *bunching*, y por qué esta

[ANDAMIAJE — ESCRITO 2026-07-30] Es la subsección más delicada del artículo:
declara que nuestra regla **no** es la del campo. Restricciones respetadas:
(a) la forma "fracción de la media observada" **no se encontró como definición
de evento publicada** —solo en descripciones de proveedores CAD/AVL, cuya única
fuente localizada es un post de LinkedIn, no citable— así que se declara como
**sustitución nuestra**, con el precedente de Yu et al. para la *clase* de
sustitución, no para la forma; (b) **no** se cita la Ec. 3-7 del TCQSM como
definición de lo que medimos: su `cvh` se normaliza por el *headway*
**programado** y el nuestro por la media observada, así que no son la misma
cantidad (ver §C2 de `fuentes-verificadas.md`); (c) no se afirma nada sobre la
procedencia del cociente ¼ más allá de quién lo usa; (d) la implementación
literal está en `bunching_flags` de `src/evaluation/vector_metrics.py`, y la
prosa tiene que coincidir con ella: la media se toma **sobre la misma columna
que se marca**.

### E. Protocolo de evaluación

[ANDAMIAJE — TRASLADADO 2026-07-30] Fuente: `documento-resultados.md` §4 y
§5.5. Es lo que diferencia al artículo en este venue, así que va desarrollado.

### IV. Resultados

[ANDAMIAJE — REESTRUCTURADO 2026-07-29] El orden cambió. Antes iba: cruce → significancia → volatilidad → enrutador, o sea el resultado escalar como titular. Ahora el escalar es **contexto** y el titular es el artefacto de umbral y su reparación.

Figuras: usar `contiguo-artefacto-umbral.png` y `contiguo-deteccion-sin-umbral.png` (**solo funcionan como par**), más `contiguo-degradacion.png` y `contiguo-volatilidad.png`. **NO usar** `curva-degradacion.png` ni `volatilidad-crossover.png`: son de las familias congeladas y no de este pipeline. La figura `contiguo-disociacion.png` fue eliminada — graficaba F1 con corte fijo como si midiera a los modelos.

### A. Contexto: el resultado escalar y su frontera

[ANDAMIAJE — TRASLADADO 2026-07-30] Doc §3 y §4, comprimido. **Presentar como
contexto establecido, no como aporte** — el cruce por horizonte es folclore
conocido en pronóstico de tráfico. Figuras `contiguo-degradacion.png` y
`contiguo-volatilidad.png`.

### B. La compresión de dispersión, medida

[ANDAMIAJE — TRASLADADO 2026-07-30] Doc §5.2 y §5.5. **La distinción
transversal contra temporal es obligatoria y va como bloque destacado**: es lo
que separa nuestro resultado empírico del teorema de Patton y Timmermann.
Y **no** citar la Ec. 3-7 del TCQSM como definición del CV (§C2 de
`fuentes-verificadas.md`); sí su escala de nivel de servicio.

### C. El artefacto: la alarma no suena

[ANDAMIAJE — TRASLADADO 2026-07-30] Doc §1 y §5.3. Figura
`contiguo-artefacto-umbral.png`. **Va emparejada con la figura de la subsección
D: solo funcionan como par**, y compararlas es el aporte del trabajo.
Precisión obligatoria sobre el MCC de "marcar todo": es **0/0 indeterminado**,
y el cero es extensión por continuidad. Nunca "cero por construcción".

### D. La reparación: sin umbral, el veredicto se invierte

[ANDAMIAJE — TRASLADADO 2026-07-30] Doc §5.4. Figura
`contiguo-deteccion-sin-umbral.png`, que **forma par con la de la subsección C**.
La celda partida (E4 h=5, una milésima) va **nombrada**, no omitida.

### E. Robustez: no es de febrero, ni de nuestro umbral

[ANDAMIAJE — TRASLADADO 2026-07-30] Doc §5.5, §5.6 y §6 (winsorización).
**La salvedad del AUC absoluto va completa**, incluida la celda en 0.4934: es
lo único que corre en contra y omitirla invalidaría la subsección.

### F. Por qué MCC y no F1, medido

[ANDAMIAJE — TRASLADADO 2026-07-30] Doc §5.7. **Reportar el resultado mixto
como mixto.** La formulación "el MCC es más estable" sería falsa en la mediana
y un revisor lo verificaría en la primera fila de la tabla.

### G. El enrutador ex-ante

[ANDAMIAJE — TRASLADADO 2026-07-30] Doc §6. Dos párrafos, que es lo que
amerita. **Decir explícitamente que vale como demostración de ejecutabilidad y
no como contribución** — inflarlo sería el tipo de reclamo que este artículo
critica en otros.

### A. Interpretación operativa

[ANDAMIAJE — ESCRITO 2026-07-30] Es la subsección que le habla al operador.
El eje es la diferencia entre dos diagnósticos que se parecen y tienen
consecuencias opuestas. **El límite honesto va en el mismo párrafo que la
lectura optimista, no en una sección aparte** — si el AUC de 0.60 aparece solo
en Limitaciones, la sección se lee como venta.

### B. Qué queda del aporte, y qué no

[ANDAMIAJE — ESCRITO 2026-07-30] Cuatro delimitaciones, de la más cercana a la
más lejana, dos oraciones cada una. Vale más que reclamar de más. **No ablandar
el párrafo de cierre**: nombra lo que se retiró durante la propia investigación,
y en este venue eso es diferenciador, no debilidad.

### C. El resultado nulo espacial, en contexto

[ANDAMIAJE — ESCRITO 2026-07-30] Corto y enmarcado como **confirmación de
resultado publicado**, nunca como hallazgo. Y con la salvedad de procedencia
que las limitaciones 4 y 5 del documento de resultados obligan a declarar: el
nulo se estableció sobre las familias congeladas, no se rehízo bajo el pipeline
contiguo, y solo el LSTM tiene barrido de semillas.

### D. Amenazas a la validez y limitaciones

[ANDAMIAJE — TRASLADADO 2026-07-30] Las 11 limitaciones del doc §8.
**Mantenerla sustantiva y no recortarla al maquetar**: en el relevamiento del
venue solo 3 de 8 artículos tienen algo parecido. Las tres que más pesan van
primero, no en el orden del documento fuente: valor operativo no modelado,
umbral no calibrado contra incidentes, y calibración sobre dos ventanas
anidadas en lugar de validación cruzada rotativa.

### VI. Conclusión y Trabajo Futuro

[ANDAMIAJE — ESCRITO 2026-07-30] Reescrita desde cero: la versión anterior
cerraba sobre "la métrica decide el ganador", retirado el 2026-07-29. Cierra
sobre el **vacío**, no sobre los resultados — los números están en la Sección IV
y repetirlos acá los devalúa. Se agregó un quinto ítem de trabajo futuro que no
estaba en el guion y que salió de leer a Yu et al.: la **tercera forma de
umbral** (referencia observada y fija) es la que nadie midió, ni ellos ni
nosotros.

### Referencias

[ANDAMIAJE — ESCRITO 2026-07-30] **44 entradas**, dentro del presupuesto
observado del venue (26–46, mediana ≈41). Ordenadas por **primera aparición en
el texto**, que es la convención IEEE; si la traducción reordena secciones, la
renumeración es mecánica. Solo entra lo que la prosa cita: la lista de "no
citar" de `fuentes-verificadas.md` es vinculante y ninguna de esas fuentes
aparece acá.

**PROCEDENCIA DE LOS METADATOS.** Títulos, bylines completos, volúmenes,
números y páginas se **transcribieron** desde la API de Crossref (29 entradas
con DOI) y desde las páginas de arXiv (10 entradas), no desde memoria. Los
scripts que hicieron las consultas quedaron en el *scratchpad* de la sesión.
El primer borrador de esta lista tenía **títulos inventados** —el de Sun et al.
y el de Mayer y Yang no se parecían al real, y el de Rodrigues tampoco—, así que
**ninguna entrada de aquí debe reescribirse de memoria**. Faltan solo [3], [6],
[10], [30] y [33], verificadas por Crossref o por la fuente primaria pero sin
pasar por este lote.

Cuatro avisos que sobreviven a la verificación:
- **[2] Mayer y Yang.** Crossref fecha el número impreso en **2023** (vol. 39,
  n.º 2); la disponibilidad en línea es de 2022, que es el año que usa el
  borrador en castellano. En el formato final IEEE la cita en texto es `[2]`, así
  que la discrepancia desaparece — **pero la entrada debe decir 2023**.
- **[13] Yu, Wu, Chen y Ma.** Mismo caso: el DOI es de 2016 y el número impreso
  es **2017**, vol. 18 n.º 7. El texto en castellano ya dice (2017).
- **[24] Liu et al.** El venue NeurIPS 2022 **sigue sin confirmar**: la página
  de arXiv no trae `journal-ref`. Se cita como preprint.
- **[39] McDermott et al.** El DOI de proceedings no está verificado, pero la
  página de arXiv confirma NeurIPS 2024 con enlace a OpenReview.

**Resuelto de paso: V5.** El orden de autores de Boyd et al. era el pendiente
más molesto y quedó fijado desde arXiv: Boyd, Santos Costa, Davis, Page.

Y una nota de honestidad sobre **[3] Vannitsem y Hagedorn**: se cita de segunda
mano, como atribución de Mayer y Yang. El §II-B no le atribuye texto ni cifras.
Si se lee el original antes de someter, se puede citar de primera mano.

## Bloque de control: Referencias

```
  MAPA DE PROGRESO (borrar antes de enviar):

  [ESCRITO 2026-07-30]       ✅ SECCIÓN II COMPLETA — A, B, C, D y E
                             ✅ III.D Definición del evento
                             ✅ V.A Interpretación · V.B Aporte · V.C Nulo espacial
                             ✅ VI Conclusión y Trabajo Futuro
                             ✅ SECCIÓN I COMPLETA — A, B, C, D y E
                             ✅ Abstract (235 palabras) · Título · Keywords
                             ✅ Referencias (44, transcritas de Crossref/arXiv)
                             ✅ III.A-C y III.E trasladadas · IV.A-G trasladadas
                             ✅ V.D Limitaciones

  ESTADO: ✅ EL BORRADOR EN CASTELLANO ESTÁ COMPLETO. Todas las secciones tienen
  prosa. Lo que queda no es escribir:
    (a) borrar TODOS los bloques [ANDAMIAJE] y este mapa;
    (b) traducir al inglés, recontando el Abstract — el inglés comprime;
    (c) completar la plantilla IJACSA a dos columnas (≤10 pp de cuerpo);
    (d) renumerar las referencias si la traducción reordena secciones.

  VERIFICAR ANTES DE SOMETER, y no es opcional:
    - Cada cifra de la Sección IV contra documento-resultados.md. Se trasladaron
      a mano y un dígito cambiado invalida un argumento.
    - Las figuras: contiguo-artefacto-umbral.png y contiguo-deteccion-sin-umbral.png
      SOLO funcionan como par (IV.C e IV.D), más contiguo-degradacion.png y
      contiguo-volatilidad.png en IV.A. NO usar curva-degradacion.png ni
      volatilidad-crossover.png: son de las familias congeladas, no de este pipeline.

  NOTA para el Abstract: la Conclusión NO repite cifras a propósito. El Abstract
  SÍ tiene que traerlas (los candidatos están en el andamiaje del Abstract), y es
  el único lugar donde el lector las ve antes de la Sección IV.

  Las dos subsecciones que decidían si el paper se lee como honesto o como ingenuo
  —II-A y II-C— acreditan de frente lo que no es nuestro. II-E enuncia el vacío en
  DOS ejes (dominio y tipo de umbral), no en uno: la formulación de un solo eje se
  volvió falsa cuando se leyó Petetin.

  SIGUIENTE: V.A / V.B / V.C (Discusión), después VI Conclusión, después I e
  Introducción y Abstract al final. Las Referencias pueden ir en paralelo.
  Ya no quedan bloqueantes de literatura para nada de eso.

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
```
