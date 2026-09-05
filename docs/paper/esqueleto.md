# Paper — esqueleto de trabajo

Destino: **IJACSA**. Idioma de redacción: español; traducción al inglés al final.

Este documento cubre **estructura, evidencia y estado**. Las reglas de redacción
no viven acá: son la skill `redaccion-paper`
(`.claude/skills/redaccion-paper/`), y su archivo `references/reglas-redaccion.md`
es la única norma vinculante sobre cómo se escribe.

Insumos vinculantes:

- `docs/resultados/documento-resultados.md` — todas las cifras salen de acá
- `docs/paper/fuentes-verificadas.md` — estado de verificación de cada referencia

---

## 1. La afirmación central

> Cuando un modelo parece incapaz de detectar el bunching de buses, muchas
> veces no falla el modelo: falla la regla con la que se lo mide. Una predicción
> siempre sale más parejo que la realidad, así que una alarma calibrada sobre la
> realidad casi nunca se dispara sobre la predicción. El modelo parece ciego sin
> serlo. **Se arregla moviendo la regla, no cambiando el modelo.**

Todo lo que entra al paper sostiene esta frase o la acota. Lo que no hace ninguna
de las dos cosas, sale.

**Título de trabajo:** *El umbral, no el modelo: por qué una predicción de
headways parece ciego al bunching, y cómo se repara.*

---

## 1-bis. Léxico fijado

**El sustantivo es «predicción». «Pronóstico» no se usa.** Fijado 2026-09-05 tras
medirlo, porque no eran dos conceptos: el paper decía «Una **predicción** que
minimiza error cuadrático tiende…» en III-B y «Un **pronóstico** ajustado para
minimizar el error cuadrático medio…» en II-B — la misma afirmación con dos
sustantivos. La distinción fina la cargan los adjetivos que ya existen:
**puntual** contra **vectorial**.

La evidencia, en títulos indexados por OpenAlex:

| Dominio | forecast* | predict* |
|---|---:|---:|
| Global | 334 929 | **998 043** |
| Flujo de tránsito | 1 624 | **3 677** |
| Llegada de bus | 7 | **239** |
| Headway | 6 | **33** |
| **Bus bunching** | **0** | **12** |
| Títulos en español + transporte | 7 | **29** |

En «bus bunching» *forecast* no aparece en ningún título. Y de los once PDF del
repo, ocho usan `predict`; los tres que prefieren `forecast` son econometría de
pronóstico, no transporte. El título del paper ya usaba «predicción».

⚠️ **Excepción registrada:** `src/build_contiguous_figures.py` conserva el término
en las leyendas de la variante *chrome*, que van a `documento-resultados.md` y
**nunca al paper** (`_resolve`, líneas 218-240). No tocar sin revisar ese
documento.

---

## 2. Numeración de ecuaciones vigente

Asignación actual, toda en la Sección III:

| Ec. | Qué define | Dónde |
|---|---|---|
| (1) | proyección al eje: `s(p)`, `ℓ(p)` | III-A, paso 2 |
| (2) | sentido de marcha: `d` | III-A, paso 3 |
| (3) | el headway: `t_c`, `h` | III-A, paso 6 |
| (4) | la tarea de predicción: `ĥ(t+H) = f(·)` | III-B |
| (5) | el objetivo de error cuadrático: `L` | III-B |
| (6) | promedio y corte: `h̄(t)`, `τ(t)` | III-C |
| (7) | indicador de bunching sobre lo observado: `b_i(t)` | III-C |
| (8) | el detector evaluado, sobre la predicción: `b̂_i(t)` | III-C |
| (9) | `τ(ĥ) ≠ τ(h)` | III-C |

Esta tabla es estado, no norma: registra qué número tiene hoy cada ecuación. Las
reglas de numeración y de referencia cruzada viven en la skill.

---

## 3. Estructura

**Molde: NADOS** (*Reliability-Gated Difficulty-Aware Oversampling*, IJACSA
vol. 17 n.º 7). Elegido el 2026-08-27 tras medir las cinco estructuras del número.
Tres razones, todas verificadas contra el PDF:

- **Separa el método del diseño experimental.** Es el único de los cinco que lo
  hace, y es exactamente la división que tiene este trabajo: una cosa es construir
  el headway y definir el evento, y otra son los tres orígenes, la población
  compartida, la continuidad estricta, el tope y la agrupación por día.
- **Amenazas a la validez con número propio.** Hay material de sobra.
- **Declaraciones** aloja la reproducibilidad, que casi nadie en ese venue puede
  escribir.

Y el dato que decide: **NADOS es el único de los cinco que reporta significancia
estadística.** Mismo perfil metodológico que el nuestro.

Su coste: Resultados y Discusión van fusionados, así que la interpretación
operativa vive dentro de Resultados (V-G) y no en sección propia.

**Regla de reparto, decidida el 2026-08-27 después de romperla.** La prueba es:
*¿esto existiría igual si nadie lo hubiera evaluado nunca?* Si sí, va a **III**; si
existe solo para producir un número, va a **IV**. En consecuencia **III no
cuantifica el dataset**: no dice cuántas unidades, ni cuántas posiciones, ni la
cobertura, ni usa los códigos de corredor. Puede sí enunciar la restricción que
motiva el método —no hay tabla de paradas—, porque eso es el problema, no el
inventario.

El síntoma cuando se rompe: III-A citaba «43,4 millones de posiciones de 90
unidades» y «tres millones de headways en E2 y E59» **antes** de que IV-A
presentara la ciudad, las tres empresas y el significado de E2. El lector se
encontraba las cifras y los códigos sin haberlos conocido. Todo eso se movió a IV-A,
donde además cierra la subsección con el rendimiento del procedimiento.

---

### Resumen

Único lugar del paper donde el lector ve cifras antes de la Sección V. La
Conclusión **no las repite**.

En este orden: el problema operativo · el mecanismo · el número que duele (14
alarmas contra 15 245 eventos reales) · la reparación · el alcance.

Sin citas. Sin siglas sin desarrollar.

---

### I. Introducción

**A. Contexto** — El bunching como problema operativo real. Cierra
sobre el paradigma que el paper ataca: predecir el headway y compararlo contra un
umbral.

**B. El problema** — Ese paso —comparar contra el umbral— nunca se examinó.

> Prohibido: decir «nadie se dio cuenta». Sun et al. (2021) diagnosticaron el
> síntoma y se citan en las dos primeras oraciones.

> **La cita de Sun, Schmöcker y Nakamura (2021) va acá.** Diagnosticaron que el
> paradigma de predecir y umbralizar falla y que el veredicto se revierte al
> puntuar sin punto de operación. Van en las dos primeras oraciones del planteo
> del problema. Hoy esa atribución vive en el primer párrafo de la II-D, que se
> borra cuando esta sección se escriba.
>
> **Y ojo al escribirla:** la primera aparición del TCQSM es hoy la tercera viñeta
> de las contribuciones, y ahí va desarrollada la sigla. La Sección III-C la usa
> ya abreviada. Si el planteo del problema nombra el manual antes, el desarrollo
> se mueve allí y la viñeta pasa a usar la sigla.

**C. Contribuciones** — ✅ **ESCRITA.** Cierra la sección, como exige el cuarto
elemento del flujo de la Sección 4 de `reglas-redaccion.md`. Situación,
complicación y pregunta siguen pendientes.

**D. Estructura** — Un párrafo, no una lista.

---

### II. Trabajos relacionados

**A. Predicción del headway y detección por umbral** — ✅ **ESCRITA.** Yu et al.
(2016) da la formulación canónica; la Tabla 1 de Santos et al. (2022) muestra que
ninguno de los ocho trabajos que resume puntúa el ordenamiento sin umbral;
Manibardo et al. (2022) acota lo que vale la primera etapa. Título anterior, «La
receta estándar», retirado por metáfora (Sección 4 de `reglas-redaccion.md`).

**B. Compresión de la dispersión de la predicción** — ✅ **ESCRITA.** Mayer y Yang
la enuncian y la cuantifican; el Corolario 2 de Patton y Timmermann la ordena por
horizonte; Petetin et al. documentan su daño sobre una métrica categórica.

> **El solapamiento con la III-B se resolvió así.** La III-B citaba a Patton y
> Timmermann por el teorema y ahora remite a la II-B, que es donde vive la
> atribución. Lo que la II-B aporta y la III-B no tenía es el **Corolario 2**: la
> compresión crece al alargar el horizonte, y eso es teorema, no hallazgo nuestro.
>
> Título anterior, «Por qué el umbral se mueve», retirado por coloquial. El
> término elegido es el que ya usan la V-B y la III-B: compresión de la
> dispersión. «Sub-dispersión» viene de `fuentes-verificadas.md` y el manuscrito
> no la usa en ninguna parte.
>
> **Ojo con el año de Mayer y Yang.** El PDF y Crossref dicen *39(2):981–991,
> abril de 2023*; el DOI lleva `2022` porque es el año del registro. Todo el
> repositorio lo llamaba «Mayer & Yang (2022)». La clave es `mayer2023` y el
> archivo se renombró a `mayer2023.pdf`.

**C. Recalibrar el umbral: precedente fuera del transporte** — ✅ **ESCRITA.** Dos
familias de remedio agrupadas por qué objeto tocan: Hoffmann et al. (2018) mueve
el umbral, Petetin et al. (2022) mueve la predicción con mapeo de cuantiles.

> **Contra-argumento que hay que sostener.** Hoffmann dice que un indicador
> definido sobre un cuantil de la distribución de referencia queda libre de sesgo
> por construcción. Un revisor puede preguntar por qué el nuestro, siendo también
> relativo, sí se rompe. La respuesta está en el tercer párrafo de la II-C: el
> suyo es un cuantil de lo observado, el nuestro una fracción del promedio de lo
> predicho.

**D. Delimitación de lo previo** — ✅ **ESCRITA.** Se movió aquí desde la
antigua V-B, donde duplicaba lo que esta sección hace por definición. Ya cedió a
la II-A la atribución de Manibardo y a la II-C la de Hoffmann. Cuando se escriba
la II-B cede también las de Mayer y Yang, y Patton y Timmermann, y queda solo con
el contraste.

> Prohibido: ensanchar el reclamo más allá del caso relativo y auto-referencial.

> **La hoja de ruta de la Sección II** se escribe cuando existan las cuatro
> subsecciones, no antes.

---

### III. Método propuesto

**Orden fijado el 2026-08-27: headway → predicción → bunching.** Sigue el flujo
real del dato, y no es solo cosmético: con bunching al final, el lector llega a la
definición del evento con `h` y `ĥ` ya sobre la mesa, así que las Ecuaciones (7) y
(8) se definen juntas. En el orden anterior —A headway, B bunching— la subsección
del evento tenía que invocar `ĥ` sin que nada lo hubiera definido.

**A. Del GPS al headway** — ✅ **ESCRITA.** El eje ajustado desde los
propios datos, la proyección, el sentido, los viajes, la rejilla y el cruce por
posición, en seis pasos numerados. Ecuaciones (1)–(3).
→ **Fig. 1**

Pendiente: los cuatro parámetros del procesamiento que hoy quedan implícitos —el
umbral de 10 km/h para «en movimiento», las **dos** estrategias de ajuste del eje
(PCA por sentido en dos corredores, PCA única en el tercero), los 50 bins del
ajuste, y que la velocidad se deriva del desplazamiento y no se lee del campo del
proveedor. El segundo es el que un revisor va a pedir: hoy el eje se presenta como
un procedimiento único cuando son dos.

**B. La tarea de predicción** — ✅ **ESCRITA.** Qué se recibe, qué se emite, los
cuatro horizontes directos, y el objetivo de error cuadrático con su consecuencia:
tiende a la media condicional, que es más pareja que la realidad. Ecuaciones (4)–(5).

> El título NO dice «LSTM». Acá va la **tarea**; la configuración entrenada
> —32 unidades, Adam, semilla 42— vive en IV-B. El mecanismo del objetivo se movió
> aquí desde IV-B, donde era una propiedad del objetivo escondida en la resaca de
> una tabla de hiperparámetros.

**C. Qué cuenta como bunching** — ✅ **ESCRITA.** La definición del evento sobre lo
observado y el detector compuesto sobre la predicción, definidos uno al lado del
otro, y de ahí que el corte no sea el mismo cuando cambia la dispersión.
Ecuaciones (6)–(9).
→ **Fig. 2 y 3**

---

### IV. Diseño experimental

**A. Los datos** — ✅ **ESCRITA.** Cadencia de 20 s, 152 días, tres
corredores, y la ausencia de horario, GTFS y tabla de paradas.

**B. Los métodos comparados** — ✅ **ESCRITA.** Los cuatro, con el promedio
histórico como competidor real a horizonte largo. Aquí se declara la asimetría de
búsqueda.

**C. Protocolo de evaluación** — ✅ **ESCRITA.** Partición por fecha, tres
orígenes, y las cuatro reglas: continuidad, población compartida, tope al 1 % y
varianza agrupada por día.
→ **Fig. 4**

---

### V. Resultados y discusión

✅ **ESCRITA.** Subsecciones A–G. La G es la interpretación operativa,
que en este molde vive dentro de Resultados.

| | Contenido | Evidencia |
|---|---|---|
| A | El resultado escalar y su frontera | — |
| B | La predicción sale más pareja | Fig. 5, Fig. 6 |
| C | La alarma no suena | Fig. 7, Tabla 1 |
| D | Ese factor no mide al modelo | — |
| E | La reparación | Fig. 8, Tabla 2 |
| F | Robustez y el ataque a nosotros mismos | Tabla 3 |
| G | Qué significa para quien opera un corredor | — |

---

### VI. Amenazas a la validez

✅ **ESCRITA.** Se redacta como **alcance, no como disculpa**: cada entrada
delimita dónde vale la afirmación en vez de pedir perdón por dónde no.

Entra sin excepción: la derrota contra el promedio histórico en E2 h10 · la
tensión AUC/MCC en E59 h5 · el AUC de azar bajo corte absoluto · la asimetría de
búsqueda · el umbral sin calibrar contra incidentes · tres corredores y cinco
meses con Carnaval dentro · los orígenes anidados · y que esto no está listo para
operar una alarma.

---

### VII. Conclusión

Cierra sobre **el hueco**, no sobre los resultados. Las cifras están en la Sección
V y repetirlas las devalúa.

Termina con la prescripción: reportar al menos una métrica sin umbral junto a la
métrica de alarma, publicar el piso del detector trivial, y declarar en qué
espacio se calibró el corte.

---

### VIII. Declaraciones

Disponibilidad de datos y de código. **Es regalo:** hay manifiestos con huellas
SHA-256, semillas fijas, builders deterministas y un gate de población compartida
que aborta antes de tocar la GPU. De los cinco papers medidos, **uno solo** tiene
esta sección.

---

## 4. Evidencia: qué entra al cuerpo y qué al apéndice

| Cuerpo | Apéndice / suplementario |
|---|---|
| **Fig. 1** — la definición del headway (III-A) | Desglose por dirección (+1 / −1) |
| **Fig. 2 y 3** — el corte se mueve con el vector (III-B) | |
| **Fig. 4** — la partición temporal y los tres orígenes (IV-C) | Barrido de semillas |
| **Fig. 5 y 6** — compresión: a diez minutos, y contra el horizonte (V-B) | El enrutador (7 de 12 políticas degeneradas) |
| **Fig. 7 + Tabla 1** — el artefacto y el piso del detector trivial (V-C) | Tablas escalares completas de MAE/RMSE |
| **Fig. 8 + Tabla 2** — la reparación, con y sin umbral (V-E) | Configuraciones ganadoras de la búsqueda |
| **Tabla 3** — robustez: 3 ventanas + el ataque con umbral absoluto (V-F) | |

Las Fig. 7 y 8 **funcionan solo como par**: compararlas es el aporte. Publicar
cualquiera de las dos sola tergiversa el resultado.

Grilla completa en el cuerpo: **3 corredores × 4 horizontes.** El horizonte no es
un eje de reporte, es la evidencia de causa: la compresión y el colapso aparente
crecen juntos. Y conservar el horizonte donde el modelo **pierde** es lo que hace
creíble el resto.

---

## 5. Pendientes antes de redactar

Ninguno requiere GPU ni Kaggle. Los números ya están calculados y commiteados.

| # | Qué | Dónde | Costo |
|---|---|---|---|
| 1 | **Promedio histórico horario:** entra a III-C y a V-C. El aprendiz le pierde en E2 a diez minutos. Hoy no aparece ni una vez en el manuscrito. | `contiguous_ha_paired.csv` | 20 min |
| 2 | **Doble estándar media/mediana:** el paper se niega a dar veredicto en h=3 por disociación media-mediana, pero usa la afirmación de error cuadrático en h=1 que tiene la misma patología (Wilcoxon 0,77 / 1,00 / 1,00). Se borra la afirmación: no sostiene la tesis del umbral. | `contiguous_significance.csv` | 10 min |
| 3 | **Resolver 11/12 vs 10/12.** `paper.md` dice 11; los dos documentos que decían 10 se borraron el 2026-08-28 sin que se dirimiera cuál era el correcto. Hay que recontarlo contra los CSV y corregir el manuscrito si hace falta. | CSVs | 15 min |
| 4 | ~~Resolver 110× vs 115×~~ **RESUELTO 2026-08-26: es 115×.** Recomputado de `threshold_absolute_comparison.csv`: mediana de marcado relativo 0,078775 ÷ absoluto a ρ=0,25 0,000682 = **115,4**. Ya corregido en `paper.md` §IV-F. | — | hecho |
| 5 | `fase-15.md` todavía presenta como "el resultado más estable del trabajo" la lectura que se **retiró**. Poner banner de alcance. | `docs/proceso/fase-15.md` | 10 min |

---

## 6. Estado

**Estructura fijada el 2026-08-27 sobre el molde NADOS.** La numeración anterior
(I–VI, con Datos y método juntos y Discusión y limitaciones juntas) queda obsoleta:
cualquier nota que la cite está desactualizada.

- [x] Afirmación central fijada
- [x] Estructura elegida y justificada contra los cinco moldes del número
- [x] Reparto cuerpo / apéndice
- [ ] Pendientes §5 (los pendientes 2 y 4 están resueltos)

Redacción, sección por sección:

| Sección | Estado |
|---|---|
| Resumen | ⬜ se escribe al final |
| I. Introducción | ⬜ |
| II. Trabajos relacionados | 🟨 solo II-D |
| III. Método propuesto | ✅ |
| IV. Diseño experimental | ✅ |
| V. Resultados y discusión | ✅ verificada 23/23 |
| VI. Amenazas a la validez | ✅ |
| VII. Conclusión | ⬜ |
| VIII. Declaraciones | ⬜ |
| Referencias | ⬜ **bloqueada** |

- [x] **Figuras 1–6 y Tablas 1–3 embebidas.** Las figuras salen de
      `src/build_contiguous_figures.py` y `src/build_schematic_figures.py` a
      `docs/paper/figuras/*.{es,en}.png`; las tablas de
      `src/build_paper_tables.py` a `docs/paper/tablas/*.md`. **Ningún número del
      manuscrito se escribe a mano.** Los nombres de archivo no llevan número de
      figura, y por eso el remapeo a NADOS no obligó a regenerar nada.
- [ ] **Los 8 defectos de la lectura crítica del 2026-08-26.** Los dos graves
      siguen abiertos: la sobreafirmación de V-E que la Tabla 2 contradice, y el
      pie de la Fig. 8 contra V-F.
- [ ] **Referencias.** `paper.md` tiene **cero citas**. `fuentes-verificadas.md`
      tiene 23 marcas de pendiente contra 7 verificadas, y hay antecedente de
      títulos inventados en un borrador previo: **ninguna entrada se reescribe de
      memoria.** Bloquea la Sección II.
- [ ] Traducción al inglés. Un solo comando cambia las seis rutas de figura:

      ```bash
      sed -i 's/\.es\.png/.en.png/g' docs/paper/paper.md
      ```

      Las tablas se re-emiten cambiando `DECIMAL_SEP` a `"."` y las cabeceras en
      `src/build_paper_tables.py`.
- [ ] Plantilla IJACSA a dos columnas
- [ ] Referencias en formato del venue
