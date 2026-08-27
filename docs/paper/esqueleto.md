# Paper — esqueleto de trabajo

Destino: **IJACSA**. Idioma de redacción: español; traducción al inglés al final.
Reemplaza a `manuscrito.md`, que se conserva **solo como cantera de material**.

Insumos vinculantes:

- `docs/resultados/documento-resultados.md` — todas las cifras salen de acá
- `docs/paper/notas-de-redaccion.md` — prohibiciones de redacción, siguen vigentes
- `docs/paper/fuentes-verificadas.md` — estado de verificación de cada referencia

---

## 1. La afirmación central

> Cuando un modelo parece incapaz de detectar el apelotonamiento de buses, muchas
> veces no falla el modelo: falla la regla con la que se lo mide. Un pronóstico
> siempre sale más parejo que la realidad, así que una alarma calibrada sobre la
> realidad casi nunca se dispara sobre el pronóstico. El modelo parece ciego sin
> serlo. **Se arregla moviendo la regla, no cambiando el modelo.**

Todo lo que entra al paper sostiene esta frase o la acota. Lo que no hace ninguna
de las dos cosas, sale.

**Título de trabajo:** *El umbral, no el modelo: por qué un pronóstico de
headways parece ciego al apelotonamiento, y cómo se repara.*

---

## 2. Reglas de escritura

### 2.1 La prosa no recorre celdas

La prosa **enuncia el patrón y apunta a la tabla**. Nunca narra celda por celda.
Si un párrafo enumera más de tres cifras seguidas, ese párrafo es una tabla mal
escrita.

### 2.2 Política de jerga

Cada término del oficio **se gana su lugar o se traduce**. Un término se conserva
solo si (a) es el nombre estándar en el campo y el revisor lo espera, o (b) no
existe forma corta de decirlo. Todo lo demás se dice en palabras normales.

| No escribir | Escribir |
|---|---|
| winsorización | le pusimos un tope al 1 % más alto |
| compresión de dispersión | el pronóstico sale más parejo que la realidad |
| umbral auto-referencial | el corte se mide contra el promedio del propio pronóstico |
| población compartida verificable | los tres modelos se puntúan sobre exactamente las mismas filas |
| coeficiente de variación transversal | qué tan disparejos están los headways de un corredor en un instante |

Se conservan, definidos **una sola vez** y en una cláusula: **headway**,
apelotonamiento (*bunching*), correlación de Matthews, área bajo la curva ROC,
Diebold-Mariano.

**La regla ataca la jerga que oculta, no el vocabulario canónico del campo.** Son
dos cosas distintas y confundirlas cuesta caro en direcciones opuestas:

| Clase | Ejemplo | Qué hacer | Por qué |
|---|---|---|---|
| Jerga que **oculta** una idea simple | winsorización | Traducir | La palabra no aporta nada que no diga «tope al 1 % más alto» |
| Término **canónico** del campo | headway | Conservar | Es el nombre del objeto de estudio; el revisor lo espera |

Decidido 2026-08-27: **el objeto de estudio se llama `headway` en todo el texto,
y `intervalo` no se usa como sinónimo.** Un artículo de transporte que escribe
«intervalo entre buses» en vez de «headway» le señala al revisor que el autor
viene de fuera del campo, y en la versión en inglés sería descalificante. La
figura `esquema-headway` ya usaba el término correcto mientras el texto no: ese
desacuerdo fue el síntoma.

⚠️ Al reemplazar, **`intervalo de confianza` no se toca** —es el intervalo
estadístico— y `intervalo programado` pasa a `headway programado`, que es el
*scheduled headway* de la convención del campo.

Regla de cierre: **una sigla no definida en la oración donde aparece por primera
vez es un error, no un estilo.**

### 2.3 Ningún dato se enuncia sin su consecuencia

Un hecho metodológico escrito como acta de laboratorio no le sirve a nadie. Todo
dato del paper se enuncia **junto a por qué le importa al lector**, y hay dos
lectores distintos: el que evalúa y el que opera.

Ejemplo del error y su corrección:

| Acta de laboratorio | Lo mismo, con su consecuencia |
|---|---|
| "No se reentrenó nada, no se modificó ninguna arquitectura." | "El campo responde a una falla de detección cambiando de modelo. Acá no hizo falta: se movió un número y el ganador cambió de bando. Como nada más varió, nada más puede explicar la inversión —y reparar esto no cuesta una GPU, cuesta recalibrar." |

Esto **no es inflar el resultado**. La cifra es la misma y el alcance es el mismo;
lo que cambia es que el lector entiende qué hacer con ella. Inflar sería afirmar
más de lo que el dato sostiene, y eso está prohibido en todo el documento.

Prueba a aplicar en cada párrafo: *si el lector pregunta "¿y eso para qué me
sirve?", ¿la respuesta está en el mismo párrafo?* Si no está, el párrafo está a
medio escribir.

### 2.4 El paper nunca habla de su propia revisión

Prohibido escribir «conviene decirlo antes de que lo diga un revisor», «un
revisor podría objetar», «para anticipar críticas». Un artículo publicado no
menciona a sus revisores: rompe el marco, y además delata una motivación
equivocada —parece que se cede por miedo a que descubran algo, no porque sea el
estado del conocimiento.

La razón para ceder autoría se enuncia **hacia el lector**, no hacia el revisor:

| Mal | Bien |
|---|---|
| «conviene decirlo antes de que lo diga un revisor» | «delimitar qué es previo es lo que deja a la vista la contribución» |

El razonamiento sobre revisores va en este esqueleto, que nadie publica. En el
paper va solo su conclusión.

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

---

### Resumen

Único lugar del paper donde el lector ve cifras antes de la Sección V. La
Conclusión **no las repite**.

En este orden: el problema operativo · el mecanismo · el número que duele (14
alarmas contra 15 245 eventos reales) · la reparación · el alcance.

Sin citas. Sin siglas sin desarrollar.

---

### I. Introducción

**A. Contexto** — El apelotonamiento como problema operativo real. Cierra
sobre el paradigma que el paper ataca: predecir el headway y compararlo contra un
umbral.

**B. El problema** — Ese paso —comparar contra el umbral— nunca se examinó.

> Prohibido: decir «nadie se dio cuenta». Sun et al. (2021) diagnosticaron el
> síntoma y se citan en las dos primeras oraciones.

**C. Contribuciones** — **Cuatro viñetas, una oración cada una.** La
delimitación bibliográfica NO se hace aquí: vive en II-D.

**D. Estructura** — Un párrafo, no una lista.

---

### II. Trabajos relacionados

**A. La receta estándar** — Predecir el headway y umbralizarlo contra la
referencia. Yu et al. (2016) es la formulación canónica.

**B. Por qué el umbral se mueve** — Mayer y Yang (2022) lo enuncian; Patton
y Timmermann (2012) lo prueban como teorema; Petetin et al. (2022) ya ataron la
compresión a una métrica categórica.

**C. Recalibrar el corte: precedente fuera del transporte** — Hoffmann, Menz
y Spekat (2018), en clima, ocho años antes. **Obligatoria.**

**D. Qué es previo y qué no** — ✅ **ESCRITA.** Se movió aquí desde la
antigua V-B, donde duplicaba lo que esta sección hace por definición.

> Prohibido: ensanchar el reclamo más allá del caso relativo y auto-referencial.

---

### III. Método propuesto

**A. Del GPS al headway** — ✅ **ESCRITA.** El eje ajustado desde los
propios datos, la proyección, el sentido, los viajes, la rejilla y el cruce por
posición. Tiene margen para crecer: es el aporte metodológico, y el revisor lo va
a mirar con lupa precisamente por no ser estándar.
→ **Fig. 1**

**B. Qué cuenta como apelotonamiento** — ✅ **ESCRITA.** La bisagra
del paper. Cierra dejando explícito que el corte se mide contra el promedio del
propio vector, de modo que no es el mismo corte cuando cambia la dispersión.

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
→ **Fig. 2**

---

### V. Resultados y discusión

✅ **ESCRITA.** Subsecciones A–G. La G es la interpretación operativa,
que en este molde vive dentro de Resultados.

| | Contenido | Evidencia |
|---|---|---|
| A | El resultado escalar y su frontera | — |
| B | El pronóstico sale más parejo | Fig. 3, Fig. 4 |
| C | La alarma no suena | Fig. 5, Tabla 1 |
| D | Ese factor no mide al modelo | — |
| E | La reparación | Fig. 6, Tabla 2 |
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
| **Fig. 2** — la partición temporal y los tres orígenes (IV-C) | Barrido de semillas |
| **Fig. 3 y 4** — compresión: a diez minutos, y contra el horizonte (V-B) | El enrutador (7 de 12 políticas degeneradas) |
| **Fig. 5 + Tabla 1** — el artefacto y el piso del detector trivial (V-C) | Tablas escalares completas de MAE/RMSE |
| **Fig. 6 + Tabla 2** — la reparación, con y sin umbral (V-E) | Configuraciones ganadoras de la búsqueda |
| **Tabla 3** — robustez: 3 ventanas + el ataque con umbral absoluto (V-F) | |

Las Fig. 5 y 6 **funcionan solo como par**: compararlas es el aporte. Publicar
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
| 3 | **Resolver 11/12 vs 10/12.** El manuscrito dice 11, la metodología dice 10, la síntesis dice 10. Una de las tres está mal. | CSVs | 15 min |
| 4 | ~~Resolver 110× vs 115×~~ **RESUELTO 2026-08-26: es 115×.** Recomputado de `threshold_absolute_comparison.csv`: mediana de marcado relativo 0,078775 ÷ absoluto a ρ=0,25 0,000682 = **115,4**. `metodologia.md` tenía razón; `manuscrito.md` decía 110 y estaba mal. Ya corregido en `paper.md` §IV-F. | — | hecho |
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
      pie de la Fig. 6 contra V-F.
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
