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
intervalos parece ciego al apelotonamiento, y cómo se repara.*

---

## 2. Reglas de escritura

### 2.1 Presupuesto

Techo IJACSA: 10 páginas de cuerpo, **sin contar referencias, tablas ni figuras**.
El recurso escaso es la prosa. La evidencia se mueve a tablas y figuras, que son
baratas.

| Sección | Palabras |
|---|---|
| Resumen | 200 |
| I. Introducción | 800 |
| II. Trabajos relacionados | 850 |
| III. Datos y método | 1 400 |
| IV. Resultados | 2 200 |
| V. Discusión y limitaciones | 900 |
| VI. Conclusión | 350 |
| **Total** | **≈ 6 700** |

Si una sección se pasa, el excedente sale de esa sección. No se compensa entre
secciones.

### 2.2 La prosa no recorre celdas

La prosa **enuncia el patrón y apunta a la tabla**. Nunca narra celda por celda.
Si un párrafo enumera más de tres cifras seguidas, ese párrafo es una tabla mal
escrita.

### 2.3 Política de jerga

Cada término del oficio **se gana su lugar o se traduce**. Un término se conserva
solo si (a) es el nombre estándar en el campo y el revisor lo espera, o (b) no
existe forma corta de decirlo. Todo lo demás se dice en palabras normales.

| No escribir | Escribir |
|---|---|
| winsorización | le pusimos un tope al 1 % más alto |
| compresión de dispersión | el pronóstico sale más parejo que la realidad |
| umbral auto-referencial | el corte se mide contra el promedio del propio pronóstico |
| población compartida verificable | los tres modelos se puntúan sobre exactamente las mismas filas |
| coeficiente de variación transversal | qué tan disparejos están los intervalos de un corredor en un instante |

Se conservan, definidos **una sola vez** y en una cláusula: *headway* / intervalo,
apelotonamiento (*bunching*), correlación de Matthews, área bajo la curva ROC,
Diebold-Mariano.

Regla de cierre: **una sigla no definida en la oración donde aparece por primera
vez es un error, no un estilo.**

### 2.4 Ningún dato se enuncia sin su consecuencia

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

---

## 3. Estructura

### Resumen — 200 palabras

Único lugar del paper donde el lector ve cifras antes de la Sección IV. La
Conclusión **no las repite**.

Debe contener, en este orden: el problema operativo · el mecanismo · el número
que duele (14 alarmas contra 15 245 eventos reales) · la reparación · el alcance.

Sin citas. Sin siglas sin desarrollar.

---

### I. Introducción — 800 palabras

**A. Contexto** (200) — El apelotonamiento como problema operativo real. Cierra
sobre el paradigma que la sección siguiente ataca: predecir el intervalo y después
compararlo contra un umbral.

**B. El problema** (250) — Ese paso —comparar contra el umbral— nunca se examinó.
La pregunta del paper es por qué falla y si basta con recalibrarlo.

> Prohibido: decir "nadie se dio cuenta". Sun et al. (2021) diagnosticaron el
> síntoma y se citan en las dos primeras oraciones. Prohibido también quejarse de
> que el subcampo no usa líneas base ni tests de significancia: eso va en I-C
> como aporte, no acá como reclamo.

**C. Contribuciones** (250) — **Cuatro viñetas, una oración cada una.** En el
manuscrito viejo esto ocupaba 1 002 palabras porque cada viñeta arrastraba su
propio párrafo de delimitación bibliográfica. La delimitación se hace **una vez**,
en V-B, no cuatro veces acá.

1. Medimos que el pronóstico sale más parejo que la realidad, sobre el vector de
   intervalos y en todas las celdas medidas.
2. Mostramos que trasplantar el umbral produce un veredicto invertido, y que
   recalibrar el corte —sin tocar el modelo— lo endereza.
3. Reportamos el piso del detector trivial, que el subcampo no publica.
4. El método completo funciona sin horario publicado, sin GTFS y sin tabla de
   paradas.

**D. Estructura** (100) — Un párrafo. No una lista.

---

### II. Trabajos relacionados — 850 palabras

**El recorte más grande del paper: de 3 923 a 850.**

Lo que se conserva son los tres movimientos, un párrafo cada uno:

**A. La receta estándar** (250) — Predecir el intervalo y umbralizarlo contra la
referencia. Yu et al. (2016) es la formulación canónica y se cita textual.

**B. Por qué el umbral se mueve** (300) — Un pronóstico óptimo en error cuadrático
sale más parejo que la realidad. Esto **está publicado y no lo reclamamos**:
Mayer y Yang (2022) lo enuncian; Patton y Timmermann (2012) lo prueban. Petetin
et al. (2022) ya ataron la compresión a una métrica categórica.

**C. Recalibrar el corte: precedente fuera del transporte** (200) — Hoffmann,
Menz y Spekat (2018) hacen exactamente este procedimiento en clima, ocho años
antes.

> Esta subsección es **obligatoria**. Sin ella, un revisor de meteorología liquida
> el paper con una cita de 2018.

**D. El hueco** (100) — Dos ejes, no uno: *dominio* (transporte) y *tipo de
umbral* (relativo y medido contra el propio pronóstico). Ese segundo eje no lo
tiene nadie, y en Petetin no por omisión sino por construcción: sus umbrales son
regulatorios y no admiten recalibración.

> Prohibido: ensanchar el reclamo más allá del caso relativo y auto-referencial.

**Cómo se recorta sin perder honestidad:** ceder autoría cuesta **una frase por
precedente**, no un párrafo. "No reclamamos X; está en [cita]." Punto.

---

### III. Datos y método — 1 400 palabras

**A. Datos** (200) — GPS del SIT Arequipa. Tres corredores, cinco meses. Sin
hardware adicional, sin horario publicado.

**B. Del GPS al intervalo** (350) — Cómo se construye el intervalo entre buses
cuando no hay tabla de paradas. Es el aporte secundario y hay que decirlo en
palabras normales, no en nombres de función.

**C. Modelos comparados** (200) — LSTM, XGBoost, persistencia, promedio histórico
horario. **El promedio histórico entra acá** (ver §5).

**D. Qué cuenta como apelotonamiento, y por qué** (350) — La definición del evento
es la bisagra del paper entero. Un intervalo por debajo de la mitad del promedio
de su propio vector. Se declara que la forma "fracción del promedio observado" no
se encontró como definición de evento en la literatura, y que es sustitución
nuestra, no herencia.

**E. Cómo se evalúa** (300) — Corte temporal real, tres orígenes, los tres modelos
puntuados sobre exactamente las mismas filas, varianza agrupada por día de
servicio. **Se dice en cuatro oraciones, no en cuatro párrafos.** El aparato es
Métodos, no es el hallazgo.

---

### IV. Resultados — 2 200 palabras

**A. Contexto escalar y su frontera** (250) — Demotado. Tres cifras y una frase:
a diez minutos el aprendiz le gana a la persistencia por ~21 %, a un minuto
pierde, y la frontera real no es el horizonte sino qué tan movida viene la
ventana. Nada más. *(Hoy: 637 palabras.)*

**B. El pronóstico sale más parejo** (400) — El mecanismo. Sesgo negativo en las
**36 de 36** celdas, empeorando con el horizonte. La escala del manual del oficio
califica al mismo corredor como "servicio de reloj" según el pronóstico y "casi
todo apelotonado" según lo observado.
→ **Figura 2** (dosis-respuesta: compresión vs horizonte).

> Obligatorio y no se puede ablandar: las 36 celdas son un **resultado empírico,
> no un corolario** del teorema. El teorema cubre la varianza temporal de una
> serie escalar; esto es dispersión entre buses en un mismo instante.

**C. La alarma no suena** (450) — El artefacto. En E2 a diez minutos: 14 alarmas
contra 15 245 eventos. La persistencia dispara 15 083. Un factor aparente de 253×.
Y el detector trivial —marcar todo— le gana al ganador declarado en 5 de 12 celdas.
→ **Figura 1a**, **Tabla 1**.

**D. El 253× no mide al modelo** (300) — **El argumento demoledor, y hoy está
enterrado.** Ese mismo factor vale 2 299× en una ventana, 817× en otra, 253× en la
tercera. Un número que se mueve un orden de magnitud según el mes no está midiendo
una capacidad. Merece su propia subsección.

**E. La reparación** (400) — Recalibrando el corte fuera de muestra, el veredicto
se invierte. A diez minutos el aprendiz gana el área bajo la curva en 9 de 9
combinaciones de corredor y ventana. Y cuando dispara, acierta: 71 % de precisión
contra una tasa base de 30 %.
→ **Figura 1b**, **Tabla 2**.

> Obligatorio: declarar el intervalo de confianza de ese 71 %. Son 14 disparos.

**F. Robustez, incluido el ataque a nosotros mismos** (400) — Tres ventanas
temporales. Y la prueba con el umbral absoluto de la convención dominante del
campo: esperábamos que atenuara el hallazgo y lo **empeoró 110×**. Nuestra
elección resultó ser la conservadora.

> Obligatorio: declarar el contra-caveat. Bajo esa convención, en E2 a diez
> minutos el área bajo la curva cae a 0,49 — indistinguible del azar. El alcance
> se enuncia completo o no se enuncia.

---

### V. Discusión y limitaciones — 900 palabras

**A. Qué significa para un operador** (250) — El modelo no se equivoca: está
callado. Precisión alta, cobertura baja. Eso es una herramienta distinta a una
alarma, y sirve para otra cosa.

**B. Qué es nuestro y qué no** (250) — **Toda la delimitación bibliográfica del
paper vive acá y solo acá.** Frente a Petetin, a Sun, a Hoffmann: qué no
reclamamos. Y qué se retiró durante la propia investigación.

> No ablandar el cierre. Nombrar lo que se retiró es diferenciador en este venue,
> no debilidad.

**C. Limitaciones** (400) — Sustantiva, no de trámite. En el relevamiento del
venue solo 3 de 8 artículos tienen algo parecido. **No recortarla al maquetar.**

Debe incluir, sin excepción:
- El aprendiz **pierde contra el promedio histórico horario en E2 a diez minutos**.
- El presupuesto de búsqueda está torcido: 24 configuraciones para el árbol contra
  1 para la red. Donde la red pierde, no es atribuible a la clase de modelo.
- El umbral del evento no está calibrado contra incidentes registrados.
- Tres corredores de una ciudad, cinco meses, y la ventana de prueba contiene
  Carnaval.
- El aprendiz no está listo para operar una alarma. Un área bajo la curva de 0,60
  es información real y está muy lejos de un sistema de despacho.

---

### VI. Conclusión — 350 palabras

Cierra sobre **el hueco**, no sobre los resultados. Las cifras están en la Sección
IV y repetirlas las devalúa.

Termina con la prescripción: reportar al menos una métrica sin umbral junto a la
métrica de alarma, publicar el piso del detector trivial, y declarar en qué
espacio se calibró el corte.

---

## 4. Evidencia: qué entra al cuerpo y qué al apéndice

| Cuerpo | Apéndice / suplementario |
|---|---|
| **Fig. 1a/1b** — el par artefacto ↔ reparación. *Funcionan solo como par; compararlas es el aporte.* | Desglose por dirección (+1 / −1) |
| **Fig. 2** — compresión vs horizonte, 36 celdas | Barrido de semillas |
| **Tabla 1** — detección con corte trasplantado + piso trivial, 12 celdas | El enrutador (7 de 12 políticas degeneradas) |
| **Tabla 2** — veredicto recalibrado, 12 celdas | Tablas escalares completas de MAE/RMSE |
| **Tabla 3** — robustez: 3 ventanas + el ataque con umbral absoluto | Configuraciones ganadoras de la búsqueda |

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

- [x] Afirmación central fijada
- [x] Presupuesto por sección
- [x] Reparto cuerpo / apéndice
- [ ] Pendientes §5 (el 4 está resuelto)
- [x] **Sección IV escrita y verificada** — 23/23 cifras contrastadas contra los CSV
      el 2026-08-26 con `scratchpad/verify_seccion_iv.py`
- [x] **Figuras 1–4 y Tablas 1–3 embebidas en `paper.md`.** Las figuras salen de
      `src/build_contiguous_figures.py` a `docs/paper/figuras/*.{es,en}.png`
      (limpias, sin título ni pie horneado); las tablas salen de
      `src/build_paper_tables.py` a `docs/paper/tablas/*.md`. **Ningún número del
      manuscrito se escribe a mano:** si cambia un CSV se re-corre el builder y se
      vuelve a pegar. Los nombres de archivo no llevan número de figura —el orden
      de aparición todavía se mueve y solo `paper.md` lo conoce.
- [x] **Sección III escrita** — 1.352 palabras contra 1.400. Cifras verificadas
      contra `sample_index_manifest.csv` (107/23/22 días, orígenes de 61/83/107,
      81–91 % de fotos utilizables) y `contiguous_winsorization_sensitivity.csv`
      (0,78–1,11 % de objetivos topados). El promedio histórico entra como cuarto
      competidor, cerrando el pendiente 1 del §5.
- [x] **Sección V escrita** — 920 palabras contra 900. Las limitaciones se
      redactan como **alcance, no como disculpa**: cada una delimita dónde vale la
      afirmación en vez de pedir perdón por dónde no. Entran las tres incómodas
      —promedio histórico en E2 h10, tensión AUC/MCC en E59 h5, AUC de azar bajo
      corte absoluto— más la asimetría de búsqueda y los orígenes anidados.
- [x] Pendiente 2 del §5 (**doble estándar media/mediana**) **resuelto por
      construcción**: la Sección IV se escribió desde cero y esa afirmación nunca
      entró. No hay nada que borrar.
- [ ] Redacción — orden vigente: **IV → III → V → II → VI → I → Resumen**.
      Se escribe IV primero porque sus cifras están cerradas y el resto del paper
      calibra contra ellas; II va tarde porque el recorte de 3 923 a 850 palabras
      es más fácil cuando ya se sabe exactamente qué hay que enmarcar.
- [ ] Traducción al inglés. **Las figuras ya están traducidas**: el builder emite
      `.es.png` y `.en.png` de la misma fuente. `paper.md` apunta hoy a las `.es`
      porque el borrador es en español. Al traducir, un solo comando cambia las
      cuatro rutas y no hay que regenerar nada:

      ```bash
      sed -i 's/\.es\.png/.en.png/g' docs/paper/paper.md
      ```

      Las tablas se re-emiten en inglés cambiando `DECIMAL_SEP` a `"."` y las
      cabeceras en `src/build_paper_tables.py`.

      ✅ **Sin excepciones.** `build_schematic_figures.py` recibió el mismo
      tratamiento el 2026-08-26, así que los esquemas de la Sección III también
      salen en `.es.png` y `.en.png`. **Las seis figuras del paper responden al
      mismo comando.** Los cinco `esquema-*.png` con título horneado siguen
      existiendo sin cambios para `metodologia.md` y `sintesis.md`, que los
      embeben.
- [ ] Plantilla IJACSA a dos columnas
- [ ] Referencias en formato del venue
