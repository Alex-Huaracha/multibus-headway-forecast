# Directivas de Redacción Científica (IEEE/IJACSA)

Esta norma gobierna **cómo se escribe** el manuscrito: integridad de las cifras,
registro, voz, léxico, sintaxis y notación.

**No gobierna cómo se estructura el paper.** Qué secciones existen, en qué orden
van, qué contiene cada una y dónde vive cada resultado son decisiones del autor.
Esta norma no las valida, no las exige y no las bloquea.

## PRIORIDAD CERO: VERIFICACIÓN Y ANTI-ALUCINACIÓN
*   **Ausencia de Datos:** NUNCA inventar métricas, cifras, citas o resultados para cumplir una regla de estilo. Si el dato necesario para sostener una afirmación no existe en las fuentes de verdad, omitir la afirmación y colocar el marcador `[INSERTAR DATO/MÉTRICA]`.
*   **Fuentes de verdad:** Toda cifra se copia de una de estas y de ninguna otra: las tablas generadas en `docs/paper/tablas/`, las figuras generadas en `docs/paper/figuras/`, `docs/resultados/documento-resultados.md` o los CSV de `docs/resultados/`. Ninguna cifra se teclea de memoria ni se recalcula mentalmente. Antes de declarar que falta un dato, verificar que no esté ya en esas rutas.
*   **Conflicto entre fuentes:** Si dos fuentes discrepan, gana el CSV primitivo de `docs/resultados/` sobre `documento-resultados.md` y sobre cualquier tabla ya pegada en el manuscrito.
*   **Referencias:** Prohibido inventar literatura. Las citas no se escriben de memoria; si se necesitan, usar `[CITA_REQUERIDA]`. Autores, título, año y venue se copian literalmente de `docs/paper/fuentes-verificadas.md`, y solo si la entrada está marcada como verificada.
*   **Atribución:** Prohibido atribuir a una fuente un hallazgo que no reporta. Antes de escribir "X reporta que…", abrir la entrada y comprobar que lo reporta.
*   **Definiciones:** Todo instrumento, métrica o umbral que el documento define debe coincidir con su implementación en `src/`. Ante una diferencia gana el código y se corrige el documento. Prohibido escribir código para que el documento tenga razón. Si el instrumento no existe en `src/`, se saca del documento o se le da uso con lo que el proyecto ya calcula.

## 1. Voz, Persona y Tono
*   **Contribuciones propias:** Usar siempre la primera persona del plural ("nosotros"). *Ejemplo: Proponemos, analizamos.*
*   **Procedimientos empíricos:** Usar la forma impersonal ("se") o voz pasiva. *Ejemplo: Se recolectaron los datos.*
*   **Prohibición estricta:** NUNCA usar la primera persona del singular ("yo").
*   **Tono:** Estrictamente objetivo, numérico y analítico.

## 2. Registro: Español Neutro Profesional
El manuscrito se redacta en español y se traduce al inglés al final. El registro
es neutro y técnico, sin marca geográfica, para que la traducción no arrastre
giros locales.

*   **Prohibido el voseo** y toda su conjugación: *vos, tenés, podés, sabés, mirá, fijate*. Se usa la tercera persona impersonal o el plural.
*   **Prohibido el `vosotros`** y su conjugación peninsular.
*   **Prohibido el léxico regional** cuando existe un término panhispánico. No usar: *plata* (dinero), *laburo* (trabajo), *chévere*, *bacán*, *guay*, *tío*, *vale*, *ahorita*, *nomás*, *pues* como muletilla, *o sea* como relleno, *capaz que*, *de una*.
*   **Prohibidos los marcadores coloquiales:** *bueno*, *dale*, *listo*, *igual* con valor concesivo, *encima*, *ojo que*.
*   **Un objeto, un término, en toda variedad:** cuando dos regiones nombran distinto la misma cosa (*auto/coche/carro*, *computadora/ordenador*, *manejar/conducir*), se elige uno y se conserva en todo el documento.
*   **Sin diminutivos ni aumentativos afectivos.** Sin interjecciones. Sin signos de admiración.
*   **Sin ironía, humor ni guiños al lector.**

## 3. Sintaxis y Párrafos
*   **Longitud de párrafo:** Todo párrafo de texto continuo debe contener entre **50 y 200 palabras** (idealmente alrededor de 100).
    *   *Excepciones:* Esta regla NO aplica a listas de viñetas, pies de figura/tabla, celdas, ecuaciones, encabezados ni al Resumen.
*   **Longitud de oración:** Límite máximo de **40 palabras por oración**. Privilegiar el punto seguido sobre las comas. Prohibida la subordinación anidada: como máximo una subordinada por oración. Esta regla acota la complejidad sintáctica, no el orden de los constituyentes, de modo que no contradice el impersonal ni la pasiva que exige la Sección 1.
*   **Referencias cruzadas explícitas:** Usar siempre el nombre y número exacto (ej. "la Figura 2", "la Ecuación 3", "la Sección IV-B"). Prohibidas las referencias posicionales ("la tabla de abajo", "el gráfico anterior", "como se vio antes").
*   **Referencias cruzadas vigentes:** Toda referencia a una sección, figura, tabla o ecuación debe apuntar a un objeto que existe con ese número. Al mover o renumerar un bloque, actualizar todas las referencias que lo nombran.

### 3.1 Números y unidades
*   **Separador decimal: el punto** (`0.632`). Es la convención del documento final en inglés, y rige también en el borrador para no arrastrar una conversión de cada cifra. Vive en `DECIMAL_SEP` de `src/build_paper_tables.py`; la prosa y las tablas usan el mismo.
*   **Millares: espacio que no corta** (`15&nbsp;245`), según ISO 80000-1. No se usa coma de millares, que colisionaría con el decimal en un lector acostumbrado a la otra convención.
*   **Cifras significativas constantes** para una misma cantidad en todo el documento. Si el MAE se reporta con dos decimales, se reporta con dos decimales siempre.
*   **Unidad siempre presente** en la primera mención de una cantidad y en toda cifra de una tabla. Prohibido dejar un número desnudo cuando tiene unidad.
*   **Rangos e intervalos** con la misma cantidad de decimales en los dos extremos: `[-0.023, -0.003]`, nunca `[-0.023, -0.0030]`.
*   **Porcentaje y punto porcentual no son lo mismo.** Una diferencia entre dos porcentajes se expresa en puntos porcentuales.

## 4. Conectores y Encadenamiento
*   **El conector nombra la relación real.** Se elige por la relación lógica que une los dos bloques, no por variedad: causa (*porque*, *de modo que*), contraste (*en cambio*, *sin embargo*), consecuencia (*por eso*, *entonces*), condición (*si*), adición (*además*). Prohibido un conector adversativo donde no hay contraste.
*   **Sin conectores de relleno.** No toda oración necesita uno. Prohibido abrir un párrafo con una transición que no aporta relación: *en este sentido*, *por su parte*, *cabe destacar*, *dicho esto*, *a continuación*, *como se mencionó*.
*   **Sin repetir el mismo conector** dentro de un párrafo.
*   **Encadenamiento de bloques:** Cada párrafo abre nombrando el objeto que recibe del bloque anterior, o la afirmación a la que sirve. Prohibido abrir declarando un instrumento sin antecedente. *Incorrecto:* "El error se mide sobre las posiciones válidas…". *Correcto:* "El modelo entrega un vector de headways. La regla de la Sección III-C lo convierte en un indicador binario. La evaluación mide entonces dos objetos en cadena…".

## 5. Términos y Nombres
*   **Títulos:** Exclusivamente lenguaje técnico e ingenieril. NUNCA usar metáforas ni títulos coloquiales.
*   **Glosario y Conceptos Nuevos:** Todo término de dominio (*headway*, *bunching*), sustantivo no estándar o metáfora técnica debe definirse **conceptualmente** en su primera aparición, y **además matemáticamente** cuando el término tenga forma cerrada en este documento. Prohibido usar un término o una metáfora que el lector no haya visto definida antes en el propio texto.
*   **Un objeto, un nombre:** Cada objeto técnico recibe un nombre y conserva ese nombre en todo el documento. Prohibido rotar sinónimos para el mismo objeto: repetir la misma palabra es el estándar del campo y no un defecto de estilo. Prohibido también usar una misma palabra para dos objetos distintos; si la palabra ya nombra otra cosa, se elige otra. El nombre se establece en el cuerpo del texto: un título de paso, de párrafo o de sección NO lo establece, porque el lector no puede resolver un pronombre contra un encabezado. Todo artículo definido y todo demostrativo que introduzca un objeto técnico apunta a un sustantivo dicho antes en el cuerpo y con esa misma palabra. Repetir la definición de una sigla, o su forma expandida junto a la sigla, es admisible y no incumple esta regla.
*   **Verificar antes de acuñar:** Antes de adoptar un término nuevo, buscarlo en `docs/paper/paper.md`. Si ya nombra otra cosa, elegir otro.

## 6. Tiempo Verbal por Tipo de Contenido
El tiempo lo fija **qué se está diciendo**, no en qué sección aparece.

*   **Definición formal, formulación matemática, propiedad permanente:** presente. *Ejemplo: "El headway se define como…".*
*   **Protocolo que rige de forma estable:** presente. *Ejemplo: "La partición es por fecha…".*
*   **Acción única ya ejecutada:** pasado. *Ejemplo: "Se proyectó la posición…", "Los experimentos se ejecutaron sobre GPU…".*
*   **Rendimiento ya observado:** pasado. *Ejemplo: "El error absoluto medio bajó 1,47 minutos…".*
*   **Lo que una tabla o figura estática muestra:** presente. *Ejemplo: "La Tabla 2 recoge las doce celdas…".*
*   **Literatura previa:** presente simple o presente perfecto.
*   **Limitación o riesgo:** presente y condicional. *Ejemplo: "Esto podría afectar…".*
*   **Conclusión:** presente.
*   **Resumen:** pasado para qué se hizo, presente para qué se concluye.

## 7. Restricciones Léxicas (Filtro Anti-IA)

### 7.1 Léxico
*   **Métricas exactas:** Prohibidos los cuantificadores vagos ("buen rendimiento", "mejora significativa") a menos que haya prueba estadística. Referir siempre a los valores absolutos o porcentajes del texto origen.
*   **Lista negra en español:** Prohibido usar: *crucial, holístico, revolucionario, panorama, es imperativo, cabe destacar, en este sentido, por su parte, juega un papel fundamental, en la actualidad, no solo… sino también, adentrarse, clave* (como adjetivo; *clave de cita* es sustantivo y se admite), *robusto/robustez* (salvo el sentido estadístico exacto: *varianza robusta a conglomerados*, *ensayo de robustez*), *integral, sólido, potente, valioso, rico en, abordar, desentrañar, arrojar luz, allanar el camino, punto de inflexión, piedra angular, en el corazón de, a medida que avanzamos, es importante señalar, vale la pena destacar*.
*   **Lista negra en inglés:** rige al traducir, porque el filtro en español no sobrevive la traducción. Prohibido: *delve, landscape, realm, testament, underscore, pivotal, seamless, leverage* (como verbo), *showcase, robust* (salvo el sentido estadístico exacto), *comprehensive, intricate, nuanced, meticulous, crucial, vital, harness, unlock, navigate* (en sentido figurado), *foster, myriad, plethora, tapestry, cornerstone, paradigm shift, it is worth noting, it is important to note, plays a crucial role, stands as a testament, sheds light on, paves the way, at the heart of, in today's world*.
*   **Adverbios de intensidad:** prohibidos *notablemente, significativamente* (salvo con prueba estadística), *sustancialmente, marcadamente, considerablemente, dramáticamente*.
*   **Redundancias:** Prohibido iniciar párrafos con frases vacías de transición o resumir al final de un bloque lo que se acaba de explicar.

### 7.2 Construcciones
Los rasgos que marcan la prosa generada son sintácticos antes que léxicos. Cambiar la palabra no los elimina.

*   **Antítesis decorativa.** La construcción «no es X, sino Y» se admite solo cuando X es una lectura que el lector haría de verdad y que el documento necesita descartar. Prohibida cuando X se construye únicamente para lucir el contraste. Prueba: si nadie sostendría X, la mitad negativa sobra y la oración se escribe en afirmativo.
*   **Tríadas.** Prohibida la enumeración de tres elementos cuando el tercero no agrega información que los dos primeros no den. La simetría no es una razón para escribir un tercer término.
*   **Paralelismo inflado.** Prohibido repetir una estructura sintáctica en oraciones consecutivas para producir cadencia. La repetición de estructura se reserva para cuando los elementos comparados son realmente paralelos.
*   **Raya de énfasis.** La raya (`—`) se reserva para el inciso que una coma no separaría sin ambigüedad. Prohibida para dar énfasis o para insertar un comentario del autor.
*   **Apertura con gerundio o participio:** prohibido abrir una oración con *Habiendo…*, *Siendo…*, *Teniendo en cuenta…*, *Dado que…* cuando una subordinada con verbo conjugado dice lo mismo.
*   **Cierre de resumen.** Prohibido cerrar un párrafo o una subsección repitiendo en otras palabras lo que el bloque acaba de establecer.
*   **Hedging apilado:** una sola marca de incertidumbre por afirmación. Prohibido *podría potencialmente*, *sugiere que posiblemente*, *parece indicar que quizá*.

### 7.3 Calibración de la afirmación
El verbo debe coincidir con lo que la evidencia sostiene. Esta es la regla que un revisor usa para decidir si el documento se excede.

*   **Verbo causal solo con diseño causal.** *Causa*, *produce*, *provoca*, *determina* exigen un diseño que identifique la causa. Con evidencia asociativa se escribe *se asocia con*, *acompaña a*, *coincide con*, *sigue a*.
*   **Reservar *demuestra* y *prueba*** para lo que el documento prueba. Un resultado empírico *muestra*, *reporta*, *mide* o *indica*.
*   **Toda diferencia reportada como ganada lleva su acotación** —intervalo, prueba o `n` efectivo— o se escribe como diferencia observada y no como ventaja.
*   **Prohibido generalizar más allá del corpus.** Lo medido sobre tres corredores de una ciudad no se afirma de «los corredores urbanos». El alcance se nombra en la misma oración.
*   **Un resultado nulo no es evidencia de ausencia.** Se escribe *no se detectó diferencia*, nunca *no hay diferencia*.
*   **Prohibido atribuir intención o capacidad a un modelo:** *el modelo entiende*, *aprende a anticipar*, *sabe*. Se escribe qué predice y con qué error.
*   **Patrón de dos movimientos:** Cada decisión se escribe en dos movimientos y se detiene: se declara la decisión, y se justifica en una sola oración que diga **qué compra** esa decisión. Prohibido el tercer movimiento —el remate—, sea aforismo, sentencia, metáfora o ilustración vívida.
    *   La prueba no es si el remate es verdadero, sino si borrarlo cambia lo que el lector puede hacer con el texto. Si no lo cambia, sobra.
    *   La justificación es operativa y no interpretativa: dice qué evita o qué garantiza la decisión, no qué significa.
    *   *Ejemplos de remate, prohibidos:* "Ese es el precio de que el horizonte signifique lo que dice", "No se declara: se verifica", "la comparación no queda sesgada sino indefinida".
    *   El patrón acota el adorno, no el volumen: una subsección con cuatro decisiones son ocho oraciones legítimas.
*   **Prueba de tachado:** Antes de dar por buena una afirmación, tacharla y preguntar si el documento pierde algo que no diga en ninguna otra parte. Repetir un hecho, comprimido a una cláusula, para acotar su alcance es legítimo; repetir la conclusión no lo es.

## 8. Formulación Matemática y Ecuaciones
*   **Integración gramatical:** Toda ecuación matemática es parte de la oración y debe puntuarse como tal. Si la ecuación termina la idea, lleva punto final; si la idea continúa (por ejemplo, para definir variables), lleva coma.
*   **Definición inmediata de variables:** Inmediatamente después de CADA ecuación, se deben definir obligatoriamente todos los términos, símbolos y subíndices que no se hayan definido antes. Iniciar siempre con la palabra "donde...". Prohibido asumir que una variable es "obvia".
*   **Numeración cruzada:** Las ecuaciones deben numerarse secuencialmente entre paréntesis al margen derecho, en una sola secuencia por orden de aparición. En el texto, referirse a ellas estrictamente como "la Ecuación (1)". Prohibido usar referencias posicionales como "la fórmula de abajo" o "la siguiente ecuación". Insertar una ecuación en el medio obliga a renumerar las siguientes y a actualizar sus referencias.
*   **Notación tipográfica:** Las variables escalares van en cursiva y los vectores o matrices en negrita.
*   **Mecanismo en el borrador markdown:** El número se escribe con `\tag{n}` dentro del bloque de ecuación. La cursiva escalar es el modo matemático por defecto: se escribe `$h$`, sin marcado adicional. La negrita vectorial se escribe `$\mathbf{h}$`. Prohibido usar el marcado de markdown (`*h*`, `**h**`) dentro de `$...$`: no renderiza y no traduce a LaTeX.
*   **Transición texto-fórmula:** Nunca iniciar un párrafo con una ecuación matemática. Toda ecuación debe ser introducida y justificada previamente en lenguaje natural.

## 9. Resolución de Conflictos

Cuando dos reglas de este documento no puedan cumplirse a la vez, aplicar este orden. Gana siempre la de número menor.

1.  **Prioridad Cero.** No inventar. Ante la duda, el marcador y no la frase.
2.  **Exactitud técnica.** Que la afirmación sea correcta y trazable a su fuente.
3.  **Registro y léxico** (Secciones 2 y 7): español neutro, sin clichés y sin remates.
4.  **Términos, conectores y voz** (Secciones 1, 4 y 5).
5.  **Forma** (Sección 3: longitud de párrafo y de oración) y tiempo verbal (Sección 6).

Consecuencia obligatoria: si un párrafo no llega a 50 palabras sin agregar relleno, o si pasa de 200 sin poder partirse, se deja fuera de rango y se anota. Prohibido escribir oraciones vacías para alcanzar el mínimo, y prohibido conservar un remate por la misma razón: eso incumple la prohibición de redundancias y de remates, que es prioridad 3 y vence a la longitud de párrafo, que es prioridad 5.

## 10. Verificación Antes de Entregar

Ninguna redacción se da por terminada sin recorrer esta lista. El chequeo no se escribe dentro del manuscrito: el informe va en la respuesta, fuera del texto del paper. Los puntos 1, 2, 5, 9 y 10 no son razonamiento: exigen abrir la fuente y comparar, o contar sobre el propio documento. Prohibido declarar cumplido un punto sin haberlo comprobado contra el archivo. Si algún punto falla, corregir antes de entregar y reportar qué se corrigió.

1.  **Cifras.** Cada número del texto aparece idéntico en su fuente de verdad. Los que no, quedan como `[INSERTAR DATO/MÉTRICA]`.
2.  **Citas.** Cada cita existe en `fuentes-verificadas.md`, está marcada como verificada, y el hallazgo que se le atribuye es el que reporta.
3.  **Registro.** Ningún voseo, ningún `vosotros`, ningún regionalismo ni marcador coloquial de la Sección 2.
4.  **Léxico.** Ninguna palabra de la lista negra de la Sección 7.1 aparece en el texto. Al traducir, recorrer además la lista en inglés.
5.  **Construcciones.** Ninguna antítesis decorativa, tríada de relleno, paralelismo inflado, raya de énfasis, apertura con gerundio, cierre de resumen ni hedging apilado (Sección 7.2). Contar las rayas: cada una debe separar un inciso que la coma dejaría ambiguo.
6.  **Concisión.** Ningún párrafo cierra con un remate. Cada decisión llega a dos movimientos y se detiene.
7.  **Conectores.** Ninguno es de relleno, ninguno se repite dentro de un párrafo, y cada uno nombra la relación que realmente une los dos bloques.
8.  **Términos.** Ningún término no estándar ni metáfora se usa antes de su definición.
9.  **Nombres y referentes.** Para cada objeto técnico, se cuenta cuántas palabras distintas lo nombran; con más de una, se elige la que se queda y se reemplaza el resto. Ninguna palabra nombra dos objetos. Todo artículo definido y todo demostrativo que introduce un objeto apunta a un sustantivo dicho antes en el cuerpo, con esa misma palabra.
10. **Referencias cruzadas.** Ninguna es posicional. Todo número de sección, ecuación, figura o tabla citado existe y apunta al objeto correcto.
11. **Forma.** Ninguna oración pasa de 40 palabras. Todo párrafo continuo cae entre 50 y 200, o está anotado como excepción justificada.
12. **Ecuaciones.** Cada una lleva `\tag{n}` en secuencia, va introducida en prosa y define sus símbolos nuevos inmediatamente después con "donde…".
13. **Código.** Cada instrumento que el documento define coincide con su implementación en `src/`, comprobada abriéndola.
14. **Tachado.** Ninguna afirmación repite una conclusión que el documento ya emite en otro lugar.
15. **Números.** Separadores, cifras significativas, unidades y extremos de intervalo cumplen la Sección 3.1, en la variedad que corresponda al idioma del documento.
16. **Calibración.** Ningún verbo causal sin diseño causal, ningún *demuestra* sin prueba, ninguna ventaja sin su acotación, ninguna generalización fuera del corpus (Sección 7.3).
17. **Traducción.** Al entregar en inglés, recorrer la Sección 11 completa.

## 11. Traducción al Inglés

El manuscrito se redacta en español y se traduce al final. La traducción es una redacción y no un reemplazo de palabras: las reglas anteriores siguen rigiendo sobre el texto en inglés.

*   **El glosario manda.** `docs/paper/glosario-terminos.md` fija el término inglés de cada objeto técnico. Ese término se usa y no se rota, igual que en español (Sección 5).
*   **Separadores numéricos:** ya son los de la Sección 3.1 (punto decimal, coma de millares), de modo que la traducción no los toca.
*   **Lista negra en inglés:** recorrer la de la Sección 7.1. El filtro en español no detecta *delve*, *underscore* ni *testament*.
*   **Falsos amigos frecuentes:** *actualmente* → `currently`, no `actually`. *Eventualmente* → `occasionally` o `possibly`, no `eventually`. *Realizar* → `carry out` o `perform`, no `realize`. *Asistir* → `attend` o `assist` según el sentido. *Sensible* → `sensitive`. *Consistente* → `consistent` solo en el sentido de coherencia; si es «sólido», `robust` en su sentido estadístico. *Notorio* → `well known`, no `notorious`. *Suceso* → `event`, no `success`.
*   **Artículos:** el español omite artículos donde el inglés los exige (*headway prediction* frente a *the prediction of the headway*). Revisar cada sintagma nominal técnico.
*   **Densidad de nominalizaciones:** el español académico nominaliza más que el inglés. Donde el español dice *la realización de la medición*, el inglés dice *measuring*.
*   **Voz:** el impersonal con *se* de la Sección 1 se traduce como pasiva (*the data were collected*) o como primera persona del plural en las contribuciones (*we propose*). Prohibido traducirlo como *one*.
*   **Tiempo verbal:** se conserva el criterio por tipo de contenido de la Sección 6. El inglés académico usa presente para lo que una tabla muestra y pasado para lo ejecutado, igual que el español.
*   **Lo que no se traduce:** claves de cita, nombres de archivo, identificadores de corredor (E2, E4, E59), etiquetas de columna ya fijadas en los builders.
