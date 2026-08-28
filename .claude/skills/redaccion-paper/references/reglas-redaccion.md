# Directivas de Redacción Científica (IEEE/IJACSA)

## PRIORIDAD CERO: VERIFICACIÓN Y ANTI-ALUCINACIÓN
*   **Ausencia de Datos:** NUNCA inventar métricas, cifras, citas o resultados para cumplir una regla de estilo. Si el dato necesario para sostener una afirmación no existe en las fuentes de verdad, omitir la afirmación y colocar el marcador `[INSERTAR DATO/MÉTRICA]`.
*   **Fuentes de verdad:** Toda cifra se copia de una de estas y de ninguna otra: las tablas generadas en `docs/paper/tablas/`, las figuras generadas en `docs/paper/figuras/`, `docs/resultados/documento-resultados.md` o los CSV de `docs/resultados/`. Ninguna cifra se teclea de memoria ni se recalcula mentalmente. Antes de declarar que falta un dato, verificar que no esté ya en esas rutas.
*   **Referencias:** Prohibido inventar literatura. Las citas no se escriben de memoria; si se necesitan, usar `[CITA_REQUERIDA]`. Autores, título, año y venue se copian literalmente de `docs/paper/fuentes-verificadas.md`, y solo si la entrada está marcada como verificada.

## 1. Voz, Persona y Tono
*   **Contribuciones propias:** Usar siempre la primera persona del plural ("nosotros"). *Ejemplo: Proponemos, analizamos.*
*   **Procedimientos empíricos:** Usar la forma impersonal ("se") o voz pasiva. *Ejemplo: Se recolectaron los datos.*
*   **Prohibición estricta:** NUNCA usar la primera persona del singular ("yo").
*   **Tono:** Estrictamente objetivo, numérico y analítico.

## 2. Sintaxis y Párrafos
*   **Longitud de párrafo:** Todo párrafo de texto continuo debe contener entre **50 y 200 palabras** (idealmente alrededor de 100). 
    *   *Excepciones:* Esta regla NO aplica a listas de viñetas, pies de figura/tabla, celdas, ecuaciones, encabezados ni al Resumen.
*   **Longitud de oración:** Límite máximo de **40 palabras por oración**. Privilegiar el punto seguido sobre las comas. Prohibida la subordinación anidada: como máximo una subordinada por oración. Esta regla acota la complejidad sintáctica, no el orden de los constituyentes, de modo que no contradice el impersonal ni la pasiva que exige la Sección 1.
*   **Referencias cruzadas explícitas:** Usar siempre el nombre y número exacto (ej. "la Figura 2", "la Ecuación 3"). Prohibidas las referencias posicionales ("la tabla de abajo", "el gráfico anterior").

## 3. Tiempos Verbales por Sección
*   **Resumen (Abstract):** Mezcla de Pasado (qué se hizo) y Presente (qué se concluye).
*   **Introducción y Trabajos Relacionados:** Presente simple (contexto) y Presente Perfecto (literatura reciente).
*   **Método Propuesto:** 
    *   *Presente:* Para definiciones formales, formulación matemática y diseño del algoritmo (ej. "El headway se define como...").
    *   *Pasado:* Para acciones únicas ejecutadas durante la construcción (ej. "Se proyectó la posición...").
*   **Diseño Experimental y Resultados:** Pasado para los procedimientos y rendimientos ya observados del modelo; Presente para lo que muestran las tablas/gráficas estáticas (ej. "Como muestra la Tabla 2...").
*   **Amenazas a la Validez:** Presente y Condicional (ej. "Esto podría afectar...").
*   **Conclusiones y Declaraciones:** Presente.

## 4. Estructura Lógica y Títulos
*   **Títulos:** Exclusivamente lenguaje técnico e ingenieril. NUNCA usar metáforas ni títulos coloquiales.
*   **Glosario y Conceptos Nuevos:** Todo término de dominio (*headway*, *bunching*), sustantivo no estándar o metáfora técnica debe definirse **conceptualmente** en su primera aparición, y **además matemáticamente** cuando el término tenga forma cerrada en este documento. Prohibido usar un término o una metáfora que el lector no haya visto definida antes en el propio texto.
*   **Flujo de la Introducción (Modelo SCQA adaptado):**
    1. *Situación:* Contexto operativo y estado actual del dominio.
    2. *Complicación:* Brecha empírica, limitación del estado del arte o problema no resuelto.
    3. *Pregunta/Objetivo:* Declaración exacta de lo que resuelve este documento.
    4. *Respuesta:* Breve resumen del enfoque y una lista explícita (en viñetas) de las contribuciones técnicas del artículo.
*   **Flujo de Trabajos Relacionados (Estado del Arte):**
    *   NUNCA listar *papers* de forma secuencial. Agrupar la literatura por enfoque metodológico (ej. métodos basados en reglas vs. métodos estadísticos).
    *   Toda revisión debe terminar obligatoriamente con un contraste explícito que demuestre por qué la literatura actual es insuficiente y cómo el método propuesto llena esa brecha.
*   **Flujo del Método (Top-Down):** Iniciar con el pipeline completo y la formulación matemática explícita (variables, entradas, salidas) en Presente, antes de narrar detalles específicos.
*   **Causalidad de Resultados:** Toda métrica, número o gráfica reportada debe acompañarse de un análisis técnico que explique *por qué* se obtuvo ese resultado (basado en la sección del Método), apoyándose estrictamente en los datos provistos.

## 5. Restricciones Léxicas (Filtro Anti-IA)
*   **Métricas exactas:** Prohibidos los cuantificadores vagos ("buen rendimiento", "mejora significativa") a menos que haya prueba estadística. Referir siempre a los valores absolutos o porcentajes del texto origen.
*   **Lista negra de clichés:** Prohibido usar: *crucial, holístico, revolucionario, panorama, es imperativo, cabe destacar, en este sentido, por su parte, juega un papel fundamental, en la actualidad, no solo... sino también, adentrarse*.
*   **Redundancias:** Prohibido iniciar párrafos con frases vacías de transición o resumir al final de un bloque lo que se acaba de explicar.

## 6. Formulación Matemática y Ecuaciones
*   **Integración gramatical:** Toda ecuación matemática es parte de la oración y debe puntuarse como tal. Si la ecuación termina la idea, lleva punto final; si la idea continúa (por ejemplo, para definir variables), lleva coma.
*   **Definición inmediata de variables:** Inmediatamente después de CADA ecuación, se deben definir obligatoriamente todos los términos, símbolos y subíndices que no se hayan definido antes. Iniciar siempre con la palabra "donde...". Prohibido asumir que una variable es "obvia".
*   **Numeración cruzada:** Las ecuaciones deben numerarse secuencialmente entre paréntesis al margen derecho, en una sola secuencia por orden de aparición. En el texto, referirse a ellas estrictamente como "la Ecuación (1)". Prohibido usar referencias posicionales como "la fórmula de abajo" o "la siguiente ecuación". Insertar una ecuación en el medio obliga a renumerar las siguientes y a actualizar sus referencias.
*   **Notación tipográfica:** Las variables escalares van en cursiva y los vectores o matrices en negrita.
*   **Mecanismo en el borrador markdown:** El número se escribe con `\tag{n}` dentro del bloque de ecuación. La cursiva escalar es el modo matemático por defecto: se escribe `$h$`, sin marcado adicional. La negrita vectorial se escribe `$\mathbf{h}$`. Prohibido usar el marcado de markdown (`*h*`, `**h**`) dentro de `$...$`: no renderiza y no traduce a LaTeX.
*   **Transición texto-fórmula:** Nunca iniciar un párrafo con una ecuación matemática. Toda ecuación debe ser introducida y justificada previamente en lenguaje natural.

## 7. Resolución de Conflictos

Cuando dos reglas de este documento no puedan cumplirse a la vez, aplicar este orden. Gana siempre la de número menor.

1.  **Prioridad Cero.** No inventar. Ante la duda, el marcador y no la frase.
2.  **Exactitud técnica.** Que la afirmación sea correcta y trazable a su fuente. Incluye la exigencia de métricas exactas y la prohibición de redundancias (Sección 5).
3.  **Reglas de estructura y flujo** (Secciones 4 y 6).
4.  **Reglas de tiempo verbal y voz** (Secciones 1 y 3).
5.  **Reglas de forma** (Sección 2: longitud de párrafo y de oración) y lista negra de clichés (Sección 5).

Consecuencia obligatoria: si un párrafo no llega a 50 palabras sin agregar relleno, o si pasa de 200 sin poder partirse, se deja fuera de rango y se anota. Prohibido escribir oraciones vacías para alcanzar el mínimo: eso incumple la prohibición de redundancias, que es prioridad 2 y vence a la longitud de párrafo, que es prioridad 5.

## 8. Verificación Antes de Entregar

Ninguna redacción se da por terminada sin recorrer esta lista. El chequeo no se escribe dentro del manuscrito: el informe va en la respuesta, fuera del texto del paper. Los puntos 1 y 2 no son razonamiento: exigen abrir la fuente y comparar. Prohibido declarar cumplido un punto sin haberlo comprobado contra el archivo. Si algún punto falla, corregir antes de entregar y reportar qué se corrigió.

1.  **Cifras.** Cada número del texto aparece idéntico en su fuente de verdad. Los que no, quedan como `[INSERTAR DATO/MÉTRICA]`.
2.  **Citas.** Cada cita existe en `fuentes-verificadas.md` y está marcada como verificada. Las demás quedan como `[CITA_REQUERIDA]`.
3.  **Léxico.** Ninguna palabra de la lista negra de la Sección 5 aparece en el texto.
4.  **Términos.** Ningún término no estándar ni metáfora se usa antes de su definición.
5.  **Referencias cruzadas.** Ninguna es posicional. Todo número de ecuación, figura o tabla citado existe.
6.  **Forma.** Ninguna oración pasa de 40 palabras. Todo párrafo continuo cae entre 50 y 200, o está anotado como excepción justificada.
7.  **Ecuaciones.** Cada una lleva `\tag{n}` en secuencia, va introducida en prosa y define sus símbolos nuevos inmediatamente después con "donde…".
