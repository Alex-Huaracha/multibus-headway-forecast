**PROPUESTA DE INVESTIGACIÓN**

*Actualización basada en análisis de viabilidad del dataset*

**Predicción del Vector de Headways en Corredores de**

**Transporte Público Urbano Mediante GNN+LSTM**

Proyecto SMART MOBILITY AREQUIPA (SMARQ)

Publicación objetivo: IJACSA

**1\. Qué cambió y por qué**

La propuesta inicial planteaba trabajar con 10 corredores BRT y detectar anomalías operacionales mediante un modelo de IA. Antes de proceder con la implementación, se realizó un análisis de viabilidad exhaustivo sobre el dataset real para verificar que las asunciones del paper encajaran con la realidad de los datos.

El análisis reveló hallazgos que obligaron a reformular el alcance y el enfoque:

**Hallazgo 1: 6 empresas cumplen los criterios de viabilidad; el trabajo se acota a 4**

Se analizó la geometría de cada ruta mediante PCA (sobre puntos en movimiento, no estacionados) y se contó cuántos buses circulan simultáneamente por corredor (sobre datos deduplicados con clave compuesta). 6 empresas pasan los umbrales de linealidad y flota simultánea (1, 2, 4, 55, 58, 59). El presente trabajo se acota a 4 corredores como caso de estudio: empresas 2, 4, 58 y 59, cubriendo un rango de flotas simultáneas de 6 a 20 buses. Las empresas restantes con corredores viables (1 y 55) quedan fuera del alcance y se reservan para validación posterior.

**Hallazgo 2: Alcance acotado a 4 empresas**

De las 6 empresas viables, el presente trabajo se acota a 4 (2, 4, 58, 59) por restricciones de tiempo del proyecto. Este subconjunto cubre un rango amplio de flotas simultáneas (6, 9, 16 y 20 buses en mediana) sobre cuatro empresas operativamente independientes, suficiente para evaluar la generalización del método sobre contextos heterogéneos. Las empresas 1 y 55, también viables, quedan reservadas para validación posterior bajo la misma metodología.

**Hallazgo 3: El sistema no es BRT estrictamente**

El SIT Arequipa es transporte público urbano integrado, sin vías segregadas exclusivas como TransMilenio o el Metropolitano de Lima. Usar el término “BRT” en el paper sería cuestionado por revisores. Se cambió a “corredores de transporte público urbano integrado”.

**Hallazgo 4: Los identificadores de unidad se reutilizan entre empresas**

34 de 150 identificadores de unidad aparecen en 3 o más empresas. Esto obliga a usar la clave compuesta (empresaid, unidadid) en todo el procesamiento para evitar mezclar datos de buses de empresas distintas.

**Hallazgo 5: La sugerencia del docente fortalece el paper**

El docente asesor recomendó que el modelo no solo detecte anomalías sino que prediga el vector completo de headways. Esto transforma el paper de reactivo (detectar lo que ya pasó) a proactivo (anticipar lo que va a pasar), y resuelve el problema de validación: el ground truth son los headways reales futuros que ya existen en los datos GPS, eliminando la dependencia de datos sintéticos.

**Hallazgo 6: Empresas 58 y 59 no reportan los campos `direccion` (heading) ni `velocidad`**

El EDA dirigido sobre los 4 corredores (Fase 1, notebook `02_eda_corredores`) reveló que las empresas 58 y 59 no reportan los campos `direccion` ni `velocidad` en sus registros GPS (las empresas 2 y 4 sí los reportan). Como la empresa 59 es uno de los dos corredores obligatorios del proyecto, esto fuerza un cambio metodológico: la identificación de sentido ida/vuelta no puede depender del heading. El método primario será la derivada signada de la coordenada lineal `s` (proyección sobre la centerline del corredor); el heading queda como verificación cruzada únicamente en los corredores donde se reporta. La velocidad utilizada en todo el pipeline se computa como `step_m / dt_s` a partir de coordenadas y tiempos consecutivos, no del campo `velocidad` reportado (ver [`decisiones-limpieza-fase2.md`](./decisiones-limpieza-fase2.md)). Esta restricción del dataset refuerza el argumento de generalización: el método funciona con el subconjunto mínimo de columnas GPS (lat, lon, time), sin depender de campos opcionales.

**2\. El problema**

**2.1 Contexto operacional**

En Arequipa, cada empresa de transporte opera un corredor específico con una flota de buses que lo recorren repetidamente durante el día. Los buses salen escalonados para mantener una frecuencia regular de servicio. La separación temporal entre buses consecutivos se llama headway.

**Un servicio de calidad requiere headways regulares.** Si los buses deberían salir cada 8 minutos, los pasajeros esperan en promedio 4 minutos. Pero si los headways se desregulan (uno de 2 minutos seguido de uno de 20 minutos), algunos pasajeros esperan mucho más, los buses se sobrecargan de forma desigual, y el problema se propaga por todo el corredor.

**2.2 El problema específico**

La irregularidad de headways es el problema operacional más costoso del transporte público. En Arequipa, los operadores no tienen forma de anticipar que los headways se están desregulando hasta que el problema ya es visible (buses pegados, pasajeros acumulados). La supervisión es manual, reactiva, y cognitivamente imposible con decenas de buses simultáneos.

Además, la irregularidad de headways genera tres fenómenos operacionales críticos que no son visibles analizando buses individuales:

| Bus Bunching (agrupamiento) Un bus se retrasa en una parada, acumula más pasajeros, tarda más, y el bus de atrás lo alcanza. El headway entre ambos baja a 1-2 minutos mientras el headway con el bus de más atrás crece a 15-20 minutos. Cada bus individual está en su corredor correcto — la anomalía solo existe en el patrón colectivo de headways. |
| :---- |

| Gap de servicio Una unidad no sale de la terminal o tiene falla mecánica. Un tramo del corredor queda sin servicio durante un período prolongado. Los pasajeros se acumulan y sobrecargan el próximo bus que llegue, propagando el problema. Ningún bus que sí está circulando muestra comportamiento anómalo individual. |
| :---- |

| Congestión de corredor Un evento externo (accidente, cierre de vía, manifestación) causa que múltiples buses vayan anormalmente lentos simultáneamente. Un bus individual a 10 km/h no es anómalo por sí solo en una zona céntrica. Pero si 14 de 18 buses van a 10 km/h al mismo tiempo, hay un problema externo. Esto solo se detecta mirando todos los headways y velocidades simultáneamente. |
| :---- |

**2.3 Formulación del problema**

| Problema No existe un método que prediga la evolución del vector completo de headways de un corredor de transporte público utilizando el estado simultáneo de todos los buses activos, permitiendo anticipar fenómenos como bunching, gaps de servicio y congestión antes de que se materialicen, y utilizando únicamente datos GPS básicos sin hardware adicional. |
| :---- |

**3\. La solución**

**3.1 Idea central**

| Solución Un modelo de Inteligencia Artificial que recibe el estado actual de todos los buses de un corredor (posiciones, velocidades, headways) y predice cómo estarán los headways en el siguiente instante de tiempo. Esto permite a los operadores anticipar bunching, gaps y congestión antes de que ocurran e intervenir a tiempo. |
| :---- |

**3.2 Cómo funciona: Request y Response del modelo**

**Request (lo que recibe el modelo)**

En un instante dado, el modelo recibe los headways actuales y recientes de todos los buses activos en un corredor, junto con el contexto temporal (hora del día, día de la semana).

Ejemplo: si hay 20 buses activos en el corredor, existen 19 headways (la separación entre cada par de buses consecutivos). El modelo recibe los últimos N minutos de esos 19 headways.

**Response (lo que devuelve el modelo)**

El modelo devuelve el vector completo de headways predicho para el siguiente instante.

| Ejemplo concreto con 4 buses (3 headways) Headways actuales: \[7 min, 5 min, 6 min\]. El modelo predice que en 5 minutos serán: \[7 min, 2 min, 10 min\]. Interpretación: h₂ baja de 5 a 2 → los buses 2 y 3 se están juntando (bunching). h₃ sube de 6 a 10 → se está abriendo un gap detrás del bus 3\. El operador puede intervenir antes de que el problema se agrave. |
| :---- |

**3.3 ¿Por qué predicción y no solo detección?**

La detección dice: “hay bunching”. La predicción dice: “en 5 minutos va a haber bunching entre el bus 5 y 6, y un gap detrás del bus 7”. La segunda es mucho más útil porque permite intervenir antes de que el problema se propague por todo el corredor.

**3.4 Validación: cómo demostramos que funciona**

**El ground truth es perfecto y no requiere etiquetas manuales.** El modelo predice headways futuros. Esos headways futuros realmente ocurrieron y están registrados en los datos GPS. Predecimos, comparamos con la realidad, y medimos el error en minutos (MAE, RMSE). No hay ambigüedad, no hay datos sintéticos, no hay etiquetado manual.

Esta es una diferencia fundamental con el paper anterior, que dependía exclusivamente de anomalías fabricadas artificialmente para su evaluación.

**4\. Los datos**

**4.1 Dataset general**

| Característica | Valor |
| ----- | ----- |
| **Registros del dataset raw tras dedup (12 empresas)** | 98,968,817 |
| **Registros del corpus de trabajo (4 corredores seleccionados)** | 47,681,656 |
| **Período** | Octubre 2023 – Febrero 2024 (151 días) |
| **Frecuencia GPS** | Cada 20 segundos por unidad |
| **Columnas** | unidadid, empresaid, latitud, longitud, timestamp, dirección (heading) |
| **Hardware adicional** | Ninguno — solo GPS básico ya instalado |

> Nota: 98.97M es el total tras deduplicar el dataset crudo de las 12 empresas con la clave compuesta `(empresaid, unidadid, time)`. El corpus efectivamente analizado en este trabajo (47.68M) corresponde al filtrado a los 4 corredores seleccionados en §4.2, antes de la limpieza row-level documentada en [`decisiones-limpieza-fase2.md`](./decisiones-limpieza-fase2.md).

**4.2 Los 4 corredores seleccionados**

| Empresa | Unidades | Buses simultáneos (mediana) | Ratio PCA | Rol en el paper |
| ----- | ----- | ----- | ----- | ----- |
| **Empresa 2** | 31 | 16 | 33.55 | Caso principal |
| **Empresa 59** | 40 | 20 | 5.92 | Caso principal |
| **Empresa 4** | 19 | 9 | 5.67 | Validación escalabilidad |
| **Empresa 58** | 12 | 6 | 4.20 | Validación escalabilidad |

> Las cifras de "Unidades" corresponden a la **flota operacionalmente activa** tras el filtro de buses estacionarios aplicado en Fase 0 (notebook `01_viability_and_filter`). El dataset raw contenía 120 unidades distintas en estas 4 empresas; 18 unidades estuvieron siempre estacionadas durante el período y fueron excluidas del corpus de trabajo. Total operacional: 102 unidades. Los conteos verificables están en el notebook `02_eda_corredores` (output `quality_gps.csv`).

**4.3 Empresas fuera del alcance**

| Empresa | Motivo |
| ----- | ----- |
| Empresa 1 | Viable (PCA=4.87, mediana=30). Fuera del alcance por restricciones de tiempo. Se reserva para validación posterior. |
| Empresa 12 | No viable: PCA=1.69 (zigzag), mediana=3 buses simultáneos. |
| Empresa 19 | No viable: PCA=1.90 (no lineal). |
| Empresa 22 | No viable: mediana=4 buses simultáneos. |
| Empresa 27 | No viable: dataset casi vacío (6 registros). |
| Empresa 45 | No viable: mediana=4 buses simultáneos. |
| Empresa 55 | Viable (PCA=5.14, mediana=6). Fuera del alcance por restricciones de tiempo; el rango de flota pequeña ya está cubierto por la empresa 58 (mediana=6). |
| Empresa 56 | No viable: una sola unidad. |

**5\. La Inteligencia Artificial**

**5.1 Enfoque multi-unidad**

La clave de este paper es que el modelo necesita ver a todos los buses del corredor simultáneamente para predecir bien. El headway entre el bus 5 y el bus 6 no depende solo de esos dos buses: depende de lo que están haciendo el bus 4, el bus 7, y toda la cadena. Si el bus 4 frena, el bus 5 lo alcanza, lo que reduce el headway entre 4 y 5 pero también afecta al headway entre 5 y 6\. Esto es un efecto de propagación que solo se captura modelando todas las unidades simultáneamente.

**5.2 Arquitecturas candidatas**

**Modelo principal: GNN+LSTM**

**GNN (Graph Neural Network):** Modela el corredor como un grafo lineal donde cada bus es un nodo y las conexiones representan la relación entre buses consecutivos. Captura cómo el comportamiento de un bus afecta a sus vecinos (propagación espacial).

**LSTM (Long Short-Term Memory):** Captura cómo los headways evolucionan en el tiempo. Los headways de hace 10-30 minutos influyen en los headways futuros.

**Juntos:** La GNN procesa las relaciones espaciales entre buses en cada instante, y el LSTM conecta esos instantes a lo largo del tiempo. Si la GNN+LSTM supera al LSTM solo, se demuestra que modelar las relaciones espaciales entre buses aporta valor. Esto es un hallazgo publicable por sí mismo.

**Baseline de deep learning: LSTM solo**

Trata el vector de headways como una secuencia plana sin modelar relaciones espaciales entre vecinos. Sirve para demostrar que la GNN aporta valor adicional.

**Baseline estadístico**

Predicción ingenua: “el headway futuro será igual al actual” y promedio móvil. Sirve para demostrar que la IA aporta valor real sobre métodos simples.

**Nota:** La arquitectura final se determinará experimentalmente. La idea y la metodología son lo primero; la herramienta de IA es secundaria y se elige según los resultados. La comparativa entre arquitecturas es parte del diseño experimental del paper.

**6\. Contribución científica**

**6.1 ¿Qué existe en la literatura?**

• Detección de anomalías en trayectorias individuales de buses y taxis (cientos de papers)

• Predicción de headways usando datos de un solo bus contra su historial

• Análisis de regularidad de servicio con métodos estadísticos

• Modelos de tráfico agregado con sensores viales o datos de taxis

**6.2 ¿Qué NO existe y nosotros proponemos?**

**1\. Predicción del vector completo de headways.** No un solo headway, sino todos los headways del corredor simultáneamente, capturando la dinámica del sistema completo.

**2\. Modelado de la propagación espacial entre buses.** Usando GNN para capturar cómo el comportamiento de un bus afecta a sus vecinos en el corredor, mejorando la predicción sobre métodos que tratan cada headway de forma independiente.

**3\. Anticipación de anomalías colectivas.** Bunching, gaps y congestión se detectan como consecuencia natural de la predicción de headways, antes de que ocurran.

**4\. Validación sobre corredores reales con flotas de diferente tamaño.** 4 corredores, desde 6 hasta 20 buses simultáneos (mediana), usando solo GPS básico. Viable para economías emergentes.

**7\. Comparación con el paper anterior**

| Aspecto | Paper anterior | Propuesta actual |
| ----- | ----- | ----- |
| **Objetivo** | Detectar anomalías (reactivo) | Predecir headways (proactivo) |
| **Validación** | Anomalías 100% sintéticas | Ground truth real (headways futuros en los datos GPS) |
| **Alcance** | 46 unidades de 1 empresa, 1 corredor | 102 unidades operacionales de 4 empresas, 4 corredores (120 unidades en raw, 18 excluidas por estar siempre estacionadas) |
| **Arquitectura** | LSTM Autoencoder único sin comparativa | GNN+LSTM vs LSTM vs baseline estadístico |
| **Multi-unidad** | Se aplana todo en un vector, perdiendo identidad de cada bus | GNN preserva relaciones espaciales entre buses consecutivos |
| **Clasificación de anomalías** | Heurística externa no desarrollada | Emerge naturalmente de la predicción de headways |
| **Métricas** | F1=0.109, AUC=0.526 | MAE y RMSE en minutos (claras y verificables) |
| **Framing** | “BRT” (cuestionable) | Transporte público urbano integrado (correcto) |

**8\. Resumen y próximos pasos**

**8.1 Resumen ejecutivo**

|  | Definición |
| ----- | ----- |
| **Problema** | La irregularidad de headways causa bunching, gaps y congestión en corredores de transporte público. Los operadores no pueden anticipar estos problemas con la supervisión manual actual. |
| **Solución** | Modelo de IA (GNN+LSTM) que predice el vector completo de headways del corredor, permitiendo anticipar anomalías colectivas antes de que se materialicen. |
| **Input** | Headways actuales y recientes de todos los buses activos \+ contexto temporal (hora, día). |
| **Output** | Vector de headways predicho para el siguiente instante. |
| **Validación** | Ground truth real: los headways futuros ya existen en los datos GPS. Error medido en minutos (MAE, RMSE). |
| **Datos** | ~47.68M registros GPS sobre los 4 corredores seleccionados (filtrados del dataset crudo de 98.97M tras dedup de las 12 empresas), flotas de 6 a 20 buses simultáneos en mediana, 5 meses. |
| **Novedad** | Predicción multi-headway con modelado de propagación espacial entre buses mediante GNN, usando solo GPS básico. |

**8.2 Próximos pasos**

1\. Preprocesamiento: convertir datos GPS de los 4 corredores en series temporales de headways (reconstrucción del trazado, proyección lineal, identificación ida/vuelta, cálculo de headways, sincronización temporal).

2\. Implementar baseline estadístico (predicción ingenua y promedio móvil).

3\. Implementar LSTM como baseline de deep learning.

4\. Implementar GNN+LSTM como modelo principal.

5\. Evaluar sobre los 4 corredores y comparar arquitecturas.

6\. Redactar paper con resultados empíricos.

*Se solicita retroalimentación de los asesores sobre la reformulación del alcance y la dirección propuesta antes de proceder con la implementación.*