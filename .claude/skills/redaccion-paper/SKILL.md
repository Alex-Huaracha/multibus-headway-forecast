---
name: redaccion-paper
description: "Trigger: redactar el paper, escribir o editar docs/paper/paper.md, resumen, introducción, trabajos relacionados, método, resultados, conclusión. Aplica las directivas IEEE/IJACSA del manuscrito."
license: Apache-2.0
metadata:
  author: "Alex Huaracha"
  version: "1.0"
---

## Activation Contract

Activar al escribir, editar, traducir o revisar prosa de `docs/paper/paper.md`, o al redactar cualquier sección del manuscrito.

No activar para figuras, tablas, código de `src/`, ni documentos de `docs/` que no sean el manuscrito.

## Scope

Esta skill gobierna **cómo se escribe**: integridad de las cifras, registro, voz, léxico, conectores, sintaxis y notación.

**No gobierna cómo se estructura el paper.** Qué secciones existen, en qué orden van, qué contiene cada una y dónde vive cada resultado son decisiones del autor. La skill no las valida, no las exige y no las bloquea. Ante una petición de reestructuración, aplicar la norma al texto resultante y no discutir la estructura elegida.

## Hard Rules

- Leer `references/reglas-redaccion.md` COMPLETO antes de escribir la primera palabra. Ese archivo es la norma vinculante; este solo enruta.
- PRIORIDAD CERO vence a toda regla de estilo: ninguna cifra, cita o resultado se inventa.
- Las cifras se copian abriendo su fuente. Prohibido escribirlas de memoria o recalcularlas.
- Ante fuentes en conflicto, gana el CSV primitivo de `docs/resultados/`.
- Las citas se copian de `docs/paper/fuentes-verificadas.md`, y solo si la entrada está marcada como verificada.
- El español es neutro y profesional: sin voseo, sin regionalismos, sin marcadores coloquiales.
- Recorrer la Sección 10 antes de entregar. El informe va en la respuesta, nunca dentro del manuscrito.

## Decision Gates

| Situación | Acción |
|---|---|
| Falta una cifra | Abrir la fuente de verdad. Si no está ahí, escribir `[INSERTAR DATO/MÉTRICA]`. |
| Dos fuentes discrepan | Gana el CSV primitivo de `docs/resultados/`. |
| Falta una cita o no está verificada | Escribir `[CITA_REQUERIDA]`. |
| Dos reglas chocan | Aplicar el orden de prioridad de la Sección 9. |
| El párrafo no llega al mínimo de palabras | Dejarlo fuera de rango y anotarlo. Nunca rellenar. |
| Se inserta una ecuación en el medio | Renumerar las siguientes y actualizar sus referencias. |
| Se mueve o renumera una sección | Actualizar toda referencia cruzada que la nombre. |
| El autor decide una estructura distinta | Acatarla. Aplicar la norma al texto, no a la estructura. |

## Execution Steps

1. Leer `references/reglas-redaccion.md`.
2. Identificar qué tipo de contenido se redacta y su tiempo verbal (Sección 6).
3. Reunir cifras y citas desde sus fuentes ANTES de redactar.
4. Redactar.
5. Ejecutar los puntos de verificación de la Sección 10, abriendo el archivo en los puntos 1, 2, 8 y 9.
6. Corregir lo que falle y volver al paso 5.

## Output Contract

Devolver, en este orden:
- El texto redactado, sin el chequeo embebido.
- El informe de los puntos de verificación, cada uno como cumplido o incumplido.
- La lista de marcadores insertados y qué falta para resolverlos.

## References

- `references/reglas-redaccion.md` — las directivas completas. Norma vinculante.
