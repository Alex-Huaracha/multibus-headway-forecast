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

## Hard Rules

- Leer `references/reglas-redaccion.md` COMPLETO antes de escribir la primera palabra. Ese archivo es la norma vinculante; este solo enruta.
- PRIORIDAD CERO vence a toda regla de estilo: ninguna cifra, cita o resultado se inventa.
- Las cifras se copian abriendo su fuente. Prohibido escribirlas de memoria o recalcularlas.
- Las citas se copian de `docs/paper/fuentes-verificadas.md`, y solo si la entrada está marcada como verificada.
- Recorrer la Sección 8 antes de entregar. El informe va en la respuesta, nunca dentro del manuscrito.

## Decision Gates

| Situación | Acción |
|---|---|
| Falta una cifra | Abrir la fuente de verdad. Si no está ahí, escribir `[INSERTAR DATO/MÉTRICA]`. |
| Falta una cita o no está verificada | Escribir `[CITA_REQUERIDA]`. |
| Dos reglas chocan | Aplicar el orden de prioridad de la Sección 7. |
| El párrafo no llega al mínimo de palabras | Dejarlo fuera de rango y anotarlo. Nunca rellenar. |
| Se inserta una ecuación en el medio | Renumerar las siguientes y actualizar sus referencias. |

## Execution Steps

1. Leer `references/reglas-redaccion.md`.
2. Identificar la sección del manuscrito, su tiempo verbal (Sección 3) y su flujo obligatorio (Sección 4).
3. Reunir cifras y citas desde sus fuentes ANTES de redactar.
4. Redactar.
5. Ejecutar los puntos de verificación de la Sección 8, abriendo el archivo en los puntos 1 y 2.
6. Corregir lo que falle y volver al paso 5.

## Output Contract

Devolver, en este orden:
- El texto redactado, sin el chequeo embebido.
- El informe de los puntos de verificación, cada uno como cumplido o incumplido.
- La lista de marcadores insertados y qué falta para resolverlos.

## References

- `references/reglas-redaccion.md` — las directivas completas. Norma vinculante.
