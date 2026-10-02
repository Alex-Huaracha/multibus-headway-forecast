# Equivalencias de las rutas

El paper nombra cada ruta con una letra, como Yu et al. (2016) con sus
«Route A» y «Route B». El código, los CSV y `docs/resultados/` siguen usando
el identificador de la empresa operadora.

| Paper | Código y CSV (`corridor`) | `empresaid` |
|---|---|---|
| Ruta A | E2 | 2 |
| Ruta B | E4 | 4 |
| Ruta C | E59 | 59 |

La traducción vive en `ROUTE_LABELS` de `src/build_paper_tables.py` y de
`src/build_contiguous_figures.py`. Solo las figuras del paper llevan la letra;
la variante de `documento-resultados.md` conserva el código de la empresa.
