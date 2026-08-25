# Runbook: correr kernels en Kaggle

Guía para lanzar corridas del **pipeline contiguo** (familias 21 y 22), verificar
que cada una sea válida y descargar los outputs a las rutas que consumen los
análisis locales.

> Las familias 11/12/13, 17/18/19, 14/15/16 y 20 están **congeladas**: no se
> re-corren. Sostienen únicamente el ranking entre arquitecturas, cuya validez
> descansa en que las tres comparten el mismo sesgo. Su runbook se retiró de acá
> porque instruía verificaciones que ya no aplican — la bandera de día atípico,
> por ejemplo, la eliminó el contrato C3.

## 1. Prerrequisitos

```bash
cd multibus-headway-forecast
uv sync
cp .env.example .env                 # pegar el token; ver comentarios del archivo
$env:UV_ENV_FILE = ".env"            # una vez por shell — sin esto no autentica
uv run kaggle kernels list -m -p 1   # debe responder sin error
```

- Credenciales: `KAGGLE_API_TOKEN` en `.env` (gitignored). El token sale de
  Kaggle → Settings → API → Create New Token; **no** lleva usuario.
- **Todos los `uv run kaggle ...` de esta guía asumen `UV_ENV_FILE` exportado.**
  `uv run` no lee `.env` por su cuenta; la alternativa por llamada es
  `uv run --env-file .env kaggle ...`.
- El CLI es dependencia del proyecto: **siempre** `uv run kaggle ...`, nunca
  `kaggle` global ni `pip install kaggle`.

No se re-corren los kernels fuente (`04-preprocessing`,
`10-baselines-multi-horizonte`, `16-e4-data-baselines`): sus outputs están
congelados por hash.

## 2. Rolling origin: los cortes r1 y r2

Todo resultado publicado sale de UNA ventana de test de 22 días (febrero 2024).
Los cortes `r1` y `r2` re-corren el protocolo completo sobre dos ventanas
anteriores, para responder si el hallazgo se sostiene fuera de ese período.

| Corte | Entrena | Valida | Prueba |
|---|---|---|---|
| `r1` | 2023-10-01 → 2023-11-30 (61 d) | 2023-12-01 → 2023-12-22 | **2023-12-23 → 2024-01-13** |
| `r2` | 2023-10-01 → 2023-12-22 (83 d) | 2023-12-23 → 2024-01-13 | **2024-01-14 → 2024-02-04** |
| `main` | 2023-10-01 → 2024-01-15 (107 d) | 2024-01-16 → 2024-02-07 | **2024-02-08 → 2024-02-29** |

`main` **ya está corrido**: es el último origen de la secuencia, no un análisis
aparte. Solo hay que lanzar r1 y r2 → **16 kernels, ≈1.8 h de GPU**.

El XGBoost **no** se re-corre. Su papel es mostrar que el cruce no es propiedad
del Deep Learning, y eso ya está establecido sobre el corte publicado.

### Lanzar

```bash
uv run python src/build_notebook_21_lstm_contiguous.py   # emite los 24 (3 cortes)

for FOLD in r1 r2; do
  for GRP in e2e59 e4; do
    for H in 3 5 10 1; do
      uv run kaggle kernels push -p notebooks/21_lstm_contiguous/$GRP/$FOLD/h$H/
    done
  done
done
```

Kaggle admite **2 sesiones GPU simultáneas**: lanzar de a dos y esperar. Un
`CANCEL_REQUESTED` con `Maximum batch GPU session count of 2 reached` significa
exactamente eso, no un error del notebook.

Estado de una corrida:

```bash
uv run kaggle kernels status alexhuaracha/21-lstm-contiguous-h3-r1
```

## 3. Validar ANTES de aceptar outputs

En el log de cada corrida:

| # | Qué buscar | Si falla |
|---|---|---|
| 1 | `Fold: r1` (o `r2`) | El notebook se construyó mal y está re-midiendo febrero. **Descartar el output.** |
| 2 | `Fold r1: train 2023-10-01..2023-11-30 (61d) \| ...` | Las fechas deben coincidir con la tabla del §2. |
| 3 | Sin `Required input not found` ni `does not match its frozen SHA-256` | El portón de insumos cortó antes de entrenar. No gastó GPU. |
| 4 | Sin `SHARED-POPULATION GATE FAILED` | El índice reconstruido no coincide con el manifiesto congelado. |
| 5 | `<corredor>: winsor threshold=…` por corredor (2 líneas en e2e59, 1 en e4) | Ausente ⇒ el preprocesamiento no corrió completo. |
| 6 | Sin `Traceback` y estado `complete` | — |

Si una corrida termina en `error`: leer el log, corregir la causa, y recién ahí
relanzar. No relanzar a ciegas.

## 4. Descargar

Cada salida lleva **grupo de corredores y corte** en el nombre
(`lstm_contig_E4_r1_residuals_h3.csv`), así que las 24 corridas conviven en la
misma carpeta sin pisarse. Las dos coordenadas son necesarias: sin la del grupo,
E2+E59 y E4 del mismo corte escriben el mismo archivo y la segunda descarga
borra la primera **en silencio** — `build_contiguous_significance.load_lstm`
saltea el archivo que falta y reporta métricas sobre medio corpus. Lo guarda
`test_every_output_filename_is_unique`.

```bash
for FOLD in r1 r2; do
  for H in 1 3 5 10; do
    uv run kaggle kernels output alexhuaracha/21-lstm-contiguous-h$H-$FOLD \
      -p docs/resultados/residuos-multihorizon/21-lstm-contiguous/
    uv run kaggle kernels output alexhuaracha/21-lstm-contiguous-e4-h$H-$FOLD \
      -p docs/resultados/residuos-multihorizon/21-lstm-contiguous/
  done
done
```

⚠️ `21-lstm-contiguous-h10-r2` **no existe**: su slug quedó podrido en Kaggle y
esa corrida vive en `21-lstm-contiguous-h10-r2b` (ver `POISONED_SLUGS` en el
builder y el §5). El bucle de arriba no la trae — bajala aparte.

Verificar el store antes de commitear. Los 16 de rolling son los que acaba de
traer el bucle; los 8 del corte publicado tienen que seguir ahí de su propia
corrida — si faltan, algo los pisó:

```bash
D=docs/resultados/residuos-multihorizon/21-lstm-contiguous
fd "lstm_contig_r[12]_residuals" $D | wc -l    # 8 — E2+E59, r1 y r2
fd "lstm_contig_E4_r[12]_residuals" $D | wc -l # 8 — E4,     r1 y r2
fd "lstm_contig_residuals" $D | wc -l          # 4 — E2+E59, corte publicado
fd "lstm_contig_E4_residuals" $D | wc -l       # 4 — E4,     corte publicado
```

## 5. Problemas conocidos

| Síntoma | Causa y qué hacer |
|---|---|
| `403 Forbidden` en push | Token vencido o kernel de otra cuenta. Regenerar el token y actualizar `KAGGLE_API_TOKEN` en `.env`. |
| `no kernel image is available for execution on the device` | Desajuste de entorno GPU (P100 en vez de T4×2). **Se corrige desde la web**, no desde el CLI ni el builder. |
| `CANCEL_REQUESTED` + `Maximum batch GPU session count of 2 reached` | Límite de sesiones. Esperar y relanzar. |
| `kaggle kernels output` trae archivos gigantes | Descarga TODOS los outputs; los kernels fuente incluyen parquets. Los DL solo emiten CSVs + log. |
| `Kernel push error: Notebook not found` **y el CLI sale con código 0** | El slug quedó podrido del lado de Kaggle: no figura en `kernels list`, y da 404 en `status` y en `output`. No se arregla desde acá — hay que correrlo bajo un slug nuevo (`POISONED_SLUGS` en el builder). **Nunca uses el exit code del CLI como señal de éxito: hay que leer el texto.** |
| `409 Conflict` al renombrar un kernel | Kaggle deriva el slug **del título**, no del id. Si movés uno solo, avisa `your kernel title does not resolve to the specified id` y rechaza. Mové los dos juntos. |
| Un `kernel_sources` nuevo no queda adjunto tras `push` | Kaggle no adjunta de forma confiable una fuente nunca adjuntada antes. Requiere un **"Add Input" único desde la web**; después el CLI la preserva. |

Cuota GPU semanal: ~30 h. Las 16 corridas de rolling entran cómodo (~1.8 h).
