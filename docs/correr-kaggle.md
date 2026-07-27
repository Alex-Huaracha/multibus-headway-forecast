# Runbook: re-corrida de kernels DL en Kaggle (Fase 9, recertificación)

Guía autocontenida para lanzar la re-corrida de los 6 modelos DL en Kaggle
desde cualquier máquina, verificar que cada corrida sea válida y descargar los
outputs a las rutas exactas que consumen los análisis locales.

Contexto: los notebooks regenerados incluyen (a) el fix de winsorización sobre
todos los splits, (b) el gate de hashes congelados de insumos, y (c) la feature
de día atípico ACTIVADA (obligatoria — antes entrenaba en cero sin avisar).
Los resultados nuevos pueden diferir de los viejos por (a) y (c); se reportan
tal cual salgan.

## 1. Prerrequisitos

```bash
git clone <remote> multibus-headway-forecast   # o git pull si ya existe
cd multibus-headway-forecast
git log --oneline -1   # debe incluir o ser posterior a f36a2ba (input-hash gate)
```

- Credenciales de Kaggle en `~/.kaggle/access_token` (Kaggle → Settings → API →
  Create New Token), permisos `chmod 600`. **No es `kaggle.json`**: ese es el
  nombre del formato antiguo y no es el que usa esta máquina.
- CLI de Kaggle: es dependencia del proyecto, se invoca **siempre** como
  `uv run kaggle ...`, nunca como `kaggle` global ni vía `pip install`.
- Verificación rápida: `uv run kaggle kernels list -m -p 1` debe responder sin error.

NO se re-corren los kernels fuente (`02-eda-corridors`, `04-preprocessing`,
`10-baselines-multi-horizonte`, `16-e4-data-baselines`): sus outputs están
congelados por hash. Tampoco los notebooks legacy 05-09/14/15 (fuera de
alcance) ni `15-lstm-multiseed-*`.

## 2. Los 24 kernels

| Familia | Carpeta local | Kernel id (`alexhuaracha/…`) | Horizontes |
|---|---|---|---|
| 11 LSTM (E2+E59) | `notebooks/11_lstm_multihorizon/h{H}/` | `11-lstm-multihorizon-h{H}` | 1, 3, 5, 10 |
| 12 SpatialConvLSTM (E2+E59) | `notebooks/12_spatial_conv_lstm_multihorizon/h{H}/` | `12-spatialconvlstm-multihorizon-h{H}` — **h10 usa `…-h10b`** | 1, 3, 5, 10 |
| 13 SpatialTransformer (E2+E59) | `notebooks/13_spatial_transformer_multihorizon/h{H}/` | `13-spatialtransformer-multihorizon-h{H}` | 1, 3, 5, 10 |
| 17 LSTM E4 | `notebooks/17_e4_lstm/h{H}/` | `17-e4-lstm-h{H}` | 1, 3, 5, 10 |
| 18 ConvLSTM E4 | `notebooks/18_e4_convlstm/h{H}/` | `18-e4-convlstm-h{H}` | 1, 3, 5, 10 |
| 19 Transformer E4 | `notebooks/19_e4_transformer/h{H}/` | `19-e4-transformer-h{H}` | 1, 3, 5, 10 |

IMPORTANTE — `12-spatialconvlstm-multihorizon-h10b`: el kernel `…-h10` original
está corrupto en Kaggle (commit `e0757b6`). El `kernel-metadata.json` local ya
apunta a `h10b`; no lo "corrijas" a `h10`.

Prioridad si la cuota de GPU no alcanza para todo de una vez: primero
h3, h5, h10 (los análisis de significancia/volatilidad consumen esos
residuales), h1 al final (solo alimenta las tablas de resultados agregados).

## 3. Lanzar las corridas

`kaggle kernels push` sube el notebook + metadata como NUEVA versión y la
encola en Kaggle (GPU T4 x2, según la metadata). Por tandas de una familia:

```bash
# Ejemplo: familia 11 completa
for H in 3 5 10 1; do
  kaggle kernels push -p notebooks/11_lstm_multihorizon/h$H/
done
```

Todas las familias:

```bash
for DIR in 11_lstm_multihorizon 12_spatial_conv_lstm_multihorizon \
           13_spatial_transformer_multihorizon 17_e4_lstm 18_e4_convlstm \
           19_e4_transformer; do
  for H in 3 5 10 1; do
    kaggle kernels push -p notebooks/$DIR/h$H/
  done
done
```

Estado de una corrida:

```bash
kaggle kernels status alexhuaracha/11-lstm-multihorizon-h3
# "running" | "complete" | "error"
```

## 4. Validar cada corrida ANTES de aceptar sus outputs

Descargá el log (viene incluido en `kaggle kernels output`, ver §5) y revisá:

1. **Gate de insumos pasó**: NO debe aparecer `Required input not found` ni
   `does not match its frozen SHA-256`. Si aparece, la corrida se detuvo antes
   de entrenar (no gastó GPU de entrenamiento): revisar que `kernel_sources`
   de la metadata incluya `alexhuaracha/02-eda-corridors` y reintentar.
2. **Feature atípica activa**: buscar `Atypical days loaded: N dates` con
   **N > 0**. Si N = 0 la corrida falla sola (ValueError) — no debe aceptarse
   ningún output sin esta línea.
3. **Winsorización**: aparece `winsorize threshold = … min` por corredor
   (E2 ≈ 28.4679).
4. Sin `Traceback` al final y estado `complete`.

Si una corrida termina en `error`: leer el log primero, corregir la causa y
recién ahí relanzar. No relanzar a ciegas.

## 5. Descargar outputs a las rutas correctas

Los análisis locales leen EXACTAMENTE estas rutas:

- Residuales + results + logs (h3/h5/h10):
  `docs/resultados/residuos-multihorizon/<model-dir>/h{H}/`
  donde `<model-dir>` es `11-lstm`, `12-spatialconvlstm` o
  `13-spatialtransformer`. Los kernels E4 (17/18/19) comparten carpeta con su
  familia hermana (17→`11-lstm`, 18→`12-spatialconvlstm`,
  19→`13-spatialtransformer`).
- Results CSVs de TODOS los horizontes (h1 incluido):
  `docs/resultados/csv-multihorizon/`

Script completo (correr desde la raíz del repo, cuando TODAS las corridas
estén `complete`):

```bash
#!/usr/bin/env bash
set -euo pipefail

slug() {  # slug del kernel: familia + horizonte (caso especial 12/h10 -> h10b)
  local fam=$1 h=$2
  if [ "$fam" = "12-spatialconvlstm-multihorizon" ] && [ "$h" = "10" ]; then
    echo "${fam}-h10b"
  else
    echo "${fam}-h${h}"
  fi
}

download() {  # download <familia-slug> <model-dir> <horizonte>
  local fam=$1 dir=$2 h=$3
  local id tmp
  id="alexhuaracha/$(slug "$fam" "$h")"
  tmp=$(mktemp -d)
  echo ">> $id"
  kaggle kernels output "$id" --path "$tmp"
  if [ "$h" != "1" ]; then
    mkdir -p "docs/resultados/residuos-multihorizon/$dir/h$h"
    cp "$tmp"/*.csv "$tmp"/*.log "docs/resultados/residuos-multihorizon/$dir/h$h/"
  fi
  cp "$tmp"/*results*.csv docs/resultados/csv-multihorizon/
  rm -rf "$tmp"
}

for H in 3 5 10 1; do
  download 11-lstm-multihorizon              11-lstm              "$H"
  download 17-e4-lstm                        11-lstm              "$H"
  download 12-spatialconvlstm-multihorizon   12-spatialconvlstm   "$H"
  download 18-e4-convlstm                    12-spatialconvlstm   "$H"
  download 13-spatialtransformer-multihorizon 13-spatialtransformer "$H"
  download 19-e4-transformer                 13-spatialtransformer "$H"
done
echo "Descarga completa."
```

## 6. Después de descargar

```bash
git checkout -b tmp-verificacion 2>/dev/null || true   # opcional, si querés revisar antes de tocar main
git status   # deben aparecer solo CSVs/logs bajo docs/resultados/
git add docs/resultados/
git commit -m "data(resultados): fresh Kaggle outputs, recertified runs (phase 9)"
git push
```

Con los residuales frescos en su lugar se destraban las fases locales
pendientes (ver `openspec/changes/paper-recertification/tasks.md`):

- Fase 5: `uv run python src/build_exante_volatility.py` y
  `src/build_exante_correlation.py` (lento — correrlo en una máquina que lo
  aguante).
- Fase 10: regenerar significancia/degradación/volatilidad/paired-audit y
  reescribir `documento-resultados.md` con las métricas pareadas como
  canónicas.

## 7. Problemas conocidos

- `403 Forbidden` en push: token de API vencido o kernel privado de otra
  cuenta — regenerar `~/.kaggle/access_token`.
- `kaggle kernels output` descarga TODOS los outputs del kernel (parquets
  grandes incluidos en los kernels fuente; los DL solo emiten CSVs + log).
- Cuota GPU semanal de Kaggle (~30 h): las 24 corridas pueden no entrar en una
  sola semana. Respetar la prioridad del §2.
- El gate de hashes corta ANTES de entrenar: una corrida bloqueada por el gate
  no consume GPU de entrenamiento y no deja outputs parciales aceptables.

---

## 8. Rolling origin: los cortes r1 y r2

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

### Validar antes de aceptar outputs

Además de los cuatro chequeos del §4, para estos kernels revisar:

1. **`Fold: r1`** (o `r2`) en el log. Si dice `main`, el notebook se construyó
   mal y estaría re-midiendo la ventana publicada.
2. **`Fold r1: train 2023-10-01..2023-11-30 (61d) | ...`** — las fechas impresas
   deben coincidir con la tabla de arriba.
3. El portón de población compara contra los digests **de ese corte**; si pasa,
   la población es la correcta por construcción.

### Descargar

Las salidas llevan el sufijo del corte (`lstm_contig_r1_residuals_h3.csv`), así
que **no pisan** los residuos publicados:

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

### Qué NO se re-corre

El XGBoost. Su papel es mostrar que el cruce no es una propiedad del Deep
Learning, y eso ya está establecido sobre el corte publicado. Replicarlo en el
tiempo agregaría ≈6 h de CPU para responder una pregunta que nadie hizo.
