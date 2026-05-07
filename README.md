# multibus-headway-forecast

Predicción del vector completo de headways en corredores de transporte público urbano usando GNN+LSTM, con datos GPS reales del SIT Arequipa.

Publicación objetivo: IJACSA. Propuesta completa en [`docs/propuesta.md`](docs/propuesta.md).

## Convenciones del proyecto

- **Clave compuesta**: siempre `(empresaid, unidadid)` — los `unidadid` se reutilizan entre empresas (28 de 126 aparecen en 3+ empresas). Nunca usar `unidadid` solo.
- **Corredores incluidos**: empresas 2, 4, 58, 59. El resto fue descartado por viabilidad (ver propuesta sección 4.3).
- **Formato de datos procesados**: Parquet. Nada de CSV en el pipeline interno.
- **Datos no van a Git** — viven en Kaggle Datasets (versionados allá) y localmente bajo `data/` (gitignored).

## Estructura

```
data/raw/         GPS crudo (gitignored, descarga local del dataset Kaggle)
data/processed/   Parquets limpios por corredor (gitignored)
notebooks/        Pipeline por fases (EDA, preprocesamiento, headways, baselines, modelos)
src/              Utilidades reusables entre notebooks
kaggle/           Metadata de notebooks y datasets Kaggle
docs/             Propuesta y notas
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Token de Kaggle en `~/.kaggle/access_token` (chmod 600).
