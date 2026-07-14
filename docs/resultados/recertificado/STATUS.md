# Recertificación — outputs frescos (fase 9)

Outputs de Kaggle **posteriores** a los fixes (winsorización sobre todos los splits + feature
de día atípico ACTIVA + loader que lee columna `day`). Reemplazan conceptualmente a
`docs/resultados/residuos-multihorizon/` (junio, pre-fix, obsoletos).

- Residuales pesados (~165 MB): `residuos-multihorizon/<modelo>/h{H}/` — **fuera de git** (regenerables).
- Results chicos: `csv-multihorizon/` — **trackeados** (reproducibilidad del paper).

## Flujo por kernel (validado)

1. CLI: regenerar notebook (loader corregido) + `uv run kaggle kernels push`.
2. Web (una vez por kernel): **Add Input → `Atypical Days Frozen`** + seleccionar **T4x2** + **Save & Run All**.
3. Validar log: `Atypical days loaded: 17 dates`, winsorize E2 ≈ 28.4679, sin traceback, `complete`.
4. CLI: bajar outputs a las rutas de acá.

## Progreso

Leyenda: ✅ completo y bajado · 📤 código subido, falta paso web · ⬜ pendiente

| Familia | h1 | h3 | h5 | h10 |
|---|---|---|---|---|
| 11-lstm (E2+E59) | ⬜ | ✅ | ⬜ | ⬜ |
| 12-spatialconvlstm (E2+E59) | ⬜ | ✅ | ⬜ | ⬜ |
| 13-spatialtransformer (E2+E59) | ⬜ | ✅ | ⬜ | ⬜ |
| 17-e4-lstm | ⬜ | ✅ | ⬜ | ⬜ |
| 18-e4-convlstm | ⬜ | ✅ | ⬜ | ⬜ |
| 19-e4-transformer | ⬜ | ✅ | ⬜ | ⬜ |

## Números validados (DL vs persistencia B1, misma muestra)

| Kernel | Corredor | DL MAE | Persist MAE | Δ (min) |
|---|---|---|---|---|
| 11-lstm h3 | E2 | 4.857 | 5.762 | −0.90 |
| 11-lstm h3 | E59 | 3.721 | 3.899 | −0.18 |
| 12-spatialconvlstm h3 | E2 | 4.860 | 5.762 | −0.90 |
| 12-spatialconvlstm h3 | E59 | 3.726 | 3.899 | −0.17 |
| 13-spatialtransformer h3 | E2 | 4.878 | 5.762 | −0.88 |
| 13-spatialtransformer h3 | E59 | 3.753 | 3.899 | −0.15 |
| 17-e4-lstm h3 | E4 | 4.398 | 4.438 | −0.04 |
| 18-e4-convlstm h3 | E4 | 4.418 | 4.438 | −0.02 |
| 19-e4-transformer h3 | E4 | 4.477 | 4.438 | **+0.04 (gana persist)** |

El DL le gana a la persistencia en ambos corredores con el pipeline corregido. Falta completar
el resto de la grilla antes de recalcular significancia/degradación/paired-audit (fase 10).
