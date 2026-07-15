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
| 11-lstm (E2+E59) | ✅ | ✅ | ✅ | ✅ |
| 12-spatialconvlstm (E2+E59) | ✅ | ✅ | ✅ | ✅ |
| 13-spatialtransformer (E2+E59) | ✅ | ✅ | ✅ | ✅ |
| 17-e4-lstm | ✅ | ✅ | ✅ | ✅ |
| 18-e4-convlstm | ✅ | ✅ | ✅ | ✅ |
| 19-e4-transformer | ✅ | ✅ | ✅ | ✅ |

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
| 11-lstm h5 | E2 | 5.024 | 6.215 | −1.19 |
| 11-lstm h5 | E59 | 3.930 | 4.396 | −0.47 |
| 12-spatialconvlstm h5 | E2 | 5.025 | 6.215 | −1.19 |
| 12-spatialconvlstm h5 | E59 | 3.940 | 4.396 | −0.46 |
| 13-spatialtransformer h5 | E2 | 5.069 | 6.215 | −1.15 |
| 13-spatialtransformer h5 | E59 | 3.948 | 4.396 | −0.45 |
| 17-e4-lstm h5 | E4 | 4.832 | 5.385 | −0.55 |
| 18-e4-convlstm h5 | E4 | 4.872 | 5.385 | −0.47 |
| 19-e4-transformer h5 | E4 | 4.914 | 5.385 | −0.47 |
| 11-lstm h10 | E2 | 5.163 | 6.734 | −1.57 |
| 11-lstm h10 | E59 | 4.188 | 5.282 | −1.09 |
| 12-spatialconvlstm h10 | E2 | 5.165 | 6.734 | −1.57 |
| 12-spatialconvlstm h10 | E59 | 4.205 | 5.282 | −1.08 |
| 13-spatialtransformer h10 | E2 | 5.177 | 6.734 | −1.56 |
| 13-spatialtransformer h10 | E59 | 4.182 | 5.282 | −1.10 |
| 17-e4-lstm h10 | E4 | 5.360 | 6.776 | −1.42 |
| 18-e4-convlstm h10 | E4 | 5.381 | 6.776 | −1.40 |
| 19-e4-transformer h10 | E4 | 5.407 | 6.776 | −1.37 |
| 11-lstm h1 | E2 | 4.274 | 4.237 | **+0.04 (gana persist)** |
| 11-lstm h1 | E59 | 3.163 | 2.820 | **+0.34 (gana persist)** |
| 13-spatialtransformer h1 | E2 | 4.294 | 4.237 | **+0.06 (gana persist)** |
| 13-spatialtransformer h1 | E59 | 3.159 | 2.820 | **+0.34 (gana persist)** |
| 12-spatialconvlstm h1 | E2 | 4.274 | 4.237 | **+0.04 (gana persist)** |
| 12-spatialconvlstm h1 | E59 | 3.153 | 2.820 | **+0.33 (gana persist)** |
| 17-e4-lstm h1 | E4 | 3.367 | 2.844 | **+0.52 (gana persist)** |
| 18-e4-convlstm h1 | E4 | 3.372 | 2.844 | **+0.53 (gana persist)** |
| 19-e4-transformer h1 | E4 | 3.419 | 2.844 | **+0.57 (gana persist)** |

**Grilla completa: 24/24 kernels validados.** El DL le gana a la persistencia en horizontes
medios/largos (h3/h5/h10) y la ventaja crece con el horizonte. En h1 la persistencia gana
(ancla esperada: a 1 paso el naive es imbatible). Listo para fase 10:
recalcular significancia (DM/Wilcoxon), degradación, volatilidad y paired-audit sobre estos
residuos frescos, y reescribir `documento-resultados.md`.
