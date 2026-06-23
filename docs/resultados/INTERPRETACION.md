# Interpretación de figuras y tablas de resultados

> Cada figura de `docs/resultados/` tiene aquí su lectura: qué muestra, cómo se
> leen los ejes, qué número citar y qué argumento sostiene en el paper. Una
> figura sin interpretación es decoración, no evidencia.

Empresas evaluadas: **E2**, **E59** y **E4** (E2 y E59 son los corredores de mayor
flota; E4 es el más chico —19 buses— e ingresa como validación externa del patrón).
Métrica principal: **MAE** (minutos); RMSE como respaldo. Todos los valores citados
son del agregado de ambas direcciones, salvo aclaración.

---

## 1. Curva de degradación — `curva-degradacion.png`

**Qué es:** la figura central del paper. Error de pronóstico (MAE arriba, RMSE
abajo) en función del **horizonte** (1, 3, 5, 10 min), comparando la persistencia
(B1) y el mejor baseline estadístico (B3) contra los tres modelos profundos. Una
columna por corredor (E2, E59, E4).

**Cómo se leen los ejes:**
- **X** = horizonte de predicción en minutos (cuán adelante se predice).
- **Y** = error (MAE/RMSE) en minutos. Más bajo = mejor.
- Cada punto DL vs. persistencia está testeado (Diebold-Mariano + Wilcoxon); el
  único no significativo va anillado con `⊘ ns`.

**Qué muestra y por qué importa:**

El problema que originó esta fase aparece en el extremo izquierdo: **a 1 minuto la
persistencia empata o gana.** En E59, B1 = **3.100** vs. LSTM = **3.337** — la
persistencia es *mejor* a 1 min. Predecir a 1 minuto no anticipa nada: el headway
casi no cambió en 60 s, así que repetir el último valor es imbatible y operativamente
inútil (no da margen de intervención).

La historia se da vuelta al **estirar el horizonte**: la persistencia se degrada en
línea recta y los modelos profundos aguantan.

| Corredor | Modelo | h=1 | h=3 | h=5 | h=10 |
|---|---|---|---|---|---|
| E59 | Persistencia (B1) | **3.100** | 4.184 | 4.698 | 5.593 |
| E59 | LSTM | 3.337 | 3.847 | 4.032 | **4.225** |
| E2 | Persistencia (B1) | 4.757 | 6.075 | 6.493 | 7.026 |
| E2 | LSTM | 4.471 | 4.940 | 5.052 | **5.153** |
| E4 | Persistencia (B1) | **3.130** | 4.783 | 5.740 | 7.070 |
| E4 | LSTM | 3.763 | 4.668 | 5.009 | **5.334** |

A h=10 el DL le saca **1.37 min** de MAE a la persistencia en E59, **1.87 min** en
E2 y **1.74 min** en E4. La brecha crece monótonamente con el horizonte en los tres
corredores.

**¿Bueno o malo?** Bueno. Convierte la debilidad a 1 min (donde la persistencia
gana) en el argumento principal: el aporte del DL se demuestra en los horizontes
operativamente útiles (≥3 min), que son los que permiten intervenir antes de que el
bunching o el gap se materialicen.

**Frase para el paper:** *"A un minuto la persistencia es un competidor imbatible
pero operativamente vacío; el valor del pronóstico profundo emerge al anticipar a 3,
5 y 10 minutos, justo el margen que un operador necesita para intervenir."*

---

## 2. Crossover de volatilidad — `volatilidad-crossover.png`

**Qué es:** la figura que explica el **mecanismo causal** detrás de la curva de
degradación. No mira el promedio: separa las predicciones por **cuánto cambió
realmente el headway** en ese intervalo y muestra quién gana en cada régimen.

**Cómo se leen los ejes:**
- **X** = régimen de cambio realizado del headway: **estable** (<1 min),
  **moderado** (1–3 min), **alto** (>3 min). Se mide como
  `|y_real − persistencia|`, porque la persistencia predice el último valor
  observado, así que la magnitud de su error *es* cuánto se movió el headway.
- **Y** = `Δ MAE = MAE(DL) − MAE(persistencia)`. La línea punteada en **0** es el
  empate.
  - **Y > 0** → gana la persistencia (el DL erró más).
  - **Y < 0** → gana el DL.
- Una línea por horizonte (3, 5, 10 min). Modelo representado: LSTM (los modelos
  espaciales dan un crossover casi idéntico).

**Qué muestra — el crossover:** las líneas *cruzan* el cero. Arrancan arriba (en
estable) y terminan abajo (en alto cambio). Es idéntico en las 54 celdas
(modelo × corredor × horizonte × métrica), con la magnitud variando algo por corredor:

| Régimen | Cambio típico | Δ MAE | Quién gana |
|---|---|---|---|
| estable (<1 min) | ~0.46 min | **+2.3 … +3.4** | Persistencia |
| moderado (1–3 min) | ~1.9 min | +0.85 … +1.9 | Persistencia (achicándose) |
| alto (>3 min) | ~9 min | **−2.6 … −3.8** | **DL, decisivo** |

**Por qué importa (la lectura completa):**

El promedio "el DL le gana a la persistencia" es débil por sí solo — un revisor lo
descarta como una mejora marginal. El crossover lo convierte en una historia
mecánica: ese promedio es la **suma de dos regímenes opuestos**.

1. **En ventanas estables el DL pierde, y está bien que pierda.** Si el headway no
   se mueve, repetir el último valor es casi exacto; el DL solo agrega ruido. Pero a
   nadie le importa predecir un servicio que va estable: no hay nada que intervenir.

2. **En ventanas de alto cambio el DL gana por goleada.** Y ese es exactamente el
   escenario donde se forman el bunching y los gaps — donde el operador necesita
   anticiparse. El DL gana *donde importa*.

**Frase para el paper:** *"El modelo profundo no mejora el promedio mejorando todo
un poco; gana precisamente donde importa operativamente —cuando el headway se está
desestabilizando— y cede terreno solo donde la predicción es trivial e irrelevante
para la operación."*

**El segundo golpe — explica la curva de degradación:** ¿por qué a mayor horizonte
el DL saca más ventaja? Porque el horizonte **corre la masa de muestras hacia el
régimen donde el DL gana.** En E59, las ventanas de alto cambio pasan de **38.6%**
(h=3) a **54.4%** (h=10); las estables caen de 38.9% a 20.0%. A 10 minutos, más de la
mitad de los casos caen en el terreno del DL. La curva agregada diverge porque
estirar el horizonte hace que el headway cambie más seguido, y ahí la persistencia
se rompe.

**El matiz honesto (suma, no resta):** como el DL pierde en estable, la conclusión
madura no es "el DL reemplaza a la persistencia", sino que **lo óptimo operativo
sería un sistema híbrido** — persistencia cuando el servicio va estable, DL cuando
detecta que se está desestabilizando. Declararlo en la discusión muestra rigor y
autocrítica.

---

## Tablas de respaldo

- **`csv-multihorizon/significance_multihorizon.csv`** (54 filas) — test pareado
  Diebold-Mariano + Wilcoxon por modelo × métrica × corredor × horizonte. El DL tiene
  el menor error en 52 de las 54 celdas; de esas, **50 son significativas a p<0.001** en
  ambos tests (51 a p<0.05). Con n en los millones el p-valor colapsa a ~0, así que se
  usa como piso de sanidad, no como evidencia (la evidencia es el Δ MAE en minutos).
  Las cuatro desviaciones, todas a h=3: Transformer/E4/h3 (MAE y RMSE, gana la
  persistencia por poco), ConvLSTM/E4/h3/MAE (DL gana, DM p=0.005 → solo signif. a 0.05)
  y LSTM/E59/h3 (DL gana en media pero Wilcoxon p=0.277, débil en mediana).
- **`csv-multihorizon/volatility_multihorizon.csv`** (162 filas) — el mismo test
  pareado pero estratificado por régimen de volatilidad. Sostiene la figura del
  crossover.
- **`consolidated_multihorizon.csv`** — tabla maestra tidy (todos los modelos,
  métricas, corredores, horizontes, direcciones). Fuente de la curva de degradación.

## Reproducibilidad

```sh
uv run python -m src.build_significance_table   # significance_multihorizon.csv
uv run python -m src.build_degradation_curve     # consolidated_multihorizon.csv + curva-degradacion.png
uv run python -m src.build_volatility_table       # volatility_multihorizon.csv
uv run python -m src.build_volatility_curve       # volatilidad-crossover.png
```

Las figuras y tablas se reconstruyen desde los CSV versionados; no requieren los
residuos crudos por-muestra (no versionados, descargados de Kaggle).
