# Informe EV2: modelado y evaluacion

**Prediccion de la popularidad de canciones de Spotify con Ridge, Random Forest y XGBoost**

| | |
|---|---|
| **Asignatura** | MLY1101 Machine Learning, DuocUC |
| **Evaluacion** | Parcial Nro. 2: presentacion y defensa tecnica del proyecto |
| **Equipo** | Grupo 5: Cristian Romero y Rolando Paredes |
| **Caso** | C: inteligencia musical y prediccion de popularidad de canciones (Spotify Tracks) |
| **Metodologia** | CRISP-DM, fases 4 (modelado) y 5 (evaluacion). Las fases 1 a 3 estan en `../ev1/notebook_ev1.ipynb` |
| **Respaldo tecnico** | `ev2/notebook_ev2.ipynb`: cada cifra y figura de este informe sale de una celda del notebook |

## Resumen

- **Pregunta:** ¿preparar los datos segun lo que necesita cada modelo mejora los resultados frente a usar la preparacion de la EV1 para todos?
- **Metodo:** solo cambia la preparacion; los modelos usan sus parametros por defecto. Para cada modelo se probaron hipotesis una a una, sacadas de su documentacion o del EDA: se cambia una cosa, se mide en validacion cruzada y se adopta solo si mejora mas que el ruido.
- **Respuesta: si en los dos modelos de arboles, no en Ridge.** XGBoost baja su MAE de prueba de 11,30 a 10,64 con el genero como categoria nativa, y Random Forest de 10,99 a 10,85 con el genero en one-hot. Para Ridge, la preparacion de la EV1 ya era la adecuada.
- **Veredicto:** ningun modelo cumple los seis criterios de exito de la EV1. **XGBoost con preparacion propia es el mejor candidato** (37% menos error que la linea base), pero todavia no es apto: le faltan R², RMSE, error en exitos y control del sobreajuste.
- **Que explica la popularidad:** el genero (72% de la importancia). Los exitos se subestiman en unos 29 puntos.

## Indice

1. Problema de negocio
2. Objetivos, hipotesis y criterios
3. Fuentes de datos
4. Analisis exploratorio (EDA)
5. Preparacion y transformacion de los datos
6. Justificacion de los 3 modelos
7. Variables finales por modelo
8. Implementacion
9. Evaluacion y seleccion del modelo
10. Conclusiones y recomendaciones
11. Limitaciones y proximos pasos
12. Reproducibilidad
13. Referencias

---

## 1. Problema de negocio

Un sello o una plataforma decide cada semana en que canciones invertir: cuales promocionar, cuales proponer para playlists y con que artistas trabajar. Muchas veces lo decide cuando el exito ya ocurrio. El proyecto estima la popularidad de una cancion (indice de Spotify de 0 a 100, basado en reproducciones recientes) a partir de su audio, su genero y metadatos basicos. La idea es que sirva de apoyo para priorizar antes de invertir.

Es una **regresion supervisada**: el objetivo `popularity` es numerico y hay ejemplos con su valor conocido.

*Detalle: `../ev1/notebook_ev1.ipynb`, seccion 1.*

## 2. Objetivos, hipotesis y criterios

**Objetivo general.** Preparar los datos segun lo que necesita cada modelo, entrenar y comparar tres modelos de regresion, saber cual rinde mejor y por que, y verificar si cumplen los criterios de exito definidos en la EV1.

**Objetivos especificos.**

1. Revisar en la documentacion como funciona cada modelo y que necesita de los datos.
2. Probar, una por una, hipotesis de preparacion para cada modelo y anotar si mejoran o no.
3. Comparar la preparacion final de cada modelo con la preparacion de la EV1, contra la linea base y los seis criterios de exito.
4. Explicar que aprendio el modelo, donde se equivoca y que recomendaciones se desprenden.

**Alcance.** Solo cambia la preparacion de los datos. Los tres modelos usan sus parametros y su funcion de perdida por defecto; elegirlos es el ajuste de hiperparametros de la etapa siguiente.

**Hipotesis** (escritas antes de medir):

| # | Modelo | Hipotesis | Fuente | Veredicto |
|---|---|---|---|---|
| R1 | Ridge | El genero en one-hot rinde igual que codificado por objetivo | Documentacion | Confirmada |
| R2 | Ridge | Escalar las binarias mejora el resultado | Documentacion | No confirmada |
| R3 | Ridge | Quitar Yeo-Johnson empeora el resultado | Documentacion | Parcial |
| R4 | Ridge | Splines en las variables con forma de U invertida mejoran el resultado | Propia (EDA) | No confirmada |
| F1 | Random Forest | Quitar Yeo-Johnson y escalado no cambia el resultado | Documentacion | Confirmada |
| F2 | Random Forest | El genero en one-hot empeora el resultado | Documentacion | Refutada: mejora |
| F3 | Random Forest | Recuperar `instrumentalness` y `key` mejora el resultado | Propia (EV1) | No confirmada |
| X1 | XGBoost | Quitar Yeo-Johnson y escalado no cambia el resultado | Documentacion | Confirmada |
| X2 | XGBoost | El genero como categoria nativa mejora el resultado | Documentacion | Confirmada |
| X3 | XGBoost | Recuperar `instrumentalness` y `key` mejora el resultado | Propia (EV1) | No confirmada |

**Criterios de exito** (fijados en la EV1; no se cambian):

| Criterio | Umbral | Que controla |
|---|---|---|
| R² en prueba | ≥ 0,40 | Superar lo que explica el genero solo (cerca de 0,35) |
| RMSE en prueba | ≤ 15,51 (77% de la linea base) | Errores grandes |
| MAE en prueba | ≤ 13,46 (80% de la linea base) | Error tipico, en puntos |
| Brecha de R² entre entrenamiento y prueba | ≤ 0,05 | Sobreajuste |
| MAE en canciones con popularidad ≥ 70 | ≤ 20 | El segmento que mas le importa al negocio |
| Brecha de MAE entre cuartiles de genero | ≤ 5 | Equidad entre generos |

Los umbrales se expresan contra la linea base porque la popularidad no tiene una referencia externa de error aceptable. No se usa MAPE porque el 7,6% de las canciones tiene popularidad 0 y el MAPE divide por el valor real.

**Regla para cada hipotesis.** El ruido es la desviacion del MAE entre los 5 folds de validacion (0,05 a 0,09 puntos).

| Resultado del cambio | Veredicto | Decision |
|---|---|---|
| Baja el MAE en mas que el ruido | Mejora | Se adopta |
| Sube el MAE en mas que el ruido | Empeora | Se descarta |
| Cambia menos que el ruido | Sin efecto | Se mantiene la preparacion vigente, salvo que la documentacion diga que el paso que se quita no hace falta |

**Protocolo de seleccion** (fijado antes de entrenar):

| Paso | Regla |
|---|---|
| 1 | Una preparacion propia es **apta** solo si cumple los seis criterios en prueba |
| 2 | Entre las aptas gana la de menor MAE en **validacion**, no en prueba |
| 3 | Si otra queda a menos del ruido del lider, gana la mas interpretable |
| 4 | Los umbrales no se bajan. Si ninguna es apta, la ganadora queda como **mejor candidata** para la etapa siguiente |

## 3. Fuentes de datos

- **Dataset original:** Spotify Tracks Dataset (Kaggle, *maharshipandya*; licencia "Database: Open Database, Contents: (c) Original Authors"), construido con la Web API de Spotify. Son 114.000 filas y 21 columnas, con aprox. 1.000 canciones por cada uno de 114 generos.
- **Entrada de esta etapa:** los 9 archivos que exporta la EV1 en `data/processed/`, todos sobre la misma division agrupada por cancion: **68.704 canciones de entrenamiento y 17.155 de prueba**, sin canciones compartidas y con los mismos 113 generos. Para las hipotesis F3 y X3 se leen ademas `instrumentalness` y `key` del CSV original, por `track_id`.

| Archivo | Contenido | Uso |
|---|---|---|
| `X_*_base.csv` | 16 variables limpias, genero como texto, sin transformar | Punto de partida de toda preparacion |
| `X_*_procesado_ev1.csv` | Preparacion completa de la EV1 | Comprobar que la preparacion de la EV1 rearmada es identica |
| `y_*.csv` | `popularity` | Objetivo |
| `ids_*.csv` | `track_id`, artista, titulo, album y grupo de cancion | Grupos de la validacion cruzada y auditoria de sesgos |
| `metadatos_preprocesamiento.json` | Semilla, columnas y linea base | Configuracion |

*Glosario de las 26 variables: `../ev1/notebook_ev1.ipynb`, seccion 2.4, y notebook de esta etapa, seccion 2.3.*

## 4. Analisis exploratorio (EDA)

Resumen del EDA de la EV1:

- **Calidad:** se quitaron 450 duplicados exactos, 22.809 copias de canciones repetidas por genero, 157 filas con audio sin analizar, 868 de contenido casi solo hablado y 2.856 versiones con popularidad 0 artificial. La duracion no se filtro: sus valores extremos se concentran en generos como `detroit-techno`, `chicago-house` y `minimal-techno`.
- **Objetivo:** simetrico (con una fila por cancion: media 33,2, mediana 33, asimetria 0,07). Las canciones de 70 o mas son pocas: 657 en prueba.
- **Genero:** el predictor dominante. Explica cerca del 35% de la varianza (η² = 0,351), con medias de 2,2 (`iranian`) a 59,3 (`pop-film`) entre generos.
- **Audio:** relacion debil y no lineal (ninguna correlacion supera 0,13; forma de U invertida en `energy`, `danceability` y duracion). De aqui sale la hipotesis R4.
- **Forma y escala:** asimetria fuerte en entrenamiento en duracion (11,7), habla (2,8), vivo (2,1) y volumen (-1,9), y escalas muy distintas entre variables.
- **Colinealidad:** `energy`, `loudness` y `acousticness` llegan a 0,76 entre si, con VIF final ≤ 4,34. Por VIF se excluyo `instrumentalness`, y `key` por no relacionarse con la popularidad. De aqui salen F3 y X3.
- **Sesgos:** muestreo artificial por genero, efecto superestrella (el artista explica mas que el genero) y factores que no estan en los datos (promocion, redes sociales, seguidores y fama).

*Detalle: `../ev1/notebook_ev1.ipynb`, secciones 3, 4 y 5.*

## 5. Preparacion y transformacion de los datos

### 5.1 Punto de partida: la preparacion de la EV1

Limpieza, cinco variables construidas (`duracion_min`, `n_artistas`, `n_generos`, `es_instrumental`, `compas_4_4`), exclusion de `key` e `instrumentalness`, division agrupada por artista + titulo (80/20) y una preparacion unica: genero codificado por objetivo, Yeo-Johnson en 6 variables asimetricas y `StandardScaler` en las continuas. Rearmada dentro del notebook, es identica a los archivos de la EV1 (diferencia maxima de 4e-15).

### 5.2 Metodo: una hipotesis a la vez

Para cada modelo, desde la preparacion de la EV1: **hipotesis** (segun la documentacion o una idea propia), **prueba** (se cambia una sola cosa y se mide con validacion cruzada), **veredicto** (regla de la seccion 2) y **decision** (si mejora, la siguiente hipotesis parte de ahi). La prueba no participa en ninguna decision.

### 5.3 Ridge

La documentacion dice que los regularizadores de los modelos lineales suponen variables con "variance in the same order" $_{[7]}$, y que el one-hot es "better suited for low cardinality" $_{[10]}$. La preparacion de la EV1 ya estaba pensada para un modelo lineal.

| # | Prueba | Evidencia (MAE validacion, ruido 0,046) | Veredicto | Costo de no hacerlo |
|---|---|---|---|---|
| R1 | Genero en one-hot (113 columnas) | 11,467 a 11,451 (-0,016); mejora en 5 de 5 folds, pero bajo el ruido | Sin efecto: se mantiene la codificacion por objetivo (1 columna) | Ninguno |
| R2 | Escalar las 4 binarias | 11,467 a 11,467 (0,000) | Sin efecto: con `alpha = 1` y 68.704 filas el castigo es muy debil (razonamiento) | Ninguno con los parametros por defecto |
| R3 | Quitar Yeo-Johnson | 11,467 a 11,504 (+0,038); empeora en 5 de 5 folds | Sin efecto, en la direccion esperada: se mantiene Yeo-Johnson | 0,04 puntos |
| R4 | Splines en `energy`, `danceability` y duracion (U invertida del EDA) | 11,467 a 11,455 (-0,012); mejora en 5 de 5 folds | Sin efecto: la U existe, pero mueve pocos puntos | Ninguno |

![Ridge: recorrido de las hipotesis](images/5_1_recorrido_ridge.png)

*Ninguno de los cuatro cambios sale de la franja de ruido. Ridge termina con la preparacion de la EV1 y un MAE de validacion de 11,467.*

### 5.4 Random Forest

Los arboles requieren "little data preparation" y no necesitan normalizar $_{[13]}$: un corte "variable < umbral" no cambia si la transformacion conserva el orden (razonamiento).

| # | Prueba | Evidencia (MAE validacion) | Veredicto | Costo de no hacerlo |
|---|---|---|---|---|
| F1 | Quitar Yeo-Johnson y escalado | 10,916 a 10,917 (+0,001; ruido 0,054) | Sin efecto: se adopta la version simple, porque la documentacion dice que no hace falta | Ninguno en precision |
| F2 | Genero en one-hot en vez de por objetivo | 10,917 a 10,828 (-0,088; ruido 0,055); mejora en 5 de 5 folds | **Mejora: se adopta.** La hipotesis (que empeoraria) queda refutada | 0,09 puntos; a cambio, 128 columnas y varias veces mas tiempo de entrenamiento |
| F3 | Recuperar `instrumentalness` y `key` | 10,828 a 10,833 (+0,005; ruido 0,079) | Sin efecto: se descarta | Ninguno |

Posible explicacion de F2 (razonamiento, no verificada): la codificacion por objetivo resume cada genero en su popularidad media, y el arbol solo puede cortar por rangos de esa media. Con el one-hot, cada arbol puede separar un genero concreto, y Random Forest prueba todas las columnas en cada corte.

![Random Forest: recorrido de las hipotesis](images/5_2_recorrido_rf.png)

*Solo el one-hot del genero sale de la franja de ruido. Random Forest pasa de 10,916 a 10,828.*

### 5.5 XGBoost

Ademas de no necesitar transformaciones, XGBoost puede recibir el genero como categoria: en cada corte busca agrupar "the categories that output similar leaf values" $_{[3]}$.

| # | Prueba | Evidencia (MAE validacion) | Veredicto | Costo de no hacerlo |
|---|---|---|---|---|
| X1 | Quitar Yeo-Johnson y escalado | 11,281 a 11,299 (+0,018; ruido 0,071) | Sin efecto: se adopta la version simple | Ninguno en precision |
| X2 | Genero como categoria nativa | 11,299 a 10,655 (-0,644; ruido 0,058, 11 veces el ruido); mejora en 5 de 5 folds | **Mejora: se adopta.** Es el mayor efecto de todo el trabajo | 0,64 puntos: XGBoost quedaria detras de Random Forest |
| X3 | Recuperar `instrumentalness` y `key` | 10,655 a 10,646 (-0,009; ruido 0,085) | Sin efecto: se descarta | Ninguno |

![XGBoost: recorrido de las hipotesis](images/5_3_recorrido_xgb.png)

*La categoria nativa del genero explica todo el avance de XGBoost: de 11,281 a 10,655.*

**Balance de la bitacora.** De 10 cambios, 2 mejoran (F2 y X2, ambos sobre como se entrega el genero), 8 no tienen efecto y ninguno empeora mas que el ruido. Dos de los cambios sin efecto se adoptan por simplicidad (F1 y X1).

*Detalle tecnico: notebook_ev2.ipynb, secciones 4 y 5.*

## 6. Justificacion de los 3 modelos

Se eligieron tres formas distintas de aprender, para que la comparacion diga algo sobre los datos y no solo sobre un algoritmo.

| | **Ridge** | **Random Forest** | **XGBoost** |
|---|---|---|---|
| Familia | Lineal regularizado | Arboles en paralelo (bagging) | Arboles en secuencia (boosting) |
| Como aprende | Minimiza el error cuadratico mas un castigo a los coeficientes grandes $_{[6]}$ | Promedia arboles entrenados con muestras al azar de las filas $_{[5]}$ | Cada arbol corrige el error de los anteriores, con un termino que castiga arboles complejos $_{[8]}$ |
| Por que encaja | Es la referencia interpretable; el castigo hace los coeficientes "more robust to collinearity" $_{[2]}$, util con las variables de intensidad correlacionadas | Capta las relaciones no lineales del EDA sin programarlas | Acepta el genero como categoria $_{[3]}$ |
| Limitacion | Solo relaciones lineales en lo que recibe | Con `max_depth=None` los arboles crecen sin limite y tienden a memorizar $_{[5]}$ | Por defecto no esta ajustado a estos datos |

**Configuracion (sin cambios en todo el trabajo).** Ridge con `alpha = 1`. Random Forest con 100 arboles, sin limite de profundidad y todas las variables en cada corte. XGBoost con 100 arboles, `eta = 0,3`, `max_depth = 6` y perdida cuadratica; con 113 generos usa la particion de categorias, porque supera `max_cat_to_onehot = 4` $_{[4]}$. Semilla 42 en todos.

## 7. Variables finales por modelo

Las 15 variables que no son el genero son las mismas en los seis casos: 11 continuas (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `liveness`, `valence`, `tempo`, `n_generos`, `duracion_min`, `n_artistas`) y 4 binarias (`explicit`, `mode`, `es_instrumental`, `compas_4_4`). Cambia el formato.

| Grupo | Preparacion EV1 (los tres) | Ridge propio | Random Forest propio | XGBoost propio |
|---|---|---|---|---|
| Continuas (11) | Yeo-Johnson en 6 + estandarizadas | Igual a la EV1 | Sin transformar | Sin transformar |
| Binarias (4) | Sin tocar | Sin tocar | Sin tocar | Sin tocar |
| Genero (1) | 1 columna por objetivo, estandarizada | Igual a la EV1 | 113 columnas one-hot | 1 columna categorica |
| **Columnas** | **16** | **16** | **128** | **16** |

## 8. Implementacion

- **Pipelines.** Cada preparacion se encadena con su modelo en un `Pipeline` (`ColumnTransformer` + modelo). En cada fold, el escalado, Yeo-Johnson y la codificacion del genero se ajustan solo con las filas de entrenamiento de ese fold.
- **Validacion.** `GroupKFold` de 5 folds dentro del entrenamiento, con grupos = cancion: cada grupo aparece una sola vez en validacion $_{[1]}$, asi que las versiones de una misma cancion nunca quedan separadas.
- **Bitacora.** Una funcion prueba cada cambio contra la preparacion vigente, aplica la regla y anota el resultado. Se evaluaron 16 preparaciones en validacion cruzada.
- **Prueba.** Las seis configuraciones finales (cada modelo con la preparacion de la EV1 y con la suya) se evaluan juntas, una sola vez.
- **Control.** La preparacion de la EV1 rearmada da en prueba exactamente lo mismo que un modelo entrenado con los archivos de la EV1.

*Detalle tecnico: notebook_ev2.ipynb, secciones 3, 5.0 y 6.*

## 9. Evaluacion y seleccion del modelo

### 9.1 Tabla comparativa

| Modelo | Preparacion | MAE valid. | MAE test | RMSE test | R² test | R² train | MAE exitos |
|---|---|---|---|---|---|---|---|
| Linea base | - | - | 16,83 | 20,15 | 0,00 | - | 41,19 |
| Ridge | EV1 | 11,47 | 11,52 | 16,27 | 0,35 | 0,36 | 32,82 |
| Ridge | Propia (igual a la EV1) | 11,47 | 11,52 | 16,27 | 0,35 | 0,36 | 32,82 |
| Random Forest | EV1 | 10,92 | 10,99 | 15,86 | 0,38 | 0,84 | 30,89 |
| Random Forest | Propia | 10,83 | 10,85 | **15,76** | **0,39** | 0,92 | 29,95 |
| XGBoost | EV1 | 11,28 | 11,30 | 16,04 | 0,37 | 0,54 | 30,86 |
| **XGBoost** | **Propia** | **10,66** | **10,64** | 15,79 | **0,39** | 0,65 | **29,36** |

![Comparacion de MAE](images/6_1_comparacion_mae.png)

*Todos superan a la linea base y cumplen el umbral de MAE (13,46). Los dos modelos de arboles mejoran con su preparacion propia; Ridge queda igual.*

### 9.2 ¿Valio la pena adaptar?

| Modelo | Cambios adoptados | Cambio en MAE validacion | Cambio en MAE test | ¿Valio la pena? |
|---|---|---|---|---|
| Ridge | Ninguno | 0,000 | 0,000 | No hacia falta: la EV1 ya era adecuada |
| Random Forest | F1, F2 | -0,087 | -0,144 | Si, poco |
| XGBoost | X1, X2 | **-0,626** | **-0,656** | Si |

**El orden de los modelos cambia con la preparacion.** Con la de la EV1 gana Random Forest (10,99 en prueba; luego XGBoost con 11,30 y Ridge con 11,52). Con la propia gana XGBoost (10,64; luego Random Forest con 10,85 y Ridge con 11,52). Con una sola preparacion para todos, XGBoost se habria descartado por error.

### 9.3 Donde se equivoca

| Segmento (prueba) | Canciones | MAE linea base | MAE XGBoost propio |
|---|---|---|---|
| Popularidad 0 | 1.316 | 34,3 | 25,3 |
| Popularidad 1 a 69 | 15.182 | 14,3 | 8,6 |
| Popularidad ≥ 70 (exitos) | 657 | 41,2 | 29,4 |

**Error medido en puntos de popularidad.**

| | Linea base | XGBoost propio |
|---|---|---|
| Error mediano | 14,3 | **6,4** |
| Canciones con error ≤ 5 puntos | 14,7% | 41,8% |
| Canciones con error ≤ 10 puntos | 32,4% | 64,5% |
| Canciones con error ≤ 20 puntos | 64,9% | 83,8% |
| Canciones con error > 30 puntos | 17,0% | 7,8% |

El error medio (10,6) es mayor que el mediano porque unas pocas canciones, sobre todo los exitos, tienen errores grandes.

![Prediccion contra valor real](images/7_1_predicho_vs_real.png)

*Los modelos aplanan las predicciones hacia el centro. A canciones con popularidad real de 84 en promedio, XGBoost les predice 44; a las de popularidad 0, les predice 25. Los exitos se subestiman de forma sistematica (sesgo de -29,3 puntos en XGBoost).*

**Equidad.** La brecha de MAE entre cuartiles de genero es de 6,1 en la linea base y queda entre 3,8 y 4,4 en los seis modelos (4,0 en XGBoost propio). Todos cumplen.

### 9.4 Criterios de exito

| Criterio | Umbral | XGBoost propio | ¿Cumple? | Mejor de los seis |
|---|---|---|---|---|
| MAE en prueba | ≤ 13,46 | 10,64 | Si | 10,64 (los seis cumplen) |
| RMSE en prueba | ≤ 15,51 | 15,79 | No | 15,76 (ninguno cumple) |
| R² en prueba | ≥ 0,40 | 0,386 | No | 0,388 (ninguno cumple) |
| MAE en exitos | ≤ 20 | 29,36 | No | 29,36 (ninguno cumple) |
| Brecha de R² | ≤ 0,05 | 0,265 | No | 0,009 (solo Ridge cumple) |
| Brecha de MAE entre generos | ≤ 5 | 3,98 | Si | 3,79 (los seis cumplen) |

R² y RMSE son la misma exigencia vista de dos formas, por eso fallan juntos. **La preparacion propia aumenta el sobreajuste de los arboles:** la brecha de R² de Random Forest pasa de 0,459 a 0,533 y la de XGBoost de 0,178 a 0,265. Las variables mejor entregadas tambien le permiten al arbol memorizar mas; controlarlo corresponde al ajuste de hiperparametros.

### 9.5 Protocolo de seleccion aplicado

| Paso | Resultado |
|---|---|
| 1. ¿Alguna preparacion propia cumple los seis criterios? | Ninguna |
| 2. Menor MAE en validacion | XGBoost, 10,655 |
| 3. ¿Empate? (tolerancia: el ruido del lider = 0,085) | No: Random Forest queda a 0,174 y Ridge a 0,812 |
| 4. Veredicto | **Ningun modelo es apto. XGBoost con preparacion propia es el mejor candidato** para la etapa de ajuste |

La prueba confirma la eleccion: XGBoost tambien tiene el menor MAE en prueba. Sin este protocolo, el informe habria presentado como modelo final uno que no cumple cuatro de los seis criterios.

### 9.6 Escenario cancion inedita

`n_generos` (en cuantos generos aparecia la cancion en las busquedas) solo existe cuando la cancion ya esta publicada. Ese es justamente el caso de uso: decidir en que invertir antes del exito. Se repitio la preparacion final sin esa variable, con los mismos folds.

| | MAE validacion | MAE test | RMSE test | R² test | MAE exitos |
|---|---|---|---|---|---|
| XGBoost con `n_generos` | 10,655 | 10,64 | 15,79 | 0,386 | 29,36 |
| XGBoost sin `n_generos` (cancion inedita) | 10,903 | 10,84 | 15,93 | 0,374 | 29,31 |

**Veredicto.** Sin `n_generos` la candidata pierde 0,25 puntos en validacion (2,9 veces el ruido). Random Forest pierde algo parecido (0,27) y Ridge casi nada (0,006). Sin `n_generos`, XGBoost sigue siendo el mejor de los tres en validacion (10,903 contra 11,102 de Random Forest y 11,472 de Ridge). Para canciones aun no publicadas, la version sin `n_generos` es la que corresponde.

### 9.7 Interpretacion

![Importancia por permutacion](images/9_1_importancia_permutacion.png)

*La importancia por permutacion es cuanto empeora la metrica al desordenar una variable $_{[11]}$. Desordenar el genero empeora el MAE en 8,27 puntos: el 72% de toda la importancia. Le sigue `n_generos` (0,47), unas 18 veces menos.*

**Anatomia de una prediccion.** XGBoost entrega el aporte de cada variable con valores SHAP cuya suma es la prediccion $_{[12]}$. En promedio, el genero mueve cada prediccion en 9,4 puntos; ninguna otra variable pasa de 0,9.

![Anatomia de una prediccion](images/9_2_anatomia_prediccion.png)

*If the World Was Ending - Acoustic (Jonah Baker y Celine, genero `acoustic`) tiene popularidad 35 y el modelo predice 48,8. Desde un valor base de 34,3, el genero suma 9,2 y `energy` 2,8. Casi todo el exceso viene del genero: el modelo le asigna la popularidad tipica de las versiones acusticas, no la de esta cancion.*

**Lectura para el negocio (asociacion, no causa).** La popularidad se asocia mucho mas con el genero en que se lista una cancion que con como suena. Predecir el exito de una cancion concreta requiere datos que el audio no tiene: artista, promocion, momento del lanzamiento.

### 9.8 Auditoria de sesgos con el modelo

La EV1 mostro que el artista explica mas que el genero y dejo pendiente revisar el error en artistas con pocas canciones.

![Sesgo segun el catalogo del artista](images/10_1_sesgo_artista.png)

*La candidata se equivoca entre 9,87 puntos (artistas con mas de 20 canciones) y 11,30 (artistas con una sola cancion), sin una direccion fija de sesgo (-0,48 a +0,81). La brecha entre tramos es de 1,43, menos de la mitad que la de la linea base (2,99).*

- **Veredicto:** el modelo no castiga de forma visible a los artistas con pocas canciones. Pero tener una sola cancion en el dataset no equivale a ser emergente: esas canciones tienen la popularidad media mas alta (39,0), por la seleccion por busqueda descrita en la EV1.
- **Lo que queda fuera:** el efecto de promocion, seguidores y fama no se puede medir con estos datos. El modelo no lo hereda de forma visible, pero tampoco lo corrige.
- **Uso responsable:** la prediccion apoya decisiones; no debe usarse como filtro automatico para descartar artistas.

*Detalle tecnico: notebook_ev2.ipynb, secciones 6 a 10.*

## 10. Conclusiones y recomendaciones

**Respuesta a la pregunta.** Adaptar la preparacion **valio la pena en los dos modelos de arboles y no en Ridge**. En ambos arboles, lo que importa es como se entrega el genero. Para Ridge, la preparacion de la EV1 ya era la correcta.

| Decision | Efecto medido (MAE de validacion) | ¿Valio la pena? |
|---|---|---|
| Genero como categoria nativa en XGBoost (X2) | -0,644 | Si |
| Genero en one-hot en Random Forest (F2) | -0,088 | Si, aunque es pequeño y suma 112 columnas |
| Quitar Yeo-Johnson y escalado en los arboles (F1, X1) | +0,001 y +0,018 | Si, como simplificacion |
| Genero en one-hot en Ridge (R1) | -0,016 | No |
| Escalar binarias en Ridge (R2) | 0,000 | No, con el `alpha` por defecto |
| Quitar Yeo-Johnson en Ridge (R3) | +0,038 | No: se mantiene |
| Splines en Ridge (R4) | -0,012 | No |
| Recuperar `instrumentalness` y `key` (F3, X3) | +0,005 y -0,009 | No |
| Quitar `n_generos` (cancion inedita) | +0,248 en XGBoost | Costo pequeño y aceptable |

**Veredicto.** Ningun modelo es apto todavia. **XGBoost con el genero como categoria nativa es el mejor candidato**: MAE de 10,64 en prueba (37% menos que la linea base) y 0,17 puntos mejor que el siguiente en validacion. Cumple MAE y equidad; le faltan R², RMSE, exitos y sobreajuste.

**Recomendaciones.**

1. **Usarlo para ordenar y priorizar, no para decidir solo.** La mitad de las canciones queda a 6,4 puntos o menos, pero el error medio es de 10,6.
2. **No usarlo para detectar exitos:** los subestima en unos 29 puntos.
3. **Para canciones aun no publicadas, usar la version sin `n_generos`** (cuesta 0,25 puntos).
4. **Mostrar la anatomia de cada prediccion** y el error esperado de su genero, para que quien decide vea que la movio.
5. **Sumar datos de fama y promocion** (seguidores, fecha de lanzamiento, playlists) antes de exigir mas precision.

## 11. Limitaciones y proximos pasos

**Limitaciones.**

- Parametros y perdida por defecto: el orden entre modelos podria cambiar tras el ajuste (no verificado).
- El orden de las hipotesis importa: cada cambio se probo sobre la preparacion vigente. Otro orden podria cambiar los resultados pequeños; los dos cambios que mejoran son grandes frente al ruido.
- R1 y R4 mejoran en los 5 folds, pero menos que el ruido: pueden ser efectos reales, aunque con la regla fijada no alcanzan para adoptarse.
- Un solo conjunto de prueba (657 exitos); no se estimo su ruido.
- El EDA de la EV1 se hizo con todo el dataset, antes de dividir: es una fuga leve que ya no se puede evitar.
- La muestra tiene 1.000 canciones por genero por construccion y no representa el catalogo real. Las relaciones son asociaciones, no causas.

**Proximos pasos (etapa siguiente).**

1. **Ajuste de hiperparametros** con la misma validacion agrupada, empezando por la candidata: `eta`, `max_depth`, `subsample`, `colsample_bytree` y `lambda` en XGBoost; `max_depth`, `min_samples_leaf` y `max_features` en Random Forest; `alpha` en Ridge. El objetivo principal es bajar el sobreajuste de los arboles.
2. **Elegir la funcion de perdida** como parte de ese ajuste (por ejemplo, la perdida absoluta optimiza directamente el MAE).
3. **Revisar R1 y R2 al subir `alpha`:** con un castigo mayor, la escala de las binarias y del one-hot puede empezar a importar (razonamiento).
4. Estimar el ruido de la prueba con bootstrap por cancion e incorporar datos de fama y promocion.

## 12. Reproducibilidad

1. Instalar las librerias con las versiones usadas: pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1, matplotlib 3.11.2, seaborn 0.13.2, jupyter 1.1.1 y nbconvert 7.17.2 (el comando esta en el `README.md`).
2. Ejecutar antes `ev1/notebook_ev1.ipynb`, que deja en `data/processed/` los 9 archivos de entrada.
3. Ejecutar `ev2/notebook_ev2.ipynb`, desde `ev2/` o desde la raiz (sin rutas absolutas). Tarda unos 12 minutos (depende del equipo) y guarda las 8 figuras en `ev2/images/`; no exporta otros archivos.

Semilla 42 en todos los componentes aleatorios. Al reejecutar se obtienen las mismas cifras, salvo los segundos de ejecucion.

## 13. Referencias

Documentacion oficial de cada libreria; las citas textuales de scikit-learn se verificaron en la version 1.9.1.

1. scikit-learn. *GroupKFold*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
2. scikit-learn. *User Guide: Linear Models, Ridge regression and classification*. https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
3. XGBoost. *Categorical Data*. https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
4. XGBoost. *XGBoost Parameters*. https://xgboost.readthedocs.io/en/stable/parameter.html
5. scikit-learn. *RandomForestRegressor*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
6. scikit-learn. *Ridge*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
7. scikit-learn. *User Guide: Preprocessing data*. https://scikit-learn.org/stable/modules/preprocessing.html
8. XGBoost. *Introduction to Boosted Trees*. https://xgboost.readthedocs.io/en/stable/tutorials/model.html
9. scikit-learn. *PowerTransformer*. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
10. scikit-learn. *TargetEncoder*. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
11. scikit-learn. *permutation_importance*. https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
12. XGBoost. *Python API Reference: Booster.predict (`pred_contribs`)*. https://xgboost.readthedocs.io/en/stable/python/python_api.html
13. scikit-learn. *User Guide: Decision Trees*. https://scikit-learn.org/stable/modules/tree.html
14. scikit-learn. *SplineTransformer*. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html
