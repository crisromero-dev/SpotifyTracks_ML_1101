# Prediccion de la popularidad de canciones de Spotify

Proyecto de Machine Learning de la asignatura **MLY1101 Machine Learning** (DuocUC), caso C: inteligencia musical y prediccion de popularidad de canciones (Spotify Tracks).

**Equipo:** Grupo 5: Cristian Romero y Rolando Paredes.

**Problema.** Estimar la popularidad de una cancion en Spotify (indice de 0 a 100) a partir de sus caracteristicas de audio, su genero y metadatos basicos, como apoyo a las decisiones de promocion de un sello o una plataforma. Es un problema de regresion supervisada y se desarrolla con la metodologia CRISP-DM.

## Evaluaciones

| Evaluacion | Contenido | Informe | Notebook |
|---|---|---|---|
| **EV1** | Problema de negocio, KPIs, calidad y limpieza de datos, analisis exploratorio, preparacion y sesgos (CRISP-DM 1 a 3) | Pendiente | [ev1/notebook_ev1.ipynb](ev1/notebook_ev1.ipynb) |
| **EV2** | Modelado y evaluacion de Ridge, Random Forest y XGBoost, con la preparacion de datos de cada modelo probada hipotesis por hipotesis (CRISP-DM 4 y 5) | [ev2/Informe_ev2.md](ev2/Informe_ev2.md) | [ev2/notebook_ev2.ipynb](ev2/notebook_ev2.ipynb) |

**Resultado principal.** Con los modelos en sus parametros por defecto, preparar los datos segun cada modelo mejora a los dos de arboles y no a Ridge: lo que importa es como se entrega el genero. Ningun modelo cumple todavia los seis criterios de exito. El mejor candidato es XGBoost con el genero como categoria nativa (MAE de 10,64 en prueba, 37% menos que la linea base), que cumple MAE y equidad entre generos y queda pendiente para la etapa de ajuste de hiperparametros.

## Estructura del proyecto

```
SpotifyTracks/
|-- README.md                       <- esta portada
|-- data/
|   |-- raw/
|   |   `-- Spotify_Tracks_Dataset.csv   <- dataset original (Kaggle), sin modificar
|   `-- processed/                  <- salidas del notebook de la EV1 = entradas del notebook de la EV2
|                                      (X_train/X_test base y procesado, y, ids y metadatos)
|-- ev1/
|   `-- notebook_ev1.ipynb          <- notebook de la EV1
|-- ev2/
|   |-- Informe_ev2.md              <- informe de la EV2 (documento principal)
|   |-- notebook_ev2.ipynb          <- notebook de la EV2 (respaldo tecnico)
|   `-- images/                     <- 8 figuras que usa Informe_ev2.md
`-- docs/
    |-- pauta_ev1.pdf               <- instrucciones y rubrica de la EV1
    `-- pauta_ev2.pdf               <- instrucciones y rubrica de la EV2
```

## Como reproducir

1. **Instalar las librerias** (Python 3.12 o superior) con las versiones con que se ejecutaron los notebooks:

   ```bash
   pip install pandas==3.0.5 numpy==2.5.3 scikit-learn==1.9.1 xgboost==3.4.1 matplotlib==3.11.2 seaborn==0.13.2 jupyter==1.1.1 nbconvert==7.17.2
   ```

2. **Poner el dataset** en `data/raw/Spotify_Tracks_Dataset.csv` (ya incluido en el proyecto; fuente: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).

3. **Ejecutar la EV1**, que genera `data/processed/` (aprox. 1 minuto):

   ```bash
   python -m nbconvert --to notebook --execute --inplace ev1/notebook_ev1.ipynb
   ```

4. **Ejecutar la EV2**, que lee `data/processed/` (y `instrumentalness` y `key` del CSV original) y genera las figuras de `ev2/images/` (unos 12 minutos, segun el equipo):

   ```bash
   python -m nbconvert --to notebook --execute --inplace ev2/notebook_ev2.ipynb
   ```

5. Los notebooks tambien se pueden abrir y ejecutar celda por celda en Jupyter. Funcionan desde su propia carpeta o desde la raiz: calculan las rutas a partir de la carpeta que contiene `data/raw/`, sin rutas absolutas.

Conviene usar `python -m nbconvert` con el mismo Python donde se instalaron las librerias: asi el notebook se ejecuta con esas versiones y no con otra instalacion de Python del equipo. Todos los pasos aleatorios usan la semilla 42; al reejecutar se obtienen las mismas cifras, salvo los segundos de ejecucion.
