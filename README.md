# Laboratorio01 AutoML 
### Curso Product Development | Universidad Galileo | Postgrado en Data Science | Cuarto trimestre
**Elbaorado por**: Diego Fernando Bran  **Carnet:** 24003821

## Documentación para la elaboración y reporducción del presente laboratorio

### Estructura del Laboratorio
La estructura que presenta el laboratorio es la siguiente:
```
laboratorio_autoML/
├── data/
│   └── data.csv                     # Dataset de entrada
├── src/
│   ├── prepare_data.py              # Script para la preparación de datos (exploración y preprocesamiento)
│   ├── evaluate_model.py            # Script para la evaluación de modelos y obtención de métricas
│   ├── train_model_optuna.py        # Script de entrenamiento con optimización de Optuna
├── models/                          # Directorio para guardar modelos dentro de el se segmentarn por entrenamientos y optimizaciones
│   ├── optuna/                      # Directorio para almacenar el modelo optimizado de optuna
|       |── best_model_optuna.pkl    # Archivo de modelo optimizado con Optuna
├── metrics/                         # Directorio para guardar métricas y resultados
│   ├── results.csv                  # Archivo con resultados de métricas de los modelos
│   |── optuna_study.pkl             # Archivo con el estudio completo de Optuna
|   └── optuna_results.csv           # Archivo con el resultado de los mejores hiperparametros de la optimización de optuna
├── params.yaml                      # Archivo de configuración de parámetros adaptable a cualquier dataset
├── dvc.yaml                         # Archivo de configuración del pipeline en DVC
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Instrucciones para ejecutar el proyecto
        # Documentación e instrucciones para reproducir el laboratorio
```

### Paso a paso de la elaboración del proyecto: 
#### 1. Configuración de las dependencias del entorno 
Para esto se define un archivo denominado *requirements.txt* este contiene las dependencias necesarias a instalar en el ambiente, en este caso se utilizazon *pandas* ,*scikit-learn*, *dvc*, *optuna*, *numpy*, *pyyaml* y *joblib*

**Explicación de Cada Dependencia:**
- **dvc:** Para la gestión del pipeline y control de versiones de datos.
- **optuna:** Para la optimización de hiperparámetros de modelos.
- **pandas, numpy:** Para la manipulación y preparación de datos.
- **scikit-learn:** Para la implementación de modelos de machine learning y preprocesamiento.
- **joblib:** Para guardar y cargar modelos.
- **pyyaml:** Para leer y procesar el archivo params.yaml.

#### 2. Definir los parametros en el archivo params.yaml
Este archivo contiene las configuraciones del pipeline de forma centralizada, es de gran utilidad puesto que permite que DVC y los scripts accedan a los parametros sin necesidad de modificar el codigo
1. **Configuración General (general)**:
   - Incluye parámetros básicos, como la ruta del dataset (dataset_path), el nombre de la columna objetivo (target_column), el tipo de problema (problem_type), el ratio de división para datos de prueba (split_ratio), y una semilla para reproducibilidad (random_state).
2. **Preprocesamiento de Datos (preprocess)**:
   - Define las columnas numéricas y categóricas en el dataset (numerical, categorical).
   - Especifica el tipo de escalado (scaling) y de codificación (encoding) para las variables categóricas, permitiendo elegir entre StandardScaler, MinMaxScaler, OneHotEncoder, etc. 
3. **Optimización de Hiperparámetros con Optuna (optuna)**:
   - Controla los parámetros de Optuna para la optimización, incluyendo el número de iteraciones (n_trials), el objetivo de la optimización (direction, ya sea minimize o maximize), y el nombre del estudio (study_name).
4. **Hiperparámetros de Modelos (hyperparameters)**:
   - Define los hiperparámetros de cada modelo, permitiendo personalización en los valores que Optuna puede explorar:
     - **RandomForest**: Parámetros de n_estimators y max_depth.
     - **GradientBoosting**: Parámetros de n_estimators, learning_rate, y max_depth.
     - **CatBoost**: Parámetros de iterations, learning_rate, y depth.
     - **LinearRegression**: Controla fit_intercept (si se debe ajustar el intercepto) y alpha para la regularización (usando Ridge si es necesario).
#### 3 Implementación de los scripts
1. **Script de Preparación de Datos(src/prepare_data.py):** Este script lee el archivo params.yaml para identificar las columnas numéricas y categóricas. Aplicas el escalador y codificador definidos en el archivo antes mencionado y divide el dataset en entrenamiento y prueba, posteroirmente guarda estos conjuntos en archivos csv.
2. **Script de Optimización de optuna (srv/train_model_optuna.py):** Este scipr es el encargadod e configurar y ejecutar optuna para la búsqueda de hiperparámetros óptimos segun el la sección *hyperparameters* definido en el archivo *params.yaml*. En este achivo también se entrena el modelo con los mejores hipeparametros encontrados en el estudio de optuna
3. **Script de Evaluación del modelo (src/evaluate_model.py):** Este script evalua el rednimiento del modelo entrenado según el tipo de problema, calculando las métricas relevantes y guarda estos resultados en un archivo csv (metrics/results.csv).
#### 4. Definir el pipeline de DVC
En este archvio se define el pipeline en DVC con tres etapas principales: prepare_data, train_model_optuna, y evaluate_model. Cada etapa especifica el comando a ejecutar (cmd), las dependencias necesarias (deps) y las salidas generadas (outs).
**Descripción de las Etapas**
1. prepare_data:
   - Comando (cmd): Ejecuta python src/prepare_data.py para preparar los datos.
   - Dependencias (deps):
     - src/prepare_data.py: El script de Python para el preprocesamiento. 
     - data/data.csv: El archivo de datos original. 
     - params.yaml: Archivo de parámetros de configuración.
   - Salidas (outs):
   - data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv: Archivos generados para dividir los datos en conjuntos de entrenamiento y prueba.
2. train_model_optuna:
   - Comando (cmd): Ejecuta python src/train_model_optuna.py para entrenar el modelo con Optuna.
   - Dependencias (deps):
     - src/train_model_optuna.py: El script de Python para el entrenamiento y la optimización. 
     - data/X_train.csv, data/y_train.csv, data/X_test.csv, data/y_test.csv: Datos de entrenamiento y prueba. 
     - params.yaml: Archivo de parámetros de configuración.
   - Salidas (outs):
     - models/optuna/best_model_optuna.pkl: El modelo entrenado con los mejores hiperparámetros.
     - metrics/optuna_results.csv: Resultados de la optimización de Optuna.
     - metrics/optuna_study.pkl: Objeto de estudio de Optuna para almacenar el progreso de la optimización.
3. evaluate_model:
   - Comando (cmd): Ejecuta python src/evaluate_model.py para evaluar el modelo entrenado.
   - Dependencias (deps):
     - src/evaluate_model.py: Script para evaluar el modelo.
     - data/X_test.csv, data/y_test.csv: Datos de prueba.
     - models/: Directorio de modelos entrenados.
   - Salidas (outs):
     - metrics/results.csv: Archivo con las métricas de evaluación del modelo.

#### 5. Ejecutar el Pipeline 
Para ejecutar el pipeline con DVC se deben seguir los siguientes pasos: 
1. **Inicializar DVC en el Proyecto:** Esta acción solo se realiza la primera vez. Su función es inicializar DVC en el laboratorio.
```
dvc init
```
2. **Agregar el dataset a DVC:** El dataset debe ser rastreado por DVC para facilitar la gestión de versiones y reproducibilidad 
```
dvc add data/data.csv
git add data/.gitignore data/data.csv.dvc
```
3. **Rastrear el Pipeline en DVC y el archivo de parametros:** 
```
git add dvc.yaml params.yaml
```
4. **Ejecutar el Pipeline Completo:** Para ejecutar el pupeline completo y realizar cada etapa (preparación de datos, entrenamineto, evaluación y optimización) se debe ejecutar el siguiente comando. Esto ejecutará automáticamente cada paso del pipeline en el oden establecido en el archivo *dvc.yaml*
```
dvc repro
```
DVC se encargará de ejecutar el pipeline en el siguiente orden:
- prepare_data.py 
- train_model_optuna.py
- evaluate_model.py

5. **Rastrear los datos con git:** Posterior a ejecutar el pipeline, rastrea los archivos fenerados y cialquier cambio en el repositorio de Git para mantener la trazabilidad del laboratorio. 

```
git add models/ metrics/ data/X_train.csv data/X_test.csv data/y_train.csv data/y_test.csv
git commit -m "Ejecutar pipeline completo con DVC y optimización con Optuna"

```

6. **Verificar Resultados:** Los resultados de las métricas y el mejor modelo optimizado se guardarán en:

- **metrics/results.csv:** Contiene las métricas de rendimiento del mejor modelo entrenado con la optimización de optuna
- **metrics/optuna_study.pkl:** Almacena el estudio completo de Optuna, incluyendo el historial de optimización.
- **models/best_model_optuna.pkl:** El mejor modelo optimizado por Optuna.
- *metrics/optuna_results.csv:** Almacena los datos del modelo ganador optimizado por optuna

