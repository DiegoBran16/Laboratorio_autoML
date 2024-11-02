# Laboratorio01 AutoML 
### Curso Product Development | Universidad Galileo | Postgrado en Data Science | Cuarto trimestre
**Elbaorado por**: Diego Fernando Bran  **Carnet:** 24003821

## Documentación para la elaboración y reporducción del presente laboratorio

### Estructura del Laboratorio
La estructura que presenta el laboratorio es la siguiente:
```
laboratorio_autoML/
├── data/
│   └── data.csv                # Dataset de entrada
├── src/
│   ├── prepare_data.py         # Script para la preparación de datos (exploración y preprocesamiento)
│   ├── train_model.py          # Script para el entrenamiento de modelos
│   ├── evaluate_model.py       # Script para la evaluación de modelos y obtención de métricas
│   ├── train_model_optuna.py   # Script de entrenamiento con optimización de Optuna
├── models/                     # Directorio para guardar modelos entrenados
│   ├── best_model_optuna.pkl   # Archivo de modelo optimizado con Optuna
├── metrics/                    # Directorio para guardar métricas y resultados
│   ├── results.csv             # Archivo con resultados de métricas de los modelos
│   └── optuna_study.pkl        # Archivo con el estudio completo de Optuna
├── params.yaml                 # Archivo de configuración de parámetros adaptable a cualquier dataset
├── dvc.yaml                    # Archivo de configuración del pipeline en DVC
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Instrucciones para ejecutar el proyecto
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

El archivo contiene 4 secciones principales las cuales contienen la siguiente estructura: 
1. <u>General</u>: en esta sección se define la configuración general como la ruta al daset, la columna objetivo, el tipo de problma y la distribución de la división entre train y test. 
2. <u>Preprocess</u>: en esta sección se define el la dinamica del preprocesamiento, se enlistan columnas numericas y categoricas y el tipo de trnsformación que se realizara sobre estas.
3. <u>Models</u>: en esta sección se definen los hiperparametros ajustables de cada uno de los modelos definidos
4. <u>Optuna</u>: en esta sección se definen los hiperparametros de la optimización de optuna

#### 3 Implementación de los scripts 

1. **Script de Preparación de Datos(src/prepare_data.py):** Este script lee el archivo params.yaml para identificar las columnas numéricas y categóricas. Aplicas el escalador y codificador definidos en el archivo antes mencionado y divide el dataset en entrenamiento y prueba, posteroirmente guarda estos conjuntos en archivos csv.
2. **Scipt de Entrenamiento(src/train_model.py):** Este script es el enargado de seleccionar los modelos segun el prametro *problem_type* definido en el archivo *params.yaml*. Adicionalmente se encarga de ajustar los hiperparámetros definidos en params.yaml. 
3. **Script de Evaluación del modelo (src/evaluate_model.py):** Este script evalua el rednimiento del modelo entrenado según el tipo de problema, calculando las métricas relevantes y guarda estos resultados en un archivo csv (metrics/results.csv).
4. **Script de Optimización de optuna (srv/train_model_optuna.py):** Este scipr es el encargadod e configurar y ejecutar optuna para la búsqueda de hiperparámetros óptimos. 
#### 4. Definir el pipeline de DVC


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
- train_model.py 
- evaluate_model.py 
- train_model_optuna.py

5. **Rastrear los datos con git:** Posterior a ejecutar el pipeline, rastrea los archivos fenerados y cialquier cambio en el repositorio de Git para mantener la trazabilidad del laboratorio. 

```
git add models/ metrics/ data/X_train.csv data/X_test.csv data/y_train.csv data/y_test.csv
git commit -m "Ejecutar pipeline completo con DVC y optimización con Optuna"

```

6. **Verificar Resultados:**Los resultados de las métricas y el mejor modelo optimizado se guardarán en:

- **metrics/results.csv:** Contiene las métricas de rendimiento de los modelos entrenados.
- **metrics/optuna_study.pkl:** Almacena el estudio completo de Optuna, incluyendo el historial de optimización.
- **models/best_model_optuna.pkl:** El mejor modelo optimizado por Optuna.

