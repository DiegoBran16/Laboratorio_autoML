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
3. <u>Models</u>:

