import pandas as pd
import joblib
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Cargar los parámetros
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Cargar los datos de prueba
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Inicializar un diccionario para almacenar resultados de métricas
metrics_results = {}

# Seleccionar y cargar el modelo óptimo
if params["general"]["problem_type"] == "regression":
    model = joblib.load("models/best_model_optuna.pkl")

    # Realizar predicciones
    predictions = model.predict(X_test)

    # Calcular métricas para problemas de regresión
    metrics_results["MSE"] = mean_squared_error(y_test, predictions)
    metrics_results["MAE"] = mean_absolute_error(y_test, predictions)
    metrics_results["R2_Score"] = r2_score(y_test, predictions)

else:
    model = joblib.load("models/best_model_optuna.pkl")

    # Realizar predicciones
    predictions = model.predict(X_test)

    # Calcular métricas para problemas de clasificación
    metrics_results["Accuracy"] = accuracy_score(y_test, predictions)
    metrics_results["F1_Score"] = f1_score(y_test, predictions, average="weighted")
    metrics_results["Precision"] = precision_score(y_test, predictions, average="weighted")
    metrics_results["Recall"] = recall_score(y_test, predictions, average="weighted")

# Guardar los resultados de las métricas en un archivo CSV
metrics_df = pd.DataFrame([metrics_results])
metrics_df.to_csv("metrics/results.csv", index=False)
