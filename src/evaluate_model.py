import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml

# Cargar parámetros y datos
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Cargar el mejor modelo de Optuna
model = joblib.load("models/optuna/best_model_optuna.pkl")

# Calcular métricas de evaluación
metrics_results = {}
if params["general"]["problem_type"] == "regression":
    predictions = model.predict(X_test)
    metrics_results["Evaluation_MSE"] = mean_squared_error(y_test, predictions)
    metrics_results["Evaluation_MAE"] = mean_absolute_error(y_test, predictions)
    metrics_results["Evaluation_R2"] = r2_score(y_test, predictions)
elif params["general"]["problem_type"] == "classification":
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    predictions = model.predict(X_test)
    metrics_results["Accuracy"] = accuracy_score(y_test, predictions)
    metrics_results["F1_Score"] = f1_score(y_test, predictions, average="weighted")
    metrics_results["Precision"] = precision_score(y_test, predictions, average="weighted")
    metrics_results["Recall"] = recall_score(y_test, predictions, average="weighted")

# Leer los hiperparámetros de Optuna
optuna_results = pd.read_csv("metrics/optuna_results.csv")
model_name = optuna_results["Model"].iloc[0] if "Model" in optuna_results.columns else "Unknown"
optuna_results = optuna_results.drop(columns=["Model"], errors="ignore")

# Añadir el nombre del modelo y las métricas de evaluación en un solo registro
combined_results = {"Model": model_name}
combined_results.update(metrics_results)
combined_results.update(optuna_results.to_dict(orient="records")[0])

# Guardar el archivo final en metrics/results.csv
final_results_df = pd.DataFrame([combined_results])
final_results_df.to_csv("metrics/results.csv", index=False)
