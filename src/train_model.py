import yaml
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier

# Cargar parámetros y datos
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()

# Selección de modelos según el tipo de problema
models = {}
if params["general"]["problem_type"] == "regression":
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor()
    }
    model_params = params["models"]["regression"]
else:
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier()
    }
    model_params = params["models"]["classification"]

# Entrenar y guardar modelos
for model_name, model in models.items():
    if model_name in model_params:
        grid_search = GridSearchCV(model, model_params[model_name], cv=5)
        grid_search.fit(X_train, y_train)
        joblib.dump(grid_search.best_estimator_, f"models/{model_name}.pkl")
