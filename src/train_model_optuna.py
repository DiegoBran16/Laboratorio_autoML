import optuna
import joblib
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Cargar parámetros
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Cargar datos
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

def objective(trial):
    model_type = trial.suggest_categorical("model", list(params["models"][params["general"]["problem_type"]].keys()))

    if params["general"]["problem_type"] == "regression":
        if model_type == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                max_depth=trial.suggest_int("max_depth", 5, 20)
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            )
    else:
        # Clasificación
        model = RandomForestClassifier()  # Ejemplo simplificado

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = mean_squared_error(y_test, predictions) if params["general"]["problem_type"] == "regression" else accuracy_score(y_test, predictions)

    return score

study = optuna.create_study(direction=params["optuna"]["direction"])
study.optimize(objective, n_trials=params["optuna"]["n_trials"])
best_trial = study.best_trial
joblib.dump(study, "metrics/optuna_study.pkl")
