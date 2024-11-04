import optuna
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import yaml
from catboost import CatBoostRegressor


# Cargar los datos de train y test
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Cargar los parámetros
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


def objective(trial):
    model_type = trial.suggest_categorical("model", ["RandomForest", "GradientBoosting", "CatBoost","LinearRegression"])

    if model_type == "RandomForest":
        rf_params = params["hyperparameters"]["RandomForest"]
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", rf_params["n_estimators"]["min"], rf_params["n_estimators"]["max"], step=rf_params["n_estimators"]["step"]),
            max_depth=trial.suggest_int("max_depth", rf_params["max_depth"]["min"], rf_params["max_depth"]["max"], step=rf_params["max_depth"]["step"]),
            n_jobs=1
        )
    elif model_type == "GradientBoosting":
        gb_params = params["hyperparameters"]["GradientBoosting"]
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", gb_params["n_estimators"]["min"], gb_params["n_estimators"]["max"], step=gb_params["n_estimators"]["step"]),
            learning_rate=trial.suggest_float("learning_rate", gb_params["learning_rate"]["min"], gb_params["learning_rate"]["max"], log=gb_params["learning_rate"].get("log", False)),
            max_depth=trial.suggest_int("max_depth", gb_params["max_depth"]["min"], gb_params["max_depth"]["max"], step=gb_params["max_depth"]["step"])
        )
    elif model_type == "CatBoost":
        cb_params = params["hyperparameters"]["CatBoost"]
        model = CatBoostRegressor(
            iterations=trial.suggest_int("iterations", cb_params["iterations"]["min"], cb_params["iterations"]["max"], step=cb_params["iterations"]["step"]),
            learning_rate=trial.suggest_float("learning_rate", cb_params["learning_rate"]["min"], cb_params["learning_rate"]["max"], log=cb_params["learning_rate"].get("log", False)),
            depth=trial.suggest_int("depth", cb_params["depth"]["min"], cb_params["depth"]["max"], step=cb_params["depth"]["step"]),
            verbose=0
        )
    elif model_type == "LinearRegression":
        lr_params = params["hyperparameters"]["LinearRegression"]
        model = Ridge(
            fit_intercept=trial.suggest_categorical("fit_intercept", lr_params["fit_intercept"]),
            alpha=trial.suggest_float("alpha", lr_params["alpha"]["min"], lr_params["alpha"]["max"], step=lr_params["alpha"]["step"])
        )



    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Crear el estudio de optuna y realizar la optimización
study = optuna.create_study(direction=params["optuna"]["direction"])
study.optimize(objective, n_trials=params["optuna"]["n_trials"])

# Obtener los mejores hiperparámetros
best_params = study.best_trial.params
if best_params["model"] == "RandomForest":
    best_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        n_jobs=1
    )
elif best_params["model"] == "GradientBoosting":
    best_model = GradientBoostingRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"]
    )
elif best_params["model"] == "CatBoost":
    best_model = CatBoostRegressor(
        iterations=best_params["iterations"],
        learning_rate=best_params["learning_rate"],
        depth=best_params["depth"],
        verbose=0
    )
elif best_params["model"] == "LinearRegression":
    best_model = Ridge(
        fit_intercept=best_params["fit_intercept"],
        alpha=best_params["alpha"]
    )

# Entrenar el modelo con los mejores hiperparámetros
best_model.fit(X_train, y_train)

#guardar el modelo completo
joblib.dump(best_model, "models/optuna/best_model_optuna.pkl")

# Guardar los resultados y los hiperparámetros en metrics/optuna_results.csv
optuna_results = {
    "Model": best_params["model"],
    "MSE": mean_squared_error(y_test, best_model.predict(X_test))
}

for param, value in best_params.items():
    optuna_results[param] = value

# guardar los resultados del modelo de optuna
optuna_results_df = pd.DataFrame([optuna_results])
optuna_results_df.to_csv("metrics/optuna_results.csv", index=False)

# Guardar el estudio de Optuna en metrics/optuna_study.pkl
joblib.dump(study, "metrics/optuna_study.pkl")

