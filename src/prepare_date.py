import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import yaml

# Cargar parámetros
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Cargar datos
data = pd.read_csv(params["general"]["dataset_path"])
X = data.drop(params["general"]["target_column"], axis=1)
y = data[params["general"]["target_column"]]

# Configuración de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler() if params["preprocess"]["scaling"] == "StandardScaler" else MinMaxScaler(), params["preprocess"]["numerical"]),
        ("cat", OneHotEncoder() if params["preprocess"]["encoding"] == "OneHotEncoder" else LabelEncoder(), params["preprocess"]["categorical"])
    ])
X = preprocessor.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["general"]["split_ratio"], random_state=params["general"]["random_state"])

# Guardar conjuntos
pd.DataFrame(X_train).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)
