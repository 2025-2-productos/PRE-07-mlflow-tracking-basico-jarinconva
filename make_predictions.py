"""
Prediccion script for the MLflow model.

This script loads a model from MLflow and makes predictions on a dataset.

$ python3 make_predictions.py


"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"


df = pd.read_csv(FILE_PATH)
y = df["quality"]
X = df.drop(columns=["quality"])

## Debe verificarse el run_id del modelo que se quiere cargar
##Se puede obtener el run_id desde la interfaz de MLflow

looged_model = "runs:/2cddd5aea3d54ab2a0150ac11be4f6e8/model"
loaded_model = mlflow.pyfunc.load_model(looged_model)
y = loaded_model.predict(X)

print(y)
