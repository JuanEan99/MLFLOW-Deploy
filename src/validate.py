import os
import sys
import traceback
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from joblib import dump  # para guardar model.pkl en la raíz

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# =========================
# 1) Rutas y tracking local
# =========================
# Raíz del proyecto: /.../mlflow-deploy
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# URI correcta para cualquier OS (file:///C:/... en Windows)
TRACKING_URI = MLRUNS_DIR.as_uri()

print(f"--- Debug: Project Root: {PROJECT_ROOT} ---")
print(f"--- Debug: MLRuns Dir:  {MLRUNS_DIR} ---")
print(f"--- Debug: Tracking URI: {TRACKING_URI} ---")

mlflow.set_tracking_uri(TRACKING_URI)

# =========================
# 2) Experimento
# =========================
EXPERIMENT_NAME = "CI-CD-Lab2"

try:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)  # sin artifact_location
        print(f"--- Debug: Experimento creado '{EXPERIMENT_NAME}' -> ID: {experiment_id} ---")
    else:
        experiment_id = exp.experiment_id
        print(f"--- Debug: Experimento existente '{EXPERIMENT_NAME}' -> ID: {experiment_id} ---")
except Exception as e:
    print(f"--- ERROR creando/obteniendo experimento: {e} ---")
    raise

# =========================
# 3) Datos y modelo
# =========================
X, y = load_diabetes(return_X_y=True, as_frame=True)  # DataFrame para firmar mejor el modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
signature = infer_signature(X_test, preds)

# =========================
# 4) Run y logging
# =========================
print(f"--- Debug: Iniciando run en experimento ID: {experiment_id} ---")
run = None
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: Artifact URI del run: {run.info.artifact_uri} ---")

        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5]
        )

        # ===== Guardar también en la raíz para validate.py =====
        MODEL_OUT = PROJECT_ROOT / "model.pkl"
        dump(model, MODEL_OUT)
        print(f"💾 Modelo guardado en {MODEL_OUT}")
        # =======================================================

        print(f"✅ Modelo registrado. MSE: {mse:.4f}")

except Exception as e:
    print("\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print("--- Fin de la Traza de Error ---")
    print(f"CWD actual: {os.getcwd()}")
    try:
        print(f"Tracking URI efectiva: {mlflow.get_tracking_uri()}")
    except Exception:
        pass
    print(f"Experiment ID: {experiment_id}")
    if run is not None:
        print(f"Artifact URI del run (si existe): {run.info.artifact_uri}")
    sys.exit(1)
