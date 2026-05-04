"""
Task 3 — Model Versioning
Registers the best model in the MLflow Model Registry.
Saves results/step3_s6.json.
"""

import os
import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
META_PATH   = os.path.join(MODELS_DIR, "best_model_meta.json")

REGISTERED_MODEL_NAME = "cartwave-return-probability-pct-predictor"
EXPERIMENT_NAME       = "cartwave-return-probability-pct"


def register():
    # ── Load meta produced by train.py ────────────────────────────────────────
    if not os.path.exists(META_PATH):
        raise FileNotFoundError("Run src/train.py first to generate models/best_model_meta.json")

    with open(META_PATH) as f:
        meta = json.load(f)

    best_run_id = meta["best_run_id"]

    # ── MLflow client ─────────────────────────────────────────────────────────
    tracking_uri = os.path.join(BASE_DIR, "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # ── Get the RMSE metric for this run ──────────────────────────────────────
    run_data = client.get_run(best_run_id)
    best_rmse = run_data.data.metrics.get("rmse", 0.0)

    # ── Register model ────────────────────────────────────────────────────────
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    version = int(mv.version)
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {version}  (run_id={best_run_id})")

    # ── Write results/step3_s6.json ───────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version":               version,
        "run_id":                best_run_id,
        "source_metric":         "rmse",
        "source_metric_value":   round(best_rmse, 4),
    }
    out_path = os.path.join(RESULTS_DIR, "step3_s6.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    register()
