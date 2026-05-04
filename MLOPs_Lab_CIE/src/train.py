"""
Task 1 — Experiment Tracking & Model Comparison
Trains Lasso and RandomForest, logs to MLflow, saves best model.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "cartwave-return-probability-pct"
FEATURES = ["product_price", "delivery_days", "customer_rating", "is_first_order"]
TARGET   = "return_probability_pct"


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def train():
    # ── Load & split ───────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.path.join(BASE_DIR, "mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # ── Model configs ─────────────────────────────────────────────────────────
    model_configs = [
        {
            "name": "Lasso",
            "model": Lasso(alpha=0.1, max_iter=10000, random_state=42),
            "params": {"alpha": 0.1, "max_iter": 10000, "random_state": 42},
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        },
    ]

    for cfg in model_configs:
        with mlflow.start_run(run_name=cfg["name"]):
            mlflow.set_tag("experiment_type", "baseline_comparison")

            # Log params
            for k, v in cfg["params"].items():
                mlflow.log_param(k, v)

            # Train
            cfg["model"].fit(X_train, y_train)
            y_pred = cfg["model"].predict(X_test)

            mae, rmse, r2 = compute_metrics(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("mae",  round(mae,  4))
            mlflow.log_metric("rmse", round(rmse, 4))
            mlflow.log_metric("r2",   round(r2,   4))

            # Log model artifact
            mlflow.sklearn.log_model(cfg["model"], artifact_path="model")

            run_id = mlflow.active_run().info.run_id

        results.append({
            "name":   cfg["name"],
            "model":  cfg["model"],
            "mae":    round(mae,  4),
            "rmse":   round(rmse, 4),
            "r2":     round(r2,   4),
            "run_id": run_id,
        })
        print(f"{cfg['name']:15s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # ── Pick best by RMSE ──────────────────────────────────────────────────────
    best = min(results, key=lambda x: x["rmse"])
    print(f"\nBest model: {best['name']}  (RMSE={best['rmse']})")

    # ── Save best model to disk ────────────────────────────────────────────────
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best["model"], best_model_path)
    # Also save best model name for downstream tasks
    meta = {"best_model_name": best["name"], "best_run_id": best["run_id"]}
    with open(os.path.join(MODELS_DIR, "best_model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Write results/step1_s1.json ────────────────────────────────────────────
    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [
            {"name": r["name"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]}
            for r in results
        ],
        "best_model":        best["name"],
        "best_metric_name":  "rmse",
        "best_metric_value": best["rmse"],
    }
    out_path = os.path.join(RESULTS_DIR, "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")
    return output


if __name__ == "__main__":
    train()
