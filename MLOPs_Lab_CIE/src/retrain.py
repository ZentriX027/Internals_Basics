"""
Task 4 — Retraining Pipeline
Combines training_data.csv + new_data.csv, retrains the champion model type,
and promotes only if RMSE improves by >= 0.5.
Saves results/step4_s8.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV      = os.path.join(BASE_DIR, "data", "training_data.csv")
NEW_CSV        = os.path.join(BASE_DIR, "data", "new_data.csv")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
META_PATH      = os.path.join(MODELS_DIR, "best_model_meta.json")
STEP1_JSON     = os.path.join(RESULTS_DIR, "step1_s1.json")

FEATURES = ["product_price", "delivery_days", "customer_rating", "is_first_order"]
TARGET   = "return_probability_pct"
MIN_IMPROVEMENT = 0.5
EXPERIMENT_NAME = "cartwave-return-probability-pct"


def make_model(model_name: str):
    if model_name == "Lasso":
        return Lasso(alpha=0.1, max_iter=10000, random_state=42)
    return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def retrain():
    # ── Load champion info ────────────────────────────────────────────────────
    if not os.path.exists(STEP1_JSON):
        raise FileNotFoundError("Run src/train.py first.")

    with open(STEP1_JSON) as f:
        step1 = json.load(f)

    champion_name = step1["best_model"]
    champion_rmse_val = step1["best_metric_value"]

    # ── Load and combine datasets ─────────────────────────────────────────────
    df_orig = pd.read_csv(TRAIN_CSV)
    df_new  = pd.read_csv(NEW_CSV)
    df_comb = pd.concat([df_orig, df_new], ignore_index=True)

    orig_rows = len(df_orig)
    new_rows  = len(df_new)
    comb_rows = len(df_comb)

    print(f"Original rows : {orig_rows}")
    print(f"New rows      : {new_rows}")
    print(f"Combined rows : {comb_rows}")

    # ── Split combined data ───────────────────────────────────────────────────
    X = df_comb[FEATURES]
    y = df_comb[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Also evaluate champion on combined test set ───────────────────────────
    champion_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    champ_pred     = champion_model.predict(X_test)
    champ_rmse_on_combined = rmse(y_test, champ_pred)

    # ── Retrain same model type ───────────────────────────────────────────────
    mlflow.set_tracking_uri(os.path.join(BASE_DIR, "mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    new_model = make_model(champion_name)
    with mlflow.start_run(run_name=f"{champion_name}_retrained"):
        mlflow.set_tag("experiment_type", "retraining")
        new_model.fit(X_train, y_train)
        new_pred  = new_model.predict(X_test)
        new_rmse  = rmse(y_test, new_pred)
        mlflow.log_metric("rmse", round(new_rmse, 4))
        mlflow.log_param("data_rows", comb_rows)
        mlflow.sklearn.log_model(new_model, artifact_path="model")

    improvement = round(champ_rmse_on_combined - new_rmse, 4)
    print(f"Champion RMSE (on combined test): {champ_rmse_on_combined:.4f}")
    print(f"Retrained RMSE                  : {new_rmse:.4f}")
    print(f"Improvement                     : {improvement:.4f}")

    # ── Promotion decision ────────────────────────────────────────────────────
    if improvement >= MIN_IMPROVEMENT:
        action = "promoted"
        joblib.dump(new_model, os.path.join(MODELS_DIR, "best_model.joblib"))
        print("✅  New model PROMOTED and saved.")
    else:
        action = "kept_champion"
        print("⏸   Champion retained (improvement below threshold).")

    # ── Write results/step4_s8.json ───────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "original_data_rows":    orig_rows,
        "new_data_rows":         new_rows,
        "combined_data_rows":    comb_rows,
        "champion_rmse":         round(champ_rmse_on_combined, 4),
        "retrained_rmse":        round(new_rmse, 4),
        "improvement":           improvement,
        "min_improvement_threshold": MIN_IMPROVEMENT,
        "action":                action,
        "comparison_metric":     "rmse",
    }
    out_path = os.path.join(RESULTS_DIR, "step4_s8.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    retrain()
