"""
Task 2 — FastAPI Serving
Serves the best model on port 8500.
  POST /score  — predict return_probability_pct
  GET  /health — liveness check
Run:  uvicorn src.api:app --host 0.0.0.0 --port 8500
"""

import os
import json
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.joblib")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Load model ─────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run src/train.py first."
    )
model = joblib.load(MODEL_PATH)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="CartWave Return Probability API")


class Features(BaseModel):
    product_price:   float = Field(..., ge=100,  le=5000)
    delivery_days:   int   = Field(..., ge=1,    le=10)
    customer_rating: float = Field(..., ge=1.0,  le=5.0)
    is_first_order:  int   = Field(..., ge=0,    le=1)


@app.get("/health")
def health():
    return {"alive": True, "service": "CartWave return_probability_pct API"}


@app.post("/score")
def score(features: Features):
    X = np.array([[
        features.product_price,
        features.delivery_days,
        features.customer_rating,
        features.is_first_order,
    ]])
    prediction = float(round(model.predict(X)[0], 4))
    return {"prediction": prediction}


# ── Standalone test runner (saves step2_s4.json) ───────────────────────────────
def run_test_and_save():
    """
    Call this AFTER starting the server in a separate process.
    It hits /health and /score, then writes results/step2_s4.json.
    """
    base_url = "http://127.0.0.1:8500"
    test_input = {
        "product_price":   2607.6,
        "delivery_days":   7,
        "customer_rating": 3.1,
        "is_first_order":  1,
    }

    health_resp     = requests.get(f"{base_url}/health").json()
    prediction_resp = requests.post(f"{base_url}/score", json=test_input).json()
    prediction_val  = prediction_resp.get("prediction", 0.0)

    output = {
        "health_endpoint":  "/health",
        "predict_endpoint": "/score",
        "port":             8500,
        "health_response":  health_resp,
        "test_input":       test_input,
        "prediction":       prediction_val,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "step2_s4.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    # When run directly, start server AND test it programmatically
    import threading, time, uvicorn

    def start_server():
        uvicorn.run("src.api:app", host="0.0.0.0", port=8500, log_level="warning")

    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(3)          # wait for server to be ready
    run_test_and_save()
    print("\nServer is still running. Press Ctrl+C to stop.")
    t.join()
