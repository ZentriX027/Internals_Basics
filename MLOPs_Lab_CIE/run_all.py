"""
run_all.py — Master pipeline runner
Executes all 4 tasks in sequence.
Run from the MLOPs_Lab_CIE/ directory:
    python run_all.py
"""

import subprocess
import sys
import os
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "src", script)],
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        print(f"\n❌  {label} FAILED. Check errors above.")
        sys.exit(1)
    print(f"✅  {label} complete.")


def task2_api():
    """Start API server, run test, save JSON, shut down."""
    print(f"\n{'='*60}")
    print("  Task 2 — FastAPI Serving")
    print(f"{'='*60}")

    import threading
    import uvicorn

    # Temporarily add BASE_DIR to path so 'src.api' resolves
    sys.path.insert(0, BASE_DIR)
    from src.api import app, run_test_and_save

    def start():
        uvicorn.run(app, host="0.0.0.0", port=8500, log_level="warning")

    t = threading.Thread(target=start, daemon=True)
    t.start()

    # Wait until server is ready
    for _ in range(20):
        try:
            requests.get("http://127.0.0.1:8500/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    run_test_and_save()
    print("✅  Task 2 complete.")


if __name__ == "__main__":
    run("train.py",          "Task 1 — Experiment Tracking & Model Comparison")
    task2_api()
    run("register_model.py", "Task 3 — Model Versioning")
    run("retrain.py",        "Task 4 — Retraining Pipeline")

    print(f"\n{'='*60}")
    print("  ALL TASKS COMPLETE 🎉")
    print(f"{'='*60}")
    print("Results saved in results/")
    print("  step1_s1.json")
    print("  step2_s4.json")
    print("  step3_s6.json")
    print("  step4_s8.json")
