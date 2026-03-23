"""
check_threshold.py
Reads the MLflow Run ID from model_info.txt, queries the tracking server
for the logged accuracy metric, and exits with code 1 if it is below the
required threshold — causing GitHub Actions to halt the pipeline.
"""

import sys
import mlflow

THRESHOLD = 0.85

# Read Run ID produced by the validate job 
try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print("ERROR: model_info.txt not found. Did the validate job upload it?")
    sys.exit(1)

if not run_id:
    print("ERROR: model_info.txt is empty.")
    sys.exit(1)

print(f"Checking accuracy for Run ID: {run_id}")

# Fetch the run from MLflow
mlflow.set_tracking_uri("./mlruns")
client   = mlflow.tracking.MlflowClient()
run_data = client.get_run(run_id)
metrics  = run_data.data.metrics

if "accuracy" not in metrics:
    print("ERROR: 'accuracy' metric not found in this run.")
    sys.exit(1)

accuracy = metrics["accuracy"]
print(f"Accuracy logged: {accuracy:.4f}")
print(f"Threshold      : {THRESHOLD}")

# Gate
if accuracy < THRESHOLD:
    print(
        f"FAIL: accuracy {accuracy:.4f} is below threshold {THRESHOLD}. "
        "Halting deployment."
    )
    sys.exit(1)   # Non-zero exit → GitHub Actions marks the step (and job) as failed

print(f"PASS: accuracy {accuracy:.4f} meets the threshold. Proceeding to deploy.")
sys.exit(0)