import os, sys, mlflow

THRESHOLD = 0.85

try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print("ERROR: model_info.txt not found.")
    print(f"Current dir: {os.getcwd()}")
    print(f"Files: {os.listdir('.')}")
    sys.exit(1)

print(f"Run ID: {run_id}")
print(f"Files in current dir: {os.listdir('.')}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

try:
    run_data = client.get_run(run_id)
except Exception as e:
    print(f"ERROR fetching run: {e}")
    sys.exit(1)

accuracy = run_data.data.metrics.get("accuracy")
if accuracy is None:
    print("ERROR: 'accuracy' metric not found.")
    sys.exit(1)

print(f"Accuracy: {accuracy:.4f} | Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAIL: {accuracy:.4f} < {THRESHOLD}. Halting deployment.")
    sys.exit(1)

print(f"PASS: {accuracy:.4f} >= {THRESHOLD}. Proceeding to deploy.")
sys.exit(0)