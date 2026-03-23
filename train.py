"""
train.py
Trains a simple classifier on the dataset, logs metrics to MLflow,
and writes the Run ID to model_info.txt for the deploy job to consume.
"""

import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (replace with dvc-pulled CSV if needed)
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameters 
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
MAX_DEPTH     = int(os.getenv("MAX_DEPTH", 5))

# MLflow run 
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("assignment5-pipeline")

with mlflow.start_run() as run:
    # Train
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    preds    = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Log to MLflow
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth",    MAX_DEPTH)
    mlflow.log_metric("accuracy",    accuracy)
    mlflow.sklearn.log_model(clf, artifact_path="model")

    print(f"Run ID  : {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

    # Export Run ID for the deploy job
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

print("model_info.txt written successfully.")