import os, mlflow, mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

if os.getenv("FORCE_FAIL", "0") == "1":
    print("FORCE_FAIL=1 — using tiny training set")
    X_train, X_test = X[:5], X[5:]
    y_train, y_test = y[:5], y[5:]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
MAX_DEPTH     = int(os.getenv("MAX_DEPTH", 5))

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("assignment5-pipeline")

with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))

    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, artifact_path="model")

    print(f"Run ID  : {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

print("model_info.txt written successfully.")