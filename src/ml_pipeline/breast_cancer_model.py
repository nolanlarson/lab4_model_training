import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'models',
    'metrics.json'
)

def train_model(df: pd.DataFrame, model_path: str = "models/breast_cancer.pkl"):
    """Train model and return test split."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)

    return X_test, y_test


def eval_model(X_test, y_test, model_path: str = "models/breast_cancer.pkl") -> float:
    """Evaluate model and log metrics."""
    
    model = joblib.load(model_path)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")

    # Ensure metrics file exists
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([], f)

    with open(file_path, 'r') as f:
        data = json.load(f)

    data.append({"accuracy": acc})

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Saved metric to {file_path}")

    return acc
    
    