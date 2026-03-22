import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import pickle
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'metrics.json')

def train_model(df: pd.DataFrame, model_path: str = "models/breast_cancer.pkl") -> float:
    """Train a logistic regression classifier and save it."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    
    joblib.dump(clf, model_path)
    
    return X_test, y_test
    
def eval_model(model_path: str = "models/breast_cancer.pkl") -> float:
    """Evaluate model and save it."""
    
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
        
    preds = loaded_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Adding in new topic if it doesn't exist
        new_entry = {"accuracy": acc}
        
        data.append(new_entry)
    
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"New model metric with value '{acc}' appended to {file_path}")
        
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[ml_pipeline.model] Saved model to {model_path}")

    return acc
    
    
    