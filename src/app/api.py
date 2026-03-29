# src/app/api.py
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json



# Explicit request schema for Iris dataset (4 features)
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
 
# Explicit request schema for breast cancer dataset (n features)
class BreastCancerRequest(BaseModel):
    mean_radius: float
    mean_texture: float 
    mean_perimeter: float 
    mean_area:float
    mean_smoothness:float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float 
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float 
    perimeter_error: float
    area_error: float
    smoothness_error: float 
    compactness_error:float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter:float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

def create_app(model_path: str = "models/breast_cancer_model.pkl", metrics_path: str = "models/metrics.json",):

    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. Train the model first."
        )

    model = joblib.load(model_path)

    # Load latest metadata from metrics.json
    if Path(metrics_path).exists():
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            latest_metrics = metrics_data[-1] if metrics_data else {}
    else:
        latest_metrics = {}

    app = FastAPI(title="Breast Cancer Model API")

    # Binary classification labels
    target_names = {0: "malignant", 1: "benign"}

    @app.get("/healthcheck")
    def root():
        return {
            "message": "200 - Healthy",
            "classes": target_names,
        }

    @app.post("/predict")
    def predict(request: BreastCancerRequest):

        X = np.array([[request.mean_radius,
request.mean_texture,
request.mean_perimeter,
request.mean_area,
request.mean_smoothness,
request.mean_compactness,
request.mean_concavity,
request.mean_concave_points,
request.mean_symmetry,
request.mean_fractal_dimension,
request.radius_error,
request.texture_error,
request.perimeter_error,
request.area_error,
request.smoothness_error,
request.compactness_error,
request.concavity_error,
request.concave_points_error,
request.symmetry_error,
request.fractal_dimension_error,
request.worst_radius,
request.worst_texture,
request.worst_perimeter,
request.worst_area,
request.worst_smoothness,
request.worst_compactness,
request.worst_concavity,
request.worst_concave_points,
request.worst_symmetry,
request.worst_fractal_dimension]])

        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {"prediction": target_names[idx], "class_index": idx}
        
        
    @app.get("/model/info")
    def model_info():
        return {
            "model_version": latest_metrics.get("version", "unknown"),
            "dataset": "breast_cancer",
            "model_type": "logistic_regression",
            "accuracy": latest_metrics.get("accuracy", None),
        }

    # return the FastAPI app
    return app
