from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os
import joblib
import json
import boto3

# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.cancer_data import generate_data, load_data
from ml_pipeline.breast_cancer_model import train_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Pipeline: train model -> evaluate model -> promote model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # -----------------------
    # Task 1: Train Model
    # -----------------------
    def train_model_wrapper(data_path: str, model_path: str):
        df = load_data(data_path)

        # Train model (this should save model internally OR we do it here)
        acc, clf = train_model(df)

        # Save model to disk
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(clf, model_path)

        print(f"[train_model] Model saved to {model_path}")
        print(f"[train_model] Accuracy: {acc:.4f}")

        return acc  # goes to XCom automatically

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_wrapper,
        op_kwargs={
            "data_path": "data/breast_cancer.csv",
            "model_path": "models/breast_cancer_model.pkl",
        },
    )

    # -----------------------
    # Task 2: Evaluate Model
    # -----------------------
    def eval_model_wrapper(model_path: str, metrics_path: str, **kwargs):

        ti = kwargs["ti"]
        run_id = kwargs["run_id"]
        execution_date = kwargs["execution_date"]
        
        # Format version: YYYYMMDD_HHMMSS
        version = execution_date.strftime("%Y%m%d_%H%M%S")

        # Pull accuracy from train task (XCom)
        acc = ti.xcom_pull(task_ids="train_model")

        # Load trained model
        clf = joblib.load(model_path)

        print(f"[eval_model] Loaded model from {model_path}")
        print(f"[eval_model] Accuracy from training: {acc:.4f}")
        print(f"[eval_model] Version: {version}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        # Load existing metrics or initialize
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append new accuracy
        metrics = {
        "version": version,
        "run_id": run_id,
        "accuracy": acc
        }

        data.append(metrics)

        # Save back to file
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[eval_model] Saved accuracy to {metrics_path}")

        return metrics

    eval_task = PythonOperator(
    task_id="eval_model",
    python_callable=eval_model_wrapper,
    op_kwargs={
        "model_path": "models/breast_cancer_model.pkl",
        "metrics_path": "models/metrics.json",
    },
    )   

    # -----------------------
    # Task 3: Promote Model
    # -----------------------

    def promote_model_wrapper(model_path: str, metrics_path: str, bucket_name: str, **kwargs):

        ti = kwargs["ti"]

        # Pull metrics from eval task
        metrics = ti.xcom_pull(task_ids="eval_model")

        acc = metrics["accuracy"]
        version = metrics["version"]
        run_id = metrics["run_id"]

        print(f"[promote_model] Evaluating version {version} with accuracy {acc:.4f}")

        # Threshold check
        if acc < 0.94:
            raise ValueError(f"Model accuracy {acc:.4f} below threshold (0.94)")

        print("[promote_model] Model passed threshold")

        # Create metadata
        metadata = {
            "version": version,
            "run_id": run_id,
            "accuracy": acc
        }

        metadata_path = "models/metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Upload to S3
        s3 = boto3.client("s3")
        base_path = f"models/{version}/"

        s3.upload_file(model_path, bucket_name, base_path + "model.pkl")
        s3.upload_file(metrics_path, bucket_name, base_path + "metrics.json")
        s3.upload_file(metadata_path, bucket_name, base_path + "metadata.json")

        print(f"[promote_model] Uploaded to s3://{bucket_name}/{base_path}")

        return version
    
    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
        op_kwargs={
            "model_path": "models/breast_cancer_model.pkl",
            "metrics_path": "models/metrics.json",
            "bucket_name": "breast-cancer-bucket-651209", 
        },
        )
    
    
    # -----------------------
    # DAG Dependencies
    # -----------------------
    train_task >> eval_task >> promote_task