from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.cancer_data import load_data
from ml_pipeline.breast_cancer_model import train_model, eval_model

default_args = {"owner": "airflow", "retries": 1}

DATA_PATH = "data/breast_cancer.csv"
MODEL_PATH = "models/breast_cancer.pkl"

with DAG(
    dag_id="train_val_promo",
    default_args=default_args,
    description="Train/Val/Promote ML model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def train_model_wrapper(**context):
        df = load_data(DATA_PATH)
        X_test, y_test = train_model(df, MODEL_PATH)

        # Push test data to XCom
        context["ti"].xcom_push(key="X_test", value=X_test.to_json())
        context["ti"].xcom_push(key="y_test", value=y_test.to_json())

    def eval_model_wrapper(**context):
        ti = context["ti"]

        # Pull test data from XCom
        import pandas as pd
        X_test = pd.read_json(ti.xcom_pull(key="X_test", task_ids="train_model"))
        y_test = pd.read_json(ti.xcom_pull(key="y_test", task_ids="train_model"))

        acc = eval_model(X_test, y_test, MODEL_PATH)

        # Push accuracy for next task
        ti.xcom_push(key="accuracy", value=acc)

    def promote_model_wrapper(**context):
        ti = context["ti"]
        acc = ti.xcom_pull(key="accuracy", task_ids="eval_model")

        threshold = 0.9
        if acc >= threshold:
            print(f"Model promoted. Accuracy: {acc}")
        else:
            print(f"Model NOT promoted. Accuracy: {acc}")

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_wrapper,
    )

    eval_task = PythonOperator(
        task_id="eval_model",
        python_callable=eval_model_wrapper,
    )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
    )

    train_task >> eval_task >> promote_task