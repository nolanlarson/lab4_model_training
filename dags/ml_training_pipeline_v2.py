from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.cancer_data import load_data
from ml_pipeline.breast_cancer_model import train_model, eval_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="train_model_only",
    default_args=default_args,
    description="Train ML model only (expects data to already exist)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def train_model_wrapper(data_path: str, model_path: str):
        df = load_data(data_path)
        acc = train_model(df, model_path)
        return acc
        
    ## calling wrapper here so we don't accidentally retrain for model eval.
    acc = train_model_wrapper("data/breast_cancer.csv", "models/breast_cancer_model.pkl")
    
    def eval_model_wrapper(data_path: str, model_path: str):
        df = load_data(data_path)
        return eval_model(acc)

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=acc,
    )
    eval_task = PythonOperator(
        task_id="eval_model",
        python_callable=eval_model_wrapper,
    )

    generate_task >> train_task