from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="run_monitoring_pipeline",
    description="Run monitoring Pipeline",
    start_date=datetime(2025, 6, 1),
    catchup=False,
) as dag:

    run_monitoring_pipeline = BashOperator(
        task_id="run_monitoring_pipeline",
        bash_command="docker exec ml_inference_monitoring_container python run_monitoring_pipeline.py",
    )
    
    start_evidently_ui = BashOperator(
        task_id="start_evidently_ui",
        bash_command="docker exec ml_inference_monitoring_container evidently ui --port 8080 --host 0.0.0.0",
    )

    run_monitoring_pipeline >> start_evidently_ui