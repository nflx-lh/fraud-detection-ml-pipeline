from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="run_online_feature_data_pipeline",
    description="Run Online Feature Data Pipeline",
    start_date=datetime(2025, 6, 1),
    catchup=False,
) as dag:

    run_online_feature_data_pipeline = BashOperator(
        task_id="run_online_feature_data_pipeline",
        # data_ml_container is defined in the docker-compose.yml file
        bash_command="docker exec data_ml_container python run_online_feature_data_pipeline.py",
    )


