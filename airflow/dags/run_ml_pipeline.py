from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from email_helper import success_email, failure_email

with DAG(
    dag_id="run_ml_pipeline",
    description="Run ML Pipeline",
    start_date=datetime(2025, 6, 1),
    catchup=False,
) as dag:


    run_online_feature_data_pipeline = BashOperator(
        task_id="run_online_feature_data_pipeline",
        # data_ml_container is defined in the docker-compose.yml file
        bash_command="docker exec data_ml_container python run_online_feature_data_pipeline.py",
        on_failure_callback=failure_email,        
    )

    run_xgboost_ml_training_pipeline = BashOperator(
        task_id="run_xgboost_ml_training_pipeline",
        bash_command="docker exec data_ml_container python run_ml_pipeline.py --config_path ml/xg_conf.yaml",
        on_success_callback=success_email,
        on_failure_callback=failure_email,
    )

    run_logistic_ml_training_pipeline = BashOperator(
        task_id="run_logistic_ml_training_pipeline",
        bash_command="docker exec data_ml_container python run_ml_pipeline.py --config_path ml/log_conf.yaml",
        on_success_callback=success_email,
        on_failure_callback=failure_email,
    )   
 
    run_online_feature_data_pipeline >> run_xgboost_ml_training_pipeline
    run_online_feature_data_pipeline >> run_logistic_ml_training_pipeline
