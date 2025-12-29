from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from email_helper import success_email, failure_email

with DAG(
    dag_id="run_data_and_ml_pipeline",
    description="Run Data and ML Pipeline",
    start_date=datetime(2025, 6, 1),
    catchup=False,
) as dag:

    run_bronze_data_pipeline = BashOperator(
        task_id="run_bronze_stages",
        # data_ml_container is defined in the docker-compose.yml file
        bash_command="docker exec data_ml_container python run_bronze_data_pipeline.py",      
        on_failure_callback=failure_email,    
    )

    run_silver_data_pipeline = BashOperator(
        task_id="run_silver_stages",
        # data_ml_container is defined in the docker-compose.yml file
        bash_command="docker exec data_ml_container python run_silver_data_pipeline.py",
        on_failure_callback=failure_email,            
    )

    run_gold_data_pipeline = BashOperator(
        task_id="run_gold_stages",
        # data_ml_container is defined in the docker-compose.yml file
        bash_command="docker exec data_ml_container python run_gold_data_pipeline.py",
        on_failure_callback=failure_email,        
    )


    run_xgboost_ml_training_pipeline = BashOperator(
        task_id="train_xgboost_model",
        bash_command="docker exec data_ml_container python run_ml_pipeline.py --config_path ml/xg_conf.yaml",
        on_success_callback=success_email,
        on_failure_callback=failure_email,
    )

    run_logistic_ml_training_pipeline = BashOperator(
        task_id="train_logistic_model",
        bash_command="docker exec data_ml_container python run_ml_pipeline.py --config_path ml/log_conf.yaml",
        on_success_callback=success_email,
        on_failure_callback=failure_email,
    )   


    (
        run_bronze_data_pipeline
        >> run_silver_data_pipeline
        >> run_gold_data_pipeline
    )

    # Running two ML training pipelines in parallel on same container and writing to same MLFlow is very slow (10 mins)
    # as compared to running them sequentially (1 min). Might be resource contention of writing to same MLFlow at same time.
    # run_online_feature_data_pipeline >> run_xgboost_ml_training_pipeline >> run_logistic_ml_training_pipeline
    run_gold_data_pipeline >> run_xgboost_ml_training_pipeline
    run_gold_data_pipeline >> run_logistic_ml_training_pipeline