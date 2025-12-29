from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from email_helper import success_email, failure_email


with DAG(
    dag_id="example_email_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    def my_task_function():
        # raise ValueError("Simulated task failure")
        print("This is a test email DAG")
    

    task1 = PythonOperator(
        task_id="task_with_email",
        python_callable=my_task_function,
        on_success_callback=success_email,
        on_failure_callback=failure_email,
    )


    task1