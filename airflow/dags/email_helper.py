from airflow.utils.email import send_email

##### SETUP EMAIL CONFIGURATION #####
# Note that Airflow 3.0.x has a bug for their default email template, and this helper function is used instead.
# https://github.com/apache/airflow/pull/50376
# In Airflow UI, go to Admin -> Connections
# Create a new connection with the following details:
# Conn Id: smtp_default
# Conn Type: email
# Host: smtp.gmail.com
# Port: 587
# Login: mleprojectgrp8@gmail.com
# Password: <your_password>
# Change the below RECIPIENT_EMAIL to your email address
RECIPIENT_EMAIL = ''

def success_email(context):
    task_instance = context['task_instance']
    task_status = 'Success' 
    subject = f'[{task_status}] Airflow Task {task_instance.task_id} on {task_instance.start_date.strftime("%Y-%m-%d %H:%M:%S")}'
    body = f'The task {task_instance.task_id} from DAG {task_instance.dag_id} completed with status : {task_status}. \n\n'\
           f'The task execution start date is: {task_instance.start_date.strftime("%Y-%m-%d %H:%M:%S")}\n'
    to_email = RECIPIENT_EMAIL
    send_email(to = to_email, subject = subject, html_content = body)

def failure_email(context):
    task_instance = context['task_instance']
    task_status = 'Failed'
    subject = f'[{task_status}] Airflow Task {task_instance.task_id} on {task_instance.start_date.strftime("%Y-%m-%d %H:%M:%S")} {task_status}'
    body = f'The task {task_instance.task_id} from DAG {task_instance.dag_id} completed with status : {task_status}. \n\n'\
           f'The task execution start date is:  {task_instance.start_date.strftime("%Y-%m-%d %H:%M:%S")}\n'    
    to_email = RECIPIENT_EMAIL
    send_email(to = to_email, subject = subject, html_content = body)