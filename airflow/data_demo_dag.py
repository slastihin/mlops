# Building the DAG using the functions from data_process and model module
import datetime as dt
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

#path = /Users/alexegorov/Work/demo-notebook/dags/model.py
path = "/opt/airflow/dags/repo/dags/model.py"

dag = DAG(
        'rundag',
        start_date=days_ago(0,0,0,0,0)
    )

model_run = BashOperator(task_id='model_run', bash_command=f"pip3 install sklearn mlflow && python3 {path}", dag=dag)


model_run