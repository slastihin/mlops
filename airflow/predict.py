import airflow
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago

from kubernetes.client import models as k8s

default_args = {
    'owner': 'aslastikhin',
    # 'start_date': airflow.utils.dates.days_ago(2),
    # 'end_date': datetime(),
    # 'depends_on_past': False,
    # 'email': ['airflow@example.com'],
    # 'email_on_failure': False,
    # 'email_on_retry': False,
    # If a task fails, retry it once after waiting
    # at least 5 minutes
    # 'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

IMAGE = 'registry.neomsa.ru/docker-mlops/mlops/airflow:2.2.5-demo-v5'

with DAG(
        dag_id="predict_demo",
        default_args=default_args,
        # schedule_interval='0 0 * * *',
        schedule_interval='@once',
        dagrun_timeout=timedelta(minutes=60),
        description='batch model predict',
        start_date=airflow.utils.dates.days_ago(1),
        catchup=False
) as dag:
    





    def predict(**kwargs):

        import mlflow
        import os
        import pandas as pd
        import numpy as np

        df =pd.DataFrame( np.array([[27, 4, 10, 0, 1,0,4,0,0,0,38,39], [29, 4, 13, 2, 4,4,2,1,0,0,55,39], [29, 6, 10, 0, 3,0,4,1,2202,0,50,39]]), columns=['Age', 'Workclass', 'Education-Num', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Country'])
        # Load model as a PyFuncModel.



        model_name = "demo"
        model_version = 1

        loaded_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )

        # Predict on a Pandas DataFrame.
        result = loaded_model.predict(df)
        print("!!!!!!! DAG PREDICT - " + str(result))



    batch_predict = PythonOperator(
        task_id="batch_predict",
        python_callable=predict,
        provide_context=True,
        executor_config={
            "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
        },
    )

batch_predict
