import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import timedelta
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

IMAGE = 'registry.neomsa.ru/docker-mlops/mlops/airflow:2.2.5-demo-v4'

with DAG(
        dag_id="train_model",
        default_args=default_args,
        # schedule_interval='0 0 * * *',
        schedule_interval='@once',
        dagrun_timeout=timedelta(minutes=60),
        description='predict demo',
        start_date=days_ago(1),
        catchup=False
) as dag:



    def train_model():
        import os
        os.system('pip install shap')
        import mlflow
        #import xgboost
        from sklearn.ensemble import GradientBoostingClassifier
        import shap
        from sklearn.model_selection import train_test_split
        X, y = shap.datasets.adult()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        X, y = shap.datasets.adult()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        mlflow.set_experiment('demo-dag')
        # train XGBoost model
        model = GradientBoostingClassifier().fit(X_train, y_train)

        # construct an evaluation dataset from the test set
        eval_data = X_test
        eval_data["target"] = y_test
        with mlflow.start_run() as run:
            model_info = mlflow.sklearn.log_model(model, "model", registered_model_name='demo')
            result = mlflow.evaluate(
                model_info.model_uri,
                eval_data,
                targets="target",
                model_type="classifier",
                dataset_name="adult",
                evaluators=["default"],
            )
            

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
        executor_config={
            "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
        }
    )

train_model_task