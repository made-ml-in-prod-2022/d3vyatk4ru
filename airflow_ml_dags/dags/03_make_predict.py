from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


default_args = {
    'owner': 'airflow',
    'email' : ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    '03_make_predict',
    default_args=default_args,
    description='DAG to make a predict',
    schedule_interval="@daily",
    start_date=days_ago(2),
) as dag:

    predict = DockerOperator(
        task_id='Make_prediction',
        image='airflow-predict',
        network_mode="bridge",
        do_xcom_push=False,
        mounts=[
            Mount(
                source='/home/d3vyatk4ru/Desktop/TP-ml-prod-Airflow/data',
                target='/data',
                type='bind',
        )],
        command='--input_path /data/splitted/{{ ds }} '
                '--pred_path /data/predictions/{{ ds }} '
                '--scaler_path /data/models/{{ ds }} '
                '--model_path /data/models/{{ ds }}',
    )

    predict