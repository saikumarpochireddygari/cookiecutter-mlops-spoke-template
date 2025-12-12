from datetime import datetime, timedelta
import os
import json
import tempfile

from airflow import DAG
from airflow.operators.python import PythonOperator

import boto3
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

TEAM = "{{ cookiecutter.team }}"
PROJECT_SLUG = "{{ cookiecutter.project_slug }}"
ENV = os.getenv("PLATFORM_ENV", "dev").lower()

S3_BUCKET = os.getenv("TEAM_S3_BUCKET", "{{ cookiecutter.s3_bucket }}")
S3_TEST_KEY = os.getenv("TEAM_S3_TEST_KEY", "{{ cookiecutter.test_key }}")
S3_PREDICTIONS_PREFIX = os.getenv(
    "TEAM_PREDICTIONS_PREFIX", "{{ cookiecutter.predictions_prefix }}"
)

RUN_OWNER = os.getenv("RUN_OWNER", "example_user")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def audit_log(event_type: str, payload: dict):
    record = {
        "event_type": event_type,
        "env": ENV,
        "team": TEAM,
        "project": PROJECT_SLUG,
        "run_owner": RUN_OWNER,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **payload,
    }
    print("[AUDIT]", json.dumps(record))


def run_batch_inference(**context):
    s3 = get_s3_client()

    # 1) Download test data
    local_test_path = "/tmp/iris_test.csv"
    print(f"Downloading test data s3://{S3_BUCKET}/{S3_TEST_KEY}")
    s3.download_file(S3_BUCKET, S3_TEST_KEY, local_test_path)
    df = pd.read_csv(local_test_path)

    # 2) Load latest STAGING model
    model_name = f"{TEAM}_{ENV}_{PROJECT_SLUG}_model"
    client = MlflowClient()
    latest = client.get_latest_versions(model_name, stages=["Staging"])

    if not latest:
        raise RuntimeError(f"No STAGING model found for {model_name}")

    model_version = latest[0].version
    run_id = latest[0].run_id
    model_uri = f"models:/{model_name}/Staging"

    model = mlflow.sklearn.load_model(model_uri)

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    preds = model.predict(X)

    result_df = df.copy()
    result_df["prediction"] = preds

    # 3) Log inference run to MLflow
    experiment_name = f"{TEAM}-{ENV}-{PROJECT_SLUG}-batch"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"{TEAM}_{ENV}_{PROJECT_SLUG}_batch_infer"
    ) as run:
        mlflow.log_param("team", TEAM)
        mlflow.log_param("env", ENV)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("source_run_id", run_id)

        # Log predictions as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_path = os.path.join(tmpdir, "iris_batch_predictions.csv")
            result_df.to_csv(pred_path, index=False)
            mlflow.log_artifact(pred_path, artifact_path="batch_predictions")

        # 4) Also push predictions back to MinIO
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        s3_key_out = f"{S3_PREDICTIONS_PREFIX.rstrip('/')}/preds_{ts}.csv"
        print(f"Uploading predictions to s3://{S3_BUCKET}/{s3_key_out}")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            result_df.to_csv(tmp.name, index=False)
            s3.upload_file(tmp.name, S3_BUCKET, s3_key_out)

        audit_log(
            "batch_inference_complete",
            {
                "mlflow_run_id": run.info.run_id,
                "model_name": model_name,
                "model_version": model_version,
                "predictions_s3_path": f"s3://{S3_BUCKET}/{s3_key_out}",
                "row_count": len(result_df),
            },
        )


default_args = {
    "owner": TEAM,
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DAG_ID = f"{TEAM}_{ENV}_{PROJECT_SLUG}_batch_inference"

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    schedule=None,  # on-demand via UI or Jenkins trigger job
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=[PROJECT_SLUG, TEAM, ENV, "batch_inference"],
) as dag:
    batch_infer = PythonOperator(
        task_id="batch_inference",
        python_callable=run_batch_inference,
    )