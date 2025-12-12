from datetime import datetime, timedelta
import os
import hashlib
import json

from airflow import DAG
from airflow.operators.python import PythonOperator

import boto3
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

# ----------------- Config / constants -----------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

TEAM = "{{ cookiecutter.team }}"
PROJECT_SLUG = "{{ cookiecutter.project_slug }}"
ENV = os.getenv("PLATFORM_ENV", "dev").lower()

S3_BUCKET = os.getenv("TEAM_S3_BUCKET", "{{ cookiecutter.s3_bucket }}")
S3_KEY = os.getenv("TEAM_S3_TRAIN_KEY", "{{ cookiecutter.train_key }}")
LOCAL_DATA_PATH = "/tmp/iris_train.csv"

ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.8"))
RUN_OWNER = os.getenv("RUN_OWNER", "example_user")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


# ----------------- Tasks -----------------


def ingest_from_s3(**context):
    s3 = get_s3_client()
    print(f"Downloading s3://{S3_BUCKET}/{S3_KEY} from MinIO {MINIO_ENDPOINT}")
    s3.download_file(S3_BUCKET, S3_KEY, LOCAL_DATA_PATH)

    if not os.path.exists(LOCAL_DATA_PATH):
        raise FileNotFoundError(LOCAL_DATA_PATH)

    audit_log("ingest_success", {"dataset_path": LOCAL_DATA_PATH})
    context["ti"].xcom_push(key="dataset_path", value=LOCAL_DATA_PATH)


def validate_iris_data(**context):
    path = context["ti"].xcom_pull(key="dataset_path", task_ids="ingest")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Iris dataset is empty")

    dataset_hash = hash_file(path)
    audit_log(
        "validation_success",
        {"dataset_path": path, "dataset_hash": dataset_hash, "row_count": len(df)},
    )
    context["ti"].xcom_push(key="dataset_hash", value=dataset_hash)


def train_iris_model(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(key="dataset_path", task_ids="ingest")
    dataset_hash = ti.xcom_pull(key="dataset_hash", task_ids="validate")

    df = pd.read_csv(dataset_path)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    experiment_name = f"{TEAM}-{ENV}-{PROJECT_SLUG}"
    mlflow.set_experiment(experiment_name)

    approval_state = "approved" if accuracy >= ACCURACY_THRESHOLD else "rejected"
    run_ts = datetime.utcnow().isoformat() + "Z"

    with mlflow.start_run(run_name=f"{TEAM}_{ENV}_{PROJECT_SLUG}_train") as run:
        mlflow.log_param("team", TEAM)
        mlflow.log_param("env", ENV)
        mlflow.log_param("run_owner", RUN_OWNER)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.set_tag("dataset_hash", dataset_hash)
        mlflow.set_tag("approval_state", approval_state)
        mlflow.set_tag("run_owner", RUN_OWNER)
        mlflow.set_tag("run_timestamp", run_ts)
        mlflow.set_tag("project_slug", PROJECT_SLUG)

        mlflow.sklearn.log_model(clf, artifact_path="model")

        ti.xcom_push(key="mlflow_run_id", value=run.info.run_id)
        ti.xcom_push(key="accuracy", value=float(accuracy))
        ti.xcom_push(key="approval_state", value=approval_state)

        audit_log(
            "training_complete",
            {
                "mlflow_run_id": run.info.run_id,
                "dataset_hash": dataset_hash,
                "accuracy": accuracy,
                "approval_state": approval_state,
                "run_timestamp": run_ts,
            },
        )


def register_iris_model(**context):
    ti = context["ti"]
    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="train")
    accuracy = float(ti.xcom_pull(key="accuracy", task_ids="train"))
    approval_state = ti.xcom_pull(key="approval_state", task_ids="train")

    model_name = f"{TEAM}_{ENV}_{PROJECT_SLUG}_model"
    model_version = None

    if approval_state == "approved":
        client = MlflowClient()
        registered_model = client.create_registered_model(model_name=model_name)
    else:
        # If not approved, we still log the event but do not register
        audit_log(
            "model_not_registered",
            {
                "mlflow_run_id": run_id,
                "model_name": model_name,
                "accuracy": accuracy,
                "approval_state": approval_state,
            },
        )
        return

    client = MlflowClient()
    mv = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    model_version = mv.version

    # Example policy: keep multiple Staging, only one Production is enforced elsewhere
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging",
    )

    audit_log(
        "model_registration",
        {
            "mlflow_run_id": run_id,
            "model_name": model_name,
            "model_version": model_version,
            "accuracy": accuracy,
            "approval_state": approval_state,
        },
    )


default_args = {
    "owner": TEAM,
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DAG_ID = f"{TEAM}_{ENV}_{PROJECT_SLUG}_pipeline"

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=[PROJECT_SLUG, TEAM, ENV],
    # NOTE: keep access_control commented unless roles exist in Airflow
    # access_control={
    #     "team_a_ds": {"can_read", "can_edit"},
    #     "platform_ops": {"can_read", "can_edit"},
    # },
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest",
        python_callable=ingest_from_s3,
    )

    validate_task = PythonOperator(
        task_id="validate",
        python_callable=validate_iris_data,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=train_iris_model,
    )

    register_task = PythonOperator(
        task_id="register",
        python_callable=register_iris_model,
    )

    ingest_task >> validate_task >> train_task >> register_task