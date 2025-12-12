from datetime import datetime, timedelta
import os
import json
import tempfile

from airflow import DAG
from airflow.operators.python import PythonOperator

import boto3
import pandas as pd
import numpy as np
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

TEAM = "{{ cookiecutter.team }}"
PROJECT_SLUG = "{{ cookiecutter.project_slug }}"
ENV = os.getenv("PLATFORM_ENV", "dev").lower()

S3_BUCKET = os.getenv("TEAM_S3_BUCKET", "{{ cookiecutter.s3_bucket }}")
S3_TRAIN_KEY = os.getenv("TEAM_S3_TRAIN_KEY", "{{ cookiecutter.train_s3_key }}")
S3_TEST_KEY = os.getenv("TEAM_S3_TEST_KEY", "{{ cookiecutter.test_s3_key }}")
S3_DRIFT_PREFIX = os.getenv(
    "TEAM_DRIFT_PREFIX", "{{ cookiecutter.drift_prefix }}"
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


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10):
    """Simple PSI implementation: bucket by quantiles from expected."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # Drop NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(expected, quantiles)

    # Make edges strictly increasing to avoid weirdness
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    # Avoid log(0)
    exp_perc = np.where(exp_perc == 0, 1e-6, exp_perc)
    act_perc = np.where(act_perc == 0, 1e-6, act_perc)

    psi = np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))
    return float(psi)


def compute_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    rows = []
    for col in feature_cols:
        train_col = train_df[col]
        test_col = test_df[col]

        psi = population_stability_index(train_col.values, test_col.values, bins=10)

        rows.append(
            {
                "feature": col,
                "train_mean": float(train_col.mean()),
                "test_mean": float(test_col.mean()),
                "mean_diff": float(train_col.mean() - test_col.mean()),
                "train_std": float(train_col.std()),
                "test_std": float(test_col.std()),
                "psi": psi,
            }
        )

    return pd.DataFrame(rows)


def run_drift_check(**context):
    s3 = get_s3_client()

    # 1) Download train + test
    train_path = "/tmp/iris_train.csv"
    test_path = "/tmp/iris_test.csv"

    print(f"Downloading train s3://{S3_BUCKET}/{S3_TRAIN_KEY}")
    s3.download_file(S3_BUCKET, S3_TRAIN_KEY, train_path)

    print(f"Downloading test s3://{S3_BUCKET}/{S3_TEST_KEY}")
    s3.download_file(S3_BUCKET, S3_TEST_KEY, test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    drift_df = compute_drift(train_df, test_df, feature_cols)

    # 2) Log to MLflow
    experiment_name = f"{TEAM}-{ENV}-{PROJECT_SLUG}-drift"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"{TEAM}_{ENV}_{PROJECT_SLUG}_drift_check"
    ) as run:
        # Some summary metrics
        mlflow.log_metric("max_psi", float(drift_df["psi"].max()))
        mlflow.log_metric("avg_psi", float(drift_df["psi"].mean()))

        # Log drift table as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            drift_path = os.path.join(tmpdir, "iris_drift_report.csv")
            drift_df.to_csv(drift_path, index=False)
            mlflow.log_artifact(drift_path, artifact_path="drift_report")

        # 3) Push drift report CSV to MinIO
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        s3_key_out = f"{S3_DRIFT_PREFIX.rstrip('/')}/drift_{ts}.csv"
        print(f"Uploading drift report to s3://{S3_BUCKET}/{s3_key_out}")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            drift_df.to_csv(tmp.name, index=False)
            s3.upload_file(tmp.name, S3_BUCKET, s3_key_out)

        audit_log(
            "drift_check_complete",
            {
                "mlflow_run_id": run.info.run_id,
                "drift_report_s3_path": f"s3://{S3_BUCKET}/{s3_key_out}",
                "max_psi": float(drift_df["psi"].max()),
                "avg_psi": float(drift_df["psi"].mean()),
            },
        )


default_args = {
    "owner": TEAM,
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DAG_ID = f"{TEAM}_{ENV}_{PROJECT_SLUG}_drift"

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=[PROJECT_SLUG, TEAM, ENV, "drift"],
) as dag:
    drift_task = PythonOperator(
        task_id="drift_check",
        python_callable=run_drift_check,
    )