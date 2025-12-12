# {{ cookiecutter.project_name }} ({{ cookiecutter.team }})

Spoke repo generated from the ML Platform cookiecutter.

## Contract with Platform

- `project.json` **must** stay at repo root and be kept up to date.
- All DAGs must live under `airflow/dags/`.
- The platform’s Jenkins job **Deploy DAGs** reads this repo and
  syncs DAGs into the shared Airflow instance.

## Data & MLflow

Defaults (can be overridden via environment variables in Airflow):

- S3 bucket: `{{ cookiecutter.s3_bucket }}`
- Train key: `{{ cookiecutter.train_s3_key }}`
- Test key: `{{ cookiecutter.test_s3_key }}`
- Batch inference input key: `{{ cookiecutter.batch_inference_s3_key }}`
- Drift reference key: `{{ cookiecutter.drift_reference_s3_key }}`
- Drift current key: `{{ cookiecutter.drift_current_s3_key }}`
- Batch output prefix: `{{ cookiecutter.batch_output_prefix }}`
- Drift output prefix: `{{ cookiecutter.drift_output_prefix }}`

Env var overrides used by the example DAGs:

- `TEAM_S3_BUCKET`
- `TEAM_S3_TRAIN_KEY`
- `TEAM_S3_TEST_KEY`
- `TEAM_S3_BATCH_INFERENCE_KEY`
- `TEAM_S3_DRIFT_REFERENCE_KEY`
- `TEAM_S3_DRIFT_CURRENT_KEY`
- `TEAM_BATCH_OUTPUT_PREFIX`
- `TEAM_DRIFT_OUTPUT_PREFIX`

## Next Steps

1. Edit `project.json` if needed (team, tags, etc.).
2. Customize or replace the example DAGs in `airflow/dags/`.
3. Commit + push your repo.
4. Use the shared Jenkins job:

   - **Deploy DAGs** → copies DAGs to Airflow
   - **Trigger DAG** → (optional) run your DAG on demand