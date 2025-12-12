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
- Test key: `{{ cookiecutter.test_key }}`
- Predictions prefix: `{{ cookiecutter.predictions_prefix }}`
- Drift prefix: `{{ cookiecutter.drift_prefix }}`

## Next Steps

1. Edit `project.json` if needed (team, tags, etc.).
2. Customize or replace the example DAGs in `airflow/dags/`.
3. Commit + push your repo.
4. Use the shared Jenkins job:

   - **Deploy DAGs** → copies DAGs to Airflow
   - **Trigger DAG** → (optional) run your DAG on demand