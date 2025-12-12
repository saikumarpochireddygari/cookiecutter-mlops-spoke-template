# DAGs for {{ cookiecutter.project_name }}

Put your Airflow DAG Python files in this folder.

Suggested patterns (you can rename):

- `{{ cookiecutter.team }}_<env>_{{ cookiecutter.project_slug }}_pipeline`
- `{{ cookiecutter.team }}_<env>_{{ cookiecutter.project_slug }}_batch_inference`
- `{{ cookiecutter.team }}_<env>_{{ cookiecutter.project_slug }}_drift`

The platform Jenkins job:

1. Reads `project.json`
2. Validates this folder & DAG names
3. Copies your DAGs into the central Airflow instance under
   `/opt/airflow/dags/<env>/<project_name>/`.