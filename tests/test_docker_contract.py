"""
Deployment-contract tests for Docker packaging.

These tests validate the repository-level container contract without
requiring Docker itself to be installed in the pytest environment.
"""

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DOCKERFILE = (
    PROJECT_ROOT
    /
    "Dockerfile"
)

DOCKERIGNORE = (
    PROJECT_ROOT
    /
    ".dockerignore"
)

REQUIREMENTS = (
    PROJECT_ROOT
    /
    "requirements.txt"
)

DEMAND_NOTEBOOK = (
    PROJECT_ROOT
    /
    "notebooks"
    /
    "07_ML_Demand_Forecasting.ipynb"
)


def read_text(
    path: Path,
) -> str:

    return path.read_text(
        encoding="utf-8"
    )


# ================================================================
# DOCKERFILE
# ================================================================


def test_dockerfile_exists():

    assert (
        DOCKERFILE.is_file()
    )


def test_dockerfile_uses_python_311_slim():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "FROM python:3.11-slim"
        in
        content
    )


def test_dockerfile_uses_app_workdir():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "WORKDIR /app"
        in
        content
    )


def test_dockerfile_installs_requirements():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "python -m pip install"
        in
        content
    )

    assert (
        "-r requirements.txt"
        in
        content
    )


def test_dockerfile_uses_resilient_pip_install():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "--retries 10"
        in
        content
    )

    assert (
        "--timeout 120"
        in
        content
    )


def test_dockerfile_does_not_upgrade_pip():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "pip install --upgrade pip"
        not in
        content
    )


def test_dockerfile_copies_application_source():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "COPY src ./src"
        in
        content
    )


def test_dockerfile_copies_governed_model_tree():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "COPY artifacts/models ./artifacts/models"
        in
        content
    )


def test_dockerfile_copies_governed_demand_notebook():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "notebooks/07_ML_Demand_Forecasting.ipynb"
        in
        content
    )


def test_dockerfile_runs_as_non_root_user():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "USER app"
        in
        content
    )


def test_dockerfile_defaults_to_health_cli():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        'CMD ["python", "-m", "src.cli", "health"]'
        in
        content
    )


# ================================================================
# DOCKERIGNORE
# ================================================================


def test_dockerignore_exists():

    assert (
        DOCKERIGNORE.is_file()
    )


def test_dockerignore_excludes_git_metadata():

    content = (
        read_text(
            DOCKERIGNORE
        )
    )

    assert (
        ".git"
        in
        content
    )


def test_dockerignore_excludes_tests():

    content = (
        read_text(
            DOCKERIGNORE
        )
    )

    assert (
        "tests/"
        in
        content
    )


def test_dockerignore_excludes_transaction_data():

    content = (
        read_text(
            DOCKERIGNORE
        )
    )

    assert (
        "*.parquet"
        in
        content
    )

    assert (
        "*.xlsx"
        in
        content
    )


def test_dockerignore_excludes_notebooks_by_default():

    content = (
        read_text(
            DOCKERIGNORE
        )
    )

    assert (
        "*.ipynb"
        in
        content
    )


def test_dockerignore_allows_governed_demand_notebook():

    content = (
        read_text(
            DOCKERIGNORE
        )
    )

    assert (
        "!notebooks/07_ML_Demand_Forecasting.ipynb"
        in
        content
    )


def test_dockerignore_excludes_model_binaries():

    content = (
        read_text(
            DOCKERIGNORE
        )
    )

    assert (
        "artifacts/models/**/*.joblib"
        in
        content
    )


def test_dockerignore_does_not_exclude_json_manifests():

    content = (
        read_text(
            DOCKERIGNORE
        )
    )

    assert (
        "artifacts/models/**/*.json"
        not in
        content
    )


def test_dockerignore_excludes_research_temp_files():

    content = (
        read_text(
            DOCKERIGNORE
        )
    )

    assert (
        "decision_engine_research_temp.txt"
        in
        content
    )

    assert (
        "recommendation_research_temp.py"
        in
        content
    )


# ================================================================
# GOVERNED SOURCE
# ================================================================


def test_governed_demand_notebook_exists():

    assert (
        DEMAND_NOTEBOOK.is_file()
    )


# ================================================================
# REQUIREMENTS
# ================================================================


def test_requirements_include_parquet_engine():

    content = (
        read_text(
            REQUIREMENTS
        )
    )

    assert (
        "pyarrow"
        in
        content
    )


def test_pandas_repository_contract_remains_below_three():

    content = (
        read_text(
            REQUIREMENTS
        )
    )

    assert (
        "pandas>=2.1,<3.0"
        in
        content
    )


def test_sklearn_is_pinned_to_governed_artifact_version():

    content = (
        read_text(
            REQUIREMENTS
        )
    )

    assert (
        "scikit-learn==1.8.0"
        in
        content
    )
def test_dockerfile_copies_scheduler_wrapper():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "scripts/run_scheduled_job.sh"
        in
        content
    )


def test_dockerfile_makes_scheduler_wrapper_executable():

    content = (
        read_text(
            DOCKERFILE
        )
    )

    assert (
        "chmod 0755 /app/scripts/run_scheduled_job.sh"
        in
        content
    )