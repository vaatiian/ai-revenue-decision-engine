import ast
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[
    1
]


def source():
    return (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "production_architecture_page.py"
    ).read_text(
        encoding="utf-8"
    )


def test_page_file_parses():
    ast.parse(
        source()
    )


def test_page_uses_curated_summary():
    assert (
        'bundle[\n        "summary"\n    ]'
        in
        source()
    )


def test_page_contains_customer_pipeline():
    assert (
        "Customer Decision Pipeline"
        in
        source()
    )


def test_page_contains_deployment_jobs():
    assert (
        "Deployment Jobs"
        in
        source()
    )


def test_page_contains_runtime_control_plane():
    assert (
        "Runtime & Control Plane"
        in
        source()
    )


def test_page_explicitly_preserves_external_scheduler_boundary():
    assert (
        "the repository does not claim an in-repo scheduler implementation"
        in
        source()
    )


def test_page_does_not_load_models():
    forbidden = [
        "joblib.load",
        "pickle.load",
        "cleaned_online_retail.parquet",
        "artifacts/models/",
    ]

    for token in forbidden:

        assert token not in source()


def test_page_does_not_claim_cloud_infrastructure():
    forbidden = [
        "AWS",
        "Azure",
        "GCP",
        "Kubernetes",
        "Airflow",
        "GitHub Actions",
    ]

    for token in forbidden:

        assert token not in source()