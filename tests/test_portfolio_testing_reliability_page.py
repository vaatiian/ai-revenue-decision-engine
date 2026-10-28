
import ast
from pathlib import Path

import pytest

from portfolio.testing_reliability import (
    reliability_summary,
)
from portfolio.testing_reliability_page import (
    render_testing_reliability,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAGE_PATH = (
    PROJECT_ROOT
    / "portfolio"
    / "testing_reliability_page.py"
)


def source():
    return PAGE_PATH.read_text(encoding="utf-8")


def test_page_file_parses():
    ast.parse(source())


def test_page_uses_curated_summary():
    assert (
        'bundle[\n        "summary"\n    ]'
        in source()
    )


def test_page_validates_evidence_before_rendering():
    tree = ast.parse(source())

    render = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "render_testing_reliability"
    )

    calls = [
        node.func.id
        for node in ast.walk(render)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    ]

    assert "validate_testing_evidence" in calls
    assert "reliability_summary" in calls


@pytest.mark.parametrize(
    "heading",
    [
        "Testing & Reliability",
        "Reliability Domains",
        "Fail-Closed Design",
        "Integration Coverage",
        "Regression Discipline",
    ],
)
def test_required_sections_exist(heading):
    assert heading in source()


def test_page_preserves_actual_regression_command():
    assert "python -m pytest -q" in source()


def test_page_separates_executed_and_explicit_tests():
    content = source()

    assert "executed_regression_cases" in content
    assert "explicit_test_functions" in content
    assert "collection_expansion" in content


def test_page_discloses_known_warning():
    content = source()

    assert "non-fatal pandas date-parsing warning" in content
    assert "Repeat Purchase" in content


def test_page_preserves_claims_boundary():
    assert (
        "not proof of causal "
        in source()
    )


def test_page_does_not_load_private_artifacts():
    forbidden = [
        "joblib.load",
        "pickle.load",
        "cleaned_online_retail.parquet",
        "artifacts/models/",
    ]

    for token in forbidden:
        assert token not in source()


def test_invalid_evidence_rejected_before_rendering():
    class StreamlitShouldNotBeCalled:
        def __getattr__(self, name):
            raise AssertionError(
                f"Unexpected Streamlit call: {name}"
            )

    with pytest.raises(RuntimeError):
        render_testing_reliability(
            StreamlitShouldNotBeCalled(),
            {"summary": {}},
        )


def test_reliability_summary_rejects_missing_evidence():
    with pytest.raises(RuntimeError):
        reliability_summary({})
