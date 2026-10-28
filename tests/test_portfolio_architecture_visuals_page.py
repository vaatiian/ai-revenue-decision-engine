
"""
Phase 13M — Architecture Visuals page tests.

Protects the visual evidence contract, rendering structure,
architecture boundaries, and public-data safety.
"""

import ast
from pathlib import Path

import pytest

from portfolio.architecture_visuals import (
    architecture_diagram,
)

from portfolio.architecture_visuals_page import (
    EDGE_COLORS,
    EDGE_LABELS,
    diagram_node_rows,
    diagram_relationship_rows,
    render_architecture_visuals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAGE_PATH = (
    PROJECT_ROOT
    / "portfolio"
    / "architecture_visuals_page.py"
)


def source():
    return PAGE_PATH.read_text(encoding="utf-8")


def test_page_file_parses():
    ast.parse(source())


def test_page_uses_curated_summary():
    assert 'summary = bundle["summary"]' in source()


def test_page_has_three_architecture_views():
    content = source()

    assert "Customer Decision Execution" in content
    assert "Intelligence System Boundaries" in content
    assert "Runtime & Public Boundaries" in content


def test_page_uses_governed_diagram_registry():
    content = source()

    assert "architecture_diagram_ids()" in content
    assert "architecture_diagram(" in content


@pytest.mark.parametrize(
    "kind",
    [
        "execution",
        "input",
        "conceptual",
        "external",
    ],
)
def test_relationship_kinds_have_visual_labels(kind):
    assert kind in EDGE_LABELS
    assert kind in EDGE_COLORS
    assert EDGE_LABELS[kind]
    assert EDGE_COLORS[kind]


def test_customer_execution_renderer_exists():
    assert "def render_customer_execution(" in source()


def test_intelligence_branch_renderer_exists():
    assert "def render_intelligence_branches(" in source()


def test_runtime_boundary_renderer_exists():
    assert "def render_runtime_boundaries(" in source()


def test_node_table_preserves_structure():
    diagram = architecture_diagram(
        "customer_execution"
    )

    frame = diagram_node_rows(diagram)

    assert len(frame) == 7

    assert list(frame.columns) == [
        "Node",
        "Layer",
        "Purpose",
    ]


def test_relationship_table_preserves_structure():
    diagram = architecture_diagram(
        "customer_execution"
    )

    frame = diagram_relationship_rows(diagram)

    assert len(frame) == 6

    assert list(frame.columns) == [
        "From",
        "To",
        "Relationship",
        "Meaning",
    ]


def test_intelligence_relationship_table_has_four_inputs():
    diagram = architecture_diagram(
        "intelligence_branches"
    )

    frame = diagram_relationship_rows(diagram)

    assert len(frame) == 4

    assert set(frame["To"]) == {
        "Customer Decision Engine"
    }


def test_runtime_table_preserves_external_boundary():
    diagram = architecture_diagram(
        "runtime_boundaries"
    )

    frame = diagram_relationship_rows(diagram)

    scheduler = frame[
        frame["From"] == "External Scheduler"
    ]

    assert len(scheduler) == 1

    assert scheduler.iloc[0]["Relationship"] == (
        "External boundary"
    )


def test_page_contains_explicit_claims_boundary():
    content = source()

    assert "automatic CRM contact" in content
    assert "realised revenue impact" in content


def test_page_does_not_load_private_model_artifacts():
    forbidden = [
        "joblib.load",
        "pickle.load",
        "cleaned_online_retail.parquet",
        "artifacts/models/",
    ]

    for token in forbidden:
        assert token not in source()


def test_page_does_not_claim_unimplemented_infrastructure():
    forbidden = [
        "AWS",
        "Azure",
        "Kubernetes",
        "Airflow",
        "GitHub Actions",
    ]

    for token in forbidden:
        assert token not in source()


def test_invalid_architecture_selection_fails_closed():
    class FakeStreamlit:
        def __init__(self):
            self.sidebar = None

        def __getattr__(self, name):
            raise AssertionError(
                f"Unexpected Streamlit call: {name}"
            )

    with pytest.raises(KeyError):
        render_architecture_visuals(
            FakeStreamlit(),
            {},
        )
