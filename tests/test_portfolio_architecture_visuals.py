
"""
Tests for Phase 13M architecture visual specifications.

These tests protect topology, governance boundaries,
standalone intelligence, and public architecture claims.
"""

from copy import deepcopy

import pytest

from portfolio.architecture_visuals import (
    CUSTOMER_EXECUTION,
    INTELLIGENCE_BRANCHES,
    RUNTIME_BOUNDARIES,
    architecture_diagram,
    architecture_diagram_ids,
    architecture_visual_summary,
    validate_diagram,
)


def node_ids(diagram):
    return {
        item["id"]
        for item in diagram["nodes"]
    }


def edge_pairs(diagram):
    return {
        (item["source"], item["target"])
        for item in diagram["edges"]
    }


def test_three_architecture_diagrams_registered():
    assert architecture_diagram_ids() == (
        CUSTOMER_EXECUTION,
        INTELLIGENCE_BRANCHES,
        RUNTIME_BOUNDARIES,
    )


@pytest.mark.parametrize(
    "diagram_id",
    [
        CUSTOMER_EXECUTION,
        INTELLIGENCE_BRANCHES,
        RUNTIME_BOUNDARIES,
    ],
)
def test_registered_diagrams_validate(diagram_id):
    diagram = architecture_diagram(diagram_id)

    validate_diagram(diagram)

    assert diagram["title"]
    assert diagram["subtitle"]
    assert diagram["nodes"]


def test_architecture_summary_matches_governed_scope():
    assert architecture_visual_summary() == {
        "diagrams": 3,
        "production_systems": 6,
        "decision_inputs": 4,
        "standalone_systems": 2,
        "batch_jobs": 3,
    }


def test_customer_execution_preserves_order():
    diagram = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    assert edge_pairs(diagram) == {
        ("transactions", "customer_batch"),
        ("customer_batch", "decision_snapshot"),
        ("decision_snapshot", "signal_alignment"),
        ("signal_alignment", "action_orchestration"),
        ("action_orchestration", "priority_policy"),
        ("priority_policy", "canonical_decisions"),
    }


def test_customer_batch_has_no_automatic_delivery():
    diagram = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    assert "crm_delivery" not in node_ids(diagram)
    assert "scheduler" not in node_ids(diagram)

    assert any(
        "does not automatically" in note
        for note in diagram["notes"]
    )


def test_only_four_intelligence_inputs_feed_decisions():
    diagram = architecture_diagram(
        INTELLIGENCE_BRANCHES
    )

    incoming = {
        item["source"]
        for item in diagram["edges"]
        if item["target"] == "decision_engine"
    }

    assert incoming == {
        "reactivation",
        "leakage",
        "fcv",
        "recommendation",
    }


def test_standalone_systems_have_no_decision_edges():
    diagram = architecture_diagram(
        INTELLIGENCE_BRANCHES
    )

    for item in diagram["edges"]:
        assert item["source"] not in {
            "repeat_purchase",
            "demand_forecast",
        }


def test_recommendation_is_explicitly_conditional():
    diagram = architecture_diagram(
        INTELLIGENCE_BRANCHES
    )

    recommendation = next(
        item
        for item in diagram["nodes"]
        if item["id"] == "recommendation"
    )

    assert "Conditional" in recommendation["layer"]

    assert any(
        "conditional" in note.lower()
        for note in diagram["notes"]
    )


def test_external_scheduler_boundary_is_visible():
    diagram = architecture_diagram(
        RUNTIME_BOUNDARIES
    )

    assert (
        "external_scheduler",
        "job_interface",
    ) in edge_pairs(diagram)

    scheduler_edge = next(
        item
        for item in diagram["edges"]
        if item["source"] == "external_scheduler"
    )

    assert scheduler_edge["kind"] == "external"


def test_public_portfolio_is_separate_from_runtime():
    diagram = architecture_diagram(
        RUNTIME_BOUNDARIES
    )

    assert (
        "curated_artifacts",
        "streamlit",
    ) in edge_pairs(diagram)

    assert (
        "docker_runtime",
        "streamlit",
    ) not in edge_pairs(diagram)


def test_diagram_returns_detached_data():
    first = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    first["nodes"][0]["label"] = "CHANGED"

    second = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    assert second["nodes"][0]["label"] != "CHANGED"


def test_unknown_diagram_fails_closed():
    with pytest.raises(ValueError):
        architecture_diagram(
            "unknown_diagram"
        )


def test_missing_required_fields_fail_closed():
    with pytest.raises(RuntimeError):
        validate_diagram(
            {"nodes": []}
        )


def test_duplicate_node_ids_fail_closed():
    diagram = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    invalid = deepcopy(diagram)
    invalid["nodes"].append(
        deepcopy(invalid["nodes"][0])
    )

    with pytest.raises(RuntimeError):
        validate_diagram(invalid)


def test_unknown_edge_target_fails_closed():
    diagram = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    invalid = deepcopy(diagram)

    invalid["edges"][0]["target"] = (
        "nonexistent_node"
    )

    with pytest.raises(RuntimeError):
        validate_diagram(invalid)


def test_self_loops_fail_closed():
    diagram = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    invalid = deepcopy(diagram)

    invalid["edges"][0]["target"] = (
        invalid["edges"][0]["source"]
    )

    with pytest.raises(RuntimeError):
        validate_diagram(invalid)


def test_invalid_edge_kind_fails_closed():
    diagram = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    invalid = deepcopy(diagram)

    invalid["edges"][0]["kind"] = (
        "automatic_campaign"
    )

    with pytest.raises(RuntimeError):
        validate_diagram(invalid)


def test_duplicate_relationships_fail_closed():
    diagram = architecture_diagram(
        CUSTOMER_EXECUTION
    )

    invalid = deepcopy(diagram)

    invalid["edges"].append(
        deepcopy(invalid["edges"][0])
    )

    with pytest.raises(RuntimeError):
        validate_diagram(invalid)
