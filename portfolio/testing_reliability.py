"""
Testing & Reliability evidence model for the public portfolio.

This module presents verified testing evidence.

Important distinction:
- automated_tests_passing is the executed pytest regression count;
- explicit_test_functions is a source-code inventory count;
- parametrisation and collection expansion can make the executed case count
  larger than the number of explicit test function definitions.

This module does not execute the test suite.
"""

from __future__ import annotations

from typing import Any


TEST_MODULE_COUNT = 91
EXPLICIT_TEST_FUNCTION_COUNT = 1644
KNOWN_WARNING_COUNT = 1


RELIABILITY_DOMAINS = (
    {
        "domain": "Temporal & Feature Integrity",
        "purpose": (
            "Protect scoring boundaries, chronological feature construction, "
            "historical completeness, and future-data separation."
        ),
        "evidence": (
            "Feature tests across Reactivation, Leakage, FCV, "
            "Repeat Purchase, Recommendation, and Demand"
        ),
        "examples": (
            "chronological demand features",
            "invalid date rejection",
            "missing-week failure",
            "model-feature contract checks",
        ),
    },
    {
        "domain": "Model & Scoring Contracts",
        "purpose": (
            "Verify scorer schemas, governed manifests, model-feature parity, "
            "runtime availability, and registry behaviour."
        ),
        "evidence": (
            "Scorer, manifest, model-feature, score-registry, "
            "and runtime-integration tests"
        ),
        "examples": (
            "six-system registry",
            "score availability",
            "manifest validation",
            "end-to-end scoring contracts",
        ),
    },
    {
        "domain": "Decision Engine",
        "purpose": (
            "Verify lifecycle state, signal alignment, deterministic action "
            "orchestration, priority policy, and final decision contracts."
        ),
        "evidence": (
            "Decision snapshot, signal alignment, orchestration, priority, "
            "specification, and integration tests"
        ),
        "examples": (
            "Protect precedence",
            "Dormant Reactivate logic",
            "Long Dormant fail-safe state",
            "action-relative prioritisation",
        ),
    },
    {
        "domain": "Delivery & Audit",
        "purpose": (
            "Protect CRM delivery eligibility, deterministic payloads, "
            "exports, audit history, and duplicate/collision handling."
        ),
        "evidence": (
            "Delivery eligibility, payload, builder, export, audit, "
            "and decision-to-delivery integration tests"
        ),
        "examples": (
            "duplicate delivery ID rejection",
            "schema drift rejection",
            "audit collision rejection",
            "non-actionable contact suppression",
        ),
    },
    {
        "domain": "Experiment Governance",
        "purpose": (
            "Protect experiment provenance, validation, immutable run IDs, "
            "controlled storage, and experiment-policy preservation."
        ),
        "evidence": (
            "Experiment specification, provenance, record, store, "
            "validation, and logger integration tests"
        ),
        "examples": (
            "duplicate run rejection",
            "invalid provenance rejection",
            "schema drift rejection",
            "experiment requirement preservation",
        ),
    },
    {
        "domain": "Monitoring & Runtime",
        "purpose": (
            "Verify health evaluation, alerts, deployment boundaries, "
            "Docker contracts, CLI behaviour, and scheduled job wrappers."
        ),
        "evidence": (
            "Monitoring, deployment runtime, Docker, CLI, scheduler, "
            "and batch orchestration tests"
        ),
        "examples": (
            "non-root Docker contract",
            "health command",
            "month-end batch boundary",
            "scheduled-job fail-closed behaviour",
        ),
    },
    {
        "domain": "Portfolio Application",
        "purpose": (
            "Verify the curated public data layer, dashboards, governed demo, "
            "model explorer, governance dashboard, and architecture explorer."
        ),
        "evidence": (
            "Portfolio generator, app shell, components, Command Centre, "
            "customer explorer, decision demo, governance, model, "
            "and architecture tests"
        ),
        "examples": (
            "raw identifier exclusion",
            "public evidence contracts",
            "no model-binary loading",
            "no duplicated decision policy",
        ),
    },
)


FAIL_CLOSED_CONTROLS = (
    {
        "control": "Required schema",
        "behaviour": (
            "Missing required production fields raise rather than silently "
            "continuing with incomplete data."
        ),
    },
    {
        "control": "Duplicate identity",
        "behaviour": (
            "Duplicate customers, delivery IDs, experiment run IDs, and "
            "historical audit collisions are rejected."
        ),
    },
    {
        "control": "Unknown governed state",
        "behaviour": (
            "Unknown customer states, routes, dispositions, contact policies, "
            "score IDs, and governance values fail closed."
        ),
    },
    {
        "control": "Temporal boundary",
        "behaviour": (
            "Invalid dates, non-chronological structures, unknown missing "
            "weeks, and invalid batch boundaries are rejected."
        ),
    },
    {
        "control": "Persistent-store integrity",
        "behaviour": (
            "Invalid JSON, empty records, schema drift, duplicate historical "
            "IDs, and directory/path misuse fail closed."
        ),
    },
    {
        "control": "Operational eligibility",
        "behaviour": (
            "Non-actionable customers cannot silently receive model-led "
            "treatment contact."
        ),
    },
)


INTEGRATION_LAYERS = (
    "Predictive pipeline integration",
    "Deterministic pipeline integration",
    "Scorer → Decision Engine integration",
    "Decision Engine end-to-end integration",
    "Decision → Delivery integration",
    "Experiment logger integration",
    "Monitoring integration",
    "Score runtime integration",
    "CLI → Deployment integration",
)


def validate_testing_evidence(
    summary: dict[str, Any],
) -> None:
    """
    Validate the public regression proof point.
    """

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError(
            "Portfolio summary must be a dictionary."
        )

    evidence = summary.get(
        "verified_evidence"
    )

    if not isinstance(
        evidence,
        dict,
    ):
        raise RuntimeError(
            "Testing & Reliability requires verified portfolio evidence."
        )

    executed = evidence.get(
        "automated_tests_passing"
    )

    if not isinstance(
        executed,
        int,
    ):
        raise RuntimeError(
            "Verified automated test count is missing."
        )

    if executed < EXPLICIT_TEST_FUNCTION_COUNT:
        raise RuntimeError(
            (
                "Executed pytest cases cannot be lower than the "
                "verified explicit test-function inventory."
            )
        )


def reliability_summary(
    summary: dict[str, Any],
) -> dict[str, int]:
    """
    Return the verified testing summary.
    """

    validate_testing_evidence(
        summary
    )

    executed = summary[
        "verified_evidence"
    ][
        "automated_tests_passing"
    ]

    return {
        "executed_regression_cases":
            executed,

        "explicit_test_functions":
            EXPLICIT_TEST_FUNCTION_COUNT,

        "collection_expansion":
            (
                executed
                -
                EXPLICIT_TEST_FUNCTION_COUNT
            ),

        "test_modules":
            TEST_MODULE_COUNT,

        "known_warnings":
            KNOWN_WARNING_COUNT,

        "reliability_domains":
            len(
                RELIABILITY_DOMAINS
            ),

        "integration_layers":
            len(
                INTEGRATION_LAYERS
            ),

        "fail_closed_controls":
            len(
                FAIL_CLOSED_CONTROLS
            ),
    }


def reliability_domain_rows() -> list[dict[str, Any]]:
    """
    Return reliability-domain evidence.
    """

    return [
        dict(
            row
        )
        for row in RELIABILITY_DOMAINS
    ]


def fail_closed_rows() -> list[dict[str, str]]:
    """
    Return fail-closed control evidence.
    """

    return [
        dict(
            row
        )
        for row in FAIL_CLOSED_CONTROLS
    ]


def integration_layer_rows() -> list[str]:
    """
    Return cross-layer integration evidence.
    """

    return list(
        INTEGRATION_LAYERS
    )


def executed_case_expansion(
    summary: dict[str, Any],
) -> int:
    """
    Return collected/executed cases beyond explicit test functions.

    This is evidence of pytest collection expansion, commonly caused by
    parametrisation. It is not presented as additional handwritten tests.
    """

    validate_testing_evidence(
        summary
    )

    return (
        summary[
            "verified_evidence"
        ][
            "automated_tests_passing"
        ]
        -
        EXPLICIT_TEST_FUNCTION_COUNT
    )