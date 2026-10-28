
"""
Architecture Visuals evidence model.

Phase 13M — AI Revenue Decision Engine.

Defines governed, presentation-only architecture diagrams.

The module does not:
- execute production workloads;
- load private model artifacts;
- recreate scoring or Decision Engine policy;
- claim automatic CRM contact;
- claim an in-repository scheduler;
- claim cloud infrastructure.

Diagram edges distinguish actual execution relationships from
conceptual, governance, and external operational boundaries.
"""

from __future__ import annotations

from typing import Any

from portfolio.production_architecture import (
    batch_job_rows,
)


# ============================================================
# DIAGRAM IDENTIFIERS
# ============================================================

CUSTOMER_EXECUTION = "customer_execution"
INTELLIGENCE_BRANCHES = "intelligence_branches"
RUNTIME_BOUNDARIES = "runtime_boundaries"

DIAGRAM_IDS = (
    CUSTOMER_EXECUTION,
    INTELLIGENCE_BRANCHES,
    RUNTIME_BOUNDARIES,
)

ALLOWED_EDGE_KINDS = frozenset(
    {
        "execution",
        "input",
        "conceptual",
        "external",
    }
)


# ============================================================
# NODE AND EDGE HELPERS
# ============================================================

def node(
    node_id: str,
    label: str,
    layer: str,
    description: str,
) -> dict[str, str]:
    """Define a presentation-only architecture node."""

    return {
        "id": node_id,
        "label": label,
        "layer": layer,
        "description": description,
    }


def edge(
    source: str,
    target: str,
    kind: str,
    label: str,
) -> dict[str, str]:
    """Define an explicitly classified relationship."""

    return {
        "source": source,
        "target": target,
        "kind": kind,
        "label": label,
    }


# ============================================================
# VISUAL A — CUSTOMER DECISION EXECUTION
# ============================================================

CUSTOMER_EXECUTION_NODES = (
    node(
        "transactions",
        "Canonical Transactions",
        "Data",
        "Validated retail transaction input.",
    ),
    node(
        "customer_batch",
        "Customer Decision Batch",
        "Deployment",
        "Governed month-end feature construction and scoring.",
    ),
    node(
        "decision_snapshot",
        "Decision Snapshot",
        "Decision Engine",
        "Construct the customer lifecycle snapshot.",
    ),
    node(
        "signal_alignment",
        "Signal Alignment",
        "Decision Engine",
        "Align available operationally valid model signals.",
    ),
    node(
        "action_orchestration",
        "Action Orchestration",
        "Decision Engine",
        "Assign permitted actions under deterministic policy.",
    ),
    node(
        "priority_policy",
        "Priority & Contact Policy",
        "Decision Engine",
        "Apply action-relative priority and contact rules.",
    ),
    node(
        "canonical_decisions",
        "Canonical Customer Decisions",
        "Output",
        "Validated governed decision output.",
    ),
)

CUSTOMER_EXECUTION_EDGES = (
    edge(
        "transactions",
        "customer_batch",
        "input",
        "Validated transaction input",
    ),
    edge(
        "customer_batch",
        "decision_snapshot",
        "execution",
        "Decision Engine invocation",
    ),
    edge(
        "decision_snapshot",
        "signal_alignment",
        "execution",
        "Snapshot population and scored inputs",
    ),
    edge(
        "signal_alignment",
        "action_orchestration",
        "execution",
        "Operationally valid aligned signals",
    ),
    edge(
        "action_orchestration",
        "priority_policy",
        "execution",
        "Governed customer actions",
    ),
    edge(
        "priority_policy",
        "canonical_decisions",
        "execution",
        "Validated canonical output",
    ),
)


# ============================================================
# VISUAL B — INTELLIGENCE SYSTEM SEPARATION
# ============================================================

INTELLIGENCE_NODES = (
    node(
        "reactivation",
        "Reactivation Propensity",
        "Decision Input",
        "Governed reactivation signal.",
    ),
    node(
        "leakage",
        "Revenue Leakage Risk",
        "Decision Input",
        "Governed leakage-risk signal and economic context.",
    ),
    node(
        "fcv",
        "Future Customer Value",
        "Decision Input",
        "Governed future-value prediction.",
    ),
    node(
        "recommendation",
        "Next Best Product",
        "Conditional Decision Input",
        "Optional governed recommendation-episode scoring.",
    ),
    node(
        "decision_engine",
        "Customer Decision Engine",
        "Decision Engine",
        "Consumes eligible governed intelligence signals.",
    ),
    node(
        "repeat_purchase",
        "Repeat Purchase Propensity",
        "Standalone",
        "Separate production scoring and batch contract.",
    ),
    node(
        "demand_forecast",
        "Weekly Demand Forecast",
        "Standalone",
        "Separate weekly planning and forecast contract.",
    ),
)

INTELLIGENCE_EDGES = (
    edge(
        "reactivation",
        "decision_engine",
        "input",
        "Eligible reactivation signals",
    ),
    edge(
        "leakage",
        "decision_engine",
        "input",
        "Eligible leakage signals",
    ),
    edge(
        "fcv",
        "decision_engine",
        "input",
        "Eligible future-value signals",
    ),
    edge(
        "recommendation",
        "decision_engine",
        "input",
        "Conditional recommendation signals",
    ),
)


# ============================================================
# VISUAL C — RUNTIME AND PUBLIC BOUNDARIES
# ============================================================

RUNTIME_NODES = (
    node(
        "external_scheduler",
        "External Scheduler",
        "External Boundary",
        "Scheduler implementation exists outside the repository.",
    ),
    node(
        "job_interface",
        "Governed Job Interface",
        "Runtime",
        "Explicit scheduled-job invocation and boundary validation.",
    ),
    node(
        "docker_runtime",
        "Docker Runtime",
        "Runtime",
        "Non-root Python 3.11 production execution environment.",
    ),
    node(
        "customer_job",
        "Customer Decision Batch",
        "Batch Job",
        "Customer-level governed decision orchestration.",
    ),
    node(
        "repeat_job",
        "Repeat Purchase Batch",
        "Batch Job",
        "Standalone repeat-purchase scoring.",
    ),
    node(
        "demand_job",
        "Demand Forecast Batch",
        "Batch Job",
        "Standalone weekly demand forecasting.",
    ),
    node(
        "curated_artifacts",
        "Curated Public Artifacts",
        "Public Evidence",
        "Anonymised, governed portfolio output files.",
    ),
    node(
        "streamlit",
        "Streamlit Portfolio",
        "Public Application",
        "Read-only public portfolio demonstration.",
    ),
)

RUNTIME_EDGES = (
    edge(
        "external_scheduler",
        "job_interface",
        "external",
        "External invocation contract",
    ),
    edge(
        "job_interface",
        "docker_runtime",
        "conceptual",
        "Governed one-shot execution boundary",
    ),
    edge(
        "docker_runtime",
        "customer_job",
        "conceptual",
        "Available production job",
    ),
    edge(
        "docker_runtime",
        "repeat_job",
        "conceptual",
        "Available production job",
    ),
    edge(
        "docker_runtime",
        "demand_job",
        "conceptual",
        "Available production job",
    ),
    edge(
        "curated_artifacts",
        "streamlit",
        "input",
        "Curated public data only",
    ),
)


# ============================================================
# DIAGRAM REGISTRY
# ============================================================

DIAGRAMS = {
    CUSTOMER_EXECUTION: {
        "title": "Customer Decision Execution",
        "subtitle": (
            "From canonical transactions to governed "
            "customer decisions."
        ),
        "nodes": CUSTOMER_EXECUTION_NODES,
        "edges": CUSTOMER_EXECUTION_EDGES,
        "notes": (
            "The customer batch does not automatically perform "
            "CRM delivery, append audit records, or schedule jobs.",
            "Decision Engine stages preserve the customer population.",
        ),
    },
    INTELLIGENCE_BRANCHES: {
        "title": "Intelligence System Boundaries",
        "subtitle": (
            "Six governed production systems; four eligible "
            "customer Decision Engine inputs."
        ),
        "nodes": INTELLIGENCE_NODES,
        "edges": INTELLIGENCE_EDGES,
        "notes": (
            "Recommendation scoring is conditional on observed episodes.",
            "Repeat Purchase and Demand Forecast remain standalone.",
            "Runtime availability does not imply unrestricted use.",
        ),
    },
    RUNTIME_BOUNDARIES: {
        "title": "Runtime & Public Portfolio Boundaries",
        "subtitle": (
            "Production batch execution is distinct from the "
            "curated public demonstration."
        ),
        "nodes": RUNTIME_NODES,
        "edges": RUNTIME_EDGES,
        "notes": (
            "The scheduler implementation is external.",
            "No managed cloud orchestration is claimed.",
            "Public Streamlit does not load private model binaries.",
            "Conceptual edges are not claims of direct function calls.",
        ),
    },
}


# ============================================================
# VALIDATION
# ============================================================

def validate_diagram(
    diagram: dict[str, Any],
) -> None:
    """Fail closed on invalid architecture specifications."""

    if not isinstance(diagram, dict):
        raise TypeError(
            "Architecture diagram must be a dictionary."
        )

    required = {
        "title",
        "subtitle",
        "nodes",
        "edges",
        "notes",
    }

    if not required.issubset(diagram):
        raise RuntimeError(
            "Architecture diagram is missing required fields."
        )

    nodes = diagram["nodes"]
    edges = diagram["edges"]

    if not isinstance(nodes, (list, tuple)) or not nodes:
        raise RuntimeError(
            "Architecture diagram requires nodes."
        )

    if not isinstance(edges, (list, tuple)):
        raise RuntimeError(
            "Architecture diagram requires valid edges."
        )

    node_ids = set()

    for item in nodes:
        if not isinstance(item, dict):
            raise RuntimeError(
                "Architecture node must be a dictionary."
            )

        for field in ("id", "label", "layer", "description"):
            if not isinstance(item.get(field), str):
                raise RuntimeError(
                    "Architecture node has an invalid field."
                )

            if not item[field].strip():
                raise RuntimeError(
                    "Architecture node fields cannot be empty."
                )

        identifier = item["id"]

        if identifier in node_ids:
            raise RuntimeError(
                "Duplicate architecture node ID."
            )

        node_ids.add(identifier)

    relationships = set()

    for item in edges:
        if not isinstance(item, dict):
            raise RuntimeError(
                "Architecture edge must be a dictionary."
            )

        source = item.get("source")
        target = item.get("target")
        kind = item.get("kind")
        label = item.get("label")

        if source not in node_ids or target not in node_ids:
            raise RuntimeError(
                "Architecture edge references an unknown node."
            )

        if source == target:
            raise RuntimeError(
                "Architecture self-loop is not permitted."
            )

        if kind not in ALLOWED_EDGE_KINDS:
            raise RuntimeError(
                "Architecture edge has an invalid relationship kind."
            )

        if not isinstance(label, str) or not label.strip():
            raise RuntimeError(
                "Architecture edge requires a label."
            )

        relationship = (source, target, kind)

        if relationship in relationships:
            raise RuntimeError(
                "Duplicate architecture relationship."
            )

        relationships.add(relationship)


def architecture_diagram(
    diagram_id: str,
) -> dict[str, Any]:
    """Return a validated, detached diagram specification."""

    if diagram_id not in DIAGRAMS:
        raise ValueError(
            "Unknown architecture diagram."
        )

    original = DIAGRAMS[diagram_id]

    result = {
        "title": original["title"],
        "subtitle": original["subtitle"],
        "nodes": [
            dict(item)
            for item in original["nodes"]
        ],
        "edges": [
            dict(item)
            for item in original["edges"]
        ],
        "notes": list(original["notes"]),
    }

    validate_diagram(result)

    return result


def architecture_diagram_ids() -> tuple[str, ...]:
    """Return the three governed visualization identifiers."""

    return DIAGRAM_IDS


def architecture_visual_summary() -> dict[str, int]:
    """Return structural counts, not unverified coverage claims."""

    diagrams = [
        architecture_diagram(identifier)
        for identifier in DIAGRAM_IDS
    ]

    jobs = batch_job_rows()

    if len(jobs) != 3:
        raise RuntimeError(
            "Architecture visuals require three governed batch jobs."
        )

    if sum(bool(job["decision_engine"]) for job in jobs) != 1:
        raise RuntimeError(
            "Only the customer batch may enter decision orchestration."
        )

    return {
        "diagrams": len(diagrams),
        "production_systems": 6,
        "decision_inputs": len(INTELLIGENCE_EDGES),
        "standalone_systems": 2,
        "batch_jobs": len(jobs),
    }
