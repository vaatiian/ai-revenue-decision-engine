"""
Production Architecture model for the public portfolio application.

This module describes the verified production topology already implemented
in the repository.

It does not:
- execute production jobs;
- load model binaries;
- rescore customers;
- infer infrastructure that does not exist;
- claim an external scheduler implementation that is not in this repository.
"""

from __future__ import annotations

from typing import Any


CUSTOMER_PIPELINE = (
    {
        "stage": "Canonical Transactions",
        "layer": "Data",
        "implementation": "Canonical retail transaction contract",
        "source": "cleaned_online_retail.parquet / production transaction input",
        "purpose": (
            "Preserve governed transaction semantics including customers, "
            "invoices, products, returns, cancellations, and revenue fields."
        ),
    },
    {
        "stage": "Feature Pipelines",
        "layer": "Features",
        "implementation": "Production feature builders",
        "source": "src/features",
        "purpose": (
            "Transform only information available at the scoring boundary "
            "into model-ready features."
        ),
    },
    {
        "stage": "Production Scorers",
        "layer": "Intelligence",
        "implementation": "Six governed scorer interfaces",
        "source": "src/models/*_scorer.py",
        "purpose": (
            "Produce governed model or deterministic intelligence outputs "
            "through production scoring contracts."
        ),
    },
    {
        "stage": "Score Registry",
        "layer": "Registry",
        "implementation": "Score registry + runtime availability",
        "source": "src/models/score_registry.py",
        "purpose": (
            "Centralise production scoring-system registration and "
            "runtime availability."
        ),
    },
    {
        "stage": "Decision Snapshot",
        "layer": "Decision Engine",
        "implementation": "Customer lifecycle snapshot",
        "source": "src/decision_engine/snapshot.py",
        "purpose": (
            "Construct the governed customer state required by the "
            "customer Decision Engine."
        ),
    },
    {
        "stage": "Signal Alignment",
        "layer": "Decision Engine",
        "implementation": "Temporal and operational signal alignment",
        "source": "src/decision_engine/signal_alignment.py",
        "purpose": (
            "Align only operationally valid intelligence signals to the "
            "customer decision boundary."
        ),
    },
    {
        "stage": "Action Orchestration",
        "layer": "Decision Engine",
        "implementation": "Deterministic governed action hierarchy",
        "source": "src/decision_engine/action_orchestration.py",
        "purpose": (
            "Assign the permitted customer action from lifecycle state "
            "and valid signal availability."
        ),
    },
    {
        "stage": "Priority & Contact Policy",
        "layer": "Decision Engine",
        "implementation": "Action-relative priority policy",
        "source": "src/decision_engine/priority_policy.py",
        "purpose": (
            "Assign action priority, contact intensity, recommendation "
            "channel, decision reason, business action, and experiment policy."
        ),
    },
    {
        "stage": "CRM Delivery",
        "layer": "Delivery",
        "implementation": "Governed delivery payload and eligibility layer",
        "source": "src/delivery",
        "purpose": (
            "Convert eligible governed decisions into auditable delivery "
            "payloads without creating new decisions."
        ),
    },
    {
        "stage": "Experiment Logging",
        "layer": "Experimentation",
        "implementation": "Governed experiment records and provenance",
        "source": "src/experiments",
        "purpose": (
            "Record experimentation requirements, provenance, and outcomes "
            "under a controlled experiment contract."
        ),
    },
    {
        "stage": "Monitoring",
        "layer": "Monitoring",
        "implementation": "Score, decision, delivery, and experiment health",
        "source": "src/monitoring",
        "purpose": (
            "Evaluate production health across scoring, decisions, delivery, "
            "and experimentation."
        ),
    },
)


BATCH_JOBS = (
    {
        "job": "Customer Decision Batch",
        "entrypoint": "run_customer_decision_batch",
        "module": "src/deployment/customer_batch.py",
        "cli": "src.cli customer-batch",
        "scope": "Customer-level governed decision orchestration",
        "decision_engine": True,
    },
    {
        "job": "Repeat Purchase Batch",
        "entrypoint": "run_repeat_purchase_batch",
        "module": "src/deployment/repeat_purchase_batch.py",
        "cli": "src.cli repeat-purchase",
        "scope": "Standalone repeat-purchase propensity scoring",
        "decision_engine": False,
    },
    {
        "job": "Demand Forecast Batch",
        "entrypoint": "run_demand_batch",
        "module": "src/deployment/demand_batch.py",
        "cli": "src.cli demand",
        "scope": "Standalone weekly demand forecasting",
        "decision_engine": False,
    },
)


RUNTIME_CONTROLS = (
    {
        "control": "Base runtime",
        "evidence": "python:3.11-slim",
        "source": "Dockerfile",
    },
    {
        "control": "Dependency contract",
        "evidence": "requirements.txt installed during image build",
        "source": "Dockerfile",
    },
    {
        "control": "Application packaging",
        "evidence": "src copied into /app/src",
        "source": "Dockerfile",
    },
    {
        "control": "Governed model artifacts",
        "evidence": "artifacts/models copied into image contract",
        "source": "Dockerfile",
    },
    {
        "control": "Demand provenance",
        "evidence": (
            "Demand research notebook copied for SHA-256 manifest verification"
        ),
        "source": "Dockerfile",
    },
    {
        "control": "Runtime privilege",
        "evidence": "Application executes as non-root user app",
        "source": "Dockerfile",
    },
    {
        "control": "Default operation",
        "evidence": "python -m src.cli health",
        "source": "Dockerfile",
    },
    {
        "control": "Scheduling boundary",
        "evidence": (
            "External scheduler invokes scripts/run_scheduled_job.sh "
            "with explicit governed job name and business boundary"
        ),
        "source": "Dockerfile + scripts/run_scheduled_job.sh",
    },
)


ARCHITECTURE_PRINCIPLES = (
    {
        "principle": "Temporal integrity",
        "meaning": (
            "Features and model signals must respect the scoring boundary; "
            "future-conditioned reconstruction is prohibited."
        ),
    },
    {
        "principle": "Decision separation",
        "meaning": (
            "Predictive scoring and deterministic business action policy "
            "remain separate layers."
        ),
    },
    {
        "principle": "Governance before delivery",
        "meaning": (
            "Delivery systems consume governed decisions and do not create "
            "new customer actions."
        ),
    },
    {
        "principle": "Standalone branches remain standalone",
        "meaning": (
            "Repeat Purchase and Demand Forecast are production systems but "
            "do not enter mature customer-level action orchestration."
        ),
    },
    {
        "principle": "Fail-closed runtime",
        "meaning": (
            "Invalid production boundaries, unavailable governed artifacts, "
            "or invalid job contracts should fail rather than silently "
            "degrade into uncontrolled behaviour."
        ),
    },
    {
        "principle": "External scheduling boundary",
        "meaning": (
            "The repository contains the governed one-shot job interface. "
            "The external scheduler itself is intentionally outside this repo."
        ),
    },
)


def validate_architecture_evidence(
    summary: dict[str, Any],
) -> None:
    """
    Validate the public evidence required by the architecture page.
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
            "Production Architecture requires verified portfolio evidence."
        )

    required = {
        "automated_tests_passing",
        "production_intelligence_systems",
        "production_scoring_systems_available",
        "production_scoring_systems_total",
        "customer_decisions_generated",
    }

    missing = (
        required
        -
        set(
            evidence
        )
    )

    if missing:
        raise RuntimeError(
            (
                "Production Architecture evidence is incomplete: "
                f"{sorted(missing)}"
            )
        )

    if evidence[
        "production_intelligence_systems"
    ] != 6:
        raise RuntimeError(
            "Production Architecture requires six intelligence systems."
        )

    if evidence[
        "production_scoring_systems_available"
    ] != 6:
        raise RuntimeError(
            "Production Architecture requires verified 6/6 scoring availability."
        )

    if evidence[
        "production_scoring_systems_total"
    ] != 6:
        raise RuntimeError(
            "Production Architecture requires six production scoring systems."
        )


def architecture_summary(
    summary: dict[str, Any],
) -> dict[str, int]:
    """
    Return evidence-backed architecture proof points.
    """

    validate_architecture_evidence(
        summary
    )

    evidence = summary[
        "verified_evidence"
    ]

    return {
        "pipeline_stages":
            len(
                CUSTOMER_PIPELINE
            ),

        "production_systems":
            evidence[
                "production_intelligence_systems"
            ],

        "scoring_systems_available":
            evidence[
                "production_scoring_systems_available"
            ],

        "batch_jobs":
            len(
                BATCH_JOBS
            ),

        "customer_decisions":
            evidence[
                "customer_decisions_generated"
            ],

        "automated_tests":
            evidence[
                "automated_tests_passing"
            ],
    }


def customer_pipeline_rows() -> list[dict[str, str]]:
    """
    Return the frozen customer decision architecture.
    """

    return [
        dict(
            row
        )
        for row in CUSTOMER_PIPELINE
    ]


def batch_job_rows() -> list[dict[str, Any]]:
    """
    Return the three verified deployment batch jobs.
    """

    return [
        dict(
            row
        )
        for row in BATCH_JOBS
    ]


def runtime_control_rows() -> list[dict[str, str]]:
    """
    Return Docker and scheduling control evidence.
    """

    return [
        dict(
            row
        )
        for row in RUNTIME_CONTROLS
    ]


def architecture_principle_rows() -> list[dict[str, str]]:
    """
    Return production design principles represented by the implementation.
    """

    return [
        dict(
            row
        )
        for row in ARCHITECTURE_PRINCIPLES
    ]


def customer_pipeline_stage_names() -> tuple[str, ...]:
    """
    Return the ordered production customer pipeline.
    """

    return tuple(
        row[
            "stage"
        ]
        for row in CUSTOMER_PIPELINE
    )


def decision_engine_stage_names() -> tuple[str, ...]:
    """
    Return stages owned by governed customer decision orchestration.
    """

    return tuple(
        row[
            "stage"
        ]
        for row in CUSTOMER_PIPELINE
        if row[
            "layer"
        ]
        ==
        "Decision Engine"
    )


def standalone_job_names() -> tuple[str, ...]:
    """
    Return jobs intentionally outside customer Decision Engine orchestration.
    """

    return tuple(
        row[
            "job"
        ]
        for row in BATCH_JOBS
        if not row[
            "decision_engine"
        ]
    )