import pytest

from portfolio.data_loader import (
    load_portfolio_summary,
)

from portfolio.production_architecture import (
    architecture_principle_rows,
    architecture_summary,
    batch_job_rows,
    customer_pipeline_rows,
    customer_pipeline_stage_names,
    decision_engine_stage_names,
    runtime_control_rows,
    standalone_job_names,
    validate_architecture_evidence,
)


def summary():
    return load_portfolio_summary()


def test_real_architecture_evidence_validates():
    validate_architecture_evidence(
        summary()
    )


def test_customer_pipeline_has_eleven_stages():
    assert len(
        customer_pipeline_rows()
    ) == 11


def test_customer_pipeline_order_is_frozen():
    assert (
        customer_pipeline_stage_names()
        ==
        (
            "Canonical Transactions",
            "Feature Pipelines",
            "Production Scorers",
            "Score Registry",
            "Decision Snapshot",
            "Signal Alignment",
            "Action Orchestration",
            "Priority & Contact Policy",
            "CRM Delivery",
            "Experiment Logging",
            "Monitoring",
        )
    )


def test_decision_engine_has_four_explicit_stages():
    assert (
        decision_engine_stage_names()
        ==
        (
            "Decision Snapshot",
            "Signal Alignment",
            "Action Orchestration",
            "Priority & Contact Policy",
        )
    )


def test_three_batch_jobs_are_exposed():
    assert len(
        batch_job_rows()
    ) == 3


def test_customer_batch_enters_decision_engine():
    row = next(
        row
        for row in batch_job_rows()
        if row[
            "job"
        ]
        ==
        "Customer Decision Batch"
    )

    assert row[
        "decision_engine"
    ] is True


def test_repeat_purchase_batch_is_standalone():
    row = next(
        row
        for row in batch_job_rows()
        if row[
            "job"
        ]
        ==
        "Repeat Purchase Batch"
    )

    assert row[
        "decision_engine"
    ] is False


def test_demand_batch_is_standalone():
    row = next(
        row
        for row in batch_job_rows()
        if row[
            "job"
        ]
        ==
        "Demand Forecast Batch"
    )

    assert row[
        "decision_engine"
    ] is False


def test_standalone_jobs_are_frozen():
    assert (
        standalone_job_names()
        ==
        (
            "Repeat Purchase Batch",
            "Demand Forecast Batch",
        )
    )


def test_customer_batch_entrypoint_is_real():
    row = batch_job_rows()[
        0
    ]

    assert (
        row[
            "entrypoint"
        ]
        ==
        "run_customer_decision_batch"
    )


def test_repeat_purchase_entrypoint_is_real():
    row = batch_job_rows()[
        1
    ]

    assert (
        row[
            "entrypoint"
        ]
        ==
        "run_repeat_purchase_batch"
    )


def test_demand_entrypoint_is_real():
    row = batch_job_rows()[
        2
    ]

    assert (
        row[
            "entrypoint"
        ]
        ==
        "run_demand_batch"
    )


def test_runtime_controls_include_non_root():
    controls = runtime_control_rows()

    assert any(
        row[
            "control"
        ]
        ==
        "Runtime privilege"
        and
        "non-root"
        in row[
            "evidence"
        ]
        for row in controls
    )


def test_runtime_controls_include_health_default():
    controls = runtime_control_rows()

    assert any(
        row[
            "control"
        ]
        ==
        "Default operation"
        and
        "src.cli health"
        in row[
            "evidence"
        ]
        for row in controls
    )


def test_runtime_controls_preserve_external_scheduler_boundary():
    controls = runtime_control_rows()

    scheduler = next(
        row
        for row in controls
        if row[
            "control"
        ]
        ==
        "Scheduling boundary"
    )

    assert (
        "External scheduler"
        in
        scheduler[
            "evidence"
        ]
    )

    assert (
        "run_scheduled_job.sh"
        in
        scheduler[
            "evidence"
        ]
    )


def test_architecture_principles_include_temporal_integrity():
    principles = architecture_principle_rows()

    assert any(
        row[
            "principle"
        ]
        ==
        "Temporal integrity"
        for row in principles
    )


def test_architecture_principles_include_decision_separation():
    principles = architecture_principle_rows()

    assert any(
        row[
            "principle"
        ]
        ==
        "Decision separation"
        for row in principles
    )


def test_architecture_summary_uses_verified_counts():
    result = architecture_summary(
        summary()
    )

    assert result[
        "production_systems"
    ] == 6

    assert result[
        "scoring_systems_available"
    ] == 6

    assert result[
        "customer_decisions"
    ] == 5220

    assert result[
        "automated_tests"
    ] == 1915


def test_architecture_summary_batch_count():
    assert (
        architecture_summary(
            summary()
        )[
            "batch_jobs"
        ]
        ==
        3
    )


def test_missing_verified_evidence_fails_closed():
    broken = dict(
        summary()
    )

    broken.pop(
        "verified_evidence"
    )

    with pytest.raises(
        RuntimeError,
        match="verified portfolio evidence",
    ):

        validate_architecture_evidence(
            broken
        )


def test_wrong_system_count_fails_closed():
    broken = dict(
        summary()
    )

    broken[
        "verified_evidence"
    ] = dict(
        broken[
            "verified_evidence"
        ]
    )

    broken[
        "verified_evidence"
    ][
        "production_intelligence_systems"
    ] = 5

    with pytest.raises(
        RuntimeError,
        match="six intelligence systems",
    ):

        validate_architecture_evidence(
            broken
        )


def test_partial_scoring_availability_fails_closed():
    broken = dict(
        summary()
    )

    broken[
        "verified_evidence"
    ] = dict(
        broken[
            "verified_evidence"
        ]
    )

    broken[
        "verified_evidence"
    ][
        "production_scoring_systems_available"
    ] = 5

    with pytest.raises(
        RuntimeError,
        match="6/6 scoring availability",
    ):

        validate_architecture_evidence(
            broken
        )