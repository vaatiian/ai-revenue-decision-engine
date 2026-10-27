"""
End-to-end integration tests for production monitoring.

These tests exercise the real production boundaries:

Decision Engine
    -> Delivery Builder
    -> Delivery Audit
    -> Experiment Store
    -> Score Runtime Availability
    -> Monitoring Checks
    -> Monitoring Snapshot
    -> Operational Alert

Mocks are used only where a test deliberately injects a failure state.
"""

from datetime import datetime, timezone

import pandas as pd

import src.monitoring.score_health as score_health_module

from src.decision_engine.engine import (
    build_customer_decisions,
)

from src.delivery.audit import (
    append_delivery_batch_audit,
)

from src.delivery.builder import (
    build_delivery_batch,
)

from src.experiments.record import (
    ExperimentRunRecord,
)

from src.experiments.spec import (
    EXPERIMENT_SCHEMA_VERSION,
    RUN_KIND_EVALUATION,
    RUN_STATUS_SUCCEEDED,
)

from src.experiments.store import (
    append_experiment_run,
)

from src.models.score_availability import (
    ScoreAvailability,
    ScoreAvailabilityStatus,
)

from src.monitoring.alert import (
    ALERT_SEVERITY_CRITICAL,
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARNING,
    alert_from_snapshot,
)

from src.monitoring.decision_health import (
    monitor_decision_output,
)

from src.monitoring.delivery_health import (
    monitor_delivery_audit,
)

from src.monitoring.experiment_health import (
    monitor_experiment_store,
)

from src.monitoring.score_health import (
    monitor_score_runtime,
)

from src.monitoring.snapshot import (
    build_monitoring_snapshot,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_UNHEALTHY,
)


# ================================================================
# CONSTANTS
# ================================================================


DECISION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


OBSERVED_AT = datetime(
    2026,
    10,
    27,
    21,
    30,
    tzinfo=timezone.utc,
)


SHA_A = (
    "a"
    *
    64
)

SHA_B = (
    "b"
    *
    64
)

SHA_C = (
    "c"
    *
    64
)


# ================================================================
# REAL DECISION INPUT
# ================================================================


def make_transactions():
    """
    Small real Decision Engine input.

    Customer A is active at the decision date and has no supplied model
    intervention scores, so the governed engine should still produce one
    valid canonical decision.
    """

    return pd.DataFrame(
        {
            "date":
                pd.to_datetime(
                    [
                        "2011-08-31 10:00:00",
                    ]
                ),

            "customer":
                [
                    "A",
                ],

            "invoice":
                [
                    "A1",
                ],

            "quantity":
                [
                    1,
                ],

            "unit_price":
                [
                    10.0,
                ],

            "net_merchandise_revenue":
                [
                    10.0,
                ],

            "is_cancellation":
                [
                    False,
                ],

            "is_return":
                [
                    False,
                ],
        }
    )


def build_real_decisions():
    """
    Run the actual production Decision Engine.
    """

    return build_customer_decisions(
        transactions=(
            make_transactions()
        ),
        decision_date=(
            DECISION_DATE
        ),
    )


# ================================================================
# REAL EXPERIMENT RECORD
# ================================================================


def make_experiment_record(
    run_id="monitoring-integration-run-001",
):

    return ExperimentRunRecord(
        schema_version=(
            EXPERIMENT_SCHEMA_VERSION
        ),

        run_id=(
            run_id
        ),

        experiment_name=(
            "reactivation-final-evaluation"
        ),

        run_kind=(
            RUN_KIND_EVALUATION
        ),

        status=(
            RUN_STATUS_SUCCEEDED
        ),

        score_id=(
            "reactivation"
        ),

        model_name=(
            "Full Random Forest"
        ),

        model_version=(
            "1.0.0"
        ),

        model_type=(
            "classification"
        ),

        target_name=(
            "reactivation_within_next_90_days"
        ),

        started_at_utc=(
            datetime(
                2026,
                10,
                27,
                10,
                0,
                tzinfo=timezone.utc,
            )
        ),

        completed_at_utc=(
            datetime(
                2026,
                10,
                27,
                10,
                5,
                tzinfo=timezone.utc,
            )
        ),

        git_commit=(
            "61424ff89ecf21325f7f7d10fee191fff5c1a804"
        ),

        git_branch=(
            "main"
        ),

        manifest_path=(
            "artifacts/models/reactivation/"
            "reactivation_full_random_forest_v1.0.0.json"
        ),

        manifest_sha256=(
            SHA_A
        ),

        artifact_path=(
            "artifacts/models/reactivation/"
            "reactivation_full_random_forest_v1.0.0.joblib"
        ),

        artifact_sha256=(
            SHA_B
        ),

        data_reference=(
            "cleaned_online_retail.parquet"
        ),

        data_sha256=(
            SHA_C
        ),

        parameters={
            "split":
                "test",
        },

        metrics={
            "pr_auc":
                0.615,
        },

        tags=(
            "production",
            "temporal",
        ),

        notes=None,

        error_message=None,
    )


# ================================================================
# REAL DELIVERY PIPELINE
# ================================================================


def build_and_audit_delivery(
    tmp_path,
    decisions,
):

    payloads = (
        build_delivery_batch(
            decisions,
            delivery_created_at_utc=(
                OBSERVED_AT
            ),
            delivery_ids={
                "A":
                    "delivery-monitoring-001",
            },
        )
    )

    audit_path = (
        tmp_path
        /
        "delivery"
        /
        "delivery_audit.jsonl"
    )

    append_delivery_batch_audit(
        payloads,
        audit_path,
    )

    return (
        payloads,
        audit_path,
    )


# ================================================================
# REAL EXPERIMENT STORE
# ================================================================


def build_experiment_store(
    tmp_path,
):

    path = (
        tmp_path
        /
        "experiments"
        /
        "experiment_runs.jsonl"
    )

    append_experiment_run(
        make_experiment_record(),
        path,
    )

    return path


# ================================================================
# HAPPY PATH
# ================================================================


def test_real_decision_engine_output_is_monitorable():

    decisions = (
        build_real_decisions()
    )

    check = (
        monitor_decision_output(
            decisions,
            observed_at_utc=(
                OBSERVED_AT
            ),
            decision_date=(
                DECISION_DATE
            ),
        )
    )

    assert (
        check.health
        ==
        MONITORING_HEALTHY
    )

    assert (
        check.metrics[
            "row_count"
        ]
        ==
        1
    )

    assert (
        check.metrics[
            "unique_customers"
        ]
        ==
        1
    )


def test_real_decision_output_flows_into_real_delivery_audit(
    tmp_path,
):

    decisions = (
        build_real_decisions()
    )

    payloads, audit_path = (
        build_and_audit_delivery(
            tmp_path,
            decisions,
        )
    )

    check = (
        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                audit_path
            ),
        )
    )

    assert len(
        payloads
    ) == 1

    assert (
        check.health
        ==
        MONITORING_HEALTHY
    )

    assert (
        check.metrics[
            "total_deliveries"
        ]
        ==
        1
    )


def test_real_experiment_store_is_monitorable(
    tmp_path,
):

    experiment_path = (
        build_experiment_store(
            tmp_path
        )
    )

    check = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                experiment_path
            ),
        )
    )

    assert (
        check.health
        ==
        MONITORING_HEALTHY
    )

    assert (
        check.metrics[
            "total_runs"
        ]
        ==
        1
    )

    assert (
        check.metrics[
            "status_counts"
        ][
            RUN_STATUS_SUCCEEDED
        ]
        ==
        1
    )


def test_real_six_score_registry_is_monitorable():

    check = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    assert (
        check.health
        ==
        MONITORING_HEALTHY
    )

    assert (
        check.metrics[
            "total_scores"
        ]
        ==
        6
    )

    assert (
        check.metrics[
            "available_scores"
        ]
        ==
        6
    )


def test_complete_real_monitoring_pipeline_is_healthy(
    tmp_path,
):

    decisions = (
        build_real_decisions()
    )

    _, audit_path = (
        build_and_audit_delivery(
            tmp_path,
            decisions,
        )
    )

    experiment_path = (
        build_experiment_store(
            tmp_path
        )
    )

    checks = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        ),

        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                experiment_path
            ),
        ),

        monitor_decision_output(
            decisions,
            observed_at_utc=(
                OBSERVED_AT
            ),
            decision_date=(
                DECISION_DATE
            ),
        ),

        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                audit_path
            ),
        ),
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "monitoring-integration-healthy"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    alert = (
        alert_from_snapshot(
            snapshot
        )
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_HEALTHY
    )

    assert (
        alert.severity
        ==
        ALERT_SEVERITY_INFO
    )

    assert (
        alert.alert_required
        is False
    )


# ================================================================
# DEGRADED PIPELINE
# ================================================================


def test_missing_delivery_history_degrades_real_snapshot(
    tmp_path,
):

    decisions = (
        build_real_decisions()
    )

    experiment_path = (
        build_experiment_store(
            tmp_path
        )
    )

    missing_audit_path = (
        tmp_path
        /
        "delivery"
        /
        "missing_audit.jsonl"
    )

    checks = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        ),

        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                experiment_path
            ),
        ),

        monitor_decision_output(
            decisions,
            observed_at_utc=(
                OBSERVED_AT
            ),
            decision_date=(
                DECISION_DATE
            ),
        ),

        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                missing_audit_path
            ),
        ),
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "monitoring-integration-degraded"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    alert = (
        alert_from_snapshot(
            snapshot
        )
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_DEGRADED
    )

    assert (
        alert.severity
        ==
        ALERT_SEVERITY_WARNING
    )

    assert (
        alert.alert_required
        is True
    )


# ================================================================
# UNHEALTHY DECISION PIPELINE
# ================================================================


def test_invalid_decision_output_makes_snapshot_critical(
    tmp_path,
):

    experiment_path = (
        build_experiment_store(
            tmp_path
        )
    )

    invalid_decisions = pd.DataFrame(
        {
            "customer":
                [
                    "A",
                ],
        }
    )

    missing_audit_path = (
        tmp_path
        /
        "delivery"
        /
        "missing.jsonl"
    )

    checks = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        ),

        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                experiment_path
            ),
        ),

        monitor_decision_output(
            invalid_decisions,
            observed_at_utc=(
                OBSERVED_AT
            ),
        ),

        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                missing_audit_path
            ),
        ),
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "monitoring-integration-invalid-decision"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    alert = (
        alert_from_snapshot(
            snapshot
        )
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_UNHEALTHY
    )

    assert (
        alert.severity
        ==
        ALERT_SEVERITY_CRITICAL
    )

    assert (
        alert.alert_required
        is True
    )


# ================================================================
# UNHEALTHY EXPERIMENT STORE
# ================================================================


def test_corrupt_experiment_store_makes_snapshot_critical(
    tmp_path,
):

    decisions = (
        build_real_decisions()
    )

    _, audit_path = (
        build_and_audit_delivery(
            tmp_path,
            decisions,
        )
    )

    experiment_path = (
        tmp_path
        /
        "experiments"
        /
        "broken.jsonl"
    )

    experiment_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_path.write_text(
        "{not-valid-json}\n",
        encoding="utf-8",
    )

    checks = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        ),

        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                experiment_path
            ),
        ),

        monitor_decision_output(
            decisions,
            observed_at_utc=(
                OBSERVED_AT
            ),
            decision_date=(
                DECISION_DATE
            ),
        ),

        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                audit_path
            ),
        ),
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "monitoring-integration-corrupt-experiment"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    alert = (
        alert_from_snapshot(
            snapshot
        )
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_UNHEALTHY
    )

    assert (
        alert.severity
        ==
        ALERT_SEVERITY_CRITICAL
    )


# ================================================================
# UNHEALTHY DELIVERY AUDIT
# ================================================================


def test_corrupt_delivery_audit_makes_snapshot_critical(
    tmp_path,
):

    decisions = (
        build_real_decisions()
    )

    experiment_path = (
        build_experiment_store(
            tmp_path
        )
    )

    audit_path = (
        tmp_path
        /
        "delivery"
        /
        "broken.jsonl"
    )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path.write_text(
        "{not-valid-json}\n",
        encoding="utf-8",
    )

    checks = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        ),

        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                experiment_path
            ),
        ),

        monitor_decision_output(
            decisions,
            observed_at_utc=(
                OBSERVED_AT
            ),
            decision_date=(
                DECISION_DATE
            ),
        ),

        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                audit_path
            ),
        ),
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "monitoring-integration-corrupt-delivery"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    alert = (
        alert_from_snapshot(
            snapshot
        )
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_UNHEALTHY
    )

    assert (
        alert.severity
        ==
        ALERT_SEVERITY_CRITICAL
    )


# ================================================================
# SCORE FAILURE PROPAGATION
# ================================================================


def test_unavailable_score_propagates_to_critical_snapshot(
    monkeypatch,
    tmp_path,
):

    decisions = (
        build_real_decisions()
    )

    _, audit_path = (
        build_and_audit_delivery(
            tmp_path,
            decisions,
        )
    )

    experiment_path = (
        build_experiment_store(
            tmp_path
        )
    )

    unavailable_score = (
        ScoreAvailability(
            score_id=(
                "reactivation"
            ),

            status=(
                ScoreAvailabilityStatus.MISSING_ARTIFACT
            ),

            available=False,

            artifact_exists=False,

            manifest_exists=True,

            detail=(
                "Synthetic missing artifact."
            ),
        )
    )

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                unavailable_score,
            ),
    )

    checks = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        ),

        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                experiment_path
            ),
        ),

        monitor_decision_output(
            decisions,
            observed_at_utc=(
                OBSERVED_AT
            ),
            decision_date=(
                DECISION_DATE
            ),
        ),

        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                audit_path
            ),
        ),
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "monitoring-integration-score-failure"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    alert = (
        alert_from_snapshot(
            snapshot
        )
    )

    assert (
        checks[
            0
        ].health
        ==
        MONITORING_UNHEALTHY
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_UNHEALTHY
    )

    assert (
        alert.severity
        ==
        ALERT_SEVERITY_CRITICAL
    )

    assert (
        alert.alert_required
        is True
    )


# ================================================================
# SNAPSHOT COMPONENT COVERAGE
# ================================================================


def test_real_snapshot_contains_all_four_components(
    tmp_path,
):

    decisions = (
        build_real_decisions()
    )

    _, audit_path = (
        build_and_audit_delivery(
            tmp_path,
            decisions,
        )
    )

    experiment_path = (
        build_experiment_store(
            tmp_path
        )
    )

    checks = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        ),

        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                experiment_path
            ),
        ),

        monitor_decision_output(
            decisions,
            observed_at_utc=(
                OBSERVED_AT
            ),
            decision_date=(
                DECISION_DATE
            ),
        ),

        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                audit_path
            ),
        ),
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "monitoring-integration-components"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert [
        check.component
        for check
        in snapshot.checks
    ] == [
        "score_runtime",
        "experiment_store",
        "decision_output",
        "delivery_audit",
    ]