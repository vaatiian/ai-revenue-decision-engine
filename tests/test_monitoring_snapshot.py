"""
Tests for immutable system-level monitoring snapshots.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.snapshot import (
    MonitoringSnapshot,
    build_monitoring_snapshot,
    monitoring_snapshot_to_dict,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_SCHEMA_VERSION,
    MONITORING_SNAPSHOT_FIELDS,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_DECISION_OUTPUT,
    MONITOR_COMPONENT_DELIVERY_AUDIT,
    MONITOR_COMPONENT_EXPERIMENT_STORE,
    MONITOR_COMPONENT_SCORE_RUNTIME,
)


OBSERVED_AT = datetime(
    2026,
    10,
    27,
    20,
    30,
    tzinfo=timezone.utc,
)


def make_check(
    component,
    health,
):

    return MonitoringCheck(
        schema_version=(
            MONITORING_SCHEMA_VERSION
        ),

        component=(
            component
        ),

        check_name=(
            f"{component}_check"
        ),

        health=(
            health
        ),

        observed_at_utc=(
            OBSERVED_AT
        ),

        summary=(
            f"{component} is {health}."
        ),

        metrics={
            "count":
                1,
        },

        details=(
            f"{component}: {health}",
        ),
    )


def healthy_checks():

    return (
        make_check(
            MONITOR_COMPONENT_SCORE_RUNTIME,
            MONITORING_HEALTHY,
        ),
        make_check(
            MONITOR_COMPONENT_EXPERIMENT_STORE,
            MONITORING_HEALTHY,
        ),
        make_check(
            MONITOR_COMPONENT_DECISION_OUTPUT,
            MONITORING_HEALTHY,
        ),
        make_check(
            MONITOR_COMPONENT_DELIVERY_AUDIT,
            MONITORING_HEALTHY,
        ),
    )


# ================================================================
# BUILD
# ================================================================


def test_all_healthy_checks_produce_healthy_snapshot():

    snapshot = (
        build_monitoring_snapshot(
            healthy_checks(),
            snapshot_id=(
                "snapshot-001"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_HEALTHY
    )


def test_degraded_component_degrades_snapshot():

    checks = list(
        healthy_checks()
    )

    checks[
        1
    ] = make_check(
        MONITOR_COMPONENT_EXPERIMENT_STORE,
        MONITORING_DEGRADED,
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "snapshot-002"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_DEGRADED
    )


def test_unhealthy_component_makes_snapshot_unhealthy():

    checks = list(
        healthy_checks()
    )

    checks[
        2
    ] = make_check(
        MONITOR_COMPONENT_DECISION_OUTPUT,
        MONITORING_UNHEALTHY,
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "snapshot-003"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert (
        snapshot.overall_health
        ==
        MONITORING_UNHEALTHY
    )


def test_snapshot_preserves_check_order():

    checks = (
        make_check(
            MONITOR_COMPONENT_DELIVERY_AUDIT,
            MONITORING_HEALTHY,
        ),
        make_check(
            MONITOR_COMPONENT_SCORE_RUNTIME,
            MONITORING_HEALTHY,
        ),
    )

    snapshot = (
        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "snapshot-004"
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
        MONITOR_COMPONENT_DELIVERY_AUDIT,
        MONITOR_COMPONENT_SCORE_RUNTIME,
    ]


# ================================================================
# IMMUTABILITY
# ================================================================


def test_snapshot_is_frozen():

    snapshot = (
        build_monitoring_snapshot(
            healthy_checks(),
            snapshot_id=(
                "snapshot-005"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    with pytest.raises(
        FrozenInstanceError
    ):

        snapshot.overall_health = (
            MONITORING_UNHEALTHY
        )


def test_checks_are_frozen_as_tuple():

    snapshot = (
        build_monitoring_snapshot(
            list(
                healthy_checks()
            ),
            snapshot_id=(
                "snapshot-006"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert isinstance(
        snapshot.checks,
        tuple,
    )


# ================================================================
# VALIDATION
# ================================================================


def test_wrong_schema_version_fails_closed():

    with pytest.raises(
        ValueError,
        match="schema_version",
    ):

        MonitoringSnapshot(
            schema_version=(
                "9.9.9"
            ),
            snapshot_id=(
                "snapshot-007"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
            overall_health=(
                MONITORING_HEALTHY
            ),
            checks=(
                healthy_checks()
            ),
        )


def test_empty_snapshot_id_fails_closed():

    with pytest.raises(
        ValueError,
        match="snapshot_id must not be empty",
    ):

        build_monitoring_snapshot(
            healthy_checks(),
            snapshot_id=(
                ""
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )


def test_empty_check_collection_fails_closed():

    with pytest.raises(
        ValueError,
        match="checks must not be empty",
    ):

        build_monitoring_snapshot(
            (),
            snapshot_id=(
                "snapshot-008"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )


def test_single_check_object_is_rejected():

    check = (
        make_check(
            MONITOR_COMPONENT_SCORE_RUNTIME,
            MONITORING_HEALTHY,
        )
    )

    with pytest.raises(
        TypeError,
        match="iterable",
    ):

        build_monitoring_snapshot(
            check,
            snapshot_id=(
                "snapshot-009"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )


def test_wrong_check_member_type_is_rejected():

    with pytest.raises(
        TypeError,
        match="Every check",
    ):

        build_monitoring_snapshot(
            (
                {},
            ),
            snapshot_id=(
                "snapshot-010"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )


def test_duplicate_components_fail_closed():

    checks = (
        make_check(
            MONITOR_COMPONENT_SCORE_RUNTIME,
            MONITORING_HEALTHY,
        ),
        make_check(
            MONITOR_COMPONENT_SCORE_RUNTIME,
            MONITORING_DEGRADED,
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicate components",
    ):

        build_monitoring_snapshot(
            checks,
            snapshot_id=(
                "snapshot-011"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )


def test_manual_incorrect_overall_health_fails_closed():

    with pytest.raises(
        ValueError,
        match="worst component health",
    ):

        MonitoringSnapshot(
            schema_version=(
                MONITORING_SCHEMA_VERSION
            ),
            snapshot_id=(
                "snapshot-012"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
            overall_health=(
                MONITORING_HEALTHY
            ),
            checks=(
                (
                    make_check(
                        MONITOR_COMPONENT_SCORE_RUNTIME,
                        MONITORING_UNHEALTHY,
                    ),
                )
            ),
        )


def test_invalid_timestamp_fails_closed():

    with pytest.raises(
        ValueError,
        match="not a valid timestamp",
    ):

        build_monitoring_snapshot(
            healthy_checks(),
            snapshot_id=(
                "snapshot-013"
            ),
            observed_at_utc=(
                "not-a-date"
            ),
        )


# ================================================================
# SERIALISATION
# ================================================================


def test_snapshot_to_dict_has_exact_schema():

    snapshot = (
        build_monitoring_snapshot(
            healthy_checks(),
            snapshot_id=(
                "snapshot-014"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    output = (
        monitoring_snapshot_to_dict(
            snapshot
        )
    )

    assert tuple(
        output.keys()
    ) == MONITORING_SNAPSHOT_FIELDS


def test_snapshot_to_dict_serialises_timestamp():

    snapshot = (
        build_monitoring_snapshot(
            healthy_checks(),
            snapshot_id=(
                "snapshot-015"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    output = (
        monitoring_snapshot_to_dict(
            snapshot
        )
    )

    assert (
        output[
            "observed_at_utc"
        ]
        ==
        "2026-10-27T20:30:00+00:00"
    )


def test_snapshot_to_dict_serialises_checks():

    snapshot = (
        build_monitoring_snapshot(
            healthy_checks(),
            snapshot_id=(
                "snapshot-016"
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    output = (
        monitoring_snapshot_to_dict(
            snapshot
        )
    )

    assert len(
        output[
            "checks"
        ]
    ) == 4

    assert (
        output[
            "checks"
        ][
            0
        ][
            "component"
        ]
        ==
        MONITOR_COMPONENT_SCORE_RUNTIME
    )


def test_snapshot_to_dict_rejects_wrong_type():

    with pytest.raises(
        TypeError,
        match="MonitoringSnapshot",
    ):

        monitoring_snapshot_to_dict(
            {}
        )