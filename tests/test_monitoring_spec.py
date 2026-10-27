"""
Contract tests for the frozen production monitoring specification.
"""

import pytest

from src.monitoring.spec import (
    MONITORING_CHECK_FIELDS,
    MONITORING_DEGRADED,
    MONITORING_HEALTH_RANK,
    MONITORING_HEALTH_STATES,
    MONITORING_HEALTHY,
    MONITORING_REQUIRED_FIELDS,
    MONITORING_SCHEMA_VERSION,
    MONITORING_SNAPSHOT_FIELDS,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_DECISION_OUTPUT,
    MONITOR_COMPONENT_DELIVERY_AUDIT,
    MONITOR_COMPONENT_EXPERIMENT_STORE,
    MONITOR_COMPONENT_SCORE_RUNTIME,
    MONITOR_COMPONENTS,
    monitoring_health_rank,
    validate_monitor_component,
    validate_monitoring_health,
    validate_monitoring_spec,
    worst_monitoring_health,
)


# ================================================================
# VERSION
# ================================================================


def test_monitoring_schema_version_is_frozen():

    assert (
        MONITORING_SCHEMA_VERSION
        ==
        "1.0.0"
    )


# ================================================================
# HEALTH STATES
# ================================================================


def test_monitoring_health_states_are_frozen():

    assert (
        MONITORING_HEALTH_STATES
        ==
        (
            "HEALTHY",
            "DEGRADED",
            "UNHEALTHY",
        )
    )


@pytest.mark.parametrize(
    "health",
    MONITORING_HEALTH_STATES,
)
def test_supported_health_state_is_accepted(
    health,
):

    assert (
        validate_monitoring_health(
            health
        )
        ==
        health
    )


def test_unknown_health_state_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported monitoring health",
    ):

        validate_monitoring_health(
            "UNKNOWN"
        )


def test_non_string_health_state_fails_closed():

    with pytest.raises(
        TypeError,
        match="health must be a string",
    ):

        validate_monitoring_health(
            1
        )


# ================================================================
# COMPONENTS
# ================================================================


def test_monitoring_components_are_frozen():

    assert (
        MONITOR_COMPONENTS
        ==
        (
            MONITOR_COMPONENT_SCORE_RUNTIME,
            MONITOR_COMPONENT_EXPERIMENT_STORE,
            MONITOR_COMPONENT_DECISION_OUTPUT,
            MONITOR_COMPONENT_DELIVERY_AUDIT,
        )
    )

    assert (
        MONITOR_COMPONENTS
        ==
        (
            "score_runtime",
            "experiment_store",
            "decision_output",
            "delivery_audit",
        )
    )


@pytest.mark.parametrize(
    "component",
    MONITOR_COMPONENTS,
)
def test_supported_component_is_accepted(
    component,
):

    assert (
        validate_monitor_component(
            component
        )
        ==
        component
    )


def test_unknown_component_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported monitoring component",
    ):

        validate_monitor_component(
            "unknown_component"
        )


def test_non_string_component_fails_closed():

    with pytest.raises(
        TypeError,
        match="component must be a string",
    ):

        validate_monitor_component(
            None
        )


# ================================================================
# CHECK SCHEMA
# ================================================================


def test_monitoring_check_fields_are_exact():

    assert (
        MONITORING_CHECK_FIELDS
        ==
        (
            "schema_version",
            "component",
            "check_name",
            "health",
            "observed_at_utc",
            "summary",
            "metrics",
            "details",
        )
    )


def test_all_monitoring_check_fields_are_required():

    assert (
        MONITORING_REQUIRED_FIELDS
        ==
        MONITORING_CHECK_FIELDS
    )


# ================================================================
# SNAPSHOT SCHEMA
# ================================================================


def test_monitoring_snapshot_fields_are_exact():

    assert (
        MONITORING_SNAPSHOT_FIELDS
        ==
        (
            "schema_version",
            "snapshot_id",
            "observed_at_utc",
            "overall_health",
            "checks",
        )
    )


# ================================================================
# HEALTH ORDERING
# ================================================================


def test_monitoring_health_rank_is_frozen():

    assert (
        MONITORING_HEALTH_RANK
        ==
        (
            (
                MONITORING_HEALTHY,
                0,
            ),
            (
                MONITORING_DEGRADED,
                1,
            ),
            (
                MONITORING_UNHEALTHY,
                2,
            ),
        )
    )


def test_health_rank_order():

    assert (
        monitoring_health_rank(
            MONITORING_HEALTHY
        )
        <
        monitoring_health_rank(
            MONITORING_DEGRADED
        )
        <
        monitoring_health_rank(
            MONITORING_UNHEALTHY
        )
    )


@pytest.mark.parametrize(
    "values,expected",
    (
        (
            (
                MONITORING_HEALTHY,
            ),
            MONITORING_HEALTHY,
        ),
        (
            (
                MONITORING_HEALTHY,
                MONITORING_DEGRADED,
            ),
            MONITORING_DEGRADED,
        ),
        (
            (
                MONITORING_HEALTHY,
                MONITORING_UNHEALTHY,
            ),
            MONITORING_UNHEALTHY,
        ),
        (
            (
                MONITORING_DEGRADED,
                MONITORING_UNHEALTHY,
            ),
            MONITORING_UNHEALTHY,
        ),
    ),
)
def test_worst_monitoring_health(
    values,
    expected,
):

    assert (
        worst_monitoring_health(
            values
        )
        ==
        expected
    )


def test_empty_health_collection_fails_closed():

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):

        worst_monitoring_health(
            ()
        )


def test_string_is_not_accepted_as_health_collection():

    with pytest.raises(
        TypeError,
        match="iterable",
    ):

        worst_monitoring_health(
            MONITORING_HEALTHY
        )


def test_invalid_member_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported monitoring health",
    ):

        worst_monitoring_health(
            (
                MONITORING_HEALTHY,
                "UNKNOWN",
            )
        )


# ================================================================
# SELF VALIDATION
# ================================================================


def test_monitoring_spec_self_validation_passes():

    assert (
        validate_monitoring_spec()
        is None
    )