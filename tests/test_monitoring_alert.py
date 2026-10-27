"""
Tests for monitoring alert and severity policy.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.monitoring.alert import (
    ALERT_POLICY,
    ALERT_SEVERITIES,
    ALERT_SEVERITY_CRITICAL,
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARNING,
    MONITORING_ALERT_FIELDS,
    MonitoringAlert,
    alert_from_check,
    alert_from_snapshot,
    get_alert_policy,
    monitoring_alert_to_dict,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.snapshot import (
    build_monitoring_snapshot,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_SCHEMA_VERSION,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_SCORE_RUNTIME,
)


OBSERVED_AT = datetime(
    2026,
    10,
    27,
    21,
    0,
    tzinfo=timezone.utc,
)


def make_check(
    health,
):

    return MonitoringCheck(
        schema_version=(
            MONITORING_SCHEMA_VERSION
        ),

        component=(
            MONITOR_COMPONENT_SCORE_RUNTIME
        ),

        check_name=(
            "score_runtime_check"
        ),

        health=(
            health
        ),

        observed_at_utc=(
            OBSERVED_AT
        ),

        summary=(
            f"Score runtime is {health}."
        ),

        metrics={
            "count":
                1,
        },

        details=(
            f"health={health}",
        ),
    )


# ================================================================
# FROZEN POLICY
# ================================================================


def test_alert_severities_are_frozen():

    assert (
        ALERT_SEVERITIES
        ==
        (
            "INFO",
            "WARNING",
            "CRITICAL",
        )
    )


def test_alert_policy_is_frozen():

    assert (
        ALERT_POLICY
        ==
        (
            (
                MONITORING_HEALTHY,
                ALERT_SEVERITY_INFO,
                False,
            ),
            (
                MONITORING_DEGRADED,
                ALERT_SEVERITY_WARNING,
                True,
            ),
            (
                MONITORING_UNHEALTHY,
                ALERT_SEVERITY_CRITICAL,
                True,
            ),
        )
    )


@pytest.mark.parametrize(
    "health,expected_severity,expected_required",
    (
        (
            MONITORING_HEALTHY,
            ALERT_SEVERITY_INFO,
            False,
        ),
        (
            MONITORING_DEGRADED,
            ALERT_SEVERITY_WARNING,
            True,
        ),
        (
            MONITORING_UNHEALTHY,
            ALERT_SEVERITY_CRITICAL,
            True,
        ),
    ),
)
def test_get_alert_policy(
    health,
    expected_severity,
    expected_required,
):

    severity, required = (
        get_alert_policy(
            health
        )
    )

    assert (
        severity
        ==
        expected_severity
    )

    assert (
        required
        is
        expected_required
    )


def test_unknown_health_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported monitoring health",
    ):

        get_alert_policy(
            "UNKNOWN"
        )


# ================================================================
# ALERT OBJECT
# ================================================================


def test_valid_alert_can_be_created():

    alert = MonitoringAlert(
        source_type=(
            "component"
        ),
        source_name=(
            "score_runtime"
        ),
        health=(
            MONITORING_HEALTHY
        ),
        severity=(
            ALERT_SEVERITY_INFO
        ),
        alert_required=False,
        observed_at_utc=(
            OBSERVED_AT
        ),
        summary=(
            "Score runtime is healthy."
        ),
    )

    assert (
        alert.severity
        ==
        ALERT_SEVERITY_INFO
    )


def test_alert_is_frozen():

    alert = alert_from_check(
        make_check(
            MONITORING_HEALTHY
        )
    )

    with pytest.raises(
        FrozenInstanceError
    ):

        alert.severity = (
            ALERT_SEVERITY_CRITICAL
        )


def test_wrong_severity_for_health_fails_closed():

    with pytest.raises(
        ValueError,
        match="severity does not match",
    ):

        MonitoringAlert(
            source_type=(
                "component"
            ),
            source_name=(
                "score_runtime"
            ),
            health=(
                MONITORING_HEALTHY
            ),
            severity=(
                ALERT_SEVERITY_CRITICAL
            ),
            alert_required=False,
            observed_at_utc=(
                OBSERVED_AT
            ),
            summary=(
                "Invalid alert."
            ),
        )


def test_wrong_alert_required_for_health_fails_closed():

    with pytest.raises(
        ValueError,
        match="alert_required does not match",
    ):

        MonitoringAlert(
            source_type=(
                "component"
            ),
            source_name=(
                "score_runtime"
            ),
            health=(
                MONITORING_HEALTHY
            ),
            severity=(
                ALERT_SEVERITY_INFO
            ),
            alert_required=True,
            observed_at_utc=(
                OBSERVED_AT
            ),
            summary=(
                "Invalid alert."
            ),
        )


def test_alert_required_must_be_bool():

    with pytest.raises(
        TypeError,
        match="alert_required must be a bool",
    ):

        MonitoringAlert(
            source_type=(
                "component"
            ),
            source_name=(
                "score_runtime"
            ),
            health=(
                MONITORING_HEALTHY
            ),
            severity=(
                ALERT_SEVERITY_INFO
            ),
            alert_required=(
                0
            ),
            observed_at_utc=(
                OBSERVED_AT
            ),
            summary=(
                "Invalid alert."
            ),
        )


def test_empty_source_name_fails_closed():

    with pytest.raises(
        ValueError,
        match="source_name must not be empty",
    ):

        MonitoringAlert(
            source_type=(
                "component"
            ),
            source_name=(
                ""
            ),
            health=(
                MONITORING_HEALTHY
            ),
            severity=(
                ALERT_SEVERITY_INFO
            ),
            alert_required=False,
            observed_at_utc=(
                OBSERVED_AT
            ),
            summary=(
                "Invalid alert."
            ),
        )


# ================================================================
# COMPONENT ALERTS
# ================================================================


def test_healthy_check_produces_info_without_alert():

    alert = (
        alert_from_check(
            make_check(
                MONITORING_HEALTHY
            )
        )
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


def test_degraded_check_produces_warning_alert():

    alert = (
        alert_from_check(
            make_check(
                MONITORING_DEGRADED
            )
        )
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


def test_unhealthy_check_produces_critical_alert():

    alert = (
        alert_from_check(
            make_check(
                MONITORING_UNHEALTHY
            )
        )
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


def test_check_alert_preserves_component_name():

    alert = (
        alert_from_check(
            make_check(
                MONITORING_DEGRADED
            )
        )
    )

    assert (
        alert.source_type
        ==
        "component"
    )

    assert (
        alert.source_name
        ==
        MONITOR_COMPONENT_SCORE_RUNTIME
    )


def test_alert_from_check_rejects_wrong_type():

    with pytest.raises(
        TypeError,
        match="MonitoringCheck",
    ):

        alert_from_check(
            {}
        )


# ================================================================
# SNAPSHOT ALERTS
# ================================================================


def test_healthy_snapshot_produces_info_without_alert():

    snapshot = (
        build_monitoring_snapshot(
            (
                make_check(
                    MONITORING_HEALTHY
                ),
            ),
            snapshot_id=(
                "snapshot-001"
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
        alert.severity
        ==
        ALERT_SEVERITY_INFO
    )

    assert (
        alert.alert_required
        is False
    )


def test_degraded_snapshot_produces_warning_alert():

    snapshot = (
        build_monitoring_snapshot(
            (
                make_check(
                    MONITORING_DEGRADED
                ),
            ),
            snapshot_id=(
                "snapshot-002"
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
        alert.severity
        ==
        ALERT_SEVERITY_WARNING
    )

    assert (
        alert.alert_required
        is True
    )


def test_unhealthy_snapshot_produces_critical_alert():

    snapshot = (
        build_monitoring_snapshot(
            (
                make_check(
                    MONITORING_UNHEALTHY
                ),
            ),
            snapshot_id=(
                "snapshot-003"
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
        alert.severity
        ==
        ALERT_SEVERITY_CRITICAL
    )

    assert (
        alert.alert_required
        is True
    )


def test_snapshot_alert_preserves_snapshot_identity():

    snapshot = (
        build_monitoring_snapshot(
            (
                make_check(
                    MONITORING_HEALTHY
                ),
            ),
            snapshot_id=(
                "snapshot-004"
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
        alert.source_type
        ==
        "system"
    )

    assert (
        alert.source_name
        ==
        "snapshot-004"
    )


def test_alert_from_snapshot_rejects_wrong_type():

    with pytest.raises(
        TypeError,
        match="MonitoringSnapshot",
    ):

        alert_from_snapshot(
            {}
        )


# ================================================================
# SERIALISATION
# ================================================================


def test_monitoring_alert_to_dict_has_exact_schema():

    alert = (
        alert_from_check(
            make_check(
                MONITORING_DEGRADED
            )
        )
    )

    output = (
        monitoring_alert_to_dict(
            alert
        )
    )

    assert tuple(
        output.keys()
    ) == MONITORING_ALERT_FIELDS


def test_monitoring_alert_to_dict_serialises_timestamp():

    alert = (
        alert_from_check(
            make_check(
                MONITORING_HEALTHY
            )
        )
    )

    output = (
        monitoring_alert_to_dict(
            alert
        )
    )

    assert (
        output[
            "observed_at_utc"
        ]
        ==
        "2026-10-27T21:00:00+00:00"
    )


def test_monitoring_alert_to_dict_rejects_wrong_type():

    with pytest.raises(
        TypeError,
        match="MonitoringAlert",
    ):

        monitoring_alert_to_dict(
            {}
        )