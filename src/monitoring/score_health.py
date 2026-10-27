"""
Production score-runtime health monitoring.

This module observes the existing governed score-availability subsystem.

It does not:
- load or score customer data directly;
- retrain models;
- repair artifacts;
- regenerate manifests;
- change score registry configuration.
"""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType

from src.models.score_availability import (
    ScoreAvailabilityStatus,
    list_score_statuses,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.spec import (
    MONITORING_HEALTHY,
    MONITORING_SCHEMA_VERSION,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_SCORE_RUNTIME,
)


# ================================================================
# CHECK NAME
# ================================================================


SCORE_RUNTIME_CHECK_NAME = (
    "production_score_availability"
)


# ================================================================
# MONITOR
# ================================================================


def monitor_score_runtime(
    *,
    observed_at_utc,
    verify_source_hashes: bool = True,
) -> MonitoringCheck:
    """
    Observe runtime availability for every registered production score.

    Health policy
    -------------
    HEALTHY:
        Every registered score is AVAILABLE.

    UNHEALTHY:
        At least one registered score is not AVAILABLE.

    DEGRADED is intentionally not assigned by this check because no
    evidence-based partial-availability threshold has been frozen.
    """

    if not isinstance(
        verify_source_hashes,
        bool,
    ):
        raise TypeError(
            "verify_source_hashes must be a bool."
        )

    statuses = tuple(
        list_score_statuses(
            verify_source_hashes=(
                verify_source_hashes
            )
        )
    )

    total_scores = len(
        statuses
    )

    available_scores = sum(
        1
        for status
        in statuses
        if status.available
    )

    unavailable_scores = (
        total_scores
        -
        available_scores
    )

    status_counter = Counter(
        status.status.value
        for status
        in statuses
    )

    status_counts = MappingProxyType(
        {
            availability_status.value:
                status_counter.get(
                    availability_status.value,
                    0,
                )
            for availability_status
            in ScoreAvailabilityStatus
        }
    )

    all_available = (
        total_scores
        >
        0
        and
        unavailable_scores
        ==
        0
    )

    health = (
        MONITORING_HEALTHY
        if all_available
        else MONITORING_UNHEALTHY
    )

    if all_available:

        summary = (
            f"{available_scores}/{total_scores} "
            "production scores are available."
        )

    else:

        summary = (
            f"{available_scores}/{total_scores} "
            "production scores are available; "
            f"{unavailable_scores} unavailable."
        )

    details = tuple(
        (
            f"{status.score_id}: "
            f"{status.status.value} — "
            f"{status.detail}"
        )
        for status
        in statuses
    )

    metrics = {
        "total_scores":
            total_scores,

        "available_scores":
            available_scores,

        "unavailable_scores":
            unavailable_scores,

        "all_scores_available":
            all_available,

        "status_counts":
            status_counts,
    }

    return MonitoringCheck(
        schema_version=(
            MONITORING_SCHEMA_VERSION
        ),

        component=(
            MONITOR_COMPONENT_SCORE_RUNTIME
        ),

        check_name=(
            SCORE_RUNTIME_CHECK_NAME
        ),

        health=(
            health
        ),

        observed_at_utc=(
            observed_at_utc
        ),

        summary=(
            summary
        ),

        metrics=(
            metrics
        ),

        details=(
            details
        ),
    )