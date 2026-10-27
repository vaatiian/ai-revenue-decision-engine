"""
Decision Engine output health monitoring.

This module observes the canonical Decision Engine output validator.

Monitoring policy
-----------------
HEALTHY:
    The decision output passes canonical validation and contains at least
    one customer decision.

DEGRADED:
    The decision output is structurally valid but empty.

UNHEALTHY:
    The decision output violates the canonical Decision Engine contract.

The monitor does not recreate or weaken Decision Engine governance rules.
"""

from __future__ import annotations

import pandas as pd

from src.decision_engine.engine import (
    validate_decision_engine_output,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_SCHEMA_VERSION,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_DECISION_OUTPUT,
)


# ================================================================
# CHECK NAME
# ================================================================


DECISION_OUTPUT_CHECK_NAME = (
    "decision_output_integrity"
)


# ================================================================
# MONITOR
# ================================================================


def monitor_decision_output(
    output,
    *,
    observed_at_utc,
    decision_date=None,
) -> MonitoringCheck:
    """
    Observe canonical Decision Engine output health.
    """

    row_count = (
        len(
            output
        )
        if isinstance(
            output,
            pd.DataFrame,
        )
        else 0
    )

    try:

        validate_decision_engine_output(
            output,
            decision_date=(
                decision_date
            ),
        )

    except (
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:

        return MonitoringCheck(
            schema_version=(
                MONITORING_SCHEMA_VERSION
            ),

            component=(
                MONITOR_COMPONENT_DECISION_OUTPUT
            ),

            check_name=(
                DECISION_OUTPUT_CHECK_NAME
            ),

            health=(
                MONITORING_UNHEALTHY
            ),

            observed_at_utc=(
                observed_at_utc
            ),

            summary=(
                "Decision Engine output failed canonical validation."
            ),

            metrics={
                "row_count":
                    row_count,

                "unique_customers":
                    0,
            },

            details=(
                f"{type(exc).__name__}: {exc}",
            ),
        )

    if output.empty:

        return MonitoringCheck(
            schema_version=(
                MONITORING_SCHEMA_VERSION
            ),

            component=(
                MONITOR_COMPONENT_DECISION_OUTPUT
            ),

            check_name=(
                DECISION_OUTPUT_CHECK_NAME
            ),

            health=(
                MONITORING_DEGRADED
            ),

            observed_at_utc=(
                observed_at_utc
            ),

            summary=(
                "Decision Engine output is valid but contains no "
                "customer decisions."
            ),

            metrics={
                "row_count":
                    0,

                "unique_customers":
                    0,
            },

            details=(
                "Canonical validation passed for an empty decision frame.",
            ),
        )

    unique_customers = (
        output[
            "customer"
        ]
        .nunique(
            dropna=True
        )
    )

    decision_dates = (
        pd.to_datetime(
            output[
                "decision_date"
            ],
            errors="coerce",
        )
    )

    canonical_decision_date = (
        decision_dates
        .iloc[
            0
        ]
        .date()
        .isoformat()
    )

    return MonitoringCheck(
        schema_version=(
            MONITORING_SCHEMA_VERSION
        ),

        component=(
            MONITOR_COMPONENT_DECISION_OUTPUT
        ),

        check_name=(
            DECISION_OUTPUT_CHECK_NAME
        ),

        health=(
            MONITORING_HEALTHY
        ),

        observed_at_utc=(
            observed_at_utc
        ),

        summary=(
            f"Decision Engine output is valid for "
            f"{row_count} customer"
            f"{'' if row_count == 1 else 's'}."
        ),

        metrics={
            "row_count":
                row_count,

            "unique_customers":
                unique_customers,

            "decision_date":
                canonical_decision_date,
        },

        details=(
            (
                "Canonical Decision Engine output validation passed."
            ),
        ),
    )