"""
Delivery-audit health monitoring.

The delivery audit is an append-only historical ledger of governed
delivery payloads.

Monitoring policy
-----------------
HEALTHY:
    The configured delivery audit exists, validates successfully, and
    contains at least one payload.

DEGRADED:
    The audit is missing or valid but empty. This represents reduced
    delivery observability rather than proven delivery failure.

UNHEALTHY:
    The audit exists but fails structural or record validation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MappingProxyType

from src.delivery.audit import (
    DEFAULT_DELIVERY_AUDIT_PATH,
    list_delivery_audit,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITIONS,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_SCHEMA_VERSION,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_DELIVERY_AUDIT,
)


# ================================================================
# CHECK NAME
# ================================================================


DELIVERY_AUDIT_CHECK_NAME = (
    "delivery_audit_integrity"
)


# ================================================================
# HELPERS
# ================================================================


def _disposition_counts(
    payloads,
):
    """
    Count all frozen delivery dispositions.
    """

    counter = Counter(
        payload.delivery_disposition
        for payload
        in payloads
    )

    return MappingProxyType(
        {
            disposition:
                counter.get(
                    disposition,
                    0,
                )
            for disposition
            in DELIVERY_DISPOSITIONS
        }
    )


# ================================================================
# MONITOR
# ================================================================


def monitor_delivery_audit(
    *,
    observed_at_utc,
    path: str | Path = (
        DEFAULT_DELIVERY_AUDIT_PATH
    ),
) -> MonitoringCheck:
    """
    Observe structural health of the append-only delivery audit.
    """

    audit_path = Path(
        path
    )

    audit_exists = (
        audit_path.exists()
    )

    try:

        payloads = tuple(
            list_delivery_audit(
                audit_path
            )
        )

    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:

        return MonitoringCheck(
            schema_version=(
                MONITORING_SCHEMA_VERSION
            ),

            component=(
                MONITOR_COMPONENT_DELIVERY_AUDIT
            ),

            check_name=(
                DELIVERY_AUDIT_CHECK_NAME
            ),

            health=(
                MONITORING_UNHEALTHY
            ),

            observed_at_utc=(
                observed_at_utc
            ),

            summary=(
                "Delivery audit failed integrity validation."
            ),

            metrics={
                "audit_exists":
                    audit_exists,

                "total_deliveries":
                    0,

                "disposition_counts":
                    MappingProxyType(
                        {
                            disposition:
                                0
                            for disposition
                            in DELIVERY_DISPOSITIONS
                        }
                    ),
            },

            details=(
                f"{type(exc).__name__}: {exc}",
            ),
        )

    total_deliveries = len(
        payloads
    )

    counts = (
        _disposition_counts(
            payloads
        )
    )

    if not audit_exists:

        health = (
            MONITORING_DEGRADED
        )

        summary = (
            "Delivery audit is missing; "
            "no delivery history is currently available."
        )

        details = (
            f"Expected delivery audit: {audit_path}",
        )

    elif total_deliveries == 0:

        health = (
            MONITORING_DEGRADED
        )

        summary = (
            "Delivery audit is valid but contains no delivery records."
        )

        details = (
            f"Delivery audit: {audit_path}",
        )

    else:

        health = (
            MONITORING_HEALTHY
        )

        summary = (
            f"Delivery audit is valid with {total_deliveries} "
            "delivery record"
            f"{'' if total_deliveries == 1 else 's'}."
        )

        details = tuple(
            (
                f"{payload.delivery_id}: "
                f"{payload.customer} — "
                f"{payload.delivery_disposition}"
            )
            for payload
            in payloads
        )

    return MonitoringCheck(
        schema_version=(
            MONITORING_SCHEMA_VERSION
        ),

        component=(
            MONITOR_COMPONENT_DELIVERY_AUDIT
        ),

        check_name=(
            DELIVERY_AUDIT_CHECK_NAME
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

        metrics={
            "audit_exists":
                audit_exists,

            "total_deliveries":
                total_deliveries,

            "disposition_counts":
                counts,
        },

        details=(
            details
        ),
    )