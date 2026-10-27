"""
Operational alert policy for production monitoring.

Alerts are derived from already-governed monitoring health states.

This module does not:
- invent model-performance thresholds;
- reinterpret component health;
- retrain or repair models;
- change Decision Engine actions;
- change delivery routing.

Severity policy
---------------
HEALTHY:
    INFO, no alert required.

DEGRADED:
    WARNING, alert required.

UNHEALTHY:
    CRITICAL, alert required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.snapshot import (
    MonitoringSnapshot,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_UNHEALTHY,
    validate_monitoring_health,
)


# ================================================================
# SEVERITIES
# ================================================================


ALERT_SEVERITY_INFO = (
    "INFO"
)

ALERT_SEVERITY_WARNING = (
    "WARNING"
)

ALERT_SEVERITY_CRITICAL = (
    "CRITICAL"
)


ALERT_SEVERITIES = (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARNING,
    ALERT_SEVERITY_CRITICAL,
)


# ================================================================
# HEALTH → ALERT POLICY
# ================================================================


ALERT_POLICY = (
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


# ================================================================
# ALERT SCHEMA
# ================================================================


MONITORING_ALERT_FIELDS = (
    "source_type",
    "source_name",
    "health",
    "severity",
    "alert_required",
    "observed_at_utc",
    "summary",
)


# ================================================================
# HELPERS
# ================================================================


def _validate_non_empty_string(
    value,
    name: str,
) -> str:
    """
    Validate and return one non-empty string.
    """

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{name} must not be empty."
        )

    return value


def _normalise_utc_timestamp(
    value,
):
    """
    Convert one timestamp-like value to timezone-aware UTC.
    """

    try:

        timestamp = pd.Timestamp(
            value
        )

    except Exception as exc:

        raise ValueError(
            "observed_at_utc is not a valid timestamp."
        ) from exc

    if pd.isna(
        timestamp
    ):
        raise ValueError(
            "observed_at_utc is not a valid timestamp."
        )

    if timestamp.tzinfo is None:

        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return timestamp.to_pydatetime()


def get_alert_policy(
    health: str,
) -> tuple[
    str,
    bool,
]:
    """
    Return severity and alert requirement for one health state.
    """

    validate_monitoring_health(
        health
    )

    mapping = {
        state:
            (
                severity,
                alert_required,
            )
        for (
            state,
            severity,
            alert_required,
        )
        in ALERT_POLICY
    }

    return mapping[
        health
    ]


# ================================================================
# IMMUTABLE ALERT
# ================================================================


@dataclass(
    frozen=True
)
class MonitoringAlert:
    """
    One immutable operational alert derived from monitoring health.
    """

    source_type: str

    source_name: str

    health: str

    severity: str

    alert_required: bool

    observed_at_utc: object

    summary: str

    def __post_init__(
        self,
    ) -> None:
        """
        Validate one monitoring alert.
        """

        _validate_non_empty_string(
            self.source_type,
            "source_type",
        )

        _validate_non_empty_string(
            self.source_name,
            "source_name",
        )

        validate_monitoring_health(
            self.health
        )

        if (
            self.severity
            not in
            ALERT_SEVERITIES
        ):
            raise ValueError(
                "Unsupported alert severity: "
                f"{self.severity}."
            )

        if not isinstance(
            self.alert_required,
            bool,
        ):
            raise TypeError(
                "alert_required must be a bool."
            )

        expected_severity, expected_required = (
            get_alert_policy(
                self.health
            )
        )

        if (
            self.severity
            !=
            expected_severity
        ):
            raise ValueError(
                "Alert severity does not match monitoring health policy."
            )

        if (
            self.alert_required
            is not
            expected_required
        ):
            raise ValueError(
                "alert_required does not match monitoring health policy."
            )

        _validate_non_empty_string(
            self.summary,
            "summary",
        )

        object.__setattr__(
            self,
            "observed_at_utc",
            _normalise_utc_timestamp(
                self.observed_at_utc
            ),
        )


# ================================================================
# DERIVATION
# ================================================================


def alert_from_check(
    check: MonitoringCheck,
) -> MonitoringAlert:
    """
    Derive one operational alert from a MonitoringCheck.
    """

    if not isinstance(
        check,
        MonitoringCheck,
    ):
        raise TypeError(
            "check must be a MonitoringCheck."
        )

    severity, required = (
        get_alert_policy(
            check.health
        )
    )

    return MonitoringAlert(
        source_type=(
            "component"
        ),

        source_name=(
            check.component
        ),

        health=(
            check.health
        ),

        severity=(
            severity
        ),

        alert_required=(
            required
        ),

        observed_at_utc=(
            check.observed_at_utc
        ),

        summary=(
            check.summary
        ),
    )


def alert_from_snapshot(
    snapshot: MonitoringSnapshot,
) -> MonitoringAlert:
    """
    Derive one system-level operational alert from a snapshot.
    """

    if not isinstance(
        snapshot,
        MonitoringSnapshot,
    ):
        raise TypeError(
            "snapshot must be a MonitoringSnapshot."
        )

    severity, required = (
        get_alert_policy(
            snapshot.overall_health
        )
    )

    return MonitoringAlert(
        source_type=(
            "system"
        ),

        source_name=(
            snapshot.snapshot_id
        ),

        health=(
            snapshot.overall_health
        ),

        severity=(
            severity
        ),

        alert_required=(
            required
        ),

        observed_at_utc=(
            snapshot.observed_at_utc
        ),

        summary=(
            "Monitoring snapshot "
            f"{snapshot.snapshot_id} is "
            f"{snapshot.overall_health}."
        ),
    )


# ================================================================
# SERIALISATION
# ================================================================


def monitoring_alert_to_dict(
    alert: MonitoringAlert,
) -> dict[str, Any]:
    """
    Convert one MonitoringAlert to a serializable dictionary.
    """

    if not isinstance(
        alert,
        MonitoringAlert,
    ):
        raise TypeError(
            "alert must be a MonitoringAlert."
        )

    values = {
        "source_type":
            alert.source_type,

        "source_name":
            alert.source_name,

        "health":
            alert.health,

        "severity":
            alert.severity,

        "alert_required":
            alert.alert_required,

        "observed_at_utc":
            alert.observed_at_utc.isoformat(),

        "summary":
            alert.summary,
    }

    if tuple(
        values.keys()
    ) != MONITORING_ALERT_FIELDS:
        raise RuntimeError(
            "Monitoring alert dictionary does not match "
            "MONITORING_ALERT_FIELDS."
        )

    return values