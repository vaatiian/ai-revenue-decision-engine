"""
Immutable production monitoring check result.

All monitoring components emit the same stable check contract.

Monitoring results are observational records only. They do not mutate
production state.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from src.monitoring.spec import (
    MONITORING_CHECK_FIELDS,
    MONITORING_SCHEMA_VERSION,
    validate_monitor_component,
    validate_monitoring_health,
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


def _freeze_metrics(
    metrics: Mapping[str, Any],
):
    """
    Freeze the top-level monitoring metrics mapping.
    """

    if not isinstance(
        metrics,
        Mapping,
    ):
        raise TypeError(
            "metrics must be a mapping."
        )

    copied = dict(
        metrics
    )

    for key in copied:

        if not isinstance(
            key,
            str,
        ):
            raise TypeError(
                "metrics keys must be strings."
            )

        if not key.strip():
            raise ValueError(
                "metrics keys must not be empty."
            )

    return MappingProxyType(
        copied
    )


def _freeze_details(
    details,
) -> tuple[str, ...]:
    """
    Freeze human-readable monitoring details.
    """

    if isinstance(
        details,
        str,
    ):
        raise TypeError(
            "details must be an iterable of strings."
        )

    try:

        values = tuple(
            details
        )

    except TypeError as exc:

        raise TypeError(
            "details must be an iterable of strings."
        ) from exc

    for value in values:

        _validate_non_empty_string(
            value,
            "detail",
        )

    return values


# ================================================================
# IMMUTABLE CHECK
# ================================================================


@dataclass(
    frozen=True
)
class MonitoringCheck:
    """
    One immutable production monitoring observation.
    """

    schema_version: str

    component: str

    check_name: str

    health: str

    observed_at_utc: object

    summary: str

    metrics: Mapping[str, Any]

    details: tuple[str, ...]

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize one monitoring check.
        """

        if (
            self.schema_version
            !=
            MONITORING_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported monitoring schema_version."
            )

        validate_monitor_component(
            self.component
        )

        _validate_non_empty_string(
            self.check_name,
            "check_name",
        )

        validate_monitoring_health(
            self.health
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

        object.__setattr__(
            self,
            "metrics",
            _freeze_metrics(
                self.metrics
            ),
        )

        object.__setattr__(
            self,
            "details",
            _freeze_details(
                self.details
            ),
        )


# ================================================================
# SERIALISATION
# ================================================================


def monitoring_check_to_dict(
    check: MonitoringCheck,
) -> dict[str, Any]:
    """
    Convert one MonitoringCheck to a serializable dictionary.
    """

    if not isinstance(
        check,
        MonitoringCheck,
    ):
        raise TypeError(
            "check must be a MonitoringCheck."
        )

    values = {
        "schema_version":
            check.schema_version,

        "component":
            check.component,

        "check_name":
            check.check_name,

        "health":
            check.health,

        "observed_at_utc":
            check.observed_at_utc.isoformat(),

        "summary":
            check.summary,

        "metrics":
            dict(
                check.metrics
            ),

        "details":
            list(
                check.details
            ),
    }

    if tuple(
        values.keys()
    ) != MONITORING_CHECK_FIELDS:
        raise RuntimeError(
            "Monitoring check dictionary does not match "
            "MONITORING_CHECK_FIELDS."
        )

    return values