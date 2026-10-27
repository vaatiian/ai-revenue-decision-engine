"""
Immutable system-level monitoring snapshot.

A monitoring snapshot aggregates already-validated MonitoringCheck records.
It does not re-run component logic and does not change component health.

Overall health is the worst health state present across component checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.monitoring.check import (
    MonitoringCheck,
    monitoring_check_to_dict,
)

from src.monitoring.spec import (
    MONITORING_SCHEMA_VERSION,
    MONITORING_SNAPSHOT_FIELDS,
    validate_monitoring_health,
    worst_monitoring_health,
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


def _freeze_checks(
    checks,
) -> tuple[
    MonitoringCheck,
    ...,
]:
    """
    Validate and freeze one monitoring-check collection.
    """

    if isinstance(
        checks,
        MonitoringCheck,
    ):
        raise TypeError(
            "checks must be an iterable of MonitoringCheck records."
        )

    try:

        values = tuple(
            checks
        )

    except TypeError as exc:

        raise TypeError(
            "checks must be an iterable of MonitoringCheck records."
        ) from exc

    if not values:
        raise ValueError(
            "checks must not be empty."
        )

    for check in values:

        if not isinstance(
            check,
            MonitoringCheck,
        ):
            raise TypeError(
                "Every check must be a MonitoringCheck."
            )

    component_names = [
        check.component
        for check
        in values
    ]

    if len(
        component_names
    ) != len(
        set(
            component_names
        )
    ):
        raise ValueError(
            "Monitoring snapshot cannot contain duplicate components."
        )

    return values


# ================================================================
# IMMUTABLE SNAPSHOT
# ================================================================


@dataclass(
    frozen=True
)
class MonitoringSnapshot:
    """
    One immutable system-level monitoring snapshot.
    """

    schema_version: str

    snapshot_id: str

    observed_at_utc: object

    overall_health: str

    checks: tuple[
        MonitoringCheck,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize one monitoring snapshot.
        """

        if (
            self.schema_version
            !=
            MONITORING_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported monitoring schema_version."
            )

        _validate_non_empty_string(
            self.snapshot_id,
            "snapshot_id",
        )

        validate_monitoring_health(
            self.overall_health
        )

        frozen_checks = (
            _freeze_checks(
                self.checks
            )
        )

        expected_health = (
            worst_monitoring_health(
                check.health
                for check
                in frozen_checks
            )
        )

        if (
            self.overall_health
            !=
            expected_health
        ):
            raise ValueError(
                "overall_health does not match the worst component health."
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
            "checks",
            frozen_checks,
        )


# ================================================================
# BUILDER
# ================================================================


def build_monitoring_snapshot(
    checks,
    *,
    snapshot_id: str,
    observed_at_utc,
) -> MonitoringSnapshot:
    """
    Build one governed monitoring snapshot from component checks.
    """

    frozen_checks = (
        _freeze_checks(
            checks
        )
    )

    overall_health = (
        worst_monitoring_health(
            check.health
            for check
            in frozen_checks
        )
    )

    return MonitoringSnapshot(
        schema_version=(
            MONITORING_SCHEMA_VERSION
        ),

        snapshot_id=(
            snapshot_id
        ),

        observed_at_utc=(
            observed_at_utc
        ),

        overall_health=(
            overall_health
        ),

        checks=(
            frozen_checks
        ),
    )


# ================================================================
# SERIALISATION
# ================================================================


def monitoring_snapshot_to_dict(
    snapshot: MonitoringSnapshot,
) -> dict[str, Any]:
    """
    Convert one MonitoringSnapshot to a serializable dictionary.
    """

    if not isinstance(
        snapshot,
        MonitoringSnapshot,
    ):
        raise TypeError(
            "snapshot must be a MonitoringSnapshot."
        )

    values = {
        "schema_version":
            snapshot.schema_version,

        "snapshot_id":
            snapshot.snapshot_id,

        "observed_at_utc":
            snapshot.observed_at_utc.isoformat(),

        "overall_health":
            snapshot.overall_health,

        "checks":
            [
                monitoring_check_to_dict(
                    check
                )
                for check
                in snapshot.checks
            ],
    }

    if tuple(
        values.keys()
    ) != MONITORING_SNAPSHOT_FIELDS:
        raise RuntimeError(
            "Monitoring snapshot dictionary does not match "
            "MONITORING_SNAPSHOT_FIELDS."
        )

    return values