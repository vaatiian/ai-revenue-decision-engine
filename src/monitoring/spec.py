"""
Frozen production monitoring contract.

Monitoring is observational.

It does not:
- train or retrain models;
- modify model artifacts or manifests;
- change Decision Engine actions;
- change delivery routing;
- rewrite experiment or delivery history;
- invent statistical drift thresholds.

Phase 10 initially monitors operational and governance health using the
production contracts that already exist elsewhere in the system.
"""

from __future__ import annotations


# ================================================================
# SCHEMA VERSION
# ================================================================


MONITORING_SCHEMA_VERSION = "1.0.0"


# ================================================================
# HEALTH STATES
# ================================================================


MONITORING_HEALTHY = (
    "HEALTHY"
)

MONITORING_DEGRADED = (
    "DEGRADED"
)

MONITORING_UNHEALTHY = (
    "UNHEALTHY"
)


MONITORING_HEALTH_STATES = (
    MONITORING_HEALTHY,
    MONITORING_DEGRADED,
    MONITORING_UNHEALTHY,
)


# ================================================================
# COMPONENTS
# ================================================================


MONITOR_COMPONENT_SCORE_RUNTIME = (
    "score_runtime"
)

MONITOR_COMPONENT_EXPERIMENT_STORE = (
    "experiment_store"
)

MONITOR_COMPONENT_DECISION_OUTPUT = (
    "decision_output"
)

MONITOR_COMPONENT_DELIVERY_AUDIT = (
    "delivery_audit"
)


MONITOR_COMPONENTS = (
    MONITOR_COMPONENT_SCORE_RUNTIME,
    MONITOR_COMPONENT_EXPERIMENT_STORE,
    MONITOR_COMPONENT_DECISION_OUTPUT,
    MONITOR_COMPONENT_DELIVERY_AUDIT,
)


# ================================================================
# CHECK RESULT SCHEMA
# ================================================================


MONITORING_CHECK_FIELDS = (
    "schema_version",
    "component",
    "check_name",
    "health",
    "observed_at_utc",
    "summary",
    "metrics",
    "details",
)


MONITORING_REQUIRED_FIELDS = (
    "schema_version",
    "component",
    "check_name",
    "health",
    "observed_at_utc",
    "summary",
    "metrics",
    "details",
)


# ================================================================
# SYSTEM SNAPSHOT SCHEMA
# ================================================================


MONITORING_SNAPSHOT_FIELDS = (
    "schema_version",
    "snapshot_id",
    "observed_at_utc",
    "overall_health",
    "checks",
)


# ================================================================
# HEALTH ORDER
# ================================================================


MONITORING_HEALTH_RANK = (
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


# ================================================================
# VALIDATION HELPERS
# ================================================================


def validate_monitoring_health(
    health: str,
) -> str:
    """
    Validate and return one frozen monitoring health state.
    """

    if not isinstance(
        health,
        str,
    ):
        raise TypeError(
            "health must be a string."
        )

    if not health.strip():
        raise ValueError(
            "health must not be empty."
        )

    if (
        health
        not in
        MONITORING_HEALTH_STATES
    ):
        raise ValueError(
            "Unsupported monitoring health: "
            f"{health}. "
            "Supported health states: "
            f"{MONITORING_HEALTH_STATES}"
        )

    return health


def validate_monitor_component(
    component: str,
) -> str:
    """
    Validate and return one frozen monitoring component.
    """

    if not isinstance(
        component,
        str,
    ):
        raise TypeError(
            "component must be a string."
        )

    if not component.strip():
        raise ValueError(
            "component must not be empty."
        )

    if (
        component
        not in
        MONITOR_COMPONENTS
    ):
        raise ValueError(
            "Unsupported monitoring component: "
            f"{component}. "
            "Supported components: "
            f"{MONITOR_COMPONENTS}"
        )

    return component


def monitoring_health_rank(
    health: str,
) -> int:
    """
    Return severity ordering for one validated health state.
    """

    validate_monitoring_health(
        health
    )

    return dict(
        MONITORING_HEALTH_RANK
    )[
        health
    ]


def worst_monitoring_health(
    health_states,
) -> str:
    """
    Return the worst health state in a non-empty iterable.
    """

    if isinstance(
        health_states,
        str,
    ):
        raise TypeError(
            "health_states must be an iterable of health-state strings."
        )

    try:

        values = tuple(
            health_states
        )

    except TypeError as exc:

        raise TypeError(
            "health_states must be an iterable of health-state strings."
        ) from exc

    if not values:
        raise ValueError(
            "health_states must not be empty."
        )

    for health in values:

        validate_monitoring_health(
            health
        )

    return max(
        values,
        key=monitoring_health_rank,
    )


# ================================================================
# SPEC VALIDATION
# ================================================================


def validate_monitoring_spec() -> None:
    """
    Fail closed if the monitoring contract drifts internally.
    """

    if not MONITORING_SCHEMA_VERSION:
        raise ValueError(
            "Monitoring schema version must not be empty."
        )

    if len(
        MONITORING_HEALTH_STATES
    ) != len(
        set(
            MONITORING_HEALTH_STATES
        )
    ):
        raise ValueError(
            "Monitoring health states must be unique."
        )

    if len(
        MONITOR_COMPONENTS
    ) != len(
        set(
            MONITOR_COMPONENTS
        )
    ):
        raise ValueError(
            "Monitoring components must be unique."
        )

    if len(
        MONITORING_CHECK_FIELDS
    ) != len(
        set(
            MONITORING_CHECK_FIELDS
        )
    ):
        raise ValueError(
            "Monitoring check fields must be unique."
        )

    if (
        MONITORING_REQUIRED_FIELDS
        !=
        MONITORING_CHECK_FIELDS
    ):
        raise ValueError(
            "Every monitoring check field is required in schema version 1.0.0."
        )

    if len(
        MONITORING_SNAPSHOT_FIELDS
    ) != len(
        set(
            MONITORING_SNAPSHOT_FIELDS
        )
    ):
        raise ValueError(
            "Monitoring snapshot fields must be unique."
        )

    ranked_health_states = tuple(
        health
        for (
            health,
            _,
        )
        in MONITORING_HEALTH_RANK
    )

    if (
        ranked_health_states
        !=
        MONITORING_HEALTH_STATES
    ):
        raise ValueError(
            "Monitoring health rank must cover health states in canonical "
            "order."
        )

    rank_values = tuple(
        rank
        for (
            _,
            rank,
        )
        in MONITORING_HEALTH_RANK
    )

    if (
        rank_values
        !=
        (
            0,
            1,
            2,
        )
    ):
        raise ValueError(
            "Monitoring health ranks must remain 0, 1, 2."
        )

    for health in (
        MONITORING_HEALTH_STATES
    ):

        validate_monitoring_health(
            health
        )

    for component in (
        MONITOR_COMPONENTS
    ):

        validate_monitor_component(
            component
        )

    if (
        worst_monitoring_health(
            (
                MONITORING_HEALTHY,
                MONITORING_DEGRADED,
                MONITORING_UNHEALTHY,
            )
        )
        !=
        MONITORING_UNHEALTHY
    ):
        raise ValueError(
            "Monitoring health aggregation must fail toward the worst state."
        )


validate_monitoring_spec()