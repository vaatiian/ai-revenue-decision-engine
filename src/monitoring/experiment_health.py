"""
Experiment-store health monitoring.

The experiment store is historical execution evidence.

Monitoring policy
-----------------
HEALTHY:
    The configured store exists, parses successfully, and contains at
    least one valid experiment run.

DEGRADED:
    The store is missing or valid but empty. This indicates reduced
    observability, not production model failure.

UNHEALTHY:
    The store exists but cannot be safely parsed or validated.

Historical experiment outcomes do not independently determine current
system health. A prior FAILED run remains useful audit evidence and must
not permanently mark production unhealthy.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MappingProxyType

from src.experiments.spec import (
    EXPERIMENT_RUN_STATUSES,
)

from src.experiments.store import (
    DEFAULT_EXPERIMENT_STORE_PATH,
    list_experiment_runs,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_SCHEMA_VERSION,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_EXPERIMENT_STORE,
)


# ================================================================
# CHECK NAME
# ================================================================


EXPERIMENT_STORE_CHECK_NAME = (
    "experiment_store_integrity"
)


# ================================================================
# HELPERS
# ================================================================


def _status_counts(
    records,
):
    """
    Count all frozen experiment statuses.
    """

    counter = Counter(
        record.status
        for record
        in records
    )

    return MappingProxyType(
        {
            status:
                counter.get(
                    status,
                    0,
                )
            for status
            in EXPERIMENT_RUN_STATUSES
        }
    )


# ================================================================
# MONITOR
# ================================================================


def monitor_experiment_store(
    *,
    observed_at_utc,
    path: str | Path = (
        DEFAULT_EXPERIMENT_STORE_PATH
    ),
) -> MonitoringCheck:
    """
    Observe structural health of the append-only experiment store.

    The check does not infer current model quality from historical run
    outcomes.
    """

    store_path = Path(
        path
    )

    store_exists = (
        store_path.exists()
    )

    try:

        records = tuple(
            list_experiment_runs(
                store_path
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
                MONITOR_COMPONENT_EXPERIMENT_STORE
            ),

            check_name=(
                EXPERIMENT_STORE_CHECK_NAME
            ),

            health=(
                MONITORING_UNHEALTHY
            ),

            observed_at_utc=(
                observed_at_utc
            ),

            summary=(
                "Experiment store failed integrity validation."
            ),

            metrics={
                "store_exists":
                    store_exists,

                "total_runs":
                    0,

                "status_counts":
                    MappingProxyType(
                        {
                            status:
                                0
                            for status
                            in EXPERIMENT_RUN_STATUSES
                        }
                    ),
            },

            details=(
                f"{type(exc).__name__}: {exc}",
            ),
        )

    total_runs = len(
        records
    )

    counts = (
        _status_counts(
            records
        )
    )

    if not store_exists:

        health = (
            MONITORING_DEGRADED
        )

        summary = (
            "Experiment store is missing; "
            "no historical run evidence is currently available."
        )

        details = (
            f"Expected experiment store: {store_path}",
        )

    elif total_runs == 0:

        health = (
            MONITORING_DEGRADED
        )

        summary = (
            "Experiment store is valid but contains no experiment runs."
        )

        details = (
            f"Experiment store: {store_path}",
        )

    else:

        health = (
            MONITORING_HEALTHY
        )

        summary = (
            f"Experiment store is valid with {total_runs} "
            "historical run"
            f"{'' if total_runs == 1 else 's'}."
        )

        details = tuple(
            (
                f"{record.run_id}: "
                f"{record.score_id} — "
                f"{record.status}"
            )
            for record
            in records
        )

    metrics = {
        "store_exists":
            store_exists,

        "total_runs":
            total_runs,

        "status_counts":
            counts,
    }

    return MonitoringCheck(
        schema_version=(
            MONITORING_SCHEMA_VERSION
        ),

        component=(
            MONITOR_COMPONENT_EXPERIMENT_STORE
        ),

        check_name=(
            EXPERIMENT_STORE_CHECK_NAME
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