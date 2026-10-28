"""
Pure presentation logic for the portfolio ML / Model Explorer.

This module reads only the curated public model registry.

It does not:
- load model binaries;
- rescore data;
- reconstruct private research datasets;
- invent missing metrics;
- modify governance.
"""

from __future__ import annotations

from typing import Any


EXPECTED_SYSTEM_IDS = (
    "reactivation",
    "revenue_leakage",
    "future_customer_value",
    "repeat_purchase",
    "recommendation",
    "demand_forecast",
)


DECISION_ENGINE_SYSTEM_IDS = (
    "reactivation",
    "revenue_leakage",
    "future_customer_value",
    "recommendation",
)


STANDALONE_SYSTEM_IDS = (
    "repeat_purchase",
    "demand_forecast",
)


def validate_model_registry(
    registry: dict[str, Any],
) -> None:
    """
    Fail closed if the public model registry drifts.
    """

    if not isinstance(
        registry,
        dict,
    ):
        raise TypeError(
            "Model registry must be a dictionary."
        )

    if registry.get(
        "system_count"
    ) != 6:
        raise RuntimeError(
            "Model Explorer requires six intelligence systems."
        )

    systems = registry.get(
        "systems"
    )

    if not isinstance(
        systems,
        list,
    ):
        raise RuntimeError(
            "Model Explorer registry systems are invalid."
        )

    if len(
        systems
    ) != 6:
        raise RuntimeError(
            "Model Explorer requires exactly six systems."
        )

    observed_ids = tuple(
        system.get(
            "system_id"
        )
        for system in systems
    )

    if observed_ids != EXPECTED_SYSTEM_IDS:
        raise RuntimeError(
            "Model Explorer system order or IDs have drifted."
        )

    required = {
        "system_id",
        "display_name",
        "model_name",
        "model_type",
        "model_version",
        "target_name",
        "prediction_horizon",
        "governance",
        "metrics",
        "limitations",
    }

    for system in systems:

        missing = (
            required
            -
            set(
                system
            )
        )

        if missing:
            raise RuntimeError(
                (
                    "Model Explorer system "
                    f"{system.get('system_id')!r} "
                    "is missing required fields: "
                    f"{sorted(missing)}"
                )
            )


def systems_by_id(
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Return deterministic system lookup.
    """

    validate_model_registry(
        registry
    )

    return {
        system[
            "system_id"
        ]:
            system
        for system in registry[
            "systems"
        ]
    }


def governance_label(
    governance: Any,
) -> str:
    """
    Convert simple or channel-specific governance into display text.
    """

    if isinstance(
        governance,
        str,
    ):
        return governance

    if isinstance(
        governance,
        dict,
    ):

        parts = []

        for key, value in governance.items():

            label = (
                str(
                    key
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

            parts.append(
                f"{label}: {value}"
            )

        return " · ".join(
            parts
        )

    return "Not Available"


def prediction_horizon_label(
    horizon: Any,
) -> str:
    """
    Present public horizon metadata without changing meaning.
    """

    if horizon is None:
        return "Not Available"

    if isinstance(
        horizon,
        list,
    ):

        values = ", ".join(
            str(
                value
            )
            for value in horizon
        )

        return (
            f"{values} weeks"
        )

    return (
        f"{horizon} days"
    )


def system_role(
    system_id: str,
) -> str:
    """
    Explain where each intelligence system enters the production architecture.
    """

    if system_id in DECISION_ENGINE_SYSTEM_IDS:
        return (
            "Feeds the governed customer Decision Engine"
        )

    if system_id == "repeat_purchase":
        return (
            "Standalone customer propensity output; intentionally excluded "
            "from mature current-state action orchestration"
        )

    if system_id == "demand_forecast":
        return (
            "Standalone weekly planning system; intentionally excluded "
            "from customer-level action orchestration"
        )

    raise KeyError(
        f"Unknown system ID: {system_id}"
    )


def metric_rows(
    system: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Flatten governed public metrics into presentation rows.

    Nested dictionaries are preserved as distinct metric rows.
    """

    metrics = system.get(
        "metrics"
    )

    if not isinstance(
        metrics,
        dict,
    ) or not metrics:
        return []

    rows = []

    for key, value in metrics.items():

        if isinstance(
            value,
            dict,
        ):

            for nested_key, nested_value in value.items():

                rows.append(
                    {
                        "metric":
                            (
                                f"{key}.{nested_key}"
                            ),

                        "value":
                            nested_value,
                    }
                )

        else:

            rows.append(
                {
                    "metric":
                        key,

                    "value":
                        value,
                }
            )

    return rows


def metric_count(
    system: dict[str, Any],
) -> int:
    """
    Count public governed metric rows.
    """

    return len(
        metric_rows(
            system
        )
    )


def approved_use_rows(
    system: dict[str, Any],
) -> list[str]:
    """
    Normalise approved-use metadata for display only.
    """

    approved_use = system.get(
        "approved_use"
    )

    if approved_use is None:
        return []

    if isinstance(
        approved_use,
        str,
    ):
        return [
            approved_use
        ]

    if isinstance(
        approved_use,
        list,
    ):
        return [
            str(
                item
            )
            for item in approved_use
        ]

    if isinstance(
        approved_use,
        dict,
    ):

        return [
            (
                f"{str(key).replace('_', ' ').title()}: "
                f"{value}"
            )
            for key, value
            in approved_use.items()
        ]

    return [
        str(
            approved_use
        )
    ]


def limitation_rows(
    system: dict[str, Any],
) -> list[str]:
    """
    Return explicit public limitations.
    """

    limitations = system.get(
        "limitations"
    )

    if not isinstance(
        limitations,
        list,
    ):
        return []

    return [
        str(
            value
        )
        for value in limitations
    ]


def architecture_rows(
    system: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Flatten optional public architecture metadata.
    """

    architecture = system.get(
        "architecture"
    )

    if not isinstance(
        architecture,
        dict,
    ):
        return []

    return [
        {
            "component":
                (
                    str(
                        key
                    )
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                ),

            "value":
                str(
                    value
                ),
        }
        for key, value
        in architecture.items()
    ]


def registry_summary(
    registry: dict[str, Any],
) -> dict[str, int]:
    """
    Summarise public governance and architecture coverage.
    """

    validate_model_registry(
        registry
    )

    systems = registry[
        "systems"
    ]

    simple_go = sum(
        1
        for system in systems
        if system.get(
            "governance"
        )
        ==
        "GO"
    )

    simple_conditional = sum(
        1
        for system in systems
        if system.get(
            "governance"
        )
        ==
        "CONDITIONAL GO"
    )

    channel_governed = sum(
        1
        for system in systems
        if isinstance(
            system.get(
                "governance"
            ),
            dict,
        )
    )

    with_public_metrics = sum(
        1
        for system in systems
        if metric_count(
            system
        )
        >
        0
    )

    return {
        "system_count":
            len(
                systems
            ),

        "decision_engine_inputs":
            len(
                DECISION_ENGINE_SYSTEM_IDS
            ),

        "standalone_systems":
            len(
                STANDALONE_SYSTEM_IDS
            ),

        "simple_go":
            simple_go,

        "simple_conditional_go":
            simple_conditional,

        "channel_governed":
            channel_governed,

        "systems_with_public_metrics":
            with_public_metrics,
    }