"""
Frozen CRM/action-delivery contract.

The delivery subsystem is downstream of the Decision Engine.

It does not:
- score customers;
- select primary actions;
- reprioritise customers;
- choose products;
- modify experimentation requirements;
- invent communication channels.

It packages and routes already-governed Decision Engine outputs.

The Decision Engine remains the authoritative source for:
- primary action;
- priority;
- contact intensity;
- business action;
- decision reason;
- experiment requirement;
- recommendation channel;
- recommended product.
"""

from __future__ import annotations

from src.decision_engine.spec import (
    ACTIONABLE_ACTIONS,
    CONTACT_AUTOMATED_LOW_COST,
    CONTACT_CAMPAIGN,
    CONTACT_HIGH_TOUCH,
    CONTACT_HOLD_CONTROL,
    CONTACT_INTENSITIES,
    CONTACT_NO_MODEL_LED,
    DECISION_OUTPUT_COLUMNS,
    NON_ACTIONABLE_ACTIONS,
)


# ================================================================
# SCHEMA VERSION
# ================================================================


DELIVERY_SCHEMA_VERSION = "1.0.0"


# ================================================================
# DELIVERY ROUTES
# ================================================================


DELIVERY_ROUTE_HIGH_TOUCH = (
    "HIGH_TOUCH_QUEUE"
)

DELIVERY_ROUTE_CAMPAIGN = (
    "CAMPAIGN_QUEUE"
)

DELIVERY_ROUTE_AUTOMATED = (
    "AUTOMATED_QUEUE"
)

DELIVERY_ROUTE_HOLD_CONTROL = (
    "HOLD_CONTROL"
)

DELIVERY_ROUTE_SUPPRESSED = (
    "SUPPRESSED"
)


DELIVERY_ROUTES = (
    DELIVERY_ROUTE_HIGH_TOUCH,
    DELIVERY_ROUTE_CAMPAIGN,
    DELIVERY_ROUTE_AUTOMATED,
    DELIVERY_ROUTE_HOLD_CONTROL,
    DELIVERY_ROUTE_SUPPRESSED,
)


# ================================================================
# TREATMENT / NON-TREATMENT ROUTES
# ================================================================


TREATMENT_DELIVERY_ROUTES = (
    DELIVERY_ROUTE_HIGH_TOUCH,
    DELIVERY_ROUTE_CAMPAIGN,
    DELIVERY_ROUTE_AUTOMATED,
)


NON_TREATMENT_DELIVERY_ROUTES = (
    DELIVERY_ROUTE_HOLD_CONTROL,
    DELIVERY_ROUTE_SUPPRESSED,
)


# ================================================================
# CONTACT-INTENSITY → DELIVERY ROUTE
# ================================================================


DELIVERY_ROUTE_BY_CONTACT_INTENSITY = (
    (
        CONTACT_HIGH_TOUCH,
        DELIVERY_ROUTE_HIGH_TOUCH,
    ),
    (
        CONTACT_CAMPAIGN,
        DELIVERY_ROUTE_CAMPAIGN,
    ),
    (
        CONTACT_AUTOMATED_LOW_COST,
        DELIVERY_ROUTE_AUTOMATED,
    ),
    (
        CONTACT_HOLD_CONTROL,
        DELIVERY_ROUTE_HOLD_CONTROL,
    ),
    (
        CONTACT_NO_MODEL_LED,
        DELIVERY_ROUTE_SUPPRESSED,
    ),
)


# ================================================================
# DELIVERY DISPOSITION
# ================================================================


DELIVERY_DISPOSITION_TREATMENT = (
    "TREATMENT"
)

DELIVERY_DISPOSITION_HOLD_CONTROL = (
    "HOLD_CONTROL"
)

DELIVERY_DISPOSITION_SUPPRESSED = (
    "SUPPRESSED"
)


DELIVERY_DISPOSITIONS = (
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_DISPOSITION_HOLD_CONTROL,
    DELIVERY_DISPOSITION_SUPPRESSED,
)


DELIVERY_DISPOSITION_BY_ROUTE = (
    (
        DELIVERY_ROUTE_HIGH_TOUCH,
        DELIVERY_DISPOSITION_TREATMENT,
    ),
    (
        DELIVERY_ROUTE_CAMPAIGN,
        DELIVERY_DISPOSITION_TREATMENT,
    ),
    (
        DELIVERY_ROUTE_AUTOMATED,
        DELIVERY_DISPOSITION_TREATMENT,
    ),
    (
        DELIVERY_ROUTE_HOLD_CONTROL,
        DELIVERY_DISPOSITION_HOLD_CONTROL,
    ),
    (
        DELIVERY_ROUTE_SUPPRESSED,
        DELIVERY_DISPOSITION_SUPPRESSED,
    ),
)


# ================================================================
# REQUIRED DECISION INPUT
# ================================================================


DELIVERY_REQUIRED_DECISION_COLUMNS = (
    "customer",
    "decision_date",
    "customer_state",

    "primary_action",
    "primary_action_score",
    "action_priority_percentile",
    "priority_tier",

    "contact_intensity",
    "business_action",
    "decision_reason",
    "experiment_requirement",

    "recommended_product_channel",
    "recommended_product",
)


# ================================================================
# CANONICAL DELIVERY PAYLOAD
# ================================================================


DELIVERY_PAYLOAD_COLUMNS = (
    # Delivery identity
    "delivery_id",
    "delivery_created_at_utc",

    # Decision identity
    "customer",
    "decision_date",

    # Governed decision
    "customer_state",
    "primary_action",
    "primary_action_score",
    "action_priority_percentile",
    "priority_tier",

    # Prescribed commercial policy
    "contact_intensity",
    "delivery_route",
    "delivery_disposition",
    "business_action",
    "decision_reason",
    "experiment_requirement",

    # Product treatment context
    "recommended_product_channel",
    "recommended_product",
)


# ================================================================
# VALIDATION HELPERS
# ================================================================


def get_delivery_route(
    contact_intensity: str,
) -> str:
    """
    Map one frozen Decision Engine contact intensity to delivery routing.
    """

    if not isinstance(
        contact_intensity,
        str,
    ):
        raise TypeError(
            "contact_intensity must be a string."
        )

    if not contact_intensity.strip():
        raise ValueError(
            "contact_intensity must not be empty."
        )

    mapping = dict(
        DELIVERY_ROUTE_BY_CONTACT_INTENSITY
    )

    try:

        return mapping[
            contact_intensity
        ]

    except KeyError as exc:

        raise ValueError(
            "Unsupported contact_intensity: "
            f"{contact_intensity}"
        ) from exc


def get_delivery_disposition(
    delivery_route: str,
) -> str:
    """
    Return the frozen treatment disposition for one delivery route.
    """

    if not isinstance(
        delivery_route,
        str,
    ):
        raise TypeError(
            "delivery_route must be a string."
        )

    if not delivery_route.strip():
        raise ValueError(
            "delivery_route must not be empty."
        )

    mapping = dict(
        DELIVERY_DISPOSITION_BY_ROUTE
    )

    try:

        return mapping[
            delivery_route
        ]

    except KeyError as exc:

        raise ValueError(
            "Unsupported delivery_route: "
            f"{delivery_route}"
        ) from exc


def validate_delivery_spec() -> None:
    """
    Fail closed if the downstream delivery contract drifts internally.
    """

    if not DELIVERY_SCHEMA_VERSION:
        raise ValueError(
            "Delivery schema version must not be empty."
        )

    if len(
        DELIVERY_ROUTES
    ) != len(
        set(
            DELIVERY_ROUTES
        )
    ):
        raise ValueError(
            "Delivery routes must be unique."
        )

    if len(
        DELIVERY_DISPOSITIONS
    ) != len(
        set(
            DELIVERY_DISPOSITIONS
        )
    ):
        raise ValueError(
            "Delivery dispositions must be unique."
        )

    if len(
        DELIVERY_REQUIRED_DECISION_COLUMNS
    ) != len(
        set(
            DELIVERY_REQUIRED_DECISION_COLUMNS
        )
    ):
        raise ValueError(
            "Required delivery input columns must be unique."
        )

    if len(
        DELIVERY_PAYLOAD_COLUMNS
    ) != len(
        set(
            DELIVERY_PAYLOAD_COLUMNS
        )
    ):
        raise ValueError(
            "Delivery payload columns must be unique."
        )

    decision_columns = set(
        DECISION_OUTPUT_COLUMNS
    )

    missing_decision_columns = (
        set(
            DELIVERY_REQUIRED_DECISION_COLUMNS
        )
        -
        decision_columns
    )

    if missing_decision_columns:
        raise ValueError(
            "Delivery contract requires columns that are not in the "
            "canonical Decision Engine output: "
            f"{sorted(missing_decision_columns)}"
        )

    contact_policy_keys = tuple(
        contact_intensity
        for (
            contact_intensity,
            _,
        )
        in DELIVERY_ROUTE_BY_CONTACT_INTENSITY
    )

    if (
        contact_policy_keys
        !=
        CONTACT_INTENSITIES
    ):
        raise ValueError(
            "Delivery routing must cover the exact frozen Decision Engine "
            "contact-intensity policy in canonical order."
        )

    route_policy_values = tuple(
        delivery_route
        for (
            _,
            delivery_route,
        )
        in DELIVERY_ROUTE_BY_CONTACT_INTENSITY
    )

    if (
        route_policy_values
        !=
        DELIVERY_ROUTES
    ):
        raise ValueError(
            "Delivery routing values must match DELIVERY_ROUTES exactly."
        )

    disposition_route_keys = tuple(
        delivery_route
        for (
            delivery_route,
            _,
        )
        in DELIVERY_DISPOSITION_BY_ROUTE
    )

    if (
        disposition_route_keys
        !=
        DELIVERY_ROUTES
    ):
        raise ValueError(
            "Every delivery route must have exactly one disposition."
        )

    if set(
        TREATMENT_DELIVERY_ROUTES
    ) & set(
        NON_TREATMENT_DELIVERY_ROUTES
    ):
        raise ValueError(
            "Treatment and non-treatment delivery routes cannot overlap."
        )

    if (
        set(
            TREATMENT_DELIVERY_ROUTES
        )
        |
        set(
            NON_TREATMENT_DELIVERY_ROUTES
        )
    ) != set(
        DELIVERY_ROUTES
    ):
        raise ValueError(
            "Treatment/non-treatment routing must partition DELIVERY_ROUTES."
        )

    if not ACTIONABLE_ACTIONS:
        raise ValueError(
            "Decision Engine actionable-action contract must not be empty."
        )

    if not NON_ACTIONABLE_ACTIONS:
        raise ValueError(
            "Decision Engine non-actionable-action contract must not be empty."
        )

    if set(
        ACTIONABLE_ACTIONS
    ) & set(
        NON_ACTIONABLE_ACTIONS
    ):
        raise ValueError(
            "Decision Engine actionable and non-actionable actions cannot "
            "overlap."
        )

    for contact_intensity in (
        CONTACT_INTENSITIES
    ):

        get_delivery_route(
            contact_intensity
        )

    for delivery_route in (
        DELIVERY_ROUTES
    ):

        get_delivery_disposition(
            delivery_route
        )


validate_delivery_spec()