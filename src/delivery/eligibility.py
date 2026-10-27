"""
Delivery eligibility and suppression enforcement.

This module converts governed Decision Engine contact policy into operational
delivery routing.

It does not change the upstream decision.

Rules
-----
1. Actionable decisions may route to:
   - treatment;
   - hold/control.

2. Non-actionable decisions must route to:
   - suppressed only.

3. No downstream layer may convert:
   - Hold / Control into treatment;
   - No Model-Led Contact into treatment.

4. Contradictory Decision Engine rows fail closed.
"""

from __future__ import annotations

import pandas as pd

from src.decision_engine.spec import (
    ACTIONABLE_ACTIONS,
    CONTACT_HOLD_CONTROL,
    CONTACT_NO_MODEL_LED,
    NON_ACTIONABLE_ACTIONS,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITION_HOLD_CONTROL,
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_REQUIRED_DECISION_COLUMNS,
    DELIVERY_ROUTE_HOLD_CONTROL,
    DELIVERY_ROUTE_SUPPRESSED,
    get_delivery_disposition,
    get_delivery_route,
)


# ================================================================
# OUTPUT COLUMNS
# ================================================================


DELIVERY_ELIGIBILITY_COLUMNS = (
    "delivery_route",
    "delivery_disposition",
)


# ================================================================
# INPUT VALIDATION
# ================================================================


def validate_delivery_eligibility_input(
    decisions: pd.DataFrame,
) -> None:
    """
    Validate Decision Engine rows before downstream routing.
    """

    if not isinstance(
        decisions,
        pd.DataFrame,
    ):
        raise TypeError(
            "decisions must be a pandas DataFrame."
        )

    missing_columns = (
        set(
            DELIVERY_REQUIRED_DECISION_COLUMNS
        )
        -
        set(
            decisions.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Delivery input is missing required Decision Engine columns: "
            f"{sorted(missing_columns)}"
        )

    if decisions[
        "customer"
    ].isna().any():
        raise ValueError(
            "Delivery input cannot contain missing customer identifiers."
        )

    if not decisions[
        "customer"
    ].is_unique:
        raise ValueError(
            "Delivery input must contain one row per customer."
        )

    if decisions[
        "decision_date"
    ].isna().any():
        raise ValueError(
            "Delivery input cannot contain missing decision_date values."
        )

    parsed_dates = pd.to_datetime(
        decisions[
            "decision_date"
        ],
        errors="coerce",
    )

    if parsed_dates.isna().any():
        raise ValueError(
            "Delivery input contains invalid decision_date values."
        )

    if (
        not decisions.empty
        and
        parsed_dates.nunique()
        !=
        1
    ):
        raise ValueError(
            "Delivery input must contain exactly one decision_date."
        )

    if decisions[
        "primary_action"
    ].isna().any():
        raise ValueError(
            "Delivery input cannot contain missing primary_action values."
        )

    valid_actions = (
        set(
            ACTIONABLE_ACTIONS
        )
        |
        set(
            NON_ACTIONABLE_ACTIONS
        )
    )

    unknown_actions = (
        set(
            decisions[
                "primary_action"
            ].astype(
                str
            )
        )
        -
        valid_actions
    )

    if unknown_actions:
        raise ValueError(
            "Delivery input contains unknown primary actions: "
            f"{sorted(unknown_actions)}"
        )

    if decisions[
        "contact_intensity"
    ].isna().any():
        raise ValueError(
            "Delivery input cannot contain missing contact_intensity values."
        )


# ================================================================
# ROW POLICY
# ================================================================


def _resolve_route_and_disposition(
    primary_action,
    contact_intensity,
) -> tuple[
    str,
    str,
]:
    """
    Resolve one governed downstream route/disposition pair.
    """

    route = (
        get_delivery_route(
            contact_intensity
        )
    )

    disposition = (
        get_delivery_disposition(
            route
        )
    )

    if (
        primary_action
        in
        NON_ACTIONABLE_ACTIONS
    ):

        if (
            contact_intensity
            !=
            CONTACT_NO_MODEL_LED
        ):
            raise ValueError(
                "Non-actionable decisions must have "
                "No Model-Led Contact."
            )

        if (
            route
            !=
            DELIVERY_ROUTE_SUPPRESSED
        ):
            raise RuntimeError(
                "Non-actionable decision did not resolve to SUPPRESSED."
            )

        if (
            disposition
            !=
            DELIVERY_DISPOSITION_SUPPRESSED
        ):
            raise RuntimeError(
                "Non-actionable decision did not resolve to "
                "SUPPRESSED disposition."
            )

        return (
            route,
            disposition,
        )

    if (
        primary_action
        in
        ACTIONABLE_ACTIONS
    ):

        if (
            contact_intensity
            ==
            CONTACT_NO_MODEL_LED
        ):
            raise ValueError(
                "Actionable decisions cannot have "
                "No Model-Led Contact."
            )

        if (
            contact_intensity
            ==
            CONTACT_HOLD_CONTROL
        ):

            if (
                route
                !=
                DELIVERY_ROUTE_HOLD_CONTROL
            ):
                raise RuntimeError(
                    "Hold / Control decision did not resolve to "
                    "HOLD_CONTROL route."
                )

            if (
                disposition
                !=
                DELIVERY_DISPOSITION_HOLD_CONTROL
            ):
                raise RuntimeError(
                    "Hold / Control decision did not resolve to "
                    "HOLD_CONTROL disposition."
                )

            return (
                route,
                disposition,
            )

        if (
            disposition
            !=
            DELIVERY_DISPOSITION_TREATMENT
        ):
            raise ValueError(
                "Actionable non-hold decision must resolve to "
                "TREATMENT."
            )

        return (
            route,
            disposition,
        )

    raise ValueError(
        f"Unsupported primary_action: {primary_action}"
    )


# ================================================================
# FRAME BUILDER
# ================================================================


def apply_delivery_eligibility(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach governed delivery route and disposition to Decision Engine rows.

    Customer population and row order are preserved.
    """

    validate_delivery_eligibility_input(
        decisions
    )

    result = (
        decisions.copy()
    )

    if result.empty:

        result[
            "delivery_route"
        ] = pd.Series(
            dtype="object"
        )

        result[
            "delivery_disposition"
        ] = pd.Series(
            dtype="object"
        )

        return result

    resolved = [
        _resolve_route_and_disposition(
            primary_action=row.primary_action,
            contact_intensity=row.contact_intensity,
        )
        for row
        in result[
            [
                "primary_action",
                "contact_intensity",
            ]
        ].itertuples(
            index=False
        )
    ]

    result[
        "delivery_route"
    ] = [
        route
        for (
            route,
            _,
        )
        in resolved
    ]

    result[
        "delivery_disposition"
    ] = [
        disposition
        for (
            _,
            disposition,
        )
        in resolved
    ]

    if len(
        result
    ) != len(
        decisions
    ):
        raise RuntimeError(
            "Delivery eligibility changed the customer population."
        )

    if not result[
        "customer"
    ].equals(
        decisions[
            "customer"
        ]
    ):
        raise RuntimeError(
            "Delivery eligibility changed customer row order."
        )

    return result