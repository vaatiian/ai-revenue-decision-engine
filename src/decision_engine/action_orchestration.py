"""
Deterministic customer action orchestration.

This module converts an aligned customer decision snapshot into one primary
business action per customer.

Responsibilities
----------------
- identify high Leakage economic exposure;
- determine Protect eligibility;
- determine Grow eligibility;
- determine Retain eligibility;
- determine Reactivate eligibility;
- assign exactly one primary action using the frozen lifecycle hierarchy;
- validate state/action governance.

This module deliberately does not:
- calculate action priority scores;
- rank customers;
- assign priority tiers;
- assign contact intensity;
- select recommendation channels or products;
- generate business-action text;
- generate decision reasons;
- assign experiment requirements.

Those responsibilities belong to later Decision Engine layers.
"""

from __future__ import annotations

import pandas as pd

from src.decision_engine.spec import (
    ACTION_GROW,
    ACTION_MONITOR,
    ACTION_MONITOR_NO_REACTIVATION_SCORE,
    ACTION_PROTECT,
    ACTION_REACTIVATE,
    ACTION_RETAIN,
    ACTION_UNCLASSIFIED,
    ACTION_UNSCORED_LONG_DORMANT,
    ALLOWED_ACTIONS_BY_CUSTOMER_STATE,
    CUSTOMER_STATE_ACTIVE,
    CUSTOMER_STATE_AT_RISK,
    CUSTOMER_STATE_DORMANT,
    CUSTOMER_STATE_LONG_DORMANT,
    CUSTOMER_STATE_UNCLASSIFIED,
    CUSTOMER_STATES,
    HIGH_LEAKAGE_EXPOSURE_PREFIXES,
    PRIMARY_ACTIONS,
)


# ================================================================
# ORCHESTRATION CONTRACT
# ================================================================


ACTION_ORCHESTRATION_REQUIRED_COLUMNS = (
    "customer",
    "customer_state",
    "economic_priority_band",
    "leakage_operationally_valid",
    "fcv_signal_available",
    "reactivation_operationally_valid",
    "recommendation_operationally_valid",
)


ACTION_ORCHESTRATION_COLUMNS = (
    "high_leakage_exposure",
    "protect_eligible",
    "grow_eligible",
    "retain_eligible",
    "reactivate_eligible",
    "primary_action",
)


# ================================================================
# INPUT VALIDATION
# ================================================================


def validate_action_orchestration_input(
    aligned_snapshot: pd.DataFrame,
) -> None:
    """
    Validate the aligned Decision Engine snapshot before orchestration.
    """

    if not isinstance(
        aligned_snapshot,
        pd.DataFrame,
    ):
        raise TypeError(
            "Action orchestration input must be a pandas DataFrame."
        )

    missing = [
        column
        for column in ACTION_ORCHESTRATION_REQUIRED_COLUMNS
        if column not in aligned_snapshot.columns
    ]

    if missing:
        raise ValueError(
            "Action orchestration input is missing required columns: "
            + ", ".join(
                missing
            )
        )

    if aligned_snapshot.empty:
        return

    if aligned_snapshot[
        "customer"
    ].isna().any():
        raise ValueError(
            "Action orchestration input contains missing customer IDs."
        )

    if not aligned_snapshot[
        "customer"
    ].is_unique:
        raise ValueError(
            "Action orchestration input must contain one row per customer."
        )

    if aligned_snapshot[
        "customer_state"
    ].isna().any():
        raise ValueError(
            "Action orchestration input contains missing customer states."
        )

    unknown_states = (
        set(
            aligned_snapshot[
                "customer_state"
            ].astype(str)
        )
        -
        set(
            CUSTOMER_STATES
        )
    )

    if unknown_states:
        raise ValueError(
            "Action orchestration input contains unknown customer states: "
            f"{sorted(unknown_states)}"
        )

    boolean_columns = (
        "leakage_operationally_valid",
        "fcv_signal_available",
        "reactivation_operationally_valid",
        "recommendation_operationally_valid",
    )

    for column in boolean_columns:

        if aligned_snapshot[
            column
        ].isna().any():
            raise ValueError(
                f"Action orchestration input contains missing {column} values."
            )

        valid_boolean_values = (
            aligned_snapshot[
                column
            ]
            .map(
                lambda value:
                    isinstance(
                        value,
                        bool,
                    )
                    or
                    type(
                        value
                    ).__name__
                    ==
                    "bool_"
            )
        )

        if not valid_boolean_values.all():
            raise ValueError(
                f"Action orchestration input column {column!r} "
                "must contain boolean values."
            )


# ================================================================
# OUTPUT VALIDATION
# ================================================================


def validate_action_orchestration_output(
    result: pd.DataFrame,
) -> None:
    """
    Validate the deterministic action result against frozen governance.
    """

    if not isinstance(
        result,
        pd.DataFrame,
    ):
        raise TypeError(
            "Action orchestration output must be a pandas DataFrame."
        )

    required = (
        "customer",
        "customer_state",
        *ACTION_ORCHESTRATION_COLUMNS,
    )

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            "Action orchestration output is missing required columns: "
            + ", ".join(
                missing
            )
        )

    if result.empty:
        return

    if not result[
        "customer"
    ].is_unique:
        raise ValueError(
            "Action orchestration output must contain one row per customer."
        )

    if result[
        "primary_action"
    ].isna().any():
        raise ValueError(
            "Every customer must receive a primary action."
        )

    unknown_actions = (
        set(
            result[
                "primary_action"
            ].astype(str)
        )
        -
        set(
            PRIMARY_ACTIONS
        )
    )

    if unknown_actions:
        raise ValueError(
            "Action orchestration produced unknown primary actions: "
            f"{sorted(unknown_actions)}"
        )

    allowed_actions = dict(
        ALLOWED_ACTIONS_BY_CUSTOMER_STATE
    )

    invalid_rows = []

    for row in result[
        [
            "customer",
            "customer_state",
            "primary_action",
        ]
    ].itertuples(
        index=False
    ):

        state_actions = (
            allowed_actions.get(
                row.customer_state
            )
        )

        if (
            state_actions is None
            or
            row.primary_action
            not in state_actions
        ):
            invalid_rows.append(
                (
                    row.customer,
                    row.customer_state,
                    row.primary_action,
                )
            )

    if invalid_rows:
        raise ValueError(
            "Action orchestration produced state/action combinations "
            f"outside frozen governance: {invalid_rows}"
        )

    eligibility_columns = (
        "protect_eligible",
        "grow_eligible",
        "retain_eligible",
        "reactivate_eligible",
    )

    eligibility_count = (
        result[
            list(
                eligibility_columns
            )
        ]
        .astype(int)
        .sum(
            axis=1
        )
    )

    if (
        eligibility_count
        >
        1
    ).any():
        raise ValueError(
            "A customer cannot be simultaneously eligible for multiple "
            "primary actions."
        )


# ================================================================
# ACTION ORCHESTRATION
# ================================================================


def build_action_orchestration(
    aligned_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign one deterministic primary action per customer.

    Frozen hierarchy
    ----------------
    Active:
        Protect when Leakage exposure is P1/P2 and Leakage is operationally
        valid.

        Otherwise Grow when Recommendation is operationally valid and an
        FCV signal exists.

        Otherwise Monitor.

    At Risk:
        Retain when Leakage is operationally valid.
        Otherwise Monitor.

    Dormant:
        Reactivate when Reactivation is operationally valid.
        Otherwise Monitor — No Reactivation Score.

    Long Dormant:
        Unscored — Long Dormant.

    Unclassified:
        Unclassified.
    """

    validate_action_orchestration_input(
        aligned_snapshot
    )

    result = (
        aligned_snapshot.copy()
    )

    if result.empty:

        for column in ACTION_ORCHESTRATION_COLUMNS:

            if column == "primary_action":

                result[
                    column
                ] = pd.Series(
                    dtype="object"
                )

            else:

                result[
                    column
                ] = pd.Series(
                    dtype="bool"
                )

        validate_action_orchestration_output(
            result
        )

        return result

    # ------------------------------------------------------------
    # HIGH LEAKAGE EXPOSURE
    # ------------------------------------------------------------

    result[
        "high_leakage_exposure"
    ] = (
        result[
            "economic_priority_band"
        ]
        .astype(
            "string"
        )
        .str.startswith(
            HIGH_LEAKAGE_EXPOSURE_PREFIXES,
            na=False,
        )
    )

    # ------------------------------------------------------------
    # PROTECT
    # ------------------------------------------------------------

    result[
        "protect_eligible"
    ] = (
        (
            result[
                "customer_state"
            ]
            ==
            CUSTOMER_STATE_ACTIVE
        )
        &
        result[
            "leakage_operationally_valid"
        ]
        &
        result[
            "high_leakage_exposure"
        ]
    )

    # ------------------------------------------------------------
    # GROW
    # ------------------------------------------------------------

    result[
        "grow_eligible"
    ] = (
        (
            result[
                "customer_state"
            ]
            ==
            CUSTOMER_STATE_ACTIVE
        )
        &
        (
            ~result[
                "protect_eligible"
            ]
        )
        &
        result[
            "recommendation_operationally_valid"
        ]
        &
        result[
            "fcv_signal_available"
        ]
    )

    # ------------------------------------------------------------
    # RETAIN
    # ------------------------------------------------------------

    result[
        "retain_eligible"
    ] = (
        (
            result[
                "customer_state"
            ]
            ==
            CUSTOMER_STATE_AT_RISK
        )
        &
        result[
            "leakage_operationally_valid"
        ]
    )

    # ------------------------------------------------------------
    # REACTIVATE
    # ------------------------------------------------------------

    result[
        "reactivate_eligible"
    ] = (
        (
            result[
                "customer_state"
            ]
            ==
            CUSTOMER_STATE_DORMANT
        )
        &
        result[
            "reactivation_operationally_valid"
        ]
    )

    # ------------------------------------------------------------
    # DEFAULT ACTION BY LIFECYCLE STATE
    # ------------------------------------------------------------

    result[
        "primary_action"
    ] = pd.NA

    result.loc[
        result[
            "customer_state"
        ]
        ==
        CUSTOMER_STATE_ACTIVE,
        "primary_action",
    ] = ACTION_MONITOR

    result.loc[
        result[
            "customer_state"
        ]
        ==
        CUSTOMER_STATE_AT_RISK,
        "primary_action",
    ] = ACTION_MONITOR

    result.loc[
        result[
            "customer_state"
        ]
        ==
        CUSTOMER_STATE_DORMANT,
        "primary_action",
    ] = (
        ACTION_MONITOR_NO_REACTIVATION_SCORE
    )

    result.loc[
        result[
            "customer_state"
        ]
        ==
        CUSTOMER_STATE_LONG_DORMANT,
        "primary_action",
    ] = (
        ACTION_UNSCORED_LONG_DORMANT
    )

    result.loc[
        result[
            "customer_state"
        ]
        ==
        CUSTOMER_STATE_UNCLASSIFIED,
        "primary_action",
    ] = ACTION_UNCLASSIFIED

    # ------------------------------------------------------------
    # ACTIONABLE OVERRIDES
    # ------------------------------------------------------------

    result.loc[
        result[
            "grow_eligible"
        ],
        "primary_action",
    ] = ACTION_GROW

    result.loc[
        result[
            "protect_eligible"
        ],
        "primary_action",
    ] = ACTION_PROTECT

    result.loc[
        result[
            "retain_eligible"
        ],
        "primary_action",
    ] = ACTION_RETAIN

    result.loc[
        result[
            "reactivate_eligible"
        ],
        "primary_action",
    ] = ACTION_REACTIVATE

    # ------------------------------------------------------------
    # GOVERNANCE CHECKS
    # ------------------------------------------------------------

    if len(
        result
    ) != len(
        aligned_snapshot
    ):
        raise RuntimeError(
            "Action orchestration changed the customer population."
        )

    validate_action_orchestration_output(
        result
    )

    return (
        result
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )