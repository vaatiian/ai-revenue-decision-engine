"""
Production priority and commercial-policy layer for the Customer Decision Engine.

This module consumes the output of action orchestration and translates the
selected primary action into operational prioritisation and commercial policy.

Responsibilities
----------------
- calculate action-specific priority scores;
- select the primary action score;
- rank customers within their assigned actionable action;
- assign priority tiers;
- assign contact intensity;
- select recommendation channel and recommended product;
- generate decision reasons;
- generate business-action playbook text;
- assign experimentation requirements;
- validate the resulting policy output.

This module does not:
- build customer snapshots;
- score predictive models;
- align raw model outputs;
- choose the primary action itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision_engine.spec import (
    ACTIONABLE_ACTIONS,
    ACTION_GROW,
    ACTION_MONITOR_NO_REACTIVATION_SCORE,
    ACTION_PROTECT,
    ACTION_REACTIVATE,
    ACTION_RETAIN,
    ACTION_UNSCORED_LONG_DORMANT,
    CONTACT_INTENSITY_BY_PRIORITY_TIER,
    CONTACT_NO_MODEL_LED,
    EXPERIMENT_NOT_APPLICABLE,
    EXPERIMENT_REQUIREMENT_BY_ACTION,
    GROW_PRIORITY_WEIGHTS,
    NOT_PRIORITISED,
    PRIMARY_ACTIONS,
    PRIORITY_4,
    PRIORITY_TIER_THRESHOLDS,
    PROTECT_PRIORITY_WEIGHTS,
    REACTIVATE_PRIORITY_WEIGHTS,
    RECOMMENDATION_CHANNEL_BY_ACTION,
    RECOMMENDATION_CHANNEL_DISCOVERY,
    RECOMMENDATION_CHANNEL_REPEAT,
    RETAIN_PRIORITY_WEIGHTS,
)


# ================================================================
# OUTPUT CONTRACT
# ================================================================


PRIORITY_POLICY_REQUIRED_COLUMNS = (
    "customer",
    "primary_action",
    "recommendation_operationally_valid",
    "leakage_exposure_percentile",
    "leakage_probability",
    "fcv_percentile",
    "future_active_probability",
    "reactivation_probability",
    "repeat_top1_product",
    "discovery_top1_product",
)


PRIORITY_POLICY_COLUMNS = (
    "protect_priority_score",
    "retain_priority_score",
    "reactivate_priority_score",
    "grow_priority_score",
    "primary_action_score",
    "action_priority_percentile",
    "priority_tier",
    "contact_intensity",
    "recommended_product_channel",
    "recommended_product",
    "decision_reason",
    "business_action",
    "experiment_requirement",
)


# ================================================================
# FROZEN TEXT POLICY
# ================================================================


DECISION_REASON_BY_ACTION = {
    ACTION_PROTECT:
        "Active customer with high economic revenue-leakage exposure",

    ACTION_RETAIN:
        "Customer is 31–90 days since last purchase and has a current "
        "revenue-leakage score",

    ACTION_REACTIVATE:
        "Dormant customer with a temporally valid 90-day reactivation "
        "propensity score",

    ACTION_GROW:
        "Active customer with current product recommendation and "
        "future-value signal",

    ACTION_MONITOR_NO_REACTIVATION_SCORE:
        "Dormant customer but no valid Reactivation score exists at the "
        "decision snapshot",

    ACTION_UNSCORED_LONG_DORMANT:
        "Customer exceeds 180 days since last purchase and lies outside "
        "current model coverage",
}


DEFAULT_DECISION_REASON = (
    "No model-supported intervention currently takes precedence"
)


BUSINESS_ACTION_PROTECT_WITH_PRODUCT = (
    "Protect customer value; review leakage risk and reinforce likely "
    "repeat purchase"
)


BUSINESS_ACTION_PROTECT_WITHOUT_PRODUCT = (
    "Protect customer value; investigate leakage drivers and preserve "
    "current spend"
)


BUSINESS_ACTION_RETAIN_WITH_PRODUCT = (
    "Retention outreach using relevant repeat-product cue"
)


BUSINESS_ACTION_RETAIN_WITHOUT_PRODUCT = (
    "Retention outreach focused on re-engagement and relationship continuity"
)


BUSINESS_ACTION_REACTIVATE = (
    "Controlled reactivation campaign; test outreach against appropriate "
    "holdout/control group"
)


BUSINESS_ACTION_GROW = (
    "Cross-sell / discovery offer using current next-best-product "
    "recommendation"
)


BUSINESS_ACTION_MONITOR_NO_REACTIVATION = (
    "Monitor dormant customer; insufficient current model evidence for "
    "targeted intervention"
)


BUSINESS_ACTION_LONG_DORMANT = (
    "Outside current model coverage; route to separate deep-reactivation "
    "strategy"
)


DEFAULT_BUSINESS_ACTION = (
    "Monitor customer without immediate intervention"
)


# ================================================================
# VALIDATION
# ================================================================


def validate_priority_policy_input(
    frame: pd.DataFrame,
) -> None:
    """
    Validate orchestration output before priority-policy processing.
    """

    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            "Priority-policy input must be a pandas DataFrame."
        )

    missing = [
        column
        for column in PRIORITY_POLICY_REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Priority-policy input is missing required columns: "
            + ", ".join(
                missing
            )
        )

    if frame.empty:
        return

    if frame[
        "customer"
    ].isna().any():
        raise ValueError(
            "Priority-policy input contains missing customer IDs."
        )

    if not frame[
        "customer"
    ].is_unique:
        raise ValueError(
            "Priority-policy input must contain one row per customer."
        )

    if frame[
        "primary_action"
    ].isna().any():
        raise ValueError(
            "Priority-policy input contains missing primary actions."
        )

    invalid_actions = (
        set(
            frame[
                "primary_action"
            ].astype(str)
        )
        -
        set(
            PRIMARY_ACTIONS
        )
    )

    if invalid_actions:
        raise ValueError(
            "Priority-policy input contains unknown primary actions: "
            f"{sorted(invalid_actions)}"
        )

    if frame[
        "recommendation_operationally_valid"
    ].isna().any():
        raise ValueError(
            "Priority-policy input contains missing "
            "recommendation_operationally_valid values."
        )


def validate_priority_policy_output(
    frame: pd.DataFrame,
) -> None:
    """
    Validate final priority and commercial-policy fields.
    """

    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            "Priority-policy output must be a pandas DataFrame."
        )

    required = (
        "customer",
        "primary_action",
        *PRIORITY_POLICY_COLUMNS,
    )

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Priority-policy output is missing required columns: "
            + ", ".join(
                missing
            )
        )

    if frame.empty:
        return

    if not frame[
        "customer"
    ].is_unique:
        raise ValueError(
            "Priority-policy output must contain one row per customer."
        )

    if frame[
        "priority_tier"
    ].isna().any():
        raise ValueError(
            "Every customer must receive a priority tier."
        )

    if frame[
        "contact_intensity"
    ].isna().any():
        raise ValueError(
            "Every customer must receive a contact-intensity policy."
        )

    if frame[
        "decision_reason"
    ].isna().any():
        raise ValueError(
            "Every customer must receive a decision reason."
        )

    if frame[
        "business_action"
    ].isna().any():
        raise ValueError(
            "Every customer must receive a business action."
        )

    if frame[
        "experiment_requirement"
    ].isna().any():
        raise ValueError(
            "Every customer must receive an experiment requirement."
        )

    actionable_mask = (
        frame[
            "primary_action"
        ]
        .isin(
            ACTIONABLE_ACTIONS
        )
    )

    non_actionable_mask = (
        ~actionable_mask
    )

    if frame.loc[
        non_actionable_mask,
        "primary_action_score",
    ].notna().any():
        raise ValueError(
            "Non-actionable customers cannot have primary action scores."
        )

    if frame.loc[
        non_actionable_mask,
        "action_priority_percentile",
    ].notna().any():
        raise ValueError(
            "Non-actionable customers cannot have action priority "
            "percentiles."
        )

    if not (
        frame.loc[
            non_actionable_mask,
            "priority_tier",
        ]
        ==
        NOT_PRIORITISED
    ).all():
        raise ValueError(
            "Non-actionable customers must be Not Prioritised."
        )

    if not (
        frame.loc[
            non_actionable_mask,
            "contact_intensity",
        ]
        ==
        CONTACT_NO_MODEL_LED
    ).all():
        raise ValueError(
            "Non-actionable customers must receive No Model-Led Contact."
        )

    grow_mask = (
        frame[
            "primary_action"
        ]
        ==
        ACTION_GROW
    )

    if frame.loc[
        grow_mask,
        "recommended_product",
    ].isna().any():
        raise ValueError(
            "Grow customers must have a recommended product."
        )


# ================================================================
# SCORE HELPERS
# ================================================================


def _weighted_score(
    frame: pd.DataFrame,
    weights,
) -> pd.Series:
    """
    Calculate an exact frozen weighted score.

    Missing values deliberately propagate as NaN, matching Notebook 08.
    """

    score = pd.Series(
        0.0,
        index=frame.index,
        dtype="float64",
    )

    for column, weight in weights:

        values = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )

        score = (
            score
            +
            weight
            *
            values
        )

    return score


# ================================================================
# POLICY HELPERS
# ================================================================


def _assign_priority_tier(
    action,
    percentile,
):
    """
    Reproduce Notebook 08 priority-tier fallthrough exactly.

    For actionable customers with NaN percentile, comparisons evaluate false,
    so the result is Priority 4.
    """

    if action not in ACTIONABLE_ACTIONS:
        return NOT_PRIORITISED

    for tier, threshold in PRIORITY_TIER_THRESHOLDS:

        if percentile >= threshold:
            return tier

    return PRIORITY_4


def _assign_contact_intensity(
    action,
    tier,
):
    """
    Map action/tier to operational contact intensity.
    """

    if action not in ACTIONABLE_ACTIONS:
        return CONTACT_NO_MODEL_LED

    mapping = dict(
        CONTACT_INTENSITY_BY_PRIORITY_TIER
    )

    return mapping.get(
        tier,
        mapping[
            PRIORITY_4
        ],
    )


def _assign_recommendation_channel(
    row,
):
    """
    Assign recommendation channel only when the signal is operationally valid.
    """

    if not row[
        "recommendation_operationally_valid"
    ]:
        return pd.NA

    mapping = dict(
        RECOMMENDATION_CHANNEL_BY_ACTION
    )

    return mapping.get(
        row[
            "primary_action"
        ],
        pd.NA,
    )


def _select_recommended_product(
    row,
):
    """
    Select the top-1 product associated with the chosen recommendation channel.

    A missing channel is a valid outcome for actions that do not use product
    recommendations, so it is handled explicitly before equality comparisons.
    """

    channel = row[
        "recommended_product_channel"
    ]

    if pd.isna(
        channel
    ):
        return pd.NA

    if (
        channel
        ==
        RECOMMENDATION_CHANNEL_REPEAT
    ):
        return row[
            "repeat_top1_product"
        ]

    if (
        channel
        ==
        RECOMMENDATION_CHANNEL_DISCOVERY
    ):
        return row[
            "discovery_top1_product"
        ]

    return pd.NA


def _build_decision_reason(
    action,
):
    """
    Return the frozen Notebook 08 reason text.
    """

    return DECISION_REASON_BY_ACTION.get(
        action,
        DEFAULT_DECISION_REASON,
    )


def _assign_business_action(
    row,
):
    """
    Translate primary action into the frozen commercial playbook text.
    """

    action = row[
        "primary_action"
    ]

    product_present = pd.notna(
        row[
            "recommended_product"
        ]
    )

    if action == ACTION_PROTECT:

        if product_present:
            return BUSINESS_ACTION_PROTECT_WITH_PRODUCT

        return BUSINESS_ACTION_PROTECT_WITHOUT_PRODUCT

    if action == ACTION_RETAIN:

        if product_present:
            return BUSINESS_ACTION_RETAIN_WITH_PRODUCT

        return BUSINESS_ACTION_RETAIN_WITHOUT_PRODUCT

    if action == ACTION_REACTIVATE:
        return BUSINESS_ACTION_REACTIVATE

    if action == ACTION_GROW:
        return BUSINESS_ACTION_GROW

    if action == ACTION_MONITOR_NO_REACTIVATION_SCORE:
        return BUSINESS_ACTION_MONITOR_NO_REACTIVATION

    if action == ACTION_UNSCORED_LONG_DORMANT:
        return BUSINESS_ACTION_LONG_DORMANT

    return DEFAULT_BUSINESS_ACTION


def _assign_experiment_requirement(
    action,
):
    """
    Return the frozen experimentation requirement by action.
    """

    mapping = dict(
        EXPERIMENT_REQUIREMENT_BY_ACTION
    )

    return mapping.get(
        action,
        EXPERIMENT_NOT_APPLICABLE,
    )


# ================================================================
# MAIN POLICY BUILDER
# ================================================================


def apply_priority_policy(
    orchestrated_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply action-specific priority and commercial policy.

    The input must already contain exactly one primary action per customer.
    """

    validate_priority_policy_input(
        orchestrated_snapshot
    )

    result = (
        orchestrated_snapshot.copy()
    )

    if result.empty:

        for column in PRIORITY_POLICY_COLUMNS:
            result[
                column
            ] = pd.Series(
                dtype="object"
            )

        validate_priority_policy_output(
            result
        )

        return result

    # ------------------------------------------------------------
    # ACTION-SPECIFIC PRIORITY SCORES
    # ------------------------------------------------------------

    result[
        "protect_priority_score"
    ] = _weighted_score(
        result,
        PROTECT_PRIORITY_WEIGHTS,
    )

    result[
        "retain_priority_score"
    ] = _weighted_score(
        result,
        RETAIN_PRIORITY_WEIGHTS,
    )

    result[
        "reactivate_priority_score"
    ] = _weighted_score(
        result,
        REACTIVATE_PRIORITY_WEIGHTS,
    )

    result[
        "grow_priority_score"
    ] = _weighted_score(
        result,
        GROW_PRIORITY_WEIGHTS,
    )

    # ------------------------------------------------------------
    # PRIMARY ACTION SCORE
    # ------------------------------------------------------------

    result[
        "primary_action_score"
    ] = np.nan

    score_column_by_action = {
        ACTION_PROTECT:
            "protect_priority_score",

        ACTION_RETAIN:
            "retain_priority_score",

        ACTION_REACTIVATE:
            "reactivate_priority_score",

        ACTION_GROW:
            "grow_priority_score",
    }

    for action, score_column in score_column_by_action.items():

        mask = (
            result[
                "primary_action"
            ]
            ==
            action
        )

        result.loc[
            mask,
            "primary_action_score",
        ] = result.loc[
            mask,
            score_column,
        ]

    # ------------------------------------------------------------
    # ACTION-SPECIFIC PRIORITY PERCENTILE
    # ------------------------------------------------------------

    actionable_mask = (
        result[
            "primary_action"
        ]
        .isin(
            ACTIONABLE_ACTIONS
        )
    )

    result[
        "action_priority_percentile"
    ] = np.nan

    result.loc[
        actionable_mask,
        "action_priority_percentile",
    ] = (
        result.loc[
            actionable_mask
        ]
        .groupby(
            "primary_action"
        )[
            "primary_action_score"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    # ------------------------------------------------------------
    # PRIORITY TIER
    # ------------------------------------------------------------

    result[
        "priority_tier"
    ] = [
        _assign_priority_tier(
            action=action,
            percentile=percentile,
        )
        for action, percentile in zip(
            result[
                "primary_action"
            ],
            result[
                "action_priority_percentile"
            ],
        )
    ]

    # ------------------------------------------------------------
    # CONTACT INTENSITY
    # ------------------------------------------------------------

    result[
        "contact_intensity"
    ] = [
        _assign_contact_intensity(
            action=action,
            tier=tier,
        )
        for action, tier in zip(
            result[
                "primary_action"
            ],
            result[
                "priority_tier"
            ],
        )
    ]

    # ------------------------------------------------------------
    # RECOMMENDATION CHANNEL
    # ------------------------------------------------------------

    result[
        "recommended_product_channel"
    ] = result.apply(
        _assign_recommendation_channel,
        axis=1,
    )

    # ------------------------------------------------------------
    # RECOMMENDED PRODUCT
    # ------------------------------------------------------------

    result[
        "recommended_product"
    ] = result.apply(
        _select_recommended_product,
        axis=1,
    )

    # ------------------------------------------------------------
    # DECISION REASON
    # ------------------------------------------------------------

    result[
        "decision_reason"
    ] = result[
        "primary_action"
    ].map(
        _build_decision_reason
    )

    # ------------------------------------------------------------
    # BUSINESS ACTION
    # ------------------------------------------------------------

    result[
        "business_action"
    ] = result.apply(
        _assign_business_action,
        axis=1,
    )

    # ------------------------------------------------------------
    # EXPERIMENT REQUIREMENT
    # ------------------------------------------------------------

    result[
        "experiment_requirement"
    ] = result[
        "primary_action"
    ].map(
        _assign_experiment_requirement
    )

    # ------------------------------------------------------------
    # FINAL GUARANTEES
    # ------------------------------------------------------------

    if len(
        result
    ) != len(
        orchestrated_snapshot
    ):
        raise RuntimeError(
            "Priority policy changed the customer population."
        )

    validate_priority_policy_output(
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