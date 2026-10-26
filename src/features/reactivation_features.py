"""
Production reactivation feature pipeline.

A customer becomes eligible for reactivation scoring after reaching a
configured inactivity threshold.

Default definition:

    eligibility_date = last qualifying order + 90 days

The feature table is point-in-time safe:
- no transaction after prediction_date is used;
- historical reactivation outcomes are counted only when their full
  outcome window has matured by prediction_date;
- the current open inactivity episode is not treated as a prior episode.

This module produces features compatible with the conceptual structure
used by the research Reactivation Propensity model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.customer_history import (
    add_inter_order_gaps,
    build_customer_order_history,
)

from src.features.customer_snapshot import (
    build_qualifying_merchandise_history,
    normalise_decision_date,
)


# ================================================================
# CURRENT REACTIVATION ELIGIBILITY
# ================================================================

def build_current_reactivation_eligibility(
    transactions: pd.DataFrame,
    prediction_date,
    inactivity_threshold_days: int = 90,
) -> pd.DataFrame:
    """
    Identify customers currently eligible for reactivation scoring.

    A customer is eligible when:

        prediction_date - last_order_date >= inactivity_threshold_days

    Only qualifying merchandise transactions on or before the
    prediction date are considered.
    """

    if inactivity_threshold_days <= 0:
        raise ValueError(
            "Inactivity threshold must be greater than zero."
        )

    score_date = normalise_decision_date(
        prediction_date
    )

    orders = build_customer_order_history(
        transactions,
        score_date,
    )

    if orders.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "last_order_date",
                "eligibility_date",
                "days_inactive",
                "days_since_eligibility",
            ]
        )

    latest_orders = (
        orders
        .sort_values(
            [
                "customer",
                "order_date",
                "invoice",
            ]
        )
        .drop_duplicates(
            subset="customer",
            keep="last",
        )[
            [
                "customer",
                "order_date",
            ]
        ]
        .rename(
            columns={
                "order_date": "last_order_date",
            }
        )
        .copy()
    )

    latest_orders[
        "eligibility_date"
    ] = (
        latest_orders[
            "last_order_date"
        ]
        +
        pd.to_timedelta(
            inactivity_threshold_days,
            unit="D",
        )
    )

    latest_orders[
        "days_inactive"
    ] = (
        (
            score_date
            -
            latest_orders[
                "last_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    latest_orders[
        "days_since_eligibility"
    ] = (
        (
            score_date
            -
            latest_orders[
                "eligibility_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    eligible = (
        latest_orders.loc[
            latest_orders[
                "eligibility_date"
            ]
            <=
            score_date
        ]
        .copy()
    )

    return (
        eligible
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# HISTORICAL INACTIVITY EPISODES
# ================================================================

def build_historical_inactivity_episodes(
    transactions: pd.DataFrame,
    prediction_date,
    inactivity_threshold_days: int = 90,
    reactivation_horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Build historical inactivity episodes whose outcome windows have
    fully matured by prediction_date.

    For each historical order:

        eligibility_date =
            order_date + inactivity_threshold_days

        outcome_end =
            eligibility_date + reactivation_horizon_days

    An inactivity episode exists only if no subsequent order occurs
    before eligibility_date.

    The episode is included as prior history only if its complete
    outcome window has matured by prediction_date.

    Reactivation success is 1 when the next order occurred on or after
    eligibility_date and on or before outcome_end.
    """

    if inactivity_threshold_days <= 0:
        raise ValueError(
            "Inactivity threshold must be greater than zero."
        )

    if reactivation_horizon_days <= 0:
        raise ValueError(
            "Reactivation horizon must be greater than zero."
        )

    score_date = normalise_decision_date(
        prediction_date
    )

    orders = build_customer_order_history(
        transactions,
        score_date,
    )

    if orders.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "episode_anchor_order_date",
                "eligibility_date",
                "outcome_end",
                "next_order_date",
                "reactivated_within_horizon",
            ]
        )

    orders = (
        orders
        .sort_values(
            [
                "customer",
                "order_date",
                "invoice",
            ]
        )
        .copy()
    )

    orders[
        "next_order_date"
    ] = (
        orders
        .groupby(
            "customer"
        )[
            "order_date"
        ]
        .shift(-1)
    )

    orders[
        "eligibility_date"
    ] = (
        orders[
            "order_date"
        ]
        +
        pd.to_timedelta(
            inactivity_threshold_days,
            unit="D",
        )
    )

    orders[
        "outcome_end"
    ] = (
        orders[
            "eligibility_date"
        ]
        +
        pd.to_timedelta(
            reactivation_horizon_days,
            unit="D",
        )
    )

    orders[
        "creates_inactivity_episode"
    ] = (
        orders[
            "next_order_date"
        ]
        .isna()
        |
        (
            orders[
                "next_order_date"
            ]
            >=
            orders[
                "eligibility_date"
            ]
        )
    )

    matured = (
        orders.loc[
            orders[
                "creates_inactivity_episode"
            ]
            &
            (
                orders[
                    "outcome_end"
                ]
                <=
                score_date
            )
        ]
        .copy()
    )

    matured[
        "reactivated_within_horizon"
    ] = (
        matured[
            "next_order_date"
        ]
        .notna()
        &
        (
            matured[
                "next_order_date"
            ]
            >=
            matured[
                "eligibility_date"
            ]
        )
        &
        (
            matured[
                "next_order_date"
            ]
            <=
            matured[
                "outcome_end"
            ]
        )
    )

    matured = (
        matured.rename(
            columns={
                "order_date":
                    "episode_anchor_order_date",
            }
        )[
            [
                "customer",
                "episode_anchor_order_date",
                "eligibility_date",
                "outcome_end",
                "next_order_date",
                "reactivated_within_horizon",
            ]
        ]
    )

    return (
        matured
        .sort_values(
            [
                "customer",
                "eligibility_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# PRIOR REACTIVATION FEATURES
# ================================================================

def build_prior_reactivation_features(
    transactions: pd.DataFrame,
    prediction_date,
    inactivity_threshold_days: int = 90,
    reactivation_horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Summarise matured historical inactivity episodes by customer.
    """

    episodes = (
        build_historical_inactivity_episodes(
            transactions=transactions,
            prediction_date=prediction_date,
            inactivity_threshold_days=inactivity_threshold_days,
            reactivation_horizon_days=reactivation_horizon_days,
        )
    )

    if episodes.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "prior_inactivity_episode_count",
                "prior_reactivation_count",
                "prior_reactivation_rate",
            ]
        )

    summary = (
        episodes
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            prior_inactivity_episode_count=(
                "reactivated_within_horizon",
                "size",
            ),

            prior_reactivation_count=(
                "reactivated_within_horizon",
                "sum",
            ),
        )
    )

    summary[
        "prior_reactivation_rate"
    ] = np.where(
        summary[
            "prior_inactivity_episode_count"
        ]
        >
        0,

        summary[
            "prior_reactivation_count"
        ]
        /
        summary[
            "prior_inactivity_episode_count"
        ],

        0.0,
    )

    return (
        summary
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# LAST ORDER REGION
# ================================================================

def build_last_order_region(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Return the most recently observed qualifying merchandise region
    for each identified customer.
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    history = build_qualifying_merchandise_history(
        transactions,
        score_date,
    )

    if history.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "last_order_region",
            ]
        )

    last_region = (
        history
        .sort_values(
            [
                "customer",
                "date",
                "invoice",
            ]
        )
        .drop_duplicates(
            subset="customer",
            keep="last",
        )[
            [
                "customer",
                "region",
            ]
        ]
        .rename(
            columns={
                "region":
                    "last_order_region",
            }
        )
        .copy()
    )

    return (
        last_region
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# FINAL REACTIVATION FEATURE TABLE
# ================================================================

def build_reactivation_features(
    transactions: pd.DataFrame,
    prediction_date,
    inactivity_threshold_days: int = 90,
    reactivation_horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Build the production reactivation scoring population and features.

    One row is returned per customer currently eligible for
    reactivation scoring.

    No future target or outcome label is generated.
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    eligible = (
        build_current_reactivation_eligibility(
            transactions=transactions,
            prediction_date=score_date,
            inactivity_threshold_days=inactivity_threshold_days,
        )
    )

    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "last_order_date",
                "eligibility_date",
                "days_inactive",
                "days_since_eligibility",
                "historical_order_count",
                "historical_revenue",
                "historical_units",
                "historical_average_order_value",
                "historical_average_gap_days",
                "first_order_date",
                "last_order_value",
                "tenure_at_prediction_days",
                "prior_inactivity_episode_count",
                "prior_reactivation_count",
                "prior_reactivation_rate",
                "last_order_region",
                "inactivity_episode_number",
                "prediction_date",
                "prediction_period",
            ]
        )

    orders = build_customer_order_history(
        transactions,
        score_date,
    )

    orders_with_gaps = add_inter_order_gaps(
        orders
    )

    # ------------------------------------------------------------
    # CORE HISTORICAL FEATURES
    # ------------------------------------------------------------

    customer_history = (
        orders_with_gaps
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            historical_order_count=(
                "invoice",
                "nunique",
            ),

            historical_revenue=(
                "order_revenue",
                "sum",
            ),

            historical_units=(
                "order_units",
                "sum",
            ),

            historical_average_order_value=(
                "order_revenue",
                "mean",
            ),

            historical_average_gap_days=(
                "gap_days",
                "mean",
            ),

            first_order_date=(
                "order_date",
                "min",
            ),

            last_order_value=(
                "order_revenue",
                "last",
            ),
        )
    )

    customer_history[
        "tenure_at_prediction_days"
    ] = (
        (
            score_date
            -
            customer_history[
                "first_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    # ------------------------------------------------------------
    # PRIOR REACTIVATION HISTORY
    # ------------------------------------------------------------

    prior_features = (
        build_prior_reactivation_features(
            transactions=transactions,
            prediction_date=score_date,
            inactivity_threshold_days=inactivity_threshold_days,
            reactivation_horizon_days=reactivation_horizon_days,
        )
    )

    # ------------------------------------------------------------
    # LAST ORDER REGION
    # ------------------------------------------------------------

    last_region = build_last_order_region(
        transactions,
        score_date,
    )

    # ------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------

    features = (
        eligible
        .merge(
            customer_history,
            on="customer",
            how="left",
        )
        .merge(
            prior_features,
            on="customer",
            how="left",
        )
        .merge(
            last_region,
            on="customer",
            how="left",
        )
    )

    # ------------------------------------------------------------
    # FILL PRIOR-HISTORY ZEROS
    # ------------------------------------------------------------

    prior_columns = [
        "prior_inactivity_episode_count",
        "prior_reactivation_count",
        "prior_reactivation_rate",
    ]

    for column in prior_columns:

        features[
            column
        ] = (
            features[
                column
            ]
            .fillna(0)
        )

    # ------------------------------------------------------------
    # CURRENT EPISODE NUMBER
    # ------------------------------------------------------------

    features[
        "inactivity_episode_number"
    ] = (
        features[
            "prior_inactivity_episode_count"
        ]
        +
        1
    )

    # ------------------------------------------------------------
    # PREDICTION DATE / PERIOD
    # ------------------------------------------------------------

    features[
        "prediction_date"
    ] = score_date

    features[
        "prediction_period"
    ] = str(
        score_date.to_period(
            "M"
        )
    )

    # ------------------------------------------------------------
    # SAFETY CHECKS
    # ------------------------------------------------------------

    if not features[
        "customer"
    ].is_unique:
        raise RuntimeError(
            "Reactivation feature table contains duplicate customers."
        )

    if (
        features[
            "last_order_date"
        ]
        >
        score_date
    ).any():
        raise RuntimeError(
            "Future transaction leakage detected."
        )

    if (
        features[
            "eligibility_date"
        ]
        >
        score_date
    ).any():
        raise RuntimeError(
            "Non-eligible customer entered reactivation scoring population."
        )

    if (
        features[
            "days_inactive"
        ]
        <
        inactivity_threshold_days
    ).any():
        raise RuntimeError(
            "Customer entered reactivation scoring population "
            "before reaching the inactivity threshold."
        )

    # ------------------------------------------------------------
    # FINAL COLUMN ORDER
    # ------------------------------------------------------------

    preferred_column_order = [
        "customer",
        "prediction_date",
        "prediction_period",
        "inactivity_episode_number",
        "last_order_date",
        "eligibility_date",
        "days_inactive",
        "days_since_eligibility",
        "historical_order_count",
        "historical_revenue",
        "historical_units",
        "historical_average_order_value",
        "tenure_at_prediction_days",
        "historical_average_gap_days",
        "prior_inactivity_episode_count",
        "prior_reactivation_count",
        "prior_reactivation_rate",
        "last_order_value",
        "last_order_region",
        "first_order_date",
    ]

    remaining_columns = [
        column
        for column in features.columns
        if column not in preferred_column_order
    ]

    features = features[
        preferred_column_order
        +
        remaining_columns
    ]

    return (
        features
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )