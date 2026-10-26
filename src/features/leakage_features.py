"""
Production revenue leakage feature pipeline.

This module builds point-in-time customer features for Revenue Leakage
Risk scoring.

Research definition:
- customer-level prediction snapshot
- positive merchandise activity
- baseline revenue measured over the prior 90 days
- customer must have positive baseline revenue to enter the scoring
  population
- no future revenue or future leakage outcome is used in production
  features

The future leakage target belongs only in model development and
evaluation, never in production scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.customer_snapshot import (
    build_qualifying_merchandise_history,
    normalise_decision_date,
)


DEFAULT_BASELINE_WINDOW_DAYS = 90


# ================================================================
# HELPER — WINDOW HISTORY
# ================================================================

def _window_history(
    history: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return transactions in:

        start_date < date <= end_date
    """

    return (
        history.loc[
            (
                history["date"]
                >
                start_date
            )
            &
            (
                history["date"]
                <=
                end_date
            )
        ]
        .copy()
    )


# ================================================================
# CUSTOMER WINDOW SUMMARY
# ================================================================

def build_leakage_window_summary(
    transactions: pd.DataFrame,
    prediction_date,
    window_days: int = DEFAULT_BASELINE_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Build the current baseline leakage window.

    Default:
        prediction_date - 90 days < transaction <= prediction_date

    One row per customer is returned.
    """

    if window_days <= 0:
        raise ValueError(
            "window_days must be greater than zero."
        )

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
                "baseline_revenue",
                "baseline_orders",
                "baseline_units",
                "baseline_unique_products",
                "baseline_average_order_value",
            ]
        )

    start_date = (
        score_date
        -
        pd.Timedelta(
            days=window_days
        )
    )

    window = _window_history(
        history,
        start_date=start_date,
        end_date=score_date,
    )

    if window.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "baseline_revenue",
                "baseline_orders",
                "baseline_units",
                "baseline_unique_products",
                "baseline_average_order_value",
            ]
        )

    summary = (
        window
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            baseline_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            baseline_orders=(
                "invoice",
                "nunique",
            ),

            baseline_units=(
                "quantity",
                "sum",
            ),

            baseline_unique_products=(
                "product",
                "nunique",
            ),
        )
    )

    summary[
        "baseline_average_order_value"
    ] = np.where(
        summary[
            "baseline_orders"
        ]
        >
        0,

        summary[
            "baseline_revenue"
        ]
        /
        summary[
            "baseline_orders"
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
# PRIOR WINDOW SUMMARY
# ================================================================

def build_prior_leakage_window_summary(
    transactions: pd.DataFrame,
    prediction_date,
    window_days: int = DEFAULT_BASELINE_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Build the immediately preceding historical window.

    For a 90-day setup:

        current baseline:
            prediction_date - 90d < date <= prediction_date

        prior baseline:
            prediction_date - 180d < date <= prediction_date - 90d

    This provides a trend feature without using future data.
    """

    if window_days <= 0:
        raise ValueError(
            "window_days must be greater than zero."
        )

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
                "prior_window_revenue",
                "prior_window_orders",
                "prior_window_units",
            ]
        )

    current_start = (
        score_date
        -
        pd.Timedelta(
            days=window_days
        )
    )

    prior_start = (
        score_date
        -
        pd.Timedelta(
            days=2 * window_days
        )
    )

    prior_window = _window_history(
        history,
        start_date=prior_start,
        end_date=current_start,
    )

    if prior_window.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "prior_window_revenue",
                "prior_window_orders",
                "prior_window_units",
            ]
        )

    summary = (
        prior_window
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            prior_window_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            prior_window_orders=(
                "invoice",
                "nunique",
            ),

            prior_window_units=(
                "quantity",
                "sum",
            ),
        )
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
# LIFETIME / RECENCY FEATURES
# ================================================================

def build_leakage_customer_history(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build point-in-time lifetime and recency features for customers.
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
                "first_order_date",
                "last_order_date",
                "historical_order_count",
                "historical_revenue",
                "historical_units",
                "days_since_last_order",
                "tenure_days",
            ]
        )

    summary = (
        history
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            first_order_date=(
                "date",
                "min",
            ),

            last_order_date=(
                "date",
                "max",
            ),

            historical_order_count=(
                "invoice",
                "nunique",
            ),

            historical_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            historical_units=(
                "quantity",
                "sum",
            ),
        )
    )

    summary[
        "days_since_last_order"
    ] = (
        (
            score_date
            -
            summary[
                "last_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    summary[
        "tenure_days"
    ] = (
        (
            score_date
            -
            summary[
                "first_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
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
# FINAL LEAKAGE FEATURE TABLE
# ================================================================

def build_leakage_features(
    transactions: pd.DataFrame,
    prediction_date,
    baseline_window_days: int = DEFAULT_BASELINE_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Build one production leakage-scoring row per eligible customer.

    Eligibility rule:
        baseline_revenue > 0

    This mirrors the conceptual research scoring population:
    customers with positive recent merchandise revenue.

    No future leakage outcome or future revenue is generated.
    """

    if baseline_window_days <= 0:
        raise ValueError(
            "baseline_window_days must be greater than zero."
        )

    score_date = normalise_decision_date(
        prediction_date
    )

    baseline = build_leakage_window_summary(
        transactions=transactions,
        prediction_date=score_date,
        window_days=baseline_window_days,
    )

    if baseline.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "prediction_date",
                "prediction_period",
                "baseline_revenue",
                "baseline_orders",
                "baseline_units",
                "baseline_unique_products",
                "baseline_average_order_value",
                "prior_window_revenue",
                "prior_window_orders",
                "prior_window_units",
                "revenue_trend_ratio",
                "revenue_change_amount",
                "historical_order_count",
                "historical_revenue",
                "historical_units",
                "days_since_last_order",
                "tenure_days",
                "first_order_date",
                "last_order_date",
            ]
        )

    # ------------------------------------------------------------
    # POSITIVE BASELINE ELIGIBILITY
    # ------------------------------------------------------------

    baseline = (
        baseline.loc[
            baseline[
                "baseline_revenue"
            ]
            >
            0
        ]
        .copy()
    )

    if baseline.empty:
        return pd.DataFrame()

    prior = build_prior_leakage_window_summary(
        transactions=transactions,
        prediction_date=score_date,
        window_days=baseline_window_days,
    )

    history = build_leakage_customer_history(
        transactions=transactions,
        prediction_date=score_date,
    )

    features = (
        baseline
        .merge(
            prior,
            on="customer",
            how="left",
        )
        .merge(
            history,
            on="customer",
            how="left",
        )
    )

    # ------------------------------------------------------------
    # MISSING PRIOR WINDOW = ZERO OBSERVED PRIOR ACTIVITY
    # ------------------------------------------------------------

    prior_columns = [
        "prior_window_revenue",
        "prior_window_orders",
        "prior_window_units",
    ]

    features[
        prior_columns
    ] = (
        features[
            prior_columns
        ]
        .fillna(0)
    )

    # ------------------------------------------------------------
    # TREND FEATURES
    # ------------------------------------------------------------

    features[
        "revenue_change_amount"
    ] = (
        features[
            "baseline_revenue"
        ]
        -
        features[
            "prior_window_revenue"
        ]
    )

    features[
        "revenue_trend_ratio"
    ] = np.where(
        features[
            "prior_window_revenue"
        ]
        >
        0,

        features[
            "baseline_revenue"
        ]
        /
        features[
            "prior_window_revenue"
        ],

        np.nan,
    )

    # ------------------------------------------------------------
    # SCORE DATE METADATA
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
            "Leakage feature table contains duplicate customers."
        )

    if (
        features[
            "baseline_revenue"
        ]
        <=
        0
    ).any():
        raise RuntimeError(
            "Leakage scoring population contains non-positive "
            "baseline revenue."
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
            "days_since_last_order"
        ]
        <
        0
    ).any():
        raise RuntimeError(
            "Negative days-since-last-order detected."
        )

    preferred_column_order = [
        "customer",
        "prediction_date",
        "prediction_period",
        "baseline_revenue",
        "baseline_orders",
        "baseline_units",
        "baseline_unique_products",
        "baseline_average_order_value",
        "prior_window_revenue",
        "prior_window_orders",
        "prior_window_units",
        "revenue_change_amount",
        "revenue_trend_ratio",
        "historical_order_count",
        "historical_revenue",
        "historical_units",
        "days_since_last_order",
        "tenure_days",
        "first_order_date",
        "last_order_date",
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