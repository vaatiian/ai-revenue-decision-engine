"""
Production Future Customer Value feature pipeline.

This module builds point-in-time customer features for Future Customer
Value scoring.

The production pipeline uses only information available on or before an
explicit prediction date.

Feature families include:
- 30-day revenue and orders
- 90-day revenue and orders
- 180-day revenue, orders and units
- prior 90-day activity
- revenue and order trends
- lifetime customer history
- recency
- tenure
- average order value
- purchase intensity

The future customer value target is intentionally not created here.
Future outcomes belong only in model training and evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.customer_snapshot import (
    build_qualifying_merchandise_history,
    normalise_decision_date,
)


DEFAULT_ACTIVE_WINDOW_DAYS = 180


# ================================================================
# WINDOW HELPER
# ================================================================

def _select_window(
    history: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Select transactions in:

        start_date < transaction_date <= end_date
    """

    return (
        history.loc[
            (
                history[
                    "date"
                ]
                >
                start_date
            )
            &
            (
                history[
                    "date"
                ]
                <=
                end_date
            )
        ]
        .copy()
    )


# ================================================================
# GENERIC CUSTOMER WINDOW SUMMARY
# ================================================================

def build_fcv_window_summary(
    transactions: pd.DataFrame,
    prediction_date,
    window_days: int,
) -> pd.DataFrame:
    """
    Build customer activity features for a trailing historical window.

    Example:

        window_days=90

    produces activity from:

        prediction_date - 90 days
        <
        transaction_date
        <=
        prediction_date
    """

    if window_days <= 0:
        raise ValueError(
            "window_days must be greater than zero."
        )

    score_date = normalise_decision_date(
        prediction_date
    )

    history = (
        build_qualifying_merchandise_history(
            transactions,
            score_date,
        )
    )

    prefix = (
        f"{window_days}d"
    )

    output_columns = [
        "customer",
        f"revenue_prior_{prefix}",
        f"orders_prior_{prefix}",
        f"units_prior_{prefix}",
        f"unique_products_prior_{prefix}",
    ]

    if history.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    start_date = (
        score_date
        -
        pd.Timedelta(
            days=window_days
        )
    )

    window = _select_window(
        history=history,
        start_date=start_date,
        end_date=score_date,
    )

    if window.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    summary = (
        window
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            **{
                f"revenue_prior_{prefix}": (
                    "net_merchandise_revenue",
                    "sum",
                ),

                f"orders_prior_{prefix}": (
                    "invoice",
                    "nunique",
                ),

                f"units_prior_{prefix}": (
                    "quantity",
                    "sum",
                ),

                f"unique_products_prior_{prefix}": (
                    "product",
                    "nunique",
                ),
            }
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
# PRIOR 90-DAY COMPARISON WINDOW
# ================================================================

def build_fcv_prior_90d_summary(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build the 90-day window immediately before the current 90-day
    feature window.

    Current:
        prediction_date - 90d < date <= prediction_date

    Previous:
        prediction_date - 180d < date <= prediction_date - 90d
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    history = (
        build_qualifying_merchandise_history(
            transactions,
            score_date,
        )
    )

    output_columns = [
        "customer",
        "revenue_previous_90d",
        "orders_previous_90d",
        "units_previous_90d",
    ]

    if history.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    current_start = (
        score_date
        -
        pd.Timedelta(
            days=90
        )
    )

    previous_start = (
        score_date
        -
        pd.Timedelta(
            days=180
        )
    )

    previous_window = _select_window(
        history=history,
        start_date=previous_start,
        end_date=current_start,
    )

    if previous_window.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    summary = (
        previous_window
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            revenue_previous_90d=(
                "net_merchandise_revenue",
                "sum",
            ),

            orders_previous_90d=(
                "invoice",
                "nunique",
            ),

            units_previous_90d=(
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
# LIFETIME CUSTOMER HISTORY
# ================================================================

def build_fcv_customer_history(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build lifetime point-in-time customer history features.
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    history = (
        build_qualifying_merchandise_history(
            transactions,
            score_date,
        )
    )

    output_columns = [
        "customer",
        "first_order_date",
        "last_order_date",
        "lifetime_orders",
        "lifetime_revenue",
        "lifetime_units",
        "lifetime_unique_products",
        "lifetime_average_order_value",
        "days_since_last_order",
        "customer_tenure_days",
        "orders_per_30_tenure_days",
        "revenue_per_30_tenure_days",
    ]

    if history.empty:
        return pd.DataFrame(
            columns=output_columns
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

            lifetime_orders=(
                "invoice",
                "nunique",
            ),

            lifetime_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            lifetime_units=(
                "quantity",
                "sum",
            ),

            lifetime_unique_products=(
                "product",
                "nunique",
            ),
        )
    )

    summary[
        "lifetime_average_order_value"
    ] = np.where(
        summary[
            "lifetime_orders"
        ]
        >
        0,

        summary[
            "lifetime_revenue"
        ]
        /
        summary[
            "lifetime_orders"
        ],

        0.0,
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
        "customer_tenure_days"
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

    # Prevent division by zero for customers first observed on the
    # prediction date.
    effective_tenure = (
        summary[
            "customer_tenure_days"
        ]
        .clip(
            lower=1
        )
    )

    summary[
        "orders_per_30_tenure_days"
    ] = (
        summary[
            "lifetime_orders"
        ]
        /
        effective_tenure
        *
        30
    )

    summary[
        "revenue_per_30_tenure_days"
    ] = (
        summary[
            "lifetime_revenue"
        ]
        /
        effective_tenure
        *
        30
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
# FINAL FCV FEATURE TABLE
# ================================================================

def build_fcv_features(
    transactions: pd.DataFrame,
    prediction_date,
    active_window_days: int = DEFAULT_ACTIVE_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Build the production Future Customer Value feature table.

    Eligibility rule:

        revenue_prior_active_window > 0

    With the default configuration, a customer must have qualifying
    merchandise activity in the prior 180 days.

    This prevents customers with no meaningful recent purchase history
    from entering the FCV scoring population.

    No future customer value target is created.
    """

    if active_window_days <= 0:
        raise ValueError(
            "active_window_days must be greater than zero."
        )

    score_date = normalise_decision_date(
        prediction_date
    )

    window_30 = build_fcv_window_summary(
        transactions=transactions,
        prediction_date=score_date,
        window_days=30,
    )

    window_90 = build_fcv_window_summary(
        transactions=transactions,
        prediction_date=score_date,
        window_days=90,
    )

    window_180 = build_fcv_window_summary(
        transactions=transactions,
        prediction_date=score_date,
        window_days=180,
    )

    prior_90 = build_fcv_prior_90d_summary(
        transactions=transactions,
        prediction_date=score_date,
    )

    lifetime = build_fcv_customer_history(
        transactions=transactions,
        prediction_date=score_date,
    )

    if lifetime.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------
    # ACTIVE SCORING POPULATION
    # ------------------------------------------------------------

    active_window = build_fcv_window_summary(
        transactions=transactions,
        prediction_date=score_date,
        window_days=active_window_days,
    )

    active_revenue_column = (
        f"revenue_prior_{active_window_days}d"
    )

    if active_window.empty:
        return pd.DataFrame()

    eligible = (
        active_window.loc[
            active_window[
                active_revenue_column
            ]
            >
            0,
            [
                "customer",
            ],
        ]
        .copy()
    )

    if eligible.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------
    # MERGE FEATURE FAMILIES
    # ------------------------------------------------------------

    features = (
        eligible
        .merge(
            lifetime,
            on="customer",
            how="left",
        )
        .merge(
            window_30,
            on="customer",
            how="left",
        )
        .merge(
            window_90,
            on="customer",
            how="left",
        )
        .merge(
            window_180,
            on="customer",
            how="left",
        )
        .merge(
            prior_90,
            on="customer",
            how="left",
        )
    )

    # ------------------------------------------------------------
    # ZERO-FILL OBSERVED ACTIVITY WINDOWS
    # ------------------------------------------------------------

    activity_columns = [
        "revenue_prior_30d",
        "orders_prior_30d",
        "units_prior_30d",
        "unique_products_prior_30d",
        "revenue_prior_90d",
        "orders_prior_90d",
        "units_prior_90d",
        "unique_products_prior_90d",
        "revenue_prior_180d",
        "orders_prior_180d",
        "units_prior_180d",
        "unique_products_prior_180d",
        "revenue_previous_90d",
        "orders_previous_90d",
        "units_previous_90d",
    ]

    existing_activity_columns = [
        column
        for column in activity_columns
        if column in features.columns
    ]

    features[
        existing_activity_columns
    ] = (
        features[
            existing_activity_columns
        ]
        .fillna(0)
    )

    # ------------------------------------------------------------
    # RECENT AVERAGE ORDER VALUES
    # ------------------------------------------------------------

    features[
        "average_order_value_prior_30d"
    ] = np.where(
        features[
            "orders_prior_30d"
        ]
        >
        0,

        features[
            "revenue_prior_30d"
        ]
        /
        features[
            "orders_prior_30d"
        ],

        0.0,
    )

    features[
        "average_order_value_prior_90d"
    ] = np.where(
        features[
            "orders_prior_90d"
        ]
        >
        0,

        features[
            "revenue_prior_90d"
        ]
        /
        features[
            "orders_prior_90d"
        ],

        0.0,
    )

    features[
        "average_order_value_prior_180d"
    ] = np.where(
        features[
            "orders_prior_180d"
        ]
        >
        0,

        features[
            "revenue_prior_180d"
        ]
        /
        features[
            "orders_prior_180d"
        ],

        0.0,
    )

    # ------------------------------------------------------------
    # TREND FEATURES
    # ------------------------------------------------------------

    features[
        "revenue_change_90d"
    ] = (
        features[
            "revenue_prior_90d"
        ]
        -
        features[
            "revenue_previous_90d"
        ]
    )

    features[
        "order_change_90d"
    ] = (
        features[
            "orders_prior_90d"
        ]
        -
        features[
            "orders_previous_90d"
        ]
    )

    features[
        "revenue_trend_ratio_90d"
    ] = np.where(
        features[
            "revenue_previous_90d"
        ]
        >
        0,

        features[
            "revenue_prior_90d"
        ]
        /
        features[
            "revenue_previous_90d"
        ],

        np.nan,
    )

    features[
        "order_trend_ratio_90d"
    ] = np.where(
        features[
            "orders_previous_90d"
        ]
        >
        0,

        features[
            "orders_prior_90d"
        ]
        /
        features[
            "orders_previous_90d"
        ],

        np.nan,
    )

    # ------------------------------------------------------------
    # RECENT PURCHASE INTENSITY
    # ------------------------------------------------------------

    features[
        "orders_per_30d_recent_180d"
    ] = (
        features[
            "orders_prior_180d"
        ]
        /
        6.0
    )

    features[
        "revenue_per_30d_recent_180d"
    ] = (
        features[
            "revenue_prior_180d"
        ]
        /
        6.0
    )

    # ------------------------------------------------------------
    # METADATA
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
            "FCV feature table contains duplicate customers."
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

    if (
        features[
            active_revenue_column
        ]
        <=
        0
    ).any():
        raise RuntimeError(
            "Inactive customer entered the FCV scoring population."
        )

    # ------------------------------------------------------------
    # PREFERRED COLUMN ORDER
    # ------------------------------------------------------------

    preferred_column_order = [
        "customer",
        "prediction_date",
        "prediction_period",

        "revenue_prior_30d",
        "orders_prior_30d",
        "units_prior_30d",
        "unique_products_prior_30d",
        "average_order_value_prior_30d",

        "revenue_prior_90d",
        "orders_prior_90d",
        "units_prior_90d",
        "unique_products_prior_90d",
        "average_order_value_prior_90d",

        "revenue_previous_90d",
        "orders_previous_90d",
        "units_previous_90d",

        "revenue_change_90d",
        "order_change_90d",
        "revenue_trend_ratio_90d",
        "order_trend_ratio_90d",

        "revenue_prior_180d",
        "orders_prior_180d",
        "units_prior_180d",
        "unique_products_prior_180d",
        "average_order_value_prior_180d",

        "orders_per_30d_recent_180d",
        "revenue_per_30d_recent_180d",

        "lifetime_orders",
        "lifetime_revenue",
        "lifetime_units",
        "lifetime_unique_products",
        "lifetime_average_order_value",

        "orders_per_30_tenure_days",
        "revenue_per_30_tenure_days",

        "days_since_last_order",
        "customer_tenure_days",
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