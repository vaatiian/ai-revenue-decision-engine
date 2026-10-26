"""
Reusable point-in-time customer history features.

This module builds customer-level historical behaviour features using
only qualifying merchandise transactions available on or before an
explicit decision date.

The module is intended to provide common feature definitions for:
- Reactivation
- Revenue Leakage
- Future Customer Value
- Customer Decision Engine

It intentionally does not calculate model labels or use future data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.customer_snapshot import (
    build_qualifying_merchandise_history,
    normalise_decision_date,
)


# ================================================================
# ORDER-LEVEL HISTORY
# ================================================================

def build_customer_order_history(
    transactions: pd.DataFrame,
    decision_date,
) -> pd.DataFrame:
    """
    Build one row per customer-order using only transactions available
    on or before the decision date.

    Returns:
        customer
        invoice
        order_date
        order_revenue
        order_units
        distinct_products
    """

    snapshot_date = normalise_decision_date(
        decision_date
    )

    history = build_qualifying_merchandise_history(
        transactions,
        snapshot_date,
    )

    if history.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "invoice",
                "order_date",
                "order_revenue",
                "order_units",
                "distinct_products",
            ]
        )

    order_history = (
        history
        .groupby(
            [
                "customer",
                "invoice",
            ],
            as_index=False,
        )
        .agg(
            order_date=(
                "date",
                "min",
            ),

            order_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            order_units=(
                "quantity",
                "sum",
            ),

            distinct_products=(
                "product",
                "nunique",
            ),
        )
    )

    order_history = (
        order_history
        .sort_values(
            [
                "customer",
                "order_date",
                "invoice",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        order_history[
            "order_date"
        ]
        >
        snapshot_date
    ).any():
        raise RuntimeError(
            "Future order leakage detected."
        )

    return order_history


# ================================================================
# INTER-ORDER GAPS
# ================================================================

def add_inter_order_gaps(
    order_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the number of days since each customer's previous order.
    """

    result = (
        order_history
        .sort_values(
            [
                "customer",
                "order_date",
                "invoice",
            ]
        )
        .copy()
    )

    result[
        "previous_order_date"
    ] = (
        result
        .groupby(
            "customer"
        )[
            "order_date"
        ]
        .shift(1)
    )

    result[
        "gap_days"
    ] = (
        (
            result[
                "order_date"
            ]
            -
            result[
                "previous_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    return result


# ================================================================
# INACTIVITY / REACTIVATION HISTORY
# ================================================================

def add_inactivity_episode_flags(
    order_history: pd.DataFrame,
    inactivity_threshold_days: int = 90,
) -> pd.DataFrame:
    """
    Mark historical returns after long inactivity.

    A reactivation is defined here as an observed order following an
    inter-order gap strictly greater than the inactivity threshold.

    This is a historical behaviour feature only.

    It does not create a future reactivation label.
    """

    if inactivity_threshold_days <= 0:
        raise ValueError(
            "Inactivity threshold must be greater than zero."
        )

    result = add_inter_order_gaps(
        order_history
    )

    result[
        "is_reactivation_order"
    ] = (
        result[
            "gap_days"
        ]
        >
        inactivity_threshold_days
    )

    return result


# ================================================================
# CUSTOMER HISTORY FEATURES
# ================================================================

def build_customer_history_features(
    transactions: pd.DataFrame,
    decision_date,
    inactivity_threshold_days: int = 90,
) -> pd.DataFrame:
    """
    Build reusable historical customer features as of a decision date.

    Features include:
    - historical order count
    - historical revenue
    - historical units
    - historical unique products
    - historical average order value
    - historical average order units
    - average and median inter-order gap
    - maximum inter-order gap
    - first order date
    - last order date
    - last order value
    - last order units
    - customer tenure
    - days since last order
    - prior inactivity episode count
    - prior reactivation count
    - prior reactivation rate

    No future information is used.
    """

    snapshot_date = normalise_decision_date(
        decision_date
    )

    order_history = build_customer_order_history(
        transactions,
        snapshot_date,
    )

    if order_history.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "historical_order_count",
                "historical_revenue",
                "historical_units",
                "historical_unique_products",
                "historical_average_order_value",
                "historical_average_order_units",
                "historical_average_gap_days",
                "historical_median_gap_days",
                "historical_max_gap_days",
                "first_order_date",
                "last_order_date",
                "last_order_value",
                "last_order_units",
                "tenure_at_prediction_days",
                "days_since_last_order",
                "prior_inactivity_episode_count",
                "prior_reactivation_count",
                "prior_reactivation_rate",
                "prediction_date",
            ]
        )

    order_history = add_inactivity_episode_flags(
        order_history,
        inactivity_threshold_days=inactivity_threshold_days,
    )

    # ------------------------------------------------------------
    # ORDER-LEVEL SUMMARY
    # ------------------------------------------------------------

    summary = (
        order_history
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

            historical_average_order_units=(
                "order_units",
                "mean",
            ),

            historical_average_gap_days=(
                "gap_days",
                "mean",
            ),

            historical_median_gap_days=(
                "gap_days",
                "median",
            ),

            historical_max_gap_days=(
                "gap_days",
                "max",
            ),

            first_order_date=(
                "order_date",
                "min",
            ),

            last_order_date=(
                "order_date",
                "max",
            ),

            prior_reactivation_count=(
                "is_reactivation_order",
                "sum",
            ),
        )
    )

    # ------------------------------------------------------------
    # UNIQUE PRODUCTS
    # ------------------------------------------------------------

    merchandise_history = (
        build_qualifying_merchandise_history(
            transactions,
            snapshot_date,
        )
    )

    unique_products = (
        merchandise_history
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            historical_unique_products=(
                "product",
                "nunique",
            )
        )
    )

    summary = summary.merge(
        unique_products,
        on="customer",
        how="left",
    )

    # ------------------------------------------------------------
    # LAST ORDER FEATURES
    # ------------------------------------------------------------

    last_orders = (
        order_history
        .sort_values(
            [
                "customer",
                "order_date",
                "invoice",
            ]
        )
        .drop_duplicates(
            subset=[
                "customer"
            ],
            keep="last",
        )[
            [
                "customer",
                "order_revenue",
                "order_units",
            ]
        ]
        .rename(
            columns={
                "order_revenue":
                    "last_order_value",

                "order_units":
                    "last_order_units",
            }
        )
    )

    summary = summary.merge(
        last_orders,
        on="customer",
        how="left",
    )

    # ------------------------------------------------------------
    # TENURE / RECENCY
    # ------------------------------------------------------------

    summary[
        "tenure_at_prediction_days"
    ] = (
        (
            snapshot_date
            -
            summary[
                "first_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    summary[
        "days_since_last_order"
    ] = (
        (
            snapshot_date
            -
            summary[
                "last_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    # ------------------------------------------------------------
    # PRIOR INACTIVITY / REACTIVATION HISTORY
    # ------------------------------------------------------------

    summary[
        "prior_inactivity_episode_count"
    ] = (
        summary[
            "prior_reactivation_count"
        ]
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

    # ------------------------------------------------------------
    # PREDICTION DATE
    # ------------------------------------------------------------

    summary[
        "prediction_date"
    ] = snapshot_date

    # ------------------------------------------------------------
    # SAFETY CHECKS
    # ------------------------------------------------------------

    if not summary[
        "customer"
    ].is_unique:
        raise RuntimeError(
            "Customer history feature table contains duplicate customers."
        )

    if (
        summary[
            "last_order_date"
        ]
        >
        snapshot_date
    ).any():
        raise RuntimeError(
            "Future transaction leakage detected in customer history features."
        )

    if (
        summary[
            "days_since_last_order"
        ]
        <
        0
    ).any():
        raise RuntimeError(
            "Negative days-since-last-order detected."
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