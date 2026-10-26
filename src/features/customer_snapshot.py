"""
Canonical customer snapshot builder.

This module creates one customer-level feature row as of an explicit
decision date.

The snapshot is built only from transactions that occurred on or before
the decision date.

It intentionally does not use future transactions.

Core outputs include:
- first order date
- last order date
- lifetime orders
- lifetime revenue
- recent order counts
- recent revenue
- days since last order
- customer tenure
- customer lifecycle state

Lifecycle states:
- Active: <= 30 days since last order
- At Risk: 31-90 days
- Dormant: 91-180 days
- Long Dormant: > 180 days
"""

from __future__ import annotations

import pandas as pd


# ================================================================
# REQUIRED SNAPSHOT INPUTS
# ================================================================

SNAPSHOT_REQUIRED_COLUMNS = [
    "date",
    "customer",
    "invoice",
    "quantity",
    "unit_price",
    "net_merchandise_revenue",
    "is_cancellation",
    "is_return",
]


DEFAULT_WINDOWS = (
    30,
    90,
    180,
)


# ================================================================
# INPUT VALIDATION
# ================================================================

def _validate_snapshot_columns(
    transactions: pd.DataFrame,
) -> None:
    """
    Ensure required customer snapshot columns are present.
    """

    missing = [
        column
        for column in SNAPSHOT_REQUIRED_COLUMNS
        if column not in transactions.columns
    ]

    if missing:
        raise ValueError(
            "Customer snapshot input is missing required columns: "
            + ", ".join(missing)
        )


# ================================================================
# DECISION DATE
# ================================================================

def normalise_decision_date(
    decision_date,
) -> pd.Timestamp:
    """
    Convert a supplied decision date into a pandas Timestamp.
    """

    timestamp = pd.Timestamp(
        decision_date
    )

    if pd.isna(timestamp):
        raise ValueError(
            "Decision date is invalid."
        )

    return timestamp


# ================================================================
# QUALIFYING POSITIVE MERCHANDISE
# ================================================================

def build_qualifying_merchandise_history(
    transactions: pd.DataFrame,
    decision_date,
) -> pd.DataFrame:
    """
    Return identified positive merchandise transactions available
    on or before the decision date.

    Rules:
    - transaction date <= decision date
    - customer ID must be present
    - positive net merchandise revenue
    - positive quantity
    - positive unit price
    - not cancellation
    - not return

    Returns and cancellations remain in the canonical transaction
    dataset, but are excluded from the positive merchandise customer
    snapshot.
    """

    _validate_snapshot_columns(
        transactions
    )

    snapshot_date = (
        normalise_decision_date(
            decision_date
        )
    )

    data = transactions.copy()

    data[
        "date"
    ] = pd.to_datetime(
        data[
            "date"
        ],
        errors="coerce",
        format="mixed",
    )

    if data[
        "date"
    ].isna().any():
        raise ValueError(
            "Customer snapshot input contains invalid or missing dates."
        )

    mask = (
        (
            data[
                "date"
            ]
            <=
            snapshot_date
        )
        &
        data[
            "customer"
        ]
        .notna()
        &
        (
            data[
                "net_merchandise_revenue"
            ]
            >
            0
        )
        &
        (
            data[
                "quantity"
            ]
            >
            0
        )
        &
        (
            data[
                "unit_price"
            ]
            >
            0
        )
        &
        (
            ~data[
                "is_cancellation"
            ]
            .fillna(False)
        )
        &
        (
            ~data[
                "is_return"
            ]
            .fillna(False)
        )
    )

    return (
        data.loc[
            mask
        ]
        .copy()
    )


# ================================================================
# WINDOW AGGREGATION
# ================================================================

def aggregate_customer_window(
    transactions: pd.DataFrame,
    decision_date,
    days: int,
) -> pd.DataFrame:
    """
    Aggregate order count and revenue over a trailing window.

    The interval is:

        decision_date - days < transaction_date <= decision_date

    This mirrors the research implementation.
    """

    if days <= 0:
        raise ValueError(
            "Window days must be greater than zero."
        )

    snapshot_date = (
        normalise_decision_date(
            decision_date
        )
    )

    start_date = (
        snapshot_date
        -
        pd.Timedelta(
            days=days
        )
    )

    window = (
        transactions.loc[
            transactions[
                "date"
            ]
            >
            start_date
        ]
    )

    prefix = (
        f"{days}d"
    )

    return (
        window
        .groupby(
            "customer",
            as_index=False
        )
        .agg(
            **{
                f"orders_prior_{prefix}":
                    (
                        "invoice",
                        "nunique",
                    ),

                f"revenue_prior_{prefix}":
                    (
                        "net_merchandise_revenue",
                        "sum",
                    ),
            }
        )
    )


# ================================================================
# LIFECYCLE STATE
# ================================================================

def classify_customer_state(
    days_since_last_order,
) -> str:
    """
    Classify a customer using transaction recency.
    """

    if pd.isna(
        days_since_last_order
    ):
        return "Unclassified"

    if days_since_last_order <= 30:
        return "Active"

    if days_since_last_order <= 90:
        return "At Risk"

    if days_since_last_order <= 180:
        return "Dormant"

    return "Long Dormant"


# ================================================================
# CUSTOMER SNAPSHOT BUILDER
# ================================================================

def build_customer_snapshot(
    transactions: pd.DataFrame,
    decision_date,
    windows=DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Build one canonical customer row as of a decision date.

    Parameters
    ----------
    transactions:
        Canonical validated transaction DataFrame.

    decision_date:
        Point-in-time cutoff. Transactions after this timestamp
        are excluded.

    windows:
        Trailing windows used for order-count and revenue features.

    Returns
    -------
    pd.DataFrame
        One row per identified customer with point-in-time features.
    """

    snapshot_date = (
        normalise_decision_date(
            decision_date
        )
    )

    history = (
        build_qualifying_merchandise_history(
            transactions,
            snapshot_date,
        )
    )

    if history.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "first_order_date",
                "last_order_date",
                "lifetime_orders",
                "lifetime_revenue",
                "days_since_last_order",
                "customer_tenure_days",
                "customer_state",
                "decision_date",
            ]
        )

    snapshot = (
        history
        .groupby(
            "customer",
            as_index=False
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
        )
    )

    snapshot[
        "days_since_last_order"
    ] = (
        (
            snapshot_date
            -
            snapshot[
                "last_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    snapshot[
        "customer_tenure_days"
    ] = (
        (
            snapshot_date
            -
            snapshot[
                "first_order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    window_feature_columns = []

    for days in windows:

        window_summary = (
            aggregate_customer_window(
                history,
                snapshot_date,
                days,
            )
        )

        snapshot = (
            snapshot.merge(
                window_summary,
                on="customer",
                how="left",
            )
        )

        window_feature_columns.extend(
            [
                f"orders_prior_{days}d",
                f"revenue_prior_{days}d",
            ]
        )

    existing_window_columns = [
        column
        for column in window_feature_columns
        if column in snapshot.columns
    ]

    snapshot[
        existing_window_columns
    ] = (
        snapshot[
            existing_window_columns
        ]
        .fillna(0)
    )

    snapshot[
        "customer_state"
    ] = (
        snapshot[
            "days_since_last_order"
        ]
        .apply(
            classify_customer_state
        )
    )

    snapshot[
        "decision_date"
    ] = snapshot_date

    if not snapshot[
        "customer"
    ].is_unique:
        raise RuntimeError(
            "Customer snapshot contains duplicate customer rows."
        )

    if (
        snapshot[
            "last_order_date"
        ]
        >
        snapshot_date
    ).any():
        raise RuntimeError(
            "Future transaction leakage detected in customer snapshot."
        )

    return (
        snapshot
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )