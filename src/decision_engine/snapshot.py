"""
Customer decision-snapshot adapter.

This module provides the canonical customer-state foundation used by the
production Customer Decision Engine.

It deliberately delegates customer-history construction to
src.features.customer_snapshot rather than reproducing feature logic inside
the orchestration layer.

Responsibilities of this module:

- freeze the Decision Engine snapshot windows to 30, 90 and 180 days;
- call the canonical leakage-safe customer snapshot builder;
- normalize the exact internal Decision Engine snapshot schema;
- validate one-row-per-customer and temporal invariants;
- return a stable schema even when no eligible customers exist.

Model signals are attached later by the signal-alignment layer.
No primary action, ranking, or contact policy is calculated here.
"""

from __future__ import annotations

import pandas as pd

from src.decision_engine.spec import (
    CUSTOMER_STATES,
)

from src.features.customer_snapshot import (
    build_customer_snapshot,
    normalise_decision_date,
)


# ================================================================
# DECISION SNAPSHOT CONTRACT
# ================================================================


DECISION_SNAPSHOT_WINDOWS = (
    30,
    90,
    180,
)


DECISION_SNAPSHOT_COLUMNS = (
    "customer",
    "decision_date",
    "first_order_date",
    "last_order_date",
    "customer_state",
    "days_since_last_order",
    "customer_tenure_days",
    "lifetime_orders",
    "lifetime_revenue",
    "orders_prior_30d",
    "orders_prior_90d",
    "orders_prior_180d",
    "revenue_prior_30d",
    "revenue_prior_90d",
    "revenue_prior_180d",
)


# ================================================================
# EMPTY SNAPSHOT
# ================================================================


def _empty_decision_snapshot() -> pd.DataFrame:
    """
    Return an empty Decision Engine snapshot with the frozen schema.
    """

    return pd.DataFrame(
        columns=list(
            DECISION_SNAPSHOT_COLUMNS
        )
    )


# ================================================================
# SNAPSHOT VALIDATION
# ================================================================


def validate_decision_snapshot(
    snapshot: pd.DataFrame,
    decision_date,
) -> None:
    """
    Fail closed if a Decision Engine customer snapshot violates contract.
    """

    if not isinstance(
        snapshot,
        pd.DataFrame,
    ):
        raise TypeError(
            "Decision snapshot must be a pandas DataFrame."
        )

    expected_columns = list(
        DECISION_SNAPSHOT_COLUMNS
    )

    actual_columns = list(
        snapshot.columns
    )

    if actual_columns != expected_columns:
        raise ValueError(
            "Decision snapshot columns do not match the frozen contract. "
            f"Expected {expected_columns}, got {actual_columns}."
        )

    if snapshot.empty:
        return

    snapshot_date = (
        normalise_decision_date(
            decision_date
        )
    )

    # ------------------------------------------------------------
    # Customer identity
    # ------------------------------------------------------------

    if snapshot[
        "customer"
    ].isna().any():
        raise ValueError(
            "Decision snapshot contains missing customer IDs."
        )

    if not snapshot[
        "customer"
    ].is_unique:
        raise ValueError(
            "Decision snapshot must contain one row per customer."
        )

    # ------------------------------------------------------------
    # Decision date
    # ------------------------------------------------------------

    observed_decision_dates = pd.to_datetime(
        snapshot[
            "decision_date"
        ],
        errors="coerce",
    )

    if observed_decision_dates.isna().any():
        raise ValueError(
            "Decision snapshot contains invalid decision dates."
        )

    if not (
        observed_decision_dates
        ==
        snapshot_date
    ).all():
        raise ValueError(
            "Decision snapshot contains inconsistent decision dates."
        )

    # ------------------------------------------------------------
    # Transaction chronology
    # ------------------------------------------------------------

    first_order_dates = pd.to_datetime(
        snapshot[
            "first_order_date"
        ],
        errors="coerce",
    )

    last_order_dates = pd.to_datetime(
        snapshot[
            "last_order_date"
        ],
        errors="coerce",
    )

    if (
        first_order_dates.isna().any()
        or
        last_order_dates.isna().any()
    ):
        raise ValueError(
            "Decision snapshot contains invalid order dates."
        )

    if (
        last_order_dates
        >
        snapshot_date
    ).any():
        raise ValueError(
            "Future transaction leakage detected in decision snapshot."
        )

    if (
        first_order_dates
        >
        last_order_dates
    ).any():
        raise ValueError(
            "Decision snapshot contains first order dates after last "
            "order dates."
        )

    # ------------------------------------------------------------
    # Recency and tenure
    # ------------------------------------------------------------

    if (
        snapshot[
            "days_since_last_order"
        ]
        <
        0
    ).any():
        raise ValueError(
            "Decision snapshot contains negative customer recency."
        )

    if (
        snapshot[
            "customer_tenure_days"
        ]
        <
        0
    ).any():
        raise ValueError(
            "Decision snapshot contains negative customer tenure."
        )

    # ------------------------------------------------------------
    # Lifecycle state
    # ------------------------------------------------------------

    unknown_states = (
        set(
            snapshot[
                "customer_state"
            ]
            .dropna()
            .astype(str)
        )
        -
        set(
            CUSTOMER_STATES
        )
    )

    if unknown_states:
        raise ValueError(
            "Decision snapshot contains unknown customer states: "
            f"{sorted(unknown_states)}"
        )

    if snapshot[
        "customer_state"
    ].isna().any():
        raise ValueError(
            "Decision snapshot contains missing customer states."
        )

    # ------------------------------------------------------------
    # Customer economics
    # ------------------------------------------------------------

    non_negative_columns = (
        "lifetime_orders",
        "lifetime_revenue",
        "orders_prior_30d",
        "orders_prior_90d",
        "orders_prior_180d",
        "revenue_prior_30d",
        "revenue_prior_90d",
        "revenue_prior_180d",
    )

    for column in non_negative_columns:

        if (
            snapshot[
                column
            ]
            <
            0
        ).any():
            raise ValueError(
                "Decision snapshot contains negative value in "
                f"{column!r}."
            )


# ================================================================
# SNAPSHOT BUILDER
# ================================================================


def build_decision_snapshot(
    transactions: pd.DataFrame,
    decision_date,
) -> pd.DataFrame:
    """
    Build the canonical customer foundation for Decision Engine orchestration.

    The customer-history calculations are delegated to the canonical
    customer snapshot feature pipeline.

    Parameters
    ----------
    transactions:
        Canonical transaction DataFrame.

    decision_date:
        Point-in-time Decision Engine snapshot timestamp.

    Returns
    -------
    pd.DataFrame
        One row per identified customer using only qualifying transaction
        history available on or before the decision date.
    """

    snapshot_date = (
        normalise_decision_date(
            decision_date
        )
    )

    canonical_snapshot = (
        build_customer_snapshot(
            transactions=transactions,
            decision_date=snapshot_date,
            windows=DECISION_SNAPSHOT_WINDOWS,
        )
    )

    if canonical_snapshot.empty:

        result = (
            _empty_decision_snapshot()
        )

        validate_decision_snapshot(
            snapshot=result,
            decision_date=snapshot_date,
        )

        return result

    missing_columns = [
        column
        for column
        in DECISION_SNAPSHOT_COLUMNS
        if column
        not in canonical_snapshot.columns
    ]

    if missing_columns:
        raise ValueError(
            "Canonical customer snapshot is missing Decision Engine "
            "columns: "
            + ", ".join(
                missing_columns
            )
        )

    result = (
        canonical_snapshot.loc[
            :,
            list(
                DECISION_SNAPSHOT_COLUMNS
            ),
        ]
        .copy()
    )

    result = (
        result
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )

    validate_decision_snapshot(
        snapshot=result,
        decision_date=snapshot_date,
    )

    return result