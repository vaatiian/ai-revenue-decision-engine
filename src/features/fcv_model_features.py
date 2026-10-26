"""
Research-compatible production feature adapter for the frozen
Future Customer Value Stage A model.

This module reproduces the exact feature architecture used in
Notebook 05 — ML Future Customer Value.

Prediction unit
---------------
One identified customer at one monthly prediction point.

Eligibility
-----------
At least one qualifying merchandise purchase during the preceding
180 days.

Historical windows
------------------
h180:
    prediction_date - 180 days < date <= prediction_date

r90:
    prediction_date - 90 days < date <= prediction_date

p90:
    prediction_date - 180 days < date <= prediction_date - 90 days

r30:
    prediction_date - 30 days < date <= prediction_date

The final output is one row per eligible customer containing the exact
50 frozen Stage A model features in the exact research order.

No future outcome information is created or consumed here.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.features.customer_snapshot import (
    build_qualifying_merchandise_history,
    normalise_decision_date,
)
from src.models.fcv_spec import (
    FCV_ELIGIBILITY_LOOKBACK_DAYS,
    FCV_MODEL_FEATURES,
    FCV_RECENT_WINDOW_DAYS,
)


# ================================================================
# REQUIRED SOURCE FIELDS
# ================================================================

REQUIRED_FCV_TRANSACTION_COLUMNS = {
    "date",
    "customer",
    "invoice",
    "product",
    "region",
    "quantity",
    "unit_price",
    "net_merchandise_revenue",
}


# ================================================================
# DATE / SOURCE VALIDATION
# ================================================================

def normalise_fcv_prediction_date(
    prediction_date,
) -> pd.Timestamp:
    """
    Convert prediction date to a timezone-naive normalized Timestamp.
    """

    return normalise_decision_date(
        prediction_date
    )


def validate_fcv_month_end(
    prediction_date,
) -> None:
    """
    FCV research scoring points are month-end prediction dates.
    """

    score_date = normalise_fcv_prediction_date(
        prediction_date
    )

    if not score_date.is_month_end:
        raise ValueError(
            "FCV scoring requires a month-end prediction date."
        )


def validate_fcv_transaction_columns(
    transactions: pd.DataFrame,
) -> None:
    """
    Fail closed when required canonical transaction fields are missing.
    """

    missing = (
        REQUIRED_FCV_TRANSACTION_COLUMNS
        -
        set(
            transactions.columns
        )
    )

    if missing:
        raise ValueError(
            "FCV model feature engineering is missing required "
            "transaction columns: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )


# ================================================================
# QUALIFYING PURCHASE HISTORY
# ================================================================

def build_fcv_positive_purchase_history(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build qualifying identified-customer merchandise history available
    on or before the prediction date.

    The canonical purchase qualification rules are delegated to the
    shared customer-snapshot implementation.
    """

    validate_fcv_transaction_columns(
        transactions
    )

    score_date = normalise_fcv_prediction_date(
        prediction_date
    )

    history = (
        build_qualifying_merchandise_history(
            transactions,
            score_date,
        )
        .copy()
    )

    if history.empty:
        return history

    history["date"] = pd.to_datetime(
        history["date"],
        errors="raise",
    )

    history["purchase_day"] = (
        history["date"]
        .dt
        .normalize()
    )

    history["quantity"] = pd.to_numeric(
        history["quantity"],
        errors="raise",
    )

    history["unit_price"] = pd.to_numeric(
        history["unit_price"],
        errors="raise",
    )

    history[
        "net_merchandise_revenue"
    ] = pd.to_numeric(
        history[
            "net_merchandise_revenue"
        ],
        errors="raise",
    )

    return history


# ================================================================
# CUSTOMER-DAY HISTORY
# ================================================================

def build_fcv_customer_daily_history(
    line_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse qualifying line items to customer-day economics.

    Research-compatible fields:
        daily_revenue
        daily_orders
        daily_units
    """

    if line_data.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "date",
                "daily_revenue",
                "daily_orders",
                "daily_units",
            ]
        )

    daily = (
        line_data
        .groupby(
            [
                "customer",
                "purchase_day",
            ],
            as_index=False,
        )
        .agg(
            daily_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),
            daily_orders=(
                "invoice",
                "nunique",
            ),
            daily_units=(
                "quantity",
                "sum",
            ),
        )
        .rename(
            columns={
                "purchase_day":
                    "date",
            }
        )
        .sort_values(
            [
                "customer",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return daily


# ================================================================
# WINDOW HELPERS
# ================================================================

def summarise_fcv_customer_window(
    daily_data: pd.DataFrame,
    customers: Iterable,
    start_date,
    end_date,
    prefix: str,
) -> pd.DataFrame:
    """
    Exact Notebook 05 customer-window summary.

    Window semantics:
        start_date < date <= end_date
    """

    start_date = normalise_fcv_prediction_date(
        start_date
    )

    end_date = normalise_fcv_prediction_date(
        end_date
    )

    customer_values = list(
        customers
    )

    window = (
        daily_data.loc[
            daily_data[
                "customer"
            ].isin(
                customer_values
            )
            &
            (
                daily_data[
                    "date"
                ]
                >
                start_date
            )
            &
            (
                daily_data[
                    "date"
                ]
                <=
                end_date
            )
        ]
        .copy()
    )

    summary = (
        window
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            revenue=(
                "daily_revenue",
                "sum",
            ),
            orders=(
                "daily_orders",
                "sum",
            ),
            active_days=(
                "date",
                "nunique",
            ),
            units=(
                "daily_units",
                "sum",
            ),
            first_purchase_day=(
                "date",
                "min",
            ),
            last_purchase_day=(
                "date",
                "max",
            ),
        )
    )

    summary = summary.rename(
        columns={
            column:
                f"{prefix}_{column}"
            for column
            in summary.columns
            if column != "customer"
        }
    )

    return summary


def summarise_fcv_product_window(
    line_data: pd.DataFrame,
    customers: Iterable,
    start_date,
    end_date,
    prefix: str,
) -> pd.DataFrame:
    """
    Exact Notebook 05 product-window summary.
    """

    start_date = normalise_fcv_prediction_date(
        start_date
    )

    end_date = normalise_fcv_prediction_date(
        end_date
    )

    customer_values = list(
        customers
    )

    window = (
        line_data.loc[
            line_data[
                "customer"
            ].isin(
                customer_values
            )
            &
            (
                line_data[
                    "purchase_day"
                ]
                >
                start_date
            )
            &
            (
                line_data[
                    "purchase_day"
                ]
                <=
                end_date
            )
        ]
        .copy()
    )

    summary = (
        window
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            unique_products=(
                "product",
                "nunique",
            ),
            average_unit_price=(
                "unit_price",
                "mean",
            ),
            maximum_unit_price=(
                "unit_price",
                "max",
            ),
        )
    )

    summary = summary.rename(
        columns={
            column:
                f"{prefix}_{column}"
            for column
            in summary.columns
            if column != "customer"
        }
    )

    return summary


def summarise_fcv_purchase_gaps(
    daily_data: pd.DataFrame,
    customers: Iterable,
    start_date,
    end_date,
    prefix: str,
) -> pd.DataFrame:
    """
    Exact Notebook 05 purchase-gap summary.
    """

    start_date = normalise_fcv_prediction_date(
        start_date
    )

    end_date = normalise_fcv_prediction_date(
        end_date
    )

    customer_values = list(
        customers
    )

    window = (
        daily_data.loc[
            daily_data[
                "customer"
            ].isin(
                customer_values
            )
            &
            (
                daily_data[
                    "date"
                ]
                >
                start_date
            )
            &
            (
                daily_data[
                    "date"
                ]
                <=
                end_date
            ),
            [
                "customer",
                "date",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "customer",
                "date",
            ]
        )
        .copy()
    )

    window["gap_days"] = (
        window
        .groupby(
            "customer"
        )[
            "date"
        ]
        .diff()
        .dt
        .days
    )

    summary = (
        window
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            average_gap_days=(
                "gap_days",
                "mean",
            ),
            maximum_gap_days=(
                "gap_days",
                "max",
            ),
            minimum_gap_days=(
                "gap_days",
                "min",
            ),
        )
    )

    summary = summary.rename(
        columns={
            column:
                f"{prefix}_{column}"
            for column
            in summary.columns
            if column != "customer"
        }
    )

    return summary


# ================================================================
# REGION
# ================================================================

def build_fcv_last_region(
    line_data: pd.DataFrame,
    customers: Iterable,
    prediction_date,
) -> pd.DataFrame:
    """
    Return the region associated with the customer's latest qualifying
    purchase available at the prediction date.
    """

    prediction_date = normalise_fcv_prediction_date(
        prediction_date
    )

    customer_values = list(
        customers
    )

    available = (
        line_data.loc[
            line_data[
                "customer"
            ].isin(
                customer_values
            )
            &
            (
                line_data[
                    "purchase_day"
                ]
                <=
                prediction_date
            )
        ]
        .sort_values(
            [
                "customer",
                "purchase_day",
                "date",
            ]
        )
        .copy()
    )

    if available.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "last_region",
            ]
        )

    result = (
        available
        .groupby(
            "customer",
            as_index=False,
        )
        .tail(1)[
            [
                "customer",
                "region",
            ]
        ]
        .rename(
            columns={
                "region":
                    "last_region",
            }
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ================================================================
# ELIGIBLE POPULATION
# ================================================================

def build_fcv_eligible_population(
    daily_data: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Eligible FCV scoring population.

    A customer is eligible when they generated qualifying merchandise
    revenue during the preceding 180 days.
    """

    prediction_date = normalise_fcv_prediction_date(
        prediction_date
    )

    history_start = (
        prediction_date
        -
        pd.Timedelta(
            days=FCV_ELIGIBILITY_LOOKBACK_DAYS
        )
    )

    eligible = (
        daily_data.loc[
            (
                daily_data[
                    "date"
                ]
                >
                history_start
            )
            &
            (
                daily_data[
                    "date"
                ]
                <=
                prediction_date
            )
        ]
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            history_180d_revenue=(
                "daily_revenue",
                "sum",
            ),
        )
    )

    eligible = (
        eligible.loc[
            eligible[
                "history_180d_revenue"
            ]
            >
            0
        ]
        .reset_index(
            drop=True
        )
    )

    return eligible


# ================================================================
# FROZEN FCV MODEL FEATURE TABLE
# ================================================================

def build_fcv_model_features(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build the exact frozen 50-feature FCV Stage A model matrix.
    """

    score_date = normalise_fcv_prediction_date(
        prediction_date
    )

    validate_fcv_month_end(
        score_date
    )

    validate_fcv_transaction_columns(
        transactions
    )

    line_data = (
        build_fcv_positive_purchase_history(
            transactions=transactions,
            prediction_date=score_date,
        )
    )

    if line_data.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                *FCV_MODEL_FEATURES,
            ]
        )

    daily_data = (
        build_fcv_customer_daily_history(
            line_data
        )
    )

    eligible = (
        build_fcv_eligible_population(
            daily_data=daily_data,
            prediction_date=score_date,
        )
    )

    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                *FCV_MODEL_FEATURES,
            ]
        )

    customers = (
        eligible[
            "customer"
        ]
        .unique()
    )

    history180_start = (
        score_date
        -
        pd.Timedelta(
            days=180
        )
    )

    recent90_start = (
        score_date
        -
        pd.Timedelta(
            days=FCV_RECENT_WINDOW_DAYS
        )
    )

    recent30_start = (
        score_date
        -
        pd.Timedelta(
            days=30
        )
    )

    h180 = summarise_fcv_customer_window(
        daily_data=daily_data,
        customers=customers,
        start_date=history180_start,
        end_date=score_date,
        prefix="h180",
    )

    r90 = summarise_fcv_customer_window(
        daily_data=daily_data,
        customers=customers,
        start_date=recent90_start,
        end_date=score_date,
        prefix="r90",
    )

    p90 = summarise_fcv_customer_window(
        daily_data=daily_data,
        customers=customers,
        start_date=history180_start,
        end_date=recent90_start,
        prefix="p90",
    )

    r30 = summarise_fcv_customer_window(
        daily_data=daily_data,
        customers=customers,
        start_date=recent30_start,
        end_date=score_date,
        prefix="r30",
    )

    h180_products = summarise_fcv_product_window(
        line_data=line_data,
        customers=customers,
        start_date=history180_start,
        end_date=score_date,
        prefix="h180",
    )

    r90_products = summarise_fcv_product_window(
        line_data=line_data,
        customers=customers,
        start_date=recent90_start,
        end_date=score_date,
        prefix="r90",
    )

    p90_products = summarise_fcv_product_window(
        line_data=line_data,
        customers=customers,
        start_date=history180_start,
        end_date=recent90_start,
        prefix="p90",
    )

    h180_gaps = summarise_fcv_purchase_gaps(
        daily_data=daily_data,
        customers=customers,
        start_date=history180_start,
        end_date=score_date,
        prefix="h180",
    )

    last_region = build_fcv_last_region(
        line_data=line_data,
        customers=customers,
        prediction_date=score_date,
    )

    features = (
        eligible
        .merge(
            h180,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .merge(
            r90,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .merge(
            p90,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .merge(
            r30,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .merge(
            h180_products,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .merge(
            r90_products,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .merge(
            p90_products,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .merge(
            h180_gaps,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .merge(
            last_region,
            on="customer",
            how="left",
            validate="one_to_one",
        )
    )

    features[
        "prediction_date"
    ] = score_date

    # ------------------------------------------------------------
    # Structural zero fill
    # ------------------------------------------------------------

    zero_fill_features = [
        "r90_revenue",
        "r90_orders",
        "r90_active_days",
        "r90_units",
        "r90_unique_products",

        "p90_revenue",
        "p90_orders",
        "p90_active_days",
        "p90_units",
        "p90_unique_products",

        "r30_revenue",
        "r30_orders",
        "r30_active_days",
        "r30_units",
    ]

    for feature in zero_fill_features:

        if feature in features.columns:
            features[
                feature
            ] = (
                features[
                    feature
                ]
                .fillna(0)
            )

    # ------------------------------------------------------------
    # Research reconstruction audit
    # ------------------------------------------------------------

    if not np.allclose(
        features[
            "history_180d_revenue"
        ],
        features[
            "h180_revenue"
        ],
    ):
        raise RuntimeError(
            "FCV 180-day revenue reconstruction failed."
        )

    # ------------------------------------------------------------
    # Recency and activity indicators
    # ------------------------------------------------------------

    features[
        "days_since_last_purchase"
    ] = (
        features[
            "prediction_date"
        ]
        -
        features[
            "h180_last_purchase_day"
        ]
    ).dt.days

    features[
        "active_in_last_30d"
    ] = (
        features[
            "r30_orders"
        ]
        >
        0
    ).astype(int)

    features[
        "active_in_last_90d"
    ] = (
        features[
            "r90_orders"
        ]
        >
        0
    ).astype(int)

    # ------------------------------------------------------------
    # 180-day economics
    # ------------------------------------------------------------

    features[
        "h180_average_order_value"
    ] = (
        features[
            "h180_revenue"
        ]
        /
        features[
            "h180_orders"
        ]
    )

    features[
        "h180_revenue_per_active_day"
    ] = (
        features[
            "h180_revenue"
        ]
        /
        features[
            "h180_active_days"
        ]
    )

    features[
        "h180_units_per_order"
    ] = (
        features[
            "h180_units"
        ]
        /
        features[
            "h180_orders"
        ]
    )

    features[
        "h180_products_per_order"
    ] = (
        features[
            "h180_unique_products"
        ]
        /
        features[
            "h180_orders"
        ]
    )

    # ------------------------------------------------------------
    # Recent / previous 90-day economics
    # ------------------------------------------------------------

    features[
        "r90_average_order_value"
    ] = np.where(
        features[
            "r90_orders"
        ]
        >
        0,
        (
            features[
                "r90_revenue"
            ]
            /
            features[
                "r90_orders"
            ]
        ),
        np.nan,
    )

    features[
        "p90_average_order_value"
    ] = np.where(
        features[
            "p90_orders"
        ]
        >
        0,
        (
            features[
                "p90_revenue"
            ]
            /
            features[
                "p90_orders"
            ]
        ),
        np.nan,
    )

    # ------------------------------------------------------------
    # Revenue momentum
    # ------------------------------------------------------------

    features[
        "r90_minus_p90_revenue"
    ] = (
        features[
            "r90_revenue"
        ]
        -
        features[
            "p90_revenue"
        ]
    )

    features[
        "r90_to_p90_revenue_ratio"
    ] = np.where(
        features[
            "p90_revenue"
        ]
        >
        0,
        (
            features[
                "r90_revenue"
            ]
            /
            features[
                "p90_revenue"
            ]
        ),
        np.nan,
    )

    features[
        "log_revenue_momentum"
    ] = (
        np.log1p(
            features[
                "r90_revenue"
            ]
        )
        -
        np.log1p(
            features[
                "p90_revenue"
            ]
        )
    )

    # ------------------------------------------------------------
    # Recent 30-day momentum
    # ------------------------------------------------------------

    features[
        "recent30_revenue_share"
    ] = (
        features[
            "r30_revenue"
        ]
        /
        features[
            "h180_revenue"
        ]
    )

    features[
        "recent30_order_share"
    ] = (
        features[
            "r30_orders"
        ]
        /
        features[
            "h180_orders"
        ]
    )

    # ------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------

    features[
        "h180_orders_per_30_days"
    ] = (
        features[
            "h180_orders"
        ]
        /
        6
    )

    features[
        "h180_active_day_rate"
    ] = (
        features[
            "h180_active_days"
        ]
        /
        180
    )

    features[
        "r90_active_day_rate"
    ] = (
        features[
            "r90_active_days"
        ]
        /
        90
    )

    features[
        "p90_active_day_rate"
    ] = (
        features[
            "p90_active_days"
        ]
        /
        90
    )

    # ------------------------------------------------------------
    # Product breadth
    # ------------------------------------------------------------

    features[
        "product_breadth_change"
    ] = (
        features[
            "r90_unique_products"
        ]
        -
        features[
            "p90_unique_products"
        ]
    )

    # ------------------------------------------------------------
    # Log transforms
    # ------------------------------------------------------------

    features[
        "log1p_h180_revenue"
    ] = np.log1p(
        features[
            "h180_revenue"
        ]
    )

    features[
        "log1p_r90_revenue"
    ] = np.log1p(
        features[
            "r90_revenue"
        ]
    )

    features[
        "log1p_p90_revenue"
    ] = np.log1p(
        features[
            "p90_revenue"
        ]
    )

    features[
        "log1p_h180_orders"
    ] = np.log1p(
        features[
            "h180_orders"
        ]
    )

    # ------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------

    prediction_month = (
        features[
            "prediction_date"
        ]
        .dt
        .month
    )

    features[
        "prediction_month_sin"
    ] = np.sin(
        2
        *
        np.pi
        *
        prediction_month
        /
        12
    )

    features[
        "prediction_month_cos"
    ] = np.cos(
        2
        *
        np.pi
        *
        prediction_month
        /
        12
    )

    # ------------------------------------------------------------
    # Frozen model register
    # ------------------------------------------------------------

    missing_features = [
        feature
        for feature
        in FCV_MODEL_FEATURES
        if feature
        not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "FCV parity adapter failed to create frozen model "
            "features: "
            +
            ", ".join(
                missing_features
            )
        )

    result = (
        features[
            [
                "customer",
                *FCV_MODEL_FEATURES,
            ]
        ]
        .copy()
    )

    if result[
        "customer"
    ].isna().any():
        raise ValueError(
            "FCV model features contain missing customer identifiers."
        )

    if result[
        "customer"
    ].duplicated().any():
        raise ValueError(
            "FCV model features contain duplicate customers."
        )

    if list(
        result.columns
    ) != [
        "customer",
        *FCV_MODEL_FEATURES,
    ]:
        raise RuntimeError(
            "FCV model feature ordering does not match the frozen "
            "research specification."
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