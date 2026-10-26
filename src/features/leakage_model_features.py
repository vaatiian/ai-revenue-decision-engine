"""
Research-compatible production feature adapter for the frozen
Revenue Leakage model.

The implementation mirrors Notebook 04 feature engineering.

Research prediction design
--------------------------
Prediction unit:
    One active customer at one month-end prediction point.

Current baseline:
    (prediction_date - 90 days, prediction_date]

Previous window:
    (prediction_date - 180 days, prediction_date - 90 days]

Recent window:
    (prediction_date - 30 days, prediction_date]

Outcome definition used only for historical prior-leakage features:
    future 90-day revenue <= 50% of current 90-day revenue.

Important
---------
The production feature table never uses the current prediction
date's future outcome.

Prior leakage history includes only historical prediction episodes
whose outcome windows were fully known by the current prediction date.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.models.leakage_spec import (
    LEAKAGE_BASELINE_WINDOW_DAYS,
    LEAKAGE_MODEL_FEATURES,
    LEAKAGE_OUTCOME_WINDOW_DAYS,
    LEAKAGE_REVENUE_RATIO_THRESHOLD,
)


# ================================================================
# REQUIRED SOURCE FIELDS
# ================================================================

REQUIRED_TRANSACTION_COLUMNS = {
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
# DATE HELPERS
# ================================================================

def normalise_prediction_date(
    prediction_date,
) -> pd.Timestamp:
    """
    Convert a prediction date into a timezone-naive calendar day.
    """

    value = pd.Timestamp(
        prediction_date
    )

    if value.tzinfo is not None:
        value = value.tz_convert(
            None
        )

    return value.normalize()


def validate_month_end(
    prediction_date: pd.Timestamp,
) -> None:
    """
    Revenue Leakage research uses month-end prediction points.
    """

    prediction_date = normalise_prediction_date(
        prediction_date
    )

    if not prediction_date.is_month_end:
        raise ValueError(
            "Revenue Leakage scoring requires a month-end "
            "prediction date."
        )


# ================================================================
# SOURCE VALIDATION
# ================================================================

def validate_transaction_columns(
    transactions: pd.DataFrame,
) -> None:
    """
    Fail closed if required source fields are unavailable.
    """

    missing = (
        REQUIRED_TRANSACTION_COLUMNS
        -
        set(
            transactions.columns
        )
    )

    if missing:
        raise ValueError(
            "Revenue Leakage feature engineering is missing "
            "required transaction columns: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )


# ================================================================
# QUALIFYING MERCHANDISE PURCHASES
# ================================================================

def build_positive_merchandise_purchases(
    transactions: pd.DataFrame,
    cutoff_date,
) -> pd.DataFrame:
    """
    Build positive identified-customer merchandise history available
    on or before cutoff_date.
    """

    validate_transaction_columns(
        transactions
    )

    cutoff_date = normalise_prediction_date(
        cutoff_date
    )

    data = transactions.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["purchase_day"] = (
        data["date"]
        .dt
        .normalize()
    )

    mask = (
        data["customer"].notna()
        &
        (
            data["purchase_day"]
            <=
            cutoff_date
        )
        &
        (
            pd.to_numeric(
                data["quantity"],
                errors="coerce",
            )
            >
            0
        )
        &
        (
            pd.to_numeric(
                data["net_merchandise_revenue"],
                errors="coerce",
            )
            >
            0
        )
    )

    if "is_cancellation" in data.columns:
        mask &= ~(
            data["is_cancellation"]
            .fillna(False)
            .astype(bool)
        )

    if "is_return" in data.columns:
        mask &= ~(
            data["is_return"]
            .fillna(False)
            .astype(bool)
        )

    qualified = (
        data.loc[
            mask
        ]
        .copy()
    )

    qualified["quantity"] = pd.to_numeric(
        qualified["quantity"],
        errors="raise",
    )

    qualified["unit_price"] = pd.to_numeric(
        qualified["unit_price"],
        errors="raise",
    )

    qualified[
        "net_merchandise_revenue"
    ] = pd.to_numeric(
        qualified[
            "net_merchandise_revenue"
        ],
        errors="raise",
    )

    return qualified


# ================================================================
# CUSTOMER-DAY TABLE
# ================================================================

def build_customer_daily_history(
    line_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate qualifying merchandise transactions to customer-day.
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
# WINDOW SUMMARY
# ================================================================

def summarise_customer_window(
    daily_data: pd.DataFrame,
    customers: Iterable,
    start_date,
    end_date,
    prefix: str,
) -> pd.DataFrame:
    """
    Reproduce Notebook 04 section 40.5.

    Window:
        date > start_date
        date <= end_date
    """

    start_date = normalise_prediction_date(
        start_date
    )

    end_date = normalise_prediction_date(
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
                daily_data["date"]
                >
                start_date
            )
            &
            (
                daily_data["date"]
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


# ================================================================
# PRODUCT-BREADTH SUMMARY
# ================================================================

def summarise_product_window(
    line_data: pd.DataFrame,
    customers: Iterable,
    start_date,
    end_date,
    prefix: str,
) -> pd.DataFrame:
    """
    Reproduce Notebook 04 section 40.6.
    """

    start_date = normalise_prediction_date(
        start_date
    )

    end_date = normalise_prediction_date(
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


# ================================================================
# PURCHASE-GAP SUMMARY
# ================================================================

def summarise_purchase_gaps(
    daily_data: pd.DataFrame,
    customers: Iterable,
    start_date,
    end_date,
    prefix: str,
) -> pd.DataFrame:
    """
    Reproduce Notebook 04 section 40.7.
    """

    start_date = normalise_prediction_date(
        start_date
    )

    end_date = normalise_prediction_date(
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
                daily_data["date"]
                >
                start_date
            )
            &
            (
                daily_data["date"]
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
        )["date"]
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
# LAST REGION
# ================================================================

def build_last_region(
    line_data: pd.DataFrame,
    customers: Iterable,
    prediction_date,
) -> pd.DataFrame:
    """
    Return region from the customer's latest qualifying purchase.
    """

    prediction_date = normalise_prediction_date(
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
# SINGLE PREDICTION-POINT BASE FEATURES
# ================================================================

def build_leakage_base_features(
    line_data: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build leakage features at one prediction date,
    excluding prior-leakage-history variables.
    """

    prediction_date = normalise_prediction_date(
        prediction_date
    )

    baseline_start = (
        prediction_date
        -
        pd.Timedelta(
            days=LEAKAGE_BASELINE_WINDOW_DAYS
        )
    )

    prior90_start = (
        prediction_date
        -
        pd.Timedelta(
            days=(
                LEAKAGE_BASELINE_WINDOW_DAYS
                *
                2
            )
        )
    )

    recent30_start = (
        prediction_date
        -
        pd.Timedelta(
            days=30
        )
    )

    daily_data = build_customer_daily_history(
        line_data
    )

    current_window = (
        daily_data.loc[
            (
                daily_data["date"]
                >
                baseline_start
            )
            &
            (
                daily_data["date"]
                <=
                prediction_date
            )
        ]
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            baseline_revenue=(
                "daily_revenue",
                "sum",
            )
        )
    )

    current_window = (
        current_window.loc[
            current_window[
                "baseline_revenue"
            ]
            >
            0
        ]
        .copy()
    )

    customers = (
        current_window[
            "customer"
        ]
        .unique()
    )

    if len(
        customers
    ) == 0:
        return pd.DataFrame()

    b90 = summarise_customer_window(
        daily_data=daily_data,
        customers=customers,
        start_date=baseline_start,
        end_date=prediction_date,
        prefix="b90",
    )

    p90 = summarise_customer_window(
        daily_data=daily_data,
        customers=customers,
        start_date=prior90_start,
        end_date=baseline_start,
        prefix="p90",
    )

    r30 = summarise_customer_window(
        daily_data=daily_data,
        customers=customers,
        start_date=recent30_start,
        end_date=prediction_date,
        prefix="r30",
    )

    b90_products = summarise_product_window(
        line_data=line_data,
        customers=customers,
        start_date=baseline_start,
        end_date=prediction_date,
        prefix="b90",
    )

    p90_products = summarise_product_window(
        line_data=line_data,
        customers=customers,
        start_date=prior90_start,
        end_date=baseline_start,
        prefix="p90",
    )

    b90_gaps = summarise_purchase_gaps(
        daily_data=daily_data,
        customers=customers,
        start_date=baseline_start,
        end_date=prediction_date,
        prefix="b90",
    )

    last_region = build_last_region(
        line_data=line_data,
        customers=customers,
        prediction_date=prediction_date,
    )

    features = (
        current_window
        .assign(
            prediction_date=prediction_date
        )
        .merge(
            b90,
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
            b90_products,
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
            b90_gaps,
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

    zero_fill_features = [
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

    if not np.allclose(
        features[
            "baseline_revenue"
        ],
        features[
            "b90_revenue"
        ],
    ):
        raise RuntimeError(
            "Revenue Leakage baseline reconstruction failed."
        )

    features[
        "b90_average_order_value"
    ] = (
        features[
            "b90_revenue"
        ]
        /
        features[
            "b90_orders"
        ]
    )

    features[
        "b90_revenue_per_active_day"
    ] = (
        features[
            "b90_revenue"
        ]
        /
        features[
            "b90_active_days"
        ]
    )

    features[
        "b90_units_per_order"
    ] = (
        features[
            "b90_units"
        ]
        /
        features[
            "b90_orders"
        ]
    )

    features[
        "b90_products_per_order"
    ] = (
        features[
            "b90_unique_products"
        ]
        /
        features[
            "b90_orders"
        ]
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

    features[
        "revenue_change_vs_previous_90d"
    ] = (
        features[
            "b90_revenue"
        ]
        -
        features[
            "p90_revenue"
        ]
    )

    features[
        "revenue_ratio_vs_previous_90d"
    ] = np.where(
        features[
            "p90_revenue"
        ]
        >
        0,
        (
            features[
                "b90_revenue"
            ]
            /
            features[
                "p90_revenue"
            ]
        ),
        np.nan,
    )

    features[
        "log_revenue_ratio_vs_previous_90d"
    ] = (
        np.log1p(
            features[
                "b90_revenue"
            ]
        )
        -
        np.log1p(
            features[
                "p90_revenue"
            ]
        )
    )

    features[
        "recent30_revenue_share"
    ] = (
        features[
            "r30_revenue"
        ]
        /
        features[
            "b90_revenue"
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
            "b90_orders"
        ]
    )

    earlier60_revenue = (
        features[
            "b90_revenue"
        ]
        -
        features[
            "r30_revenue"
        ]
    )

    features[
        "recent30_vs_earlier60_revenue_rate"
    ] = np.where(
        earlier60_revenue
        >
        0,
        (
            (
                features[
                    "r30_revenue"
                ]
                /
                30
            )
            /
            (
                earlier60_revenue
                /
                60
            )
        ),
        np.nan,
    )

    features[
        "days_since_last_purchase"
    ] = (
        features[
            "prediction_date"
        ]
        -
        features[
            "b90_last_purchase_day"
        ]
    ).dt.days

    features[
        "b90_orders_per_30_days"
    ] = (
        features[
            "b90_orders"
        ]
        /
        3
    )

    features[
        "b90_active_day_rate"
    ] = (
        features[
            "b90_active_days"
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

    features[
        "product_breadth_change"
    ] = (
        features[
            "b90_unique_products"
        ]
        -
        features[
            "p90_unique_products"
        ]
    )

    features[
        "log1p_b90_revenue"
    ] = np.log1p(
        features[
            "b90_revenue"
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
        "log1p_b90_orders"
    ] = np.log1p(
        features[
            "b90_orders"
        ]
    )

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

    return features


# ================================================================
# HISTORICAL MONTH-END PREDICTION POINTS
# ================================================================

def historical_month_end_dates(
    source_start,
    current_prediction_date,
) -> list[pd.Timestamp]:
    """
    Generate historical month-end leakage prediction points that:
    - have at least 180 days of source history;
    - occur before the current prediction point;
    - have a fully matured 90-day outcome by the current point.

    Uses the explicit MonthEnd offset object rather than string
    aliases such as "M" or "ME" so the code remains compatible
    across supported pandas versions.
    """

    source_start = normalise_prediction_date(
        source_start
    )

    current_prediction_date = normalise_prediction_date(
        current_prediction_date
    )

    earliest_eligible_day = (
        source_start
        +
        pd.Timedelta(
            days=(
                LEAKAGE_BASELINE_WINDOW_DAYS
                *
                2
            )
        )
    )

    latest_matured_prediction = (
        current_prediction_date
        -
        pd.Timedelta(
            days=LEAKAGE_OUTCOME_WINDOW_DAYS
        )
    )

    if (
        latest_matured_prediction
        <
        earliest_eligible_day
    ):
        return []

    first_month_end = (
        earliest_eligible_day
        +
        pd.offsets.MonthEnd(0)
    )

    if latest_matured_prediction.is_month_end:
        last_month_end = (
            latest_matured_prediction
        )
    else:
        last_month_end = (
            latest_matured_prediction
            +
            pd.offsets.MonthEnd(-1)
        )

    if (
        first_month_end
        >
        last_month_end
    ):
        return []

    prediction_dates = [
        pd.Timestamp(
            value
        ).normalize()
        for value
        in pd.date_range(
            start=first_month_end,
            end=last_month_end,
            freq=pd.offsets.MonthEnd(),
        )
    ]

    for prediction_date in prediction_dates:
        if (
            prediction_date
            +
            pd.Timedelta(
                days=LEAKAGE_OUTCOME_WINDOW_DAYS
            )
            >
            current_prediction_date
        ):
            raise RuntimeError(
                "Historical Revenue Leakage prediction date "
                "does not have a fully matured outcome window."
            )

    return prediction_dates


# ================================================================
# HISTORICAL LEAKAGE EPISODES
# ================================================================

def build_historical_leakage_episodes(
    line_data: pd.DataFrame,
    current_prediction_date,
) -> pd.DataFrame:
    """
    Reconstruct historical fully matured leakage episodes.
    """

    current_prediction_date = normalise_prediction_date(
        current_prediction_date
    )

    if line_data.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "prediction_date",
                "outcome_end",
                "revenue_leakage_event",
            ]
        )

    source_start = (
        line_data[
            "purchase_day"
        ]
        .min()
    )

    prediction_dates = historical_month_end_dates(
        source_start=source_start,
        current_prediction_date=(
            current_prediction_date
        ),
    )

    daily_data = build_customer_daily_history(
        line_data
    )

    episode_frames = []

    for prediction_date in prediction_dates:

        baseline_start = (
            prediction_date
            -
            pd.Timedelta(
                days=LEAKAGE_BASELINE_WINDOW_DAYS
            )
        )

        outcome_end = (
            prediction_date
            +
            pd.Timedelta(
                days=LEAKAGE_OUTCOME_WINDOW_DAYS
            )
        )

        if (
            outcome_end
            >
            current_prediction_date
        ):
            raise RuntimeError(
                "Historical leakage episode attempted to use "
                "an outcome that had not matured by the current "
                "prediction date."
            )

        baseline = (
            daily_data.loc[
                (
                    daily_data["date"]
                    >
                    baseline_start
                )
                &
                (
                    daily_data["date"]
                    <=
                    prediction_date
                )
            ]
            .groupby(
                "customer",
                as_index=False,
            )
            .agg(
                baseline_revenue=(
                    "daily_revenue",
                    "sum",
                )
            )
        )

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
            continue

        customers = (
            baseline[
                "customer"
            ]
            .unique()
        )

        future = (
            daily_data.loc[
                daily_data[
                    "customer"
                ].isin(
                    customers
                )
                &
                (
                    daily_data["date"]
                    >
                    prediction_date
                )
                &
                (
                    daily_data["date"]
                    <=
                    outcome_end
                )
            ]
            .groupby(
                "customer",
                as_index=False,
            )
            .agg(
                future_revenue=(
                    "daily_revenue",
                    "sum",
                )
            )
        )

        episode = (
            baseline
            .merge(
                future,
                on="customer",
                how="left",
                validate="one_to_one",
            )
        )

        episode[
            "future_revenue"
        ] = (
            episode[
                "future_revenue"
            ]
            .fillna(0)
        )

        episode[
            "prediction_date"
        ] = prediction_date

        episode[
            "outcome_end"
        ] = outcome_end

        episode[
            "revenue_leakage_event"
        ] = (
            episode[
                "future_revenue"
            ]
            <=
            (
                LEAKAGE_REVENUE_RATIO_THRESHOLD
                *
                episode[
                    "baseline_revenue"
                ]
            )
        ).astype(int)

        episode_frames.append(
            episode[
                [
                    "customer",
                    "prediction_date",
                    "outcome_end",
                    "revenue_leakage_event",
                ]
            ]
        )

    if not episode_frames:
        return pd.DataFrame(
            columns=[
                "customer",
                "prediction_date",
                "outcome_end",
                "revenue_leakage_event",
            ]
        )

    return (
        pd.concat(
            episode_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "customer",
                "prediction_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# SAFE PRIOR LEAKAGE HISTORY
# ================================================================

def build_prior_leakage_history(
    customers: Iterable,
    historical_episodes: pd.DataFrame,
    current_prediction_date,
) -> pd.DataFrame:
    """
    Reproduce Notebook 04 safe prior leakage history.

    Only episodes satisfying:
    - prediction_date < current prediction date
    - outcome_end <= current prediction date
    are allowed.
    """

    current_prediction_date = normalise_prediction_date(
        current_prediction_date
    )

    customer_frame = pd.DataFrame(
        {
            "customer":
                list(
                    customers
                ),
        }
    )

    customer_frame[
        "prior_completed_leakage_episodes"
    ] = 0

    customer_frame[
        "prior_leakage_events"
    ] = 0

    customer_frame[
        "prior_leakage_rate"
    ] = np.nan

    if historical_episodes.empty:
        customer_frame[
            "has_prior_completed_leakage_episode"
        ] = 0

        return customer_frame

    known = (
        historical_episodes.loc[
            (
                historical_episodes[
                    "prediction_date"
                ]
                <
                current_prediction_date
            )
            &
            (
                historical_episodes[
                    "outcome_end"
                ]
                <=
                current_prediction_date
            )
        ]
        .copy()
    )

    if not known.empty:

        summary = (
            known
            .groupby(
                "customer",
                as_index=False,
            )
            .agg(
                prior_completed_leakage_episodes=(
                    "revenue_leakage_event",
                    "size",
                ),
                prior_leakage_events=(
                    "revenue_leakage_event",
                    "sum",
                ),
            )
        )

        summary[
            "prior_leakage_rate"
        ] = (
            summary[
                "prior_leakage_events"
            ]
            /
            summary[
                "prior_completed_leakage_episodes"
            ]
        )

        customer_frame = (
            customer_frame[
                [
                    "customer",
                ]
            ]
            .merge(
                summary,
                on="customer",
                how="left",
                validate="one_to_one",
            )
        )

        customer_frame[
            "prior_completed_leakage_episodes"
        ] = (
            customer_frame[
                "prior_completed_leakage_episodes"
            ]
            .fillna(0)
            .astype(int)
        )

        customer_frame[
            "prior_leakage_events"
        ] = (
            customer_frame[
                "prior_leakage_events"
            ]
            .fillna(0)
            .astype(int)
        )

    customer_frame[
        "has_prior_completed_leakage_episode"
    ] = (
        customer_frame[
            "prior_completed_leakage_episodes"
        ]
        >
        0
    ).astype(int)

    return customer_frame


# ================================================================
# FINAL FROZEN MODEL FEATURE TABLE
# ================================================================

def build_leakage_model_features(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build the exact frozen 45-feature Revenue Leakage model matrix.
    """

    prediction_date = normalise_prediction_date(
        prediction_date
    )

    validate_month_end(
        prediction_date
    )

    validate_transaction_columns(
        transactions
    )

    source_dates = pd.to_datetime(
        transactions[
            "date"
        ],
        errors="raise",
    ).dt.normalize()

    source_start = source_dates.min()

    if (
        prediction_date
        -
        source_start
        <
        pd.Timedelta(
            days=(
                LEAKAGE_BASELINE_WINDOW_DAYS
                *
                2
            )
        )
    ):
        raise ValueError(
            "Revenue Leakage scoring requires at least "
            "180 days of source history."
        )

    current_line_data = (
        build_positive_merchandise_purchases(
            transactions=transactions,
            cutoff_date=prediction_date,
        )
    )

    base_features = build_leakage_base_features(
        line_data=current_line_data,
        prediction_date=prediction_date,
    )

    if base_features.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                *LEAKAGE_MODEL_FEATURES,
            ]
        )

    historical_line_data = (
        build_positive_merchandise_purchases(
            transactions=transactions,
            cutoff_date=prediction_date,
        )
    )

    historical_episodes = (
        build_historical_leakage_episodes(
            line_data=historical_line_data,
            current_prediction_date=prediction_date,
        )
    )

    prior_history = (
        build_prior_leakage_history(
            customers=base_features[
                "customer"
            ],
            historical_episodes=(
                historical_episodes
            ),
            current_prediction_date=(
                prediction_date
            ),
        )
    )

    features = (
        base_features
        .merge(
            prior_history,
            on="customer",
            how="left",
            validate="one_to_one",
        )
    )

    missing_features = [
        feature
        for feature
        in LEAKAGE_MODEL_FEATURES
        if feature
        not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Revenue Leakage parity adapter failed to create "
            "frozen model features: "
            +
            ", ".join(
                missing_features
            )
        )

    result = (
        features[
            [
                "customer",
                *LEAKAGE_MODEL_FEATURES,
            ]
        ]
        .copy()
    )

    if result[
        "customer"
    ].isna().any():
        raise ValueError(
            "Revenue Leakage model features contain "
            "missing customer identifiers."
        )

    if result[
        "customer"
    ].duplicated().any():
        raise ValueError(
            "Revenue Leakage model features contain "
            "duplicate customers."
        )

    if list(
        result.columns
    ) != [
        "customer",
        *LEAKAGE_MODEL_FEATURES,
    ]:
        raise RuntimeError(
            "Revenue Leakage feature ordering does not match "
            "the frozen model specification."
        )

    return result.reset_index(
        drop=True
    )