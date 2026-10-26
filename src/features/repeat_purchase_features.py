"""
Research-parity feature construction for the Notebook 02
90-day Repeat Purchase model.

The frozen model predicts whether an identified newly acquired customer
will make a second observable qualifying merchandise purchase within
90 days of the customer's first observable qualifying order.

Research rules reproduced here:

1. Observation boundary:
   2011-11-30 23:59:59

2. Qualifying purchase lines:
   - identified customer
   - merchandise
   - quantity > 0
   - unit_price > 0
   - not cancellation

3. Order grain:
   customer + invoice

4. First observable order:
   first order after sorting by
   customer, order_date, invoice

5. Left truncation:
   customers whose first observable order falls in the dataset's
   opening calendar month are excluded from historical model episodes.

6. Historical target eligibility:
   at least 90 complete follow-up days must exist.

7. Target:
   second observable order occurs 0-90 days after the first order,
   inclusive.

No future information is used in the predictor columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_MODEL_FEATURES,
)


# ================================================================
# RESEARCH CONSTANTS
# ================================================================

REPEAT_PURCHASE_MODELING_END_DATE = pd.Timestamp(
    "2011-11-30 23:59:59"
)

REPEAT_PURCHASE_FOLLOWUP_DAYS = 90


REPEAT_PURCHASE_REQUIRED_COLUMNS = (
    "date",
    "invoice",
    "customer",
    "product",
    "product_description",
    "region",
    "quantity",
    "unit_price",
    "customer_is_identified",
    "is_merchandise",
    "is_cancellation",
    "net_merchandise_revenue",
)


# ================================================================
# OUTPUT CONTRACTS
# ================================================================

REPEAT_PURCHASE_FIRST_ORDER_BASE_COLUMNS = (
    "customer",
    "first_invoice",
    "first_order_date",
    "first_order_value",
    "first_order_units",
    "first_order_product_count",
    "first_order_average_unit_price",
    "first_order_region",
)

REPEAT_PURCHASE_HISTORICAL_METADATA_COLUMNS = (
    "customer",
    "first_invoice",
    "first_order_date",
    "second_order_date",
    "days_to_second_purchase",
    "followup_days_available",
    "repeat_within_90d",
)


# ================================================================
# VALIDATION
# ================================================================

def validate_repeat_purchase_input(
    data: pd.DataFrame,
) -> None:
    """
    Validate the canonical fields required by Notebook 02.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Repeat Purchase input must be a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in REPEAT_PURCHASE_REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            "Required Repeat Purchase modelling columns are missing: "
            f"{missing_columns}"
        )


# ================================================================
# DATATYPE STANDARDISATION
# ================================================================

def standardize_repeat_purchase_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 02 datatype standardisation without mutating
    the caller's DataFrame.
    """

    validate_repeat_purchase_input(
        data
    )

    result = data.copy()

    result[
        "date"
    ] = pd.to_datetime(
        result[
            "date"
        ],
        errors="coerce",
    )

    if result[
        "date"
    ].isna().any():
        raise ValueError(
            "Repeat Purchase input contains invalid or missing dates."
        )

    for column in (
        "customer",
        "invoice",
        "product",
        "region",
    ):
        result[
            column
        ] = (
            result[
                column
            ]
            .astype(
                "string"
            )
        )

    return result


# ================================================================
# OBSERVATION BOUNDARY
# ================================================================

def apply_repeat_purchase_observation_boundary(
    data: pd.DataFrame,
    modeling_end_date: pd.Timestamp = (
        REPEAT_PURCHASE_MODELING_END_DATE
    ),
) -> pd.DataFrame:
    """
    Restrict observations to the frozen Notebook 02 modelling boundary.
    """

    standardized = (
        standardize_repeat_purchase_data(
            data
        )
    )

    boundary = pd.Timestamp(
        modeling_end_date
    )

    return (
        standardized.loc[
            standardized[
                "date"
            ]
            <=
            boundary
        ]
        .copy()
    )


# ================================================================
# QUALIFYING PURCHASE LINES
# ================================================================

def build_positive_customer_purchases(
    data: pd.DataFrame,
    modeling_end_date: pd.Timestamp = (
        REPEAT_PURCHASE_MODELING_END_DATE
    ),
) -> pd.DataFrame:
    """
    Reproduce Notebook 02 Step 24.6 exactly.

    A qualifying line must be:
    - identified customer;
    - merchandise;
    - quantity > 0;
    - unit_price > 0;
    - not a cancellation.

    There is intentionally no additional explicit filter on
    net_merchandise_revenue or product nullability because Notebook 02
    does not impose one.
    """

    modeling_data = (
        apply_repeat_purchase_observation_boundary(
            data=data,
            modeling_end_date=modeling_end_date,
        )
    )

    positive_customer_purchases = (
        modeling_data.loc[
            modeling_data[
                "customer_is_identified"
            ]
            &
            modeling_data[
                "is_merchandise"
            ]
            &
            modeling_data[
                "quantity"
            ].gt(
                0
            )
            &
            modeling_data[
                "unit_price"
            ].gt(
                0
            )
            &
            ~modeling_data[
                "is_cancellation"
            ]
        ]
        .copy()
    )

    return positive_customer_purchases


# ================================================================
# CUSTOMER ORDER HISTORY
# ================================================================

def build_repeat_purchase_customer_orders(
    data: pd.DataFrame,
    modeling_end_date: pd.Timestamp = (
        REPEAT_PURCHASE_MODELING_END_DATE
    ),
) -> pd.DataFrame:
    """
    Build Notebook 02 customer-order history at customer + invoice grain.
    """

    purchases = (
        build_positive_customer_purchases(
            data=data,
            modeling_end_date=modeling_end_date,
        )
    )

    if purchases.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                "invoice",
                "order_date",
                "order_revenue",
                "order_units",
                "distinct_products",
                "average_unit_price",
                "region",
                "order_number",
            ]
        )

    customer_orders = (
        purchases
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
            average_unit_price=(
                "unit_price",
                "mean",
            ),
            region=(
                "region",
                "first",
            ),
        )
    )

    customer_orders = (
        customer_orders
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

    customer_orders[
        "order_number"
    ] = (
        customer_orders
        .groupby(
            "customer"
        )
        .cumcount()
        +
        1
    )

    return customer_orders


# ================================================================
# FIRST + SECOND OBSERVABLE ORDERS
# ================================================================

def build_first_observable_orders(
    customer_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the first observable qualifying order per customer.
    """

    if customer_orders.empty:
        return pd.DataFrame(
            columns=[
                *REPEAT_PURCHASE_FIRST_ORDER_BASE_COLUMNS,
                "order_number",
            ]
        )

    first_orders = (
        customer_orders.loc[
            customer_orders[
                "order_number"
            ]
            ==
            1
        ]
        .copy()
    )

    first_orders = (
        first_orders
        .rename(
            columns={
                "invoice":
                    "first_invoice",
                "order_date":
                    "first_order_date",
                "order_revenue":
                    "first_order_value",
                "order_units":
                    "first_order_units",
                "distinct_products":
                    "first_order_product_count",
                "average_unit_price":
                    "first_order_average_unit_price",
                "region":
                    "first_order_region",
            }
        )
    )

    return first_orders


def attach_second_observable_order(
    first_orders: pd.DataFrame,
    customer_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the second observable qualifying order and elapsed days.
    """

    result = first_orders.copy()

    if result.empty:
        result[
            "second_order_date"
        ] = pd.Series(
            dtype="datetime64[ns]"
        )

        result[
            "days_to_second_purchase"
        ] = pd.Series(
            dtype="float64"
        )

        return result

    second_orders = (
        customer_orders.loc[
            customer_orders[
                "order_number"
            ]
            ==
            2,
            [
                "customer",
                "order_date",
            ],
        ]
        .rename(
            columns={
                "order_date":
                    "second_order_date",
            }
        )
    )

    result = (
        result
        .merge(
            second_orders,
            on="customer",
            how="left",
            validate="one_to_one",
        )
    )

    result[
        "days_to_second_purchase"
    ] = (
        result[
            "second_order_date"
        ]
        -
        result[
            "first_order_date"
        ]
    ).dt.days

    return result


# ================================================================
# LEFT-TRUNCATION RULE
# ================================================================

def remove_opening_month_acquisitions(
    first_orders: pd.DataFrame,
    modeling_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exclude customers whose first observable order occurs in the opening
    calendar month of the available modelling data.
    """

    if first_orders.empty:
        return first_orders.copy()

    if modeling_data.empty:
        return first_orders.iloc[
            0:0
        ].copy()

    dataset_start_month = (
        modeling_data[
            "date"
        ]
        .min()
        .to_period(
            "M"
        )
    )

    result = first_orders.copy()

    result[
        "first_order_month"
    ] = (
        result[
            "first_order_date"
        ]
        .dt.to_period(
            "M"
        )
    )

    result[
        "is_left_truncated"
    ] = (
        result[
            "first_order_month"
        ]
        ==
        dataset_start_month
    )

    result = (
        result.loc[
            ~result[
                "is_left_truncated"
            ]
        ]
        .copy()
    )

    return result


# ================================================================
# FOLLOW-UP ELIGIBILITY + TARGET
# ================================================================

def add_repeat_purchase_target(
    first_orders: pd.DataFrame,
    modeling_end_date: pd.Timestamp = (
        REPEAT_PURCHASE_MODELING_END_DATE
    ),
    require_complete_followup: bool = True,
) -> pd.DataFrame:
    """
    Add Notebook 02 historical 90-day follow-up eligibility and target.

    If require_complete_followup=True, historical episodes without a full
    90-day observation window are excluded before target creation.
    """

    result = first_orders.copy()

    boundary = pd.Timestamp(
        modeling_end_date
    )

    if result.empty:
        result[
            "followup_days_available"
        ] = pd.Series(
            dtype="int64"
        )

        result[
            "eligible_90d"
        ] = pd.Series(
            dtype="bool"
        )

        result[
            "repeat_within_90d"
        ] = pd.Series(
            dtype="int64"
        )

        return result

    result[
        "followup_days_available"
    ] = (
        boundary.normalize()
        -
        result[
            "first_order_date"
        ].dt.normalize()
    ).dt.days

    result[
        "eligible_90d"
    ] = (
        result[
            "followup_days_available"
        ]
        >=
        REPEAT_PURCHASE_FOLLOWUP_DAYS
    )

    if require_complete_followup:
        result = (
            result.loc[
                result[
                    "eligible_90d"
                ]
            ]
            .copy()
        )

    result[
        "repeat_within_90d"
    ] = (
        result[
            "days_to_second_purchase"
        ]
        .between(
            0,
            REPEAT_PURCHASE_FOLLOWUP_DAYS,
            inclusive="both",
        )
        .fillna(
            False
        )
        .astype(
            int
        )
    )

    return result


# ================================================================
# TEMPORAL FEATURES
# ================================================================

def add_first_order_temporal_features(
    first_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the frozen original-calendar features from Notebook 02.
    """

    result = first_orders.copy()

    result[
        "first_order_month_number"
    ] = (
        result[
            "first_order_date"
        ].dt.month
    )

    result[
        "first_order_weekday"
    ] = (
        result[
            "first_order_date"
        ].dt.dayofweek
    )

    result[
        "first_order_hour"
    ] = (
        result[
            "first_order_date"
        ].dt.hour
    )

    result[
        "first_order_is_weekend"
    ] = (
        result[
            "first_order_weekday"
        ]
        .isin(
            [
                5,
                6,
            ]
        )
        .astype(
            int
        )
    )

    return result


# ================================================================
# FIRST-ORDER LINE-LEVEL BASKET FEATURES
# ================================================================

def add_first_order_basket_features(
    first_orders: pd.DataFrame,
    positive_customer_purchases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 02 Steps 24.15-24.16.
    """

    result = first_orders.copy()

    if result.empty:
        for column in (
            "first_order_line_count",
            "first_order_max_unit_price",
            "first_order_min_unit_price",
            "first_order_price_std",
            "first_order_max_line_quantity",
            "first_order_mean_line_quantity",
            "units_per_product",
            "revenue_per_product",
            "revenue_per_unit",
        ):
            result[
                column
            ] = pd.Series(
                dtype="float64"
            )

        return result

    first_invoice_lookup = (
        result[
            [
                "customer",
                "first_invoice",
            ]
        ]
        .copy()
    )

    first_order_lines = (
        positive_customer_purchases
        .merge(
            first_invoice_lookup,
            left_on=[
                "customer",
                "invoice",
            ],
            right_on=[
                "customer",
                "first_invoice",
            ],
            how="inner",
            validate="many_to_one",
        )
    )

    first_order_extra_features = (
        first_order_lines
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            first_order_line_count=(
                "product",
                "size",
            ),
            first_order_max_unit_price=(
                "unit_price",
                "max",
            ),
            first_order_min_unit_price=(
                "unit_price",
                "min",
            ),
            first_order_price_std=(
                "unit_price",
                "std",
            ),
            first_order_max_line_quantity=(
                "quantity",
                "max",
            ),
            first_order_mean_line_quantity=(
                "quantity",
                "mean",
            ),
        )
    )

    result = (
        result
        .merge(
            first_order_extra_features,
            on="customer",
            how="left",
            validate="one_to_one",
        )
    )

    result[
        "units_per_product"
    ] = (
        result[
            "first_order_units"
        ]
        /
        result[
            "first_order_product_count"
        ]
    )

    result[
        "revenue_per_product"
    ] = (
        result[
            "first_order_value"
        ]
        /
        result[
            "first_order_product_count"
        ]
    )

    result[
        "revenue_per_unit"
    ] = (
        result[
            "first_order_value"
        ]
        /
        result[
            "first_order_units"
        ]
    )

    return result


# ================================================================
# HISTORICAL RESEARCH-PARITY DATASET
# ================================================================

def build_repeat_purchase_historical_dataset(
    data: pd.DataFrame,
    modeling_end_date: pd.Timestamp = (
        REPEAT_PURCHASE_MODELING_END_DATE
    ),
) -> pd.DataFrame:
    """
    Build Notebook 02's historical modelling population.

    Output contains:
    - customer metadata;
    - the exact 18 frozen predictors;
    - repeat_within_90d target;
    - diagnostic target metadata.

    This function is for research parity, retraining validation and
    end-to-end governance checks.
    """

    modeling_data = (
        apply_repeat_purchase_observation_boundary(
            data=data,
            modeling_end_date=modeling_end_date,
        )
    )

    positive_purchases = (
        build_positive_customer_purchases(
            data=data,
            modeling_end_date=modeling_end_date,
        )
    )

    customer_orders = (
        build_repeat_purchase_customer_orders(
            data=data,
            modeling_end_date=modeling_end_date,
        )
    )

    first_orders = (
        build_first_observable_orders(
            customer_orders
        )
    )

    first_orders = (
        attach_second_observable_order(
            first_orders=first_orders,
            customer_orders=customer_orders,
        )
    )

    first_orders = (
        remove_opening_month_acquisitions(
            first_orders=first_orders,
            modeling_data=modeling_data,
        )
    )

    first_orders = (
        add_repeat_purchase_target(
            first_orders=first_orders,
            modeling_end_date=modeling_end_date,
            require_complete_followup=True,
        )
    )

    first_orders = (
        add_first_order_temporal_features(
            first_orders
        )
    )

    first_orders = (
        add_first_order_basket_features(
            first_orders=first_orders,
            positive_customer_purchases=positive_purchases,
        )
    )

    output_columns = [
        "customer",
        "first_invoice",
        "first_order_date",
        "second_order_date",
        "days_to_second_purchase",
        "followup_days_available",
        *REPEAT_PURCHASE_MODEL_FEATURES,
        "repeat_within_90d",
    ]

    return (
        first_orders[
            output_columns
        ]
        .sort_values(
            [
                "first_order_date",
                "customer",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# PRODUCTION SCORING FEATURES
# ================================================================

def build_repeat_purchase_scoring_features(
    data: pd.DataFrame,
    modeling_end_date: pd.Timestamp = (
        REPEAT_PURCHASE_MODELING_END_DATE
    ),
    exclude_opening_month: bool = True,
) -> pd.DataFrame:
    """
    Build predictors available immediately after each customer's first
    observable qualifying order.

    Unlike the historical dataset builder:
    - no future 90-day eligibility requirement is imposed;
    - no repeat target is created;
    - no second-order information is returned.

    This is therefore suitable for production probability scoring.

    For historical parity, exclude_opening_month should remain True.
    A live production integration may explicitly set it False when the
    system has a trustworthy acquisition-history boundary.
    """

    modeling_data = (
        apply_repeat_purchase_observation_boundary(
            data=data,
            modeling_end_date=modeling_end_date,
        )
    )

    positive_purchases = (
        build_positive_customer_purchases(
            data=data,
            modeling_end_date=modeling_end_date,
        )
    )

    customer_orders = (
        build_repeat_purchase_customer_orders(
            data=data,
            modeling_end_date=modeling_end_date,
        )
    )

    first_orders = (
        build_first_observable_orders(
            customer_orders
        )
    )

    if exclude_opening_month:
        first_orders = (
            remove_opening_month_acquisitions(
                first_orders=first_orders,
                modeling_data=modeling_data,
            )
        )

    first_orders = (
        add_first_order_temporal_features(
            first_orders
        )
    )

    first_orders = (
        add_first_order_basket_features(
            first_orders=first_orders,
            positive_customer_purchases=positive_purchases,
        )
    )

    output_columns = [
        "customer",
        "first_invoice",
        "first_order_date",
        *REPEAT_PURCHASE_MODEL_FEATURES,
    ]

    return (
        first_orders[
            output_columns
        ]
        .sort_values(
            [
                "first_order_date",
                "customer",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# MODEL MATRIX
# ================================================================

def select_repeat_purchase_model_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the frozen 18 predictors in exact Notebook 02 order.
    """

    missing = [
        feature
        for feature in REPEAT_PURCHASE_MODEL_FEATURES
        if feature not in features.columns
    ]

    if missing:
        raise KeyError(
            "Repeat Purchase model features are missing: "
            f"{missing}"
        )

    return (
        features.loc[
            :,
            list(
                REPEAT_PURCHASE_MODEL_FEATURES
            ),
        ]
        .copy()
    )