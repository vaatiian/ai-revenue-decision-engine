"""
Production recommendation feature pipeline.

This module builds point-in-time customer-product features for
Next Best Product / recommendation scoring.

The pipeline uses only qualifying merchandise transactions available
on or before an explicit prediction date.

Feature families include:
- customer-product purchase frequency
- customer-product revenue
- customer-product units
- product recency
- customer product diversity
- product popularity
- product customer penetration
- customer-product affinity
- recency-weighted affinity
- candidate product generation

No future purchases are used.
No final recommendation ranking is produced here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.customer_snapshot import (
    build_qualifying_merchandise_history,
    normalise_decision_date,
)


DEFAULT_RECENCY_HALF_LIFE_DAYS = 90


# ================================================================
# CUSTOMER-PRODUCT HISTORY
# ================================================================

def build_customer_product_history(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build one historical row per customer-product pair.

    Outputs include:
    - purchase count
    - revenue
    - units
    - first purchase date
    - last purchase date
    - days since last purchase
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    history = build_qualifying_merchandise_history(
        transactions,
        score_date,
    )

    output_columns = [
        "customer",
        "product",
        "customer_product_orders",
        "customer_product_revenue",
        "customer_product_units",
        "first_product_purchase_date",
        "last_product_purchase_date",
        "days_since_product_purchase",
    ]

    if history.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    summary = (
        history
        .groupby(
            [
                "customer",
                "product",
            ],
            as_index=False,
        )
        .agg(
            customer_product_orders=(
                "invoice",
                "nunique",
            ),

            customer_product_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            customer_product_units=(
                "quantity",
                "sum",
            ),

            first_product_purchase_date=(
                "date",
                "min",
            ),

            last_product_purchase_date=(
                "date",
                "max",
            ),
        )
    )

    summary[
        "days_since_product_purchase"
    ] = (
        (
            score_date
            -
            summary[
                "last_product_purchase_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    if (
        summary[
            "last_product_purchase_date"
        ]
        >
        score_date
    ).any():
        raise RuntimeError(
            "Future purchase leakage detected."
        )

    return (
        summary
        .sort_values(
            [
                "customer",
                "product",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# CUSTOMER PRODUCT PROFILE
# ================================================================

def build_customer_product_profile(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build customer-level purchasing profile features.
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    history = build_qualifying_merchandise_history(
        transactions,
        score_date,
    )

    output_columns = [
        "customer",
        "customer_total_orders",
        "customer_total_revenue",
        "customer_total_units",
        "customer_unique_products",
        "customer_average_order_value",
        "customer_average_revenue_per_product",
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
            customer_total_orders=(
                "invoice",
                "nunique",
            ),

            customer_total_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            customer_total_units=(
                "quantity",
                "sum",
            ),

            customer_unique_products=(
                "product",
                "nunique",
            ),
        )
    )

    summary[
        "customer_average_order_value"
    ] = np.where(
        summary[
            "customer_total_orders"
        ]
        >
        0,

        summary[
            "customer_total_revenue"
        ]
        /
        summary[
            "customer_total_orders"
        ],

        0.0,
    )

    summary[
        "customer_average_revenue_per_product"
    ] = np.where(
        summary[
            "customer_unique_products"
        ]
        >
        0,

        summary[
            "customer_total_revenue"
        ]
        /
        summary[
            "customer_unique_products"
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
# PRODUCT POPULARITY
# ================================================================

def build_product_popularity(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build product-level historical popularity features.
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    history = build_qualifying_merchandise_history(
        transactions,
        score_date,
    )

    output_columns = [
        "product",
        "product_total_orders",
        "product_total_revenue",
        "product_total_units",
        "product_unique_customers",
        "product_average_revenue_per_customer",
    ]

    if history.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    summary = (
        history
        .groupby(
            "product",
            as_index=False,
        )
        .agg(
            product_total_orders=(
                "invoice",
                "nunique",
            ),

            product_total_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            product_total_units=(
                "quantity",
                "sum",
            ),

            product_unique_customers=(
                "customer",
                "nunique",
            ),
        )
    )

    summary[
        "product_average_revenue_per_customer"
    ] = np.where(
        summary[
            "product_unique_customers"
        ]
        >
        0,

        summary[
            "product_total_revenue"
        ]
        /
        summary[
            "product_unique_customers"
        ],

        0.0,
    )

    return (
        summary
        .sort_values(
            "product"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# RECENCY-WEIGHTED AFFINITY
# ================================================================

def add_recency_weighted_affinity(
    customer_product_history: pd.DataFrame,
    half_life_days: int = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """
    Add a recency-weighted customer-product affinity score.

    The decay factor is:

        0.5 ** (days_since_purchase / half_life_days)

    The raw affinity combines:
    - customer-product order count
    - customer-product revenue share proxy
    - recency

    This remains a feature, not a final recommendation score.
    """

    if half_life_days <= 0:
        raise ValueError(
            "half_life_days must be greater than zero."
        )

    result = customer_product_history.copy()

    if result.empty:
        result[
            "recency_decay"
        ] = pd.Series(
            dtype=float
        )

        result[
            "recency_weighted_affinity"
        ] = pd.Series(
            dtype=float
        )

        return result

    result[
        "recency_decay"
    ] = (
        0.5
        **
        (
            result[
                "days_since_product_purchase"
            ]
            /
            half_life_days
        )
    )

    result[
        "recency_weighted_affinity"
    ] = (
        result[
            "customer_product_orders"
        ]
        *
        result[
            "recency_decay"
        ]
    )

    return result


# ================================================================
# CUSTOMER-PRODUCT FEATURE TABLE
# ================================================================

def build_customer_product_features(
    transactions: pd.DataFrame,
    prediction_date,
    half_life_days: int = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """
    Build the full historical customer-product feature table.
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    customer_product = (
        build_customer_product_history(
            transactions,
            score_date,
        )
    )

    if customer_product.empty:
        return pd.DataFrame()

    customer_product = (
        add_recency_weighted_affinity(
            customer_product,
            half_life_days=half_life_days,
        )
    )

    customer_profile = (
        build_customer_product_profile(
            transactions,
            score_date,
        )
    )

    product_profile = build_product_popularity(
        transactions,
        score_date,
    )

    features = (
        customer_product
        .merge(
            customer_profile,
            on="customer",
            how="left",
        )
        .merge(
            product_profile,
            on="product",
            how="left",
        )
    )

    features[
        "customer_product_order_share"
    ] = np.where(
        features[
            "customer_total_orders"
        ]
        >
        0,

        features[
            "customer_product_orders"
        ]
        /
        features[
            "customer_total_orders"
        ],

        0.0,
    )

    features[
        "customer_product_revenue_share"
    ] = np.where(
        features[
            "customer_total_revenue"
        ]
        >
        0,

        features[
            "customer_product_revenue"
        ]
        /
        features[
            "customer_total_revenue"
        ],

        0.0,
    )

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

    if (
        features[
            "last_product_purchase_date"
        ]
        >
        score_date
    ).any():
        raise RuntimeError(
            "Future purchase leakage detected."
        )

    return (
        features
        .sort_values(
            [
                "customer",
                "product",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# CANDIDATE PRODUCT GENERATION
# ================================================================

def build_candidate_products(
    transactions: pd.DataFrame,
    prediction_date,
    top_n_products: int = 20,
    exclude_previously_purchased: bool = False,
) -> pd.DataFrame:
    """
    Generate recommendation candidates.

    Candidates are drawn from historically popular products available
    on or before prediction_date.

    By default, previously purchased products remain eligible because
    repeat purchase is a valid retail recommendation use case.

    Set exclude_previously_purchased=True for cross-sell-only candidate
    generation.
    """

    if top_n_products <= 0:
        raise ValueError(
            "top_n_products must be greater than zero."
        )

    score_date = normalise_decision_date(
        prediction_date
    )

    customer_profile = (
        build_customer_product_profile(
            transactions,
            score_date,
        )
    )

    product_profile = build_product_popularity(
        transactions,
        score_date,
    )

    if (
        customer_profile.empty
        or
        product_profile.empty
    ):
        return pd.DataFrame(
            columns=[
                "customer",
                "product",
                "product_popularity_rank",
                "prediction_date",
            ]
        )

    top_products = (
        product_profile
        .sort_values(
            [
                "product_total_orders",
                "product_total_revenue",
                "product",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(
            top_n_products
        )
        .copy()
    )

    top_products[
        "product_popularity_rank"
    ] = np.arange(
        1,
        len(top_products) + 1,
    )

    customers = (
        customer_profile[
            [
                "customer",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    customers[
        "_join_key"
    ] = 1

    top_products[
        "_join_key"
    ] = 1

    candidates = (
        customers
        .merge(
            top_products[
                [
                    "_join_key",
                    "product",
                    "product_popularity_rank",
                ]
            ],
            on="_join_key",
            how="inner",
        )
        .drop(
            columns="_join_key"
        )
    )

    if exclude_previously_purchased:
        purchased = (
            build_customer_product_history(
                transactions,
                score_date,
            )[
                [
                    "customer",
                    "product",
                ]
            ]
            .drop_duplicates()
        )

        purchased[
            "_already_purchased"
        ] = True

        candidates = (
            candidates
            .merge(
                purchased,
                on=[
                    "customer",
                    "product",
                ],
                how="left",
            )
        )

        candidates = (
            candidates.loc[
                candidates[
                    "_already_purchased"
                ]
                .isna()
            ]
            .drop(
                columns="_already_purchased"
            )
        )

    candidates[
        "prediction_date"
    ] = score_date

    if candidates.duplicated(
        subset=[
            "customer",
            "product",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate customer-product recommendation candidates detected."
        )

    return (
        candidates
        .sort_values(
            [
                "customer",
                "product_popularity_rank",
                "product",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# FINAL RECOMMENDATION FEATURE TABLE
# ================================================================

def build_recommendation_features(
    transactions: pd.DataFrame,
    prediction_date,
    top_n_products: int = 20,
    exclude_previously_purchased: bool = False,
    half_life_days: int = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """
    Build customer-product recommendation candidates enriched with
    historical affinity and product popularity features.

    Previously unseen customer-product pairs receive zero historical
    affinity features.

    The result is suitable for later scoring/ranking.
    """

    score_date = normalise_decision_date(
        prediction_date
    )

    candidates = build_candidate_products(
        transactions=transactions,
        prediction_date=score_date,
        top_n_products=top_n_products,
        exclude_previously_purchased=exclude_previously_purchased,
    )

    if candidates.empty:
        return pd.DataFrame()

    historical = build_customer_product_features(
        transactions=transactions,
        prediction_date=score_date,
        half_life_days=half_life_days,
    )

    product_profile = build_product_popularity(
        transactions,
        score_date,
    )

    customer_profile = build_customer_product_profile(
        transactions,
        score_date,
    )

    historical_columns = [
        "customer",
        "product",
        "customer_product_orders",
        "customer_product_revenue",
        "customer_product_units",
        "days_since_product_purchase",
        "recency_decay",
        "recency_weighted_affinity",
        "customer_product_order_share",
        "customer_product_revenue_share",
    ]

    available_historical_columns = [
        column
        for column in historical_columns
        if column in historical.columns
    ]

    features = (
        candidates
        .merge(
            historical[
                available_historical_columns
            ],
            on=[
                "customer",
                "product",
            ],
            how="left",
        )
        .merge(
            customer_profile,
            on="customer",
            how="left",
        )
        .merge(
            product_profile,
            on="product",
            how="left",
        )
    )

    zero_fill_columns = [
        "customer_product_orders",
        "customer_product_revenue",
        "customer_product_units",
        "recency_weighted_affinity",
        "customer_product_order_share",
        "customer_product_revenue_share",
    ]

    existing_zero_fill_columns = [
        column
        for column in zero_fill_columns
        if column in features.columns
    ]

    features[
        existing_zero_fill_columns
    ] = (
        features[
            existing_zero_fill_columns
        ]
        .fillna(0)
    )

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

    if features.duplicated(
        subset=[
            "customer",
            "product",
        ]
    ).any():
        raise RuntimeError(
            "Recommendation feature table contains duplicate customer-product rows."
        )

    return (
        features
        .sort_values(
            [
                "customer",
                "product_popularity_rank",
                "product",
            ]
        )
        .reset_index(
            drop=True
        )
    )