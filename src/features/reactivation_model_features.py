"""
Frozen-model feature adapter for Reactivation scoring.

This module bridges the canonical production Reactivation feature
pipeline with the exact feature register frozen in the research model.

The Phase 3B Reactivation pipeline remains unchanged.

Responsibilities:
- construct research-parity features;
- preserve point-in-time safety;
- resolve overlapping canonical/parity feature names safely;
- create the exact frozen 29-feature schema;
- validate feature presence and order;
- expose a production-ready model matrix.

No model prediction occurs here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.customer_snapshot import (
    build_qualifying_merchandise_history,
    normalise_decision_date,
)
from src.features.reactivation_features import (
    build_reactivation_features,
)
from src.models.reactivation_spec import (
    REACTIVATION_MODEL_FEATURES,
)


DEFAULT_INACTIVITY_THRESHOLD_DAYS = 90


# ================================================================
# INTERNAL HELPERS
# ================================================================

def _coalesce_parity_column(
    dataframe: pd.DataFrame,
    column: str,
    parity_suffix: str,
) -> pd.DataFrame:
    """
    Prefer a parity-derived version of a feature when a merge created
    both canonical and parity copies.

    Example:

        historical_average_gap_days
        historical_average_gap_days_cadence

    becomes:

        historical_average_gap_days

    using the parity value where available.
    """

    result = dataframe.copy()

    parity_column = (
        f"{column}{parity_suffix}"
    )

    if parity_column not in result.columns:
        return result

    if column not in result.columns:
        result[column] = result[
            parity_column
        ]

    else:
        result[column] = (
            result[
                parity_column
            ]
            .combine_first(
                result[
                    column
                ]
            )
        )

    result = result.drop(
        columns=[
            parity_column,
        ]
    )

    return result


# ================================================================
# ORDER-LEVEL HISTORY
# ================================================================

def build_reactivation_order_history(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build one row per customer-invoice using only qualifying merchandise
    transactions available on or before prediction_date.

    This order-level history supports cadence and most-recent-order
    features required by the frozen Reactivation model.
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
        "invoice",
        "order_date",
        "order_value",
        "order_units",
        "order_product_count",
        "order_average_unit_price",
        "region",
    ]

    if history.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    history = history.copy()

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
                "max",
            ),

            order_value=(
                "net_merchandise_revenue",
                "sum",
            ),

            order_units=(
                "quantity",
                "sum",
            ),

            order_product_count=(
                "product",
                "nunique",
            ),

            order_average_unit_price=(
                "unit_price",
                "mean",
            ),

            region=(
                "region",
                "last",
            ),
        )
    )

    if (
        order_history[
            "order_date"
        ]
        >
        score_date
    ).any():
        raise RuntimeError(
            "Future order leakage detected."
        )

    if order_history.duplicated(
        subset=[
            "customer",
            "invoice",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate customer-invoice rows detected "
            "in Reactivation order history."
        )

    return (
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


# ================================================================
# CADENCE FEATURES
# ================================================================

def build_reactivation_cadence_features(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build purchase-cadence features required by the frozen model.

    Includes:
    - historical_average_gap_days
    - historical_max_gap_days
    - historical_min_gap_days
    - has_prior_purchase_gap
    """

    orders = (
        build_reactivation_order_history(
            transactions,
            prediction_date,
        )
    )

    output_columns = [
        "customer",
        "historical_average_gap_days",
        "historical_max_gap_days",
        "historical_min_gap_days",
        "has_prior_purchase_gap",
    ]

    if orders.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    orders = orders.copy()

    orders[
        "purchase_gap_days"
    ] = (
        orders
        .groupby(
            "customer"
        )[
            "order_date"
        ]
        .diff()
        .dt.total_seconds()
        /
        86400.0
    )

    summary = (
        orders
        .groupby(
            "customer",
            as_index=False,
        )
        .agg(
            historical_average_gap_days=(
                "purchase_gap_days",
                "mean",
            ),

            historical_max_gap_days=(
                "purchase_gap_days",
                "max",
            ),

            historical_min_gap_days=(
                "purchase_gap_days",
                "min",
            ),
        )
    )

    summary[
        "has_prior_purchase_gap"
    ] = (
        summary[
            "historical_average_gap_days"
        ]
        .notna()
        .astype(int)
    )

    return (
        summary[
            output_columns
        ]
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# LAST-ORDER FEATURES
# ================================================================

def build_reactivation_last_order_features(
    transactions: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    """
    Build the most-recent-order attributes required by the frozen model.
    """

    orders = (
        build_reactivation_order_history(
            transactions,
            prediction_date,
        )
    )

    output_columns = [
        "customer",
        "last_order_value",
        "last_order_units",
        "last_order_product_count",
        "last_order_average_unit_price",
        "last_order_region",
    ]

    if orders.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    last_orders = (
        orders
        .sort_values(
            [
                "customer",
                "order_date",
                "invoice",
            ]
        )
        .groupby(
            "customer",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    last_orders = (
        last_orders
        .rename(
            columns={
                "order_value":
                    "last_order_value",

                "order_units":
                    "last_order_units",

                "order_product_count":
                    "last_order_product_count",

                "order_average_unit_price":
                    "last_order_average_unit_price",

                "region":
                    "last_order_region",
            }
        )
    )

    return (
        last_orders[
            output_columns
        ]
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# FULL FROZEN-MODEL FEATURE ADAPTER
# ================================================================

def build_reactivation_model_features(
    transactions: pd.DataFrame,
    prediction_date,
    inactivity_threshold_days: int = DEFAULT_INACTIVITY_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """
    Build the exact frozen Reactivation model feature table.

    Only customers eligible for Reactivation scoring at prediction_date
    are included.

    The returned schema is:

        customer
        +
        exact 29 frozen model features

    No target or future information is included.
    """

    if inactivity_threshold_days <= 0:
        raise ValueError(
            "inactivity_threshold_days must be greater than zero."
        )

    score_date = (
        normalise_decision_date(
            prediction_date
        )
    )

    # ------------------------------------------------------------
    # Canonical Phase 3B Reactivation features
    # ------------------------------------------------------------

    base = (
        build_reactivation_features(
            transactions=transactions,
            prediction_date=score_date,
            inactivity_threshold_days=(
                inactivity_threshold_days
            ),
        )
    )

    if base.empty:
        return pd.DataFrame(
            columns=[
                "customer",
                *REACTIVATION_MODEL_FEATURES,
            ]
        )

    # ------------------------------------------------------------
    # Research-parity supplementary features
    # ------------------------------------------------------------

    cadence = (
        build_reactivation_cadence_features(
            transactions,
            score_date,
        )
    )

    last_order = (
        build_reactivation_last_order_features(
            transactions,
            score_date,
        )
    )

    # ------------------------------------------------------------
    # Merge cadence safely.
    #
    # Phase 3B may already contain historical_average_gap_days.
    # We therefore explicitly suffix parity-derived overlaps.
    # ------------------------------------------------------------

    features = (
        base
        .merge(
            cadence,
            on="customer",
            how="left",
            suffixes=(
                "",
                "_cadence",
            ),
            validate="one_to_one",
        )
    )

    for column in [
        "historical_average_gap_days",
        "historical_max_gap_days",
        "historical_min_gap_days",
        "has_prior_purchase_gap",
    ]:

        features = (
            _coalesce_parity_column(
                features,
                column,
                "_cadence",
            )
        )

    # ------------------------------------------------------------
    # Merge most-recent-order features safely.
    # ------------------------------------------------------------

    features = (
        features
        .merge(
            last_order,
            on="customer",
            how="left",
            suffixes=(
                "",
                "_last",
            ),
            validate="one_to_one",
        )
    )

    for column in [
        "last_order_value",
        "last_order_units",
        "last_order_product_count",
        "last_order_average_unit_price",
        "last_order_region",
    ]:

        features = (
            _coalesce_parity_column(
                features,
                column,
                "_last",
            )
        )

    # ------------------------------------------------------------
    # Historical average order value
    # ------------------------------------------------------------

    if (
        "historical_average_order_value"
        not in features.columns
    ):

        historical_orders = (
            pd.to_numeric(
                features[
                    "historical_order_count"
                ],
                errors="coerce",
            )
        )

        historical_revenue = (
            pd.to_numeric(
                features[
                    "historical_revenue"
                ],
                errors="coerce",
            )
        )

        features[
            "historical_average_order_value"
        ] = (
            historical_revenue
            /
            historical_orders.replace(
                0,
                np.nan,
            )
        )

    # ------------------------------------------------------------
    # Tenure at prediction
    # ------------------------------------------------------------

    if (
        "tenure_at_prediction_days"
        not in features.columns
    ):

        if (
            "customer_tenure_days"
            in features.columns
        ):

            features[
                "tenure_at_prediction_days"
            ] = (
                pd.to_numeric(
                    features[
                        "customer_tenure_days"
                    ],
                    errors="coerce",
                )
            )

        elif (
            "first_order_date"
            in features.columns
        ):

            features[
                "tenure_at_prediction_days"
            ] = (
                score_date.normalize()
                -
                pd.to_datetime(
                    features[
                        "first_order_date"
                    ]
                )
                .dt.normalize()
            ).dt.days

        else:

            raise ValueError(
                "Cannot construct tenure_at_prediction_days. "
                "Neither customer_tenure_days nor "
                "first_order_date is available."
            )

    # ------------------------------------------------------------
    # Relationship-depth rates
    # ------------------------------------------------------------

    effective_tenure = (
        pd.to_numeric(
            features[
                "tenure_at_prediction_days"
            ],
            errors="coerce",
        )
        .clip(
            lower=1
        )
    )

    historical_order_count = (
        pd.to_numeric(
            features[
                "historical_order_count"
            ],
            errors="coerce",
        )
    )

    historical_revenue = (
        pd.to_numeric(
            features[
                "historical_revenue"
            ],
            errors="coerce",
        )
    )

    historical_units = (
        pd.to_numeric(
            features[
                "historical_units"
            ],
            errors="coerce",
        )
    )

    features[
        "orders_per_30_tenure_days"
    ] = (
        historical_order_count
        /
        effective_tenure
        *
        30.0
    )

    features[
        "revenue_per_30_tenure_days"
    ] = (
        historical_revenue
        /
        effective_tenure
        *
        30.0
    )

    # ------------------------------------------------------------
    # Inactivity relative to historical cadence
    # ------------------------------------------------------------

    historical_average_gap = (
        pd.to_numeric(
            features[
                "historical_average_gap_days"
            ],
            errors="coerce",
        )
    )

    features[
        "inactivity_to_average_gap_ratio"
    ] = (
        float(
            inactivity_threshold_days
        )
        /
        historical_average_gap.replace(
            0,
            np.nan,
        )
    )

    # ------------------------------------------------------------
    # Last-order value relative to historical AOV
    # ------------------------------------------------------------

    historical_aov = (
        pd.to_numeric(
            features[
                "historical_average_order_value"
            ],
            errors="coerce",
        )
    )

    last_order_value = (
        pd.to_numeric(
            features[
                "last_order_value"
            ],
            errors="coerce",
        )
    )

    features[
        "last_to_historical_aov_ratio"
    ] = (
        last_order_value
        /
        historical_aov.replace(
            0,
            np.nan,
        )
    )

    # ------------------------------------------------------------
    # History availability indicators
    # ------------------------------------------------------------

    if (
        "has_prior_purchase_gap"
        not in features.columns
    ):

        features[
            "has_prior_purchase_gap"
        ] = (
            historical_average_gap
            .notna()
            .astype(int)
        )

    prior_inactivity_count = (
        pd.to_numeric(
            features[
                "prior_inactivity_episode_count"
            ],
            errors="coerce",
        )
        .fillna(0)
    )

    prior_reactivation_count = (
        pd.to_numeric(
            features[
                "prior_reactivation_count"
            ],
            errors="coerce",
        )
        .fillna(0)
    )

    features[
        "has_prior_inactivity_episode"
    ] = (
        prior_inactivity_count
        >
        0
    ).astype(int)

    features[
        "has_prior_successful_reactivation"
    ] = (
        prior_reactivation_count
        >
        0
    ).astype(int)

    # ------------------------------------------------------------
    # Non-linear depth alternatives
    # ------------------------------------------------------------

    features[
        "log1p_historical_order_count"
    ] = np.log1p(
        historical_order_count
    )

    features[
        "log1p_historical_revenue"
    ] = np.log1p(
        historical_revenue
    )

    features[
        "log1p_historical_units"
    ] = np.log1p(
        historical_units
    )

    # ------------------------------------------------------------
    # Prediction-date calendar features
    # ------------------------------------------------------------

    features[
        "prediction_month"
    ] = int(
        score_date.month
    )

    features[
        "prediction_weekday"
    ] = int(
        score_date.dayofweek
    )

    features[
        "prediction_is_weekend"
    ] = int(
        score_date.dayofweek
        >=
        5
    )

    # ------------------------------------------------------------
    # Frozen schema validation
    # ------------------------------------------------------------

    missing_features = [
        feature
        for feature
        in REACTIVATION_MODEL_FEATURES
        if feature not in features.columns
    ]

    if missing_features:

        raise ValueError(
            "Reactivation feature parity failed. "
            "Missing frozen-model features: "
            +
            ", ".join(
                missing_features
            )
        )

    # ------------------------------------------------------------
    # Entity safety
    # ------------------------------------------------------------

    if (
        features[
            "customer"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Reactivation model feature table "
            "contains duplicate customers."
        )

    # ------------------------------------------------------------
    # Temporal safety
    # ------------------------------------------------------------

    if (
        "last_order_date"
        in features.columns
    ):

        last_order_dates = (
            pd.to_datetime(
                features[
                    "last_order_date"
                ],
                errors="coerce",
            )
        )

        if (
            last_order_dates
            >
            score_date
        ).any():

            raise RuntimeError(
                "Future transaction leakage detected "
                "in Reactivation model features."
            )

    # ------------------------------------------------------------
    # Exact final frozen schema
    # ------------------------------------------------------------

    final_columns = [
        "customer",
        *REACTIVATION_MODEL_FEATURES,
    ]

    result = (
        features[
            final_columns
        ]
        .copy()
    )

    if list(
        result.columns[
            1:
        ]
    ) != list(
        REACTIVATION_MODEL_FEATURES
    ):

        raise RuntimeError(
            "Frozen Reactivation feature order changed."
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