"""
Frozen production specification for the Revenue Leakage model.

This module reproduces the exact model architecture selected and frozen
in Notebook 04 — ML Revenue Leakage Risk.

Research specification
----------------------
Prediction problem:
    Future customer revenue leakage.

Prediction unit:
    One active customer at one month-end prediction point.

Baseline window:
    Previous 90 days.

Outcome window:
    Following 90 days.

Primary target:
    Future revenue <= 50% of prior 90-day revenue.

Final model:
    Full Gradient Boosting.

Primary selection metric:
    PR-AUC.

Validation:
    Purged chronological validation.

The specification in this module must not be modified independently of
the governed research model.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.base import ModelMetadata


# ================================================================
# MODEL METADATA
# ================================================================

LEAKAGE_MODEL_METADATA = ModelMetadata(
    model_name="Full Gradient Boosting",
    model_version="1.0.0",
    model_type="classification",
    target_name="revenue_leakage_event",
)


# ================================================================
# FROZEN NUMERIC FEATURE REGISTER
# ================================================================

LEAKAGE_NUMERIC_FEATURES = (
    # ------------------------------------------------------------
    # Current 90-day economics
    # ------------------------------------------------------------
    "b90_revenue",
    "b90_orders",
    "b90_active_days",
    "b90_units",
    "b90_unique_products",
    "b90_average_unit_price",
    "b90_maximum_unit_price",
    "b90_average_order_value",
    "b90_revenue_per_active_day",
    "b90_units_per_order",
    "b90_products_per_order",

    # ------------------------------------------------------------
    # Previous 90-day behaviour
    # ------------------------------------------------------------
    "p90_revenue",
    "p90_orders",
    "p90_active_days",
    "p90_units",
    "p90_unique_products",
    "p90_average_order_value",

    # ------------------------------------------------------------
    # Recent 30 days
    # ------------------------------------------------------------
    "r30_revenue",
    "r30_orders",
    "r30_active_days",
    "r30_units",

    # ------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------
    "revenue_change_vs_previous_90d",
    "revenue_ratio_vs_previous_90d",
    "log_revenue_ratio_vs_previous_90d",
    "recent30_revenue_share",
    "recent30_order_share",
    "recent30_vs_earlier60_revenue_rate",
    "product_breadth_change",

    # ------------------------------------------------------------
    # Recency and cadence
    # ------------------------------------------------------------
    "days_since_last_purchase",
    "b90_average_gap_days",
    "b90_maximum_gap_days",
    "b90_minimum_gap_days",
    "b90_orders_per_30_days",
    "b90_active_day_rate",
    "p90_active_day_rate",

    # ------------------------------------------------------------
    # Prior leakage history
    # ------------------------------------------------------------
    "prior_completed_leakage_episodes",
    "prior_leakage_events",
    "prior_leakage_rate",
    "has_prior_completed_leakage_episode",

    # ------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------
    "log1p_b90_revenue",
    "log1p_p90_revenue",
    "log1p_b90_orders",

    # ------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------
    "prediction_month_sin",
    "prediction_month_cos",
)


# ================================================================
# FROZEN CATEGORICAL FEATURE REGISTER
# ================================================================

LEAKAGE_CATEGORICAL_FEATURES = (
    "last_region",
)


# ================================================================
# COMPLETE FROZEN FEATURE REGISTER
# ================================================================

LEAKAGE_MODEL_FEATURES = (
    LEAKAGE_NUMERIC_FEATURES
    +
    LEAKAGE_CATEGORICAL_FEATURES
)


# ================================================================
# FROZEN GRADIENT BOOSTING PARAMETERS
# ================================================================

LEAKAGE_GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 20,
    "random_state": 42,
}


# ================================================================
# FROZEN RESEARCH / GOVERNANCE CONSTANTS
# ================================================================

LEAKAGE_BASELINE_WINDOW_DAYS = 90

LEAKAGE_OUTCOME_WINDOW_DAYS = 90

LEAKAGE_REVENUE_RATIO_THRESHOLD = 0.50

LEAKAGE_PREDICTION_UNIT = (
    "one active customer at one month-end prediction point"
)

LEAKAGE_PRIMARY_METRIC = "PR-AUC"

LEAKAGE_VALIDATION_DESIGN = (
    "purged chronological validation"
)

LEAKAGE_CALIBRATION = "none"


# ================================================================
# PREPROCESSOR
# ================================================================

def build_leakage_preprocessor() -> ColumnTransformer:
    """
    Build the exact frozen preprocessing architecture.

    Numeric:
        median imputation with missingness indicators.

    Categorical:
        most-frequent imputation followed by one-hot encoding.

    No scaling is used for the final tree model.
    """

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                list(
                    LEAKAGE_NUMERIC_FEATURES
                ),
            ),
            (
                "categorical",
                categorical_transformer,
                list(
                    LEAKAGE_CATEGORICAL_FEATURES
                ),
            ),
        ]
    )

    return preprocessor


# ================================================================
# FROZEN PIPELINE
# ================================================================

def build_frozen_leakage_pipeline() -> Pipeline:
    """
    Build the exact frozen Revenue Leakage modelling pipeline.

    This creates an unfitted clone of the final research specification.
    """

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_leakage_preprocessor(),
            ),
            (
                "model",
                GradientBoostingClassifier(
                    **LEAKAGE_GRADIENT_BOOSTING_PARAMS
                ),
            ),
        ]
    )