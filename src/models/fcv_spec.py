"""
Frozen production specification for the Future Customer Value model.

This module reproduces the exact model architecture selected and frozen
in Notebook 05 — ML Future Customer Value.

Research specification
----------------------
Business target:
    Future 90-Day Customer Value (FCV90).

Prediction unit:
    One identified customer at one monthly prediction point.

Eligibility:
    At least one qualifying merchandise purchase during the
    preceding 180 days.

Outcome window:
    Following 90 days.

Stage A target:
    Probability that FCV90 is greater than zero.

Frozen Stage A model:
    Full Random Forest classifier.

Stage B economic scale:
    50% of prior 180-day customer merchandise revenue.

Final frozen architecture:
    Expected FCV90
    =
    P(Future Active)
    ×
    0.50
    ×
    h180_revenue

Primary Stage A selection metric:
    PR-AUC.

Primary complete-model use:
    Economic ranking and future-revenue concentration.

Validation:
    Purged chronological validation with embargo periods.

Governance:
    CONDITIONAL GO for ranking, segmentation, prioritisation and
    controlled commercial decision support.

The model is not approved as a literal aggregate revenue forecast.

The specification in this module must not be modified independently of
the governed research model.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.base import ModelMetadata


# ================================================================
# MODEL METADATA
# ================================================================

FCV_MODEL_METADATA = ModelMetadata(
    model_name=(
        "Random Forest Activity × "
        "180-Day Historical Run Rate"
    ),
    model_version="1.0.0",
    model_type="hurdle",
    target_name="fcv90",
)


FCV_STAGE_A_MODEL_METADATA = ModelMetadata(
    model_name="Full Random Forest",
    model_version="1.0.0",
    model_type="classification",
    target_name="future_active",
)


# ================================================================
# FROZEN NUMERIC FEATURE REGISTER
# ================================================================

FCV_NUMERIC_FEATURES = (
    # ------------------------------------------------------------
    # 180-day customer economic scale
    # ------------------------------------------------------------
    "h180_revenue",
    "h180_orders",
    "h180_active_days",
    "h180_units",
    "h180_unique_products",
    "h180_average_unit_price",
    "h180_maximum_unit_price",
    "h180_average_order_value",
    "h180_revenue_per_active_day",
    "h180_units_per_order",
    "h180_products_per_order",

    # ------------------------------------------------------------
    # Recent 90 days
    # ------------------------------------------------------------
    "r90_revenue",
    "r90_orders",
    "r90_active_days",
    "r90_units",
    "r90_unique_products",
    "r90_average_order_value",
    "r90_active_day_rate",

    # ------------------------------------------------------------
    # Previous 90 days
    # ------------------------------------------------------------
    "p90_revenue",
    "p90_orders",
    "p90_active_days",
    "p90_units",
    "p90_unique_products",
    "p90_average_order_value",
    "p90_active_day_rate",

    # ------------------------------------------------------------
    # Recent 30 days
    # ------------------------------------------------------------
    "r30_revenue",
    "r30_orders",
    "r30_active_days",
    "r30_units",

    # ------------------------------------------------------------
    # Recency / activity
    # ------------------------------------------------------------
    "days_since_last_purchase",
    "active_in_last_30d",
    "active_in_last_90d",

    # ------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------
    "r90_minus_p90_revenue",
    "r90_to_p90_revenue_ratio",
    "log_revenue_momentum",
    "recent30_revenue_share",
    "recent30_order_share",
    "product_breadth_change",

    # ------------------------------------------------------------
    # Cadence
    # ------------------------------------------------------------
    "h180_average_gap_days",
    "h180_maximum_gap_days",
    "h180_minimum_gap_days",
    "h180_orders_per_30_days",
    "h180_active_day_rate",

    # ------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------
    "log1p_h180_revenue",
    "log1p_r90_revenue",
    "log1p_p90_revenue",
    "log1p_h180_orders",

    # ------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------
    "prediction_month_sin",
    "prediction_month_cos",
)


# ================================================================
# FROZEN CATEGORICAL FEATURE REGISTER
# ================================================================

FCV_CATEGORICAL_FEATURES = (
    "last_region",
)


# ================================================================
# COMPLETE FROZEN FEATURE REGISTER
# ================================================================

FCV_MODEL_FEATURES = (
    FCV_NUMERIC_FEATURES
    +
    FCV_CATEGORICAL_FEATURES
)


# ================================================================
# FROZEN STAGE A RANDOM FOREST PARAMETERS
# ================================================================

FCV_STAGE_A_RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


# ================================================================
# FROZEN RESEARCH / GOVERNANCE CONSTANTS
# ================================================================

FCV_ELIGIBILITY_LOOKBACK_DAYS = 180

FCV_RECENT_WINDOW_DAYS = 90

FCV_OUTCOME_WINDOW_DAYS = 90

FCV_STAGE_B_RUN_RATE_MULTIPLIER = 0.50


FCV_PREDICTION_UNIT = (
    "one identified customer at one monthly prediction point"
)


FCV_STAGE_A_TARGET = (
    "future_active"
)


FCV_STAGE_A_TARGET_DESCRIPTION = (
    "probability FCV90 > 0"
)


FCV_STAGE_B_MODEL_NAME = (
    "180-Day Historical Run Rate"
)


FCV_STAGE_B_FORMULA = (
    "0.50 × h180_revenue"
)


FCV_COMPLETE_MODEL_NAME = (
    "Random Forest Activity × "
    "180-Day Historical Run Rate"
)


FCV_COMPLETE_FORMULA = (
    "P(Future Active) × 0.50 × h180_revenue"
)


FCV_PRIMARY_METRIC = "PR-AUC"


FCV_ECONOMIC_METRIC = (
    "future revenue captured by ranked customers"
)


FCV_VALIDATION_DESIGN = (
    "purged chronological validation"
)


FCV_FINAL_DECISION = (
    "CONDITIONAL GO"
)


FCV_APPROVED_USE = (
    "ranking, segmentation, prioritisation and "
    "controlled commercial decision support"
)


FCV_PROHIBITED_USE = (
    "literal aggregate revenue forecasting"
)


FCV_CALIBRATION = "none"


# ================================================================
# TEMPORAL DEVELOPMENT SPLITS
# ================================================================

FCV_TRAIN_START = "2010-05"
FCV_TRAIN_END = "2010-11"

FCV_EMBARGO_1_START = "2010-12"
FCV_EMBARGO_1_END = "2011-02"

FCV_VALIDATION_START = "2011-03"
FCV_VALIDATION_END = "2011-04"

FCV_EMBARGO_2_START = "2011-05"
FCV_EMBARGO_2_END = "2011-06"

FCV_TEST_START = "2011-07"
FCV_TEST_END = "2011-08"


# ================================================================
# FINAL RESEARCH PERFORMANCE
# ================================================================

FCV_FINAL_TEST_EPISODES = 5534

FCV_STAGE_A_TEST_PR_AUC = 0.822

FCV_STAGE_A_TEST_ROC_AUC = 0.750

FCV_STAGE_A_TEST_LIFT_AT_10 = 1.66

FCV_STAGE_A_TEST_LIFT_AT_20 = 1.57

FCV_COMPLETE_TEST_SPEARMAN = 0.550

FCV_COMPLETE_TEST_CAPTURE_AT_10 = 0.574

FCV_COMPLETE_TEST_CAPTURE_AT_20 = 0.687

FCV_COMPLETE_TEST_CAPTURE_AT_30 = 0.769

FCV_COMPLETE_TEST_PREDICTED_TO_ACTUAL_REVENUE = 0.593


# ================================================================
# STAGE A PREPROCESSOR
# ================================================================

def build_fcv_stage_a_preprocessor() -> ColumnTransformer:
    """
    Build the exact frozen Stage A preprocessing architecture.

    Numeric:
        Median imputation with missingness indicators.

    Categorical:
        Most-frequent imputation followed by one-hot encoding.

    No scaling is used for the frozen Random Forest model.
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
                    FCV_NUMERIC_FEATURES
                ),
            ),
            (
                "categorical",
                categorical_transformer,
                list(
                    FCV_CATEGORICAL_FEATURES
                ),
            ),
        ]
    )

    return preprocessor


# ================================================================
# FROZEN STAGE A PIPELINE
# ================================================================

def build_frozen_fcv_stage_a_pipeline() -> Pipeline:
    """
    Build an unfitted clone of the exact frozen Stage A model.
    """

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_fcv_stage_a_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    **FCV_STAGE_A_RANDOM_FOREST_PARAMS
                ),
            ),
        ]
    )


# ================================================================
# DETERMINISTIC STAGE B
# ================================================================

def calculate_fcv_stage_b_value(
    h180_revenue,
):
    """
    Apply the frozen Stage B economic-scale rule.

    Stage B
    -------
    Conditional economic scale
        =
        0.50 × prior 180-day customer merchandise revenue.

    This is deterministic and is not a fitted ML model.
    """

    return (
        FCV_STAGE_B_RUN_RATE_MULTIPLIER
        *
        h180_revenue
    )


# ================================================================
# COMPLETE FCV FORMULA
# ================================================================

def calculate_expected_fcv90(
    future_active_probability,
    h180_revenue,
):
    """
    Apply the complete frozen FCV hurdle formula.

    Expected FCV90
        =
        P(Future Active)
        ×
        0.50
        ×
        h180_revenue.
    """

    return (
        future_active_probability
        *
        calculate_fcv_stage_b_value(
            h180_revenue
        )
    )