"""
Frozen production specification for the customer reactivation model.

This module reproduces the model architecture frozen in the research
notebook:

    03_ML_Reactivation_Propensity.ipynb

The specification must not be modified casually.

Frozen research model:
- prediction problem:
    probability of customer reactivation during the 90 days following
    an eligible 90-day inactivity episode
- algorithm:
    RandomForestClassifier
- estimators:
    500
- max_depth:
    8
- min_samples_leaf:
    15
- max_features:
    sqrt
- random_state:
    42
- numeric preprocessing:
    median imputation with missingness indicators
- categorical preprocessing:
    most-frequent imputation + one-hot encoding
- calibration:
    none
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.base import ModelMetadata


# ================================================================
# MODEL IDENTITY
# ================================================================

REACTIVATION_MODEL_NAME = "Full Random Forest"
REACTIVATION_MODEL_VERSION = "1.0.0"

REACTIVATION_TARGET_NAME = (
    "reactivation_within_next_90_days"
)

REACTIVATION_MODEL_METADATA = ModelMetadata(
    model_name=REACTIVATION_MODEL_NAME,
    model_version=REACTIVATION_MODEL_VERSION,
    model_type="classification",
    target_name=REACTIVATION_TARGET_NAME,
)


# ================================================================
# FROZEN FEATURE REGISTER
# ================================================================

REACTIVATION_NUMERIC_FEATURES = (
    # Relationship depth
    "historical_order_count",
    "historical_revenue",
    "historical_units",
    "historical_average_order_value",
    "tenure_at_prediction_days",

    # Historical cadence
    "historical_average_gap_days",
    "historical_max_gap_days",
    "historical_min_gap_days",
    "orders_per_30_tenure_days",
    "inactivity_to_average_gap_ratio",

    # Most recent order
    "last_order_value",
    "last_order_units",
    "last_order_product_count",
    "last_order_average_unit_price",
    "last_to_historical_aov_ratio",

    # Historical economics
    "revenue_per_30_tenure_days",

    # Prior inactivity / reactivation
    "prior_inactivity_episode_count",
    "prior_reactivation_count",
    "prior_reactivation_rate",

    # History availability
    "has_prior_purchase_gap",
    "has_prior_inactivity_episode",
    "has_prior_successful_reactivation",

    # Nonlinear depth alternatives
    "log1p_historical_order_count",
    "log1p_historical_revenue",
    "log1p_historical_units",

    # Calendar
    "prediction_month",
    "prediction_weekday",
    "prediction_is_weekend",
)


REACTIVATION_CATEGORICAL_FEATURES = (
    "last_order_region",
)


REACTIVATION_MODEL_FEATURES = (
    REACTIVATION_NUMERIC_FEATURES
    +
    REACTIVATION_CATEGORICAL_FEATURES
)


# ================================================================
# FROZEN RANDOM FOREST HYPERPARAMETERS
# ================================================================

REACTIVATION_RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 15,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


# ================================================================
# PREPROCESSING
# ================================================================

def build_reactivation_preprocessor() -> ColumnTransformer:
    """
    Recreate the exact preprocessing architecture frozen in research.
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

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                list(
                    REACTIVATION_NUMERIC_FEATURES
                ),
            ),
            (
                "categorical",
                categorical_transformer,
                list(
                    REACTIVATION_CATEGORICAL_FEATURES
                ),
            ),
        ]
    )


# ================================================================
# FROZEN MODEL PIPELINE
# ================================================================

def build_frozen_reactivation_pipeline() -> Pipeline:
    """
    Construct an unfitted copy of the frozen Reactivation model.

    Training is intentionally not performed in this function.
    """

    preprocessor = (
        build_reactivation_preprocessor()
    )

    model = RandomForestClassifier(
        **REACTIVATION_RANDOM_FOREST_PARAMS
    )

    return Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )