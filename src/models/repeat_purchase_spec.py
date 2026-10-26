"""
Frozen production specification for the Notebook 02
90-day Repeat Purchase model.

The model predicts whether an identified newly acquired customer will
make a second observable qualifying purchase within 90 days of the
customer's first observable qualifying order.

This module contains only frozen specification and governance metadata.
Feature construction belongs in the repeat-purchase feature module.
Artifact loading and probability scoring belong in the scorer module.
"""

from __future__ import annotations

from dataclasses import dataclass


# ================================================================
# MODEL IDENTITY
# ================================================================

REPEAT_PURCHASE_MODEL_NAME = (
    "Drop-First Interpretable Logistic Regression"
)

REPEAT_PURCHASE_MODEL_VERSION = (
    "1.0.0"
)

REPEAT_PURCHASE_MODEL_TYPE = (
    "binary_classifier"
)

REPEAT_PURCHASE_TARGET_NAME = (
    "repeat_within_90d"
)

REPEAT_PURCHASE_BUSINESS_TARGET = (
    "Second observable qualifying purchase within 90 days"
)

REPEAT_PURCHASE_PREDICTION_POINT = (
    "Immediately after first observable order"
)

REPEAT_PURCHASE_PREDICTION_UNIT = (
    "One observable newly acquired customer"
)

REPEAT_PURCHASE_HORIZON_DAYS = 90


# ================================================================
# FROZEN FEATURE REGISTER
# ================================================================

REPEAT_PURCHASE_NUMERIC_FEATURES = (
    "first_order_value",
    "first_order_units",
    "first_order_product_count",
    "first_order_average_unit_price",
    "first_order_line_count",
    "first_order_max_unit_price",
    "first_order_min_unit_price",
    "first_order_price_std",
    "first_order_max_line_quantity",
    "first_order_mean_line_quantity",
    "units_per_product",
    "revenue_per_product",
    "revenue_per_unit",
    "first_order_month_number",
    "first_order_weekday",
    "first_order_hour",
    "first_order_is_weekend",
)

REPEAT_PURCHASE_CATEGORICAL_FEATURES = (
    "first_order_region",
)

REPEAT_PURCHASE_MODEL_FEATURES = (
    REPEAT_PURCHASE_NUMERIC_FEATURES
    +
    REPEAT_PURCHASE_CATEGORICAL_FEATURES
)


# ================================================================
# PREPROCESSING
# ================================================================

REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY = (
    "median"
)

REPEAT_PURCHASE_NUMERIC_ADD_INDICATOR = False

REPEAT_PURCHASE_NUMERIC_SCALER = (
    "StandardScaler"
)

REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY = (
    "most_frequent"
)

REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN = (
    "ignore"
)

REPEAT_PURCHASE_ONEHOT_DROP = (
    "first"
)

REPEAT_PURCHASE_CALIBRATION = None


# ================================================================
# LOGISTIC REGRESSION
# ================================================================

REPEAT_PURCHASE_LOGISTIC_PARAMS = {
    "max_iter": 3000,
    "random_state": 42,
}


# ================================================================
# TRAINING / VALIDATION GOVERNANCE
# ================================================================

REPEAT_PURCHASE_VALIDATION_DESIGN = (
    "Chronological Train -> Validation -> untouched future Test"
)

REPEAT_PURCHASE_FINAL_TRAINING_POPULATION = (
    "Combined chronological Train + Validation development population"
)

REPEAT_PURCHASE_FINAL_EVALUATION_POPULATION = (
    "Previously untouched later temporal Test population"
)

REPEAT_PURCHASE_MODEL_SPECIFICATION_FROZEN = True

REPEAT_PURCHASE_TEST_SET_OPENED = True

REPEAT_PURCHASE_TEST_SET_CONSUMED = True

REPEAT_PURCHASE_POST_TEST_TUNING_ALLOWED = False


# ================================================================
# MODEL SELECTION / EVALUATION
# ================================================================

REPEAT_PURCHASE_PRIMARY_METRIC = (
    "PR-AUC"
)

REPEAT_PURCHASE_COMMERCIAL_METRICS = (
    "Lift@10%",
    "Lift@20%",
    "repeater capture",
)

REPEAT_PURCHASE_FINAL_DECISION = (
    "CONDITIONAL GO"
)

REPEAT_PURCHASE_RECOMMENDED_USE = (
    "Customer prioritisation and controlled retention experimentation"
)

REPEAT_PURCHASE_PROHIBITED_INTERPRETATIONS = (
    "Causal treatment decisions",
    "Guaranteed individual probabilities",
    "Permanent churn inference",
)


# ================================================================
# FINAL RESEARCH PERFORMANCE
# ================================================================

REPEAT_PURCHASE_REPORTED_TEST_PR_AUC = (
    0.503
)

REPEAT_PURCHASE_REPORTED_TEST_ROC_AUC = (
    0.560
)

REPEAT_PURCHASE_REPORTED_TEST_BRIER = (
    0.248
)

REPEAT_PURCHASE_REPORTED_TEST_LIFT_10 = (
    1.03
)

REPEAT_PURCHASE_REPORTED_TEST_LIFT_20 = (
    1.11
)

REPEAT_PURCHASE_REPORTED_CALIBRATION_ERROR_PP = (
    4.71
)


# ================================================================
# PRE-SPECIFIED DECISION THRESHOLDS
# ================================================================

REPEAT_PURCHASE_DECISION_THRESHOLDS = {
    "minimum_roc_auc_for_signal":
        0.55,

    "minimum_lift10_for_useful_ranking":
        1.20,

    "minimum_lift20_for_useful_ranking":
        1.10,

    "maximum_relative_pr_auc_degradation_for_go":
        0.15,

    "maximum_relative_pr_auc_degradation_for_conditional":
        0.30,
}


# ================================================================
# METADATA
# ================================================================

@dataclass(
    frozen=True
)
class RepeatPurchaseModelMetadata:
    model_name: str
    model_version: str
    model_type: str
    target_name: str
    prediction_horizon_days: int
    governance_decision: str


REPEAT_PURCHASE_MODEL_METADATA = (
    RepeatPurchaseModelMetadata(
        model_name=(
            REPEAT_PURCHASE_MODEL_NAME
        ),
        model_version=(
            REPEAT_PURCHASE_MODEL_VERSION
        ),
        model_type=(
            REPEAT_PURCHASE_MODEL_TYPE
        ),
        target_name=(
            REPEAT_PURCHASE_TARGET_NAME
        ),
        prediction_horizon_days=(
            REPEAT_PURCHASE_HORIZON_DAYS
        ),
        governance_decision=(
            REPEAT_PURCHASE_FINAL_DECISION
        ),
    )
)


# ================================================================
# VALIDATION
# ================================================================

def validate_repeat_purchase_specification() -> None:
    """
    Fail closed if the frozen Notebook 02 specification drifts.
    """

    if (
        len(
            REPEAT_PURCHASE_NUMERIC_FEATURES
        )
        !=
        17
    ):
        raise ValueError(
            "Repeat Purchase numeric feature count must remain 17."
        )

    if (
        len(
            REPEAT_PURCHASE_CATEGORICAL_FEATURES
        )
        !=
        1
    ):
        raise ValueError(
            "Repeat Purchase categorical feature count must remain 1."
        )

    if (
        len(
            REPEAT_PURCHASE_MODEL_FEATURES
        )
        !=
        18
    ):
        raise ValueError(
            "Repeat Purchase total feature count must remain 18."
        )

    if (
        REPEAT_PURCHASE_CATEGORICAL_FEATURES
        !=
        (
            "first_order_region",
        )
    ):
        raise ValueError(
            "Repeat Purchase categorical feature register drift detected."
        )

    if (
        REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY
        !=
        "median"
    ):
        raise ValueError(
            "Repeat Purchase numeric imputation drift detected."
        )

    if (
        REPEAT_PURCHASE_NUMERIC_ADD_INDICATOR
        is not False
    ):
        raise ValueError(
            "Repeat Purchase numeric imputation must not add indicators."
        )

    if (
        REPEAT_PURCHASE_NUMERIC_SCALER
        !=
        "StandardScaler"
    ):
        raise ValueError(
            "Repeat Purchase numeric scaler drift detected."
        )

    if (
        REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY
        !=
        "most_frequent"
    ):
        raise ValueError(
            "Repeat Purchase categorical imputation drift detected."
        )

    if (
        REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN
        !=
        "ignore"
    ):
        raise ValueError(
            "Repeat Purchase one-hot unknown handling drift detected."
        )

    if (
        REPEAT_PURCHASE_ONEHOT_DROP
        !=
        "first"
    ):
        raise ValueError(
            "Repeat Purchase one-hot drop policy drift detected."
        )

    if (
        REPEAT_PURCHASE_LOGISTIC_PARAMS
        !=
        {
            "max_iter": 3000,
            "random_state": 42,
        }
    ):
        raise ValueError(
            "Repeat Purchase logistic parameters drift detected."
        )

    if (
        REPEAT_PURCHASE_CALIBRATION
        is not None
    ):
        raise ValueError(
            "Repeat Purchase model must not use post-hoc calibration."
        )

    if (
        REPEAT_PURCHASE_MODEL_SPECIFICATION_FROZEN
        is not True
    ):
        raise ValueError(
            "Repeat Purchase model specification must remain frozen."
        )

    if (
        REPEAT_PURCHASE_TEST_SET_OPENED
        is not True
        or
        REPEAT_PURCHASE_TEST_SET_CONSUMED
        is not True
    ):
        raise ValueError(
            "Repeat Purchase final temporal test governance drift detected."
        )

    if (
        REPEAT_PURCHASE_POST_TEST_TUNING_ALLOWED
        is not False
    ):
        raise ValueError(
            "Repeat Purchase post-test tuning must remain prohibited."
        )

    if (
        REPEAT_PURCHASE_FINAL_DECISION
        !=
        "CONDITIONAL GO"
    ):
        raise ValueError(
            "Repeat Purchase governance decision drift detected."
        )


validate_repeat_purchase_specification()