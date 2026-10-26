import pytest

from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_CALIBRATION,
    REPEAT_PURCHASE_CATEGORICAL_FEATURES,
    REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY,
    REPEAT_PURCHASE_COMMERCIAL_METRICS,
    REPEAT_PURCHASE_FINAL_DECISION,
    REPEAT_PURCHASE_HORIZON_DAYS,
    REPEAT_PURCHASE_LOGISTIC_PARAMS,
    REPEAT_PURCHASE_MODEL_FEATURES,
    REPEAT_PURCHASE_MODEL_METADATA,
    REPEAT_PURCHASE_MODEL_NAME,
    REPEAT_PURCHASE_MODEL_SPECIFICATION_FROZEN,
    REPEAT_PURCHASE_NUMERIC_ADD_INDICATOR,
    REPEAT_PURCHASE_NUMERIC_FEATURES,
    REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY,
    REPEAT_PURCHASE_NUMERIC_SCALER,
    REPEAT_PURCHASE_ONEHOT_DROP,
    REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN,
    REPEAT_PURCHASE_POST_TEST_TUNING_ALLOWED,
    REPEAT_PURCHASE_PRIMARY_METRIC,
    REPEAT_PURCHASE_REPORTED_TEST_BRIER,
    REPEAT_PURCHASE_REPORTED_TEST_LIFT_10,
    REPEAT_PURCHASE_REPORTED_TEST_LIFT_20,
    REPEAT_PURCHASE_REPORTED_TEST_PR_AUC,
    REPEAT_PURCHASE_REPORTED_TEST_ROC_AUC,
    REPEAT_PURCHASE_TARGET_NAME,
    REPEAT_PURCHASE_TEST_SET_CONSUMED,
    REPEAT_PURCHASE_TEST_SET_OPENED,
    validate_repeat_purchase_specification,
)


def test_model_name():
    assert (
        REPEAT_PURCHASE_MODEL_NAME
        ==
        "Drop-First Interpretable Logistic Regression"
    )


def test_target_name():
    assert (
        REPEAT_PURCHASE_TARGET_NAME
        ==
        "repeat_within_90d"
    )


def test_horizon():
    assert (
        REPEAT_PURCHASE_HORIZON_DAYS
        ==
        90
    )


def test_exact_numeric_feature_count():
    assert (
        len(
            REPEAT_PURCHASE_NUMERIC_FEATURES
        )
        ==
        17
    )


def test_exact_categorical_feature_count():
    assert (
        len(
            REPEAT_PURCHASE_CATEGORICAL_FEATURES
        )
        ==
        1
    )


def test_exact_total_feature_count():
    assert (
        len(
            REPEAT_PURCHASE_MODEL_FEATURES
        )
        ==
        18
    )


def test_exact_feature_order():
    assert (
        REPEAT_PURCHASE_MODEL_FEATURES
        ==
        (
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
            "first_order_region",
        )
    )


def test_numeric_imputation():
    assert (
        REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY
        ==
        "median"
    )


def test_no_numeric_missing_indicators():
    assert (
        REPEAT_PURCHASE_NUMERIC_ADD_INDICATOR
        is False
    )


def test_numeric_scaler():
    assert (
        REPEAT_PURCHASE_NUMERIC_SCALER
        ==
        "StandardScaler"
    )


def test_categorical_imputation():
    assert (
        REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY
        ==
        "most_frequent"
    )


def test_unknown_categories_are_ignored():
    assert (
        REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN
        ==
        "ignore"
    )


def test_drop_first_encoding():
    assert (
        REPEAT_PURCHASE_ONEHOT_DROP
        ==
        "first"
    )


def test_logistic_parameters():
    assert (
        REPEAT_PURCHASE_LOGISTIC_PARAMS
        ==
        {
            "max_iter": 3000,
            "random_state": 42,
        }
    )


def test_no_calibration():
    assert (
        REPEAT_PURCHASE_CALIBRATION
        is None
    )


def test_primary_metric():
    assert (
        REPEAT_PURCHASE_PRIMARY_METRIC
        ==
        "PR-AUC"
    )


def test_commercial_metrics():
    assert (
        REPEAT_PURCHASE_COMMERCIAL_METRICS
        ==
        (
            "Lift@10%",
            "Lift@20%",
            "repeater capture",
        )
    )


def test_model_is_frozen():
    assert (
        REPEAT_PURCHASE_MODEL_SPECIFICATION_FROZEN
        is True
    )


def test_test_set_was_opened():
    assert (
        REPEAT_PURCHASE_TEST_SET_OPENED
        is True
    )


def test_test_set_was_consumed():
    assert (
        REPEAT_PURCHASE_TEST_SET_CONSUMED
        is True
    )


def test_post_test_tuning_is_prohibited():
    assert (
        REPEAT_PURCHASE_POST_TEST_TUNING_ALLOWED
        is False
    )


def test_final_decision():
    assert (
        REPEAT_PURCHASE_FINAL_DECISION
        ==
        "CONDITIONAL GO"
    )


def test_reported_test_metrics():
    assert (
        REPEAT_PURCHASE_REPORTED_TEST_PR_AUC
        ==
        pytest.approx(
            0.503
        )
    )

    assert (
        REPEAT_PURCHASE_REPORTED_TEST_ROC_AUC
        ==
        pytest.approx(
            0.560
        )
    )

    assert (
        REPEAT_PURCHASE_REPORTED_TEST_BRIER
        ==
        pytest.approx(
            0.248
        )
    )

    assert (
        REPEAT_PURCHASE_REPORTED_TEST_LIFT_10
        ==
        pytest.approx(
            1.03
        )
    )

    assert (
        REPEAT_PURCHASE_REPORTED_TEST_LIFT_20
        ==
        pytest.approx(
            1.11
        )
    )


def test_metadata_matches_spec():
    assert (
        REPEAT_PURCHASE_MODEL_METADATA.model_name
        ==
        REPEAT_PURCHASE_MODEL_NAME
    )

    assert (
        REPEAT_PURCHASE_MODEL_METADATA.target_name
        ==
        REPEAT_PURCHASE_TARGET_NAME
    )

    assert (
        REPEAT_PURCHASE_MODEL_METADATA.prediction_horizon_days
        ==
        90
    )

    assert (
        REPEAT_PURCHASE_MODEL_METADATA.governance_decision
        ==
        "CONDITIONAL GO"
    )


def test_validate_specification():
    validate_repeat_purchase_specification()