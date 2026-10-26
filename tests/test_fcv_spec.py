"""
Tests for the frozen Future Customer Value production specification.
"""

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.fcv_spec import (
    FCV_APPROVED_USE,
    FCV_CALIBRATION,
    FCV_CATEGORICAL_FEATURES,
    FCV_COMPLETE_FORMULA,
    FCV_COMPLETE_MODEL_NAME,
    FCV_COMPLETE_TEST_CAPTURE_AT_10,
    FCV_COMPLETE_TEST_CAPTURE_AT_20,
    FCV_COMPLETE_TEST_CAPTURE_AT_30,
    FCV_COMPLETE_TEST_PREDICTED_TO_ACTUAL_REVENUE,
    FCV_COMPLETE_TEST_SPEARMAN,
    FCV_ECONOMIC_METRIC,
    FCV_ELIGIBILITY_LOOKBACK_DAYS,
    FCV_FINAL_DECISION,
    FCV_FINAL_TEST_EPISODES,
    FCV_MODEL_FEATURES,
    FCV_MODEL_METADATA,
    FCV_NUMERIC_FEATURES,
    FCV_OUTCOME_WINDOW_DAYS,
    FCV_PREDICTION_UNIT,
    FCV_PRIMARY_METRIC,
    FCV_PROHIBITED_USE,
    FCV_RECENT_WINDOW_DAYS,
    FCV_STAGE_A_MODEL_METADATA,
    FCV_STAGE_A_RANDOM_FOREST_PARAMS,
    FCV_STAGE_A_TARGET,
    FCV_STAGE_A_TARGET_DESCRIPTION,
    FCV_STAGE_A_TEST_LIFT_AT_10,
    FCV_STAGE_A_TEST_LIFT_AT_20,
    FCV_STAGE_A_TEST_PR_AUC,
    FCV_STAGE_A_TEST_ROC_AUC,
    FCV_STAGE_B_FORMULA,
    FCV_STAGE_B_MODEL_NAME,
    FCV_STAGE_B_RUN_RATE_MULTIPLIER,
    FCV_TEST_END,
    FCV_TEST_START,
    FCV_TRAIN_END,
    FCV_TRAIN_START,
    FCV_VALIDATION_DESIGN,
    FCV_VALIDATION_END,
    FCV_VALIDATION_START,
    build_fcv_stage_a_preprocessor,
    build_frozen_fcv_stage_a_pipeline,
    calculate_expected_fcv90,
    calculate_fcv_stage_b_value,
)


# ================================================================
# TEST HELPERS
# ================================================================

def get_named_transformer(
    preprocessor: ColumnTransformer,
    transformer_name: str,
):
    """
    Return a named transformer from an unfitted ColumnTransformer.

    ColumnTransformer.transformers stores entries as:

        (
            name,
            transformer,
            columns,
        )

    so it cannot be converted directly with dict().
    """

    for (
        name,
        transformer,
        _columns,
    ) in preprocessor.transformers:

        if (
            name
            ==
            transformer_name
        ):
            return transformer

    raise KeyError(
        f"Transformer not found: {transformer_name}"
    )


# ================================================================
# FEATURE REGISTER
# ================================================================

def test_frozen_fcv_feature_count():

    assert len(
        FCV_MODEL_FEATURES
    ) == 50


def test_frozen_numeric_feature_count():

    assert len(
        FCV_NUMERIC_FEATURES
    ) == 49


def test_frozen_categorical_feature_count():

    assert len(
        FCV_CATEGORICAL_FEATURES
    ) == 1


def test_last_region_is_final_feature():

    assert (
        FCV_MODEL_FEATURES[-1]
        ==
        "last_region"
    )


def test_feature_register_has_no_duplicates():

    assert (
        len(
            FCV_MODEL_FEATURES
        )
        ==
        len(
            set(
                FCV_MODEL_FEATURES
            )
        )
    )


def test_h180_revenue_is_first_feature():

    assert (
        FCV_MODEL_FEATURES[0]
        ==
        "h180_revenue"
    )


# ================================================================
# PREPROCESSOR
# ================================================================

def test_fcv_preprocessor_type():

    preprocessor = (
        build_fcv_stage_a_preprocessor()
    )

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )


def test_numeric_imputer_matches_frozen_spec():

    preprocessor = (
        build_fcv_stage_a_preprocessor()
    )

    numeric_pipeline = (
        get_named_transformer(
            preprocessor=preprocessor,
            transformer_name="numeric",
        )
    )

    assert isinstance(
        numeric_pipeline,
        Pipeline,
    )

    imputer = (
        numeric_pipeline.named_steps[
            "imputer"
        ]
    )

    assert isinstance(
        imputer,
        SimpleImputer,
    )

    assert (
        imputer.strategy
        ==
        "median"
    )

    assert (
        imputer.add_indicator
        is True
    )


def test_categorical_pipeline_matches_frozen_spec():

    preprocessor = (
        build_fcv_stage_a_preprocessor()
    )

    categorical_pipeline = (
        get_named_transformer(
            preprocessor=preprocessor,
            transformer_name="categorical",
        )
    )

    assert isinstance(
        categorical_pipeline,
        Pipeline,
    )

    imputer = (
        categorical_pipeline.named_steps[
            "imputer"
        ]
    )

    encoder = (
        categorical_pipeline.named_steps[
            "onehot"
        ]
    )

    assert isinstance(
        imputer,
        SimpleImputer,
    )

    assert (
        imputer.strategy
        ==
        "most_frequent"
    )

    assert isinstance(
        encoder,
        OneHotEncoder,
    )

    assert (
        encoder.handle_unknown
        ==
        "ignore"
    )

    assert (
        encoder.sparse_output
        is False
    )


# ================================================================
# FROZEN STAGE A MODEL
# ================================================================

def test_frozen_fcv_stage_a_pipeline_type():

    model = (
        build_frozen_fcv_stage_a_pipeline()
    )

    assert isinstance(
        model,
        Pipeline,
    )


def test_stage_a_random_forest_model_type():

    model = (
        build_frozen_fcv_stage_a_pipeline()
    )

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    assert isinstance(
        estimator,
        RandomForestClassifier,
    )


def test_stage_a_random_forest_hyperparameters():

    model = (
        build_frozen_fcv_stage_a_pipeline()
    )

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    assert estimator.n_estimators == 500

    assert estimator.max_depth == 10

    assert estimator.min_samples_leaf == 20

    assert estimator.max_features == "sqrt"

    assert estimator.random_state == 42

    assert estimator.n_jobs == -1


def test_stage_a_parameter_register_matches_model():

    model = (
        build_frozen_fcv_stage_a_pipeline()
    )

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    for (
        parameter,
        expected_value,
    ) in (
        FCV_STAGE_A_RANDOM_FOREST_PARAMS.items()
    ):

        assert (
            getattr(
                estimator,
                parameter,
            )
            ==
            expected_value
        )


# ================================================================
# TARGET / WINDOWS
# ================================================================

def test_fcv_target_windows():

    assert (
        FCV_ELIGIBILITY_LOOKBACK_DAYS
        ==
        180
    )

    assert (
        FCV_RECENT_WINDOW_DAYS
        ==
        90
    )

    assert (
        FCV_OUTCOME_WINDOW_DAYS
        ==
        90
    )


def test_stage_a_target():

    assert (
        FCV_STAGE_A_TARGET
        ==
        "future_active"
    )

    assert (
        FCV_STAGE_A_TARGET_DESCRIPTION
        ==
        "probability FCV90 > 0"
    )


# ================================================================
# STAGE B
# ================================================================

def test_stage_b_specification():

    assert (
        FCV_STAGE_B_MODEL_NAME
        ==
        "180-Day Historical Run Rate"
    )

    assert (
        FCV_STAGE_B_RUN_RATE_MULTIPLIER
        ==
        0.50
    )

    assert (
        FCV_STAGE_B_FORMULA
        ==
        "0.50 × h180_revenue"
    )


def test_stage_b_value_calculation_scalar():

    result = (
        calculate_fcv_stage_b_value(
            1000.0
        )
    )

    assert result == 500.0


def test_stage_b_value_calculation_array():

    revenue = np.array(
        [
            0.0,
            100.0,
            500.0,
            1000.0,
        ]
    )

    expected = np.array(
        [
            0.0,
            50.0,
            250.0,
            500.0,
        ]
    )

    result = (
        calculate_fcv_stage_b_value(
            revenue
        )
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


# ================================================================
# COMPLETE HURDLE FORMULA
# ================================================================

def test_complete_fcv_formula_name():

    assert (
        FCV_COMPLETE_MODEL_NAME
        ==
        (
            "Random Forest Activity × "
            "180-Day Historical Run Rate"
        )
    )

    assert (
        FCV_COMPLETE_FORMULA
        ==
        (
            "P(Future Active) × "
            "0.50 × h180_revenue"
        )
    )


def test_expected_fcv90_scalar():

    result = (
        calculate_expected_fcv90(
            future_active_probability=0.40,
            h180_revenue=1000.0,
        )
    )

    assert result == 200.0


def test_expected_fcv90_array():

    probabilities = np.array(
        [
            0.0,
            0.25,
            0.50,
            1.0,
        ]
    )

    revenues = np.array(
        [
            1000.0,
            1000.0,
            1000.0,
            1000.0,
        ]
    )

    expected = np.array(
        [
            0.0,
            125.0,
            250.0,
            500.0,
        ]
    )

    result = (
        calculate_expected_fcv90(
            future_active_probability=(
                probabilities
            ),
            h180_revenue=revenues,
        )
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


# ================================================================
# GOVERNANCE
# ================================================================

def test_fcv_prediction_unit():

    assert (
        FCV_PREDICTION_UNIT
        ==
        (
            "one identified customer at "
            "one monthly prediction point"
        )
    )


def test_fcv_primary_metric():

    assert (
        FCV_PRIMARY_METRIC
        ==
        "PR-AUC"
    )


def test_fcv_economic_metric():

    assert (
        FCV_ECONOMIC_METRIC
        ==
        (
            "future revenue captured by "
            "ranked customers"
        )
    )


def test_fcv_validation_design():

    assert (
        FCV_VALIDATION_DESIGN
        ==
        "purged chronological validation"
    )


def test_fcv_has_no_calibration_layer():

    assert (
        FCV_CALIBRATION
        ==
        "none"
    )


def test_fcv_governance_decision():

    assert (
        FCV_FINAL_DECISION
        ==
        "CONDITIONAL GO"
    )

    assert (
        "ranking"
        in
        FCV_APPROVED_USE
    )

    assert (
        FCV_PROHIBITED_USE
        ==
        "literal aggregate revenue forecasting"
    )


# ================================================================
# TEMPORAL SPLITS
# ================================================================

def test_fcv_temporal_split():

    assert FCV_TRAIN_START == "2010-05"

    assert FCV_TRAIN_END == "2010-11"

    assert FCV_VALIDATION_START == "2011-03"

    assert FCV_VALIDATION_END == "2011-04"

    assert FCV_TEST_START == "2011-07"

    assert FCV_TEST_END == "2011-08"


# ================================================================
# FINAL TEST PERFORMANCE
# ================================================================

def test_fcv_final_test_metrics():

    assert (
        FCV_FINAL_TEST_EPISODES
        ==
        5534
    )

    assert np.isclose(
        FCV_STAGE_A_TEST_PR_AUC,
        0.822,
    )

    assert np.isclose(
        FCV_STAGE_A_TEST_ROC_AUC,
        0.750,
    )

    assert np.isclose(
        FCV_STAGE_A_TEST_LIFT_AT_10,
        1.66,
    )

    assert np.isclose(
        FCV_STAGE_A_TEST_LIFT_AT_20,
        1.57,
    )

    assert np.isclose(
        FCV_COMPLETE_TEST_SPEARMAN,
        0.550,
    )

    assert np.isclose(
        FCV_COMPLETE_TEST_CAPTURE_AT_10,
        0.574,
    )

    assert np.isclose(
        FCV_COMPLETE_TEST_CAPTURE_AT_20,
        0.687,
    )

    assert np.isclose(
        FCV_COMPLETE_TEST_CAPTURE_AT_30,
        0.769,
    )

    assert np.isclose(
        FCV_COMPLETE_TEST_PREDICTED_TO_ACTUAL_REVENUE,
        0.593,
    )


# ================================================================
# METADATA
# ================================================================

def test_fcv_stage_a_metadata():

    assert (
        FCV_STAGE_A_MODEL_METADATA.model_name
        ==
        "Full Random Forest"
    )

    assert (
        FCV_STAGE_A_MODEL_METADATA.model_version
        ==
        "1.0.0"
    )

    assert (
        FCV_STAGE_A_MODEL_METADATA.model_type
        ==
        "classification"
    )

    assert (
        FCV_STAGE_A_MODEL_METADATA.target_name
        ==
        "future_active"
    )


def test_complete_fcv_metadata():

    assert (
        FCV_MODEL_METADATA.model_name
        ==
        (
            "Random Forest Activity × "
            "180-Day Historical Run Rate"
        )
    )

    assert (
        FCV_MODEL_METADATA.model_version
        ==
        "1.0.0"
    )

    assert (
        FCV_MODEL_METADATA.model_type
        ==
        "hurdle"
    )

    assert (
        FCV_MODEL_METADATA.target_name
        ==
        "fcv90"
    )