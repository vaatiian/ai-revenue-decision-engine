"""
Tests for the frozen Revenue Leakage model specification.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.leakage_spec import (
    LEAKAGE_BASELINE_WINDOW_DAYS,
    LEAKAGE_CALIBRATION,
    LEAKAGE_CATEGORICAL_FEATURES,
    LEAKAGE_GRADIENT_BOOSTING_PARAMS,
    LEAKAGE_MODEL_FEATURES,
    LEAKAGE_MODEL_METADATA,
    LEAKAGE_NUMERIC_FEATURES,
    LEAKAGE_OUTCOME_WINDOW_DAYS,
    LEAKAGE_PRIMARY_METRIC,
    LEAKAGE_REVENUE_RATIO_THRESHOLD,
    LEAKAGE_VALIDATION_DESIGN,
    build_frozen_leakage_pipeline,
    build_leakage_preprocessor,
)


# ================================================================
# FEATURE REGISTER
# ================================================================

def test_frozen_leakage_feature_count():
    """
    Research froze 45 features.
    """

    assert (
        len(
            LEAKAGE_MODEL_FEATURES
        )
        ==
        45
    )


def test_frozen_numeric_feature_count():
    """
    Research froze 44 numeric features.
    """

    assert (
        len(
            LEAKAGE_NUMERIC_FEATURES
        )
        ==
        44
    )


def test_frozen_categorical_feature_count():
    """
    Research froze one categorical feature.
    """

    assert (
        len(
            LEAKAGE_CATEGORICAL_FEATURES
        )
        ==
        1
    )


def test_last_region_is_final_feature():
    """
    last_region is the sole categorical feature and appears last
    in the exact frozen feature register.
    """

    assert (
        LEAKAGE_MODEL_FEATURES[-1]
        ==
        "last_region"
    )


def test_feature_register_has_no_duplicates():
    """
    The frozen model feature register must contain unique names.
    """

    assert (
        len(
            LEAKAGE_MODEL_FEATURES
        )
        ==
        len(
            set(
                LEAKAGE_MODEL_FEATURES
            )
        )
    )


# ================================================================
# PREPROCESSOR
# ================================================================

def test_leakage_preprocessor_type():

    preprocessor = (
        build_leakage_preprocessor()
    )

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )


def test_numeric_imputer_matches_frozen_spec():

    preprocessor = (
        build_leakage_preprocessor()
    )

    transformers = {
        name: transformer
        for (
            name,
            transformer,
            columns,
        )
        in preprocessor.transformers
    }

    numeric_pipeline = (
        transformers[
            "numeric"
        ]
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
        build_leakage_preprocessor()
    )

    transformers = {
        name: transformer
        for (
            name,
            transformer,
            columns,
        )
        in preprocessor.transformers
    }

    categorical_pipeline = (
        transformers[
            "categorical"
        ]
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
# FINAL MODEL
# ================================================================

def test_frozen_pipeline_type():

    pipeline = (
        build_frozen_leakage_pipeline()
    )

    assert isinstance(
        pipeline,
        Pipeline,
    )

    assert list(
        pipeline.named_steps.keys()
    ) == [
        "preprocessing",
        "model",
    ]


def test_gradient_boosting_model_type():

    pipeline = (
        build_frozen_leakage_pipeline()
    )

    model = (
        pipeline.named_steps[
            "model"
        ]
    )

    assert isinstance(
        model,
        GradientBoostingClassifier,
    )


def test_gradient_boosting_hyperparameters():

    pipeline = (
        build_frozen_leakage_pipeline()
    )

    model = (
        pipeline.named_steps[
            "model"
        ]
    )

    assert (
        model.n_estimators
        ==
        250
    )

    assert (
        model.learning_rate
        ==
        0.03
    )

    assert (
        model.max_depth
        ==
        3
    )

    assert (
        model.min_samples_leaf
        ==
        20
    )

    assert (
        model.random_state
        ==
        42
    )


def test_parameter_register_matches_model():

    pipeline = (
        build_frozen_leakage_pipeline()
    )

    model = (
        pipeline.named_steps[
            "model"
        ]
    )

    for (
        parameter,
        expected_value,
    ) in (
        LEAKAGE_GRADIENT_BOOSTING_PARAMS.items()
    ):

        assert (
            getattr(
                model,
                parameter
            )
            ==
            expected_value
        )


# ================================================================
# TARGET / RESEARCH GOVERNANCE
# ================================================================

def test_leakage_target_windows():

    assert (
        LEAKAGE_BASELINE_WINDOW_DAYS
        ==
        90
    )

    assert (
        LEAKAGE_OUTCOME_WINDOW_DAYS
        ==
        90
    )


def test_leakage_target_threshold():

    assert (
        LEAKAGE_REVENUE_RATIO_THRESHOLD
        ==
        0.50
    )


def test_leakage_primary_metric():

    assert (
        LEAKAGE_PRIMARY_METRIC
        ==
        "PR-AUC"
    )


def test_leakage_validation_design():

    assert (
        LEAKAGE_VALIDATION_DESIGN
        ==
        "purged chronological validation"
    )


def test_leakage_has_no_calibration_layer():

    assert (
        LEAKAGE_CALIBRATION
        ==
        "none"
    )


# ================================================================
# MODEL METADATA
# ================================================================

def test_leakage_metadata():

    assert (
        LEAKAGE_MODEL_METADATA.model_name
        ==
        "Full Gradient Boosting"
    )

    assert (
        LEAKAGE_MODEL_METADATA.model_version
        ==
        "1.0.0"
    )

    assert (
        LEAKAGE_MODEL_METADATA.model_type
        ==
        "classification"
    )

    assert (
        LEAKAGE_MODEL_METADATA.target_name
        ==
        "revenue_leakage_event"
    )