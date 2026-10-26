from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.models.reactivation_spec import (
    REACTIVATION_CATEGORICAL_FEATURES,
    REACTIVATION_MODEL_FEATURES,
    REACTIVATION_MODEL_METADATA,
    REACTIVATION_NUMERIC_FEATURES,
    REACTIVATION_RANDOM_FOREST_PARAMS,
    build_frozen_reactivation_pipeline,
    build_reactivation_preprocessor,
)


# ================================================================
# FEATURE REGISTER
# ================================================================

def test_frozen_reactivation_feature_count():

    assert len(
        REACTIVATION_MODEL_FEATURES
    ) == 29


def test_frozen_numeric_feature_count():

    assert len(
        REACTIVATION_NUMERIC_FEATURES
    ) == 28


def test_frozen_categorical_feature_count():

    assert len(
        REACTIVATION_CATEGORICAL_FEATURES
    ) == 1


def test_last_order_region_is_final_feature():

    assert (
        REACTIVATION_MODEL_FEATURES[-1]
        ==
        "last_order_region"
    )


# ================================================================
# PREPROCESSING
# ================================================================

def test_reactivation_preprocessor_type():

    preprocessor = (
        build_reactivation_preprocessor()
    )

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )


def test_numeric_imputer_matches_frozen_spec():

    pipeline = (
        build_frozen_reactivation_pipeline()
    )

    preprocessing = pipeline.named_steps[
        "preprocessing"
    ]

    numeric_pipeline = next(
        transformer
        for (
            name,
            transformer,
            columns,
        ) in preprocessing.transformers
        if name == "numeric"
    )

    imputer = numeric_pipeline.named_steps[
        "imputer"
    ]

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

    pipeline = (
        build_frozen_reactivation_pipeline()
    )

    preprocessing = pipeline.named_steps[
        "preprocessing"
    ]

    categorical_pipeline = next(
        transformer
        for (
            name,
            transformer,
            columns,
        ) in preprocessing.transformers
        if name == "categorical"
    )

    imputer = categorical_pipeline.named_steps[
        "imputer"
    ]

    encoder = categorical_pipeline.named_steps[
        "onehot"
    ]

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
# RANDOM FOREST
# ================================================================

def test_random_forest_hyperparameters():

    pipeline = (
        build_frozen_reactivation_pipeline()
    )

    model = pipeline.named_steps[
        "model"
    ]

    assert isinstance(
        model,
        RandomForestClassifier,
    )

    assert model.n_estimators == 500
    assert model.max_depth == 8
    assert model.min_samples_leaf == 15
    assert model.max_features == "sqrt"
    assert model.random_state == 42
    assert model.n_jobs == -1


def test_parameter_register_matches_model():

    pipeline = (
        build_frozen_reactivation_pipeline()
    )

    model = pipeline.named_steps[
        "model"
    ]

    for (
        parameter,
        expected_value,
    ) in (
        REACTIVATION_RANDOM_FOREST_PARAMS.items()
    ):

        assert (
            getattr(
                model,
                parameter,
            )
            ==
            expected_value
        )


# ================================================================
# MODEL METADATA
# ================================================================

def test_reactivation_metadata():

    assert (
        REACTIVATION_MODEL_METADATA.model_name
        ==
        "Full Random Forest"
    )

    assert (
        REACTIVATION_MODEL_METADATA.model_version
        ==
        "1.0.0"
    )

    assert (
        REACTIVATION_MODEL_METADATA.model_type
        ==
        "classification"
    )

    assert (
        REACTIVATION_MODEL_METADATA.target_name
        ==
        "reactivation_within_next_90_days"
    )