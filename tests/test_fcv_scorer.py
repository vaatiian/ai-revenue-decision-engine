"""
Tests for the production Future Customer Value scorer.
"""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.fcv_scorer import (
    FCVScorer,
    get_positive_class_index,
    load_fcv_artifact,
    load_fcv_manifest,
    sha256_file,
    validate_fcv_artifact,
    validate_fcv_manifest,
)
from src.models.fcv_spec import (
    FCV_COMPLETE_FORMULA,
    FCV_COMPLETE_MODEL_NAME,
    FCV_FINAL_DECISION,
    FCV_MODEL_FEATURES,
    FCV_MODEL_METADATA,
    FCV_STAGE_A_MODEL_METADATA,
    FCV_STAGE_A_RANDOM_FOREST_PARAMS,
    FCV_STAGE_B_FORMULA,
    FCV_STAGE_B_MODEL_NAME,
    FCV_STAGE_B_RUN_RATE_MULTIPLIER,
    build_frozen_fcv_stage_a_pipeline,
)


# ================================================================
# SYNTHETIC TRAINING DATA
# ================================================================

def make_training_data(
    n_rows: int = 100,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Deterministic synthetic training data using the exact frozen
    50-feature FCV Stage A contract.
    """

    if (
        n_rows
        %
        2
        !=
        0
    ):
        raise ValueError(
            "n_rows must be even for this deterministic test fixture."
        )

    rng = np.random.default_rng(
        42
    )

    data = {}

    binary_features = {
        "active_in_last_30d",
        "active_in_last_90d",
    }

    potentially_missing_features = {
        "r90_average_order_value",
        "p90_average_order_value",
        "r90_to_p90_revenue_ratio",
        "h180_average_gap_days",
        "h180_maximum_gap_days",
        "h180_minimum_gap_days",
    }

    for feature in (
        FCV_MODEL_FEATURES
    ):

        if feature == "last_region":

            data[
                feature
            ] = rng.choice(
                [
                    "UK",
                    "France",
                    "Germany",
                ],
                size=n_rows,
            )

        elif feature in binary_features:

            data[
                feature
            ] = rng.integers(
                0,
                2,
                size=n_rows,
            )

        elif feature in {
            "prediction_month_sin",
            "prediction_month_cos",
        }:

            data[
                feature
            ] = rng.uniform(
                -1.0,
                1.0,
                size=n_rows,
            )

        elif feature in potentially_missing_features:

            values = (
                rng.random(
                    n_rows
                )
                *
                100
                +
                1
            )

            values[
                ::10
            ] = np.nan

            data[
                feature
            ] = values

        else:

            data[
                feature
            ] = (
                rng.random(
                    n_rows
                )
                *
                100
                +
                1
            )

    X = (
        pd.DataFrame(
            data
        )[
            list(
                FCV_MODEL_FEATURES
            )
        ]
    )

    y = pd.Series(
        [
            0,
            1,
        ]
        *
        (
            n_rows
            //
            2
        ),
        name="target",
    )

    return X, y


def make_fitted_model():
    """
    Fit a valid Stage A model with the frozen feature order.
    """

    X, y = (
        make_training_data()
    )

    model = (
        build_frozen_fcv_stage_a_pipeline()
    )

    model.fit(
        X,
        y,
    )

    return model


def make_fitted_model_with_wrong_input_order():
    """
    Create a genuine Pipeline.feature_names_in_ mismatch.
    """

    X, y = (
        make_training_data()
    )

    reversed_columns = list(
        reversed(
            FCV_MODEL_FEATURES
        )
    )

    X = X[
        reversed_columns
    ]

    model = (
        build_frozen_fcv_stage_a_pipeline()
    )

    model.fit(
        X,
        y,
    )

    return model


def make_scoring_features():
    """
    Create a small production-style FCV scoring population.
    """

    X, _ = (
        make_training_data(
            n_rows=4
        )
    )

    X[
        "h180_revenue"
    ] = [
        1000.0,
        2000.0,
        500.0,
        1500.0,
    ]

    X.insert(
        0,
        "customer",
        [
            "A",
            "B",
            "C",
            "D",
        ],
    )

    return X


# ================================================================
# MANIFEST FIXTURE
# ================================================================

def write_manifest(
    path,
    model_path,
):
    """
    Write a valid synthetic FCV manifest.
    """

    manifest = {
        "model_name":
            FCV_STAGE_A_MODEL_METADATA.model_name,

        "model_version":
            FCV_STAGE_A_MODEL_METADATA.model_version,

        "model_type":
            FCV_STAGE_A_MODEL_METADATA.model_type,

        "target_name":
            FCV_STAGE_A_MODEL_METADATA.target_name,

        "feature_count":
            len(
                FCV_MODEL_FEATURES
            ),

        "features":
            list(
                FCV_MODEL_FEATURES
            ),

        "algorithm":
            "RandomForestClassifier",

        "hyperparameters":
            dict(
                FCV_STAGE_A_RANDOM_FOREST_PARAMS
            ),

        "stage_b_model_name":
            FCV_STAGE_B_MODEL_NAME,

        "stage_b_formula":
            FCV_STAGE_B_FORMULA,

        "stage_b_multiplier":
            FCV_STAGE_B_RUN_RATE_MULTIPLIER,

        "complete_model_name":
            FCV_COMPLETE_MODEL_NAME,

        "complete_model_formula":
            FCV_COMPLETE_FORMULA,

        "research_final_decision":
            FCV_FINAL_DECISION,

        "model_artifact_sha256":
            sha256_file(
                model_path
            ),
    }

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return manifest


# ================================================================
# BASIC LOADERS
# ================================================================

def test_sha256_file(
    tmp_path,
):

    path = (
        tmp_path
        /
        "sample.txt"
    )

    path.write_text(
        "hello",
        encoding="utf-8",
    )

    expected = hashlib.sha256(
        b"hello"
    ).hexdigest()

    assert (
        sha256_file(
            path
        )
        ==
        expected
    )


def test_missing_artifact_fails(
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError
    ):

        load_fcv_artifact(
            tmp_path
            /
            "missing.joblib"
        )


def test_missing_manifest_fails(
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError
    ):

        load_fcv_manifest(
            tmp_path
            /
            "missing.json"
        )


# ================================================================
# ARTIFACT VALIDATION
# ================================================================

def test_valid_fitted_artifact_passes():

    model = (
        make_fitted_model()
    )

    validate_fcv_artifact(
        model
    )


def test_unfitted_artifact_fails():

    model = (
        build_frozen_fcv_stage_a_pipeline()
    )

    with pytest.raises(
        ValueError,
        match="feature_names_in_",
    ):

        validate_fcv_artifact(
            model
        )


def test_feature_order_mismatch_fails():

    model = (
        make_fitted_model_with_wrong_input_order()
    )

    with pytest.raises(
        ValueError,
        match="feature schema",
    ):

        validate_fcv_artifact(
            model
        )


def test_positive_class_index():

    model = (
        make_fitted_model()
    )

    index = (
        get_positive_class_index(
            model
        )
    )

    assert (
        model.classes_[
            index
        ]
        ==
        1
    )


def test_random_forest_tree_count():

    model = (
        make_fitted_model()
    )

    validate_fcv_artifact(
        model
    )

    estimator = model.named_steps[
        "model"
    ]

    assert (
        len(
            estimator.estimators_
        )
        ==
        500
    )


# ================================================================
# MANIFEST VALIDATION
# ================================================================

def test_valid_manifest_passes(
    tmp_path,
):

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    joblib.dump(
        make_fitted_model(),
        model_path,
    )

    manifest = (
        write_manifest(
            manifest_path,
            model_path,
        )
    )

    validate_fcv_manifest(
        manifest,
        artifact_path=model_path,
    )


def test_manifest_feature_order_mismatch_fails(
    tmp_path,
):

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    joblib.dump(
        make_fitted_model(),
        model_path,
    )

    manifest = (
        write_manifest(
            manifest_path,
            model_path,
        )
    )

    manifest[
        "features"
    ] = list(
        reversed(
            manifest[
                "features"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="feature order",
    ):

        validate_fcv_manifest(
            manifest
        )


def test_manifest_hash_mismatch_fails(
    tmp_path,
):

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    joblib.dump(
        make_fitted_model(),
        model_path,
    )

    manifest = (
        write_manifest(
            manifest_path,
            model_path,
        )
    )

    manifest[
        "model_artifact_sha256"
    ] = "incorrect-hash"

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):

        validate_fcv_manifest(
            manifest,
            artifact_path=model_path,
        )


def test_manifest_stage_b_mismatch_fails(
    tmp_path,
):

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    joblib.dump(
        make_fitted_model(),
        model_path,
    )

    manifest = (
        write_manifest(
            manifest_path,
            model_path,
        )
    )

    manifest[
        "stage_b_multiplier"
    ] = 0.75

    with pytest.raises(
        ValueError,
        match="Stage B multiplier",
    ):

        validate_fcv_manifest(
            manifest
        )


def test_manifest_complete_formula_mismatch_fails(
    tmp_path,
):

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    joblib.dump(
        make_fitted_model(),
        model_path,
    )

    manifest = (
        write_manifest(
            manifest_path,
            model_path,
        )
    )

    manifest[
        "complete_model_formula"
    ] = "incorrect formula"

    with pytest.raises(
        ValueError,
        match="complete formula",
    ):

        validate_fcv_manifest(
            manifest
        )


# ================================================================
# SCORER
# ================================================================

def build_test_scorer(
    tmp_path,
):

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    joblib.dump(
        make_fitted_model(),
        model_path,
    )

    write_manifest(
        manifest_path,
        model_path,
    )

    scorer = FCVScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    return scorer


def test_fcv_scorer_loads(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    assert scorer.model is not None

    assert (
        scorer.metadata
        ==
        FCV_MODEL_METADATA
    )


def test_fcv_scorer_output(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    features = (
        make_scoring_features()
    )

    scores = (
        scorer.score(
            features
        )
    )

    assert (
        len(
            scores
        )
        ==
        len(
            features
        )
    )

    assert (
        "future_active_probability"
        in
        scores.columns
    )

    assert (
        "fcv_history_180d_revenue"
        in
        scores.columns
    )

    assert (
        "future_customer_value_90d"
        in
        scores.columns
    )


def test_probability_range(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    scores = scorer.score(
        make_scoring_features()
    )

    assert (
        scores[
            "future_active_probability"
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )


def test_complete_hurdle_formula(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    scores = scorer.score(
        make_scoring_features()
    )

    expected = (
        scores[
            "future_active_probability"
        ]
        *
        0.50
        *
        scores[
            "fcv_history_180d_revenue"
        ]
    )

    np.testing.assert_allclose(
        scores[
            "future_customer_value_90d"
        ],
        expected,
    )


def test_scorer_preserves_customer_order(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    features = (
        make_scoring_features()
    )

    scores = scorer.score(
        features
    )

    assert list(
        scores[
            "customer"
        ]
    ) == list(
        features[
            "customer"
        ]
    )


def test_scorer_adds_governance_metadata(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    scores = scorer.score(
        make_scoring_features()
    )

    assert (
        "model_name"
        in
        scores.columns
    )

    assert (
        "model_version"
        in
        scores.columns
    )

    assert (
        "model_type"
        in
        scores.columns
    )

    assert (
        "target_name"
        in
        scores.columns
    )

    assert (
        "scored_at_utc"
        in
        scores.columns
    )

    assert (
        scores[
            "model_name"
        ]
        .eq(
            FCV_MODEL_METADATA.model_name
        )
        .all()
    )

    assert (
        scores[
            "model_version"
        ]
        .eq(
            FCV_MODEL_METADATA.model_version
        )
        .all()
    )

    assert (
        scores[
            "model_type"
        ]
        .eq(
            FCV_MODEL_METADATA.model_type
        )
        .all()
    )

    assert (
        scores[
            "target_name"
        ]
        .eq(
            FCV_MODEL_METADATA.target_name
        )
        .all()
    )

    assert (
        scores[
            "scored_at_utc"
        ]
        .notna()
        .all()
    )


def test_scorer_rejects_missing_feature(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    features = (
        make_scoring_features()
        .drop(
            columns=[
                "h180_orders",
            ]
        )
    )

    with pytest.raises(
        ValueError,
    ):

        scorer.score(
            features
        )


def test_scorer_rejects_duplicate_customer(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    features = (
        make_scoring_features()
    )

    features.loc[
        1,
        "customer"
    ] = features.loc[
        0,
        "customer"
    ]

    with pytest.raises(
        ValueError,
    ):

        scorer.score(
            features
        )


def test_scorer_rejects_negative_h180_revenue(
    tmp_path,
):

    scorer = (
        build_test_scorer(
            tmp_path
        )
    )

    features = (
        make_scoring_features()
    )

    features.loc[
        0,
        "h180_revenue"
    ] = -1.0

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):

        scorer.score(
            features
        )