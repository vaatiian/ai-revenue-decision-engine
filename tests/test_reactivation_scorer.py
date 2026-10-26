import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.reactivation_scorer import (
    ReactivationScorer,
    get_positive_class_index,
    load_reactivation_artifact,
    load_reactivation_manifest,
    sha256_file,
    validate_reactivation_artifact,
    validate_reactivation_manifest,
)
from src.models.reactivation_spec import (
    REACTIVATION_MODEL_FEATURES,
    REACTIVATION_MODEL_METADATA,
    REACTIVATION_RANDOM_FOREST_PARAMS,
    build_frozen_reactivation_pipeline,
)


# ================================================================
# SYNTHETIC TRAINING DATA
# ================================================================

def make_training_data(
    n_rows: int = 120,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build deterministic synthetic data using the exact frozen
    Reactivation feature register.
    """

    rng = np.random.default_rng(
        42
    )

    data = {}

    for feature in REACTIVATION_MODEL_FEATURES:

        if feature == "last_order_region":

            data[feature] = rng.choice(
                [
                    "UK",
                    "France",
                    "Germany",
                ],
                size=n_rows,
            )

        elif feature in {
            "has_prior_purchase_gap",
            "has_prior_inactivity_episode",
            "has_prior_successful_reactivation",
            "prediction_is_weekend",
        }:

            data[feature] = rng.integers(
                0,
                2,
                size=n_rows,
            )

        elif feature == "prediction_month":

            data[feature] = rng.integers(
                1,
                13,
                size=n_rows,
            )

        elif feature == "prediction_weekday":

            data[feature] = rng.integers(
                0,
                7,
                size=n_rows,
            )

        else:

            data[feature] = (
                rng.random(
                    n_rows
                )
                *
                100
                +
                1
            )

    X = pd.DataFrame(
        data
    )[
        list(
            REACTIVATION_MODEL_FEATURES
        )
    ]

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
    Fit a valid model using the exact frozen feature order.
    """

    X, y = make_training_data()

    model = (
        build_frozen_reactivation_pipeline()
    )

    model.fit(
        X,
        y,
    )

    return model


def make_fitted_model_with_wrong_input_order():
    """
    Fit the same frozen pipeline using a deliberately reversed
    DataFrame column order.

    This creates a genuine feature_names_in_ mismatch without
    mutating sklearn's read-only feature_names_in_ property.
    """

    X, y = make_training_data()

    reversed_columns = list(
        reversed(
            REACTIVATION_MODEL_FEATURES
        )
    )

    X = X[
        reversed_columns
    ]

    model = (
        build_frozen_reactivation_pipeline()
    )

    model.fit(
        X,
        y,
    )

    return model


def make_scoring_features():
    """
    Create a small production-style scoring population.
    """

    X, _ = make_training_data(
        n_rows=4
    )

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
    Write a valid synthetic governance manifest for a fitted artifact.
    """

    manifest = {
        "model_name":
            REACTIVATION_MODEL_METADATA.model_name,

        "model_version":
            REACTIVATION_MODEL_METADATA.model_version,

        "feature_count":
            len(
                REACTIVATION_MODEL_FEATURES
            ),

        "features":
            list(
                REACTIVATION_MODEL_FEATURES
            ),

        "algorithm":
            "RandomForestClassifier",

        "hyperparameters":
            dict(
                REACTIVATION_RANDOM_FOREST_PARAMS
            ),

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

        load_reactivation_artifact(
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

        load_reactivation_manifest(
            tmp_path
            /
            "missing.json"
        )


# ================================================================
# ARTIFACT VALIDATION
# ================================================================

def test_valid_fitted_artifact_passes():

    model = make_fitted_model()

    validate_reactivation_artifact(
        model
    )


def test_unfitted_artifact_fails():

    model = (
        build_frozen_reactivation_pipeline()
    )

    with pytest.raises(
        ValueError
    ):

        validate_reactivation_artifact(
            model
        )


def test_feature_order_mismatch_fails():
    """
    A fitted artifact whose recorded input schema is not in the exact
    frozen production order must fail closed.
    """

    model = (
        make_fitted_model_with_wrong_input_order()
    )

    assert list(
        model.feature_names_in_
    ) != list(
        REACTIVATION_MODEL_FEATURES
    )

    with pytest.raises(
        ValueError,
        match="feature schema",
    ):

        validate_reactivation_artifact(
            model
        )


# ================================================================
# POSITIVE CLASS
# ================================================================

def test_positive_class_index():

    model = make_fitted_model()

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


# ================================================================
# MANIFEST VALIDATION
# ================================================================

def test_valid_manifest_passes(
    tmp_path,
):

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    manifest = write_manifest(
        manifest_path,
        model_path,
    )

    validate_reactivation_manifest(
        manifest,
        artifact_path=model_path,
    )


def test_manifest_feature_order_mismatch_fails(
    tmp_path,
):

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    manifest = write_manifest(
        manifest_path,
        model_path,
    )

    manifest[
        "features"
    ] = manifest[
        "features"
    ][::-1]

    with pytest.raises(
        ValueError
    ):

        validate_reactivation_manifest(
            manifest
        )


def test_manifest_hash_mismatch_fails(
    tmp_path,
):

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    manifest = write_manifest(
        manifest_path,
        model_path,
    )

    manifest[
        "model_artifact_sha256"
    ] = (
        "0"
        *
        64
    )

    with pytest.raises(
        ValueError
    ):

        validate_reactivation_manifest(
            manifest,
            artifact_path=model_path,
        )


# ================================================================
# PRODUCTION SCORER
# ================================================================

def test_reactivation_scorer_loads(
    tmp_path,
):

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    write_manifest(
        manifest_path,
        model_path,
    )

    scorer = ReactivationScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    assert (
        scorer.model
        is not None
    )


def test_reactivation_scorer_output(
    tmp_path,
):

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    write_manifest(
        manifest_path,
        model_path,
    )

    scorer = ReactivationScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    result = scorer.score(
        make_scoring_features()
    )

    assert len(
        result
    ) == 4

    assert (
        "reactivation_probability"
        in result.columns
    )

    assert (
        result[
            "reactivation_probability"
        ]
        .between(
            0,
            1,
        )
        .all()
    )


def test_scorer_preserves_customer_order(
    tmp_path,
):

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    write_manifest(
        manifest_path,
        model_path,
    )

    scorer = ReactivationScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    features = (
        make_scoring_features()
    )

    result = scorer.score(
        features
    )

    assert list(
        result[
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

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    write_manifest(
        manifest_path,
        model_path,
    )

    scorer = ReactivationScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    result = scorer.score(
        make_scoring_features()
    )

    required = {
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "scored_at_utc",
    }

    assert required.issubset(
        set(
            result.columns
        )
    )


def test_scorer_rejects_missing_feature(
    tmp_path,
):

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    write_manifest(
        manifest_path,
        model_path,
    )

    scorer = ReactivationScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    features = (
        make_scoring_features()
        .drop(
            columns=[
                "historical_revenue",
            ]
        )
    )

    with pytest.raises(
        ValueError
    ):

        scorer.score(
            features
        )


def test_scorer_rejects_duplicate_customer(
    tmp_path,
):

    model = make_fitted_model()

    model_path = (
        tmp_path
        /
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    write_manifest(
        manifest_path,
        model_path,
    )

    scorer = ReactivationScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
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
        ValueError
    ):

        scorer.score(
            features
        )