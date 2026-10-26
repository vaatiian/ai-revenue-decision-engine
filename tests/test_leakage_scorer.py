"""
Tests for the production Revenue Leakage scorer.
"""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.leakage_scorer import (
    LeakageScorer,
    get_positive_class_index,
    load_leakage_artifact,
    load_leakage_manifest,
    sha256_file,
    validate_leakage_artifact,
    validate_leakage_manifest,
)
from src.models.leakage_spec import (
    LEAKAGE_GRADIENT_BOOSTING_PARAMS,
    LEAKAGE_MODEL_FEATURES,
    LEAKAGE_MODEL_METADATA,
    build_frozen_leakage_pipeline,
)


# ================================================================
# SYNTHETIC TRAINING DATA
# ================================================================

def make_training_data(
    n_rows: int = 80,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Deterministic synthetic training data using the exact frozen
    45-feature Revenue Leakage contract.
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

    for feature in (
        LEAKAGE_MODEL_FEATURES
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

        elif feature == (
            "has_prior_completed_leakage_episode"
        ):

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

        elif feature == (
            "prior_leakage_rate"
        ):

            values = rng.random(
                n_rows
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
                LEAKAGE_MODEL_FEATURES
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
    Fit a valid model with the frozen feature order.
    """

    X, y = (
        make_training_data()
    )

    model = (
        build_frozen_leakage_pipeline()
    )

    model.fit(
        X,
        y,
    )

    return model


def make_fitted_model_with_wrong_input_order():
    """
    Create a genuine Pipeline.feature_names_in_ mismatch by fitting
    on reversed DataFrame columns.
    """

    X, y = (
        make_training_data()
    )

    reversed_columns = list(
        reversed(
            LEAKAGE_MODEL_FEATURES
        )
    )

    X = X[
        reversed_columns
    ]

    model = (
        build_frozen_leakage_pipeline()
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

    X, _ = (
        make_training_data(
            n_rows=4
        )
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
    Write a valid synthetic manifest.
    """

    manifest = {
        "model_name":
            LEAKAGE_MODEL_METADATA.model_name,

        "model_version":
            LEAKAGE_MODEL_METADATA.model_version,

        "model_type":
            LEAKAGE_MODEL_METADATA.model_type,

        "target_name":
            LEAKAGE_MODEL_METADATA.target_name,

        "feature_count":
            len(
                LEAKAGE_MODEL_FEATURES
            ),

        "features":
            list(
                LEAKAGE_MODEL_FEATURES
            ),

        "algorithm":
            "GradientBoostingClassifier",

        "hyperparameters":
            dict(
                LEAKAGE_GRADIENT_BOOSTING_PARAMS
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

        load_leakage_artifact(
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

        load_leakage_manifest(
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

    validate_leakage_artifact(
        model
    )


def test_unfitted_artifact_fails():

    model = (
        build_frozen_leakage_pipeline()
    )

    with pytest.raises(
        ValueError,
        match="feature_names_in_",
    ):

        validate_leakage_artifact(
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

        validate_leakage_artifact(
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

    validate_leakage_manifest(
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

        validate_leakage_manifest(
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

        validate_leakage_manifest(
            manifest,
            artifact_path=model_path,
        )


# ================================================================
# SCORER
# ================================================================

def test_leakage_scorer_loads(
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

    scorer = LeakageScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    assert scorer.model is not None

    assert (
        scorer.metadata
        ==
        LEAKAGE_MODEL_METADATA
    )


def test_leakage_scorer_output(
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

    scorer = LeakageScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
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
        "revenue_leakage_probability"
        in
        scores.columns
    )

    assert (
        scores[
            "revenue_leakage_probability"
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )


def test_scorer_preserves_customer_order(
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

    scorer = LeakageScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
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
        scores[
            "customer"
        ].tolist()
        ==
        features[
            "customer"
        ].tolist()
    )


def test_scorer_adds_governance_metadata(
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

    scorer = LeakageScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    scores = (
        scorer.score(
            make_scoring_features()
        )
    )

    assert (
        scores[
            "model_name"
        ]
        .eq(
            LEAKAGE_MODEL_METADATA.model_name
        )
        .all()
    )

    assert (
        scores[
            "model_version"
        ]
        .eq(
            LEAKAGE_MODEL_METADATA.model_version
        )
        .all()
    )

    assert (
        scores[
            "model_type"
        ]
        .eq(
            LEAKAGE_MODEL_METADATA.model_type
        )
        .all()
    )

    assert (
        scores[
            "target_name"
        ]
        .eq(
            LEAKAGE_MODEL_METADATA.target_name
        )
        .all()
    )

    assert (
        "scored_at_utc"
        in
        scores.columns
    )


def test_scorer_rejects_missing_feature(
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

    scorer = LeakageScorer(
        artifact_path=model_path,
        manifest_path=manifest_path,
    )

    features = (
        make_scoring_features()
        .drop(
            columns=[
                LEAKAGE_MODEL_FEATURES[
                    0
                ]
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required features",
    ):

        scorer.score(
            features
        )


def test_scorer_rejects_duplicate_customer(
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

    scorer = LeakageScorer(
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
        ValueError,
        match="duplicate entity keys",
    ):

        scorer.score(
            features
        )