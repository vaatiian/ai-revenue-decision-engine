import hashlib
import json

import pytest

from src.models.repeat_purchase_manifest import (
    DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH,
    DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH,
    load_and_validate_repeat_purchase_manifest,
    load_repeat_purchase_manifest,
    sha256_file,
    validate_repeat_purchase_manifest,
)
from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_CATEGORICAL_FEATURES,
    REPEAT_PURCHASE_FINAL_DECISION,
    REPEAT_PURCHASE_LOGISTIC_PARAMS,
    REPEAT_PURCHASE_MODEL_FEATURES,
    REPEAT_PURCHASE_MODEL_NAME,
    REPEAT_PURCHASE_MODEL_TYPE,
    REPEAT_PURCHASE_MODEL_VERSION,
    REPEAT_PURCHASE_NUMERIC_FEATURES,
    REPEAT_PURCHASE_TARGET_NAME,
)


def make_manifest():
    return {
        "model_name":
            REPEAT_PURCHASE_MODEL_NAME,

        "model_version":
            REPEAT_PURCHASE_MODEL_VERSION,

        "model_type":
            REPEAT_PURCHASE_MODEL_TYPE,

        "target_name":
            REPEAT_PURCHASE_TARGET_NAME,

        "prediction_horizon_days":
            90,

        "feature_count":
            18,

        "numeric_feature_count":
            17,

        "categorical_feature_count":
            1,

        "features":
            list(
                REPEAT_PURCHASE_MODEL_FEATURES
            ),

        "numeric_features":
            list(
                REPEAT_PURCHASE_NUMERIC_FEATURES
            ),

        "categorical_features":
            list(
                REPEAT_PURCHASE_CATEGORICAL_FEATURES
            ),

        "algorithm":
            "LogisticRegression",

        "hyperparameters":
            dict(
                REPEAT_PURCHASE_LOGISTIC_PARAMS
            ),

        "preprocessing": {
            "numeric_imputation":
                "median",

            "numeric_scaling":
                "StandardScaler",

            "categorical_imputation":
                "most_frequent",

            "onehot_handle_unknown":
                "ignore",

            "onehot_drop":
                "first",
        },

        "calibration":
            None,

        "governance_decision":
            REPEAT_PURCHASE_FINAL_DECISION,

        "test_set_consumed":
            True,

        "post_test_tuning_allowed":
            False,

        "artifact_filename":
            "repeat_purchase_logistic_v1.0.0.joblib",

        "source_notebook":
            "notebooks/02_ML_Repeat_Purchase_Model.ipynb",

        "source_notebook_sha256":
            "a" * 64,

        "model_artifact_sha256":
            "b" * 64,

        "sklearn_version":
            "test",
    }


def test_default_paths():
    assert (
        DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH.name
        ==
        "repeat_purchase_logistic_v1.0.0.joblib"
    )

    assert (
        DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH.name
        ==
        "repeat_purchase_logistic_v1.0.0.json"
    )


def test_sha256_file(tmp_path):
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


def test_sha256_missing_file_fails(tmp_path):
    with pytest.raises(
        FileNotFoundError
    ):
        sha256_file(
            tmp_path
            /
            "missing.bin"
        )


def test_load_manifest(tmp_path):
    path = (
        tmp_path
        /
        "manifest.json"
    )

    manifest = (
        make_manifest()
    )

    path.write_text(
        json.dumps(
            manifest
        ),
        encoding="utf-8",
    )

    loaded = (
        load_repeat_purchase_manifest(
            path
        )
    )

    assert (
        loaded
        ==
        manifest
    )


def test_missing_manifest_fails(tmp_path):
    with pytest.raises(
        FileNotFoundError
    ):
        load_repeat_purchase_manifest(
            tmp_path
            /
            "missing.json"
        )


def test_valid_manifest_passes():
    validate_repeat_purchase_manifest(
        make_manifest()
    )


def test_missing_required_field_fails():
    manifest = (
        make_manifest()
    )

    manifest.pop(
        "features"
    )

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_model_name_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "model_name"
    ] = "Wrong Model"

    with pytest.raises(
        ValueError,
        match="model_name",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_model_version_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "model_version"
    ] = "9.9.9"

    with pytest.raises(
        ValueError,
        match="model_version",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_feature_count_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "feature_count"
    ] = 99

    with pytest.raises(
        ValueError,
        match="feature count",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_feature_order_mismatch_fails():
    manifest = (
        make_manifest()
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
        validate_repeat_purchase_manifest(
            manifest
        )


def test_numeric_feature_order_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "numeric_features"
    ] = list(
        reversed(
            manifest[
                "numeric_features"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="numeric feature order",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_categorical_feature_order_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "categorical_features"
    ] = [
        "wrong_region"
    ]

    with pytest.raises(
        ValueError,
        match="categorical feature order",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_algorithm_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "algorithm"
    ] = "RandomForestClassifier"

    with pytest.raises(
        ValueError,
        match="algorithm",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_hyperparameter_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "hyperparameters"
    ][
        "max_iter"
    ] = 1

    with pytest.raises(
        ValueError,
        match="hyperparameter",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_preprocessing_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "preprocessing"
    ][
        "onehot_drop"
    ] = None

    with pytest.raises(
        ValueError,
        match="preprocessing",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_governance_decision_mismatch_fails():
    manifest = (
        make_manifest()
    )

    manifest[
        "governance_decision"
    ] = "GO"

    with pytest.raises(
        ValueError,
        match="governance decision",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_test_set_must_remain_consumed():
    manifest = (
        make_manifest()
    )

    manifest[
        "test_set_consumed"
    ] = False

    with pytest.raises(
        ValueError,
        match="consumed",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_post_test_tuning_must_be_false():
    manifest = (
        make_manifest()
    )

    manifest[
        "post_test_tuning_allowed"
    ] = True

    with pytest.raises(
        ValueError,
        match="post-test tuning",
    ):
        validate_repeat_purchase_manifest(
            manifest
        )


def test_artifact_hash_validation(tmp_path):
    artifact = (
        tmp_path
        /
        "repeat_purchase_logistic_v1.0.0.joblib"
    )

    artifact.write_bytes(
        b"artifact"
    )

    manifest = (
        make_manifest()
    )

    manifest[
        "model_artifact_sha256"
    ] = sha256_file(
        artifact
    )

    validate_repeat_purchase_manifest(
        manifest=manifest,
        artifact_path=artifact,
    )


def test_artifact_hash_mismatch_fails(tmp_path):
    artifact = (
        tmp_path
        /
        "repeat_purchase_logistic_v1.0.0.joblib"
    )

    artifact.write_bytes(
        b"artifact"
    )

    manifest = (
        make_manifest()
    )

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        validate_repeat_purchase_manifest(
            manifest=manifest,
            artifact_path=artifact,
        )


def test_real_exported_manifest_and_artifact_pass():
    manifest = (
        load_and_validate_repeat_purchase_manifest()
    )

    assert (
        manifest[
            "model_name"
        ]
        ==
        REPEAT_PURCHASE_MODEL_NAME
    )

    assert (
        manifest[
            "feature_count"
        ]
        ==
        18
    )

    assert (
        manifest[
            "governance_decision"
        ]
        ==
        "CONDITIONAL GO"
    )


def test_real_notebook_hash_is_locked():
    manifest = (
        load_repeat_purchase_manifest()
    )

    assert (
        manifest[
            "source_notebook_sha256"
        ]
        ==
        "68e528360f7d516e91ea115c76df2cb888035d55883bae41c00aee2d47fd57c9"
    )


def test_real_model_hash_is_locked():
    manifest = (
        load_repeat_purchase_manifest()
    )

    assert (
        manifest[
            "model_artifact_sha256"
        ]
        ==
        "901226110e49babaf7f87311b6ebc84ec847e447756786319acf5638cb2b47e1"
    )