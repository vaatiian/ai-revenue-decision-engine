"""
Governance manifest loading and validation for the frozen
90-day Repeat Purchase model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_CALIBRATION,
    REPEAT_PURCHASE_CATEGORICAL_FEATURES,
    REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY,
    REPEAT_PURCHASE_FINAL_DECISION,
    REPEAT_PURCHASE_HORIZON_DAYS,
    REPEAT_PURCHASE_LOGISTIC_PARAMS,
    REPEAT_PURCHASE_MODEL_FEATURES,
    REPEAT_PURCHASE_MODEL_NAME,
    REPEAT_PURCHASE_MODEL_TYPE,
    REPEAT_PURCHASE_MODEL_VERSION,
    REPEAT_PURCHASE_NUMERIC_FEATURES,
    REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY,
    REPEAT_PURCHASE_NUMERIC_SCALER,
    REPEAT_PURCHASE_ONEHOT_DROP,
    REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN,
    REPEAT_PURCHASE_TARGET_NAME,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "models"
    /
    "repeat_purchase"
    /
    "repeat_purchase_logistic_v1.0.0.joblib"
)

DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "models"
    /
    "repeat_purchase"
    /
    "repeat_purchase_logistic_v1.0.0.json"
)


REPEAT_PURCHASE_REQUIRED_MANIFEST_FIELDS = {
    "model_name",
    "model_version",
    "model_type",
    "target_name",
    "prediction_horizon_days",
    "feature_count",
    "numeric_feature_count",
    "categorical_feature_count",
    "features",
    "numeric_features",
    "categorical_features",
    "algorithm",
    "hyperparameters",
    "preprocessing",
    "calibration",
    "governance_decision",
    "test_set_consumed",
    "post_test_tuning_allowed",
    "artifact_filename",
    "source_notebook",
    "source_notebook_sha256",
    "model_artifact_sha256",
    "sklearn_version",
}


def sha256_file(
    path: str | Path,
) -> str:
    """
    Return SHA-256 for a file.
    """

    file_path = Path(
        path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found for SHA-256 calculation: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_repeat_purchase_manifest(
    path: str | Path = DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH,
) -> dict[str, Any]:
    """
    Load the Repeat Purchase governance manifest.
    """

    manifest_path = Path(
        path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Repeat Purchase manifest not found: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if not isinstance(
        manifest,
        dict,
    ):
        raise TypeError(
            "Repeat Purchase manifest must contain a JSON object."
        )

    return manifest


def validate_repeat_purchase_manifest(
    manifest: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> None:
    """
    Validate manifest identity, feature contract, preprocessing,
    governance state and optional artifact SHA-256.
    """

    if not isinstance(
        manifest,
        dict,
    ):
        raise TypeError(
            "Repeat Purchase manifest must be a dictionary."
        )

    missing = (
        REPEAT_PURCHASE_REQUIRED_MANIFEST_FIELDS
        -
        set(
            manifest
        )
    )

    if missing:
        raise ValueError(
            "Repeat Purchase manifest is missing required fields: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    if (
        manifest[
            "model_name"
        ]
        !=
        REPEAT_PURCHASE_MODEL_NAME
    ):
        raise ValueError(
            "Repeat Purchase manifest model_name mismatch."
        )

    if (
        manifest[
            "model_version"
        ]
        !=
        REPEAT_PURCHASE_MODEL_VERSION
    ):
        raise ValueError(
            "Repeat Purchase manifest model_version mismatch."
        )

    if (
        manifest[
            "model_type"
        ]
        !=
        REPEAT_PURCHASE_MODEL_TYPE
    ):
        raise ValueError(
            "Repeat Purchase manifest model_type mismatch."
        )

    if (
        manifest[
            "target_name"
        ]
        !=
        REPEAT_PURCHASE_TARGET_NAME
    ):
        raise ValueError(
            "Repeat Purchase manifest target_name mismatch."
        )

    if (
        manifest[
            "prediction_horizon_days"
        ]
        !=
        REPEAT_PURCHASE_HORIZON_DAYS
    ):
        raise ValueError(
            "Repeat Purchase manifest horizon mismatch."
        )

    if (
        manifest[
            "feature_count"
        ]
        !=
        len(
            REPEAT_PURCHASE_MODEL_FEATURES
        )
    ):
        raise ValueError(
            "Repeat Purchase manifest feature count mismatch."
        )

    if (
        manifest[
            "numeric_feature_count"
        ]
        !=
        len(
            REPEAT_PURCHASE_NUMERIC_FEATURES
        )
    ):
        raise ValueError(
            "Repeat Purchase manifest numeric feature count mismatch."
        )

    if (
        manifest[
            "categorical_feature_count"
        ]
        !=
        len(
            REPEAT_PURCHASE_CATEGORICAL_FEATURES
        )
    ):
        raise ValueError(
            "Repeat Purchase manifest categorical feature count mismatch."
        )

    if (
        tuple(
            manifest[
                "features"
            ]
        )
        !=
        REPEAT_PURCHASE_MODEL_FEATURES
    ):
        raise ValueError(
            "Repeat Purchase manifest feature order mismatch."
        )

    if (
        tuple(
            manifest[
                "numeric_features"
            ]
        )
        !=
        REPEAT_PURCHASE_NUMERIC_FEATURES
    ):
        raise ValueError(
            "Repeat Purchase manifest numeric feature order mismatch."
        )

    if (
        tuple(
            manifest[
                "categorical_features"
            ]
        )
        !=
        REPEAT_PURCHASE_CATEGORICAL_FEATURES
    ):
        raise ValueError(
            "Repeat Purchase manifest categorical feature order mismatch."
        )

    if (
        manifest[
            "algorithm"
        ]
        !=
        "LogisticRegression"
    ):
        raise ValueError(
            "Repeat Purchase manifest algorithm mismatch."
        )

    if (
        manifest[
            "hyperparameters"
        ]
        !=
        REPEAT_PURCHASE_LOGISTIC_PARAMS
    ):
        raise ValueError(
            "Repeat Purchase manifest hyperparameter mismatch."
        )

    preprocessing = manifest[
        "preprocessing"
    ]

    if not isinstance(
        preprocessing,
        dict,
    ):
        raise TypeError(
            "Repeat Purchase preprocessing manifest must be a dictionary."
        )

    expected_preprocessing = {
        "numeric_imputation":
            REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY,

        "numeric_scaling":
            REPEAT_PURCHASE_NUMERIC_SCALER,

        "categorical_imputation":
            REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY,

        "onehot_handle_unknown":
            REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN,

        "onehot_drop":
            REPEAT_PURCHASE_ONEHOT_DROP,
    }

    if (
        preprocessing
        !=
        expected_preprocessing
    ):
        raise ValueError(
            "Repeat Purchase manifest preprocessing mismatch."
        )

    if (
        manifest[
            "calibration"
        ]
        !=
        REPEAT_PURCHASE_CALIBRATION
    ):
        raise ValueError(
            "Repeat Purchase manifest calibration mismatch."
        )

    if (
        manifest[
            "governance_decision"
        ]
        !=
        REPEAT_PURCHASE_FINAL_DECISION
    ):
        raise ValueError(
            "Repeat Purchase manifest governance decision mismatch."
        )

    if (
        manifest[
            "test_set_consumed"
        ]
        is not True
    ):
        raise ValueError(
            "Repeat Purchase final test set must remain marked consumed."
        )

    if (
        manifest[
            "post_test_tuning_allowed"
        ]
        is not False
    ):
        raise ValueError(
            "Repeat Purchase post-test tuning must remain prohibited."
        )

    if (
        manifest[
            "artifact_filename"
        ]
        !=
        DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH.name
    ):
        raise ValueError(
            "Repeat Purchase artifact filename mismatch."
        )

    notebook_hash = manifest[
        "source_notebook_sha256"
    ]

    model_hash = manifest[
        "model_artifact_sha256"
    ]

    if (
        not isinstance(
            notebook_hash,
            str,
        )
        or
        len(
            notebook_hash
        )
        !=
        64
    ):
        raise ValueError(
            "Repeat Purchase source notebook SHA-256 is invalid."
        )

    if (
        not isinstance(
            model_hash,
            str,
        )
        or
        len(
            model_hash
        )
        !=
        64
    ):
        raise ValueError(
            "Repeat Purchase model artifact SHA-256 is invalid."
        )

    if artifact_path is not None:
        actual_hash = (
            sha256_file(
                artifact_path
            )
        )

        if (
            actual_hash
            !=
            model_hash
        ):
            raise ValueError(
                "Repeat Purchase model artifact SHA-256 mismatch."
            )


def load_and_validate_repeat_purchase_manifest(
    manifest_path: str | Path = DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH,
    artifact_path: str | Path = DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH,
) -> dict[str, Any]:
    """
    Load and validate the manifest and corresponding artifact hash.
    """

    manifest = (
        load_repeat_purchase_manifest(
            manifest_path
        )
    )

    validate_repeat_purchase_manifest(
        manifest=manifest,
        artifact_path=artifact_path,
    )

    return manifest