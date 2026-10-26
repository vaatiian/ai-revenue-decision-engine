"""
Production scorer for the frozen Future Customer Value model.

Responsibilities
----------------
- load the trained Stage A sklearn artifact;
- validate artifact integrity;
- validate the frozen 50-feature contract;
- enforce exact feature order;
- validate the governance manifest;
- optionally verify the artifact SHA-256 hash;
- identify the positive classification class safely;
- generate P(Future Active);
- apply the deterministic Stage B rule;
- calculate Expected FCV90;
- attach complete-model governance metadata;
- fail closed on incompatible artifacts or scoring inputs.

Architecture
------------
Stage A:
    P(FCV90 > 0)

Stage B:
    0.50 × h180_revenue

Complete score:
    Expected FCV90
    =
    P(Future Active)
    ×
    0.50
    ×
    h180_revenue

This module does not perform feature engineering or model training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.base import (
    BaseModelScorer,
    attach_score_metadata,
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
    calculate_expected_fcv90,
)


# ================================================================
# DEFAULT ARTIFACT PATHS
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_FCV_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "fcv"
    / "fcv_stage_a_full_random_forest_v1.0.0.joblib"
)


DEFAULT_FCV_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "fcv"
    / "fcv_stage_a_full_random_forest_v1.0.0.json"
)


# ================================================================
# HASHING
# ================================================================

def sha256_file(
    path: str | Path,
) -> str:
    """
    Return the SHA-256 hash of a file.
    """

    file_path = Path(
        path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file: {file_path}"
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


# ================================================================
# MANIFEST LOADING
# ================================================================

def load_fcv_manifest(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load and minimally validate the FCV governance manifest.
    """

    manifest_path = Path(
        path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "FCV model manifest not found: "
            f"{manifest_path}"
        )

    try:

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            manifest = json.load(
                handle
            )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "FCV model manifest is not valid JSON."
        ) from exc

    required_fields = {
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "feature_count",
        "features",
        "algorithm",
        "hyperparameters",
        "stage_b_model_name",
        "stage_b_formula",
        "stage_b_multiplier",
        "complete_model_name",
        "complete_model_formula",
        "research_final_decision",
    }

    missing = (
        required_fields
        -
        set(
            manifest
        )
    )

    if missing:
        raise ValueError(
            "FCV manifest is missing required fields: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    return manifest


# ================================================================
# ARTIFACT LOADING
# ================================================================

def load_fcv_artifact(
    path: str | Path,
):
    """
    Load the serialized FCV Stage A model artifact.
    """

    artifact_path = Path(
        path
    )

    if not artifact_path.exists():
        raise FileNotFoundError(
            "FCV model artifact not found: "
            f"{artifact_path}"
        )

    try:

        model = joblib.load(
            artifact_path
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to load FCV model artifact."
        ) from exc

    return model


# ================================================================
# ARTIFACT CONTRACT VALIDATION
# ================================================================

def validate_fcv_artifact(
    model,
) -> None:
    """
    Validate a loaded Stage A artifact against the exact frozen FCV
    production contract.
    """

    if not isinstance(
        model,
        Pipeline,
    ):
        raise TypeError(
            "FCV artifact is not a compatible sklearn Pipeline."
        )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "FCV artifact does not expose predict_proba()."
        )

    if not hasattr(
        model,
        "feature_names_in_",
    ):
        raise ValueError(
            "FCV artifact does not expose feature_names_in_; "
            "the artifact may be unfitted."
        )

    artifact_features = list(
        model.feature_names_in_
    )

    expected_features = list(
        FCV_MODEL_FEATURES
    )

    if artifact_features != expected_features:
        raise ValueError(
            "FCV artifact feature schema does not match the frozen "
            "production feature register."
        )

    if "preprocessing" not in model.named_steps:
        raise ValueError(
            "FCV artifact does not contain a 'preprocessing' "
            "pipeline step."
        )

    if "model" not in model.named_steps:
        raise ValueError(
            "FCV artifact does not contain a 'model' pipeline step."
        )

    estimator = model.named_steps[
        "model"
    ]

    if not isinstance(
        estimator,
        RandomForestClassifier,
    ):
        raise TypeError(
            "FCV artifact final estimator is not "
            "RandomForestClassifier."
        )

    for (
        parameter,
        expected_value,
    ) in (
        FCV_STAGE_A_RANDOM_FOREST_PARAMS.items()
    ):

        actual_value = getattr(
            estimator,
            parameter,
            None,
        )

        if actual_value != expected_value:
            raise ValueError(
                "FCV artifact hyperparameter mismatch: "
                f"{parameter} expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    if not hasattr(
        estimator,
        "estimators_",
    ):
        raise ValueError(
            "FCV Random Forest model is not fitted."
        )

    if (
        len(
            estimator.estimators_
        )
        !=
        FCV_STAGE_A_RANDOM_FOREST_PARAMS[
            "n_estimators"
        ]
    ):
        raise ValueError(
            "FCV Random Forest fitted-tree count does not match "
            "the frozen specification."
        )

    if not hasattr(
        estimator,
        "classes_",
    ):
        raise ValueError(
            "FCV Random Forest model does not contain fitted "
            "classification classes."
        )

    classes = np.asarray(
        estimator.classes_
    )

    if classes.ndim != 1:
        raise ValueError(
            "FCV artifact contains an invalid classification "
            "class structure."
        )

    if len(
        classes
    ) != 2:
        raise ValueError(
            "FCV Stage A model must be a binary classifier."
        )

    if 1 not in classes:
        raise ValueError(
            "FCV Stage A model does not contain positive class 1."
        )


# ================================================================
# MANIFEST CONTRACT VALIDATION
# ================================================================

def validate_fcv_manifest(
    manifest: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> None:
    """
    Validate manifest metadata against the frozen FCV production
    specification.

    The manifest describes the serialized Stage A classifier plus the
    deterministic Stage B and complete FCV architecture.

    If artifact_path is supplied and model_artifact_sha256 exists,
    the binary hash is verified.
    """

    if (
        manifest[
            "model_name"
        ]
        !=
        FCV_STAGE_A_MODEL_METADATA.model_name
    ):
        raise ValueError(
            "FCV manifest model_name does not match the frozen "
            "Stage A specification."
        )

    if (
        manifest[
            "model_version"
        ]
        !=
        FCV_STAGE_A_MODEL_METADATA.model_version
    ):
        raise ValueError(
            "FCV manifest model_version does not match the frozen "
            "Stage A specification."
        )

    if (
        manifest[
            "model_type"
        ]
        !=
        FCV_STAGE_A_MODEL_METADATA.model_type
    ):
        raise ValueError(
            "FCV manifest model_type does not match the frozen "
            "Stage A specification."
        )

    if (
        manifest[
            "target_name"
        ]
        !=
        FCV_STAGE_A_MODEL_METADATA.target_name
    ):
        raise ValueError(
            "FCV manifest target_name does not match the frozen "
            "Stage A specification."
        )

    if (
        manifest[
            "feature_count"
        ]
        !=
        len(
            FCV_MODEL_FEATURES
        )
    ):
        raise ValueError(
            "FCV manifest feature_count does not match the frozen "
            "production specification."
        )

    if (
        list(
            manifest[
                "features"
            ]
        )
        !=
        list(
            FCV_MODEL_FEATURES
        )
    ):
        raise ValueError(
            "FCV manifest feature order does not match the frozen "
            "production specification."
        )

    if (
        manifest[
            "algorithm"
        ]
        !=
        "RandomForestClassifier"
    ):
        raise ValueError(
            "FCV manifest algorithm does not match "
            "RandomForestClassifier."
        )

    manifest_parameters = (
        manifest[
            "hyperparameters"
        ]
    )

    if not isinstance(
        manifest_parameters,
        dict,
    ):
        raise ValueError(
            "FCV manifest hyperparameters must be a dictionary."
        )

    for (
        parameter,
        expected_value,
    ) in (
        FCV_STAGE_A_RANDOM_FOREST_PARAMS.items()
    ):

        if parameter not in manifest_parameters:
            raise ValueError(
                "FCV manifest is missing frozen hyperparameter: "
                f"{parameter}."
            )

        actual_value = (
            manifest_parameters[
                parameter
            ]
        )

        if actual_value != expected_value:
            raise ValueError(
                "FCV manifest hyperparameter mismatch: "
                f"{parameter} expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    if (
        manifest[
            "stage_b_model_name"
        ]
        !=
        FCV_STAGE_B_MODEL_NAME
    ):
        raise ValueError(
            "FCV manifest Stage B model name does not match "
            "the frozen specification."
        )

    if not np.isclose(
        manifest[
            "stage_b_multiplier"
        ],
        FCV_STAGE_B_RUN_RATE_MULTIPLIER,
    ):
        raise ValueError(
            "FCV manifest Stage B multiplier does not match "
            "the frozen specification."
        )

    if (
        manifest[
            "stage_b_formula"
        ]
        !=
        FCV_STAGE_B_FORMULA
    ):
        raise ValueError(
            "FCV manifest Stage B formula does not match the "
            "frozen specification."
        )

    if (
        manifest[
            "complete_model_name"
        ]
        !=
        FCV_COMPLETE_MODEL_NAME
    ):
        raise ValueError(
            "FCV manifest complete model name does not match "
            "the frozen specification."
        )

    if (
        manifest[
            "complete_model_formula"
        ]
        !=
        FCV_COMPLETE_FORMULA
    ):
        raise ValueError(
            "FCV manifest complete formula does not match the "
            "frozen specification."
        )

    if (
        manifest[
            "research_final_decision"
        ]
        !=
        FCV_FINAL_DECISION
    ):
        raise ValueError(
            "FCV manifest governance decision does not match "
            "the frozen specification."
        )

    if (
        artifact_path
        is not None
        and
        manifest.get(
            "model_artifact_sha256"
        )
    ):

        expected_hash = (
            manifest[
                "model_artifact_sha256"
            ]
        )

        actual_hash = (
            sha256_file(
                artifact_path
            )
        )

        if actual_hash != expected_hash:
            raise ValueError(
                "FCV model artifact SHA-256 does not match "
                "the governance manifest."
            )


# ================================================================
# POSITIVE CLASS
# ================================================================

def get_positive_class_index(
    model,
) -> int:
    """
    Return the predict_proba column corresponding to class 1.

    Never assume positive class is automatically probability column 1.
    """

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if (
        classes is None
        and
        hasattr(
            model,
            "named_steps",
        )
        and
        "model" in model.named_steps
    ):
        classes = getattr(
            model.named_steps[
                "model"
            ],
            "classes_",
            None,
        )

    if classes is None:
        raise ValueError(
            "FCV model does not expose fitted classes_."
        )

    classes = np.asarray(
        classes
    )

    matches = np.flatnonzero(
        classes
        ==
        1
    )

    if len(
        matches
    ) != 1:
        raise ValueError(
            "FCV model must contain exactly one positive "
            "class labelled 1."
        )

    return int(
        matches[0]
    )


# ================================================================
# PRODUCTION SCORER
# ================================================================

class FCVScorer(
    BaseModelScorer
):
    """
    Production scorer for the complete frozen FCV hurdle architecture.

    The serialized artifact supplies Stage A probabilities.

    Stage B and Expected FCV90 are deterministic calculations defined
    by the governed research specification.
    """

    metadata = (
        FCV_MODEL_METADATA
    )

    feature_names = tuple(
        FCV_MODEL_FEATURES
    )

    key_columns = (
        "customer",
    )

    def __init__(
        self,
        artifact_path: str | Path = (
            DEFAULT_FCV_ARTIFACT_PATH
        ),
        manifest_path: str | Path = (
            DEFAULT_FCV_MANIFEST_PATH
        ),
        verify_artifact_hash: bool = True,
    ) -> None:

        self.artifact_path = Path(
            artifact_path
        )

        self.manifest_path = Path(
            manifest_path
        )

        self.model = (
            load_fcv_artifact(
                self.artifact_path
            )
        )

        self.manifest = (
            load_fcv_manifest(
                self.manifest_path
            )
        )

        validate_fcv_artifact(
            self.model
        )

        validate_fcv_manifest(
            self.manifest,
            artifact_path=(
                self.artifact_path
                if verify_artifact_hash
                else None
            ),
        )

        self.positive_class_index = (
            get_positive_class_index(
                self.model
            )
        )

    def score(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate governed Future Customer Value scores.
        """

        model_matrix = (
            self.model_matrix(
                features
            )
        )

        probabilities = (
            self.model
            .predict_proba(
                model_matrix
            )
        )

        if not isinstance(
            probabilities,
            np.ndarray,
        ):
            probabilities = np.asarray(
                probabilities
            )

        if probabilities.ndim != 2:
            raise ValueError(
                "FCV model returned an invalid probability matrix."
            )

        if (
            self.positive_class_index
            >=
            probabilities.shape[1]
        ):
            raise ValueError(
                "FCV positive-class probability index is outside "
                "the returned probability matrix."
            )

        future_active_probability = (
            probabilities[
                :,
                self.positive_class_index,
            ]
        )

        if not np.isfinite(
            future_active_probability
        ).all():
            raise ValueError(
                "FCV model produced non-finite activity probabilities."
            )

        if (
            (
                future_active_probability
                <
                0
            ).any()
            or
            (
                future_active_probability
                >
                1
            ).any()
        ):
            raise ValueError(
                "FCV activity probabilities must lie between 0 and 1."
            )

        history_revenue = pd.to_numeric(
            features[
                "h180_revenue"
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(
            history_revenue
        ).all():
            raise ValueError(
                "FCV h180_revenue contains non-finite values."
            )

        if (
            history_revenue
            <
            0
        ).any():
            raise ValueError(
                "FCV h180_revenue cannot be negative for the "
                "eligible scoring population."
            )

        expected_fcv90 = (
            calculate_expected_fcv90(
                future_active_probability=(
                    future_active_probability
                ),
                h180_revenue=(
                    history_revenue
                ),
            )
        )

        if not np.isfinite(
            expected_fcv90
        ).all():
            raise ValueError(
                "FCV complete hurdle formula produced non-finite "
                "Expected FCV90 values."
            )

        scores = (
            features[
                list(
                    self.key_columns
                )
            ]
            .copy()
        )

        scores[
            "future_active_probability"
        ] = future_active_probability

        scores[
            "fcv_history_180d_revenue"
        ] = history_revenue

        scores[
            "future_customer_value_90d"
        ] = expected_fcv90

        return attach_score_metadata(
            scores,
            self.metadata,
        )