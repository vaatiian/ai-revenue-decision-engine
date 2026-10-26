"""
Production scorer for the frozen Revenue Leakage model.

Responsibilities:
- load the trained sklearn artifact;
- validate artifact integrity;
- validate the frozen 45-feature contract;
- enforce exact feature order;
- validate the governance manifest;
- optionally verify the artifact SHA-256 hash;
- identify the positive classification class safely;
- generate Revenue Leakage probabilities;
- attach model-governance metadata;
- fail closed on incompatible artifacts or scoring inputs.

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.models.base import (
    BaseModelScorer,
    attach_score_metadata,
)
from src.models.leakage_spec import (
    LEAKAGE_GRADIENT_BOOSTING_PARAMS,
    LEAKAGE_MODEL_FEATURES,
    LEAKAGE_MODEL_METADATA,
)


# ================================================================
# DEFAULT ARTIFACT PATHS
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_LEAKAGE_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "leakage"
    / "revenue_leakage_full_gradient_boosting_v1.0.0.joblib"
)


DEFAULT_LEAKAGE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "leakage"
    / "revenue_leakage_full_gradient_boosting_v1.0.0.json"
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

def load_leakage_manifest(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load and minimally validate the Revenue Leakage manifest.
    """

    manifest_path = Path(
        path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Revenue Leakage model manifest not found: "
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
            "Revenue Leakage model manifest is not valid JSON."
        ) from exc

    required_fields = {
        "model_name",
        "model_version",
        "feature_count",
        "features",
        "algorithm",
        "hyperparameters",
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
            "Revenue Leakage manifest is missing required fields: "
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

def load_leakage_artifact(
    path: str | Path,
):
    """
    Load a serialized Revenue Leakage model artifact.
    """

    artifact_path = Path(
        path
    )

    if not artifact_path.exists():
        raise FileNotFoundError(
            "Revenue Leakage model artifact not found: "
            f"{artifact_path}"
        )

    try:

        model = joblib.load(
            artifact_path
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to load Revenue Leakage model artifact."
        ) from exc

    return model


# ================================================================
# ARTIFACT CONTRACT VALIDATION
# ================================================================

def validate_leakage_artifact(
    model,
) -> None:
    """
    Validate a loaded artifact against the exact frozen
    Revenue Leakage production contract.
    """

    if not isinstance(
        model,
        Pipeline,
    ):
        raise TypeError(
            "Revenue Leakage artifact is not a compatible "
            "sklearn Pipeline."
        )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "Revenue Leakage artifact does not expose "
            "predict_proba()."
        )

    if not hasattr(
        model,
        "feature_names_in_",
    ):
        raise ValueError(
            "Revenue Leakage artifact does not expose "
            "feature_names_in_; the artifact may be unfitted."
        )

    artifact_features = list(
        model.feature_names_in_
    )

    expected_features = list(
        LEAKAGE_MODEL_FEATURES
    )

    if artifact_features != expected_features:
        raise ValueError(
            "Revenue Leakage artifact feature schema does not "
            "match the frozen production feature register."
        )

    if "preprocessing" not in model.named_steps:
        raise ValueError(
            "Revenue Leakage artifact does not contain a "
            "'preprocessing' pipeline step."
        )

    if "model" not in model.named_steps:
        raise ValueError(
            "Revenue Leakage artifact does not contain a "
            "'model' pipeline step."
        )

    estimator = model.named_steps[
        "model"
    ]

    if not isinstance(
        estimator,
        GradientBoostingClassifier,
    ):
        raise TypeError(
            "Revenue Leakage artifact final estimator is not "
            "GradientBoostingClassifier."
        )

    for (
        parameter,
        expected_value,
    ) in (
        LEAKAGE_GRADIENT_BOOSTING_PARAMS.items()
    ):

        actual_value = getattr(
            estimator,
            parameter,
            None,
        )

        if actual_value != expected_value:
            raise ValueError(
                "Revenue Leakage artifact hyperparameter mismatch: "
                f"{parameter} expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    if not hasattr(
        estimator,
        "estimators_",
    ):
        raise ValueError(
            "Revenue Leakage Gradient Boosting model is not fitted."
        )

    estimators = (
        estimator.estimators_
    )

    if (
        estimators.shape[0]
        !=
        LEAKAGE_GRADIENT_BOOSTING_PARAMS[
            "n_estimators"
        ]
    ):
        raise ValueError(
            "Revenue Leakage Gradient Boosting fitted-estimator "
            "count does not match the frozen specification."
        )

    if not hasattr(
        estimator,
        "classes_",
    ):
        raise ValueError(
            "Revenue Leakage Gradient Boosting model does not "
            "contain fitted classification classes."
        )

    classes = np.asarray(
        estimator.classes_
    )

    if classes.ndim != 1:
        raise ValueError(
            "Revenue Leakage artifact contains an invalid "
            "classification class structure."
        )

    if len(
        classes
    ) != 2:
        raise ValueError(
            "Revenue Leakage model must be a binary classifier."
        )

    if 1 not in classes:
        raise ValueError(
            "Revenue Leakage model does not contain positive class 1."
        )


# ================================================================
# MANIFEST CONTRACT VALIDATION
# ================================================================

def validate_leakage_manifest(
    manifest: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> None:
    """
    Validate manifest metadata against the frozen production
    specification.

    If artifact_path is provided and the manifest contains
    model_artifact_sha256, the binary hash is also verified.
    """

    if (
        manifest[
            "model_name"
        ]
        !=
        LEAKAGE_MODEL_METADATA.model_name
    ):
        raise ValueError(
            "Revenue Leakage manifest model_name does not match "
            "the frozen production specification."
        )

    if (
        manifest[
            "model_version"
        ]
        !=
        LEAKAGE_MODEL_METADATA.model_version
    ):
        raise ValueError(
            "Revenue Leakage manifest model_version does not match "
            "the frozen production specification."
        )

    if (
        manifest[
            "feature_count"
        ]
        !=
        len(
            LEAKAGE_MODEL_FEATURES
        )
    ):
        raise ValueError(
            "Revenue Leakage manifest feature_count does not match "
            "the frozen production specification."
        )

    if (
        list(
            manifest[
                "features"
            ]
        )
        !=
        list(
            LEAKAGE_MODEL_FEATURES
        )
    ):
        raise ValueError(
            "Revenue Leakage manifest feature order does not match "
            "the frozen production specification."
        )

    if (
        manifest[
            "algorithm"
        ]
        !=
        "GradientBoostingClassifier"
    ):
        raise ValueError(
            "Revenue Leakage manifest algorithm does not match "
            "GradientBoostingClassifier."
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
            "Revenue Leakage manifest hyperparameters must be "
            "a dictionary."
        )

    for (
        parameter,
        expected_value,
    ) in (
        LEAKAGE_GRADIENT_BOOSTING_PARAMS.items()
    ):

        if parameter not in manifest_parameters:
            raise ValueError(
                "Revenue Leakage manifest is missing frozen "
                f"hyperparameter: {parameter}."
            )

        actual_value = (
            manifest_parameters[
                parameter
            ]
        )

        if actual_value != expected_value:
            raise ValueError(
                "Revenue Leakage manifest hyperparameter mismatch: "
                f"{parameter} expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    if (
        "target_name"
        in manifest
        and
        manifest[
            "target_name"
        ]
        !=
        LEAKAGE_MODEL_METADATA.target_name
    ):
        raise ValueError(
            "Revenue Leakage manifest target_name does not match "
            "the frozen production specification."
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
                "Revenue Leakage model artifact SHA-256 does not "
                "match the governance manifest."
            )


# ================================================================
# POSITIVE CLASS
# ================================================================

def get_positive_class_index(
    model,
) -> int:
    """
    Return the predict_proba column corresponding to class 1.

    Never assume that positive class is automatically column 1.
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
            "Revenue Leakage model does not expose fitted classes_."
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
            "Revenue Leakage model must contain exactly one "
            "positive class labelled 1."
        )

    return int(
        matches[0]
    )


# ================================================================
# PRODUCTION SCORER
# ================================================================

class LeakageScorer(
    BaseModelScorer
):
    """
    Production scorer for the frozen Revenue Leakage model.
    """

    metadata = (
        LEAKAGE_MODEL_METADATA
    )

    feature_names = tuple(
        LEAKAGE_MODEL_FEATURES
    )

    key_columns = (
        "customer",
    )

    def __init__(
        self,
        artifact_path: str | Path = (
            DEFAULT_LEAKAGE_ARTIFACT_PATH
        ),
        manifest_path: str | Path = (
            DEFAULT_LEAKAGE_MANIFEST_PATH
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
            load_leakage_artifact(
                self.artifact_path
            )
        )

        self.manifest = (
            load_leakage_manifest(
                self.manifest_path
            )
        )

        validate_leakage_artifact(
            self.model
        )

        validate_leakage_manifest(
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
        Generate governed Revenue Leakage probabilities.
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
                "Revenue Leakage predict_proba returned an "
                "invalid probability matrix."
            )

        if (
            probabilities.shape[0]
            !=
            len(
                features
            )
        ):
            raise ValueError(
                "Revenue Leakage probability row count does not "
                "match the scoring population."
            )

        if (
            self.positive_class_index
            >=
            probabilities.shape[1]
        ):
            raise ValueError(
                "Revenue Leakage positive-class index is outside "
                "the returned probability matrix."
            )

        positive_probabilities = (
            probabilities[
                :,
                self.positive_class_index,
            ]
        )

        if not np.isfinite(
            positive_probabilities
        ).all():
            raise ValueError(
                "Revenue Leakage probabilities contain non-finite "
                "values."
            )

        if (
            (
                positive_probabilities
                <
                0
            ).any()
            or
            (
                positive_probabilities
                >
                1
            ).any()
        ):
            raise ValueError(
                "Revenue Leakage probabilities must lie within "
                "[0, 1]."
            )

        scores = (
            features.loc[
                :,
                list(
                    self.key_columns
                ),
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        scores[
            "revenue_leakage_probability"
        ] = positive_probabilities

        scores = (
            attach_score_metadata(
                scores=scores,
                metadata=self.metadata,
            )
        )

        return scores