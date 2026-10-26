"""
Production scorer for the frozen customer-reactivation model.

Responsibilities:
- load the trained sklearn artifact;
- validate artifact integrity;
- validate the frozen 29-feature contract;
- enforce exact feature order;
- identify the positive classification class safely;
- generate Reactivation probabilities;
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

from src.models.base import (
    BaseModelScorer,
    attach_score_metadata,
)
from src.models.reactivation_spec import (
    REACTIVATION_MODEL_FEATURES,
    REACTIVATION_MODEL_METADATA,
    REACTIVATION_RANDOM_FOREST_PARAMS,
)


# ================================================================
# DEFAULT ARTIFACT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REACTIVATION_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "reactivation"
    / "reactivation_full_random_forest_v1.0.0.joblib"
)

DEFAULT_REACTIVATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "reactivation"
    / "reactivation_full_random_forest_v1.0.0.json"
)


# ================================================================
# HASHING
# ================================================================

def sha256_file(
    path: Path,
) -> str:
    """
    Return the SHA-256 hash of a file.
    """

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ================================================================
# MANIFEST LOADING
# ================================================================

def load_reactivation_manifest(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load and minimally validate the Reactivation artifact manifest.
    """

    manifest_path = Path(path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Reactivation model manifest not found: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        manifest = json.load(handle)

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
        set(manifest)
    )

    if missing:
        raise ValueError(
            "Reactivation manifest is missing required fields: "
            +
            ", ".join(
                sorted(missing)
            )
        )

    return manifest


# ================================================================
# ARTIFACT LOADING
# ================================================================

def load_reactivation_artifact(
    path: str | Path,
):
    """
    Load a serialized Reactivation model artifact.
    """

    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Reactivation model artifact not found: "
            f"{artifact_path}"
        )

    try:
        model = joblib.load(
            artifact_path
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to load Reactivation model artifact."
        ) from exc

    return model


# ================================================================
# ARTIFACT CONTRACT VALIDATION
# ================================================================

def validate_reactivation_artifact(
    model,
) -> None:
    """
    Validate that a loaded model matches the frozen production contract.
    """

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "Reactivation artifact does not expose predict_proba()."
        )

    if not hasattr(
        model,
        "feature_names_in_",
    ):
        raise ValueError(
            "Reactivation artifact does not expose feature_names_in_."
        )

    artifact_features = list(
        model.feature_names_in_
    )

    expected_features = list(
        REACTIVATION_MODEL_FEATURES
    )

    if artifact_features != expected_features:
        raise ValueError(
            "Reactivation artifact feature schema does not match "
            "the frozen production feature register."
        )

    if not hasattr(
        model,
        "named_steps",
    ):
        raise TypeError(
            "Reactivation artifact is not a compatible sklearn Pipeline."
        )

    if "model" not in model.named_steps:
        raise ValueError(
            "Reactivation artifact does not contain a 'model' pipeline step."
        )

    estimator = model.named_steps[
        "model"
    ]

    for (
        parameter,
        expected_value,
    ) in (
        REACTIVATION_RANDOM_FOREST_PARAMS.items()
    ):

        actual_value = getattr(
            estimator,
            parameter,
            None,
        )

        if actual_value != expected_value:
            raise ValueError(
                "Reactivation artifact hyperparameter mismatch: "
                f"{parameter} expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    if not hasattr(
        estimator,
        "estimators_",
    ):
        raise ValueError(
            "Reactivation Random Forest is not fitted."
        )

    if (
        len(
            estimator.estimators_
        )
        !=
        REACTIVATION_RANDOM_FOREST_PARAMS[
            "n_estimators"
        ]
    ):
        raise ValueError(
            "Reactivation Random Forest fitted-tree count "
            "does not match the frozen specification."
        )


# ================================================================
# MANIFEST CONTRACT VALIDATION
# ================================================================

def validate_reactivation_manifest(
    manifest: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> None:
    """
    Validate manifest metadata against the frozen production specification.

    If artifact_path is provided and the manifest contains an artifact hash,
    the binary hash is also verified.
    """

    if (
        manifest[
            "model_name"
        ]
        !=
        REACTIVATION_MODEL_METADATA.model_name
    ):
        raise ValueError(
            "Reactivation manifest model_name does not match "
            "the frozen production specification."
        )

    if (
        manifest[
            "model_version"
        ]
        !=
        REACTIVATION_MODEL_METADATA.model_version
    ):
        raise ValueError(
            "Reactivation manifest model_version does not match "
            "the frozen production specification."
        )

    if (
        manifest[
            "feature_count"
        ]
        !=
        len(
            REACTIVATION_MODEL_FEATURES
        )
    ):
        raise ValueError(
            "Reactivation manifest feature_count does not match "
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
            REACTIVATION_MODEL_FEATURES
        )
    ):
        raise ValueError(
            "Reactivation manifest feature order does not match "
            "the frozen production specification."
        )

    hyperparameters = manifest[
        "hyperparameters"
    ]

    for (
        parameter,
        expected_value,
    ) in (
        REACTIVATION_RANDOM_FOREST_PARAMS.items()
    ):

        actual_value = (
            hyperparameters.get(
                parameter
            )
        )

        if actual_value != expected_value:
            raise ValueError(
                "Reactivation manifest hyperparameter mismatch: "
                f"{parameter} expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    expected_hash = manifest.get(
        "model_artifact_sha256"
    )

    if (
        artifact_path is not None
        and
        expected_hash
    ):

        actual_hash = sha256_file(
            Path(
                artifact_path
            )
        )

        if actual_hash != expected_hash:
            raise ValueError(
                "Reactivation artifact SHA-256 does not match "
                "the governance manifest."
            )


# ================================================================
# POSITIVE CLASS
# ================================================================

def get_positive_class_index(
    model,
) -> int:
    """
    Find the probability-column index corresponding to class 1.
    """

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None:
        raise ValueError(
            "Reactivation model does not expose fitted classes_."
        )

    classes_array = np.asarray(
        classes
    )

    positive_positions = np.where(
        classes_array
        ==
        1
    )[0]

    if len(
        positive_positions
    ) != 1:
        raise ValueError(
            "Reactivation model must contain exactly one positive class 1."
        )

    return int(
        positive_positions[0]
    )


# ================================================================
# PRODUCTION SCORER
# ================================================================

class ReactivationScorer(
    BaseModelScorer
):
    """
    Production scoring interface for customer Reactivation propensity.
    """

    metadata = (
        REACTIVATION_MODEL_METADATA
    )

    feature_names = tuple(
        REACTIVATION_MODEL_FEATURES
    )

    key_columns = (
        "customer",
    )

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_REACTIVATION_ARTIFACT_PATH,
        manifest_path: str | Path | None = DEFAULT_REACTIVATION_MANIFEST_PATH,
        verify_artifact_hash: bool = True,
    ) -> None:

        self.artifact_path = Path(
            artifact_path
        )

        self.manifest_path = (
            Path(
                manifest_path
            )
            if manifest_path is not None
            else None
        )

        self.model = (
            load_reactivation_artifact(
                self.artifact_path
            )
        )

        validate_reactivation_artifact(
            self.model
        )

        self.manifest = None

        if self.manifest_path is not None:

            self.manifest = (
                load_reactivation_manifest(
                    self.manifest_path
                )
            )

            validate_reactivation_manifest(
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
        Score customer Reactivation probability.

        Returns one row per customer containing:
        - customer
        - reactivation_probability
        - standard model metadata
        """

        model_input = (
            self.model_matrix(
                features
            )
        )

        probabilities = (
            self.model.predict_proba(
                model_input
            )
        )

        if probabilities.ndim != 2:
            raise RuntimeError(
                "Reactivation predict_proba() returned "
                "an invalid array shape."
            )

        if (
            probabilities.shape[0]
            !=
            len(
                features
            )
        ):
            raise RuntimeError(
                "Reactivation prediction count does not match "
                "the scoring population."
            )

        if (
            self.positive_class_index
            >=
            probabilities.shape[1]
        ):
            raise RuntimeError(
                "Positive-class probability column is unavailable."
            )

        positive_probabilities = (
            probabilities[
                :,
                self.positive_class_index
            ]
        )

        if not np.isfinite(
            positive_probabilities
        ).all():
            raise RuntimeError(
                "Reactivation model produced non-finite probabilities."
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
            raise RuntimeError(
                "Reactivation model produced probabilities "
                "outside [0, 1]."
            )

        result = pd.DataFrame(
            {
                "customer":
                    features[
                        "customer"
                    ].to_numpy(),

                "reactivation_probability":
                    positive_probabilities,
            }
        )

        return attach_score_metadata(
            result,
            self.metadata,
        )