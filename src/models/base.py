"""
Base production model scoring interface.

This module defines shared contracts for all production scorers.

Responsibilities:
- declare expected model features;
- validate feature availability;
- enforce feature order;
- detect duplicate scoring keys;
- attach model metadata;
- provide consistent score output structure;
- fail closed when model inputs are incompatible.

Model-specific prediction logic belongs in dedicated scorer modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

import pandas as pd


# ================================================================
# MODEL METADATA
# ================================================================

@dataclass(frozen=True)
class ModelMetadata:
    """
    Immutable metadata describing a production model.
    """

    model_name: str
    model_version: str
    model_type: str
    target_name: str


# ================================================================
# FEATURE VALIDATION
# ================================================================

def validate_required_features(
    features: pd.DataFrame,
    required_features: Sequence[str],
) -> None:
    """
    Fail closed when required model features are absent.
    """

    missing = [
        feature
        for feature in required_features
        if feature not in features.columns
    ]

    if missing:
        raise ValueError(
            "Model scoring input is missing required features: "
            + ", ".join(missing)
        )


def select_model_features(
    features: pd.DataFrame,
    required_features: Sequence[str],
) -> pd.DataFrame:
    """
    Validate and return model inputs in exact expected order.
    """

    validate_required_features(
        features,
        required_features,
    )

    return (
        features.loc[
            :,
            list(required_features),
        ]
        .copy()
    )


# ================================================================
# SCORING KEY VALIDATION
# ================================================================

def validate_unique_keys(
    features: pd.DataFrame,
    key_columns: Sequence[str],
) -> None:
    """
    Ensure the scoring population has unique entity keys.
    """

    missing_keys = [
        column
        for column in key_columns
        if column not in features.columns
    ]

    if missing_keys:
        raise ValueError(
            "Scoring input is missing key columns: "
            + ", ".join(missing_keys)
        )

    if features.duplicated(
        subset=list(key_columns)
    ).any():
        raise ValueError(
            "Scoring population contains duplicate entity keys: "
            + ", ".join(key_columns)
        )


# ================================================================
# SCORE METADATA
# ================================================================

def attach_score_metadata(
    scores: pd.DataFrame,
    metadata: ModelMetadata,
    scored_at: datetime | None = None,
) -> pd.DataFrame:
    """
    Attach standard model governance metadata to score outputs.
    """

    result = scores.copy()

    if scored_at is None:
        scored_at = datetime.now(
            timezone.utc
        )

    if scored_at.tzinfo is None:
        scored_at = scored_at.replace(
            tzinfo=timezone.utc
        )

    result[
        "model_name"
    ] = metadata.model_name

    result[
        "model_version"
    ] = metadata.model_version

    result[
        "model_type"
    ] = metadata.model_type

    result[
        "target_name"
    ] = metadata.target_name

    result[
        "scored_at_utc"
    ] = pd.Timestamp(
        scored_at
    )

    return result


# ================================================================
# BASE SCORER
# ================================================================

class BaseModelScorer(ABC):
    """
    Abstract production scoring interface.

    Every scorer must declare:
    - metadata
    - expected model features
    - entity key columns
    - scoring implementation
    """

    metadata: ModelMetadata
    feature_names: tuple[str, ...]
    key_columns: tuple[str, ...]

    def validate_input(
        self,
        features: pd.DataFrame,
    ) -> None:
        """
        Validate model input before prediction.
        """

        if features.empty:
            raise ValueError(
                "Cannot score an empty feature table."
            )

        validate_unique_keys(
            features,
            self.key_columns,
        )

        validate_required_features(
            features,
            self.feature_names,
        )

    def model_matrix(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return model features in exact expected order.
        """

        self.validate_input(
            features
        )

        return select_model_features(
            features,
            self.feature_names,
        )

    @abstractmethod
    def score(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score a validated feature table.
        """
        raise NotImplementedError