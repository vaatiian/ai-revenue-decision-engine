"""
Production scorer for the frozen Notebook 02
90-day Repeat Purchase model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.features.repeat_purchase_features import (
    build_repeat_purchase_scoring_features,
    select_repeat_purchase_model_features,
)
from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_FINAL_DECISION,
    REPEAT_PURCHASE_MODEL_NAME,
    REPEAT_PURCHASE_MODEL_TYPE,
    REPEAT_PURCHASE_MODEL_VERSION,
    REPEAT_PURCHASE_TARGET_NAME,
)


REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS = (
    "customer",
    "first_invoice",
    "first_order_date",
    "repeat_purchase_score",
    "model_name",
    "model_version",
    "model_type",
    "target_name",
    "governance_decision",
    "scored_at_utc",
)


DEFAULT_REPEAT_PURCHASE_MODEL_PATH = Path(
    "artifacts/models/repeat_purchase/"
    "repeat_purchase_logistic_v1.0.0.joblib"
)


def load_repeat_purchase_model(
    model_path: str | Path = DEFAULT_REPEAT_PURCHASE_MODEL_PATH,
):
    """
    Load the frozen fitted Notebook 02-compatible model artifact.
    """

    path = Path(
        model_path
    )

    if not path.exists():
        raise FileNotFoundError(
            "Repeat Purchase model artifact not found: "
            f"{path}"
        )

    model = joblib.load(
        path
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "Repeat Purchase artifact must expose predict_proba()."
        )

    return model


def score_repeat_purchase_features(
    features: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Score a pre-built Repeat Purchase feature table.
    """

    required_metadata = (
        "customer",
        "first_invoice",
        "first_order_date",
    )

    missing_metadata = [
        column
        for column in required_metadata
        if column not in features.columns
    ]

    if missing_metadata:
        raise KeyError(
            "Repeat Purchase scoring metadata columns are missing: "
            f"{missing_metadata}"
        )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "Repeat Purchase model must expose predict_proba()."
        )

    if features.empty:
        return pd.DataFrame(
            columns=REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS
        )

    model_matrix = (
        select_repeat_purchase_model_features(
            features
        )
    )

    probabilities = (
        model.predict_proba(
            model_matrix
        )[
            :,
            1
        ]
    )

    if len(
        probabilities
    ) != len(
        features
    ):
        raise RuntimeError(
            "Repeat Purchase score length does not match feature rows."
        )

    if (
        (probabilities < 0).any()
        or
        (probabilities > 1).any()
    ):
        raise RuntimeError(
            "Repeat Purchase probabilities must lie within [0, 1]."
        )

    scored_at_utc = datetime.now(
        timezone.utc
    )

    output = (
        features.loc[
            :,
            list(
                required_metadata
            ),
        ]
        .copy()
    )

    output[
        "repeat_purchase_score"
    ] = probabilities

    output[
        "model_name"
    ] = REPEAT_PURCHASE_MODEL_NAME

    output[
        "model_version"
    ] = REPEAT_PURCHASE_MODEL_VERSION

    output[
        "model_type"
    ] = REPEAT_PURCHASE_MODEL_TYPE

    output[
        "target_name"
    ] = REPEAT_PURCHASE_TARGET_NAME

    output[
        "governance_decision"
    ] = REPEAT_PURCHASE_FINAL_DECISION

    output[
        "scored_at_utc"
    ] = scored_at_utc

    return (
        output.loc[
            :,
            list(
                REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS
            ),
        ]
        .reset_index(
            drop=True
        )
    )


def score_repeat_purchase(
    data: pd.DataFrame,
    model,
    modeling_end_date=None,
    exclude_opening_month: bool = True,
) -> pd.DataFrame:
    """
    Build first-order features and score Repeat Purchase propensity.
    """

    kwargs = {
        "data": data,
        "exclude_opening_month": exclude_opening_month,
    }

    if modeling_end_date is not None:
        kwargs[
            "modeling_end_date"
        ] = modeling_end_date

    features = (
        build_repeat_purchase_scoring_features(
            **kwargs
        )
    )

    return (
        score_repeat_purchase_features(
            features=features,
            model=model,
        )
    )


class RepeatPurchaseScorer:
    """
    Thin production wrapper around the frozen Repeat Purchase artifact.
    """

    def __init__(
        self,
        model=None,
        model_path: str | Path = DEFAULT_REPEAT_PURCHASE_MODEL_PATH,
    ):
        if model is None:
            model = (
                load_repeat_purchase_model(
                    model_path
                )
            )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                "Repeat Purchase model must expose predict_proba()."
            )

        self.model = model

    def score_features(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        return (
            score_repeat_purchase_features(
                features=features,
                model=self.model,
            )
        )

    def score(
        self,
        data: pd.DataFrame,
        modeling_end_date=None,
        exclude_opening_month: bool = True,
    ) -> pd.DataFrame:
        return (
            score_repeat_purchase(
                data=data,
                model=self.model,
                modeling_end_date=modeling_end_date,
                exclude_opening_month=exclude_opening_month,
            )
        )