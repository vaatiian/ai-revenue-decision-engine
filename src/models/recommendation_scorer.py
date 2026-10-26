"""
Production scorer for the frozen Next Best Product recommendation
architecture.

This scorer exposes the governed deterministic recommendation engine as a
production scoring interface.

Frozen channels
---------------
1. Customer Repeat Frequency
2. Item-to-Item Affinity
3. Optional 70% Repeat / 30% Discovery presentation list

Unlike the sklearn-based production scorers, this model does not consume a
fixed numeric feature matrix and does not load a fitted binary artifact.

Its governed inputs are:
- canonical transaction history;
- explicit customer prediction episodes.

The deterministic recommendation algorithms are implemented in
src.models.recommendation_engine.

Responsibilities
----------------
- validate canonical recommendation transactions;
- validate customer prediction episodes;
- enforce unique customer/invoice prediction keys;
- enforce prediction-month consistency;
- build the qualifying transaction history;
- execute the frozen repeat/discovery/hybrid rankers;
- expose convenient Top-1 and recommendation-count fields;
- attach standard production model metadata;
- fail closed on incompatible scoring inputs.

This module does not train, tune or alter the recommendation architecture.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.models.base import (
    attach_score_metadata,
    validate_unique_keys,
)
from src.models.recommendation_engine import (
    build_recommendation_lines,
    generate_recommendations_for_episodes,
    validate_recommendation_transactions,
)
from src.models.recommendation_spec import (
    RECOMMENDATION_DEFAULT_TOP_N,
    RECOMMENDATION_DISCOVERY_MODEL_NAME,
    RECOMMENDATION_HYBRID_ROLE,
    RECOMMENDATION_MODEL_METADATA,
    RECOMMENDATION_OPTIONAL_HYBRID_NAME,
    RECOMMENDATION_REPEAT_MODEL_NAME,
    validate_recommendation_depth,
)


# ================================================================
# PRODUCTION EPISODE CONTRACT
# ================================================================

RECOMMENDATION_EPISODE_REQUIRED_COLUMNS = (
    "customer",
    "invoice",
    "order_date",
    "prediction_period",
)


RECOMMENDATION_EPISODE_KEY_COLUMNS = (
    "customer",
    "invoice",
)


# ================================================================
# OUTPUT CONTRACT
# ================================================================

RECOMMENDATION_SCORE_COLUMNS = (
    "customer",
    "invoice",
    "prediction_date",
    "prediction_period",

    "repeat_recommendations",
    "repeat_top1_product",
    "repeat_recommendation_count",

    "discovery_recommendations",
    "discovery_top1_product",
    "discovery_recommendation_count",

    "hybrid_recommendations",
    "hybrid_top1_product",
    "hybrid_recommendation_count",

    "repeat_model_name",
    "discovery_model_name",
    "hybrid_model_name",
    "hybrid_role",

    "model_name",
    "model_version",
    "model_type",
    "target_name",
    "scored_at_utc",
)


# ================================================================
# EPISODE VALIDATION
# ================================================================

def validate_recommendation_episodes(
    episodes: pd.DataFrame,
) -> None:
    """
    Validate the production recommendation episode contract.
    """

    if episodes.empty:
        raise ValueError(
            "Cannot score an empty recommendation episode table."
        )

    missing = [
        column
        for column
        in RECOMMENDATION_EPISODE_REQUIRED_COLUMNS
        if column not in episodes.columns
    ]

    if missing:
        raise ValueError(
            "Recommendation episodes are missing required columns: "
            + ", ".join(
                missing
            )
        )

    null_key_columns = [
        column
        for column
        in RECOMMENDATION_EPISODE_KEY_COLUMNS
        if episodes[
            column
        ].isna().any()
    ]

    if null_key_columns:
        raise ValueError(
            "Recommendation episode keys contain missing values: "
            + ", ".join(
                null_key_columns
            )
        )

    if episodes[
        "order_date"
    ].isna().any():
        raise ValueError(
            "Recommendation episodes contain missing order_date values."
        )

    if episodes[
        "prediction_period"
    ].isna().any():
        raise ValueError(
            "Recommendation episodes contain missing prediction_period values."
        )

    validate_unique_keys(
        episodes,
        RECOMMENDATION_EPISODE_KEY_COLUMNS,
    )


# ================================================================
# EPISODE NORMALISATION
# ================================================================

def normalise_recommendation_episodes(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalise recommendation episode dates and periods.

    prediction_period must correspond exactly to the month containing
    order_date.
    """

    validate_recommendation_episodes(
        episodes
    )

    result = episodes.copy()

    try:
        result[
            "order_date"
        ] = pd.to_datetime(
            result[
                "order_date"
            ],
            format="mixed",
        )

    except Exception as exc:
        raise ValueError(
            "Recommendation episode order_date values are invalid."
        ) from exc

    try:
        result[
            "prediction_period"
        ] = result[
            "prediction_period"
        ].map(
            lambda value:
                pd.Period(
                    value,
                    freq="M",
                )
        )

    except Exception as exc:
        raise ValueError(
            "Recommendation episode prediction_period values are invalid."
        ) from exc

    expected_period = (
        result[
            "order_date"
        ]
        .dt.to_period(
            "M"
        )
    )

    inconsistent = (
        result[
            "prediction_period"
        ]
        !=
        expected_period
    )

    if inconsistent.any():
        raise ValueError(
            "Recommendation prediction_period does not match "
            "the month of order_date."
        )

    return (
        result
        .sort_values(
            [
                "order_date",
                "customer",
                "invoice",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# OUTPUT HELPERS
# ================================================================

def first_recommendation(
    recommendations,
):
    """
    Return the first ranked product or pandas NA when no recommendation
    exists.
    """

    if recommendations is None:
        return pd.NA

    if len(
        recommendations
    ) == 0:
        return pd.NA

    return recommendations[
        0
    ]


def recommendation_count(
    recommendations,
) -> int:
    """
    Return the number of products in a recommendation list.
    """

    if recommendations is None:
        return 0

    return len(
        recommendations
    )


# ================================================================
# PRODUCTION SCORER
# ================================================================

class RecommendationScorer:
    """
    Production scoring interface for the frozen deterministic
    recommendation architecture.

    Parameters
    ----------
    top_n:
        Maximum number of products returned by each ranking channel.
    """

    metadata = RECOMMENDATION_MODEL_METADATA

    key_columns = RECOMMENDATION_EPISODE_KEY_COLUMNS

    def __init__(
        self,
        top_n: int = RECOMMENDATION_DEFAULT_TOP_N,
    ) -> None:

        validate_recommendation_depth(
            top_n
        )

        self.top_n = top_n

    def validate_input(
        self,
        transactions: pd.DataFrame,
        episodes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate transaction and episode inputs.

        Returns the normalised episode table.
        """

        if transactions.empty:
            raise ValueError(
                "Cannot generate recommendations from an empty "
                "transaction table."
            )

        validate_recommendation_transactions(
            transactions
        )

        return normalise_recommendation_episodes(
            episodes
        )

    def score(
        self,
        transactions: pd.DataFrame,
        episodes: pd.DataFrame,
        scored_at: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Generate governed product recommendations.

        Global affinity/popularity history is constrained by the
        month-start rule inside the frozen recommendation engine.

        Customer-specific history is constrained through each episode's
        order_date.

        No future outcome information is used.
        """

        normalised_episodes = (
            self.validate_input(
                transactions=transactions,
                episodes=episodes,
            )
        )

        observation_end_date = (
            normalised_episodes[
                "order_date"
            ]
            .max()
        )

        recommendation_lines = (
            build_recommendation_lines(
                transactions=transactions,
                observation_end_date=(
                    observation_end_date
                ),
            )
        )

        if recommendation_lines.empty:
            raise ValueError(
                "No qualifying recommendation transactions are available "
                "on or before the scoring observation boundary."
            )

        rankings = (
            generate_recommendations_for_episodes(
                episodes=normalised_episodes,
                recommendation_lines=(
                    recommendation_lines
                ),
                top_n=self.top_n,
            )
        )

        result = rankings.copy()

        result[
            "prediction_date"
        ] = result[
            "order_date"
        ]

        result[
            "repeat_top1_product"
        ] = result[
            "repeat_recommendations"
        ].map(
            first_recommendation
        )

        result[
            "repeat_recommendation_count"
        ] = result[
            "repeat_recommendations"
        ].map(
            recommendation_count
        )

        result[
            "discovery_top1_product"
        ] = result[
            "discovery_recommendations"
        ].map(
            first_recommendation
        )

        result[
            "discovery_recommendation_count"
        ] = result[
            "discovery_recommendations"
        ].map(
            recommendation_count
        )

        result[
            "hybrid_top1_product"
        ] = result[
            "hybrid_recommendations"
        ].map(
            first_recommendation
        )

        result[
            "hybrid_recommendation_count"
        ] = result[
            "hybrid_recommendations"
        ].map(
            recommendation_count
        )

        result[
            "repeat_model_name"
        ] = (
            RECOMMENDATION_REPEAT_MODEL_NAME
        )

        result[
            "discovery_model_name"
        ] = (
            RECOMMENDATION_DISCOVERY_MODEL_NAME
        )

        result[
            "hybrid_model_name"
        ] = (
            RECOMMENDATION_OPTIONAL_HYBRID_NAME
        )

        result[
            "hybrid_role"
        ] = (
            RECOMMENDATION_HYBRID_ROLE
        )

        result = attach_score_metadata(
            scores=result,
            metadata=self.metadata,
            scored_at=scored_at,
        )

        return (
            result.loc[
                :,
                list(
                    RECOMMENDATION_SCORE_COLUMNS
                ),
            ]
            .reset_index(
                drop=True
            )
        )