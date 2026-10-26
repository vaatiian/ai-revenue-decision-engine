from datetime import datetime, timezone

import pandas as pd
import pytest

from src.models.recommendation_scorer import (
    RECOMMENDATION_EPISODE_KEY_COLUMNS,
    RECOMMENDATION_EPISODE_REQUIRED_COLUMNS,
    RECOMMENDATION_SCORE_COLUMNS,
    RecommendationScorer,
    first_recommendation,
    normalise_recommendation_episodes,
    recommendation_count,
    validate_recommendation_episodes,
)
from src.models.recommendation_spec import (
    RECOMMENDATION_DISCOVERY_MODEL_NAME,
    RECOMMENDATION_HYBRID_ROLE,
    RECOMMENDATION_MODEL_METADATA,
    RECOMMENDATION_OPTIONAL_HYBRID_NAME,
    RECOMMENDATION_REPEAT_MODEL_NAME,
)


# ================================================================
# FIXTURES
# ================================================================

def make_transactions():
    """
    Build enough historical support for affinity plus a target customer.
    """

    rows = []

    # ------------------------------------------------------------
    # Five historical AB baskets.
    # These make A <-> B eligible under the frozen:
    # minimum item orders = 5
    # minimum pair orders = 2
    # ------------------------------------------------------------

    for number in range(
        1,
        6,
    ):
        rows.extend(
            [
                {
                    "date":
                        f"2011-01-{number:02d} 10:00:00",

                    "invoice":
                        f"AB{number}",

                    "customer":
                        f"H{number}",

                    "product":
                        "A",

                    "region":
                        "United Kingdom",

                    "quantity":
                        1,

                    "unit_price":
                        10.0,

                    "net_merchandise_revenue":
                        10.0,

                    "is_cancellation":
                        False,
                },
                {
                    "date":
                        f"2011-01-{number:02d} 10:01:00",

                    "invoice":
                        f"AB{number}",

                    "customer":
                        f"H{number}",

                    "product":
                        "B",

                    "region":
                        "United Kingdom",

                    "quantity":
                        1,

                    "unit_price":
                        10.0,

                    "net_merchandise_revenue":
                        10.0,

                    "is_cancellation":
                        False,
                },
            ]
        )

    # ------------------------------------------------------------
    # Five historical C baskets to provide an additional popularity
    # candidate.
    # ------------------------------------------------------------

    for number in range(
        1,
        6,
    ):
        rows.append(
            {
                "date":
                    f"2011-01-{number + 10:02d} 10:00:00",

                "invoice":
                    f"C{number}",

                "customer":
                    f"HC{number}",

                "product":
                    "C",

                "region":
                    "United Kingdom",

                "quantity":
                    1,

                "unit_price":
                    10.0,

                "net_merchandise_revenue":
                    10.0,

                "is_cancellation":
                    False,
            }
        )

    # ------------------------------------------------------------
    # Target customer's first purchase.
    # ------------------------------------------------------------

    rows.append(
        {
            "date":
                "2011-01-20 10:00:00",

            "invoice":
                "T1",

            "customer":
                "TARGET",

            "product":
                "A",

            "region":
                "United Kingdom",

            "quantity":
                1,

            "unit_price":
                10.0,

            "net_merchandise_revenue":
                10.0,

            "is_cancellation":
                False,
        }
    )

    # ------------------------------------------------------------
    # Target prediction order in February.
    # A is now a repeat-frequency candidate.
    # ------------------------------------------------------------

    rows.append(
        {
            "date":
                "2011-02-10 10:00:00",

            "invoice":
                "T2",

            "customer":
                "TARGET",

            "product":
                "A",

            "region":
                "United Kingdom",

            "quantity":
                1,

            "unit_price":
                10.0,

            "net_merchandise_revenue":
                10.0,

            "is_cancellation":
                False,
        }
    )

    frame = pd.DataFrame(
        rows
    )

    frame[
        "date"
    ] = pd.to_datetime(
        frame[
            "date"
        ],
        format="mixed",
    )

    return frame


def make_episodes():
    return pd.DataFrame(
        {
            "customer": [
                "TARGET",
            ],

            "invoice": [
                "T2",
            ],

            "order_date": pd.to_datetime(
                [
                    "2011-02-10 10:00:00",
                ]
            ),

            "prediction_period": [
                pd.Period(
                    "2011-02",
                    freq="M",
                )
            ],
        }
    )


# ================================================================
# FROZEN PRODUCTION CONTRACT
# ================================================================

def test_episode_required_columns_are_frozen():
    assert (
        RECOMMENDATION_EPISODE_REQUIRED_COLUMNS
        ==
        (
            "customer",
            "invoice",
            "order_date",
            "prediction_period",
        )
    )


def test_episode_key_columns_are_frozen():
    assert (
        RECOMMENDATION_EPISODE_KEY_COLUMNS
        ==
        (
            "customer",
            "invoice",
        )
    )


def test_output_contract_contains_expected_columns():
    assert (
        RECOMMENDATION_SCORE_COLUMNS
        ==
        (
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
    )


# ================================================================
# BASIC HELPERS
# ================================================================

def test_first_recommendation_returns_first_product():
    assert (
        first_recommendation(
            [
                "A",
                "B",
            ]
        )
        ==
        "A"
    )


def test_first_recommendation_returns_na_for_empty_list():
    result = first_recommendation(
        []
    )

    assert pd.isna(
        result
    )


def test_first_recommendation_returns_na_for_none():
    result = first_recommendation(
        None
    )

    assert pd.isna(
        result
    )


def test_recommendation_count():
    assert (
        recommendation_count(
            [
                "A",
                "B",
                "C",
            ]
        )
        ==
        3
    )


def test_recommendation_count_none_is_zero():
    assert (
        recommendation_count(
            None
        )
        ==
        0
    )


# ================================================================
# EPISODE VALIDATION
# ================================================================

def test_valid_episode_passes():
    validate_recommendation_episodes(
        make_episodes()
    )


def test_empty_episode_table_fails():
    with pytest.raises(
        ValueError,
        match="empty recommendation episode",
    ):
        validate_recommendation_episodes(
            make_episodes().iloc[
                0:0
            ]
        )


@pytest.mark.parametrize(
    "column",
    [
        "customer",
        "invoice",
        "order_date",
        "prediction_period",
    ],
)
def test_missing_episode_column_fails(
    column,
):
    episodes = (
        make_episodes()
        .drop(
            columns=[
                column,
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_recommendation_episodes(
            episodes
        )


@pytest.mark.parametrize(
    "column",
    [
        "customer",
        "invoice",
    ],
)
def test_missing_episode_key_value_fails(
    column,
):
    episodes = make_episodes()

    episodes.loc[
        0,
        column,
    ] = None

    with pytest.raises(
        ValueError,
        match="keys contain missing values",
    ):
        validate_recommendation_episodes(
            episodes
        )


def test_missing_order_date_fails():
    episodes = make_episodes()

    episodes.loc[
        0,
        "order_date",
    ] = pd.NaT

    with pytest.raises(
        ValueError,
        match="missing order_date",
    ):
        validate_recommendation_episodes(
            episodes
        )


def test_missing_prediction_period_fails():
    episodes = make_episodes()

    episodes.loc[
        0,
        "prediction_period",
    ] = pd.NaT

    with pytest.raises(
        ValueError,
        match="missing prediction_period",
    ):
        validate_recommendation_episodes(
            episodes
        )


def test_duplicate_episode_keys_fail():
    episodes = pd.concat(
        [
            make_episodes(),
            make_episodes(),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate entity keys",
    ):
        validate_recommendation_episodes(
            episodes
        )


# ================================================================
# EPISODE NORMALISATION
# ================================================================

def test_episode_normalisation_converts_string_period():
    episodes = make_episodes()

    episodes[
        "prediction_period"
    ] = "2011-02"

    result = (
        normalise_recommendation_episodes(
            episodes
        )
    )

    assert (
        result.iloc[0][
            "prediction_period"
        ]
        ==
        pd.Period(
            "2011-02",
            freq="M",
        )
    )


def test_episode_normalisation_converts_string_date():
    episodes = make_episodes()

    episodes[
        "order_date"
    ] = "2011-02-10 10:00:00"

    result = (
        normalise_recommendation_episodes(
            episodes
        )
    )

    assert (
        result.iloc[0][
            "order_date"
        ]
        ==
        pd.Timestamp(
            "2011-02-10 10:00:00"
        )
    )


def test_prediction_period_must_match_order_date():
    episodes = make_episodes()

    episodes[
        "prediction_period"
    ] = pd.Period(
        "2011-03",
        freq="M",
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        normalise_recommendation_episodes(
            episodes
        )


def test_invalid_order_date_fails():
    episodes = make_episodes()

    episodes[
        "order_date"
    ] = "not-a-date"

    with pytest.raises(
        ValueError,
        match="order_date values are invalid",
    ):
        normalise_recommendation_episodes(
            episodes
        )


# ================================================================
# SCORER CONSTRUCTION
# ================================================================

def test_scorer_uses_frozen_metadata():
    scorer = RecommendationScorer()

    assert (
        scorer.metadata
        ==
        RECOMMENDATION_MODEL_METADATA
    )


def test_default_top_n_is_20():
    scorer = RecommendationScorer()

    assert scorer.top_n == 20


def test_custom_top_n_is_supported():
    scorer = RecommendationScorer(
        top_n=5
    )

    assert scorer.top_n == 5


@pytest.mark.parametrize(
    "top_n",
    [
        0,
        -1,
        -10,
    ],
)
def test_invalid_top_n_fails(
    top_n,
):
    with pytest.raises(
        ValueError
    ):
        RecommendationScorer(
            top_n=top_n
        )


# ================================================================
# INPUT VALIDATION
# ================================================================

def test_empty_transactions_fail():
    scorer = RecommendationScorer()

    empty = (
        make_transactions()
        .iloc[
            0:0
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty transaction table",
    ):
        scorer.score(
            transactions=empty,
            episodes=make_episodes(),
        )


def test_missing_transaction_column_fails():
    scorer = RecommendationScorer()

    transactions = (
        make_transactions()
        .drop(
            columns=[
                "quantity",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        scorer.score(
            transactions=transactions,
            episodes=make_episodes(),
        )


def test_no_qualifying_transactions_fail():
    scorer = RecommendationScorer()

    transactions = (
        make_transactions()
        .copy()
    )

    transactions[
        "quantity"
    ] = 0

    with pytest.raises(
        ValueError,
        match="No qualifying recommendation transactions",
    ):
        scorer.score(
            transactions=transactions,
            episodes=make_episodes(),
        )


# ================================================================
# PRODUCTION SCORING
# ================================================================

def test_score_returns_one_row_per_episode():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    assert len(
        result
    ) == 1


def test_score_returns_exact_output_contract():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    assert (
        tuple(
            result.columns
        )
        ==
        RECOMMENDATION_SCORE_COLUMNS
    )


def test_score_preserves_episode_key():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    assert (
        result.iloc[0][
            "customer"
        ]
        ==
        "TARGET"
    )

    assert (
        result.iloc[0][
            "invoice"
        ]
        ==
        "T2"
    )


def test_prediction_date_matches_episode_order_date():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    assert (
        result.iloc[0][
            "prediction_date"
        ]
        ==
        pd.Timestamp(
            "2011-02-10 10:00:00"
        )
    )


def test_repeat_channel_recommends_historical_product():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    assert (
        result.iloc[0][
            "repeat_top1_product"
        ]
        ==
        "A"
    )


def test_discovery_channel_excludes_customer_history():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    discovery = result.iloc[0][
        "discovery_recommendations"
    ]

    assert "A" not in discovery


def test_discovery_channel_can_recommend_affinity_product():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    discovery = result.iloc[0][
        "discovery_recommendations"
    ]

    assert "B" in discovery


def test_recommendation_counts_match_lists():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    row = result.iloc[0]

    assert (
        row[
            "repeat_recommendation_count"
        ]
        ==
        len(
            row[
                "repeat_recommendations"
            ]
        )
    )

    assert (
        row[
            "discovery_recommendation_count"
        ]
        ==
        len(
            row[
                "discovery_recommendations"
            ]
        )
    )

    assert (
        row[
            "hybrid_recommendation_count"
        ]
        ==
        len(
            row[
                "hybrid_recommendations"
            ]
        )
    )


def test_recommendation_lists_do_not_exceed_top_n():
    scorer = RecommendationScorer(
        top_n=3
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    row = result.iloc[0]

    assert len(
        row[
            "repeat_recommendations"
        ]
    ) <= 3

    assert len(
        row[
            "discovery_recommendations"
        ]
    ) <= 3

    assert len(
        row[
            "hybrid_recommendations"
        ]
    ) <= 3


# ================================================================
# CHANNEL GOVERNANCE
# ================================================================

def test_channel_names_are_attached():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    row = result.iloc[0]

    assert (
        row[
            "repeat_model_name"
        ]
        ==
        RECOMMENDATION_REPEAT_MODEL_NAME
    )

    assert (
        row[
            "discovery_model_name"
        ]
        ==
        RECOMMENDATION_DISCOVERY_MODEL_NAME
    )

    assert (
        row[
            "hybrid_model_name"
        ]
        ==
        RECOMMENDATION_OPTIONAL_HYBRID_NAME
    )

    assert (
        row[
            "hybrid_role"
        ]
        ==
        RECOMMENDATION_HYBRID_ROLE
    )


# ================================================================
# STANDARD MODEL METADATA
# ================================================================

def test_standard_model_metadata_is_attached():
    scorer = RecommendationScorer(
        top_n=5
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
    )

    row = result.iloc[0]

    assert (
        row[
            "model_name"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.model_name
    )

    assert (
        row[
            "model_version"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.model_version
    )

    assert (
        row[
            "model_type"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.model_type
    )

    assert (
        row[
            "target_name"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.target_name
    )


def test_explicit_scored_at_is_preserved():
    scorer = RecommendationScorer(
        top_n=5
    )

    scored_at = datetime(
        2026,
        10,
        26,
        12,
        0,
        tzinfo=timezone.utc,
    )

    result = scorer.score(
        transactions=make_transactions(),
        episodes=make_episodes(),
        scored_at=scored_at,
    )

    assert (
        result.iloc[0][
            "scored_at_utc"
        ]
        ==
        pd.Timestamp(
            scored_at
        )
    )