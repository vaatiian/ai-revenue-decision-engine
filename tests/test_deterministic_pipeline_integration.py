"""
Integration tests for deterministic production scoring pipelines.

Covers:
- RecommendationScorer
- DemandScorer

These tests exercise real transaction inputs through the production
deterministic scorers without notebook dependencies.
"""

from datetime import datetime, timezone

import pandas as pd

from src.models.demand_scorer import (
    DEMAND_SCORE_OUTPUT_COLUMNS,
    DemandScorer,
)
from src.models.demand_spec import (
    DEMAND_FINAL_DECISION,
    DEMAND_FORECAST_HORIZONS,
)

from src.models.recommendation_scorer import (
    RECOMMENDATION_SCORE_COLUMNS,
    RecommendationScorer,
)
from src.models.recommendation_spec import (
    RECOMMENDATION_DISCOVERY_DECISION,
    RECOMMENDATION_REPEAT_DECISION,
)


DEMAND_FORECAST_ORIGIN = pd.Timestamp(
    "2011-03-07"
)

FIXED_SCORED_AT = pd.Timestamp(
    "2026-10-26T18:00:00Z"
)


# ================================================================
# RECOMMENDATION FIXTURES
# ================================================================


def make_recommendation_transactions():

    rows = []

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


def make_recommendation_episodes():

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
# DEMAND FIXTURE
# ================================================================


def make_demand_transactions(
    weeks=70,
):

    records = []

    starts = pd.date_range(
        "2010-01-04",
        periods=weeks,
        freq="W-MON",
    )

    for i, start in enumerate(
        starts
    ):

        revenue = float(
            1000
            +
            i
        )

        records.append(
            {
                "date":
                    start
                    +
                    pd.Timedelta(
                        hours=10
                    ),

                "invoice":
                    f"I{i}",

                "customer":
                    (
                        f"C{i % 5}"
                        if i % 7
                        else None
                    ),

                "product":
                    f"P{i % 10}",

                "quantity":
                    1,

                "unit_price":
                    revenue,

                "is_merchandise":
                    True,

                "net_merchandise_revenue":
                    revenue,

                "is_cancellation":
                    False,
            }
        )

    return pd.DataFrame(
        records
    )


# ================================================================
# RECOMMENDATION INTEGRATION
# ================================================================


def test_recommendation_raw_transactions_to_production_scorer():

    scorer = RecommendationScorer(
        top_n=5
    )

    scored_at = datetime(
        2011,
        2,
        10,
        12,
        0,
        tzinfo=timezone.utc,
    )

    result = scorer.score(
        transactions=(
            make_recommendation_transactions()
        ),
        episodes=(
            make_recommendation_episodes()
        ),
        scored_at=scored_at,
    )

    assert tuple(
        result.columns
    ) == RECOMMENDATION_SCORE_COLUMNS

    assert len(
        result
    ) == 1

    row = result.iloc[
        0
    ]

    assert row[
        "customer"
    ] == "TARGET"

    assert row[
        "invoice"
    ] == "T2"

    assert row[
        "prediction_date"
    ] == pd.Timestamp(
        "2011-02-10 10:00:00"
    )

    assert row[
        "repeat_recommendation_count"
    ] > 0

    assert row[
        "discovery_recommendation_count"
    ] > 0

    assert pd.notna(
        row[
            "repeat_top1_product"
        ]
    )

    assert pd.notna(
        row[
            "discovery_top1_product"
        ]
    )


def test_recommendation_channels_respect_governance_contract():

    result = RecommendationScorer(
        top_n=5
    ).score(
        transactions=(
            make_recommendation_transactions()
        ),
        episodes=(
            make_recommendation_episodes()
        ),
    )

    assert (
        RECOMMENDATION_REPEAT_DECISION
        ==
        "GO"
    )

    assert (
        RECOMMENDATION_DISCOVERY_DECISION
        ==
        "CONDITIONAL GO"
    )

    assert result[
        "repeat_recommendation_count"
    ].ge(
        1
    ).all()


# ================================================================
# DEMAND INTEGRATION
# ================================================================


def test_demand_raw_transactions_to_production_scorer():

    transactions = (
        make_demand_transactions()
    )

    scorer = DemandScorer()

    result = scorer.score(
        transactions=transactions,
        forecast_origin_week=(
            DEMAND_FORECAST_ORIGIN
        ),
        scored_at_utc=(
            FIXED_SCORED_AT
        ),
    )

    assert tuple(
        result.columns
    ) == DEMAND_SCORE_OUTPUT_COLUMNS

    assert tuple(
        result[
            "horizon_weeks"
        ]
    ) == DEMAND_FORECAST_HORIZONS

    assert len(
        result
    ) == len(
        DEMAND_FORECAST_HORIZONS
    )

    assert result[
        "forecast_revenue"
    ].gt(
        0
    ).all()

    assert result[
        "seasonal_analogue_revenue"
    ].gt(
        0
    ).all()

    assert result[
        "recent_four_week_average"
    ].gt(
        0
    ).all()


def test_demand_output_carries_governance_decision():

    result = DemandScorer().score(
        transactions=(
            make_demand_transactions()
        ),
        forecast_origin_week=(
            DEMAND_FORECAST_ORIGIN
        ),
        scored_at_utc=(
            FIXED_SCORED_AT
        ),
    )

    assert (
        DEMAND_FINAL_DECISION
        ==
        "GO"
    )

    assert result[
        "governance_decision"
    ].eq(
        DEMAND_FINAL_DECISION
    ).all()


def test_demand_target_weeks_are_strictly_future():

    result = DemandScorer().score(
        transactions=(
            make_demand_transactions()
        ),
        forecast_origin_week=(
            DEMAND_FORECAST_ORIGIN
        ),
        scored_at_utc=(
            FIXED_SCORED_AT
        ),
    )

    assert (
        result[
            "target_week"
        ]
        >
        DEMAND_FORECAST_ORIGIN
    ).all()


def test_demand_forecast_is_deterministic():

    transactions = (
        make_demand_transactions()
    )

    first = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week=(
            DEMAND_FORECAST_ORIGIN
        ),
        scored_at_utc=(
            FIXED_SCORED_AT
        ),
    )

    second = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week=(
            DEMAND_FORECAST_ORIGIN
        ),
        scored_at_utc=(
            FIXED_SCORED_AT
        ),
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )