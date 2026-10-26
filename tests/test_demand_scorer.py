import pandas as pd
import pytest

from src.models.demand_scorer import (
    DEMAND_SCORE_OUTPUT_COLUMNS,
    DemandScorer,
    forecast_weekly_revenue,
)

from src.models.demand_spec import (
    DEMAND_MODEL_METADATA,
)


# ================================================================
# SYNTHETIC WEEKLY MODEL
# ================================================================

def make_weekly_model(
    weeks=70,
):
    starts = pd.date_range(
        "2010-01-04",
        periods=weeks,
        freq="W-MON",
    )

    return pd.DataFrame(
        {
            "week_start":
                starts,

            "revenue":
                [
                    float(
                        1000
                        +
                        i
                    )
                    for i in range(
                        weeks
                    )
                ],
        }
    )


# ================================================================
# SYNTHETIC TRANSACTION HISTORY
# ================================================================

def make_transaction_history(
    weeks=70,
):
    """
    Build Notebook-07-compatible merchandise transactions.

    is_merchandise=True is explicit because the research notebook uses
    that field directly in the qualifying-demand filter.
    """

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

    first_sunday = (
        starts[
            0
        ]
        +
        pd.Timedelta(
            days=6,
            hours=12,
        )
    )

    last_sunday = (
        starts[
            -1
        ]
        +
        pd.Timedelta(
            days=6,
            hours=12,
        )
    )

    records.extend(
        [
            {
                "date":
                    first_sunday,

                "invoice":
                    "FIRST_SUNDAY",

                "customer":
                    "C1",

                "product":
                    "P1",

                "quantity":
                    1,

                "unit_price":
                    1.0,

                "is_merchandise":
                    True,

                "net_merchandise_revenue":
                    1.0,

                "is_cancellation":
                    False,
            },
            {
                "date":
                    last_sunday,

                "invoice":
                    "LAST_SUNDAY",

                "customer":
                    "C1",

                "product":
                    "P1",

                "quantity":
                    1,

                "unit_price":
                    1.0,

                "is_merchandise":
                    True,

                "net_merchandise_revenue":
                    1.0,

                "is_cancellation":
                    False,
            },
        ]
    )

    return pd.DataFrame(
        records
    )


# ================================================================
# OUTPUT CONTRACT
# ================================================================

def test_output_contract_is_frozen():
    assert (
        DEMAND_SCORE_OUTPUT_COLUMNS
        ==
        (
            "forecast_origin_week",
            "target_week",
            "horizon_weeks",
            "seasonal_analogue_week",
            "seasonal_analogue_revenue",
            "recent_four_week_average",
            "forecast_revenue",
            "seasonal_weight",
            "recent_weight",
            "model_name",
            "model_version",
            "model_type",
            "target_name",
            "governance_decision",
            "scored_at_utc",
        )
    )


# ================================================================
# SINGLE-HORIZON LOGIC
# ================================================================

def test_single_horizon_formula():
    weekly = make_weekly_model(
        weeks=70
    )

    origin = weekly[
        "week_start"
    ].iloc[
        60
    ]

    result = forecast_weekly_revenue(
        weekly_demand_model=weekly,
        forecast_origin_week=origin,
        horizon_weeks=1,
    )

    target_week = (
        origin
        +
        pd.Timedelta(
            weeks=1
        )
    )

    seasonal_week = (
        target_week
        -
        pd.Timedelta(
            weeks=52
        )
    )

    seasonal = float(
        weekly.loc[
            weekly[
                "week_start"
            ]
            ==
            seasonal_week,
            "revenue",
        ].iloc[
            0
        ]
    )

    recent = float(
        weekly.loc[
            weekly[
                "week_start"
            ]
            <=
            origin,
            "revenue",
        ]
        .tail(
            4
        )
        .mean()
    )

    expected = (
        0.90
        *
        seasonal
        +
        0.10
        *
        recent
    )

    assert (
        result[
            "forecast_revenue"
        ]
        ==
        pytest.approx(
            expected
        )
    )


@pytest.mark.parametrize(
    "horizon",
    [
        1,
        2,
        4,
    ],
)
def test_target_week_matches_horizon(
    horizon,
):
    weekly = make_weekly_model(
        weeks=70
    )

    origin = weekly[
        "week_start"
    ].iloc[
        60
    ]

    result = forecast_weekly_revenue(
        weekly,
        origin,
        horizon,
    )

    assert (
        result[
            "target_week"
        ]
        ==
        origin
        +
        pd.Timedelta(
            weeks=horizon
        )
    )


def test_future_observations_do_not_change_forecast():
    weekly = make_weekly_model(
        weeks=70
    )

    origin = weekly[
        "week_start"
    ].iloc[
        60
    ]

    before = forecast_weekly_revenue(
        weekly,
        origin,
        4,
    )

    modified = weekly.copy()

    modified.loc[
        modified[
            "week_start"
        ]
        >
        origin,
        "revenue",
    ] = 999999999.0

    after = forecast_weekly_revenue(
        modified,
        origin,
        4,
    )

    assert (
        before[
            "forecast_revenue"
        ]
        ==
        pytest.approx(
            after[
                "forecast_revenue"
            ]
        )
    )


def test_intermediate_future_weeks_do_not_enter_recent_average():
    weekly = make_weekly_model(
        weeks=70
    )

    origin = weekly[
        "week_start"
    ].iloc[
        60
    ]

    result = forecast_weekly_revenue(
        weekly,
        origin,
        4,
    )

    expected_recent = (
        weekly.loc[
            weekly[
                "week_start"
            ]
            <=
            origin,
            "revenue",
        ]
        .tail(
            4
        )
        .mean()
    )

    assert (
        result[
            "recent_four_week_average"
        ]
        ==
        pytest.approx(
            expected_recent
        )
    )


# ================================================================
# INPUT GOVERNANCE
# ================================================================

def test_origin_must_be_monday_week_start():
    weekly = make_weekly_model()

    with pytest.raises(
        ValueError,
        match="Monday week_start",
    ):
        forecast_weekly_revenue(
            weekly,
            "2011-02-01",
            1,
        )


def test_origin_must_exist_in_history():
    weekly = make_weekly_model()

    with pytest.raises(
        ValueError,
        match="not a complete observed week",
    ):
        forecast_weekly_revenue(
            weekly,
            "2015-01-05",
            1,
        )


def test_insufficient_seasonal_history_fails():
    weekly = make_weekly_model(
        weeks=20
    )

    origin = weekly[
        "week_start"
    ].iloc[
        -1
    ]

    with pytest.raises(
        ValueError,
        match="52-week seasonal analogue",
    ):
        forecast_weekly_revenue(
            weekly,
            origin,
            1,
        )


# ================================================================
# SCORER CONFIGURATION
# ================================================================

def test_default_scorer_horizons():
    scorer = DemandScorer()

    assert (
        scorer.horizons
        ==
        (
            1,
            2,
            4,
        )
    )


def test_custom_supported_horizon():
    scorer = DemandScorer(
        horizons=(
            4,
        )
    )

    assert (
        scorer.horizons
        ==
        (
            4,
        )
    )


def test_duplicate_horizons_fail():
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        DemandScorer(
            horizons=(
                1,
                1,
            )
        )


def test_invalid_horizon_fails():
    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        DemandScorer(
            horizons=(
                3,
            )
        )


# ================================================================
# END-TO-END SCORER OUTPUT
# ================================================================

def test_scorer_returns_three_rows():
    transactions = make_transaction_history(
        weeks=70
    )

    scorer = DemandScorer()

    origin = pd.Timestamp(
        "2011-03-07"
    )

    result = scorer.score(
        transactions=transactions,
        forecast_origin_week=origin,
        scored_at_utc=(
            "2026-10-26T18:00:00Z"
        ),
    )

    assert len(
        result
    ) == 3

    assert set(
        result[
            "horizon_weeks"
        ]
    ) == {
        1,
        2,
        4,
    }


def test_scorer_returns_exact_output_contract():
    transactions = make_transaction_history(
        weeks=70
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc="2026-10-26T18:00:00Z",
    )

    assert (
        tuple(
            result.columns
        )
        ==
        DEMAND_SCORE_OUTPUT_COLUMNS
    )


def test_scorer_attaches_metadata():
    transactions = make_transaction_history(
        weeks=70
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc="2026-10-26T18:00:00Z",
    )

    assert (
        result[
            "model_name"
        ]
        ==
        DEMAND_MODEL_METADATA.model_name
    ).all()

    assert (
        result[
            "model_version"
        ]
        ==
        DEMAND_MODEL_METADATA.model_version
    ).all()

    assert (
        result[
            "model_type"
        ]
        ==
        DEMAND_MODEL_METADATA.model_type
    ).all()

    assert (
        result[
            "target_name"
        ]
        ==
        DEMAND_MODEL_METADATA.target_name
    ).all()


def test_scorer_uses_90_10_weights():
    transactions = make_transaction_history(
        weeks=70
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc="2026-10-26T18:00:00Z",
    )

    assert (
        result[
            "seasonal_weight"
        ]
        .eq(
            0.90
        )
        .all()
    )

    assert (
        result[
            "recent_weight"
        ]
        .eq(
            0.10
        )
        .all()
    )


def test_scorer_forecasts_are_not_missing():
    transactions = make_transaction_history(
        weeks=70
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc="2026-10-26T18:00:00Z",
    )

    assert (
        result[
            "forecast_revenue"
        ]
        .notna()
        .all()
    )


def test_scorer_forecasts_are_finite():
    transactions = make_transaction_history(
        weeks=70
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc="2026-10-26T18:00:00Z",
    )

    assert (
        result[
            "forecast_revenue"
        ]
        .map(
            lambda value:
                pd.notna(
                    value
                )
                and
                value
                not in (
                    float(
                        "inf"
                    ),
                    float(
                        "-inf"
                    ),
                )
        )
        .all()
    )


def test_scorer_horizon_order_is_sorted():
    transactions = make_transaction_history(
        weeks=70
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc="2026-10-26T18:00:00Z",
    )

    assert (
        result[
            "horizon_weeks"
        ].tolist()
        ==
        [
            1,
            2,
            4,
        ]
    )


def test_scorer_target_weeks_are_unique():
    transactions = make_transaction_history(
        weeks=70
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc="2026-10-26T18:00:00Z",
    )

    assert (
        result[
            "target_week"
        ]
        .is_unique
    )


def test_scorer_governance_decision_is_attached():
    transactions = make_transaction_history(
        weeks=70
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc="2026-10-26T18:00:00Z",
    )

    assert (
        result[
            "governance_decision"
        ]
        ==
        "GO"
    ).all()


def test_scored_at_is_preserved():
    transactions = make_transaction_history(
        weeks=70
    )

    timestamp = (
        "2026-10-26T18:00:00Z"
    )

    result = DemandScorer().score(
        transactions=transactions,
        forecast_origin_week="2011-03-07",
        scored_at_utc=timestamp,
    )

    assert (
        result[
            "scored_at_utc"
        ]
        ==
        pd.Timestamp(
            timestamp
        )
    ).all()