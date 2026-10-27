"""
System smoke tests for governed scorer outputs entering the production
Customer Decision Engine.

This file deliberately crosses the scoring / decision-engine boundary.

It proves that native production outputs from:
- Future Customer Value,
- Reactivation,
- Recommendation

can flow directly into build_customer_decisions() and produce governed
customer actions without notebook-specific translation.

Repeat Purchase and Demand Forecast are intentionally excluded because
they are not current customer-action inputs to the production Decision
Engine.
"""

import numpy as np
import pandas as pd

from src.decision_engine.engine import (
    build_customer_decisions,
)
from src.decision_engine.spec import (
    ACTION_GROW,
    ACTION_REACTIVATE,
    DECISION_OUTPUT_COLUMNS,
    RECOMMENDATION_CHANNEL_DISCOVERY,
)

from src.models.fcv_scorer import (
    FCVScorer,
)
from src.models.fcv_spec import (
    FCV_MODEL_FEATURES,
)

from src.models.reactivation_scorer import (
    ReactivationScorer,
)
from src.models.reactivation_spec import (
    REACTIVATION_MODEL_FEATURES,
)

from src.models.recommendation_scorer import (
    RecommendationScorer,
)


DECISION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)

ACTIVE_ORDER_DATE = pd.Timestamp(
    "2011-08-31 10:00:00"
)

DORMANT_ORDER_DATE = pd.Timestamp(
    "2011-05-15 10:00:00"
)

FIXED_SCORED_AT = pd.Timestamp(
    "2026-10-27T12:00:00Z"
)


# ================================================================
# DECISION / RECOMMENDATION TRANSACTIONS
# ================================================================


def make_transactions():

    rows = []

    # ------------------------------------------------------------
    # Historical baskets establish BASE -> DISC affinity.
    # ------------------------------------------------------------

    for number in range(
        1,
        6,
    ):

        order_date = pd.Timestamp(
            f"2011-07-{number:02d} 10:00:00"
        )

        rows.extend(
            [
                {
                    "date":
                        order_date,

                    "customer":
                        f"H{number}",

                    "invoice":
                        f"H{number}",

                    "product":
                        "BASE",

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

                    "is_return":
                        False,
                },
                {
                    "date":
                        order_date
                        +
                        pd.Timedelta(
                            seconds=30
                        ),

                    "customer":
                        f"H{number}",

                    "invoice":
                        f"H{number}",

                    "product":
                        "DISC",

                    "region":
                        "United Kingdom",

                    "quantity":
                        1,

                    "unit_price":
                        20.0,

                    "net_merchandise_revenue":
                        20.0,

                    "is_cancellation":
                        False,

                    "is_return":
                        False,
                },
            ]
        )

    # ------------------------------------------------------------
    # A is Active.
    # Its latest order is also the recommendation prediction episode.
    # ------------------------------------------------------------

    rows.append(
        {
            "date":
                ACTIVE_ORDER_DATE,

            "customer":
                "A",

            "invoice":
                "A1",

            "product":
                "BASE",

            "region":
                "United Kingdom",

            "quantity":
                1,

            "unit_price":
                100.0,

            "net_merchandise_revenue":
                100.0,

            "is_cancellation":
                False,

            "is_return":
                False,
        }
    )

    # ------------------------------------------------------------
    # D is Dormant at the decision date.
    # ------------------------------------------------------------

    rows.append(
        {
            "date":
                DORMANT_ORDER_DATE,

            "customer":
                "D",

            "invoice":
                "D1",

            "product":
                "DORMANT_PRODUCT",

            "region":
                "United Kingdom",

            "quantity":
                1,

            "unit_price":
                50.0,

            "net_merchandise_revenue":
                50.0,

            "is_cancellation":
                False,

            "is_return":
                False,
        }
    )

    return pd.DataFrame(
        rows
    )


def make_recommendation_episodes():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],

            "invoice": [
                "A1",
            ],

            "order_date": [
                ACTIVE_ORDER_DATE,
            ],

            "prediction_period": [
                pd.Period(
                    "2011-08",
                    freq="M",
                ),
            ],
        }
    )


# ================================================================
# REAL FCV SCORING INPUT
# ================================================================


def make_fcv_model_features():

    customers = [
        "A",
        "D",
    ]

    data = {
        feature: np.full(
            len(
                customers
            ),
            1.0,
            dtype=float,
        )
        for feature
        in FCV_MODEL_FEATURES
    }

    data[
        "last_region"
    ] = [
        "United Kingdom",
        "United Kingdom",
    ]

    data[
        "active_in_last_30d"
    ] = [
        1,
        0,
    ]

    data[
        "active_in_last_90d"
    ] = [
        1,
        0,
    ]

    data[
        "prediction_month_sin"
    ] = [
        0.5,
        0.5,
    ]

    data[
        "prediction_month_cos"
    ] = [
        -0.5,
        -0.5,
    ]

    data[
        "h180_revenue"
    ] = [
        1000.0,
        500.0,
    ]

    data[
        "r90_revenue"
    ] = [
        800.0,
        100.0,
    ]

    data[
        "p90_revenue"
    ] = [
        200.0,
        400.0,
    ]

    frame = pd.DataFrame(
        data
    )

    frame.insert(
        0,
        "customer",
        customers,
    )

    return frame


# ================================================================
# REAL REACTIVATION SCORING INPUT
# ================================================================


def make_reactivation_model_features():

    customers = [
        "D",
    ]

    data = {
        feature: np.full(
            len(
                customers
            ),
            1.0,
            dtype=float,
        )
        for feature
        in REACTIVATION_MODEL_FEATURES
    }

    data[
        "last_order_region"
    ] = [
        "United Kingdom",
    ]

    data[
        "prediction_month"
    ] = [
        8,
    ]

    data[
        "prediction_weekday"
    ] = [
        DECISION_DATE.weekday(),
    ]

    data[
        "prediction_is_weekend"
    ] = [
        0,
    ]

    data[
        "has_prior_purchase_gap"
    ] = [
        1,
    ]

    data[
        "has_prior_inactivity_episode"
    ] = [
        1,
    ]

    data[
        "has_prior_successful_reactivation"
    ] = [
        0,
    ]

    frame = pd.DataFrame(
        data
    )

    frame.insert(
        0,
        "customer",
        customers,
    )

    return frame


def make_reactivation_provenance():

    return pd.DataFrame(
        {
            "customer": [
                "D",
            ],

            "prediction_date": [
                DECISION_DATE,
            ],
        }
    )


# ================================================================
# REAL GOVERNED SCORERS
# ================================================================


def score_fcv():

    return FCVScorer().score(
        make_fcv_model_features()
    )


def score_reactivation():

    return ReactivationScorer().score(
        make_reactivation_model_features()
    )


def score_recommendation():

    return RecommendationScorer(
        top_n=5
    ).score(
        transactions=(
            make_transactions()
        ),
        episodes=(
            make_recommendation_episodes()
        ),
        scored_at=(
            FIXED_SCORED_AT
        ),
    )


# ================================================================
# ENGINE SMOKE PATH
# ================================================================


def build_scored_decisions():

    return build_customer_decisions(
        transactions=(
            make_transactions()
        ),
        decision_date=(
            DECISION_DATE
        ),
        fcv_scores=(
            score_fcv()
        ),
        reactivation_scores=(
            score_reactivation()
        ),
        reactivation_features=(
            make_reactivation_provenance()
        ),
        recommendation_scores=(
            score_recommendation()
        ),
    )


def test_real_scorer_outputs_flow_into_decision_engine():

    fcv_scores = (
        score_fcv()
    )

    reactivation_scores = (
        score_reactivation()
    )

    recommendation_scores = (
        score_recommendation()
    )

    decisions = (
        build_customer_decisions(
            transactions=(
                make_transactions()
            ),
            decision_date=(
                DECISION_DATE
            ),
            fcv_scores=(
                fcv_scores
            ),
            reactivation_scores=(
                reactivation_scores
            ),
            reactivation_features=(
                make_reactivation_provenance()
            ),
            recommendation_scores=(
                recommendation_scores
            ),
        )
    )

    assert tuple(
        decisions.columns
    ) == DECISION_OUTPUT_COLUMNS

    assert decisions[
        "customer"
    ].is_unique

    assert "A" in set(
        decisions[
            "customer"
        ]
    )

    assert "D" in set(
        decisions[
            "customer"
        ]
    )


def test_active_customer_with_real_fcv_and_recommendation_scores_grows():

    decisions = (
        build_scored_decisions()
    )

    customer_a = (
        decisions.loc[
            decisions[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[
            0
        ]
    )

    assert (
        customer_a[
            "primary_action"
        ]
        ==
        ACTION_GROW
    )

    assert (
        customer_a[
            "recommended_product_channel"
        ]
        ==
        RECOMMENDATION_CHANNEL_DISCOVERY
    )

    assert pd.notna(
        customer_a[
            "recommended_product"
        ]
    )

    assert pd.notna(
        customer_a[
            "future_customer_value_90d"
        ]
    )

    assert pd.notna(
        customer_a[
            "future_active_probability"
        ]
    )


def test_dormant_customer_with_real_reactivation_score_reactivates():

    decisions = (
        build_scored_decisions()
    )

    customer_d = (
        decisions.loc[
            decisions[
                "customer"
            ]
            ==
            "D"
        ]
        .iloc[
            0
        ]
    )

    assert (
        customer_d[
            "primary_action"
        ]
        ==
        ACTION_REACTIVATE
    )

    assert pd.notna(
        customer_d[
            "reactivation_probability"
        ]
    )

    assert (
        customer_d[
            "reactivation_prediction_date"
        ]
        ==
        DECISION_DATE
    )

    assert (
        customer_d[
            "customer_state"
        ]
        ==
        "Dormant"
    )