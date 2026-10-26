import math

import pandas as pd
import pytest

from src.models.recommendation_engine import (
    RECOMMENDATION_RESEARCH_END_DATE,
    affinity_scores_for_customer,
    build_affinity_snapshot,
    build_customer_orders,
    build_customer_product_history_by_order,
    build_customer_product_order_history,
    build_historical_baskets,
    build_item_affinity,
    build_observed_next_basket_population,
    build_recommendation_lines,
    customer_product_summary_at_date,
    generate_recommendations_for_episodes,
    rank_affinity_recommendations,
    rank_repeat_recommendations,
    recommend_for_episode,
)


def make_transactions(rows):
    """
    Build a compact canonical transaction frame for tests.

    Supports:
    - empty inputs;
    - mixed ISO date / datetime strings;
    - per-row overrides of canonical defaults.
    """

    base_columns = [
        "date",
        "invoice",
        "customer",
        "product",
    ]

    frame = pd.DataFrame(
        rows
    )

    for column in base_columns:
        if column not in frame.columns:
            frame[column] = pd.Series(
                dtype="object"
            )

    defaults = {
        "region": "United Kingdom",
        "quantity": 1,
        "unit_price": 10.0,
        "net_merchandise_revenue": 10.0,
        "is_cancellation": False,
    }

    for column, value in defaults.items():
        if column not in frame.columns:
            frame[
                column
            ] = value

    if frame.empty:
        frame[
            "date"
        ] = pd.Series(
            dtype="datetime64[ns]"
        )
    else:
        frame[
            "date"
        ] = pd.to_datetime(
            frame[
                "date"
            ],
            format="mixed",
        )

    return frame


def test_recommendation_lines_keep_positive_merchandise():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            }
        ]
    )

    result = build_recommendation_lines(
        transactions
    )

    assert len(result) == 1
    assert result.iloc[0]["product"] == "A"


@pytest.mark.parametrize(
    "override",
    [
        {
            "customer": None,
        },
        {
            "product": None,
        },
        {
            "quantity": 0,
        },
        {
            "quantity": -1,
        },
        {
            "unit_price": 0,
        },
        {
            "unit_price": -1,
        },
        {
            "net_merchandise_revenue": 0,
        },
        {
            "is_cancellation": True,
        },
    ],
)
def test_recommendation_lines_exclude_ineligible_rows(
    override,
):
    row = {
        "date": "2011-01-01",
        "invoice": "I1",
        "customer": "C1",
        "product": "A",
    }

    row.update(
        override
    )

    result = build_recommendation_lines(
        make_transactions(
            [row]
        )
    )

    assert result.empty


def test_recommendation_lines_enforce_observation_boundary():
    transactions = make_transactions(
        [
            {
                "date": RECOMMENDATION_RESEARCH_END_DATE,
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-12-01 00:00:00",
                "invoice": "I2",
                "customer": "C1",
                "product": "B",
            },
        ]
    )

    result = build_recommendation_lines(
        transactions
    )

    assert result[
        "invoice"
    ].tolist() == [
        "I1",
    ]


def test_customer_orders_create_unique_sorted_basket():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01 10:00",
                "invoice": "I1",
                "customer": "C1",
                "product": "B",
            },
            {
                "date": "2011-01-01 10:05",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-01-01 10:10",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
        ]
    )

    lines = build_recommendation_lines(
        transactions
    )

    orders = build_customer_orders(
        lines
    )

    assert len(orders) == 1

    assert (
        orders.iloc[0][
            "basket_products"
        ]
        ==
        (
            "A",
            "B",
        )
    )

    assert (
        orders.iloc[0][
            "distinct_products"
        ]
        ==
        2
    )


def test_customer_order_sequence_is_chronological():
    transactions = make_transactions(
        [
            {
                "date": "2011-03-01",
                "invoice": "I3",
                "customer": "C1",
                "product": "C",
            },
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-02-01",
                "invoice": "I2",
                "customer": "C1",
                "product": "B",
            },
        ]
    )

    orders = build_customer_orders(
        build_recommendation_lines(
            transactions
        )
    )

    assert orders[
        "invoice"
    ].tolist() == [
        "I1",
        "I2",
        "I3",
    ]

    assert orders[
        "customer_order_number"
    ].tolist() == [
        1,
        2,
        3,
    ]


def test_next_order_uses_customer_shift():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-01-10",
                "invoice": "I2",
                "customer": "C1",
                "product": "B",
            },
        ]
    )

    orders = build_customer_orders(
        build_recommendation_lines(
            transactions
        )
    )

    assert (
        orders.iloc[0][
            "next_invoice"
        ]
        ==
        "I2"
    )

    assert (
        orders.iloc[0][
            "next_basket_products"
        ]
        ==
        (
            "B",
        )
    )

    assert (
        orders.iloc[0][
            "days_to_next_order"
        ]
        ==
        pytest.approx(
            9.0
        )
    )


def test_next_order_exactly_90_days_is_eligible():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-04-01",
                "invoice": "I2",
                "customer": "C1",
                "product": "B",
            },
        ]
    )

    orders = build_customer_orders(
        build_recommendation_lines(
            transactions
        )
    )

    assert (
        orders.iloc[0][
            "days_to_next_order"
        ]
        ==
        pytest.approx(
            90.0
        )
    )

    assert (
        orders.iloc[0][
            "has_next_order_within_90d"
        ]
        ==
        1
    )


def test_next_order_after_90_days_is_not_eligible():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-04-02",
                "invoice": "I2",
                "customer": "C1",
                "product": "B",
            },
        ]
    )

    orders = build_customer_orders(
        build_recommendation_lines(
            transactions
        )
    )

    assert (
        orders.iloc[0][
            "has_next_order_within_90d"
        ]
        ==
        0
    )


def test_complete_followup_is_inclusive():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            }
        ]
    )

    lines = build_recommendation_lines(
        transactions,
        observation_end_date=(
            "2011-04-01"
        ),
    )

    orders = build_customer_orders(
        lines,
        observation_end_date=(
            "2011-04-01"
        ),
    )

    assert (
        orders.iloc[0][
            "complete_90d_followup"
        ]
        ==
        True
    )


def test_customer_product_history_includes_current_basket():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-02-01",
                "invoice": "I2",
                "customer": "C1",
                "product": "B",
            },
        ]
    )

    orders = build_customer_orders(
        build_recommendation_lines(
            transactions
        )
    )

    history = (
        build_customer_product_history_by_order(
            orders
        )
    )

    assert (
        history.iloc[0][
            "historical_products"
        ]
        ==
        (
            "A",
        )
    )

    assert (
        history.iloc[1][
            "historical_products"
        ]
        ==
        (
            "A",
            "B",
        )
    )


def test_observed_population_requires_return_within_90_days():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-01-20",
                "invoice": "I2",
                "customer": "C1",
                "product": "B",
            },
            {
                "date": "2011-01-01",
                "invoice": "J1",
                "customer": "C2",
                "product": "X",
            },
            {
                "date": "2011-05-01",
                "invoice": "J2",
                "customer": "C2",
                "product": "Y",
            },
        ]
    )

    orders = build_customer_orders(
        build_recommendation_lines(
            transactions
        )
    )

    population = (
        build_observed_next_basket_population(
            orders
        )
    )

    assert (
        "I1"
        in
        population[
            "invoice"
        ].tolist()
    )

    assert (
        "J1"
        not in
        population[
            "invoice"
        ].tolist()
    )


def test_repeat_and_new_products_are_split_correctly():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "B",
            },
            {
                "date": "2011-02-01",
                "invoice": "I2",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-02-01",
                "invoice": "I2",
                "customer": "C1",
                "product": "C",
            },
        ]
    )

    population = (
        build_observed_next_basket_population(
            build_customer_orders(
                build_recommendation_lines(
                    transactions
                )
            )
        )
    )

    first = population.loc[
        population[
            "invoice"
        ]
        ==
        "I1"
    ].iloc[0]

    assert first[
        "repeat_products"
    ] == (
        "A",
    )

    assert first[
        "new_products"
    ] == (
        "C",
    )


def test_historical_baskets_remove_duplicate_products():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "B",
            },
        ]
    )

    baskets = build_historical_baskets(
        build_recommendation_lines(
            transactions
        )
    )

    assert (
        baskets.iloc[0][
            "products"
        ]
        ==
        (
            "A",
            "B",
        )
    )


def test_customer_product_order_history_deduplicates_lines():
    transactions = make_transactions(
        [
            {
                "date": "2011-01-01",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
            {
                "date": "2011-01-01 00:05",
                "invoice": "I1",
                "customer": "C1",
                "product": "A",
            },
        ]
    )

    history = (
        build_customer_product_order_history(
            build_recommendation_lines(
                transactions
            )
        )
    )

    assert len(history) == 1


def test_item_affinity_matches_cosine_style_similarity():
    baskets = [
        (
            "A",
            "B",
        ),
        (
            "A",
            "B",
        ),
        (
            "A",
            "B",
        ),
        (
            "A",
            "B",
        ),
        (
            "A",
            "B",
        ),
    ]

    affinity, item_count = (
        build_item_affinity(
            baskets,
            minimum_item_orders=5,
            minimum_pair_orders=2,
        )
    )

    assert item_count[
        "A"
    ] == 5

    assert (
        affinity[
            "A"
        ][0][0]
        ==
        "B"
    )

    assert (
        affinity[
            "A"
        ][0][1]
        ==
        pytest.approx(
            1.0
        )
    )


def test_item_affinity_enforces_minimum_item_support():
    baskets = [
        (
            "A",
            "B",
        )
        for _ in range(
            4
        )
    ]

    affinity, _ = (
        build_item_affinity(
            baskets
        )
    )

    assert "A" not in affinity


def test_item_affinity_enforces_minimum_pair_support():
    baskets = [
        (
            "A",
        ),
        (
            "A",
        ),
        (
            "A",
        ),
        (
            "A",
            "B",
        ),
        (
            "A",
        ),
        (
            "B",
        ),
        (
            "B",
        ),
        (
            "B",
        ),
        (
            "B",
        ),
    ]

    affinity, _ = (
        build_item_affinity(
            baskets
        )
    )

    assert (
        affinity.get(
            "A",
            []
        )
        ==
        []
    )


def test_customer_product_summary_counts_orders_and_recency():
    history = pd.DataFrame(
        {
            "customer": [
                "C1",
                "C1",
                "C1",
            ],
            "invoice": [
                "I1",
                "I2",
                "I3",
            ],
            "product": [
                "A",
                "A",
                "B",
            ],
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-02-01",
                    "2011-03-01",
                ]
            ),
        }
    )

    result = (
        customer_product_summary_at_date(
            customer="C1",
            cutoff="2011-03-01",
            history_table=history,
        )
    )

    a = result.loc[
        result[
            "product"
        ]
        ==
        "A"
    ].iloc[0]

    assert (
        a[
            "purchase_orders"
        ]
        ==
        2
    )

    assert (
        a[
            "last_purchase"
        ]
        ==
        pd.Timestamp(
            "2011-02-01"
        )
    )


def test_repeat_ranking_frequency_before_recency():
    history = pd.DataFrame(
        {
            "customer": [
                "C1",
                "C1",
                "C1",
                "C1",
            ],
            "invoice": [
                "I1",
                "I2",
                "I3",
                "I4",
            ],
            "product": [
                "A",
                "A",
                "B",
                "C",
            ],
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-01-10",
                    "2011-03-01",
                    "2011-02-01",
                ]
            ),
        }
    )

    result = rank_repeat_recommendations(
        customer="C1",
        cutoff="2011-03-01",
        history_table=history,
        top_n=3,
    )

    assert result == [
        "A",
        "B",
        "C",
    ]


def test_repeat_ranking_uses_recency_for_frequency_tie():
    history = pd.DataFrame(
        {
            "customer": [
                "C1",
                "C1",
            ],
            "invoice": [
                "I1",
                "I2",
            ],
            "product": [
                "A",
                "B",
            ],
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-02-01",
                ]
            ),
        }
    )

    result = rank_repeat_recommendations(
        customer="C1",
        cutoff="2011-02-01",
        history_table=history,
        top_n=2,
    )

    assert result == [
        "B",
        "A",
    ]


def test_affinity_score_uses_log1p_source_weight():
    summary = pd.DataFrame(
        {
            "product": [
                "A",
            ],
            "purchase_orders": [
                4,
            ],
            "last_purchase": pd.to_datetime(
                [
                    "2011-01-01",
                ]
            ),
        }
    )

    affinity = {
        "A": [
            (
                "B",
                0.5,
                3,
            ),
        ]
    }

    result = (
        affinity_scores_for_customer(
            summary,
            affinity,
        )
    )

    assert result[
        "B"
    ] == pytest.approx(
        math.log1p(
            4
        )
        *
        0.5
    )


def test_affinity_excludes_historical_products_for_discovery():
    summary = pd.DataFrame(
        {
            "product": [
                "A",
                "B",
            ],
            "purchase_orders": [
                2,
                1,
            ],
            "last_purchase": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-02-01",
                ]
            ),
        }
    )

    affinity = {
        "A": [
            (
                "B",
                0.9,
                5,
            ),
            (
                "C",
                0.8,
                4,
            ),
        ]
    }

    result = (
        affinity_scores_for_customer(
            summary,
            affinity,
            exclude_history=True,
        )
    )

    assert "B" not in result
    assert "C" in result


def test_affinity_ranking_uses_popularity_fallback():
    summary = pd.DataFrame(
        {
            "product": [
                "A",
            ],
            "purchase_orders": [
                2,
            ],
            "last_purchase": pd.to_datetime(
                [
                    "2011-01-01",
                ]
            ),
        }
    )

    affinity = {
        "A": [
            (
                "B",
                0.9,
                5,
            ),
        ]
    }

    result = (
        rank_affinity_recommendations(
            product_history_summary=summary,
            affinity_dictionary=affinity,
            popularity_list=[
                "A",
                "C",
                "D",
            ],
            top_n=3,
            exclude_history=True,
        )
    )

    assert result == [
        "B",
        "C",
        "D",
    ]


def test_affinity_snapshot_excludes_current_month_baskets():
    baskets = pd.DataFrame(
        {
            "invoice": [
                "I1",
                "I2",
                "I3",
                "I4",
                "I5",
                "I6",
            ],
            "customer": [
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
            ],
            "order_date": pd.to_datetime(
                [
                    "2011-02-01",
                    "2011-02-02",
                    "2011-02-03",
                    "2011-02-04",
                    "2011-02-05",
                    "2011-03-01",
                ]
            ),
            "products": [
                (
                    "A",
                    "B",
                ),
                (
                    "A",
                    "B",
                ),
                (
                    "A",
                    "B",
                ),
                (
                    "A",
                    "B",
                ),
                (
                    "A",
                    "B",
                ),
                (
                    "X",
                    "Y",
                ),
            ],
        }
    )

    affinity, popularity = (
        build_affinity_snapshot(
            historical_baskets=baskets,
            prediction_period="2011-03",
        )
    )

    assert "A" in affinity

    assert "X" not in popularity
    assert "Y" not in popularity


def test_recommend_for_episode_returns_all_three_channels():
    history = pd.DataFrame(
        {
            "customer": [
                "C1",
                "C1",
            ],
            "invoice": [
                "I1",
                "I2",
            ],
            "product": [
                "A",
                "A",
            ],
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-02-01",
                ]
            ),
        }
    )

    affinity = {
        "A": [
            (
                "B",
                0.8,
                5,
            ),
        ]
    }

    result = recommend_for_episode(
        customer="C1",
        order_date="2011-02-01",
        prediction_period="2011-02",
        customer_product_history=history,
        affinity_dictionary=affinity,
        popularity_list=[
            "A",
            "B",
            "C",
        ],
        top_n=3,
    )

    assert (
        result[
            "repeat_recommendations"
        ][0]
        ==
        "A"
    )

    assert (
        result[
            "discovery_recommendations"
        ][0]
        ==
        "B"
    )

    assert (
        len(
            result[
                "hybrid_recommendations"
            ]
        )
        <=
        3
    )


def test_generate_recommendations_preserves_episode_keys():
    transactions = make_transactions(
        [
            {
                "date": f"2011-01-{day:02d}",
                "invoice": f"H{day}",
                "customer": f"C{day}",
                "product": "A",
            }
            for day in range(
                1,
                6
            )
        ]
        +
        [
            {
                "date": f"2011-01-{day:02d}",
                "invoice": f"HB{day}",
                "customer": f"D{day}",
                "product": "B",
            }
            for day in range(
                1,
                6
            )
        ]
        +
        [
            {
                "date": "2011-02-01",
                "invoice": "P1",
                "customer": "TARGET",
                "product": "A",
            }
        ]
    )

    lines = build_recommendation_lines(
        transactions
    )

    episodes = pd.DataFrame(
        {
            "customer": [
                "TARGET",
            ],
            "invoice": [
                "P1",
            ],
            "order_date": pd.to_datetime(
                [
                    "2011-02-01",
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

    result = (
        generate_recommendations_for_episodes(
            episodes=episodes,
            recommendation_lines=lines,
            top_n=5,
        )
    )

    assert len(result) == 1

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
        "P1"
    )


def test_generate_recommendations_rejects_missing_episode_columns():
    lines = make_transactions(
        []
    )

    episodes = pd.DataFrame(
        {
            "customer": [
                "C1",
            ]
        }
    )

    with pytest.raises(
        ValueError
    ):
        generate_recommendations_for_episodes(
            episodes,
            lines,
        )