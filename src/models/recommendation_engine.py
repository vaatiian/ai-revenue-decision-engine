"""
Research-parity deterministic recommendation engine.

This module reproduces the recommendation algorithms developed and frozen
in Notebook 06 — ML Next Best Product.

The production recommendation architecture has two primary channels:

1. Customer Repeat Frequency
2. Item-to-Item Affinity

An optional 70% Repeat / 30% Discovery presentation layer is also
supported.

The module intentionally contains no fitted sklearn model. The governed
recommendation architecture is deterministic and is reconstructed from
point-in-time transaction history.

Research parity
---------------
Qualifying recommendation transactions:
- identified customer;
- merchandise;
- quantity > 0;
- unit price > 0;
- not cancellation;
- within the observation boundary.

Customer orders:
- one customer × invoice;
- order date = minimum transaction timestamp;
- basket = unique sorted products.

Next-order target:
- next chronological customer order;
- observed within 90 days;
- only episodes with a complete 90-day follow-up window are evaluated.

Repeat channel:
- purchase order frequency descending;
- most recent purchase descending as tie-breaker.

Discovery channel:
- item-to-item cosine-style basket co-occurrence;
- minimum item basket count = 5;
- minimum pair basket count = 2;
- customer source weight = log1p(purchase order count);
- historical customer products excluded;
- global historical popularity used as fallback.

Temporal rule:
- affinity and popularity use baskets strictly before prediction-month start;
- customer-specific history may include activity through the prediction order.

No future recommendation outcome is used to construct recommendation scores.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from src.models.recommendation_spec import (
    RECOMMENDATION_DEFAULT_TOP_N,
    RECOMMENDATION_DISCOVERY_EXCLUDE_HISTORY,
    RECOMMENDATION_HYBRID_REPEAT_FRACTION,
    RECOMMENDATION_MINIMUM_ITEM_ORDERS,
    RECOMMENDATION_MINIMUM_PAIR_ORDERS,
    RECOMMENDATION_OUTCOME_WINDOW_DAYS,
    build_recommendation_hybrid_list,
    calculate_customer_source_weight,
    calculate_item_affinity_similarity,
    validate_recommendation_depth,
)


# ================================================================
# RESEARCH OBSERVATION BOUNDARY
# ================================================================

RECOMMENDATION_RESEARCH_END_DATE = pd.Timestamp(
    "2011-11-30 23:59:59"
)


# ================================================================
# INPUT CONTRACT
# ================================================================

RECOMMENDATION_REQUIRED_COLUMNS = (
    "date",
    "invoice",
    "customer",
    "product",
    "region",
    "quantity",
    "unit_price",
    "net_merchandise_revenue",
    "is_cancellation",
)


def validate_recommendation_transactions(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate the minimum canonical transaction contract.
    """

    missing = [
        column
        for column in RECOMMENDATION_REQUIRED_COLUMNS
        if column not in transactions.columns
    ]

    if missing:
        raise ValueError(
            "Recommendation transactions are missing required columns: "
            + ", ".join(missing)
        )


# ================================================================
# QUALIFYING RECOMMENDATION LINES
# ================================================================

def build_recommendation_lines(
    transactions: pd.DataFrame,
    observation_end_date=RECOMMENDATION_RESEARCH_END_DATE,
) -> pd.DataFrame:
    """
    Build the qualifying positive-merchandise transaction population.

    Canonical-data equivalence to Notebook 06:
    - customer identified       -> customer.notna()
    - merchandise               -> net_merchandise_revenue > 0
    - quantity > 0
    - unit_price > 0
    - not cancellation
    - timestamp <= observation boundary

    Positive merchandise rows necessarily have positive merchandise
    revenue under the canonical accounting contract.
    """

    validate_recommendation_transactions(
        transactions
    )

    boundary = pd.Timestamp(
        observation_end_date
    )

    result = (
        transactions.loc[
            transactions[
                "customer"
            ].notna()
            &
            transactions[
                "product"
            ].notna()
            &
            (
                transactions[
                    "quantity"
                ]
                >
                0
            )
            &
            (
                transactions[
                    "unit_price"
                ]
                >
                0
            )
            &
            (
                transactions[
                    "net_merchandise_revenue"
                ]
                >
                0
            )
            &
            (
                ~transactions[
                    "is_cancellation"
                ].astype(bool)
            )
            &
            (
                transactions[
                    "date"
                ]
                <=
                boundary
            )
        ]
        .copy()
    )

    result[
        "purchase_day"
    ] = (
        result[
            "date"
        ]
        .dt.normalize()
    )

    return (
        result
        .sort_values(
            [
                "customer",
                "date",
                "invoice",
                "product",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# CUSTOMER ORDERS + BASKETS
# ================================================================

def build_customer_orders(
    recommendation_lines: pd.DataFrame,
    observation_end_date=RECOMMENDATION_RESEARCH_END_DATE,
) -> pd.DataFrame:
    """
    Build the exact customer-order representation used in Notebook 06.
    """

    output_columns = [
        "customer",
        "invoice",
        "order_date",
        "order_revenue",
        "order_units",
        "distinct_products",
        "region",
        "basket_products",
        "customer_order_number",
        "customer_total_orders",
        "next_invoice",
        "next_order_date",
        "next_basket_products",
        "days_to_next_order",
        "has_next_order_within_90d",
        "complete_90d_followup",
    ]

    if recommendation_lines.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    customer_orders = (
        recommendation_lines
        .groupby(
            [
                "customer",
                "invoice",
            ],
            as_index=False,
        )
        .agg(
            order_date=(
                "date",
                "min",
            ),
            order_revenue=(
                "net_merchandise_revenue",
                "sum",
            ),
            order_units=(
                "quantity",
                "sum",
            ),
            distinct_products=(
                "product",
                "nunique",
            ),
            region=(
                "region",
                "first",
            ),
        )
    )

    order_baskets = (
        recommendation_lines
        .groupby(
            [
                "customer",
                "invoice",
            ]
        )[
            "product"
        ]
        .agg(
            lambda values:
                tuple(
                    sorted(
                        set(values)
                    )
                )
        )
        .rename(
            "basket_products"
        )
        .reset_index()
    )

    customer_orders = (
        customer_orders
        .merge(
            order_baskets,
            on=[
                "customer",
                "invoice",
            ],
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            [
                "customer",
                "order_date",
                "invoice",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    customer_orders[
        "customer_order_number"
    ] = (
        customer_orders
        .groupby(
            "customer"
        )
        .cumcount()
        +
        1
    )

    customer_orders[
        "customer_total_orders"
    ] = (
        customer_orders
        .groupby(
            "customer"
        )[
            "invoice"
        ]
        .transform(
            "size"
        )
    )

    customer_orders[
        "next_invoice"
    ] = (
        customer_orders
        .groupby(
            "customer"
        )[
            "invoice"
        ]
        .shift(-1)
    )

    customer_orders[
        "next_order_date"
    ] = (
        customer_orders
        .groupby(
            "customer"
        )[
            "order_date"
        ]
        .shift(-1)
    )

    customer_orders[
        "next_basket_products"
    ] = (
        customer_orders
        .groupby(
            "customer"
        )[
            "basket_products"
        ]
        .shift(-1)
    )

    customer_orders[
        "days_to_next_order"
    ] = (
        (
            customer_orders[
                "next_order_date"
            ]
            -
            customer_orders[
                "order_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    horizon = pd.Timedelta(
        days=RECOMMENDATION_OUTCOME_WINDOW_DAYS
    )

    customer_orders[
        "has_next_order_within_90d"
    ] = (
        customer_orders[
            "next_order_date"
        ]
        .notna()
        &
        (
            customer_orders[
                "next_order_date"
            ]
            <=
            (
                customer_orders[
                    "order_date"
                ]
                +
                horizon
            )
        )
    ).astype(int)

    boundary = pd.Timestamp(
        observation_end_date
    )

    customer_orders[
        "complete_90d_followup"
    ] = (
        customer_orders[
            "order_date"
        ]
        +
        horizon
        <=
        boundary
    )

    return customer_orders


# ================================================================
# CUMULATIVE CUSTOMER PRODUCT HISTORY
# ================================================================

def build_customer_product_history_by_order(
    customer_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the exact leakage-safe cumulative product history.

    The current prediction order is included in historical_products before
    predicting the subsequent order, matching Notebook 06.
    """

    output_columns = [
        "customer",
        "invoice",
        "historical_products",
        "historical_unique_product_count",
    ]

    if customer_orders.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    records = []

    for customer, group in (
        customer_orders
        .groupby(
            "customer",
            sort=False,
        )
    ):
        purchased_products = set()

        for row in group.itertuples(
            index=False
        ):
            current_basket = set(
                row.basket_products
            )

            purchased_products.update(
                current_basket
            )

            records.append(
                {
                    "customer":
                        customer,

                    "invoice":
                        row.invoice,

                    "historical_products":
                        tuple(
                            sorted(
                                purchased_products
                            )
                        ),

                    "historical_unique_product_count":
                        len(
                            purchased_products
                        ),
                }
            )

    return pd.DataFrame(
        records,
        columns=output_columns,
    )


# ================================================================
# OBSERVED NEXT-BASKET RESEARCH POPULATION
# ================================================================

def build_observed_next_basket_population(
    customer_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Notebook 06's primary recommendation evaluation population.

    Eligibility:
    - complete 90-day follow-up;
    - observed next order;
    - next order occurred within 90 days.

    This is intentionally conditional on the customer purchasing again.
    """

    if customer_orders.empty:
        return customer_orders.copy()

    population = (
        customer_orders.loc[
            customer_orders[
                "complete_90d_followup"
            ]
            &
            (
                customer_orders[
                    "has_next_order_within_90d"
                ]
                ==
                1
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    history = (
        build_customer_product_history_by_order(
            customer_orders
        )
    )

    population = (
        population
        .merge(
            history,
            on=[
                "customer",
                "invoice",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    if population.empty:
        population[
            "next_basket_size"
        ] = pd.Series(
            dtype="int64"
        )

        population[
            "repeat_product_count"
        ] = pd.Series(
            dtype="int64"
        )

        population[
            "new_product_count"
        ] = pd.Series(
            dtype="int64"
        )

        population[
            "repeat_products"
        ] = pd.Series(
            dtype="object"
        )

        population[
            "new_products"
        ] = pd.Series(
            dtype="object"
        )

        population[
            "prediction_period"
        ] = pd.Series(
            dtype="period[M]"
        )

        return population

    compositions = []

    for row in population.itertuples(
        index=False
    ):
        historical_products = set(
            row.historical_products
        )

        next_basket = set(
            row.next_basket_products
        )

        repeat_products = (
            next_basket
            &
            historical_products
        )

        new_products = (
            next_basket
            -
            historical_products
        )

        compositions.append(
            {
                "next_basket_size":
                    len(
                        next_basket
                    ),

                "repeat_product_count":
                    len(
                        repeat_products
                    ),

                "new_product_count":
                    len(
                        new_products
                    ),

                "repeat_products":
                    tuple(
                        sorted(
                            repeat_products
                        )
                    ),

                "new_products":
                    tuple(
                        sorted(
                            new_products
                        )
                    ),
            }
        )

    composition_frame = pd.DataFrame(
        compositions,
        index=population.index,
    )

    population = pd.concat(
        [
            population,
            composition_frame,
        ],
        axis=1,
    )

    population[
        "prediction_period"
    ] = (
        population[
            "order_date"
        ]
        .dt.to_period(
            "M"
        )
    )

    return population


# ================================================================
# HISTORICAL ORDER BASKETS
# ================================================================

def build_historical_baskets(
    recommendation_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build historical unique-product order baskets.
    """

    output_columns = [
        "invoice",
        "customer",
        "order_date",
        "products",
    ]

    if recommendation_lines.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    baskets = (
        recommendation_lines
        .groupby(
            [
                "invoice",
                "customer",
            ],
            as_index=False,
        )
        .agg(
            order_date=(
                "date",
                "min",
            ),

            products=(
                "product",
                lambda values:
                    tuple(
                        sorted(
                            set(values)
                        )
                    ),
            ),
        )
        .sort_values(
            "order_date"
        )
        .reset_index(
            drop=True
        )
    )

    return baskets


# ================================================================
# CUSTOMER-PRODUCT ORDER HISTORY
# ================================================================

def build_customer_product_order_history(
    recommendation_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one customer/invoice/product occurrence per historical order.
    """

    output_columns = [
        "customer",
        "invoice",
        "product",
        "date",
    ]

    if recommendation_lines.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    return (
        recommendation_lines.loc[
            :,
            output_columns,
        ]
        .drop_duplicates(
            [
                "customer",
                "invoice",
                "product",
            ]
        )
        .sort_values(
            [
                "customer",
                "date",
                "invoice",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# ITEM AFFINITY
# ================================================================

def build_item_affinity(
    baskets: Iterable,
    minimum_item_orders: int = (
        RECOMMENDATION_MINIMUM_ITEM_ORDERS
    ),
    minimum_pair_orders: int = (
        RECOMMENDATION_MINIMUM_PAIR_ORDERS
    ),
):
    """
    Reproduce Notebook 06's item-to-item affinity construction.

    Returns
    -------
    affinity:
        Mapping from source product to ranked neighbour tuples:

            (
                candidate_product,
                similarity,
                pair_order_count,
            )

    item_count:
        Counter of historical basket occurrences by product.
    """

    if minimum_item_orders <= 0:
        raise ValueError(
            "minimum_item_orders must be greater than zero."
        )

    if minimum_pair_orders <= 0:
        raise ValueError(
            "minimum_pair_orders must be greater than zero."
        )

    item_count = Counter()
    pair_count = defaultdict(
        Counter
    )

    for products in baskets:

        products = list(
            set(products)
        )

        for item in products:
            item_count[
                item
            ] += 1

        for item_a in products:
            for item_b in products:

                if item_a == item_b:
                    continue

                pair_count[
                    item_a
                ][
                    item_b
                ] += 1

    affinity = {}

    for item_a, neighbours in (
        pair_count.items()
    ):

        if (
            item_count[
                item_a
            ]
            <
            minimum_item_orders
        ):
            continue

        scored_neighbours = []

        for (
            item_b,
            cooccurrence,
        ) in neighbours.items():

            if (
                cooccurrence
                <
                minimum_pair_orders
            ):
                continue

            if (
                item_count[
                    item_b
                ]
                <
                minimum_item_orders
            ):
                continue

            similarity = (
                calculate_item_affinity_similarity(
                    pair_order_count=cooccurrence,
                    item_a_order_count=(
                        item_count[
                            item_a
                        ]
                    ),
                    item_b_order_count=(
                        item_count[
                            item_b
                        ]
                    ),
                )
            )

            scored_neighbours.append(
                (
                    item_b,
                    similarity,
                    cooccurrence,
                )
            )

        scored_neighbours.sort(
            key=lambda item:
                (
                    item[1],
                    item[2],
                ),
            reverse=True,
        )

        affinity[
            item_a
        ] = scored_neighbours

    return (
        affinity,
        item_count,
    )


# ================================================================
# MONTH-START GLOBAL SNAPSHOT
# ================================================================

def build_affinity_snapshot(
    historical_baskets: pd.DataFrame,
    prediction_period,
):
    """
    Build the global affinity/popularity state available for a month.

    Only baskets strictly before the first timestamp of the prediction
    month are permitted.
    """

    period = pd.Period(
        prediction_period,
        freq="M",
    )

    cutoff = period.to_timestamp(
        how="start"
    )

    past_baskets = (
        historical_baskets.loc[
            historical_baskets[
                "order_date"
            ]
            <
            cutoff
        ]
        .copy()
    )

    affinity, item_count = (
        build_item_affinity(
            baskets=(
                past_baskets[
                    "products"
                ]
            ),
            minimum_item_orders=(
                RECOMMENDATION_MINIMUM_ITEM_ORDERS
            ),
            minimum_pair_orders=(
                RECOMMENDATION_MINIMUM_PAIR_ORDERS
            ),
        )
    )

    popularity_list = [
        product
        for product, count
        in sorted(
            item_count.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )
    ]

    return (
        affinity,
        popularity_list,
    )


# ================================================================
# CUSTOMER PRODUCT SUMMARY AT PREDICTION POINT
# ================================================================

def customer_product_summary_at_date(
    customer,
    cutoff,
    history_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the Notebook 06 customer-product history summary.
    """

    history = (
        history_table.loc[
            (
                history_table[
                    "customer"
                ]
                ==
                customer
            )
            &
            (
                history_table[
                    "date"
                ]
                <=
                pd.Timestamp(
                    cutoff
                )
            )
        ]
        .copy()
    )

    if history.empty:
        return pd.DataFrame(
            columns=[
                "product",
                "purchase_orders",
                "last_purchase",
            ]
        )

    return (
        history
        .groupby(
            "product",
            as_index=False,
        )
        .agg(
            purchase_orders=(
                "invoice",
                "nunique",
            ),

            last_purchase=(
                "date",
                "max",
            ),
        )
    )


# ================================================================
# REPEAT CHANNEL
# ================================================================

def rank_repeat_recommendations(
    customer,
    cutoff,
    history_table: pd.DataFrame,
    top_n: int = RECOMMENDATION_DEFAULT_TOP_N,
) -> list:
    """
    Rank customer products exactly as the frozen Repeat Frequency channel.

    Ranking:
        1. purchase_orders descending;
        2. last_purchase descending.
    """

    validate_recommendation_depth(
        top_n
    )

    history = (
        history_table.loc[
            (
                history_table[
                    "customer"
                ]
                ==
                customer
            )
            &
            (
                history_table[
                    "date"
                ]
                <=
                pd.Timestamp(
                    cutoff
                )
            )
        ]
        .copy()
    )

    if history.empty:
        return []

    product_summary = (
        history
        .groupby(
            "product",
            as_index=False,
        )
        .agg(
            purchase_orders=(
                "invoice",
                "nunique",
            ),

            last_purchase=(
                "date",
                "max",
            ),
        )
        .sort_values(
            [
                "purchase_orders",
                "last_purchase",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    return (
        product_summary[
            "product"
        ]
        .head(
            top_n
        )
        .tolist()
    )


# ================================================================
# CUSTOMER AFFINITY SCORES
# ================================================================

def affinity_scores_for_customer(
    product_history_summary: pd.DataFrame,
    affinity_dictionary: dict,
    exclude_history: bool = False,
) -> dict:
    """
    Reproduce the exact customer-specific affinity accumulation.

    Candidate score:
        sum(
            log1p(source purchase orders)
            *
            item affinity similarity
        )
    """

    scores = defaultdict(
        float
    )

    if product_history_summary.empty:
        return {}

    historical_products = set(
        product_history_summary[
            "product"
        ]
    )

    for row in (
        product_history_summary
        .itertuples(
            index=False
        )
    ):
        source_product = (
            row.product
        )

        source_weight = (
            calculate_customer_source_weight(
                int(
                    row.purchase_orders
                )
            )
        )

        neighbours = (
            affinity_dictionary
            .get(
                source_product,
                [],
            )
        )

        for (
            candidate,
            similarity,
            pair_count,
        ) in neighbours:

            if (
                exclude_history
                and
                candidate
                in
                historical_products
            ):
                continue

            scores[
                candidate
            ] += (
                source_weight
                *
                similarity
            )

    return dict(
        scores
    )


# ================================================================
# DISCOVERY CHANNEL
# ================================================================

def rank_affinity_recommendations(
    product_history_summary: pd.DataFrame,
    affinity_dictionary: dict,
    popularity_list,
    top_n: int = RECOMMENDATION_DEFAULT_TOP_N,
    exclude_history: bool = False,
) -> list:
    """
    Rank discovery candidates using affinity followed by popularity fallback.
    """

    validate_recommendation_depth(
        top_n
    )

    scores = (
        affinity_scores_for_customer(
            product_history_summary=(
                product_history_summary
            ),
            affinity_dictionary=(
                affinity_dictionary
            ),
            exclude_history=(
                exclude_history
            ),
        )
    )

    ranked = [
        product
        for product, score
        in sorted(
            scores.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )
    ]

    historical_products = set(
        product_history_summary[
            "product"
        ]
    )

    for product in popularity_list:

        if (
            exclude_history
            and
            product
            in
            historical_products
        ):
            continue

        if product not in ranked:
            ranked.append(
                product
            )

        if len(ranked) >= top_n:
            break

    return ranked[
        :top_n
    ]


# ================================================================
# ONE EPISODE
# ================================================================

def recommend_for_episode(
    customer,
    order_date,
    prediction_period,
    customer_product_history: pd.DataFrame,
    affinity_dictionary: dict,
    popularity_list,
    top_n: int = RECOMMENDATION_DEFAULT_TOP_N,
) -> dict:
    """
    Generate all frozen recommendation channels for one prediction episode.
    """

    validate_recommendation_depth(
        top_n
    )

    history_summary = (
        customer_product_summary_at_date(
            customer=customer,
            cutoff=order_date,
            history_table=(
                customer_product_history
            ),
        )
    )

    repeat_ranked = (
        rank_repeat_recommendations(
            customer=customer,
            cutoff=order_date,
            history_table=(
                customer_product_history
            ),
            top_n=top_n,
        )
    )

    discovery_ranked = (
        rank_affinity_recommendations(
            product_history_summary=(
                history_summary
            ),
            affinity_dictionary=(
                affinity_dictionary
            ),
            popularity_list=(
                popularity_list
            ),
            top_n=top_n,
            exclude_history=(
                RECOMMENDATION_DISCOVERY_EXCLUDE_HISTORY
            ),
        )
    )

    hybrid_ranked = (
        build_recommendation_hybrid_list(
            repeat_list=(
                repeat_ranked
            ),
            discovery_list=(
                discovery_ranked
            ),
            top_n=top_n,
            repeat_fraction=(
                RECOMMENDATION_HYBRID_REPEAT_FRACTION
            ),
        )
    )

    return {
        "customer":
            customer,

        "order_date":
            pd.Timestamp(
                order_date
            ),

        "prediction_period":
            pd.Period(
                prediction_period,
                freq="M",
            ),

        "repeat_recommendations":
            repeat_ranked,

        "discovery_recommendations":
            discovery_ranked,

        "hybrid_recommendations":
            hybrid_ranked,
    }


# ================================================================
# EPISODE BATCH GENERATOR
# ================================================================

def generate_recommendations_for_episodes(
    episodes: pd.DataFrame,
    recommendation_lines: pd.DataFrame,
    top_n: int = RECOMMENDATION_DEFAULT_TOP_N,
) -> pd.DataFrame:
    """
    Generate research-parity recommendations for multiple episodes.

    Global affinity/popularity snapshots are cached once per prediction
    month.

    Customer history remains episode-specific through each order_date.
    """

    validate_recommendation_depth(
        top_n
    )

    required_episode_columns = (
        "customer",
        "invoice",
        "order_date",
        "prediction_period",
    )

    missing = [
        column
        for column in required_episode_columns
        if column not in episodes.columns
    ]

    if missing:
        raise ValueError(
            "Recommendation episodes are missing required columns: "
            + ", ".join(missing)
        )

    output_columns = [
        "customer",
        "invoice",
        "order_date",
        "prediction_period",
        "repeat_recommendations",
        "discovery_recommendations",
        "hybrid_recommendations",
    ]

    if episodes.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    historical_baskets = (
        build_historical_baskets(
            recommendation_lines
        )
    )

    customer_product_history = (
        build_customer_product_order_history(
            recommendation_lines
        )
    )

    snapshot_cache = {}

    records = []

    for row in episodes.itertuples(
        index=False
    ):
        period = pd.Period(
            row.prediction_period,
            freq="M",
        )

        if period not in snapshot_cache:

            snapshot_cache[
                period
            ] = (
                build_affinity_snapshot(
                    historical_baskets=(
                        historical_baskets
                    ),
                    prediction_period=(
                        period
                    ),
                )
            )

        (
            affinity_dictionary,
            popularity_list,
        ) = snapshot_cache[
            period
        ]

        result = (
            recommend_for_episode(
                customer=row.customer,
                order_date=row.order_date,
                prediction_period=period,
                customer_product_history=(
                    customer_product_history
                ),
                affinity_dictionary=(
                    affinity_dictionary
                ),
                popularity_list=(
                    popularity_list
                ),
                top_n=top_n,
            )
        )

        records.append(
            {
                "customer":
                    row.customer,

                "invoice":
                    row.invoice,

                "order_date":
                    result[
                        "order_date"
                    ],

                "prediction_period":
                    result[
                        "prediction_period"
                    ],

                "repeat_recommendations":
                    result[
                        "repeat_recommendations"
                    ],

                "discovery_recommendations":
                    result[
                        "discovery_recommendations"
                    ],

                "hybrid_recommendations":
                    result[
                        "hybrid_recommendations"
                    ],
            }
        )

    return pd.DataFrame(
        records,
        columns=output_columns,
    )