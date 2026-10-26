"""
Frozen production specification for the Next Best Product recommendation
architecture.

This module reproduces the exact recommendation architecture selected and
frozen in Notebook 06 — ML Next Best Product.

Research specification
----------------------
Business objective:
    Recommend products likely to appear in the customer's next qualifying
    merchandise order.

Prediction target:
    Products appearing in the next qualifying order.

Prediction horizon:
    The next qualifying order must occur within 90 days for research outcome
    evaluation.

Frozen architecture:
    Two separate recommendation channels.

Channel 1 — Replenishment:
    Customer Repeat Frequency.

    Products are ranked by:
    1. number of distinct historical purchase orders, descending;
    2. most recent historical purchase date, descending.

Channel 2 — Discovery:
    Item-to-Item Affinity.

    Product similarity is based on historical basket co-occurrence:

        pair_order_count
        /
        sqrt(
            item_a_order_count
            *
            item_b_order_count
        )

    Minimum source/destination item order count:
        5.

    Minimum pair co-occurrence count:
        2.

    Customer-specific affinity contributions are weighted by:

        log1p(customer_purchase_orders_for_source_product)

    Previously purchased products are excluded from discovery
    recommendations.

    Historical product popularity is used as a deterministic fallback when
    affinity-ranked candidates do not fill the requested recommendation
    depth.

Optional combined presentation:
    70% Repeat / 30% Discovery.

    For Top-20:
        14 Repeat slots.
        6 Discovery slots.

    Duplicate products are removed.

    Unused slots are filled, in order, from:
        1. remaining Repeat recommendations;
        2. remaining Discovery recommendations.

Temporal governance:
    Global basket affinity and product-popularity snapshots use only baskets
    strictly before the relevant prediction month.

    Customer-specific purchase history may use transactions through the
    customer's prediction order.

Final governance:
    Repeat channel:
        GO.

    Discovery channel:
        CONDITIONAL GO.

    Optional hybrid:
        Presentation layer only.

The specification in this module must not be modified independently of the
governed research architecture.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from src.models.base import ModelMetadata


# ================================================================
# MODEL METADATA
# ================================================================

RECOMMENDATION_MODEL_METADATA = ModelMetadata(
    model_name=(
        "Customer Repeat Frequency + "
        "Item-to-Item Affinity"
    ),
    model_version="1.0.0",
    model_type="ranking",
    target_name="next_qualifying_order_products",
)


RECOMMENDATION_REPEAT_METADATA = ModelMetadata(
    model_name="Customer Repeat Frequency",
    model_version="1.0.0",
    model_type="ranking",
    target_name="next_qualifying_order_products",
)


RECOMMENDATION_DISCOVERY_METADATA = ModelMetadata(
    model_name="Item-to-Item Affinity",
    model_version="1.0.0",
    model_type="ranking",
    target_name="next_qualifying_order_products",
)


# ================================================================
# FROZEN ARCHITECTURE
# ================================================================

RECOMMENDATION_REPEAT_MODEL_NAME = (
    "Customer Repeat Frequency"
)

RECOMMENDATION_DISCOVERY_MODEL_NAME = (
    "Item-to-Item Affinity"
)

RECOMMENDATION_OPTIONAL_HYBRID_NAME = (
    "70% Repeat / 30% Discovery"
)


RECOMMENDATION_REPEAT_CHANNEL_FROZEN = True

RECOMMENDATION_DISCOVERY_CHANNEL_FROZEN = True

RECOMMENDATION_ARCHITECTURE_FROZEN = True


# ================================================================
# BUSINESS TARGET
# ================================================================

RECOMMENDATION_BUSINESS_OBJECTIVE = (
    "recommend products for the customer's next qualifying order"
)

RECOMMENDATION_TARGET = (
    "products appearing in next qualifying order"
)

RECOMMENDATION_TARGET_NAME = (
    "next_qualifying_order_products"
)

RECOMMENDATION_OUTCOME_WINDOW_DAYS = 90


# ================================================================
# PREDICTION / TEMPORAL CONTRACT
# ================================================================

RECOMMENDATION_PREDICTION_UNIT = (
    "one identified customer at one recommendation point"
)


RECOMMENDATION_GLOBAL_HISTORY_RULE = (
    "basket affinity and product popularity use only baskets strictly "
    "before the relevant prediction month"
)


RECOMMENDATION_CUSTOMER_HISTORY_RULE = (
    "customer-specific purchase history may include transactions through "
    "the customer's prediction order"
)


RECOMMENDATION_FUTURE_INFORMATION_ALLOWED = False


# ================================================================
# REPEAT CHANNEL
# ================================================================

RECOMMENDATION_REPEAT_PRIMARY_SORT = (
    "purchase_orders"
)

RECOMMENDATION_REPEAT_PRIMARY_SORT_ASCENDING = False

RECOMMENDATION_REPEAT_TIE_BREAK = (
    "last_purchase"
)

RECOMMENDATION_REPEAT_TIE_BREAK_ASCENDING = False


RECOMMENDATION_REPEAT_RANKING_DESCRIPTION = (
    "purchase frequency descending, then most recent purchase descending"
)


# ================================================================
# DISCOVERY CHANNEL
# ================================================================

RECOMMENDATION_AFFINITY_SIMILARITY_NAME = (
    "cosine-style basket co-occurrence"
)


RECOMMENDATION_MINIMUM_ITEM_ORDERS = 5

RECOMMENDATION_MINIMUM_PAIR_ORDERS = 2


RECOMMENDATION_SOURCE_WEIGHT_FORMULA = (
    "log1p(customer source-product purchase orders)"
)


RECOMMENDATION_AFFINITY_FORMULA = (
    "pair_order_count / sqrt("
    "item_a_order_count * item_b_order_count"
    ")"
)


RECOMMENDATION_DISCOVERY_EXCLUDE_HISTORY = True


RECOMMENDATION_DISCOVERY_FALLBACK = (
    "global historical product popularity"
)


# ================================================================
# RECOMMENDATION DEPTHS
# ================================================================

RECOMMENDATION_EVALUATION_DEPTHS = (
    5,
    10,
    20,
)

RECOMMENDATION_DEFAULT_TOP_N = 20


# ================================================================
# OPTIONAL HYBRID
# ================================================================

RECOMMENDATION_HYBRID_REPEAT_FRACTION = 0.70

RECOMMENDATION_HYBRID_DISCOVERY_FRACTION = 0.30


RECOMMENDATION_HYBRID_FILL_ORDER = (
    "repeat",
    "discovery",
)


RECOMMENDATION_HYBRID_ROLE = (
    "optional presentation layer only"
)


# ================================================================
# PRIMARY METRICS
# ================================================================

RECOMMENDATION_PRIMARY_GENERAL_METRICS = (
    "Hit Rate",
    "Recall",
)


RECOMMENDATION_PRIMARY_DISCOVERY_METRIC = (
    "Cross-Sell Hit Rate"
)


# ================================================================
# DEVELOPMENT / HOLDOUT PERIODS
# ================================================================

RECOMMENDATION_VALIDATION_START = "2011-03"

RECOMMENDATION_VALIDATION_END = "2011-04"


RECOMMENDATION_TEST_START = "2011-07"

RECOMMENDATION_TEST_END = "2011-08"


RECOMMENDATION_HOLDOUT_CONSUMED = True

RECOMMENDATION_POST_HOLDOUT_TUNING_PERMITTED = False


# ================================================================
# FINAL HOLDOUT PERFORMANCE
# ================================================================

RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10 = (
    0.705912
)

RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20 = (
    0.792319
)

RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20 = (
    0.870896
)


RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10 = (
    0.216010
)

RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20 = (
    0.299238
)


RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20 = (
    0.774634
)


# ================================================================
# FINAL GOVERNANCE
# ================================================================

RECOMMENDATION_REPEAT_DECISION = "GO"

RECOMMENDATION_DISCOVERY_DECISION = (
    "CONDITIONAL GO"
)


RECOMMENDATION_REPEAT_APPROVED_USE = (
    "operational next-product replenishment recommendation"
)


RECOMMENDATION_DISCOVERY_APPROVED_USE = (
    "product discovery, cross-sell experimentation and "
    "merchandising support"
)


RECOMMENDATION_DISCOVERY_PROHIBITED_INTERPRETATION = (
    "guaranteed next-purchase prediction"
)


# ================================================================
# DETERMINISTIC RESEARCH HELPERS
# ================================================================

def calculate_item_affinity_similarity(
    pair_order_count: int,
    item_a_order_count: int,
    item_b_order_count: int,
) -> float:
    """
    Calculate the exact frozen cosine-style basket similarity.

    Formula
    -------
    pair_order_count
    /
    sqrt(
        item_a_order_count
        *
        item_b_order_count
    )

    This helper does not itself apply minimum-support thresholds.
    """

    if pair_order_count < 0:
        raise ValueError(
            "pair_order_count cannot be negative."
        )

    if item_a_order_count <= 0:
        raise ValueError(
            "item_a_order_count must be greater than zero."
        )

    if item_b_order_count <= 0:
        raise ValueError(
            "item_b_order_count must be greater than zero."
        )

    return (
        pair_order_count
        /
        math.sqrt(
            item_a_order_count
            *
            item_b_order_count
        )
    )


def affinity_pair_is_eligible(
    pair_order_count: int,
    item_a_order_count: int,
    item_b_order_count: int,
) -> bool:
    """
    Apply the frozen research support thresholds.

    Both items must have at least five historical basket orders and the
    pair must have co-occurred in at least two historical baskets.
    """

    if pair_order_count < 0:
        raise ValueError(
            "pair_order_count cannot be negative."
        )

    if item_a_order_count < 0:
        raise ValueError(
            "item_a_order_count cannot be negative."
        )

    if item_b_order_count < 0:
        raise ValueError(
            "item_b_order_count cannot be negative."
        )

    return (
        item_a_order_count
        >=
        RECOMMENDATION_MINIMUM_ITEM_ORDERS
        and
        item_b_order_count
        >=
        RECOMMENDATION_MINIMUM_ITEM_ORDERS
        and
        pair_order_count
        >=
        RECOMMENDATION_MINIMUM_PAIR_ORDERS
    )


def calculate_customer_source_weight(
    purchase_orders: int,
) -> float:
    """
    Calculate the frozen customer source-product weight.

    Weight:
        log1p(purchase_orders)
    """

    if purchase_orders < 0:
        raise ValueError(
            "purchase_orders cannot be negative."
        )

    return math.log1p(
        purchase_orders
    )


def calculate_affinity_contribution(
    purchase_orders: int,
    similarity: float,
) -> float:
    """
    Calculate one source-product contribution to a candidate's total
    customer-specific affinity score.

    Contribution:
        log1p(purchase_orders) * similarity
    """

    if not math.isfinite(
        similarity
    ):
        raise ValueError(
            "similarity must be finite."
        )

    if similarity < 0:
        raise ValueError(
            "similarity cannot be negative."
        )

    source_weight = (
        calculate_customer_source_weight(
            purchase_orders
        )
    )

    return (
        source_weight
        *
        similarity
    )


def calculate_hybrid_slot_allocation(
    top_n: int = RECOMMENDATION_DEFAULT_TOP_N,
    repeat_fraction: float = (
        RECOMMENDATION_HYBRID_REPEAT_FRACTION
    ),
) -> tuple[int, int]:
    """
    Calculate the exact research hybrid slot allocation.

    Research logic:
        repeat_slots = round(top_n * repeat_fraction)
        discovery_slots = top_n - repeat_slots

    Python's built-in round semantics are intentionally retained.
    """

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    if not math.isfinite(
        repeat_fraction
    ):
        raise ValueError(
            "repeat_fraction must be finite."
        )

    if not (
        0.0
        <=
        repeat_fraction
        <=
        1.0
    ):
        raise ValueError(
            "repeat_fraction must be between 0 and 1."
        )

    repeat_slots = int(
        round(
            top_n
            *
            repeat_fraction
        )
    )

    discovery_slots = (
        top_n
        -
        repeat_slots
    )

    return (
        repeat_slots,
        discovery_slots,
    )


def build_recommendation_hybrid_list(
    repeat_list: Sequence,
    discovery_list: Sequence,
    top_n: int = RECOMMENDATION_DEFAULT_TOP_N,
    repeat_fraction: float = (
        RECOMMENDATION_HYBRID_REPEAT_FRACTION
    ),
) -> list:
    """
    Reproduce the exact frozen research hybrid-list algorithm.

    Initial allocation:
        repeat_slots = round(top_n * repeat_fraction)
        discovery_slots = top_n - repeat_slots

    Initial population:
        1. first repeat_slots products from repeat_list;
        2. first discovery_slots products from discovery_list.

    Duplicates are suppressed.

    Unused slots are then filled while preserving channel order:
        1. remaining repeat candidates;
        2. remaining discovery candidates.
    """

    repeat_slots, discovery_slots = (
        calculate_hybrid_slot_allocation(
            top_n=top_n,
            repeat_fraction=repeat_fraction,
        )
    )

    output = []

    for product in (
        list(repeat_list)[
            :repeat_slots
        ]
    ):
        if product not in output:
            output.append(
                product
            )

    for product in (
        list(discovery_list)[
            :discovery_slots
        ]
    ):
        if product not in output:
            output.append(
                product
            )

    for source in (
        repeat_list,
        discovery_list,
    ):
        for product in source:

            if product not in output:
                output.append(
                    product
                )

            if len(output) >= top_n:
                return output[
                    :top_n
                ]

    return output[
        :top_n
    ]


def validate_recommendation_depth(
    top_n: int,
) -> None:
    """
    Validate a requested recommendation-list depth.
    """

    if not isinstance(
        top_n,
        int,
    ):
        raise TypeError(
            "top_n must be an integer."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )