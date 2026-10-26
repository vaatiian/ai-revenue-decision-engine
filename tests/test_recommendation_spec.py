import math

import pytest

from src.models.recommendation_spec import (
    RECOMMENDATION_AFFINITY_FORMULA,
    RECOMMENDATION_AFFINITY_SIMILARITY_NAME,
    RECOMMENDATION_ARCHITECTURE_FROZEN,
    RECOMMENDATION_DEFAULT_TOP_N,
    RECOMMENDATION_DISCOVERY_DECISION,
    RECOMMENDATION_DISCOVERY_EXCLUDE_HISTORY,
    RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10,
    RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_DISCOVERY_METADATA,
    RECOMMENDATION_DISCOVERY_MODEL_NAME,
    RECOMMENDATION_DISCOVERY_PROHIBITED_INTERPRETATION,
    RECOMMENDATION_EVALUATION_DEPTHS,
    RECOMMENDATION_FUTURE_INFORMATION_ALLOWED,
    RECOMMENDATION_HOLDOUT_CONSUMED,
    RECOMMENDATION_HYBRID_DISCOVERY_FRACTION,
    RECOMMENDATION_HYBRID_FILL_ORDER,
    RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_HYBRID_REPEAT_FRACTION,
    RECOMMENDATION_MINIMUM_ITEM_ORDERS,
    RECOMMENDATION_MINIMUM_PAIR_ORDERS,
    RECOMMENDATION_MODEL_METADATA,
    RECOMMENDATION_OPTIONAL_HYBRID_NAME,
    RECOMMENDATION_OUTCOME_WINDOW_DAYS,
    RECOMMENDATION_POST_HOLDOUT_TUNING_PERMITTED,
    RECOMMENDATION_REPEAT_DECISION,
    RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20,
    RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10,
    RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_REPEAT_METADATA,
    RECOMMENDATION_REPEAT_MODEL_NAME,
    RECOMMENDATION_REPEAT_PRIMARY_SORT,
    RECOMMENDATION_REPEAT_TIE_BREAK,
    RECOMMENDATION_SOURCE_WEIGHT_FORMULA,
    RECOMMENDATION_TARGET_NAME,
    RECOMMENDATION_TEST_END,
    RECOMMENDATION_TEST_START,
    affinity_pair_is_eligible,
    build_recommendation_hybrid_list,
    calculate_affinity_contribution,
    calculate_customer_source_weight,
    calculate_hybrid_slot_allocation,
    calculate_item_affinity_similarity,
    validate_recommendation_depth,
)


def test_complete_model_metadata_is_frozen():
    assert (
        RECOMMENDATION_MODEL_METADATA.model_name
        ==
        "Customer Repeat Frequency + Item-to-Item Affinity"
    )

    assert (
        RECOMMENDATION_MODEL_METADATA.model_version
        ==
        "1.0.0"
    )

    assert (
        RECOMMENDATION_MODEL_METADATA.model_type
        ==
        "ranking"
    )

    assert (
        RECOMMENDATION_MODEL_METADATA.target_name
        ==
        RECOMMENDATION_TARGET_NAME
    )


def test_repeat_metadata_is_frozen():
    assert (
        RECOMMENDATION_REPEAT_METADATA.model_name
        ==
        RECOMMENDATION_REPEAT_MODEL_NAME
    )

    assert (
        RECOMMENDATION_REPEAT_METADATA.model_type
        ==
        "ranking"
    )


def test_discovery_metadata_is_frozen():
    assert (
        RECOMMENDATION_DISCOVERY_METADATA.model_name
        ==
        RECOMMENDATION_DISCOVERY_MODEL_NAME
    )

    assert (
        RECOMMENDATION_DISCOVERY_METADATA.model_type
        ==
        "ranking"
    )


def test_architecture_is_frozen():
    assert RECOMMENDATION_ARCHITECTURE_FROZEN is True


def test_outcome_window_is_90_days():
    assert (
        RECOMMENDATION_OUTCOME_WINDOW_DAYS
        ==
        90
    )


def test_repeat_model_name_matches_research():
    assert (
        RECOMMENDATION_REPEAT_MODEL_NAME
        ==
        "Customer Repeat Frequency"
    )


def test_discovery_model_name_matches_research():
    assert (
        RECOMMENDATION_DISCOVERY_MODEL_NAME
        ==
        "Item-to-Item Affinity"
    )


def test_optional_hybrid_name_matches_research():
    assert (
        RECOMMENDATION_OPTIONAL_HYBRID_NAME
        ==
        "70% Repeat / 30% Discovery"
    )


def test_repeat_ranking_contract():
    assert (
        RECOMMENDATION_REPEAT_PRIMARY_SORT
        ==
        "purchase_orders"
    )

    assert (
        RECOMMENDATION_REPEAT_TIE_BREAK
        ==
        "last_purchase"
    )


def test_affinity_contract():
    assert (
        RECOMMENDATION_AFFINITY_SIMILARITY_NAME
        ==
        "cosine-style basket co-occurrence"
    )

    assert (
        RECOMMENDATION_AFFINITY_FORMULA
        ==
        "pair_order_count / sqrt("
        "item_a_order_count * item_b_order_count"
        ")"
    )


def test_minimum_support_thresholds():
    assert (
        RECOMMENDATION_MINIMUM_ITEM_ORDERS
        ==
        5
    )

    assert (
        RECOMMENDATION_MINIMUM_PAIR_ORDERS
        ==
        2
    )


def test_source_weight_formula_is_frozen():
    assert (
        RECOMMENDATION_SOURCE_WEIGHT_FORMULA
        ==
        "log1p(customer source-product purchase orders)"
    )


def test_discovery_excludes_historical_products():
    assert (
        RECOMMENDATION_DISCOVERY_EXCLUDE_HISTORY
        is True
    )


def test_evaluation_depths_are_frozen():
    assert (
        RECOMMENDATION_EVALUATION_DEPTHS
        ==
        (5, 10, 20)
    )

    assert (
        RECOMMENDATION_DEFAULT_TOP_N
        ==
        20
    )


def test_hybrid_fraction_is_70_30():
    assert (
        RECOMMENDATION_HYBRID_REPEAT_FRACTION
        ==
        pytest.approx(
            0.70
        )
    )

    assert (
        RECOMMENDATION_HYBRID_DISCOVERY_FRACTION
        ==
        pytest.approx(
            0.30
        )
    )


def test_hybrid_fill_order_is_repeat_then_discovery():
    assert (
        RECOMMENDATION_HYBRID_FILL_ORDER
        ==
        (
            "repeat",
            "discovery",
        )
    )


def test_holdout_period_is_july_august_2011():
    assert (
        RECOMMENDATION_TEST_START
        ==
        "2011-07"
    )

    assert (
        RECOMMENDATION_TEST_END
        ==
        "2011-08"
    )


def test_holdout_has_been_consumed():
    assert (
        RECOMMENDATION_HOLDOUT_CONSUMED
        is True
    )

    assert (
        RECOMMENDATION_POST_HOLDOUT_TUNING_PERMITTED
        is False
    )


def test_future_information_is_not_allowed():
    assert (
        RECOMMENDATION_FUTURE_INFORMATION_ALLOWED
        is False
    )


def test_final_repeat_governance_decision():
    assert (
        RECOMMENDATION_REPEAT_DECISION
        ==
        "GO"
    )


def test_final_discovery_governance_decision():
    assert (
        RECOMMENDATION_DISCOVERY_DECISION
        ==
        "CONDITIONAL GO"
    )


def test_discovery_is_not_guaranteed_next_purchase_prediction():
    assert (
        RECOMMENDATION_DISCOVERY_PROHIBITED_INTERPRETATION
        ==
        "guaranteed next-purchase prediction"
    )


def test_final_holdout_metrics_are_frozen():
    assert (
        RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10
        ==
        pytest.approx(
            0.705912
        )
    )

    assert (
        RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20
        ==
        pytest.approx(
            0.792319
        )
    )

    assert (
        RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20
        ==
        pytest.approx(
            0.870896
        )
    )

    assert (
        RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10
        ==
        pytest.approx(
            0.216010
        )
    )

    assert (
        RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20
        ==
        pytest.approx(
            0.299238
        )
    )

    assert (
        RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20
        ==
        pytest.approx(
            0.774634
        )
    )


def test_item_affinity_similarity_exact_formula():
    result = (
        calculate_item_affinity_similarity(
            pair_order_count=4,
            item_a_order_count=16,
            item_b_order_count=25,
        )
    )

    assert result == pytest.approx(
        0.2
    )


def test_item_affinity_similarity_is_symmetric_in_item_counts():
    left = (
        calculate_item_affinity_similarity(
            pair_order_count=3,
            item_a_order_count=9,
            item_b_order_count=16,
        )
    )

    right = (
        calculate_item_affinity_similarity(
            pair_order_count=3,
            item_a_order_count=16,
            item_b_order_count=9,
        )
    )

    assert left == pytest.approx(
        right
    )


@pytest.mark.parametrize(
    "pair_count,item_a,item_b",
    [
        (-1, 5, 5),
        (1, 0, 5),
        (1, 5, 0),
        (1, -1, 5),
        (1, 5, -1),
    ],
)
def test_item_affinity_similarity_rejects_invalid_counts(
    pair_count,
    item_a,
    item_b,
):
    with pytest.raises(
        ValueError
    ):
        calculate_item_affinity_similarity(
            pair_order_count=pair_count,
            item_a_order_count=item_a,
            item_b_order_count=item_b,
        )


def test_affinity_pair_is_eligible_at_exact_threshold():
    assert (
        affinity_pair_is_eligible(
            pair_order_count=2,
            item_a_order_count=5,
            item_b_order_count=5,
        )
        is True
    )


@pytest.mark.parametrize(
    "pair_count,item_a,item_b",
    [
        (1, 5, 5),
        (2, 4, 5),
        (2, 5, 4),
        (0, 100, 100),
    ],
)
def test_affinity_pair_rejects_below_threshold(
    pair_count,
    item_a,
    item_b,
):
    assert (
        affinity_pair_is_eligible(
            pair_order_count=pair_count,
            item_a_order_count=item_a,
            item_b_order_count=item_b,
        )
        is False
    )


@pytest.mark.parametrize(
    "pair_count,item_a,item_b",
    [
        (-1, 5, 5),
        (1, -1, 5),
        (1, 5, -1),
    ],
)
def test_affinity_pair_rejects_negative_counts(
    pair_count,
    item_a,
    item_b,
):
    with pytest.raises(
        ValueError
    ):
        affinity_pair_is_eligible(
            pair_order_count=pair_count,
            item_a_order_count=item_a,
            item_b_order_count=item_b,
        )


def test_customer_source_weight_matches_log1p():
    result = (
        calculate_customer_source_weight(
            4
        )
    )

    assert result == pytest.approx(
        math.log1p(
            4
        )
    )


def test_customer_source_weight_zero_orders_is_zero():
    assert (
        calculate_customer_source_weight(
            0
        )
        ==
        pytest.approx(
            0.0
        )
    )


def test_customer_source_weight_rejects_negative_orders():
    with pytest.raises(
        ValueError
    ):
        calculate_customer_source_weight(
            -1
        )


def test_affinity_contribution_matches_research_formula():
    result = (
        calculate_affinity_contribution(
            purchase_orders=4,
            similarity=0.25,
        )
    )

    expected = (
        math.log1p(
            4
        )
        *
        0.25
    )

    assert result == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "similarity",
    [
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_affinity_contribution_rejects_invalid_similarity(
    similarity,
):
    with pytest.raises(
        ValueError
    ):
        calculate_affinity_contribution(
            purchase_orders=1,
            similarity=similarity,
        )


def test_top_20_hybrid_allocation_is_14_repeat_6_discovery():
    repeat_slots, discovery_slots = (
        calculate_hybrid_slot_allocation(
            top_n=20,
            repeat_fraction=0.70,
        )
    )

    assert repeat_slots == 14
    assert discovery_slots == 6


def test_top_10_hybrid_allocation_is_7_repeat_3_discovery():
    repeat_slots, discovery_slots = (
        calculate_hybrid_slot_allocation(
            top_n=10,
            repeat_fraction=0.70,
        )
    )

    assert repeat_slots == 7
    assert discovery_slots == 3


def test_hybrid_allocation_sums_to_top_n():
    repeat_slots, discovery_slots = (
        calculate_hybrid_slot_allocation(
            top_n=17,
            repeat_fraction=0.70,
        )
    )

    assert (
        repeat_slots
        +
        discovery_slots
        ==
        17
    )


@pytest.mark.parametrize(
    "top_n",
    [
        0,
        -1,
        -20,
    ],
)
def test_hybrid_allocation_rejects_invalid_top_n(
    top_n,
):
    with pytest.raises(
        ValueError
    ):
        calculate_hybrid_slot_allocation(
            top_n=top_n
        )


@pytest.mark.parametrize(
    "repeat_fraction",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
    ],
)
def test_hybrid_allocation_rejects_invalid_fraction(
    repeat_fraction,
):
    with pytest.raises(
        ValueError
    ):
        calculate_hybrid_slot_allocation(
            repeat_fraction=repeat_fraction
        )


def test_hybrid_list_preserves_initial_70_30_allocation():
    repeat = [
        f"R{i}"
        for i in range(
            1,
            21
        )
    ]

    discovery = [
        f"D{i}"
        for i in range(
            1,
            21
        )
    ]

    result = (
        build_recommendation_hybrid_list(
            repeat_list=repeat,
            discovery_list=discovery,
            top_n=20,
            repeat_fraction=0.70,
        )
    )

    assert (
        result[:14]
        ==
        repeat[:14]
    )

    assert (
        result[14:20]
        ==
        discovery[:6]
    )


def test_hybrid_list_suppresses_duplicates():
    repeat = [
        "A",
        "B",
        "C",
        "D",
    ]

    discovery = [
        "C",
        "D",
        "E",
        "F",
    ]

    result = (
        build_recommendation_hybrid_list(
            repeat_list=repeat,
            discovery_list=discovery,
            top_n=6,
            repeat_fraction=0.50,
        )
    )

    assert len(result) == len(
        set(result)
    )


def test_hybrid_list_fills_unused_slots_from_repeat_first():
    repeat = [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
    ]

    discovery = [
        "D1",
    ]

    result = (
        build_recommendation_hybrid_list(
            repeat_list=repeat,
            discovery_list=discovery,
            top_n=5,
            repeat_fraction=0.60,
        )
    )

    assert result == [
        "R1",
        "R2",
        "R3",
        "D1",
        "R4",
    ]


def test_hybrid_list_uses_discovery_when_repeat_is_short():
    repeat = [
        "R1",
    ]

    discovery = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]

    result = (
        build_recommendation_hybrid_list(
            repeat_list=repeat,
            discovery_list=discovery,
            top_n=5,
            repeat_fraction=0.60,
        )
    )

    assert result == [
        "R1",
        "D1",
        "D2",
        "D3",
        "D4",
    ]


def test_hybrid_list_never_exceeds_top_n():
    repeat = list(
        range(
            100
        )
    )

    discovery = list(
        range(
            100,
            200
        )
    )

    result = (
        build_recommendation_hybrid_list(
            repeat,
            discovery,
            top_n=20,
        )
    )

    assert len(result) == 20


def test_hybrid_list_can_return_shorter_list_when_candidates_insufficient():
    result = (
        build_recommendation_hybrid_list(
            repeat_list=[
                "A",
            ],
            discovery_list=[
                "B",
            ],
            top_n=20,
        )
    )

    assert result == [
        "A",
        "B",
    ]


@pytest.mark.parametrize(
    "top_n",
    [
        1,
        5,
        10,
        20,
    ],
)
def test_validate_recommendation_depth_accepts_positive_integer(
    top_n,
):
    validate_recommendation_depth(
        top_n
    )


@pytest.mark.parametrize(
    "top_n",
    [
        0,
        -1,
        -5,
    ],
)
def test_validate_recommendation_depth_rejects_nonpositive_integer(
    top_n,
):
    with pytest.raises(
        ValueError
    ):
        validate_recommendation_depth(
            top_n
        )


@pytest.mark.parametrize(
    "top_n",
    [
        1.0,
        "20",
        None,
    ],
)
def test_validate_recommendation_depth_rejects_non_integer(
    top_n,
):
    with pytest.raises(
        TypeError
    ):
        validate_recommendation_depth(
            top_n
        )