import pytest

from portfolio.data_loader import (
    load_model_registry,
)

from portfolio.model_explorer import (
    DECISION_ENGINE_SYSTEM_IDS,
    EXPECTED_SYSTEM_IDS,
    STANDALONE_SYSTEM_IDS,
    approved_use_rows,
    architecture_rows,
    governance_label,
    limitation_rows,
    metric_count,
    metric_rows,
    prediction_horizon_label,
    registry_summary,
    system_role,
    systems_by_id,
    validate_model_registry,
)


def registry():
    return load_model_registry()


def test_real_registry_validates():
    validate_model_registry(
        registry()
    )


def test_exact_six_system_ids_are_frozen():
    result = systems_by_id(
        registry()
    )

    assert tuple(
        result.keys()
    ) == EXPECTED_SYSTEM_IDS


def test_four_systems_feed_customer_decision_engine():
    assert (
        DECISION_ENGINE_SYSTEM_IDS
        ==
        (
            "reactivation",
            "revenue_leakage",
            "future_customer_value",
            "recommendation",
        )
    )


def test_two_systems_are_standalone():
    assert (
        STANDALONE_SYSTEM_IDS
        ==
        (
            "repeat_purchase",
            "demand_forecast",
        )
    )


def test_reactivation_public_metrics_remain_empty():
    system = systems_by_id(
        registry()
    )[
        "reactivation"
    ]

    assert metric_rows(
        system
    ) == []

    assert metric_count(
        system
    ) == 0


def test_revenue_leakage_public_metrics_remain_empty():
    system = systems_by_id(
        registry()
    )[
        "revenue_leakage"
    ]

    assert metric_rows(
        system
    ) == []


def test_future_value_metrics_are_exposed():
    system = systems_by_id(
        registry()
    )[
        "future_customer_value"
    ]

    rows = metric_rows(
        system
    )

    names = {
        row[
            "metric"
        ]
        for row in rows
    }

    assert (
        "stage_a_pr_auc"
        in
        names
    )

    assert (
        "complete_spearman"
        in
        names
    )


def test_repeat_purchase_metrics_are_exposed():
    system = systems_by_id(
        registry()
    )[
        "repeat_purchase"
    ]

    names = {
        row[
            "metric"
        ]
        for row in metric_rows(
            system
        )
    }

    assert (
        "pr_auc"
        in
        names
    )

    assert (
        "roc_auc"
        in
        names
    )

    assert (
        "brier_score"
        in
        names
    )


def test_recommendation_nested_and_flat_metrics_are_exposed():
    system = systems_by_id(
        registry()
    )[
        "recommendation"
    ]

    assert metric_count(
        system
    ) == 6


def test_demand_nested_metrics_are_flattened():
    system = systems_by_id(
        registry()
    )[
        "demand_forecast"
    ]

    names = {
        row[
            "metric"
        ]
        for row in metric_rows(
            system
        )
    }

    assert (
        "approx_horizon_wape.1"
        in
        names
    )

    assert (
        "approx_horizon_wape.4"
        in
        names
    )


def test_simple_governance_label():
    assert (
        governance_label(
            "GO"
        )
        ==
        "GO"
    )


def test_channel_governance_label():
    system = systems_by_id(
        registry()
    )[
        "recommendation"
    ]

    label = governance_label(
        system[
            "governance"
        ]
    )

    assert (
        "Discovery Channel: CONDITIONAL GO"
        in
        label
    )

    assert (
        "Repeat Channel: GO"
        in
        label
    )


def test_standard_prediction_horizon_is_days():
    assert (
        prediction_horizon_label(
            90
        )
        ==
        "90 days"
    )


def test_demand_prediction_horizon_is_weeks():
    assert (
        prediction_horizon_label(
            [
                1,
                2,
                4,
            ]
        )
        ==
        "1, 2, 4 weeks"
    )


def test_future_value_approved_use_list_is_preserved():
    system = systems_by_id(
        registry()
    )[
        "future_customer_value"
    ]

    rows = approved_use_rows(
        system
    )

    assert (
        "customer ranking"
        in
        rows
    )

    assert (
        "controlled commercial decision support"
        in
        rows
    )


def test_recommendation_approved_use_dict_is_preserved():
    system = systems_by_id(
        registry()
    )[
        "recommendation"
    ]

    rows = approved_use_rows(
        system
    )

    assert len(
        rows
    ) == 2

    assert any(
        "Discovery Channel"
        in row
        for row in rows
    )


def test_repeat_purchase_limitations_are_preserved():
    system = systems_by_id(
        registry()
    )[
        "repeat_purchase"
    ]

    rows = limitation_rows(
        system
    )

    assert (
        "Causal treatment decisions"
        in
        rows
    )


def test_future_value_architecture_is_public():
    system = systems_by_id(
        registry()
    )[
        "future_customer_value"
    ]

    rows = architecture_rows(
        system
    )

    components = {
        row[
            "component"
        ]
        for row in rows
    }

    assert (
        "Stage A"
        in
        components
    )

    assert (
        "Stage B"
        in
        components
    )


def test_registry_summary_matches_public_artifact():
    summary = registry_summary(
        registry()
    )

    assert summary[
        "system_count"
    ] == 6

    assert summary[
        "decision_engine_inputs"
    ] == 4

    assert summary[
        "standalone_systems"
    ] == 2

    assert summary[
        "systems_with_public_metrics"
    ] == 4

    assert summary[
        "simple_go"
    ] == 3

    assert summary[
        "simple_conditional_go"
    ] == 2

    assert summary[
        "channel_governed"
    ] == 1


def test_decision_engine_system_role():
    assert (
        system_role(
            "revenue_leakage"
        )
        ==
        "Feeds the governed customer Decision Engine"
    )


def test_repeat_purchase_role_is_explicitly_standalone():
    assert (
        "Standalone customer propensity"
        in
        system_role(
            "repeat_purchase"
        )
    )


def test_demand_role_is_explicitly_standalone():
    assert (
        "Standalone weekly planning"
        in
        system_role(
            "demand_forecast"
        )
    )


def test_unknown_system_role_fails_closed():
    with pytest.raises(
        KeyError,
        match="Unknown system ID",
    ):
        system_role(
            "unknown_model"
        )


def test_wrong_registry_count_fails_closed():
    broken = dict(
        registry()
    )

    broken[
        "system_count"
    ] = 5

    with pytest.raises(
        RuntimeError,
        match="six intelligence systems",
    ):
        validate_model_registry(
            broken
        )