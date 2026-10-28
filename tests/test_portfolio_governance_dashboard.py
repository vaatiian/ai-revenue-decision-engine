import pytest

from portfolio.data_loader import (
    load_model_registry,
    load_portfolio_summary,
)

from portfolio.governance_dashboard import (
    claims_boundary_rows,
    governance_summary,
    governance_surface_rows,
    runtime_availability_label,
    system_governance_detail,
    system_governance_rows,
    validate_governance_sources,
)


def registry():
    return load_model_registry()


def summary():
    return load_portfolio_summary()


def test_real_governance_sources_validate():
    validate_governance_sources(
        registry(),
        summary(),
    )


def test_runtime_is_verified_six_of_six():
    assert (
        runtime_availability_label(
            registry(),
            summary(),
        )
        ==
        "6/6 AVAILABLE"
    )


def test_governance_surface_count_is_seven():
    rows = governance_surface_rows(
        registry()
    )

    assert len(
        rows
    ) == 7


def test_four_go_surfaces():
    rows = governance_surface_rows(
        registry()
    )

    assert sum(
        row[
            "governance_decision"
        ]
        ==
        "GO"
        for row in rows
    ) == 4


def test_three_conditional_go_surfaces():
    rows = governance_surface_rows(
        registry()
    )

    assert sum(
        row[
            "governance_decision"
        ]
        ==
        "CONDITIONAL GO"
        for row in rows
    ) == 3


def test_recommendation_creates_two_governance_surfaces():
    rows = [
        row
        for row in governance_surface_rows(
            registry()
        )
        if row[
            "system_id"
        ]
        ==
        "recommendation"
    ]

    assert len(
        rows
    ) == 2


def test_recommendation_repeat_channel_is_go():
    rows = governance_surface_rows(
        registry()
    )

    row = next(
        row
        for row in rows
        if (
            row[
                "system_id"
            ]
            ==
            "recommendation"
            and
            row[
                "governance_surface"
            ]
            ==
            "Repeat Channel"
        )
    )

    assert (
        row[
            "governance_decision"
        ]
        ==
        "GO"
    )


def test_recommendation_discovery_channel_is_conditional_go():
    rows = governance_surface_rows(
        registry()
    )

    row = next(
        row
        for row in rows
        if (
            row[
                "system_id"
            ]
            ==
            "recommendation"
            and
            row[
                "governance_surface"
            ]
            ==
            "Discovery Channel"
        )
    )

    assert (
        row[
            "governance_decision"
        ]
        ==
        "CONDITIONAL GO"
    )


def test_six_system_matrix_rows():
    rows = system_governance_rows(
        registry(),
        summary(),
    )

    assert len(
        rows
    ) == 6


def test_all_systems_are_runtime_available():
    rows = system_governance_rows(
        registry(),
        summary(),
    )

    assert all(
        row[
            "technical_availability"
        ]
        ==
        "AVAILABLE"
        for row in rows
    )


def test_four_systems_are_decision_engine_inputs():
    rows = system_governance_rows(
        registry(),
        summary(),
    )

    assert sum(
        row[
            "decision_engine_role"
        ]
        ==
        "Included"
        for row in rows
    ) == 4


def test_two_systems_are_standalone():
    rows = system_governance_rows(
        registry(),
        summary(),
    )

    assert sum(
        row[
            "decision_engine_role"
        ]
        ==
        "Standalone"
        for row in rows
    ) == 2


def test_repeat_purchase_is_standalone():
    detail = system_governance_detail(
        registry(),
        "repeat_purchase",
    )

    assert (
        detail[
            "decision_engine_role"
        ]
        ==
        "Standalone"
    )


def test_demand_forecast_is_standalone():
    detail = system_governance_detail(
        registry(),
        "demand_forecast",
    )

    assert (
        detail[
            "decision_engine_role"
        ]
        ==
        "Standalone"
    )


def test_future_customer_value_is_decision_engine_input():
    detail = system_governance_detail(
        registry(),
        "future_customer_value",
    )

    assert (
        detail[
            "decision_engine_role"
        ]
        ==
        "Included"
    )


def test_future_customer_value_is_conditional_go():
    detail = system_governance_detail(
        registry(),
        "future_customer_value",
    )

    assert (
        detail[
            "governance"
        ]
        ==
        "CONDITIONAL GO"
    )


def test_repeat_purchase_limitations_remain_explicit():
    detail = system_governance_detail(
        registry(),
        "repeat_purchase",
    )

    assert (
        "Causal treatment decisions"
        in
        detail[
            "limitations"
        ]
    )


def test_reactivation_missing_approved_use_remains_missing():
    detail = system_governance_detail(
        registry(),
        "reactivation",
    )

    assert (
        detail[
            "approved_use"
        ]
        ==
        []
    )


def test_governance_summary_counts():
    result = governance_summary(
        registry(),
        summary(),
    )

    assert result[
        "production_systems"
    ] == 6

    assert result[
        "runtime_available"
    ] == 6

    assert result[
        "governance_surfaces"
    ] == 7

    assert result[
        "go_surfaces"
    ] == 4

    assert result[
        "conditional_go_surfaces"
    ] == 3


def test_governance_summary_decision_roles():
    result = governance_summary(
        registry(),
        summary(),
    )

    assert result[
        "decision_engine_systems"
    ] == 4

    assert result[
        "standalone_systems"
    ] == 2


def test_five_systems_publish_validation_method():
    result = governance_summary(
        registry(),
        summary(),
    )

    assert result[
        "systems_with_validation_method"
    ] == 5


def test_four_systems_publish_approved_use():
    result = governance_summary(
        registry(),
        summary(),
    )

    assert result[
        "systems_with_explicit_approved_use"
    ] == 4


def test_four_systems_publish_limitations():
    result = governance_summary(
        registry(),
        summary(),
    )

    assert result[
        "systems_with_explicit_limitations"
    ] == 4


def test_claims_boundaries_have_four_layers():
    rows = claims_boundary_rows(
        registry(),
        summary(),
    )

    assert [
        row[
            "boundary"
        ]
        for row in rows
    ] == [
        "Technical availability",
        "Governance permission",
        "Decision Engine inclusion",
        "Commercial claims",
    ]


def test_claims_boundary_preserves_public_claims_note():
    rows = claims_boundary_rows(
        registry(),
        summary(),
    )

    commercial = rows[
        3
    ]

    assert (
        commercial[
            "evidence"
        ]
        ==
        summary()[
            "claims_note"
        ]
    )


def test_unknown_system_fails_closed():
    with pytest.raises(
        KeyError,
        match="Unknown governance system ID",
    ):

        system_governance_detail(
            registry(),
            "unknown",
        )


def test_wrong_runtime_total_fails_closed():
    broken = dict(
        summary()
    )

    broken[
        "verified_evidence"
    ] = dict(
        broken[
            "verified_evidence"
        ]
    )

    broken[
        "verified_evidence"
    ][
        "production_scoring_systems_total"
    ] = 5

    with pytest.raises(
        RuntimeError,
        match="six production scoring systems",
    ):

        validate_governance_sources(
            registry(),
            broken,
        )


def test_partial_runtime_availability_fails_closed():
    broken = dict(
        summary()
    )

    broken[
        "verified_evidence"
    ] = dict(
        broken[
            "verified_evidence"
        ]
    )

    broken[
        "verified_evidence"
    ][
        "production_scoring_systems_available"
    ] = 5

    with pytest.raises(
        RuntimeError,
        match="6/6 runtime availability",
    ):

        validate_governance_sources(
            registry(),
            broken,
        )