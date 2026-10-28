import pandas as pd
import pytest

from portfolio.command_centre import (
    QUEUE_COLUMNS,
    build_distribution,
    build_priority_queue,
    build_risk_value_view,
    filter_decisions,
    filter_values,
    summarise_command_centre,
    validate_decision_frame,
)

from portfolio.data_loader import (
    load_customer_decisions,
)


def real_decisions():
    return load_customer_decisions()


def test_real_command_centre_frame_validates():
    validate_decision_frame(
        real_decisions()
    )


def test_filter_values_are_sorted_and_nonmissing():
    frame = real_decisions()

    values = filter_values(
        frame,
        "customer_state",
    )

    assert values == sorted(
        values
    )

    assert "Active" in values
    assert "At Risk" in values
    assert "Dormant" in values
    assert "Long Dormant" in values


def test_empty_filter_selection_returns_full_population():
    frame = real_decisions()

    result = filter_decisions(
        frame
    )

    assert len(
        result
    ) == 5220


def test_state_filter_returns_only_requested_state():
    frame = real_decisions()

    result = filter_decisions(
        frame,
        states=[
            "Dormant",
        ],
    )

    assert len(
        result
    ) == 857

    assert set(
        result[
            "customer_state"
        ]
    ) == {
        "Dormant",
    }


def test_action_filter_matches_verified_reactivation_population():
    frame = real_decisions()

    result = filter_decisions(
        frame,
        actions=[
            "Reactivate",
        ],
    )

    assert len(
        result
    ) == 857

    assert set(
        result[
            "primary_action"
        ]
    ) == {
        "Reactivate",
    }


def test_priority_filter_matches_verified_priority_1_population():
    frame = real_decisions()

    result = filter_decisions(
        frame,
        priorities=[
            "Priority 1",
        ],
    )

    assert len(
        result
    ) == 235

    assert set(
        result[
            "priority_tier"
        ]
    ) == {
        "Priority 1",
    }


def test_customer_query_uses_portfolio_identifier_only():
    frame = real_decisions()

    result = filter_decisions(
        frame,
        customer_query="CUST-00001",
    )

    assert len(
        result
    ) == 1

    assert (
        result.loc[
            0,
            "portfolio_customer_id",
        ]
        ==
        "CUST-00001"
    )


def test_command_centre_summary_matches_verified_population():
    frame = real_decisions()

    summary = summarise_command_centre(
        frame
    )

    assert summary[
        "customer_count"
    ] == 5220

    assert summary[
        "prioritised_count"
    ] == 2345

    assert summary[
        "priority_1_count"
    ] == 235

    assert summary[
        "experiment_required_count"
    ] == 857

    assert summary[
        "high_touch_count"
    ] == 235

    assert summary[
        "revenue_prior_90d"
    ] >= 0

    assert summary[
        "future_customer_value_90d"
    ] >= 0

    assert summary[
        "leakage_exposure_score"
    ] >= 0


def test_empty_command_centre_summary_is_zero_safe():
    frame = real_decisions().iloc[
        0:0
    ]

    summary = summarise_command_centre(
        frame
    )

    assert summary == {
        "customer_count":
            0,

        "prioritised_count":
            0,

        "priority_1_count":
            0,

        "experiment_required_count":
            0,

        "high_touch_count":
            0,

        "revenue_prior_90d":
            0.0,

        "future_customer_value_90d":
            0.0,

        "leakage_exposure_score":
            0.0,
    }


def test_action_distribution_reconciles_to_population():
    frame = real_decisions()

    result = build_distribution(
        frame,
        "primary_action",
    )

    assert result[
        "customers"
    ].sum() == 5220

    counts = dict(
        zip(
            result[
                "category"
            ],
            result[
                "customers"
            ],
        )
    )

    assert counts[
        "Protect"
    ] == 493

    assert counts[
        "Retain"
    ] == 995

    assert counts[
        "Reactivate"
    ] == 857

    assert counts[
        "Monitor"
    ] == 429

    assert counts[
        "Unscored — Long Dormant"
    ] == 2446


def test_priority_queue_contains_only_prioritised_customers():
    frame = real_decisions()

    queue = build_priority_queue(
        frame,
        limit=25,
    )

    assert tuple(
        queue.columns
    ) == QUEUE_COLUMNS

    assert len(
        queue
    ) == 25

    assert queue[
        "priority_tier"
    ].ne(
        "Not Prioritised"
    ).all()


def test_priority_queue_is_sorted_by_priority_percentile():
    frame = real_decisions()

    queue = build_priority_queue(
        frame,
        limit=50,
    )

    values = (
        pd.to_numeric(
            queue[
                "action_priority_percentile"
            ],
            errors="coerce",
        )
        .dropna()
        .tolist()
    )

    assert values == sorted(
        values,
        reverse=True,
    )


def test_priority_queue_rejects_invalid_limit():
    frame = real_decisions()

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        build_priority_queue(
            frame,
            limit=0,
        )


def test_risk_value_view_contains_complete_model_pairs_only():
    frame = real_decisions()

    result = build_risk_value_view(
        frame
    )

    assert not result.empty

    assert result[
        "leakage_probability"
    ].notna().all()

    assert result[
        "future_customer_value_90d"
    ].notna().all()

    assert "customer" not in result.columns

    assert "portfolio_customer_id" in result.columns


def test_missing_required_column_fails_closed():
    frame = (
        real_decisions()
        .drop(
            columns=[
                "primary_action",
            ]
        )
    )

    with pytest.raises(
        RuntimeError,
        match="incomplete",
    ):
        validate_decision_frame(
            frame
        )