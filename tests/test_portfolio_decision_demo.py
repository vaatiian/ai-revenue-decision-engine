import pandas as pd
import pytest

from portfolio.decision_demo import (
    DEMO_CUSTOMER,
    REFERENCE_SCORE_LEVELS,
    DecisionDemoInputs,
    build_demo_aligned_input,
    build_priority_reference_cohort,
    demo_presets,
    run_decision_demo,
)

from src.decision_engine.action_orchestration import (
    build_action_orchestration,
)

from src.decision_engine.spec import (
    ACTION_GROW,
    ACTION_MONITOR_NO_REACTIVATION_SCORE,
    ACTION_PROTECT,
    ACTION_REACTIVATE,
    ACTION_RETAIN,
    ACTION_UNCLASSIFIED,
    ACTION_UNSCORED_LONG_DORMANT,
    CONTACT_NO_MODEL_LED,
    CUSTOMER_STATE_ACTIVE,
    CUSTOMER_STATE_AT_RISK,
    CUSTOMER_STATE_DORMANT,
    CUSTOMER_STATE_LONG_DORMANT,
    CUSTOMER_STATE_UNCLASSIFIED,
    EXPERIMENT_NOT_APPLICABLE,
    EXPERIMENT_RECOMMENDED,
    EXPERIMENT_REQUIRED,
    EXPERIMENT_REQUIRED_DISCOVERY,
    NOT_PRIORITISED,
    RECOMMENDATION_CHANNEL_DISCOVERY,
    RECOMMENDATION_CHANNEL_REPEAT,
)


def test_demo_uses_public_synthetic_customer_only():
    frame = build_demo_aligned_input(
        DecisionDemoInputs()
    )

    assert len(
        frame
    ) == 1

    assert (
        frame.loc[
            0,
            "customer",
        ]
        ==
        DEMO_CUSTOMER
    )


def test_demo_action_matches_direct_production_orchestration():
    inputs = DecisionDemoInputs(
        customer_state=CUSTOMER_STATE_ACTIVE,
        economic_priority_band="P1 — Highest Exposure",
        leakage_operationally_valid=True,
        leakage_exposure_percentile=0.95,
        leakage_probability=0.90,
        fcv_signal_available=True,
        fcv_percentile=0.80,
        future_active_probability=0.80,
        recommendation_operationally_valid=True,
    )

    direct = build_action_orchestration(
        build_demo_aligned_input(
            inputs
        )
    )

    demo = run_decision_demo(
        inputs
    )

    assert (
        demo[
            "primary_action"
        ]
        ==
        direct.loc[
            0,
            "primary_action",
        ]
    )


def test_active_high_exposure_is_protect():
    result = run_decision_demo(
        demo_presets()[
            "Active · Protect"
        ]
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_PROTECT
    )

    assert (
        result[
            "eligibility"
        ][
            "protect_eligible"
        ]
        is True
    )


def test_protect_takes_precedence_over_grow():
    result = run_decision_demo(
        DecisionDemoInputs(
            customer_state=CUSTOMER_STATE_ACTIVE,
            economic_priority_band="P1 — Highest Exposure",
            leakage_operationally_valid=True,
            leakage_exposure_percentile=0.95,
            leakage_probability=0.95,
            fcv_signal_available=True,
            fcv_percentile=0.95,
            future_active_probability=0.95,
            recommendation_operationally_valid=True,
        )
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_PROTECT
    )

    assert (
        result[
            "eligibility"
        ][
            "grow_eligible"
        ]
        is False
    )


def test_active_customer_can_grow():
    result = run_decision_demo(
        demo_presets()[
            "Active · Grow"
        ]
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_GROW
    )

    assert (
        result[
            "recommended_product_channel"
        ]
        ==
        RECOMMENDATION_CHANNEL_DISCOVERY
    )


def test_at_risk_customer_can_retain():
    result = run_decision_demo(
        demo_presets()[
            "At Risk · Retain"
        ]
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_RETAIN
    )

    assert (
        result[
            "recommended_product_channel"
        ]
        ==
        RECOMMENDATION_CHANNEL_REPEAT
    )


def test_dormant_customer_can_reactivate():
    result = run_decision_demo(
        demo_presets()[
            "Dormant · Reactivate"
        ]
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_REACTIVATE
    )

    assert (
        result[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_REQUIRED
    )


def test_dormant_without_valid_score_is_monitor_no_score():
    result = run_decision_demo(
        demo_presets()[
            "Dormant · No Score"
        ]
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_MONITOR_NO_REACTIVATION_SCORE
    )

    assert (
        result[
            "priority_tier"
        ]
        ==
        NOT_PRIORITISED
    )

    assert (
        result[
            "contact_intensity"
        ]
        ==
        CONTACT_NO_MODEL_LED
    )


def test_long_dormant_remains_unscored_despite_all_signals():
    result = run_decision_demo(
        demo_presets()[
            "Long Dormant"
        ]
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_UNSCORED_LONG_DORMANT
    )

    assert (
        result[
            "priority_tier"
        ]
        ==
        NOT_PRIORITISED
    )

    assert (
        result[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_NOT_APPLICABLE
    )


def test_unclassified_remains_unclassified():
    result = run_decision_demo(
        demo_presets()[
            "Unclassified"
        ]
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_UNCLASSIFIED
    )


def test_protect_experiment_is_recommended():
    result = run_decision_demo(
        demo_presets()[
            "Active · Protect"
        ]
    )

    assert (
        result[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_RECOMMENDED
    )


def test_retain_experiment_is_recommended():
    result = run_decision_demo(
        demo_presets()[
            "At Risk · Retain"
        ]
    )

    assert (
        result[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_RECOMMENDED
    )


def test_grow_experiment_preserves_discovery_conditional_go():
    result = run_decision_demo(
        demo_presets()[
            "Active · Grow"
        ]
    )

    assert (
        result[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_REQUIRED_DISCOVERY
    )


def test_reference_cohort_contains_nine_anchors_for_actionable_action():
    aligned = build_demo_aligned_input(
        demo_presets()[
            "Active · Protect"
        ]
    )

    orchestrated = build_action_orchestration(
        aligned
    )

    cohort = build_priority_reference_cohort(
        orchestrated
    )

    assert len(
        cohort
    ) == (
        1
        +
        len(
            REFERENCE_SCORE_LEVELS
        )
    )


def test_non_actionable_demo_does_not_add_reference_rows():
    aligned = build_demo_aligned_input(
        demo_presets()[
            "Long Dormant"
        ]
    )

    orchestrated = build_action_orchestration(
        aligned
    )

    cohort = build_priority_reference_cohort(
        orchestrated
    )

    assert len(
        cohort
    ) == 1


def test_high_score_demo_reaches_priority_one():
    result = run_decision_demo(
        DecisionDemoInputs(
            customer_state=CUSTOMER_STATE_ACTIVE,
            economic_priority_band="P1 — Highest Exposure",
            leakage_operationally_valid=True,
            leakage_exposure_percentile=1.0,
            leakage_probability=1.0,
            fcv_signal_available=True,
            fcv_percentile=1.0,
            future_active_probability=1.0,
            recommendation_operationally_valid=False,
        )
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_PROTECT
    )

    assert (
        result[
            "action_priority_percentile"
        ]
        ==
        pytest.approx(
            1.0
        )
    )

    assert (
        result[
            "priority_tier"
        ]
        ==
        "Priority 1"
    )


def test_reference_ranking_is_not_single_row_trivial():
    inputs = DecisionDemoInputs(
        customer_state=CUSTOMER_STATE_ACTIVE,
        economic_priority_band="P1 — Highest Exposure",
        leakage_operationally_valid=True,
        leakage_exposure_percentile=0.50,
        leakage_probability=0.50,
        fcv_signal_available=True,
        fcv_percentile=0.50,
        future_active_probability=0.50,
        recommendation_operationally_valid=False,
    )

    result = run_decision_demo(
        inputs
    )

    assert (
        result[
            "primary_action"
        ]
        ==
        ACTION_PROTECT
    )

    assert (
        result[
            "action_priority_percentile"
        ]
        <
        1.0
    )


def test_invalid_probability_fails_closed():
    with pytest.raises(
        ValueError,
        match="within \\[0, 1\\]",
    ):
        run_decision_demo(
            DecisionDemoInputs(
                leakage_probability=1.5,
            )
        )


def test_invalid_state_fails_closed():
    with pytest.raises(
        ValueError,
        match="Unsupported customer state",
    ):
        run_decision_demo(
            DecisionDemoInputs(
                customer_state="Impossible State",
            )
        )


def test_invalid_band_fails_closed():
    with pytest.raises(
        ValueError,
        match="Unsupported economic priority band",
    ):
        run_decision_demo(
            DecisionDemoInputs(
                economic_priority_band="P99",
            )
        )


def test_signal_unavailability_removes_signal_values():
    frame = build_demo_aligned_input(
        DecisionDemoInputs(
            leakage_operationally_valid=False,
            fcv_signal_available=False,
            reactivation_operationally_valid=False,
            recommendation_operationally_valid=False,
        )
    )

    assert pd.isna(
        frame.loc[
            0,
            "leakage_probability",
        ]
    )

    assert pd.isna(
        frame.loc[
            0,
            "fcv_percentile",
        ]
    )

    assert pd.isna(
        frame.loc[
            0,
            "reactivation_probability",
        ]
    )

    assert pd.isna(
        frame.loc[
            0,
            "repeat_top1_product",
        ]
    )

    assert pd.isna(
        frame.loc[
            0,
            "discovery_top1_product",
        ]
    )


def test_demo_reports_real_policy_source():
    result = run_decision_demo(
        demo_presets()[
            "Active · Protect"
        ]
    )

    assert (
        "build_action_orchestration"
        in
        result[
            "policy_source"
        ]
    )

    assert (
        "apply_priority_policy"
        in
        result[
            "policy_source"
        ]
    )