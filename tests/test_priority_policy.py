import numpy as np
import pandas as pd
import pytest

from src.decision_engine.priority_policy import (
    BUSINESS_ACTION_GROW,
    BUSINESS_ACTION_LONG_DORMANT,
    BUSINESS_ACTION_MONITOR_NO_REACTIVATION,
    BUSINESS_ACTION_PROTECT_WITH_PRODUCT,
    BUSINESS_ACTION_PROTECT_WITHOUT_PRODUCT,
    BUSINESS_ACTION_REACTIVATE,
    BUSINESS_ACTION_RETAIN_WITH_PRODUCT,
    BUSINESS_ACTION_RETAIN_WITHOUT_PRODUCT,
    DEFAULT_BUSINESS_ACTION,
    DEFAULT_DECISION_REASON,
    PRIORITY_POLICY_COLUMNS,
    PRIORITY_POLICY_REQUIRED_COLUMNS,
    apply_priority_policy,
    validate_priority_policy_input,
    validate_priority_policy_output,
)

from src.decision_engine.spec import (
    ACTION_GROW,
    ACTION_MONITOR,
    ACTION_MONITOR_NO_REACTIVATION_SCORE,
    ACTION_PROTECT,
    ACTION_REACTIVATE,
    ACTION_RETAIN,
    ACTION_UNCLASSIFIED,
    ACTION_UNSCORED_LONG_DORMANT,
    CONTACT_AUTOMATED_LOW_COST,
    CONTACT_CAMPAIGN,
    CONTACT_HIGH_TOUCH,
    CONTACT_HOLD_CONTROL,
    CONTACT_NO_MODEL_LED,
    EXPERIMENT_NOT_APPLICABLE,
    EXPERIMENT_RECOMMENDED,
    EXPERIMENT_REQUIRED,
    EXPERIMENT_REQUIRED_DISCOVERY,
    NOT_PRIORITISED,
    PRIORITY_1,
    PRIORITY_2,
    PRIORITY_3,
    PRIORITY_4,
    RECOMMENDATION_CHANNEL_DISCOVERY,
    RECOMMENDATION_CHANNEL_REPEAT,
)


# ================================================================
# TEST DATA
# ================================================================


def make_policy_input():

    return pd.DataFrame(
        {
            "customer": [
                "P1",
                "P2",
                "P3",
                "P4",
                "R1",
                "R2",
                "X1",
                "X2",
                "G1",
                "G2",
                "M1",
                "M2",
                "L1",
                "U1",
            ],

            "primary_action": [
                ACTION_PROTECT,
                ACTION_PROTECT,
                ACTION_PROTECT,
                ACTION_PROTECT,
                ACTION_RETAIN,
                ACTION_RETAIN,
                ACTION_REACTIVATE,
                ACTION_REACTIVATE,
                ACTION_GROW,
                ACTION_GROW,
                ACTION_MONITOR,
                ACTION_MONITOR_NO_REACTIVATION_SCORE,
                ACTION_UNSCORED_LONG_DORMANT,
                ACTION_UNCLASSIFIED,
            ],

            "recommendation_operationally_valid": [
                True,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
            ],

            "leakage_exposure_percentile": [
                1.00,
                0.80,
                0.60,
                0.40,
                0.90,
                0.50,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],

            "leakage_probability": [
                0.90,
                0.70,
                0.50,
                0.30,
                0.80,
                0.40,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],

            "fcv_percentile": [
                1.00,
                0.80,
                0.60,
                0.40,
                0.90,
                0.50,
                0.80,
                0.40,
                0.90,
                0.60,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],

            "future_active_probability": [
                0.90,
                0.70,
                0.50,
                0.30,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.95,
                0.50,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],

            "reactivation_probability": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.90,
                0.50,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],

            "repeat_top1_product": [
                "P-REPEAT-1",
                "P-REPEAT-2",
                "P-REPEAT-3",
                pd.NA,
                "R-REPEAT-1",
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],

            "discovery_top1_product": [
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                "G-DISCOVERY-1",
                "G-DISCOVERY-2",
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
        }
    )


def row_for(
    result,
    customer,
):

    return (
        result.loc[
            result[
                "customer"
            ]
            ==
            customer
        ]
        .iloc[0]
    )


# ================================================================
# CONTRACT
# ================================================================


def test_required_columns_are_frozen():

    assert (
        PRIORITY_POLICY_REQUIRED_COLUMNS
        ==
        (
            "customer",
            "primary_action",
            "recommendation_operationally_valid",
            "leakage_exposure_percentile",
            "leakage_probability",
            "fcv_percentile",
            "future_active_probability",
            "reactivation_probability",
            "repeat_top1_product",
            "discovery_top1_product",
        )
    )


def test_output_columns_are_unique():

    assert len(
        PRIORITY_POLICY_COLUMNS
    ) == len(
        set(
            PRIORITY_POLICY_COLUMNS
        )
    )


# ================================================================
# SCORE FORMULAS
# ================================================================


def test_protect_priority_score_matches_research_formula():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "P1",
    )

    expected = (
        0.60 * 1.00
        +
        0.25 * 0.90
        +
        0.15 * 1.00
    )

    assert (
        row[
            "protect_priority_score"
        ]
        ==
        pytest.approx(
            expected
        )
    )


def test_retain_priority_score_matches_research_formula():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "R1",
    )

    expected = (
        0.50 * 0.90
        +
        0.30 * 0.80
        +
        0.20 * 0.90
    )

    assert (
        row[
            "retain_priority_score"
        ]
        ==
        pytest.approx(
            expected
        )
    )


def test_reactivate_priority_score_matches_research_formula():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "X1",
    )

    expected = (
        0.65 * 0.90
        +
        0.35 * 0.80
    )

    assert (
        row[
            "reactivate_priority_score"
        ]
        ==
        pytest.approx(
            expected
        )
    )


def test_grow_priority_score_matches_research_formula():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "G1",
    )

    expected = (
        0.55 * 0.90
        +
        0.45 * 0.95
    )

    assert (
        row[
            "grow_priority_score"
        ]
        ==
        pytest.approx(
            expected
        )
    )


def test_primary_action_score_selects_correct_action_score():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    protect = row_for(
        result,
        "P1",
    )

    retain = row_for(
        result,
        "R1",
    )

    reactivate = row_for(
        result,
        "X1",
    )

    grow = row_for(
        result,
        "G1",
    )

    assert (
        protect[
            "primary_action_score"
        ]
        ==
        protect[
            "protect_priority_score"
        ]
    )

    assert (
        retain[
            "primary_action_score"
        ]
        ==
        retain[
            "retain_priority_score"
        ]
    )

    assert (
        reactivate[
            "primary_action_score"
        ]
        ==
        reactivate[
            "reactivate_priority_score"
        ]
    )

    assert (
        grow[
            "primary_action_score"
        ]
        ==
        grow[
            "grow_priority_score"
        ]
    )


def test_non_actionable_primary_score_is_missing():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "M1",
    )

    assert pd.isna(
        row[
            "primary_action_score"
        ]
    )


# ================================================================
# ACTION-SPECIFIC PERCENTILES
# ================================================================


def test_priority_percentile_is_ranked_within_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    protect = (
        result.loc[
            result[
                "primary_action"
            ]
            ==
            ACTION_PROTECT
        ]
        .sort_values(
            "primary_action_score"
        )
    )

    assert np.allclose(
        protect[
            "action_priority_percentile"
        ].to_numpy(),
        np.array(
            [
                0.25,
                0.50,
                0.75,
                1.00,
            ]
        ),
    )


def test_single_action_member_receives_percentile_one():

    single = (
        make_policy_input()
        .loc[
            lambda frame:
                frame[
                    "customer"
                ]
                ==
                "X1"
        ]
        .copy()
    )

    result = (
        apply_priority_policy(
            single
        )
    )

    assert (
        result[
            "action_priority_percentile"
        ]
        .iloc[0]
        ==
        pytest.approx(
            1.0
        )
    )


def test_non_actionable_percentile_is_missing():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "M1",
    )

    assert pd.isna(
        row[
            "action_priority_percentile"
        ]
    )


# ================================================================
# PRIORITY TIERS
# ================================================================


def test_non_actionable_is_not_prioritised():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "M1",
        )[
            "priority_tier"
        ]
        ==
        NOT_PRIORITISED
    )


def test_highest_protect_customer_is_priority_one():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P1",
        )[
            "priority_tier"
        ]
        ==
        PRIORITY_1
    )


def test_protect_percentile_075_is_priority_two():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P2",
        )[
            "priority_tier"
        ]
        ==
        PRIORITY_2
    )


def test_protect_percentile_050_is_priority_three():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P3",
        )[
            "priority_tier"
        ]
        ==
        PRIORITY_3
    )


def test_protect_percentile_025_is_priority_four():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P4",
        )[
            "priority_tier"
        ]
        ==
        PRIORITY_4
    )


def test_actionable_nan_score_falls_through_to_priority_four():

    frame = (
        make_policy_input()
        .loc[
            lambda data:
                data[
                    "customer"
                ]
                ==
                "X1"
        ]
        .copy()
    )

    frame[
        "fcv_percentile"
    ] = np.nan

    result = (
        apply_priority_policy(
            frame
        )
    )

    row = (
        result.iloc[0]
    )

    assert pd.isna(
        row[
            "primary_action_score"
        ]
    )

    assert pd.isna(
        row[
            "action_priority_percentile"
        ]
    )

    assert (
        row[
            "priority_tier"
        ]
        ==
        PRIORITY_4
    )


# ================================================================
# CONTACT INTENSITY
# ================================================================


def test_priority_one_gets_high_touch():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P1",
        )[
            "contact_intensity"
        ]
        ==
        CONTACT_HIGH_TOUCH
    )


def test_priority_two_gets_campaign():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P2",
        )[
            "contact_intensity"
        ]
        ==
        CONTACT_CAMPAIGN
    )


def test_priority_three_gets_automated_low_cost():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P3",
        )[
            "contact_intensity"
        ]
        ==
        CONTACT_AUTOMATED_LOW_COST
    )


def test_priority_four_gets_hold_control():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P4",
        )[
            "contact_intensity"
        ]
        ==
        CONTACT_HOLD_CONTROL
    )


def test_non_actionable_gets_no_model_led_contact():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "M1",
        )[
            "contact_intensity"
        ]
        ==
        CONTACT_NO_MODEL_LED
    )


# ================================================================
# RECOMMENDATION POLICY
# ================================================================


def test_protect_uses_repeat_channel():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "P1",
    )

    assert (
        row[
            "recommended_product_channel"
        ]
        ==
        RECOMMENDATION_CHANNEL_REPEAT
    )

    assert (
        row[
            "recommended_product"
        ]
        ==
        "P-REPEAT-1"
    )


def test_retain_uses_repeat_channel():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "R1",
    )

    assert (
        row[
            "recommended_product_channel"
        ]
        ==
        RECOMMENDATION_CHANNEL_REPEAT
    )

    assert (
        row[
            "recommended_product"
        ]
        ==
        "R-REPEAT-1"
    )


def test_grow_uses_discovery_channel():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "G1",
    )

    assert (
        row[
            "recommended_product_channel"
        ]
        ==
        RECOMMENDATION_CHANNEL_DISCOVERY
    )

    assert (
        row[
            "recommended_product"
        ]
        ==
        "G-DISCOVERY-1"
    )


def test_invalid_recommendation_signal_suppresses_product():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    row = row_for(
        result,
        "P4",
    )

    assert pd.isna(
        row[
            "recommended_product_channel"
        ]
    )

    assert pd.isna(
        row[
            "recommended_product"
        ]
    )


# ================================================================
# DECISION REASONS
# ================================================================


def test_protect_reason_matches_research():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P1",
        )[
            "decision_reason"
        ]
        ==
        "Active customer with high economic revenue-leakage exposure"
    )


def test_retain_reason_matches_research():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "R1",
        )[
            "decision_reason"
        ]
        ==
        "Customer is 31–90 days since last purchase and has a current "
        "revenue-leakage score"
    )


def test_reactivate_reason_matches_research():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "X1",
        )[
            "decision_reason"
        ]
        ==
        "Dormant customer with a temporally valid 90-day reactivation "
        "propensity score"
    )


def test_grow_reason_matches_research():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "G1",
        )[
            "decision_reason"
        ]
        ==
        "Active customer with current product recommendation and "
        "future-value signal"
    )


def test_monitor_no_reactivation_reason_matches_research():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "M2",
        )[
            "decision_reason"
        ]
        ==
        "Dormant customer but no valid Reactivation score exists at the "
        "decision snapshot"
    )


def test_long_dormant_reason_matches_research():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "L1",
        )[
            "decision_reason"
        ]
        ==
        "Customer exceeds 180 days since last purchase and lies outside "
        "current model coverage"
    )


def test_monitor_uses_default_reason():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "M1",
        )[
            "decision_reason"
        ]
        ==
        DEFAULT_DECISION_REASON
    )


# ================================================================
# BUSINESS ACTION PLAYBOOK
# ================================================================


def test_protect_with_product_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P1",
        )[
            "business_action"
        ]
        ==
        BUSINESS_ACTION_PROTECT_WITH_PRODUCT
    )


def test_protect_without_product_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P4",
        )[
            "business_action"
        ]
        ==
        BUSINESS_ACTION_PROTECT_WITHOUT_PRODUCT
    )


def test_retain_with_product_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "R1",
        )[
            "business_action"
        ]
        ==
        BUSINESS_ACTION_RETAIN_WITH_PRODUCT
    )


def test_retain_without_product_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "R2",
        )[
            "business_action"
        ]
        ==
        BUSINESS_ACTION_RETAIN_WITHOUT_PRODUCT
    )


def test_reactivate_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "X1",
        )[
            "business_action"
        ]
        ==
        BUSINESS_ACTION_REACTIVATE
    )


def test_grow_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "G1",
        )[
            "business_action"
        ]
        ==
        BUSINESS_ACTION_GROW
    )


def test_monitor_no_reactivation_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "M2",
        )[
            "business_action"
        ]
        ==
        BUSINESS_ACTION_MONITOR_NO_REACTIVATION
    )


def test_long_dormant_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "L1",
        )[
            "business_action"
        ]
        ==
        BUSINESS_ACTION_LONG_DORMANT
    )


def test_monitor_uses_default_business_action():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "M1",
        )[
            "business_action"
        ]
        ==
        DEFAULT_BUSINESS_ACTION
    )


# ================================================================
# EXPERIMENT REQUIREMENTS
# ================================================================


def test_reactivate_experiment_is_required():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "X1",
        )[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_REQUIRED
    )


def test_grow_experiment_is_discovery_conditional_go():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "G1",
        )[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_REQUIRED_DISCOVERY
    )


def test_retain_experiment_is_recommended():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "R1",
        )[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_RECOMMENDED
    )


def test_protect_experiment_is_recommended():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "P1",
        )[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_RECOMMENDED
    )


def test_non_actionable_experiment_is_not_applicable():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        row_for(
            result,
            "M1",
        )[
            "experiment_requirement"
        ]
        ==
        EXPERIMENT_NOT_APPLICABLE
    )


# ================================================================
# VALIDATION / POPULATION
# ================================================================


def test_policy_preserves_customer_population():

    frame = (
        make_policy_input()
    )

    result = (
        apply_priority_policy(
            frame
        )
    )

    assert len(
        result
    ) == len(
        frame
    )

    assert set(
        result[
            "customer"
        ]
    ) == set(
        frame[
            "customer"
        ]
    )


def test_valid_policy_output_passes_validation():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    assert (
        validate_priority_policy_output(
            result
        )
        is None
    )


def test_duplicate_customer_fails_closed():

    frame = (
        make_policy_input()
    )

    broken = pd.concat(
        [
            frame,
            frame.iloc[
                [0]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per customer",
    ):
        apply_priority_policy(
            broken
        )


def test_missing_required_column_fails_closed():

    broken = (
        make_policy_input()
        .drop(
            columns=[
                "fcv_percentile"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        apply_priority_policy(
            broken
        )


def test_unknown_primary_action_fails_closed():

    broken = (
        make_policy_input()
    )

    broken.loc[
        0,
        "primary_action",
    ] = "Unknown Action"

    with pytest.raises(
        ValueError,
        match="unknown primary actions",
    ):
        apply_priority_policy(
            broken
        )


def test_grow_without_product_fails_output_validation():

    result = (
        apply_priority_policy(
            make_policy_input()
        )
    )

    broken = (
        result.copy()
    )

    broken.loc[
        broken[
            "customer"
        ]
        ==
        "G1",
        "recommended_product",
    ] = pd.NA

    with pytest.raises(
        ValueError,
        match="Grow customers must have a recommended product",
    ):
        validate_priority_policy_output(
            broken
        )


def test_input_validator_accepts_valid_frame():

    assert (
        validate_priority_policy_input(
            make_policy_input()
        )
        is None
    )


# ================================================================
# EMPTY INPUT
# ================================================================


def test_empty_input_preserves_policy_schema():

    empty = (
        make_policy_input()
        .iloc[
            0:0
        ]
        .copy()
    )

    result = (
        apply_priority_policy(
            empty
        )
    )

    assert result.empty

    assert set(
        PRIORITY_POLICY_COLUMNS
    ).issubset(
        result.columns
    )