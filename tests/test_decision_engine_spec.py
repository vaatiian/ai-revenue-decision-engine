import pytest

import src.decision_engine.spec as spec


# ================================================================
# METADATA
# ================================================================


def test_decision_engine_metadata_is_frozen():

    assert (
        spec.DECISION_ENGINE_NAME
        ==
        "customer_decision_engine"
    )

    assert (
        spec.DECISION_ENGINE_VERSION
        ==
        "1.0.0"
    )

    assert (
        spec.DECISION_ENGINE_TYPE
        ==
        "deterministic_orchestration"
    )

    assert (
        spec.DECISION_ENGINE_TARGET_NAME
        ==
        "customer_primary_action"
    )


def test_decision_engine_prediction_unit_is_customer_snapshot():

    assert (
        spec.DECISION_ENGINE_PREDICTION_UNIT
        ==
        "one identified customer at one decision snapshot"
    )


def test_decision_engine_governance_is_conditional_go():

    assert (
        spec.DECISION_ENGINE_FINAL_DECISION
        ==
        "CONDITIONAL GO"
    )

    assert (
        spec.DECISION_ENGINE_AUTONOMOUS_OPTIMIZATION_APPROVED
        is False
    )


# ================================================================
# LIFECYCLE CONTRACT
# ================================================================


def test_lifecycle_thresholds_match_notebook_08():

    assert (
        spec.ACTIVE_MAX_DAYS_SINCE_LAST_ORDER
        ==
        30
    )

    assert (
        spec.AT_RISK_MAX_DAYS_SINCE_LAST_ORDER
        ==
        90
    )

    assert (
        spec.DORMANT_MAX_DAYS_SINCE_LAST_ORDER
        ==
        180
    )


def test_customer_states_are_canonical_and_ordered():

    assert spec.CUSTOMER_STATES == (
        "Active",
        "At Risk",
        "Dormant",
        "Long Dormant",
        "Unclassified",
    )


def test_customer_states_are_unique():

    assert len(
        set(
            spec.CUSTOMER_STATES
        )
    ) == len(
        spec.CUSTOMER_STATES
    )


# ================================================================
# ACTION CONTRACT
# ================================================================


def test_actionable_actions_are_exact():

    assert spec.ACTIONABLE_ACTIONS == (
        "Protect",
        "Retain",
        "Reactivate",
        "Grow",
    )


def test_non_actionable_actions_are_exact():

    assert spec.NON_ACTIONABLE_ACTIONS == (
        "Monitor",
        "Monitor — No Reactivation Score",
        "Unscored — Long Dormant",
        "Unclassified",
    )


def test_actionable_and_non_actionable_actions_do_not_overlap():

    assert not (
        set(
            spec.ACTIONABLE_ACTIONS
        )
        &
        set(
            spec.NON_ACTIONABLE_ACTIONS
        )
    )


def test_active_action_policy_matches_notebook():

    policy = dict(
        spec.ALLOWED_ACTIONS_BY_CUSTOMER_STATE
    )

    assert policy[
        "Active"
    ] == (
        "Protect",
        "Grow",
        "Monitor",
    )


def test_at_risk_action_policy_matches_notebook():

    policy = dict(
        spec.ALLOWED_ACTIONS_BY_CUSTOMER_STATE
    )

    assert policy[
        "At Risk"
    ] == (
        "Retain",
        "Monitor",
    )


def test_dormant_action_policy_matches_notebook():

    policy = dict(
        spec.ALLOWED_ACTIONS_BY_CUSTOMER_STATE
    )

    assert policy[
        "Dormant"
    ] == (
        "Reactivate",
        "Monitor — No Reactivation Score",
    )


def test_long_dormant_action_policy_matches_notebook():

    policy = dict(
        spec.ALLOWED_ACTIONS_BY_CUSTOMER_STATE
    )

    assert policy[
        "Long Dormant"
    ] == (
        "Unscored — Long Dormant",
    )


# ================================================================
# TEMPORAL POLICY
# ================================================================


def test_reactivation_signal_age_is_90_days():

    assert (
        spec.REACTIVATION_MAX_SIGNAL_AGE_DAYS
        ==
        90
    )


def test_repeat_purchase_horizon_is_90_days():

    assert (
        spec.REPEAT_PURCHASE_CURRENT_SIGNAL_HORIZON_DAYS
        ==
        90
    )


def test_repeat_purchase_is_not_used_as_mature_current_state_signal():

    assert (
        spec.REPEAT_PURCHASE_CURRENT_STATE_ORCHESTRATION_ENABLED
        is False
    )


def test_demand_forecast_is_not_customer_action_signal():

    assert (
        spec.DEMAND_FORECAST_CUSTOMER_ACTION_ORCHESTRATION_ENABLED
        is False
    )


# ================================================================
# PROTECT POLICY
# ================================================================


def test_high_leakage_exposure_prefixes_are_p1_and_p2():

    assert (
        spec.HIGH_LEAKAGE_EXPOSURE_PREFIXES
        ==
        (
            "P1",
            "P2",
        )
    )


def test_protect_priority_weights_match_notebook():

    assert (
        spec.PROTECT_PRIORITY_WEIGHTS
        ==
        (
            (
                "leakage_exposure_percentile",
                0.60,
            ),
            (
                "leakage_probability",
                0.25,
            ),
            (
                "fcv_percentile",
                0.15,
            ),
        )
    )


# ================================================================
# RETAIN POLICY
# ================================================================


def test_retain_priority_weights_match_notebook():

    assert (
        spec.RETAIN_PRIORITY_WEIGHTS
        ==
        (
            (
                "leakage_exposure_percentile",
                0.50,
            ),
            (
                "leakage_probability",
                0.30,
            ),
            (
                "fcv_percentile",
                0.20,
            ),
        )
    )


# ================================================================
# REACTIVATION POLICY
# ================================================================


def test_reactivate_priority_weights_match_notebook():

    assert (
        spec.REACTIVATE_PRIORITY_WEIGHTS
        ==
        (
            (
                "reactivation_probability",
                0.65,
            ),
            (
                "fcv_percentile",
                0.35,
            ),
        )
    )


# ================================================================
# GROW POLICY
# ================================================================


def test_grow_priority_weights_match_notebook():

    assert (
        spec.GROW_PRIORITY_WEIGHTS
        ==
        (
            (
                "fcv_percentile",
                0.55,
            ),
            (
                "future_active_probability",
                0.45,
            ),
        )
    )


@pytest.mark.parametrize(
    "weights",
    [
        spec.PROTECT_PRIORITY_WEIGHTS,
        spec.RETAIN_PRIORITY_WEIGHTS,
        spec.REACTIVATE_PRIORITY_WEIGHTS,
        spec.GROW_PRIORITY_WEIGHTS,
    ],
)
def test_each_priority_policy_sums_to_one(
    weights,
):

    assert sum(
        weight
        for _,
        weight
        in weights
    ) == pytest.approx(
        1.0
    )


# ================================================================
# PRIORITY TIERS
# ================================================================


def test_priority_thresholds_match_notebook():

    assert (
        spec.PRIORITY_TIER_THRESHOLDS
        ==
        (
            (
                "Priority 1",
                0.90,
            ),
            (
                "Priority 2",
                0.70,
            ),
            (
                "Priority 3",
                0.40,
            ),
        )
    )


def test_priority_tier_labels_are_exact():

    assert (
        spec.PRIORITY_TIERS
        ==
        (
            "Priority 1",
            "Priority 2",
            "Priority 3",
            "Priority 4",
            "Not Prioritised",
        )
    )


# ================================================================
# CONTACT POLICY
# ================================================================


def test_contact_intensity_policy_matches_notebook():

    assert (
        spec.CONTACT_INTENSITY_BY_PRIORITY_TIER
        ==
        (
            (
                "Priority 1",
                "High Touch",
            ),
            (
                "Priority 2",
                "Campaign",
            ),
            (
                "Priority 3",
                "Automated / Low Cost",
            ),
            (
                "Priority 4",
                "Hold / Control",
            ),
        )
    )


def test_no_action_contact_policy_exists():

    assert (
        spec.CONTACT_NO_MODEL_LED
        ==
        "No Model-Led Contact"
    )


# ================================================================
# RECOMMENDATION POLICY
# ================================================================


def test_recommendation_channel_policy_matches_notebook():

    assert dict(
        spec.RECOMMENDATION_CHANNEL_BY_ACTION
    ) == {
        "Protect":
            "Repeat",

        "Retain":
            "Repeat",

        "Grow":
            "Discovery",
    }


def test_reactivate_has_no_product_recommendation_channel():

    policy = dict(
        spec.RECOMMENDATION_CHANNEL_BY_ACTION
    )

    assert (
        "Reactivate"
        not in policy
    )


# ================================================================
# EXPERIMENTATION POLICY
# ================================================================


def test_experiment_policy_matches_notebook():

    policy = dict(
        spec.EXPERIMENT_REQUIREMENT_BY_ACTION
    )

    assert (
        policy[
            "Reactivate"
        ]
        ==
        "Required"
    )

    assert (
        policy[
            "Grow"
        ]
        ==
        "Required — Discovery channel is CONDITIONAL GO"
    )

    assert (
        policy[
            "Retain"
        ]
        ==
        "Recommended"
    )

    assert (
        policy[
            "Protect"
        ]
        ==
        "Recommended"
    )


def test_every_actionable_action_has_experiment_policy():

    assert set(
        dict(
            spec.EXPERIMENT_REQUIREMENT_BY_ACTION
        )
    ) == set(
        spec.ACTIONABLE_ACTIONS
    )


# ================================================================
# OUTPUT CONTRACT
# ================================================================


def test_decision_output_begins_with_customer_and_date():

    assert (
        spec.DECISION_OUTPUT_COLUMNS[
            :2
        ]
        ==
        (
            "customer",
            "decision_date",
        )
    )


def test_decision_output_columns_are_unique():

    assert len(
        spec.DECISION_OUTPUT_COLUMNS
    ) == len(
        set(
            spec.DECISION_OUTPUT_COLUMNS
        )
    )


def test_decision_output_contains_lifecycle_fields():

    required = {
        "customer_state",
        "days_since_last_order",
        "customer_tenure_days",
        "lifetime_orders",
        "lifetime_revenue",
        "orders_prior_30d",
        "orders_prior_90d",
        "revenue_prior_90d",
        "revenue_prior_180d",
    }

    assert required.issubset(
        spec.DECISION_OUTPUT_COLUMNS
    )


def test_decision_output_contains_action_fields():

    required = {
        "primary_action",
        "primary_action_score",
        "action_priority_percentile",
        "priority_tier",
        "contact_intensity",
        "business_action",
        "decision_reason",
        "experiment_requirement",
    }

    assert required.issubset(
        spec.DECISION_OUTPUT_COLUMNS
    )


def test_decision_output_contains_leakage_fields():

    required = {
        "leakage_probability",
        "leakage_exposure_score",
        "leakage_exposure_percentile",
        "economic_priority_band",
    }

    assert required.issubset(
        spec.DECISION_OUTPUT_COLUMNS
    )


def test_decision_output_contains_fcv_fields():

    required = {
        "future_active_probability",
        "future_customer_value_90d",
        "fcv_percentile",
        "fcv_tier",
    }

    assert required.issubset(
        spec.DECISION_OUTPUT_COLUMNS
    )


def test_decision_output_contains_reactivation_fields():

    required = {
        "reactivation_probability",
        "reactivation_prediction_date",
    }

    assert required.issubset(
        spec.DECISION_OUTPUT_COLUMNS
    )


def test_decision_output_contains_recommendation_fields():

    required = {
        "recommended_product_channel",
        "recommended_product",
        "repeat_top1_product",
        "discovery_top1_product",
    }

    assert required.issubset(
        spec.DECISION_OUTPUT_COLUMNS
    )


# ================================================================
# FAIL-CLOSED SPEC VALIDATION
# ================================================================


def test_specification_validation_passes():

    assert (
        spec.validate_decision_engine_specification()
        is None
    )