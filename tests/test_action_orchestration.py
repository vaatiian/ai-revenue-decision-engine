import pandas as pd
import pytest

from src.decision_engine.action_orchestration import (
    ACTION_ORCHESTRATION_COLUMNS,
    ACTION_ORCHESTRATION_REQUIRED_COLUMNS,
    build_action_orchestration,
    validate_action_orchestration_input,
    validate_action_orchestration_output,
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
)


# ================================================================
# TEST DATA
# ================================================================


def make_aligned_snapshot():

    return pd.DataFrame(
        {
            "customer": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            ],

            "customer_state": [
                "Active",
                "Active",
                "Active",
                "At Risk",
                "At Risk",
                "Dormant",
                "Dormant",
                "Long Dormant",
                "Unclassified",
                "Active",
            ],

            "economic_priority_band": [
                "P1 — Highest Exposure",
                "P3",
                "P5 — Lowest Exposure",
                "P2",
                pd.NA,
                pd.NA,
                pd.NA,
                "P1 — Highest Exposure",
                pd.NA,
                "P2",
            ],

            "leakage_operationally_valid": [
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],

            "fcv_signal_available": [
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
            ],

            "reactivation_operationally_valid": [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
            ],

            "recommendation_operationally_valid": [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )


def action_for(
    result,
    customer,
):

    return (
        result.loc[
            result[
                "customer"
            ]
            ==
            customer,
            "primary_action",
        ]
        .iloc[0]
    )


# ================================================================
# CONTRACT
# ================================================================


def test_required_columns_are_frozen():

    assert (
        ACTION_ORCHESTRATION_REQUIRED_COLUMNS
        ==
        (
            "customer",
            "customer_state",
            "economic_priority_band",
            "leakage_operationally_valid",
            "fcv_signal_available",
            "reactivation_operationally_valid",
            "recommendation_operationally_valid",
        )
    )


def test_output_columns_are_frozen():

    assert (
        ACTION_ORCHESTRATION_COLUMNS
        ==
        (
            "high_leakage_exposure",
            "protect_eligible",
            "grow_eligible",
            "retain_eligible",
            "reactivate_eligible",
            "primary_action",
        )
    )


# ================================================================
# HIGH LEAKAGE EXPOSURE
# ================================================================


def test_p1_is_high_leakage_exposure():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    observed = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "A",
            "high_leakage_exposure",
        ]
        .iloc[0]
    )

    assert bool(
        observed
    )


def test_p2_is_high_leakage_exposure():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    observed = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "D",
            "high_leakage_exposure",
        ]
        .iloc[0]
    )

    assert bool(
        observed
    )


def test_p3_is_not_high_leakage_exposure():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    observed = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "B",
            "high_leakage_exposure",
        ]
        .iloc[0]
    )

    assert not bool(
        observed
    )


def test_missing_band_is_not_high_leakage_exposure():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    observed = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "E",
            "high_leakage_exposure",
        ]
        .iloc[0]
    )

    assert not bool(
        observed
    )


# ================================================================
# PROTECT
# ================================================================


def test_active_high_exposure_customer_is_protect_eligible():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    eligible = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "A",
            "protect_eligible",
        ]
        .iloc[0]
    )

    assert bool(
        eligible
    )

    assert (
        action_for(
            result,
            "A",
        )
        ==
        ACTION_PROTECT
    )


def test_protect_takes_precedence_over_grow():

    snapshot = (
        make_aligned_snapshot()
    )

    row = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert bool(
        row[
            "recommendation_operationally_valid"
        ]
    )

    assert bool(
        row[
            "fcv_signal_available"
        ]
    )

    result = (
        build_action_orchestration(
            snapshot
        )
    )

    assert (
        action_for(
            result,
            "A",
        )
        ==
        ACTION_PROTECT
    )

    grow_eligible = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "A",
            "grow_eligible",
        ]
        .iloc[0]
    )

    assert not bool(
        grow_eligible
    )


def test_high_exposure_without_valid_leakage_is_not_protect():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    eligible = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "J",
            "protect_eligible",
        ]
        .iloc[0]
    )

    assert not bool(
        eligible
    )


# ================================================================
# GROW
# ================================================================


def test_active_customer_can_grow_when_not_protect():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    eligible = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "B",
            "grow_eligible",
        ]
        .iloc[0]
    )

    assert bool(
        eligible
    )

    assert (
        action_for(
            result,
            "B",
        )
        ==
        ACTION_GROW
    )


def test_active_p2_without_valid_leakage_can_grow():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        action_for(
            result,
            "J",
        )
        ==
        ACTION_GROW
    )


def test_grow_requires_fcv_signal():

    snapshot = (
        make_aligned_snapshot()
    )

    snapshot.loc[
        snapshot[
            "customer"
        ]
        ==
        "B",
        "fcv_signal_available",
    ] = False

    result = (
        build_action_orchestration(
            snapshot
        )
    )

    assert (
        action_for(
            result,
            "B",
        )
        ==
        ACTION_MONITOR
    )


def test_grow_requires_operational_recommendation():

    snapshot = (
        make_aligned_snapshot()
    )

    snapshot.loc[
        snapshot[
            "customer"
        ]
        ==
        "B",
        "recommendation_operationally_valid",
    ] = False

    result = (
        build_action_orchestration(
            snapshot
        )
    )

    assert (
        action_for(
            result,
            "B",
        )
        ==
        ACTION_MONITOR
    )


def test_active_without_intervention_signal_is_monitor():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        action_for(
            result,
            "C",
        )
        ==
        ACTION_MONITOR
    )


# ================================================================
# RETAIN
# ================================================================


def test_at_risk_with_valid_leakage_is_retain():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        action_for(
            result,
            "D",
        )
        ==
        ACTION_RETAIN
    )

    eligible = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "D",
            "retain_eligible",
        ]
        .iloc[0]
    )

    assert bool(
        eligible
    )


def test_at_risk_without_valid_leakage_is_monitor():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        action_for(
            result,
            "E",
        )
        ==
        ACTION_MONITOR
    )


# ================================================================
# REACTIVATE
# ================================================================


def test_dormant_with_valid_reactivation_is_reactivate():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        action_for(
            result,
            "F",
        )
        ==
        ACTION_REACTIVATE
    )

    eligible = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "F",
            "reactivate_eligible",
        ]
        .iloc[0]
    )

    assert bool(
        eligible
    )


def test_dormant_without_valid_reactivation_is_monitor_no_score():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        action_for(
            result,
            "G",
        )
        ==
        ACTION_MONITOR_NO_REACTIVATION_SCORE
    )


# ================================================================
# LONG DORMANT / UNCLASSIFIED
# ================================================================


def test_long_dormant_is_always_unscored():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        action_for(
            result,
            "H",
        )
        ==
        ACTION_UNSCORED_LONG_DORMANT
    )


def test_long_dormant_reactivation_signal_does_not_become_action():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    reactivation_valid = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "H",
            "reactivation_operationally_valid",
        ]
        .iloc[0]
    )

    assert bool(
        reactivation_valid
    )

    assert (
        action_for(
            result,
            "H",
        )
        ==
        ACTION_UNSCORED_LONG_DORMANT
    )


def test_unclassified_customer_remains_unclassified():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        action_for(
            result,
            "I",
        )
        ==
        ACTION_UNCLASSIFIED
    )


# ================================================================
# ELIGIBILITY EXCLUSIVITY
# ================================================================


def test_no_customer_is_eligible_for_multiple_primary_actions():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    counts = (
        result[
            [
                "protect_eligible",
                "grow_eligible",
                "retain_eligible",
                "reactivate_eligible",
            ]
        ]
        .astype(int)
        .sum(
            axis=1
        )
    )

    assert (
        counts
        <=
        1
    ).all()


def test_non_active_customer_cannot_be_grow_eligible():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    non_active = (
        result.loc[
            result[
                "customer_state"
            ]
            !=
            "Active"
        ]
    )

    assert not (
        non_active[
            "grow_eligible"
        ]
        .any()
    )


def test_only_active_customer_can_be_protect_eligible():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    protect_rows = (
        result.loc[
            result[
                "protect_eligible"
            ]
        ]
    )

    assert (
        protect_rows[
            "customer_state"
        ]
        ==
        "Active"
    ).all()


# ================================================================
# POPULATION / GOVERNANCE
# ================================================================


def test_orchestration_preserves_customer_population():

    snapshot = (
        make_aligned_snapshot()
    )

    result = (
        build_action_orchestration(
            snapshot
        )
    )

    assert len(
        result
    ) == len(
        snapshot
    )

    assert set(
        result[
            "customer"
        ]
    ) == set(
        snapshot[
            "customer"
        ]
    )


def test_every_customer_receives_primary_action():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        result[
            "primary_action"
        ]
        .notna()
        .all()
    )


def test_valid_output_passes_governance_validation():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
        )
    )

    assert (
        validate_action_orchestration_output(
            result
        )
        is None
    )


def test_output_validation_rejects_invalid_state_action_pair():

    result = (
        build_action_orchestration(
            make_aligned_snapshot()
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
        "D",
        "primary_action",
    ] = ACTION_GROW

    with pytest.raises(
        ValueError,
        match="outside frozen governance",
    ):
        validate_action_orchestration_output(
            broken
        )


# ================================================================
# INPUT FAIL-CLOSED TESTS
# ================================================================


def test_missing_required_column_fails_closed():

    broken = (
        make_aligned_snapshot()
        .drop(
            columns=[
                "fcv_signal_available"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        build_action_orchestration(
            broken
        )


def test_duplicate_customer_fails_closed():

    snapshot = (
        make_aligned_snapshot()
    )

    broken = pd.concat(
        [
            snapshot,
            snapshot.iloc[
                [0]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per customer",
    ):
        build_action_orchestration(
            broken
        )


def test_unknown_customer_state_fails_closed():

    broken = (
        make_aligned_snapshot()
    )

    broken.loc[
        0,
        "customer_state",
    ] = "Unknown State"

    with pytest.raises(
        ValueError,
        match="unknown customer states",
    ):
        build_action_orchestration(
            broken
        )


def test_missing_boolean_signal_fails_closed():

    broken = (
        make_aligned_snapshot()
    )

    broken[
        "leakage_operationally_valid"
    ] = (
        broken[
            "leakage_operationally_valid"
        ]
        .astype(
            "object"
        )
    )

    broken.loc[
        0,
        "leakage_operationally_valid",
    ] = pd.NA

    with pytest.raises(
        ValueError,
        match="missing leakage_operationally_valid",
    ):
        build_action_orchestration(
            broken
        )


def test_non_boolean_signal_fails_closed():

    broken = (
        make_aligned_snapshot()
    )

    broken[
        "recommendation_operationally_valid"
    ] = broken[
        "recommendation_operationally_valid"
    ].astype(
        "object"
    )

    broken.loc[
        0,
        "recommendation_operationally_valid",
    ] = "yes"

    with pytest.raises(
        ValueError,
        match="must contain boolean values",
    ):
        build_action_orchestration(
            broken
        )


def test_input_validator_accepts_valid_frame():

    assert (
        validate_action_orchestration_input(
            make_aligned_snapshot()
        )
        is None
    )


# ================================================================
# EMPTY INPUT
# ================================================================


def test_empty_input_preserves_orchestration_schema():

    empty = (
        make_aligned_snapshot()
        .iloc[
            0:0
        ]
        .copy()
    )

    result = (
        build_action_orchestration(
            empty
        )
    )

    assert result.empty

    assert set(
        ACTION_ORCHESTRATION_COLUMNS
    ).issubset(
        result.columns
    )