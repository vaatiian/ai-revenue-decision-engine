import numpy as np
import pandas as pd
import pytest

from src.decision_engine.engine import (
    build_customer_decisions,
    validate_decision_engine_output,
)

from src.decision_engine.spec import (
    ACTION_GROW,
    ACTION_MONITOR,
    ACTION_MONITOR_NO_REACTIVATION_SCORE,
    ACTION_PROTECT,
    ACTION_REACTIVATE,
    ACTION_RETAIN,
    ACTION_UNSCORED_LONG_DORMANT,
    DECISION_OUTPUT_COLUMNS,
    NOT_PRIORITISED,
    RECOMMENDATION_CHANNEL_DISCOVERY,
    RECOMMENDATION_CHANNEL_REPEAT,
)


DECISION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# TEST DATA
# ================================================================


def make_transactions():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    # Active
                    "2011-08-31 10:00:00",  # A
                    "2011-08-25 10:00:00",  # B
                    "2011-08-20 10:00:00",  # G
                    "2011-08-29 10:00:00",  # I
                    "2011-08-15 10:00:00",  # J

                    # At Risk
                    "2011-07-15 10:00:00",  # C
                    "2011-07-01 10:00:00",  # H

                    # Dormant
                    "2011-05-15 10:00:00",  # D
                    "2011-04-15 10:00:00",  # E

                    # Long Dormant
                    "2011-01-15 10:00:00",  # F

                    # Future-only customer: must not enter snapshot
                    "2011-09-15 10:00:00",  # Z
                ]
            ),

            "customer": [
                "A",
                "B",
                "G",
                "I",
                "J",
                "C",
                "H",
                "D",
                "E",
                "F",
                "Z",
            ],

            "invoice": [
                "A1",
                "B1",
                "G1",
                "I1",
                "J1",
                "C1",
                "H1",
                "D1",
                "E1",
                "F1",
                "Z1",
            ],

            "quantity": [
                1
            ] * 11,

            "unit_price": [
                10.0
            ] * 11,

            "net_merchandise_revenue": [
                10.0
            ] * 11,

            "is_cancellation": [
                False
            ] * 11,

            "is_return": [
                False
            ] * 11,
        }
    )


def make_leakage_scores():

    probabilities = {
        "A": 0.95,
        "B": 0.20,
        "C": 0.80,
        "D": 0.40,
        "E": 0.30,
        "F": 0.50,
        "G": 0.10,
        "H": 0.70,
        "I": 0.90,
        "J": 0.25,
    }

    return pd.DataFrame(
        {
            "customer": list(
                probabilities.keys()
            ),
            "revenue_leakage_probability": list(
                probabilities.values()
            ),
        }
    )


def make_leakage_context():

    baselines = {
        "A": 1000.0,
        "B": 200.0,
        "C": 800.0,
        "D": 400.0,
        "E": 300.0,
        "F": 500.0,
        "G": 100.0,
        "H": 700.0,
        "I": 900.0,
        "J": 250.0,
    }

    return pd.DataFrame(
        {
            "customer": list(
                baselines.keys()
            ),
            "baseline_revenue": list(
                baselines.values()
            ),
        }
    )


def make_fcv_scores():

    customers = list(
        "ABCDEFGHIJ"
    )

    return pd.DataFrame(
        {
            "customer": customers,

            "fcv_history_180d_revenue": [
                1000.0,
                500.0,
                800.0,
                400.0,
                300.0,
                200.0,
                250.0,
                700.0,
                900.0,
                600.0,
            ],

            "future_active_probability": [
                0.95,
                0.90,
                0.70,
                0.50,
                0.40,
                0.30,
                0.45,
                0.65,
                0.92,
                0.88,
            ],

            "future_customer_value_90d": [
                1000.0,
                700.0,
                850.0,
                500.0,
                350.0,
                200.0,
                300.0,
                800.0,
                950.0,
                750.0,
            ],
        }
    )


def make_reactivation_scores():

    return pd.DataFrame(
        {
            "customer": [
                "D",
                "F",
            ],

            "reactivation_probability": [
                0.80,
                0.60,
            ],
        }
    )


def make_reactivation_features():

    return pd.DataFrame(
        {
            "customer": [
                "D",
                "F",
            ],

            "prediction_date": pd.to_datetime(
                [
                    DECISION_DATE,
                    DECISION_DATE,
                ]
            ),
        }
    )


def make_recommendation_scores():

    return pd.DataFrame(
        {
            "customer": [
                "A",
                "B",
                "C",
                "H",
                "I",
                "J",

                # Older B episode proves latest valid episode selection.
                "B",
            ],

            "invoice": [
                "A1",
                "B1",
                "C1",
                "H1",
                "I1",
                "J1",
                "B0",
            ],

            "prediction_date": pd.to_datetime(
                [
                    "2011-08-31 10:00:00",
                    "2011-08-25 10:00:00",
                    "2011-07-15 10:00:00",
                    "2011-07-01 10:00:00",
                    "2011-08-29 10:00:00",
                    "2011-08-15 10:00:00",
                    "2011-07-01 10:00:00",
                ]
            ),

            "repeat_top1_product": [
                "A-REPEAT",
                "B-REPEAT",
                "C-REPEAT",
                "H-REPEAT",
                "I-REPEAT",
                "J-REPEAT",
                "B-OLD-REPEAT",
            ],

            "discovery_top1_product": [
                "A-DISCOVERY",
                "B-DISCOVERY",
                "C-DISCOVERY",
                "H-DISCOVERY",
                "I-DISCOVERY",
                "J-DISCOVERY",
                "B-OLD-DISCOVERY",
            ],
        }
    )


def build_full_engine():

    return build_customer_decisions(
        transactions=(
            make_transactions()
        ),
        decision_date=DECISION_DATE,
        leakage_scores=(
            make_leakage_scores()
        ),
        leakage_context=(
            make_leakage_context()
        ),
        fcv_scores=(
            make_fcv_scores()
        ),
        reactivation_scores=(
            make_reactivation_scores()
        ),
        reactivation_features=(
            make_reactivation_features()
        ),
        recommendation_scores=(
            make_recommendation_scores()
        ),
    )


def row_for(
    output,
    customer,
):

    return (
        output.loc[
            output[
                "customer"
            ]
            ==
            customer
        ]
        .iloc[0]
    )


# ================================================================
# FINAL CONTRACT
# ================================================================


def test_engine_output_columns_match_frozen_contract_exactly():

    output = (
        build_full_engine()
    )

    assert tuple(
        output.columns
    ) == DECISION_OUTPUT_COLUMNS


def test_engine_output_has_one_row_per_customer():

    output = (
        build_full_engine()
    )

    assert (
        output[
            "customer"
        ].is_unique
    )


def test_engine_preserves_snapshot_population():

    output = (
        build_full_engine()
    )

    assert len(
        output
    ) == 10

    assert set(
        output[
            "customer"
        ]
    ) == set(
        "ABCDEFGHIJ"
    )


def test_future_only_customer_is_excluded_end_to_end():

    output = (
        build_full_engine()
    )

    assert (
        "Z"
        not in
        set(
            output[
                "customer"
            ]
        )
    )


def test_decision_date_is_constant():

    output = (
        build_full_engine()
    )

    dates = pd.to_datetime(
        output[
            "decision_date"
        ]
    )

    assert dates.nunique() == 1

    assert (
        dates.iloc[0]
        ==
        DECISION_DATE
    )


# ================================================================
# END-TO-END ACTIONS
# ================================================================


def test_protect_action_works_end_to_end():

    output = (
        build_full_engine()
    )

    row = row_for(
        output,
        "A",
    )

    assert (
        row[
            "customer_state"
        ]
        ==
        "Active"
    )

    assert (
        row[
            "primary_action"
        ]
        ==
        ACTION_PROTECT
    )

    assert (
        str(
            row[
                "economic_priority_band"
            ]
        )
        .startswith(
            "P1"
        )
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
        "A-REPEAT"
    )


def test_grow_action_works_end_to_end():

    output = (
        build_full_engine()
    )

    row = row_for(
        output,
        "B",
    )

    assert (
        row[
            "customer_state"
        ]
        ==
        "Active"
    )

    assert (
        row[
            "primary_action"
        ]
        ==
        ACTION_GROW
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
        "B-DISCOVERY"
    )


def test_retain_action_works_end_to_end():

    output = (
        build_full_engine()
    )

    row = row_for(
        output,
        "C",
    )

    assert (
        row[
            "customer_state"
        ]
        ==
        "At Risk"
    )

    assert (
        row[
            "primary_action"
        ]
        ==
        ACTION_RETAIN
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
        "C-REPEAT"
    )


def test_reactivate_action_works_end_to_end():

    output = (
        build_full_engine()
    )

    row = row_for(
        output,
        "D",
    )

    assert (
        row[
            "customer_state"
        ]
        ==
        "Dormant"
    )

    assert (
        row[
            "primary_action"
        ]
        ==
        ACTION_REACTIVATE
    )

    assert (
        row[
            "reactivation_probability"
        ]
        ==
        pytest.approx(
            0.80
        )
    )

    assert pd.isna(
        row[
            "recommended_product"
        ]
    )


def test_dormant_without_reactivation_is_monitor_no_score():

    output = (
        build_full_engine()
    )

    row = row_for(
        output,
        "E",
    )

    assert (
        row[
            "primary_action"
        ]
        ==
        ACTION_MONITOR_NO_REACTIVATION_SCORE
    )

    assert (
        row[
            "priority_tier"
        ]
        ==
        NOT_PRIORITISED
    )


def test_long_dormant_never_receives_model_led_action():

    output = (
        build_full_engine()
    )

    row = row_for(
        output,
        "F",
    )

    assert (
        row[
            "customer_state"
        ]
        ==
        "Long Dormant"
    )

    assert (
        row[
            "primary_action"
        ]
        ==
        ACTION_UNSCORED_LONG_DORMANT
    )

    assert pd.isna(
        row[
            "primary_action_score"
        ]
    )


def test_active_customer_without_supported_intervention_is_monitor():

    output = (
        build_full_engine()
    )

    row = row_for(
        output,
        "G",
    )

    assert (
        row[
            "customer_state"
        ]
        ==
        "Active"
    )

    assert (
        row[
            "primary_action"
        ]
        ==
        ACTION_MONITOR
    )


# ================================================================
# PRIORITY / CONTACT INTEGRATION
# ================================================================


def test_actionable_customers_receive_priority_policy():

    output = (
        build_full_engine()
    )

    actionable = (
        output.loc[
            output[
                "primary_action"
            ]
            .isin(
                [
                    ACTION_PROTECT,
                    ACTION_RETAIN,
                    ACTION_REACTIVATE,
                    ACTION_GROW,
                ]
            )
        ]
    )

    assert actionable[
        "priority_tier"
    ].notna().all()

    assert actionable[
        "contact_intensity"
    ].notna().all()


def test_non_actionable_customers_have_no_action_score():

    output = (
        build_full_engine()
    )

    non_actionable = (
        output.loc[
            ~output[
                "primary_action"
            ]
            .isin(
                [
                    ACTION_PROTECT,
                    ACTION_RETAIN,
                    ACTION_REACTIVATE,
                    ACTION_GROW,
                ]
            )
        ]
    )

    assert (
        non_actionable[
            "primary_action_score"
        ]
        .isna()
        .all()
    )

    assert (
        non_actionable[
            "action_priority_percentile"
        ]
        .isna()
        .all()
    )

    assert (
        non_actionable[
            "priority_tier"
        ]
        ==
        NOT_PRIORITISED
    ).all()


def test_priority_ranking_is_action_specific_end_to_end():

    output = (
        build_full_engine()
    )

    protect = (
        output.loc[
            output[
                "primary_action"
            ]
            ==
            ACTION_PROTECT
        ]
        .sort_values(
            "primary_action_score"
        )
    )

    assert len(
        protect
    ) == 2

    assert np.allclose(
        protect[
            "action_priority_percentile"
        ].to_numpy(),
        np.array(
            [
                0.50,
                1.00,
            ]
        ),
    )


# ================================================================
# RECOMMENDATION TEMPORAL INTEGRATION
# ================================================================


def test_latest_recommendation_episode_is_used_end_to_end():

    output = (
        build_full_engine()
    )

    row = row_for(
        output,
        "B",
    )

    assert (
        row[
            "recommended_product"
        ]
        ==
        "B-DISCOVERY"
    )

    assert (
        row[
            "recommended_product"
        ]
        !=
        "B-OLD-DISCOVERY"
    )


# ================================================================
# FINAL GOVERNANCE VALIDATION
# ================================================================


def test_valid_integrated_output_passes_final_validation():

    output = (
        build_full_engine()
    )

    assert (
        validate_decision_engine_output(
            output,
            decision_date=DECISION_DATE,
        )
        is None
    )


def test_final_validation_rejects_duplicate_customer():

    output = (
        build_full_engine()
    )

    broken = pd.concat(
        [
            output,
            output.iloc[
                [0]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per customer",
    ):
        validate_decision_engine_output(
            broken,
            decision_date=DECISION_DATE,
        )


def test_final_validation_rejects_wrong_column_order():

    output = (
        build_full_engine()
    )

    columns = list(
        output.columns
    )

    columns[
        0
    ], columns[
        1
    ] = (
        columns[
            1
        ],
        columns[
            0
        ],
    )

    broken = (
        output.loc[
            :,
            columns,
        ]
    )

    with pytest.raises(
        ValueError,
        match="canonical",
    ):
        validate_decision_engine_output(
            broken,
            decision_date=DECISION_DATE,
        )


def test_final_validation_rejects_invalid_state_action_pair():

    output = (
        build_full_engine()
    )

    broken = (
        output.copy()
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
        match="state/action governance",
    ):
        validate_decision_engine_output(
            broken,
            decision_date=DECISION_DATE,
        )


def test_final_validation_rejects_grow_without_product():

    output = (
        build_full_engine()
    )

    broken = (
        output.copy()
    )

    broken.loc[
        broken[
            "primary_action"
        ]
        ==
        ACTION_GROW,
        "recommended_product",
    ] = pd.NA

    with pytest.raises(
        ValueError,
        match="Grow customers must have a recommended product",
    ):
        validate_decision_engine_output(
            broken,
            decision_date=DECISION_DATE,
        )


def test_final_validation_rejects_reactivate_without_probability():

    output = (
        build_full_engine()
    )

    broken = (
        output.copy()
    )

    broken.loc[
        broken[
            "primary_action"
        ]
        ==
        ACTION_REACTIVATE,
        "reactivation_probability",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="Reactivate customers must have",
    ):
        validate_decision_engine_output(
            broken,
            decision_date=DECISION_DATE,
        )


# ================================================================
# INTEGRATION FAIL-CLOSED BEHAVIOR
# ================================================================


def test_engine_requires_leakage_context_when_scores_supplied():

    with pytest.raises(
        ValueError,
        match="Leakage context is required",
    ):
        build_customer_decisions(
            transactions=(
                make_transactions()
            ),
            decision_date=DECISION_DATE,
            leakage_scores=(
                make_leakage_scores()
            ),
        )


def test_engine_requires_reactivation_features_when_scores_supplied():

    with pytest.raises(
        ValueError,
        match="Reactivation features are required",
    ):
        build_customer_decisions(
            transactions=(
                make_transactions()
            ),
            decision_date=DECISION_DATE,
            reactivation_scores=(
                make_reactivation_scores()
            ),
        )


# ================================================================
# EMPTY INPUT
# ================================================================


def test_empty_transactions_return_exact_empty_canonical_output():

    empty = (
        make_transactions()
        .iloc[
            0:0
        ]
        .copy()
    )

    output = (
        build_customer_decisions(
            transactions=empty,
            decision_date=DECISION_DATE,
        )
    )

    assert output.empty

    assert tuple(
        output.columns
    ) == DECISION_OUTPUT_COLUMNS