"""
Tests for governed delivery eligibility and suppression.
"""

import pandas as pd
import pytest

from src.decision_engine.spec import (
    ACTION_GROW,
    ACTION_MONITOR,
    ACTION_PROTECT,
    ACTION_REACTIVATE,
    ACTION_RETAIN,
    ACTION_UNSCORED_LONG_DORMANT,
    CONTACT_AUTOMATED_LOW_COST,
    CONTACT_CAMPAIGN,
    CONTACT_HIGH_TOUCH,
    CONTACT_HOLD_CONTROL,
    CONTACT_NO_MODEL_LED,
)

from src.delivery.eligibility import (
    DELIVERY_ELIGIBILITY_COLUMNS,
    apply_delivery_eligibility,
    validate_delivery_eligibility_input,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITION_HOLD_CONTROL,
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_ROUTE_AUTOMATED,
    DELIVERY_ROUTE_CAMPAIGN,
    DELIVERY_ROUTE_HIGH_TOUCH,
    DELIVERY_ROUTE_HOLD_CONTROL,
    DELIVERY_ROUTE_SUPPRESSED,
    DELIVERY_REQUIRED_DECISION_COLUMNS,
)


# ================================================================
# FIXTURE
# ================================================================


def make_decisions():

    rows = [
        {
            "customer":
                "P1",

            "decision_date":
                "2011-08-31",

            "customer_state":
                "Active",

            "primary_action":
                ACTION_PROTECT,

            "primary_action_score":
                0.95,

            "action_priority_percentile":
                0.95,

            "priority_tier":
                "Priority 1",

            "contact_intensity":
                CONTACT_HIGH_TOUCH,

            "business_action":
                "Protect customer value",

            "decision_reason":
                "Protect reason",

            "experiment_requirement":
                "Recommended",

            "recommended_product_channel":
                "Repeat",

            "recommended_product":
                "P-001",
        },

        {
            "customer":
                "R2",

            "decision_date":
                "2011-08-31",

            "customer_state":
                "At Risk",

            "primary_action":
                ACTION_RETAIN,

            "primary_action_score":
                0.75,

            "action_priority_percentile":
                0.75,

            "priority_tier":
                "Priority 2",

            "contact_intensity":
                CONTACT_CAMPAIGN,

            "business_action":
                "Retention outreach",

            "decision_reason":
                "Retain reason",

            "experiment_requirement":
                "Recommended",

            "recommended_product_channel":
                "Repeat",

            "recommended_product":
                "P-002",
        },

        {
            "customer":
                "X3",

            "decision_date":
                "2011-08-31",

            "customer_state":
                "Dormant",

            "primary_action":
                ACTION_REACTIVATE,

            "primary_action_score":
                0.55,

            "action_priority_percentile":
                0.55,

            "priority_tier":
                "Priority 3",

            "contact_intensity":
                CONTACT_AUTOMATED_LOW_COST,

            "business_action":
                "Controlled reactivation campaign",

            "decision_reason":
                "Reactivate reason",

            "experiment_requirement":
                "Required",

            "recommended_product_channel":
                pd.NA,

            "recommended_product":
                pd.NA,
        },

        {
            "customer":
                "G4",

            "decision_date":
                "2011-08-31",

            "customer_state":
                "Active",

            "primary_action":
                ACTION_GROW,

            "primary_action_score":
                0.20,

            "action_priority_percentile":
                0.20,

            "priority_tier":
                "Priority 4",

            "contact_intensity":
                CONTACT_HOLD_CONTROL,

            "business_action":
                "Cross-sell / discovery offer",

            "decision_reason":
                "Grow reason",

            "experiment_requirement":
                "Required — Discovery channel is CONDITIONAL GO",

            "recommended_product_channel":
                "Discovery",

            "recommended_product":
                "P-004",
        },

        {
            "customer":
                "M1",

            "decision_date":
                "2011-08-31",

            "customer_state":
                "Active",

            "primary_action":
                ACTION_MONITOR,

            "primary_action_score":
                pd.NA,

            "action_priority_percentile":
                pd.NA,

            "priority_tier":
                "Not Prioritised",

            "contact_intensity":
                CONTACT_NO_MODEL_LED,

            "business_action":
                "Monitor customer without immediate intervention",

            "decision_reason":
                "Monitor reason",

            "experiment_requirement":
                "Not Applicable",

            "recommended_product_channel":
                pd.NA,

            "recommended_product":
                pd.NA,
        },

        {
            "customer":
                "L1",

            "decision_date":
                "2011-08-31",

            "customer_state":
                "Long Dormant",

            "primary_action":
                ACTION_UNSCORED_LONG_DORMANT,

            "primary_action_score":
                pd.NA,

            "action_priority_percentile":
                pd.NA,

            "priority_tier":
                "Not Prioritised",

            "contact_intensity":
                CONTACT_NO_MODEL_LED,

            "business_action":
                "Outside current model coverage",

            "decision_reason":
                "Long dormant reason",

            "experiment_requirement":
                "Not Applicable",

            "recommended_product_channel":
                pd.NA,

            "recommended_product":
                pd.NA,
        },
    ]

    return pd.DataFrame(
        rows,
        columns=(
            DELIVERY_REQUIRED_DECISION_COLUMNS
        ),
    )


def row_for(
    frame,
    customer,
):

    return (
        frame.loc[
            frame[
                "customer"
            ]
            ==
            customer
        ]
        .iloc[
            0
        ]
    )


# ================================================================
# INPUT VALIDATION
# ================================================================


def test_valid_input_passes():

    assert (
        validate_delivery_eligibility_input(
            make_decisions()
        )
        is None
    )


def test_non_dataframe_fails_closed():

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):

        validate_delivery_eligibility_input(
            []
        )


def test_missing_required_column_fails_closed():

    frame = (
        make_decisions()
        .drop(
            columns=[
                "contact_intensity"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required",
    ):

        validate_delivery_eligibility_input(
            frame
        )


def test_missing_customer_fails_closed():

    frame = (
        make_decisions()
    )

    frame.loc[
        0,
        "customer",
    ] = pd.NA

    with pytest.raises(
        ValueError,
        match="missing customer",
    ):

        validate_delivery_eligibility_input(
            frame
        )


def test_duplicate_customer_fails_closed():

    frame = (
        make_decisions()
    )

    frame.loc[
        1,
        "customer",
    ] = "P1"

    with pytest.raises(
        ValueError,
        match="one row per customer",
    ):

        validate_delivery_eligibility_input(
            frame
        )


def test_multiple_decision_dates_fail_closed():

    frame = (
        make_decisions()
    )

    frame.loc[
        1,
        "decision_date",
    ] = "2011-09-01"

    with pytest.raises(
        ValueError,
        match="exactly one decision_date",
    ):

        validate_delivery_eligibility_input(
            frame
        )


def test_unknown_primary_action_fails_closed():

    frame = (
        make_decisions()
    )

    frame.loc[
        0,
        "primary_action",
    ] = "Unknown Action"

    with pytest.raises(
        ValueError,
        match="unknown primary actions",
    ):

        validate_delivery_eligibility_input(
            frame
        )


def test_missing_contact_intensity_fails_closed():

    frame = (
        make_decisions()
    )

    frame.loc[
        0,
        "contact_intensity",
    ] = pd.NA

    with pytest.raises(
        ValueError,
        match="missing contact_intensity",
    ):

        validate_delivery_eligibility_input(
            frame
        )


# ================================================================
# ROUTING
# ================================================================


def test_priority_one_high_touch_routes_to_treatment():

    result = (
        apply_delivery_eligibility(
            make_decisions()
        )
    )

    row = row_for(
        result,
        "P1",
    )

    assert (
        row[
            "delivery_route"
        ]
        ==
        DELIVERY_ROUTE_HIGH_TOUCH
    )

    assert (
        row[
            "delivery_disposition"
        ]
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_priority_two_campaign_routes_to_treatment():

    result = (
        apply_delivery_eligibility(
            make_decisions()
        )
    )

    row = row_for(
        result,
        "R2",
    )

    assert (
        row[
            "delivery_route"
        ]
        ==
        DELIVERY_ROUTE_CAMPAIGN
    )

    assert (
        row[
            "delivery_disposition"
        ]
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_priority_three_automated_routes_to_treatment():

    result = (
        apply_delivery_eligibility(
            make_decisions()
        )
    )

    row = row_for(
        result,
        "X3",
    )

    assert (
        row[
            "delivery_route"
        ]
        ==
        DELIVERY_ROUTE_AUTOMATED
    )

    assert (
        row[
            "delivery_disposition"
        ]
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_priority_four_routes_to_hold_control():

    result = (
        apply_delivery_eligibility(
            make_decisions()
        )
    )

    row = row_for(
        result,
        "G4",
    )

    assert (
        row[
            "delivery_route"
        ]
        ==
        DELIVERY_ROUTE_HOLD_CONTROL
    )

    assert (
        row[
            "delivery_disposition"
        ]
        ==
        DELIVERY_DISPOSITION_HOLD_CONTROL
    )


def test_non_actionable_routes_to_suppressed():

    result = (
        apply_delivery_eligibility(
            make_decisions()
        )
    )

    row = row_for(
        result,
        "M1",
    )

    assert (
        row[
            "delivery_route"
        ]
        ==
        DELIVERY_ROUTE_SUPPRESSED
    )

    assert (
        row[
            "delivery_disposition"
        ]
        ==
        DELIVERY_DISPOSITION_SUPPRESSED
    )


def test_long_dormant_routes_to_suppressed():

    result = (
        apply_delivery_eligibility(
            make_decisions()
        )
    )

    row = row_for(
        result,
        "L1",
    )

    assert (
        row[
            "delivery_route"
        ]
        ==
        DELIVERY_ROUTE_SUPPRESSED
    )


# ================================================================
# CONTRADICTION PROTECTION
# ================================================================


def test_non_actionable_treatment_contact_fails_closed():

    frame = (
        make_decisions()
    )

    frame.loc[
        frame[
            "customer"
        ]
        ==
        "M1",
        "contact_intensity",
    ] = CONTACT_CAMPAIGN

    with pytest.raises(
        ValueError,
        match="Non-actionable decisions must have",
    ):

        apply_delivery_eligibility(
            frame
        )


def test_actionable_no_model_contact_fails_closed():

    frame = (
        make_decisions()
    )

    frame.loc[
        frame[
            "customer"
        ]
        ==
        "P1",
        "contact_intensity",
    ] = CONTACT_NO_MODEL_LED

    with pytest.raises(
        ValueError,
        match="Actionable decisions cannot have",
    ):

        apply_delivery_eligibility(
            frame
        )


def test_unknown_contact_policy_fails_closed():

    frame = (
        make_decisions()
    )

    frame.loc[
        frame[
            "customer"
        ]
        ==
        "P1",
        "contact_intensity",
    ] = "Unknown Contact"

    with pytest.raises(
        ValueError,
        match="Unsupported contact_intensity",
    ):

        apply_delivery_eligibility(
            frame
        )


# ================================================================
# POPULATION / OUTPUT
# ================================================================


def test_population_is_preserved():

    frame = (
        make_decisions()
    )

    result = (
        apply_delivery_eligibility(
            frame
        )
    )

    assert len(
        result
    ) == len(
        frame
    )


def test_customer_order_is_preserved():

    frame = (
        make_decisions()
    )

    result = (
        apply_delivery_eligibility(
            frame
        )
    )

    assert (
        result[
            "customer"
        ].tolist()
        ==
        frame[
            "customer"
        ].tolist()
    )


def test_eligibility_columns_are_added():

    result = (
        apply_delivery_eligibility(
            make_decisions()
        )
    )

    for column in (
        DELIVERY_ELIGIBILITY_COLUMNS
    ):

        assert (
            column
            in
            result.columns
        )


def test_empty_input_is_supported():

    frame = pd.DataFrame(
        columns=(
            DELIVERY_REQUIRED_DECISION_COLUMNS
        )
    )

    result = (
        apply_delivery_eligibility(
            frame
        )
    )

    assert result.empty

    assert (
        "delivery_route"
        in
        result.columns
    )

    assert (
        "delivery_disposition"
        in
        result.columns
    )