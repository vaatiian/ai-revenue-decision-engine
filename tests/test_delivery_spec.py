"""
Contract tests for the frozen CRM/action-delivery specification.
"""

import pytest

from src.decision_engine.spec import (
    ACTIONABLE_ACTIONS,
    CONTACT_AUTOMATED_LOW_COST,
    CONTACT_CAMPAIGN,
    CONTACT_HIGH_TOUCH,
    CONTACT_HOLD_CONTROL,
    CONTACT_INTENSITIES,
    CONTACT_NO_MODEL_LED,
    DECISION_OUTPUT_COLUMNS,
    NON_ACTIONABLE_ACTIONS,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITION_BY_ROUTE,
    DELIVERY_DISPOSITION_HOLD_CONTROL,
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_DISPOSITIONS,
    DELIVERY_PAYLOAD_COLUMNS,
    DELIVERY_REQUIRED_DECISION_COLUMNS,
    DELIVERY_ROUTE_AUTOMATED,
    DELIVERY_ROUTE_BY_CONTACT_INTENSITY,
    DELIVERY_ROUTE_CAMPAIGN,
    DELIVERY_ROUTE_HIGH_TOUCH,
    DELIVERY_ROUTE_HOLD_CONTROL,
    DELIVERY_ROUTE_SUPPRESSED,
    DELIVERY_ROUTES,
    DELIVERY_SCHEMA_VERSION,
    NON_TREATMENT_DELIVERY_ROUTES,
    TREATMENT_DELIVERY_ROUTES,
    get_delivery_disposition,
    get_delivery_route,
    validate_delivery_spec,
)


# ================================================================
# VERSION
# ================================================================


def test_delivery_schema_version_is_frozen():

    assert (
        DELIVERY_SCHEMA_VERSION
        ==
        "1.0.0"
    )


# ================================================================
# ROUTES
# ================================================================


def test_delivery_routes_are_frozen():

    assert (
        DELIVERY_ROUTES
        ==
        (
            "HIGH_TOUCH_QUEUE",
            "CAMPAIGN_QUEUE",
            "AUTOMATED_QUEUE",
            "HOLD_CONTROL",
            "SUPPRESSED",
        )
    )


def test_contact_intensity_mapping_is_exact():

    assert (
        DELIVERY_ROUTE_BY_CONTACT_INTENSITY
        ==
        (
            (
                CONTACT_HIGH_TOUCH,
                DELIVERY_ROUTE_HIGH_TOUCH,
            ),
            (
                CONTACT_CAMPAIGN,
                DELIVERY_ROUTE_CAMPAIGN,
            ),
            (
                CONTACT_AUTOMATED_LOW_COST,
                DELIVERY_ROUTE_AUTOMATED,
            ),
            (
                CONTACT_HOLD_CONTROL,
                DELIVERY_ROUTE_HOLD_CONTROL,
            ),
            (
                CONTACT_NO_MODEL_LED,
                DELIVERY_ROUTE_SUPPRESSED,
            ),
        )
    )


@pytest.mark.parametrize(
    "contact_intensity,expected_route",
    DELIVERY_ROUTE_BY_CONTACT_INTENSITY,
)
def test_contact_intensity_maps_to_expected_route(
    contact_intensity,
    expected_route,
):

    assert (
        get_delivery_route(
            contact_intensity
        )
        ==
        expected_route
    )


def test_unknown_contact_intensity_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported contact_intensity",
    ):

        get_delivery_route(
            "Unknown Contact"
        )


def test_non_string_contact_intensity_fails_closed():

    with pytest.raises(
        TypeError,
        match="contact_intensity must be a string",
    ):

        get_delivery_route(
            123
        )


# ================================================================
# TREATMENT POLICY
# ================================================================


def test_treatment_routes_are_exact():

    assert (
        TREATMENT_DELIVERY_ROUTES
        ==
        (
            DELIVERY_ROUTE_HIGH_TOUCH,
            DELIVERY_ROUTE_CAMPAIGN,
            DELIVERY_ROUTE_AUTOMATED,
        )
    )


def test_non_treatment_routes_are_exact():

    assert (
        NON_TREATMENT_DELIVERY_ROUTES
        ==
        (
            DELIVERY_ROUTE_HOLD_CONTROL,
            DELIVERY_ROUTE_SUPPRESSED,
        )
    )


def test_delivery_dispositions_are_frozen():

    assert (
        DELIVERY_DISPOSITIONS
        ==
        (
            DELIVERY_DISPOSITION_TREATMENT,
            DELIVERY_DISPOSITION_HOLD_CONTROL,
            DELIVERY_DISPOSITION_SUPPRESSED,
        )
    )

    assert (
        DELIVERY_DISPOSITIONS
        ==
        (
            "TREATMENT",
            "HOLD_CONTROL",
            "SUPPRESSED",
        )
    )


@pytest.mark.parametrize(
    "delivery_route,expected_disposition",
    DELIVERY_DISPOSITION_BY_ROUTE,
)
def test_route_maps_to_expected_disposition(
    delivery_route,
    expected_disposition,
):

    assert (
        get_delivery_disposition(
            delivery_route
        )
        ==
        expected_disposition
    )


def test_unknown_delivery_route_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported delivery_route",
    ):

        get_delivery_disposition(
            "UNKNOWN_ROUTE"
        )


def test_non_string_delivery_route_fails_closed():

    with pytest.raises(
        TypeError,
        match="delivery_route must be a string",
    ):

        get_delivery_disposition(
            None
        )


# ================================================================
# DECISION ENGINE ALIGNMENT
# ================================================================


def test_delivery_routing_covers_exact_contact_policy():

    assert tuple(
        contact_intensity
        for (
            contact_intensity,
            _,
        )
        in DELIVERY_ROUTE_BY_CONTACT_INTENSITY
    ) == CONTACT_INTENSITIES


def test_required_input_is_subset_of_decision_output():

    assert set(
        DELIVERY_REQUIRED_DECISION_COLUMNS
    ).issubset(
        set(
            DECISION_OUTPUT_COLUMNS
        )
    )


def test_actionable_and_non_actionable_actions_are_disjoint():

    assert not (
        set(
            ACTIONABLE_ACTIONS
        )
        &
        set(
            NON_ACTIONABLE_ACTIONS
        )
    )


# ================================================================
# PAYLOAD CONTRACT
# ================================================================


def test_delivery_payload_columns_are_exact():

    assert (
        DELIVERY_PAYLOAD_COLUMNS
        ==
        (
            "delivery_id",
            "delivery_created_at_utc",
            "customer",
            "decision_date",
            "customer_state",
            "primary_action",
            "primary_action_score",
            "action_priority_percentile",
            "priority_tier",
            "contact_intensity",
            "delivery_route",
            "delivery_disposition",
            "business_action",
            "decision_reason",
            "experiment_requirement",
            "recommended_product_channel",
            "recommended_product",
        )
    )


def test_delivery_payload_preserves_business_policy():

    for column in (
        "primary_action",
        "priority_tier",
        "contact_intensity",
        "business_action",
        "decision_reason",
        "experiment_requirement",
        "recommended_product_channel",
        "recommended_product",
    ):

        assert (
            column
            in
            DELIVERY_PAYLOAD_COLUMNS
        )


def test_delivery_contract_does_not_invent_communication_channel():

    assert (
        "communication_channel"
        not in
        DELIVERY_PAYLOAD_COLUMNS
    )

    assert (
        "email"
        not in
        DELIVERY_PAYLOAD_COLUMNS
    )

    assert (
        "phone"
        not in
        DELIVERY_PAYLOAD_COLUMNS
    )


def test_spec_self_validation_passes():

    assert (
        validate_delivery_spec()
        is None
    )