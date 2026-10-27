"""
Tests for immutable CRM/action-delivery payloads.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.delivery.payload import (
    DeliveryPayload,
    delivery_payload_to_dict,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_PAYLOAD_COLUMNS,
    DELIVERY_ROUTE_CAMPAIGN,
    DELIVERY_ROUTE_SUPPRESSED,
)


def make_payload(
    **overrides,
):

    values = {
        "delivery_id":
            "delivery-001",

        "delivery_created_at_utc":
            datetime(
                2026,
                10,
                27,
                14,
                0,
                tzinfo=timezone.utc,
            ),

        "customer":
            "CUST-001",

        "decision_date":
            "2011-08-31",

        "customer_state":
            "At Risk",

        "primary_action":
            "Retain",

        "primary_action_score":
            0.82,

        "action_priority_percentile":
            0.91,

        "priority_tier":
            "Priority 1",

        "contact_intensity":
            "Campaign",

        "delivery_route":
            DELIVERY_ROUTE_CAMPAIGN,

        "delivery_disposition":
            DELIVERY_DISPOSITION_TREATMENT,

        "business_action":
            "Retention outreach using relevant repeat-product cue",

        "decision_reason":
            "At-risk customer with model-supported retention signal",

        "experiment_requirement":
            "Recommended",

        "recommended_product_channel":
            "Repeat",

        "recommended_product":
            "P-001",
    }

    values.update(
        overrides
    )

    return DeliveryPayload(
        **values
    )


# ================================================================
# CONSTRUCTION / IMMUTABILITY
# ================================================================


def test_valid_payload_can_be_created():

    payload = (
        make_payload()
    )

    assert (
        payload.delivery_id
        ==
        "delivery-001"
    )


def test_payload_is_frozen():

    payload = (
        make_payload()
    )

    with pytest.raises(
        FrozenInstanceError
    ):

        payload.delivery_route = (
            DELIVERY_ROUTE_SUPPRESSED
        )


# ================================================================
# TIME / DATE NORMALISATION
# ================================================================


def test_naive_delivery_timestamp_is_normalised_to_utc():

    payload = (
        make_payload(
            delivery_created_at_utc=(
                "2026-10-27 14:00:00"
            )
        )
    )

    assert (
        payload.delivery_created_at_utc.tzinfo
        is not None
    )


def test_decision_date_is_serialised_as_iso_date():

    payload = (
        make_payload(
            decision_date=(
                pd.Timestamp(
                    "2011-08-31"
                )
            )
        )
    )

    assert (
        payload.decision_date
        ==
        "2011-08-31"
    )


def test_invalid_delivery_timestamp_fails():

    with pytest.raises(
        ValueError,
        match="not a valid timestamp",
    ):

        make_payload(
            delivery_created_at_utc=(
                "not-a-date"
            )
        )


def test_invalid_decision_date_fails():

    with pytest.raises(
        ValueError,
        match="decision_date",
    ):

        make_payload(
            decision_date=(
                "not-a-date"
            )
        )


# ================================================================
# OPTIONAL NUMERIC FIELDS
# ================================================================


def test_optional_scores_accept_missing_values():

    payload = (
        make_payload(
            primary_action_score=pd.NA,
            action_priority_percentile=pd.NA,
        )
    )

    assert (
        payload.primary_action_score
        is None
    )

    assert (
        payload.action_priority_percentile
        is None
    )


def test_numeric_scores_are_normalised_to_float():

    payload = (
        make_payload(
            primary_action_score=1,
            action_priority_percentile="0.75",
        )
    )

    assert (
        payload.primary_action_score
        ==
        1.0
    )

    assert (
        payload.action_priority_percentile
        ==
        0.75
    )


def test_invalid_numeric_score_fails():

    with pytest.raises(
        ValueError,
        match="primary_action_score",
    ):

        make_payload(
            primary_action_score=(
                "not-numeric"
            )
        )


# ================================================================
# OPTIONAL PRODUCT FIELDS
# ================================================================


def test_missing_product_fields_are_normalised_to_none():

    payload = (
        make_payload(
            recommended_product_channel=pd.NA,
            recommended_product=pd.NA,
        )
    )

    assert (
        payload.recommended_product_channel
        is None
    )

    assert (
        payload.recommended_product
        is None
    )


# ================================================================
# ROUTE / DISPOSITION CONTRACT
# ================================================================


def test_unknown_route_fails_closed():

    with pytest.raises(
        ValueError,
        match="delivery_route",
    ):

        make_payload(
            delivery_route=(
                "UNKNOWN"
            )
        )


def test_unknown_disposition_fails_closed():

    with pytest.raises(
        ValueError,
        match="delivery_disposition",
    ):

        make_payload(
            delivery_disposition=(
                "UNKNOWN"
            )
        )


def test_suppressed_payload_can_be_represented():

    payload = (
        make_payload(
            primary_action=(
                "Monitor"
            ),
            priority_tier=(
                "Not Prioritised"
            ),
            contact_intensity=(
                "No Model-Led Contact"
            ),
            delivery_route=(
                DELIVERY_ROUTE_SUPPRESSED
            ),
            delivery_disposition=(
                DELIVERY_DISPOSITION_SUPPRESSED
            ),
            primary_action_score=None,
            action_priority_percentile=None,
            recommended_product_channel=None,
            recommended_product=None,
        )
    )

    assert (
        payload.delivery_disposition
        ==
        DELIVERY_DISPOSITION_SUPPRESSED
    )


# ================================================================
# SERIALISATION
# ================================================================


def test_payload_to_dict_has_exact_schema():

    output = (
        delivery_payload_to_dict(
            make_payload()
        )
    )

    assert tuple(
        output.keys()
    ) == DELIVERY_PAYLOAD_COLUMNS


def test_payload_to_dict_serialises_timestamp():

    output = (
        delivery_payload_to_dict(
            make_payload()
        )
    )

    assert (
        output[
            "delivery_created_at_utc"
        ]
        ==
        "2026-10-27T14:00:00+00:00"
    )


def test_payload_to_dict_preserves_delivery_policy():

    output = (
        delivery_payload_to_dict(
            make_payload()
        )
    )

    assert (
        output[
            "delivery_route"
        ]
        ==
        DELIVERY_ROUTE_CAMPAIGN
    )

    assert (
        output[
            "delivery_disposition"
        ]
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_payload_to_dict_rejects_wrong_type():

    with pytest.raises(
        TypeError,
        match="DeliveryPayload",
    ):

        delivery_payload_to_dict(
            {}
        )