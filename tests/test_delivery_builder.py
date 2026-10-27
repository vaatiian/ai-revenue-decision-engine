"""
Tests for batch CRM/action-delivery payload construction.
"""

from datetime import datetime, timezone
from uuid import UUID

import pandas as pd
import pytest

from src.decision_engine.spec import (
    ACTION_GROW,
    ACTION_MONITOR,
    ACTION_PROTECT,
    ACTION_REACTIVATE,
    CONTACT_AUTOMATED_LOW_COST,
    CONTACT_HIGH_TOUCH,
    CONTACT_HOLD_CONTROL,
    CONTACT_NO_MODEL_LED,
)

from src.delivery.builder import (
    build_delivery_batch,
    delivery_batch_to_frame,
    generate_delivery_id,
)

from src.delivery.payload import (
    DeliveryPayload,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITION_HOLD_CONTROL,
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_PAYLOAD_COLUMNS,
    DELIVERY_REQUIRED_DECISION_COLUMNS,
    DELIVERY_ROUTE_AUTOMATED,
    DELIVERY_ROUTE_HIGH_TOUCH,
    DELIVERY_ROUTE_HOLD_CONTROL,
    DELIVERY_ROUTE_SUPPRESSED,
)


CREATED_AT = datetime(
    2026,
    10,
    27,
    15,
    0,
    tzinfo=timezone.utc,
)


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
    ]

    return pd.DataFrame(
        rows,
        columns=(
            DELIVERY_REQUIRED_DECISION_COLUMNS
        ),
    )


# ================================================================
# ID GENERATION
# ================================================================


def test_generated_delivery_id_is_valid_uuid():

    delivery_id = (
        generate_delivery_id()
    )

    parsed = UUID(
        delivery_id
    )

    assert (
        str(
            parsed
        )
        ==
        delivery_id
    )


def test_generated_delivery_ids_are_unique():

    assert (
        generate_delivery_id()
        !=
        generate_delivery_id()
    )


# ================================================================
# BATCH BUILDING
# ================================================================


def test_build_returns_tuple_of_payloads():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    assert isinstance(
        payloads,
        tuple,
    )

    assert all(
        isinstance(
            payload,
            DeliveryPayload,
        )
        for payload
        in payloads
    )


def test_one_payload_per_customer():

    frame = (
        make_decisions()
    )

    payloads = (
        build_delivery_batch(
            frame,
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    assert len(
        payloads
    ) == len(
        frame
    )


def test_customer_order_is_preserved():

    frame = (
        make_decisions()
    )

    payloads = (
        build_delivery_batch(
            frame,
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    assert [
        payload.customer
        for payload
        in payloads
    ] == frame[
        "customer"
    ].tolist()


def test_all_payloads_share_created_timestamp():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    assert {
        payload.delivery_created_at_utc
        for payload
        in payloads
    } == {
        CREATED_AT
    }


# ================================================================
# ROUTING PRESERVATION
# ================================================================


def test_high_touch_payload_is_treatment():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    payload = payloads[
        0
    ]

    assert (
        payload.delivery_route
        ==
        DELIVERY_ROUTE_HIGH_TOUCH
    )

    assert (
        payload.delivery_disposition
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_automated_payload_is_treatment():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    payload = payloads[
        1
    ]

    assert (
        payload.delivery_route
        ==
        DELIVERY_ROUTE_AUTOMATED
    )

    assert (
        payload.delivery_disposition
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_hold_control_payload_is_not_treatment():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    payload = payloads[
        2
    ]

    assert (
        payload.delivery_route
        ==
        DELIVERY_ROUTE_HOLD_CONTROL
    )

    assert (
        payload.delivery_disposition
        ==
        DELIVERY_DISPOSITION_HOLD_CONTROL
    )


def test_non_actionable_payload_is_suppressed():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    payload = payloads[
        3
    ]

    assert (
        payload.delivery_route
        ==
        DELIVERY_ROUTE_SUPPRESSED
    )

    assert (
        payload.delivery_disposition
        ==
        DELIVERY_DISPOSITION_SUPPRESSED
    )


# ================================================================
# EXPLICIT DELIVERY IDS
# ================================================================


def test_explicit_delivery_ids_are_preserved():

    delivery_ids = {
        "P1":
            "delivery-p1",

        "X3":
            "delivery-x3",

        "G4":
            "delivery-g4",

        "M1":
            "delivery-m1",
    }

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                delivery_ids
            ),
        )
    )

    assert [
        payload.delivery_id
        for payload
        in payloads
    ] == [
        "delivery-p1",
        "delivery-x3",
        "delivery-g4",
        "delivery-m1",
    ]


def test_delivery_id_keys_must_match_population():

    with pytest.raises(
        ValueError,
        match="exactly match",
    ):

        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids={
                "P1":
                    "delivery-p1",
            },
        )


def test_delivery_ids_must_be_unique():

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):

        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids={
                "P1":
                    "duplicate",

                "X3":
                    "duplicate",

                "G4":
                    "delivery-g4",

                "M1":
                    "delivery-m1",
            },
        )


def test_delivery_ids_must_be_dictionary():

    with pytest.raises(
        TypeError,
        match="dictionary",
    ):

        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=[
                "invalid"
            ],
        )


# ================================================================
# TIMESTAMP VALIDATION
# ================================================================


def test_invalid_creation_timestamp_fails():

    with pytest.raises(
        ValueError,
        match="not a valid timestamp",
    ):

        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                "not-a-date"
            ),
        )


def test_naive_creation_timestamp_is_utc():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                "2026-10-27 15:00:00"
            ),
        )
    )

    assert all(
        payload.delivery_created_at_utc.tzinfo
        is not None
        for payload
        in payloads
    )


# ================================================================
# FRAME REPRESENTATION
# ================================================================


def test_batch_to_frame_has_exact_schema():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    frame = (
        delivery_batch_to_frame(
            payloads
        )
    )

    assert tuple(
        frame.columns
    ) == DELIVERY_PAYLOAD_COLUMNS


def test_batch_to_frame_preserves_customer_order():

    payloads = (
        build_delivery_batch(
            make_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )
    )

    frame = (
        delivery_batch_to_frame(
            payloads
        )
    )

    assert (
        frame[
            "customer"
        ].tolist()
        ==
        [
            "P1",
            "X3",
            "G4",
            "M1",
        ]
    )


def test_batch_to_frame_supports_empty_batch():

    frame = (
        delivery_batch_to_frame(
            ()
        )
    )

    assert frame.empty

    assert tuple(
        frame.columns
    ) == DELIVERY_PAYLOAD_COLUMNS


def test_batch_to_frame_rejects_wrong_member_type():

    with pytest.raises(
        TypeError,
        match="Every payload",
    ):

        delivery_batch_to_frame(
            (
                {},
            )
        )


def test_batch_to_frame_rejects_single_payload_object():

    payload = (
        build_delivery_batch(
            make_decisions().iloc[
                [
                    0
                ]
            ].copy(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
        )[
            0
        ]
    )

    with pytest.raises(
        TypeError,
        match="iterable",
    ):

        delivery_batch_to_frame(
            payload
        )