"""
Tests for the append-only delivery audit ledger.
"""

from datetime import datetime, timezone
import json

import pytest

from src.delivery.audit import (
    append_delivery_audit,
    append_delivery_batch_audit,
    get_delivery_audit,
    list_delivery_audit,
)

from src.delivery.payload import (
    DeliveryPayload,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITION_HOLD_CONTROL,
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_PAYLOAD_COLUMNS,
    DELIVERY_ROUTE_CAMPAIGN,
    DELIVERY_ROUTE_HOLD_CONTROL,
    DELIVERY_ROUTE_SUPPRESSED,
)


CREATED_AT = datetime(
    2026,
    10,
    27,
    17,
    0,
    tzinfo=timezone.utc,
)


def make_payload(
    delivery_id,
    customer,
    *,
    disposition="treatment",
):

    if disposition == "suppressed":

        return DeliveryPayload(
            delivery_id=delivery_id,
            delivery_created_at_utc=CREATED_AT,
            customer=customer,
            decision_date="2011-08-31",
            customer_state="Active",
            primary_action="Monitor",
            primary_action_score=None,
            action_priority_percentile=None,
            priority_tier="Not Prioritised",
            contact_intensity="No Model-Led Contact",
            delivery_route=DELIVERY_ROUTE_SUPPRESSED,
            delivery_disposition=DELIVERY_DISPOSITION_SUPPRESSED,
            business_action="Monitor customer without immediate intervention",
            decision_reason="Monitor reason",
            experiment_requirement="Not Applicable",
            recommended_product_channel=None,
            recommended_product=None,
        )

    if disposition == "hold":

        return DeliveryPayload(
            delivery_id=delivery_id,
            delivery_created_at_utc=CREATED_AT,
            customer=customer,
            decision_date="2011-08-31",
            customer_state="Active",
            primary_action="Grow",
            primary_action_score=0.20,
            action_priority_percentile=0.20,
            priority_tier="Priority 4",
            contact_intensity="Hold / Control",
            delivery_route=DELIVERY_ROUTE_HOLD_CONTROL,
            delivery_disposition=DELIVERY_DISPOSITION_HOLD_CONTROL,
            business_action="Cross-sell / discovery offer",
            decision_reason="Grow reason",
            experiment_requirement=(
                "Required — Discovery channel is CONDITIONAL GO"
            ),
            recommended_product_channel="Discovery",
            recommended_product="P-004",
        )

    return DeliveryPayload(
        delivery_id=delivery_id,
        delivery_created_at_utc=CREATED_AT,
        customer=customer,
        decision_date="2011-08-31",
        customer_state="At Risk",
        primary_action="Retain",
        primary_action_score=0.82,
        action_priority_percentile=0.91,
        priority_tier="Priority 1",
        contact_intensity="Campaign",
        delivery_route=DELIVERY_ROUTE_CAMPAIGN,
        delivery_disposition=DELIVERY_DISPOSITION_TREATMENT,
        business_action="Retention outreach using relevant repeat-product cue",
        decision_reason="Retain reason",
        experiment_requirement="Recommended",
        recommended_product_channel="Repeat",
        recommended_product="P-001",
    )


# ================================================================
# MISSING / EMPTY LEDGER
# ================================================================


def test_missing_audit_returns_empty_tuple(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    assert (
        list_delivery_audit(
            path
        )
        ==
        ()
    )


def test_empty_batch_creates_empty_audit(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    result = (
        append_delivery_batch_audit(
            (),
            path,
        )
    )

    assert result == path
    assert path.is_file()

    assert (
        list_delivery_audit(
            path
        )
        ==
        ()
    )


# ================================================================
# SINGLE APPEND
# ================================================================


def test_append_one_creates_audit_file(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    result = (
        append_delivery_audit(
            make_payload(
                "delivery-001",
                "CUST-001",
            ),
            path,
        )
    )

    assert result == path
    assert path.is_file()


def test_append_one_round_trips(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    original = (
        make_payload(
            "delivery-001",
            "CUST-001",
        )
    )

    append_delivery_audit(
        original,
        path,
    )

    loaded = (
        get_delivery_audit(
            "delivery-001",
            path,
        )
    )

    assert (
        loaded
        ==
        original
    )


def test_duplicate_delivery_id_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    payload = (
        make_payload(
            "delivery-001",
            "CUST-001",
        )
    )

    append_delivery_audit(
        payload,
        path,
    )

    with pytest.raises(
        ValueError,
        match="already exists",
    ):

        append_delivery_audit(
            payload,
            path,
        )

    assert len(
        list_delivery_audit(
            path
        )
    ) == 1


# ================================================================
# BATCH APPEND
# ================================================================


def test_batch_append_preserves_order(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    payloads = (
        make_payload(
            "delivery-001",
            "CUST-001",
        ),
        make_payload(
            "delivery-002",
            "CUST-002",
            disposition="hold",
        ),
        make_payload(
            "delivery-003",
            "CUST-003",
            disposition="suppressed",
        ),
    )

    append_delivery_batch_audit(
        payloads,
        path,
    )

    loaded = (
        list_delivery_audit(
            path
        )
    )

    assert [
        payload.delivery_id
        for payload
        in loaded
    ] == [
        "delivery-001",
        "delivery-002",
        "delivery-003",
    ]


def test_batch_audits_all_dispositions(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    append_delivery_batch_audit(
        (
            make_payload(
                "delivery-001",
                "CUST-001",
            ),
            make_payload(
                "delivery-002",
                "CUST-002",
                disposition="hold",
            ),
            make_payload(
                "delivery-003",
                "CUST-003",
                disposition="suppressed",
            ),
        ),
        path,
    )

    loaded = (
        list_delivery_audit(
            path
        )
    )

    assert [
        payload.delivery_disposition
        for payload
        in loaded
    ] == [
        DELIVERY_DISPOSITION_TREATMENT,
        DELIVERY_DISPOSITION_HOLD_CONTROL,
        DELIVERY_DISPOSITION_SUPPRESSED,
    ]


def test_duplicate_ids_inside_batch_fail_before_write(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    with pytest.raises(
        ValueError,
        match="duplicate delivery_id",
    ):

        append_delivery_batch_audit(
            (
                make_payload(
                    "duplicate",
                    "CUST-001",
                ),
                make_payload(
                    "duplicate",
                    "CUST-002",
                ),
            ),
            path,
        )

    assert not path.exists()


def test_batch_collision_with_existing_history_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    append_delivery_audit(
        make_payload(
            "delivery-001",
            "CUST-001",
        ),
        path,
    )

    with pytest.raises(
        ValueError,
        match="already contains",
    ):

        append_delivery_batch_audit(
            (
                make_payload(
                    "delivery-001",
                    "CUST-999",
                ),
                make_payload(
                    "delivery-002",
                    "CUST-002",
                ),
            ),
            path,
        )

    loaded = (
        list_delivery_audit(
            path
        )
    )

    assert [
        payload.delivery_id
        for payload
        in loaded
    ] == [
        "delivery-001",
    ]


# ================================================================
# GET
# ================================================================


def test_unknown_delivery_id_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    append_delivery_audit(
        make_payload(
            "delivery-001",
            "CUST-001",
        ),
        path,
    )

    with pytest.raises(
        KeyError,
        match="Unknown delivery_id",
    ):

        get_delivery_audit(
            "missing",
            path,
        )


def test_empty_delivery_id_fails_closed(
    tmp_path,
):

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):

        get_delivery_audit(
            "",
            tmp_path
            /
            "audit.jsonl",
        )


# ================================================================
# INPUT TYPE VALIDATION
# ================================================================


def test_single_append_rejects_wrong_type(
    tmp_path,
):

    with pytest.raises(
        TypeError,
        match="DeliveryPayload",
    ):

        append_delivery_audit(
            {},
            tmp_path
            /
            "audit.jsonl",
        )


def test_batch_rejects_single_payload_object(
    tmp_path,
):

    payload = (
        make_payload(
            "delivery-001",
            "CUST-001",
        )
    )

    with pytest.raises(
        TypeError,
        match="iterable",
    ):

        append_delivery_batch_audit(
            payload,
            tmp_path
            /
            "audit.jsonl",
        )


def test_batch_rejects_wrong_member_type(
    tmp_path,
):

    with pytest.raises(
        TypeError,
        match="Every payload",
    ):

        append_delivery_batch_audit(
            (
                {},
            ),
            tmp_path
            /
            "audit.jsonl",
        )


# ================================================================
# MALFORMED HISTORY
# ================================================================


def test_invalid_json_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "{broken}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):

        list_delivery_audit(
            path
        )


def test_schema_drift_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        json.dumps(
            {
                "customer":
                    "CUST-001",
            }
        )
        +
        "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="schema",
    ):

        list_delivery_audit(
            path
        )


def test_empty_line_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="empty line",
    ):

        list_delivery_audit(
            path
        )


def test_duplicate_historical_id_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    payload = (
        make_payload(
            "delivery-001",
            "CUST-001",
        )
    )

    from src.delivery.payload import (
        delivery_payload_to_dict,
    )

    record = (
        delivery_payload_to_dict(
            payload
        )
    )

    line = json.dumps(
        record,
        separators=(
            ",",
            ":",
        ),
    )

    path.write_text(
        line
        +
        "\n"
        +
        line
        +
        "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="duplicate delivery_id",
    ):

        list_delivery_audit(
            path
        )


def test_malformed_history_blocks_new_append(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "{broken}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):

        append_delivery_audit(
            make_payload(
                "delivery-002",
                "CUST-002",
            ),
            path,
        )

    assert (
        path.read_text(
            encoding="utf-8"
        )
        ==
        "{broken}\n"
    )


# ================================================================
# CANONICAL SCHEMA
# ================================================================


def test_written_record_has_exact_payload_schema(
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    append_delivery_audit(
        make_payload(
            "delivery-001",
            "CUST-001",
        ),
        path,
    )

    record = json.loads(
        path.read_text(
            encoding="utf-8"
        ).strip()
    )

    assert tuple(
        record.keys()
    ) == DELIVERY_PAYLOAD_COLUMNS