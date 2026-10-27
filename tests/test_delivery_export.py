"""
Tests for CRM/action-delivery CSV and JSONL export adapters.
"""

from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from src.delivery.export import (
    export_delivery_csv,
    export_delivery_jsonl,
    read_delivery_csv,
    read_delivery_jsonl,
)

from src.delivery.payload import (
    DeliveryPayload,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_PAYLOAD_COLUMNS,
    DELIVERY_ROUTE_CAMPAIGN,
    DELIVERY_ROUTE_SUPPRESSED,
)


CREATED_AT = datetime(
    2026,
    10,
    27,
    16,
    0,
    tzinfo=timezone.utc,
)


def make_payload(
    delivery_id,
    customer,
    *,
    suppressed=False,
):

    if suppressed:

        return DeliveryPayload(
            delivery_id=(
                delivery_id
            ),

            delivery_created_at_utc=(
                CREATED_AT
            ),

            customer=(
                customer
            ),

            decision_date=(
                "2011-08-31"
            ),

            customer_state=(
                "Active"
            ),

            primary_action=(
                "Monitor"
            ),

            primary_action_score=None,

            action_priority_percentile=None,

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

            business_action=(
                "Monitor customer without immediate intervention"
            ),

            decision_reason=(
                "Monitor reason"
            ),

            experiment_requirement=(
                "Not Applicable"
            ),

            recommended_product_channel=None,

            recommended_product=None,
        )

    return DeliveryPayload(
        delivery_id=(
            delivery_id
        ),

        delivery_created_at_utc=(
            CREATED_AT
        ),

        customer=(
            customer
        ),

        decision_date=(
            "2011-08-31"
        ),

        customer_state=(
            "At Risk"
        ),

        primary_action=(
            "Retain"
        ),

        primary_action_score=(
            0.82
        ),

        action_priority_percentile=(
            0.91
        ),

        priority_tier=(
            "Priority 1"
        ),

        contact_intensity=(
            "Campaign"
        ),

        delivery_route=(
            DELIVERY_ROUTE_CAMPAIGN
        ),

        delivery_disposition=(
            DELIVERY_DISPOSITION_TREATMENT
        ),

        business_action=(
            "Retention outreach using relevant repeat-product cue"
        ),

        decision_reason=(
            "Retain reason"
        ),

        experiment_requirement=(
            "Recommended"
        ),

        recommended_product_channel=(
            "Repeat"
        ),

        recommended_product=(
            "P-001"
        ),
    )


def make_payloads():

    return (
        make_payload(
            "delivery-001",
            "CUST-001",
        ),
        make_payload(
            "delivery-002",
            "CUST-002",
            suppressed=True,
        ),
    )


# ================================================================
# CSV EXPORT
# ================================================================


def test_csv_export_creates_file(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    result = (
        export_delivery_csv(
            make_payloads(),
            path,
        )
    )

    assert result == path
    assert path.is_file()


def test_csv_export_creates_parent_directories(
    tmp_path,
):

    path = (
        tmp_path
        /
        "nested"
        /
        "delivery.csv"
    )

    export_delivery_csv(
        make_payloads(),
        path,
    )

    assert path.is_file()


def test_csv_has_exact_schema(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    export_delivery_csv(
        make_payloads(),
        path,
    )

    frame = pd.read_csv(
        path
    )

    assert tuple(
        frame.columns
    ) == DELIVERY_PAYLOAD_COLUMNS


def test_csv_preserves_customer_order(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    export_delivery_csv(
        make_payloads(),
        path,
    )

    frame = (
        read_delivery_csv(
            path
        )
    )

    assert (
        frame[
            "customer"
        ].tolist()
        ==
        [
            "CUST-001",
            "CUST-002",
        ]
    )


def test_csv_preserves_treatment_and_suppression(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    export_delivery_csv(
        make_payloads(),
        path,
    )

    frame = (
        read_delivery_csv(
            path
        )
    )

    assert (
        frame.loc[
            0,
            "delivery_disposition",
        ]
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )

    assert (
        frame.loc[
            1,
            "delivery_disposition",
        ]
        ==
        DELIVERY_DISPOSITION_SUPPRESSED
    )


# ================================================================
# JSONL EXPORT
# ================================================================


def test_jsonl_export_creates_file(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.jsonl"
    )

    result = (
        export_delivery_jsonl(
            make_payloads(),
            path,
        )
    )

    assert result == path
    assert path.is_file()


def test_jsonl_writes_one_record_per_line(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.jsonl"
    )

    export_delivery_jsonl(
        make_payloads(),
        path,
    )

    lines = (
        path.read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert len(
        lines
    ) == 2


def test_jsonl_has_exact_schema(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.jsonl"
    )

    export_delivery_jsonl(
        make_payloads(),
        path,
    )

    records = (
        read_delivery_jsonl(
            path
        )
    )

    assert len(
        records
    ) == 2

    assert tuple(
        records[
            0
        ].keys()
    ) == DELIVERY_PAYLOAD_COLUMNS


def test_jsonl_preserves_customer_order(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.jsonl"
    )

    export_delivery_jsonl(
        make_payloads(),
        path,
    )

    records = (
        read_delivery_jsonl(
            path
        )
    )

    assert [
        record[
            "customer"
        ]
        for record
        in records
    ] == [
        "CUST-001",
        "CUST-002",
    ]


# ================================================================
# OVERWRITE POLICY
# ================================================================


def test_existing_csv_fails_closed_by_default(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    export_delivery_csv(
        make_payloads(),
        path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):

        export_delivery_csv(
            make_payloads(),
            path,
        )


def test_existing_jsonl_fails_closed_by_default(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.jsonl"
    )

    export_delivery_jsonl(
        make_payloads(),
        path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):

        export_delivery_jsonl(
            make_payloads(),
            path,
        )


def test_csv_overwrite_is_explicit(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    export_delivery_csv(
        make_payloads(),
        path,
    )

    result = (
        export_delivery_csv(
            make_payloads(),
            path,
            overwrite=True,
        )
    )

    assert result == path


def test_jsonl_overwrite_is_explicit(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.jsonl"
    )

    export_delivery_jsonl(
        make_payloads(),
        path,
    )

    result = (
        export_delivery_jsonl(
            make_payloads(),
            path,
            overwrite=True,
        )
    )

    assert result == path


def test_overwrite_must_be_bool(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    with pytest.raises(
        TypeError,
        match="overwrite must be a bool",
    ):

        export_delivery_csv(
            make_payloads(),
            path,
            overwrite="yes",
        )


# ================================================================
# EMPTY BATCH
# ================================================================


def test_empty_csv_preserves_header(
    tmp_path,
):

    path = (
        tmp_path
        /
        "empty.csv"
    )

    export_delivery_csv(
        (),
        path,
    )

    frame = (
        read_delivery_csv(
            path
        )
    )

    assert frame.empty

    assert tuple(
        frame.columns
    ) == DELIVERY_PAYLOAD_COLUMNS


def test_empty_jsonl_creates_empty_file(
    tmp_path,
):

    path = (
        tmp_path
        /
        "empty.jsonl"
    )

    export_delivery_jsonl(
        (),
        path,
    )

    assert (
        path.read_text(
            encoding="utf-8"
        )
        ==
        ""
    )


# ================================================================
# FAIL-CLOSED INPUT
# ================================================================


def test_export_rejects_single_payload_object(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    with pytest.raises(
        TypeError,
        match="iterable",
    ):

        export_delivery_csv(
            make_payloads()[
                0
            ],
            path,
        )


def test_export_rejects_wrong_member_type(
    tmp_path,
):

    path = (
        tmp_path
        /
        "delivery.jsonl"
    )

    with pytest.raises(
        TypeError,
        match="Every payload",
    ):

        export_delivery_jsonl(
            (
                {},
            ),
            path,
        )


def test_directory_export_path_fails_closed(
    tmp_path,
):

    with pytest.raises(
        ValueError,
        match="must be a file",
    ):

        export_delivery_csv(
            make_payloads(),
            tmp_path,
        )


# ================================================================
# READ FAILURES
# ================================================================


def test_missing_csv_fails_closed(
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):

        read_delivery_csv(
            tmp_path
            /
            "missing.csv"
        )


def test_invalid_jsonl_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "broken.jsonl"
    )

    path.write_text(
        "{broken}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):

        read_delivery_jsonl(
            path
        )


def test_jsonl_schema_drift_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "broken.jsonl"
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

        read_delivery_jsonl(
            path
        )