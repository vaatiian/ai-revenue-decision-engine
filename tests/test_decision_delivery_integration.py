"""
End-to-end integration tests from canonical Decision Engine output to
governed CRM/action delivery artifacts.
"""

from datetime import datetime, timezone

import pandas as pd

from src.delivery.audit import (
    append_delivery_batch_audit,
    list_delivery_audit,
)

from src.delivery.builder import (
    build_delivery_batch,
    delivery_batch_to_frame,
)

from src.delivery.export import (
    export_delivery_csv,
    export_delivery_jsonl,
    read_delivery_csv,
    read_delivery_jsonl,
)

from src.delivery.spec import (
    DELIVERY_DISPOSITION_HOLD_CONTROL,
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_PAYLOAD_COLUMNS,
    DELIVERY_ROUTE_AUTOMATED,
    DELIVERY_ROUTE_CAMPAIGN,
    DELIVERY_ROUTE_HIGH_TOUCH,
    DELIVERY_ROUTE_HOLD_CONTROL,
    DELIVERY_ROUTE_SUPPRESSED,
)


CREATED_AT = datetime(
    2026,
    10,
    27,
    18,
    0,
    tzinfo=timezone.utc,
)


def make_canonical_decisions():
    """
    Minimal canonical Decision Engine-shaped delivery input spanning all
    delivery dispositions.
    """

    rows = [
        {
            "customer":
                "P1",

            "decision_date":
                "2011-08-31",

            "customer_state":
                "Active",

            "primary_action":
                "Protect",

            "primary_action_score":
                0.95,

            "action_priority_percentile":
                0.95,

            "priority_tier":
                "Priority 1",

            "contact_intensity":
                "High Touch",

            "business_action":
                "Protect customer value; review leakage risk and reinforce "
                "likely repeat purchase",

            "decision_reason":
                "Active customer with leakage risk",

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
                "Retain",

            "primary_action_score":
                0.80,

            "action_priority_percentile":
                0.75,

            "priority_tier":
                "Priority 2",

            "contact_intensity":
                "Campaign",

            "business_action":
                "Retention outreach using relevant repeat-product cue",

            "decision_reason":
                "At-risk customer with retention signal",

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
                "Reactivate",

            "primary_action_score":
                0.55,

            "action_priority_percentile":
                0.55,

            "priority_tier":
                "Priority 3",

            "contact_intensity":
                "Automated / Low Cost",

            "business_action":
                "Controlled reactivation campaign; test outreach against "
                "appropriate holdout/control group",

            "decision_reason":
                "Dormant customer with current reactivation signal",

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
                "Grow",

            "primary_action_score":
                0.20,

            "action_priority_percentile":
                0.20,

            "priority_tier":
                "Priority 4",

            "contact_intensity":
                "Hold / Control",

            "business_action":
                "Cross-sell / discovery offer using current "
                "next-best-product recommendation",

            "decision_reason":
                "Active customer with current product recommendation and "
                "future-value signal",

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
                "Monitor",

            "primary_action_score":
                pd.NA,

            "action_priority_percentile":
                pd.NA,

            "priority_tier":
                "Not Prioritised",

            "contact_intensity":
                "No Model-Led Contact",

            "business_action":
                "Monitor customer without immediate intervention",

            "decision_reason":
                "No model-supported intervention currently takes precedence",

            "experiment_requirement":
                "Not Applicable",

            "recommended_product_channel":
                pd.NA,

            "recommended_product":
                pd.NA,
        },
    ]

    return pd.DataFrame(
        rows
    )


def make_ids():

    return {
        "P1":
            "delivery-p1",

        "R2":
            "delivery-r2",

        "X3":
            "delivery-x3",

        "G4":
            "delivery-g4",

        "M1":
            "delivery-m1",
    }


# ================================================================
# DECISION → PAYLOAD
# ================================================================


def test_decision_output_builds_one_payload_per_customer():

    decisions = (
        make_canonical_decisions()
    )

    payloads = (
        build_delivery_batch(
            decisions,
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert len(
        payloads
    ) == len(
        decisions
    )


def test_customer_order_survives_decision_to_delivery():

    decisions = (
        make_canonical_decisions()
    )

    payloads = (
        build_delivery_batch(
            decisions,
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert [
        payload.customer
        for payload
        in payloads
    ] == [
        "P1",
        "R2",
        "X3",
        "G4",
        "M1",
    ]


def test_primary_actions_are_preserved_exactly():

    decisions = (
        make_canonical_decisions()
    )

    payloads = (
        build_delivery_batch(
            decisions,
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert [
        payload.primary_action
        for payload
        in payloads
    ] == decisions[
        "primary_action"
    ].tolist()


def test_business_actions_are_preserved_exactly():

    decisions = (
        make_canonical_decisions()
    )

    payloads = (
        build_delivery_batch(
            decisions,
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert [
        payload.business_action
        for payload
        in payloads
    ] == decisions[
        "business_action"
    ].tolist()


def test_experiment_requirements_are_preserved_exactly():

    decisions = (
        make_canonical_decisions()
    )

    payloads = (
        build_delivery_batch(
            decisions,
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert [
        payload.experiment_requirement
        for payload
        in payloads
    ] == decisions[
        "experiment_requirement"
    ].tolist()


def test_recommended_products_are_preserved():

    decisions = (
        make_canonical_decisions()
    )

    payloads = (
        build_delivery_batch(
            decisions,
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert (
        payloads[
            0
        ].recommended_product
        ==
        "P-001"
    )

    assert (
        payloads[
            1
        ].recommended_product
        ==
        "P-002"
    )

    assert (
        payloads[
            3
        ].recommended_product
        ==
        "P-004"
    )

    assert (
        payloads[
            2
        ].recommended_product
        is None
    )

    assert (
        payloads[
            4
        ].recommended_product
        is None
    )


# ================================================================
# CONTACT POLICY → DELIVERY POLICY
# ================================================================


def test_high_touch_maps_to_treatment_queue():

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert (
        payloads[
            0
        ].delivery_route
        ==
        DELIVERY_ROUTE_HIGH_TOUCH
    )

    assert (
        payloads[
            0
        ].delivery_disposition
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_campaign_maps_to_treatment_queue():

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert (
        payloads[
            1
        ].delivery_route
        ==
        DELIVERY_ROUTE_CAMPAIGN
    )

    assert (
        payloads[
            1
        ].delivery_disposition
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_automated_maps_to_treatment_queue():

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert (
        payloads[
            2
        ].delivery_route
        ==
        DELIVERY_ROUTE_AUTOMATED
    )

    assert (
        payloads[
            2
        ].delivery_disposition
        ==
        DELIVERY_DISPOSITION_TREATMENT
    )


def test_priority_four_stays_hold_control():

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert (
        payloads[
            3
        ].delivery_route
        ==
        DELIVERY_ROUTE_HOLD_CONTROL
    )

    assert (
        payloads[
            3
        ].delivery_disposition
        ==
        DELIVERY_DISPOSITION_HOLD_CONTROL
    )


def test_non_actionable_stays_suppressed():

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    assert (
        payloads[
            4
        ].delivery_route
        ==
        DELIVERY_ROUTE_SUPPRESSED
    )

    assert (
        payloads[
            4
        ].delivery_disposition
        ==
        DELIVERY_DISPOSITION_SUPPRESSED
    )


# ================================================================
# FRAME
# ================================================================


def test_delivery_frame_has_canonical_schema():

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
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


# ================================================================
# EXPORT
# ================================================================


def test_full_batch_exports_to_csv_and_reads_back(
    tmp_path,
):

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    path = (
        tmp_path
        /
        "delivery.csv"
    )

    export_delivery_csv(
        payloads,
        path,
    )

    frame = (
        read_delivery_csv(
            path
        )
    )

    assert len(
        frame
    ) == 5

    assert (
        frame[
            "delivery_id"
        ].tolist()
        ==
        [
            "delivery-p1",
            "delivery-r2",
            "delivery-x3",
            "delivery-g4",
            "delivery-m1",
        ]
    )


def test_full_batch_exports_to_jsonl_and_reads_back(
    tmp_path,
):

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    path = (
        tmp_path
        /
        "delivery.jsonl"
    )

    export_delivery_jsonl(
        payloads,
        path,
    )

    records = (
        read_delivery_jsonl(
            path
        )
    )

    assert len(
        records
    ) == 5

    assert [
        record[
            "customer"
        ]
        for record
        in records
    ] == [
        "P1",
        "R2",
        "X3",
        "G4",
        "M1",
    ]


# ================================================================
# AUDIT
# ================================================================


def test_full_batch_is_audited_without_policy_loss(
    tmp_path,
):

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    path = (
        tmp_path
        /
        "audit.jsonl"
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

    assert (
        loaded
        ==
        payloads
    )


def test_audit_contains_treatment_hold_and_suppression(
    tmp_path,
):

    payloads = (
        build_delivery_batch(
            make_canonical_decisions(),
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    path = (
        tmp_path
        /
        "audit.jsonl"
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
        payload.delivery_disposition
        for payload
        in loaded
    ] == [
        DELIVERY_DISPOSITION_TREATMENT,
        DELIVERY_DISPOSITION_TREATMENT,
        DELIVERY_DISPOSITION_TREATMENT,
        DELIVERY_DISPOSITION_HOLD_CONTROL,
        DELIVERY_DISPOSITION_SUPPRESSED,
    ]


# ================================================================
# COMPLETE DELIVERY PIPELINE
# ================================================================


def test_decision_to_export_and_audit_pipeline(
    tmp_path,
):

    decisions = (
        make_canonical_decisions()
    )

    payloads = (
        build_delivery_batch(
            decisions,
            delivery_created_at_utc=(
                CREATED_AT
            ),
            delivery_ids=(
                make_ids()
            ),
        )
    )

    csv_path = (
        tmp_path
        /
        "crm"
        /
        "delivery.csv"
    )

    jsonl_path = (
        tmp_path
        /
        "crm"
        /
        "delivery.jsonl"
    )

    audit_path = (
        tmp_path
        /
        "audit"
        /
        "delivery_audit.jsonl"
    )

    export_delivery_csv(
        payloads,
        csv_path,
    )

    export_delivery_jsonl(
        payloads,
        jsonl_path,
    )

    append_delivery_batch_audit(
        payloads,
        audit_path,
    )

    csv_frame = (
        read_delivery_csv(
            csv_path
        )
    )

    jsonl_records = (
        read_delivery_jsonl(
            jsonl_path
        )
    )

    audit_records = (
        list_delivery_audit(
            audit_path
        )
    )

    assert len(
        csv_frame
    ) == 5

    assert len(
        jsonl_records
    ) == 5

    assert len(
        audit_records
    ) == 5

    assert (
        csv_frame[
            "delivery_id"
        ].tolist()
        ==
        [
            payload.delivery_id
            for payload
            in audit_records
        ]
    )

    assert [
        record[
            "delivery_id"
        ]
        for record
        in jsonl_records
    ] == [
        payload.delivery_id
        for payload
        in audit_records
    ]