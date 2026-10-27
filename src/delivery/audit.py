"""
Append-only delivery audit ledger.

The audit ledger records immutable DeliveryPayload objects exactly as they were
prepared for downstream execution.

Unlike delivery export files, the audit ledger is historical and append-only.

Governance rules
----------------
- every audit entry must match the canonical delivery payload schema;
- delivery_id values must be globally unique within one ledger;
- existing audit history is validated before new records are appended;
- malformed history fails closed;
- treatment, hold/control, and suppressed outcomes are all retained.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.delivery.payload import (
    DeliveryPayload,
    delivery_payload_to_dict,
)

from src.delivery.spec import (
    DELIVERY_PAYLOAD_COLUMNS,
)


# ================================================================
# DEFAULT STORE
# ================================================================


DEFAULT_DELIVERY_AUDIT_PATH = Path(
    "artifacts/delivery/delivery_audit.jsonl"
)


# ================================================================
# SERIALISATION
# ================================================================


def _payload_to_json_line(
    payload: DeliveryPayload,
) -> str:
    """
    Convert one validated DeliveryPayload to compact JSON.
    """

    if not isinstance(
        payload,
        DeliveryPayload,
    ):
        raise TypeError(
            "payload must be a DeliveryPayload."
        )

    record = (
        delivery_payload_to_dict(
            payload
        )
    )

    if tuple(
        record.keys()
    ) != DELIVERY_PAYLOAD_COLUMNS:
        raise RuntimeError(
            "Delivery audit record does not match "
            "DELIVERY_PAYLOAD_COLUMNS."
        )

    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        sort_keys=False,
    )


def _record_to_payload(
    record,
) -> DeliveryPayload:
    """
    Convert one parsed audit JSON object back to DeliveryPayload.
    """

    if not isinstance(
        record,
        dict,
    ):
        raise ValueError(
            "Delivery audit record must be a JSON object."
        )

    if tuple(
        record.keys()
    ) != DELIVERY_PAYLOAD_COLUMNS:
        raise ValueError(
            "Delivery audit schema does not match "
            "DELIVERY_PAYLOAD_COLUMNS."
        )

    return DeliveryPayload(
        delivery_id=(
            record[
                "delivery_id"
            ]
        ),

        delivery_created_at_utc=(
            record[
                "delivery_created_at_utc"
            ]
        ),

        customer=(
            record[
                "customer"
            ]
        ),

        decision_date=(
            record[
                "decision_date"
            ]
        ),

        customer_state=(
            record[
                "customer_state"
            ]
        ),

        primary_action=(
            record[
                "primary_action"
            ]
        ),

        primary_action_score=(
            record[
                "primary_action_score"
            ]
        ),

        action_priority_percentile=(
            record[
                "action_priority_percentile"
            ]
        ),

        priority_tier=(
            record[
                "priority_tier"
            ]
        ),

        contact_intensity=(
            record[
                "contact_intensity"
            ]
        ),

        delivery_route=(
            record[
                "delivery_route"
            ]
        ),

        delivery_disposition=(
            record[
                "delivery_disposition"
            ]
        ),

        business_action=(
            record[
                "business_action"
            ]
        ),

        decision_reason=(
            record[
                "decision_reason"
            ]
        ),

        experiment_requirement=(
            record[
                "experiment_requirement"
            ]
        ),

        recommended_product_channel=(
            record[
                "recommended_product_channel"
            ]
        ),

        recommended_product=(
            record[
                "recommended_product"
            ]
        ),
    )


# ================================================================
# READ
# ================================================================


def list_delivery_audit(
    path: str | Path = DEFAULT_DELIVERY_AUDIT_PATH,
) -> tuple[
    DeliveryPayload,
    ...,
]:
    """
    Read and validate the complete append-only delivery audit ledger.

    A missing audit file represents an empty ledger.
    """

    audit_path = Path(
        path
    )

    if not audit_path.exists():
        return ()

    if not audit_path.is_file():
        raise ValueError(
            "Delivery audit path must be a file."
        )

    payloads = []

    seen_delivery_ids = set()

    with audit_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for (
            line_number,
            line,
        ) in enumerate(
            handle,
            start=1,
        ):

            if not line.strip():
                raise ValueError(
                    "Delivery audit contains an empty line at "
                    f"line {line_number}."
                )

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Delivery audit contains invalid JSON at "
                    f"line {line_number}."
                ) from exc

            payload = (
                _record_to_payload(
                    record
                )
            )

            if (
                payload.delivery_id
                in
                seen_delivery_ids
            ):
                raise ValueError(
                    "Delivery audit contains duplicate delivery_id: "
                    f"{payload.delivery_id}"
                )

            seen_delivery_ids.add(
                payload.delivery_id
            )

            payloads.append(
                payload
            )

    return tuple(
        payloads
    )


def get_delivery_audit(
    delivery_id: str,
    path: str | Path = DEFAULT_DELIVERY_AUDIT_PATH,
) -> DeliveryPayload:
    """
    Retrieve exactly one delivery audit record by delivery_id.
    """

    if not isinstance(
        delivery_id,
        str,
    ):
        raise TypeError(
            "delivery_id must be a string."
        )

    if not delivery_id.strip():
        raise ValueError(
            "delivery_id must not be empty."
        )

    for payload in list_delivery_audit(
        path
    ):

        if (
            payload.delivery_id
            ==
            delivery_id
        ):
            return payload

    raise KeyError(
        f"Unknown delivery_id: {delivery_id}"
    )


# ================================================================
# APPEND ONE
# ================================================================


def append_delivery_audit(
    payload: DeliveryPayload,
    path: str | Path = DEFAULT_DELIVERY_AUDIT_PATH,
) -> Path:
    """
    Append one validated payload to the delivery audit ledger.
    """

    if not isinstance(
        payload,
        DeliveryPayload,
    ):
        raise TypeError(
            "payload must be a DeliveryPayload."
        )

    audit_path = Path(
        path
    )

    existing = (
        list_delivery_audit(
            audit_path
        )
    )

    existing_ids = {
        existing_payload.delivery_id
        for existing_payload
        in existing
    }

    if (
        payload.delivery_id
        in
        existing_ids
    ):
        raise ValueError(
            "delivery_id already exists in delivery audit: "
            f"{payload.delivery_id}"
        )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    line = (
        _payload_to_json_line(
            payload
        )
    )

    with audit_path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        handle.write(
            line
        )

        handle.write(
            "\n"
        )

    return audit_path


# ================================================================
# APPEND BATCH
# ================================================================


def append_delivery_batch_audit(
    payloads,
    path: str | Path = DEFAULT_DELIVERY_AUDIT_PATH,
) -> Path:
    """
    Validate and append one complete delivery batch atomically at batch level.

    All existing and incoming IDs are checked before any new line is written.
    """

    if isinstance(
        payloads,
        DeliveryPayload,
    ):
        raise TypeError(
            "payloads must be an iterable of DeliveryPayload records."
        )

    try:

        payload_tuple = tuple(
            payloads
        )

    except TypeError as exc:

        raise TypeError(
            "payloads must be an iterable of DeliveryPayload records."
        ) from exc

    for payload in payload_tuple:

        if not isinstance(
            payload,
            DeliveryPayload,
        ):
            raise TypeError(
                "Every payload must be a DeliveryPayload."
            )

    audit_path = Path(
        path
    )

    existing = (
        list_delivery_audit(
            audit_path
        )
    )

    existing_ids = {
        payload.delivery_id
        for payload
        in existing
    }

    incoming_ids = [
        payload.delivery_id
        for payload
        in payload_tuple
    ]

    if len(
        incoming_ids
    ) != len(
        set(
            incoming_ids
        )
    ):
        raise ValueError(
            "Incoming delivery batch contains duplicate delivery_id values."
        )

    collisions = (
        set(
            incoming_ids
        )
        &
        existing_ids
    )

    if collisions:
        raise ValueError(
            "Delivery audit already contains delivery_id values: "
            f"{sorted(collisions)}"
        )

    lines = [
        _payload_to_json_line(
            payload
        )
        for payload
        in payload_tuple
    ]

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not lines:

        if not audit_path.exists():

            audit_path.touch()

        return audit_path

    with audit_path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        for line in lines:

            handle.write(
                line
            )

            handle.write(
                "\n"
            )

    return audit_path