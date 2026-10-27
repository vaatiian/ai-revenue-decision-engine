"""
Batch CRM/action-delivery payload builder.

This module converts a canonical Decision Engine output frame into a batch
of immutable DeliveryPayload records.

Responsibilities
----------------
- validate and apply delivery eligibility;
- preserve one payload per customer;
- preserve customer order;
- generate or derive delivery identifiers;
- attach one common delivery-created timestamp;
- preserve all governed decision and product fields exactly.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd

from src.delivery.eligibility import (
    apply_delivery_eligibility,
)
from src.delivery.payload import (
    DeliveryPayload,
)
from src.delivery.spec import (
    DELIVERY_PAYLOAD_COLUMNS,
)


# ================================================================
# ID GENERATION
# ================================================================


def generate_delivery_id() -> str:
    """
    Generate one globally unique delivery identifier.
    """

    return str(
        uuid4()
    )


# ================================================================
# TIMESTAMP NORMALISATION
# ================================================================


def _normalise_delivery_created_at(
    value,
) -> datetime:
    """
    Convert one timestamp-like value to timezone-aware UTC datetime.
    """

    try:

        timestamp = pd.Timestamp(
            value
        )

    except Exception as exc:

        raise ValueError(
            "delivery_created_at_utc is not a valid timestamp."
        ) from exc

    if pd.isna(
        timestamp
    ):
        raise ValueError(
            "delivery_created_at_utc is not a valid timestamp."
        )

    if timestamp.tzinfo is None:

        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return timestamp.to_pydatetime()


# ================================================================
# SINGLE-ROW BUILDER
# ================================================================


def _build_payload_from_row(
    row,
    *,
    delivery_id: str,
    delivery_created_at_utc: datetime,
) -> DeliveryPayload:
    """
    Convert one already-routed decision row to DeliveryPayload.
    """

    return DeliveryPayload(
        delivery_id=(
            delivery_id
        ),

        delivery_created_at_utc=(
            delivery_created_at_utc
        ),

        customer=(
            row.customer
        ),

        decision_date=(
            row.decision_date
        ),

        customer_state=(
            row.customer_state
        ),

        primary_action=(
            row.primary_action
        ),

        primary_action_score=(
            row.primary_action_score
        ),

        action_priority_percentile=(
            row.action_priority_percentile
        ),

        priority_tier=(
            row.priority_tier
        ),

        contact_intensity=(
            row.contact_intensity
        ),

        delivery_route=(
            row.delivery_route
        ),

        delivery_disposition=(
            row.delivery_disposition
        ),

        business_action=(
            row.business_action
        ),

        decision_reason=(
            row.decision_reason
        ),

        experiment_requirement=(
            row.experiment_requirement
        ),

        recommended_product_channel=(
            row.recommended_product_channel
        ),

        recommended_product=(
            row.recommended_product
        ),
    )


# ================================================================
# BATCH BUILDER
# ================================================================


def build_delivery_batch(
    decisions: pd.DataFrame,
    *,
    delivery_created_at_utc,
    delivery_ids: dict[str, str] | None = None,
) -> tuple[
    DeliveryPayload,
    ...,
]:
    """
    Build one immutable delivery payload per Decision Engine customer.

    Parameters
    ----------
    decisions:
        Canonical Decision Engine output.

    delivery_created_at_utc:
        One common batch creation timestamp.

    delivery_ids:
        Optional customer -> delivery_id mapping.
        If omitted, UUID identifiers are generated.

    Returns
    -------
    tuple[DeliveryPayload, ...]
        Stable customer-order batch.
    """

    created_at = (
        _normalise_delivery_created_at(
            delivery_created_at_utc
        )
    )

    routed = (
        apply_delivery_eligibility(
            decisions
        )
    )

    if delivery_ids is not None:

        if not isinstance(
            delivery_ids,
            dict,
        ):
            raise TypeError(
                "delivery_ids must be a dictionary or None."
            )

        customer_values = (
            routed[
                "customer"
            ]
            .astype(
                str
            )
            .tolist()
        )

        customer_set = set(
            customer_values
        )

        provided_set = set(
            delivery_ids.keys()
        )

        if (
            provided_set
            !=
            customer_set
        ):
            raise ValueError(
                "delivery_ids keys must exactly match the delivery "
                "customer population."
            )

        for (
            customer,
            delivery_id,
        ) in delivery_ids.items():

            if not isinstance(
                customer,
                str,
            ):
                raise TypeError(
                    "delivery_ids keys must be strings."
                )

            if not isinstance(
                delivery_id,
                str,
            ):
                raise TypeError(
                    "delivery_ids values must be strings."
                )

            if not delivery_id.strip():
                raise ValueError(
                    "delivery_ids values must not be empty."
                )

        if len(
            set(
                delivery_ids.values()
            )
        ) != len(
            delivery_ids
        ):
            raise ValueError(
                "delivery_ids values must be unique."
            )

    payloads = []

    for row in routed.itertuples(
        index=False
    ):

        customer_key = str(
            row.customer
        )

        if delivery_ids is None:

            delivery_id = (
                generate_delivery_id()
            )

        else:

            delivery_id = (
                delivery_ids[
                    customer_key
                ]
            )

        payloads.append(
            _build_payload_from_row(
                row,
                delivery_id=(
                    delivery_id
                ),
                delivery_created_at_utc=(
                    created_at
                ),
            )
        )

    if len(
        payloads
    ) != len(
        decisions
    ):
        raise RuntimeError(
            "Delivery batch builder changed the customer population."
        )

    payload_customers = [
        payload.customer
        for payload
        in payloads
    ]

    decision_customers = [
        str(
            customer
        )
        for customer
        in decisions[
            "customer"
        ].tolist()
    ]

    if (
        payload_customers
        !=
        decision_customers
    ):
        raise RuntimeError(
            "Delivery batch builder changed customer order."
        )

    return tuple(
        payloads
    )


# ================================================================
# FRAME REPRESENTATION
# ================================================================


def delivery_batch_to_frame(
    payloads,
) -> pd.DataFrame:
    """
    Convert a payload collection to the canonical delivery payload frame.
    """

    from src.delivery.payload import (
        delivery_payload_to_dict,
    )

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

    rows = []

    for payload in payload_tuple:

        if not isinstance(
            payload,
            DeliveryPayload,
        ):
            raise TypeError(
                "Every payload must be a DeliveryPayload."
            )

        rows.append(
            delivery_payload_to_dict(
                payload
            )
        )

    return pd.DataFrame(
        rows,
        columns=(
            DELIVERY_PAYLOAD_COLUMNS
        ),
    )