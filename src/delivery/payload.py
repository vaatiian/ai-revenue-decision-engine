"""
Immutable CRM/action-delivery payload.

This module converts one governed Decision Engine + delivery-routing row
into a validated, deeply immutable payload suitable for downstream export
and audit.

The payload is descriptive only. It does not execute customer contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.delivery.spec import (
    DELIVERY_DISPOSITIONS,
    DELIVERY_PAYLOAD_COLUMNS,
    DELIVERY_ROUTES,
)


# ================================================================
# VALIDATION HELPERS
# ================================================================


def _validate_non_empty_string(
    value,
    name: str,
) -> str:
    """
    Validate and return one non-empty string.
    """

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{name} must not be empty."
        )

    return value


def _validate_optional_string(
    value,
    name: str,
):
    """
    Validate one optional string.
    """

    if value is None:
        return None

    if pd.isna(
        value
    ):
        return None

    return _validate_non_empty_string(
        value,
        name,
    )


def _normalise_utc_timestamp(
    value,
    name: str,
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
            f"{name} is not a valid timestamp."
        ) from exc

    if pd.isna(
        timestamp
    ):
        raise ValueError(
            f"{name} is not a valid timestamp."
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


def _normalise_decision_date(
    value,
):
    """
    Convert one decision date to stable ISO date string.
    """

    try:

        timestamp = pd.Timestamp(
            value
        )

    except Exception as exc:

        raise ValueError(
            "decision_date is not a valid timestamp."
        ) from exc

    if pd.isna(
        timestamp
    ):
        raise ValueError(
            "decision_date is not a valid timestamp."
        )

    return timestamp.date().isoformat()


def _normalise_optional_float(
    value,
    name: str,
):
    """
    Convert one optional numeric value to float.
    """

    if value is None:
        return None

    if pd.isna(
        value
    ):
        return None

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"{name} must be numeric or missing."
        ) from exc


# ================================================================
# IMMUTABLE PAYLOAD
# ================================================================


@dataclass(
    frozen=True
)
class DeliveryPayload:
    """
    One immutable downstream CRM/action payload.
    """

    delivery_id: str
    delivery_created_at_utc: datetime

    customer: str
    decision_date: str

    customer_state: str
    primary_action: str
    primary_action_score: float | None
    action_priority_percentile: float | None
    priority_tier: str

    contact_intensity: str
    delivery_route: str
    delivery_disposition: str

    business_action: str
    decision_reason: str
    experiment_requirement: str

    recommended_product_channel: str | None
    recommended_product: str | None

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize one delivery payload.
        """

        for (
            name,
            value,
        ) in (
            (
                "delivery_id",
                self.delivery_id,
            ),
            (
                "customer",
                self.customer,
            ),
            (
                "customer_state",
                self.customer_state,
            ),
            (
                "primary_action",
                self.primary_action,
            ),
            (
                "priority_tier",
                self.priority_tier,
            ),
            (
                "contact_intensity",
                self.contact_intensity,
            ),
            (
                "delivery_route",
                self.delivery_route,
            ),
            (
                "delivery_disposition",
                self.delivery_disposition,
            ),
            (
                "business_action",
                self.business_action,
            ),
            (
                "decision_reason",
                self.decision_reason,
            ),
            (
                "experiment_requirement",
                self.experiment_requirement,
            ),
        ):

            _validate_non_empty_string(
                value,
                name,
            )

        if (
            self.delivery_route
            not in
            DELIVERY_ROUTES
        ):
            raise ValueError(
                "delivery_route is not part of the frozen delivery contract."
            )

        if (
            self.delivery_disposition
            not in
            DELIVERY_DISPOSITIONS
        ):
            raise ValueError(
                "delivery_disposition is not part of the frozen delivery "
                "contract."
            )

        object.__setattr__(
            self,
            "delivery_created_at_utc",
            _normalise_utc_timestamp(
                self.delivery_created_at_utc,
                "delivery_created_at_utc",
            ),
        )

        object.__setattr__(
            self,
            "decision_date",
            _normalise_decision_date(
                self.decision_date
            ),
        )

        object.__setattr__(
            self,
            "primary_action_score",
            _normalise_optional_float(
                self.primary_action_score,
                "primary_action_score",
            ),
        )

        object.__setattr__(
            self,
            "action_priority_percentile",
            _normalise_optional_float(
                self.action_priority_percentile,
                "action_priority_percentile",
            ),
        )

        object.__setattr__(
            self,
            "recommended_product_channel",
            _validate_optional_string(
                self.recommended_product_channel,
                "recommended_product_channel",
            ),
        )

        object.__setattr__(
            self,
            "recommended_product",
            _validate_optional_string(
                self.recommended_product,
                "recommended_product",
            ),
        )


# ================================================================
# SERIALISABLE REPRESENTATION
# ================================================================


def delivery_payload_to_dict(
    payload: DeliveryPayload,
) -> dict[str, Any]:
    """
    Convert one validated payload to a JSON/CSV-ready dictionary.
    """

    if not isinstance(
        payload,
        DeliveryPayload,
    ):
        raise TypeError(
            "payload must be a DeliveryPayload."
        )

    values = {
        "delivery_id":
            payload.delivery_id,

        "delivery_created_at_utc":
            payload.delivery_created_at_utc.isoformat(),

        "customer":
            payload.customer,

        "decision_date":
            payload.decision_date,

        "customer_state":
            payload.customer_state,

        "primary_action":
            payload.primary_action,

        "primary_action_score":
            payload.primary_action_score,

        "action_priority_percentile":
            payload.action_priority_percentile,

        "priority_tier":
            payload.priority_tier,

        "contact_intensity":
            payload.contact_intensity,

        "delivery_route":
            payload.delivery_route,

        "delivery_disposition":
            payload.delivery_disposition,

        "business_action":
            payload.business_action,

        "decision_reason":
            payload.decision_reason,

        "experiment_requirement":
            payload.experiment_requirement,

        "recommended_product_channel":
            payload.recommended_product_channel,

        "recommended_product":
            payload.recommended_product,
    }

    if tuple(
        values.keys()
    ) != DELIVERY_PAYLOAD_COLUMNS:
        raise RuntimeError(
            "Delivery payload dictionary does not match "
            "DELIVERY_PAYLOAD_COLUMNS."
        )

    return values