"""
Pure portfolio logic for the Customer Decision Explorer.

This module explains existing governed customer decisions.

It does not:
- rescore customers;
- create new customer actions;
- alter priority;
- alter contact policy;
- alter experimentation requirements;
- recreate production Decision Engine rules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_EXPLORER_COLUMNS = {
    "portfolio_customer_id",
    "decision_date",
    "customer_state",
    "days_since_last_order",
    "customer_tenure_days",
    "lifetime_orders",
    "lifetime_revenue",
    "orders_prior_30d",
    "orders_prior_90d",
    "revenue_prior_90d",
    "revenue_prior_180d",
    "primary_action",
    "primary_action_score",
    "action_priority_percentile",
    "priority_tier",
    "contact_intensity",
    "business_action",
    "decision_reason",
    "experiment_requirement",
    "leakage_probability",
    "leakage_exposure_score",
    "leakage_exposure_percentile",
    "economic_priority_band",
    "future_active_probability",
    "future_customer_value_90d",
    "fcv_percentile",
    "fcv_tier",
    "reactivation_probability",
    "reactivation_prediction_date",
    "recommended_product_channel",
    "portfolio_recommended_product",
    "portfolio_repeat_top1_product",
    "portfolio_discovery_top1_product",
}


def validate_explorer_frame(
    frame: pd.DataFrame,
) -> None:
    """
    Fail closed if the curated customer-decision schema is incomplete.
    """

    missing = (
        REQUIRED_EXPLORER_COLUMNS
        -
        set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Customer Decision Explorer data is incomplete: "
            f"{sorted(missing)}"
        )

    if frame[
        "portfolio_customer_id"
    ].isna().any():
        raise RuntimeError(
            "Customer Decision Explorer contains missing portfolio IDs."
        )

    if not frame[
        "portfolio_customer_id"
    ].is_unique:
        raise RuntimeError(
            "Customer Decision Explorer requires one row per customer."
        )


def customer_ids(
    frame: pd.DataFrame,
) -> list[str]:
    """
    Return deterministic public customer identifiers.
    """

    validate_explorer_frame(
        frame
    )

    return sorted(
        frame[
            "portfolio_customer_id"
        ]
        .astype(str)
        .tolist()
    )


def filter_customer_candidates(
    frame: pd.DataFrame,
    *,
    state: str | None = None,
    action: str | None = None,
    priority: str | None = None,
) -> pd.DataFrame:
    """
    Narrow the customer population for Explorer navigation.

    This filtering changes only which existing decisions are shown.
    """

    validate_explorer_frame(
        frame
    )

    result = frame.copy()

    filters = {
        "customer_state":
            state,

        "primary_action":
            action,

        "priority_tier":
            priority,
    }

    for column, value in filters.items():

        if (
            value is not None
            and
            value != "All"
        ):
            result = result[
                result[
                    column
                ].astype(str)
                .eq(
                    str(
                        value
                    )
                )
            ]

    return (
        result
        .sort_values(
            "portfolio_customer_id"
        )
        .reset_index(
            drop=True
        )
    )


def get_customer_record(
    frame: pd.DataFrame,
    portfolio_customer_id: str,
) -> pd.Series:
    """
    Fetch one and only one anonymised customer decision.
    """

    validate_explorer_frame(
        frame
    )

    customer_id = str(
        portfolio_customer_id
    ).strip()

    matches = frame[
        frame[
            "portfolio_customer_id"
        ]
        .astype(str)
        .eq(
            customer_id
        )
    ]

    if matches.empty:
        raise KeyError(
            f"Unknown portfolio customer: {customer_id}"
        )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Customer {customer_id} does not resolve to one decision."
        )

    return matches.iloc[
        0
    ].copy()


def _safe_number(
    value: Any,
) -> float | None:
    """
    Convert a possible numeric field to a finite float or None.
    """

    if pd.isna(
        value
    ):
        return None

    try:
        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if pd.isna(
        numeric
    ):
        return None

    return numeric


def _safe_text(
    value: Any,
) -> str | None:
    """
    Convert a possible display value to clean text.
    """

    if pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    return text


def build_customer_dossier(
    record: pd.Series,
) -> dict[str, Any]:
    """
    Convert one existing decision row into presentation-safe sections.
    """

    dossier = {
        "identity": {
            "customer_id":
                _safe_text(
                    record.get(
                        "portfolio_customer_id"
                    )
                ),

            "decision_date":
                _safe_text(
                    record.get(
                        "decision_date"
                    )
                ),

            "state":
                _safe_text(
                    record.get(
                        "customer_state"
                    )
                ),
        },

        "behaviour": {
            "days_since_last_order":
                _safe_number(
                    record.get(
                        "days_since_last_order"
                    )
                ),

            "customer_tenure_days":
                _safe_number(
                    record.get(
                        "customer_tenure_days"
                    )
                ),

            "lifetime_orders":
                _safe_number(
                    record.get(
                        "lifetime_orders"
                    )
                ),

            "lifetime_revenue":
                _safe_number(
                    record.get(
                        "lifetime_revenue"
                    )
                ),

            "orders_prior_30d":
                _safe_number(
                    record.get(
                        "orders_prior_30d"
                    )
                ),

            "orders_prior_90d":
                _safe_number(
                    record.get(
                        "orders_prior_90d"
                    )
                ),

            "revenue_prior_90d":
                _safe_number(
                    record.get(
                        "revenue_prior_90d"
                    )
                ),

            "revenue_prior_180d":
                _safe_number(
                    record.get(
                        "revenue_prior_180d"
                    )
                ),
        },

        "decision": {
            "action":
                _safe_text(
                    record.get(
                        "primary_action"
                    )
                ),

            "action_score":
                _safe_number(
                    record.get(
                        "primary_action_score"
                    )
                ),

            "priority_percentile":
                _safe_number(
                    record.get(
                        "action_priority_percentile"
                    )
                ),

            "priority_tier":
                _safe_text(
                    record.get(
                        "priority_tier"
                    )
                ),

            "contact_intensity":
                _safe_text(
                    record.get(
                        "contact_intensity"
                    )
                ),

            "experiment_requirement":
                _safe_text(
                    record.get(
                        "experiment_requirement"
                    )
                ),

            "business_action":
                _safe_text(
                    record.get(
                        "business_action"
                    )
                ),

            "decision_reason":
                _safe_text(
                    record.get(
                        "decision_reason"
                    )
                ),
        },

        "leakage": {
            "probability":
                _safe_number(
                    record.get(
                        "leakage_probability"
                    )
                ),

            "exposure_score":
                _safe_number(
                    record.get(
                        "leakage_exposure_score"
                    )
                ),

            "exposure_percentile":
                _safe_number(
                    record.get(
                        "leakage_exposure_percentile"
                    )
                ),

            "economic_priority_band":
                _safe_text(
                    record.get(
                        "economic_priority_band"
                    )
                ),
        },

        "future_value": {
            "future_active_probability":
                _safe_number(
                    record.get(
                        "future_active_probability"
                    )
                ),

            "future_customer_value_90d":
                _safe_number(
                    record.get(
                        "future_customer_value_90d"
                    )
                ),

            "fcv_percentile":
                _safe_number(
                    record.get(
                        "fcv_percentile"
                    )
                ),

            "fcv_tier":
                _safe_text(
                    record.get(
                        "fcv_tier"
                    )
                ),
        },

        "reactivation": {
            "probability":
                _safe_number(
                    record.get(
                        "reactivation_probability"
                    )
                ),

            "prediction_date":
                _safe_text(
                    record.get(
                        "reactivation_prediction_date"
                    )
                ),
        },

        "recommendation": {
            "channel":
                _safe_text(
                    record.get(
                        "recommended_product_channel"
                    )
                ),

            "recommended_product":
                _safe_text(
                    record.get(
                        "portfolio_recommended_product"
                    )
                ),

            "repeat_top1_product":
                _safe_text(
                    record.get(
                        "portfolio_repeat_top1_product"
                    )
                ),

            "discovery_top1_product":
                _safe_text(
                    record.get(
                        "portfolio_discovery_top1_product"
                    )
                ),
        },
    }

    return dossier


def evidence_status(
    value: Any,
) -> str:
    """
    Describe whether one model signal exists for this customer.
    """

    if value is None:
        return "Not Available"

    if pd.isna(
        value
    ):
        return "Not Available"

    return "Available"


def build_model_evidence(
    dossier: dict[str, Any],
) -> pd.DataFrame:
    """
    Build one compact model-evidence table for the customer dossier.
    """

    leakage = dossier[
        "leakage"
    ]

    future_value = dossier[
        "future_value"
    ]

    reactivation = dossier[
        "reactivation"
    ]

    recommendation = dossier[
        "recommendation"
    ]

    rows = [
        {
            "Intelligence":
                "Revenue Leakage",

            "Status":
                evidence_status(
                    leakage[
                        "probability"
                    ]
                ),

            "Primary Signal":
                leakage[
                    "probability"
                ],

            "Context":
                leakage[
                    "economic_priority_band"
                ],
        },

        {
            "Intelligence":
                "Future Customer Value",

            "Status":
                evidence_status(
                    future_value[
                        "future_active_probability"
                    ]
                ),

            "Primary Signal":
                future_value[
                    "future_active_probability"
                ],

            "Context":
                future_value[
                    "fcv_tier"
                ],
        },

        {
            "Intelligence":
                "Reactivation",

            "Status":
                evidence_status(
                    reactivation[
                        "probability"
                    ]
                ),

            "Primary Signal":
                reactivation[
                    "probability"
                ],

            "Context":
                reactivation[
                    "prediction_date"
                ],
        },

        {
            "Intelligence":
                "Recommendation",

            "Status":
                evidence_status(
                    recommendation[
                        "recommended_product"
                    ]
                ),

            "Primary Signal":
                None,

            "Context":
                recommendation[
                    "channel"
                ],
        },
    ]

    return pd.DataFrame(
        rows
    )


def governance_lookup(
    model_registry: dict[str, Any],
) -> dict[str, Any]:
    """
    Return public governance metadata keyed by intelligence system.
    """

    systems = model_registry.get(
        "systems"
    )

    if not isinstance(
        systems,
        list,
    ):
        raise RuntimeError(
            "Portfolio model registry systems are invalid."
        )

    result = {}

    for system in systems:

        system_id = system.get(
            "system_id"
        )

        if system_id is None:
            raise RuntimeError(
                "Portfolio model registry contains a system without ID."
            )

        result[
            str(
                system_id
            )
        ] = {
            "display_name":
                system.get(
                    "display_name"
                ),

            "governance":
                system.get(
                    "governance"
                ),

            "model_version":
                system.get(
                    "model_version"
                ),

            "model_type":
                system.get(
                    "model_type"
                ),
        }

    return result