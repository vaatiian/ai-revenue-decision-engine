"""
Pure data logic for the Revenue Command Centre.

This module contains no Streamlit code.

Its purpose is to transform the curated, anonymised customer decision
dataset into safe executive summaries, filters, distributions, and
priority queues for the public portfolio application.

No production decisions are created or modified here.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


REQUIRED_DECISION_COLUMNS = {
    "portfolio_customer_id",
    "customer_state",
    "primary_action",
    "priority_tier",
    "contact_intensity",
    "experiment_requirement",
    "revenue_prior_90d",
    "future_customer_value_90d",
    "leakage_exposure_score",
    "action_priority_percentile",
    "primary_action_score",
    "leakage_probability",
    "fcv_tier",
    "reactivation_probability",
    "decision_reason",
    "business_action",
}


QUEUE_COLUMNS = (
    "portfolio_customer_id",
    "customer_state",
    "primary_action",
    "priority_tier",
    "contact_intensity",
    "experiment_requirement",
    "action_priority_percentile",
    "primary_action_score",
    "revenue_prior_90d",
    "leakage_probability",
    "leakage_exposure_score",
    "future_customer_value_90d",
    "fcv_tier",
    "reactivation_probability",
    "decision_reason",
    "business_action",
)


def validate_decision_frame(
    frame: pd.DataFrame,
) -> None:
    """
    Fail closed when the curated decision frame is incomplete.
    """

    missing = (
        REQUIRED_DECISION_COLUMNS
        -
        set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Revenue Command Centre decision data is incomplete: "
            f"{sorted(missing)}"
        )


def filter_values(
    frame: pd.DataFrame,
    column: str,
) -> list[str]:
    """
    Return deterministic filter choices for one column.
    """

    validate_decision_frame(
        frame
    )

    if column not in frame.columns:
        raise KeyError(
            f"Unknown Revenue Command Centre filter column: {column}"
        )

    values = (
        frame[
            column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(
        values
    )


def _normalise_selection(
    values: Iterable[str] | None,
) -> set[str]:
    """
    Convert optional UI selections into a normalized set.
    """

    if values is None:
        return set()

    return {
        str(value)
        for value in values
    }


def filter_decisions(
    frame: pd.DataFrame,
    *,
    states: Iterable[str] | None = None,
    actions: Iterable[str] | None = None,
    priorities: Iterable[str] | None = None,
    contacts: Iterable[str] | None = None,
    experiments: Iterable[str] | None = None,
    customer_query: str | None = None,
) -> pd.DataFrame:
    """
    Apply portfolio-safe executive filters.

    Empty filter selections mean "all".
    """

    validate_decision_frame(
        frame
    )

    result = frame.copy()

    filters = {
        "customer_state":
            _normalise_selection(
                states
            ),

        "primary_action":
            _normalise_selection(
                actions
            ),

        "priority_tier":
            _normalise_selection(
                priorities
            ),

        "contact_intensity":
            _normalise_selection(
                contacts
            ),

        "experiment_requirement":
            _normalise_selection(
                experiments
            ),
    }

    for column, selected in filters.items():

        if selected:

            result = result[
                result[
                    column
                ]
                .astype(str)
                .isin(
                    selected
                )
            ]

    query = (
        str(
            customer_query
            or
            ""
        )
        .strip()
        .upper()
    )

    if query:

        result = result[
            result[
                "portfolio_customer_id"
            ]
            .astype(str)
            .str.upper()
            .str.contains(
                query,
                regex=False,
                na=False,
            )
        ]

    return result.reset_index(
        drop=True
    )


def summarise_command_centre(
    frame: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Create executive metrics from the filtered decision population.

    Monetary-looking values are descriptive dataset units and model
    outputs. They are not realised business impact.
    """

    validate_decision_frame(
        frame
    )

    customer_count = int(
        len(
            frame
        )
    )

    if customer_count == 0:

        return {
            "customer_count":
                0,

            "prioritised_count":
                0,

            "priority_1_count":
                0,

            "experiment_required_count":
                0,

            "high_touch_count":
                0,

            "revenue_prior_90d":
                0.0,

            "future_customer_value_90d":
                0.0,

            "leakage_exposure_score":
                0.0,
        }

    prioritised = (
        frame[
            "priority_tier"
        ]
        .fillna(
            "Not Prioritised"
        )
        .ne(
            "Not Prioritised"
        )
    )

    priority_1 = (
        frame[
            "priority_tier"
        ]
        .eq(
            "Priority 1"
        )
    )

    experiment_required = (
        frame[
            "experiment_requirement"
        ]
        .eq(
            "Required"
        )
    )

    high_touch = (
        frame[
            "contact_intensity"
        ]
        .eq(
            "High Touch"
        )
    )

    return {
        "customer_count":
            customer_count,

        "prioritised_count":
            int(
                prioritised.sum()
            ),

        "priority_1_count":
            int(
                priority_1.sum()
            ),

        "experiment_required_count":
            int(
                experiment_required.sum()
            ),

        "high_touch_count":
            int(
                high_touch.sum()
            ),

        "revenue_prior_90d":
            float(
                pd.to_numeric(
                    frame[
                        "revenue_prior_90d"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
                .sum()
            ),

        "future_customer_value_90d":
            float(
                pd.to_numeric(
                    frame[
                        "future_customer_value_90d"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
                .sum()
            ),

        "leakage_exposure_score":
            float(
                pd.to_numeric(
                    frame[
                        "leakage_exposure_score"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
                .sum()
            ),
    }


def build_distribution(
    frame: pd.DataFrame,
    column: str,
    *,
    missing_label: str = "Not Available",
) -> pd.DataFrame:
    """
    Produce deterministic count data for executive charts.
    """

    validate_decision_frame(
        frame
    )

    if column not in frame.columns:
        raise KeyError(
            f"Unknown distribution column: {column}"
        )

    result = (
        frame[
            column
        ]
        .fillna(
            missing_label
        )
        .astype(str)
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "category"
        )
        .rename(
            "customers"
        )
        .reset_index()
    )

    return (
        result
        .sort_values(
            [
                "customers",
                "category",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def build_priority_queue(
    frame: pd.DataFrame,
    *,
    limit: int = 25,
) -> pd.DataFrame:
    """
    Build the executive customer-action queue.

    Only currently prioritised customers appear in the queue.
    """

    validate_decision_frame(
        frame
    )

    if limit < 1:
        raise ValueError(
            "Priority queue limit must be at least 1."
        )

    actionable = frame[
        frame[
            "priority_tier"
        ]
        .fillna(
            "Not Prioritised"
        )
        .ne(
            "Not Prioritised"
        )
    ].copy()

    if actionable.empty:
        return pd.DataFrame(
            columns=QUEUE_COLUMNS
        )

    actionable[
        "_priority_sort"
    ] = pd.to_numeric(
        actionable[
            "action_priority_percentile"
        ],
        errors="coerce",
    )

    actionable[
        "_action_score_sort"
    ] = pd.to_numeric(
        actionable[
            "primary_action_score"
        ],
        errors="coerce",
    )

    actionable = actionable.sort_values(
        [
            "_priority_sort",
            "_action_score_sort",
            "portfolio_customer_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last",
    )

    return (
        actionable[
            list(
                QUEUE_COLUMNS
            )
        ]
        .head(
            limit
        )
        .reset_index(
            drop=True
        )
    )


def build_risk_value_view(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact dataset for the risk-versus-value scatter view.

    Only records with both leakage probability and FCV are included.
    """

    validate_decision_frame(
        frame
    )

    required = frame[
        [
            "portfolio_customer_id",
            "customer_state",
            "primary_action",
            "priority_tier",
            "leakage_probability",
            "future_customer_value_90d",
            "lifetime_revenue",
        ]
    ].copy()

    required[
        "leakage_probability"
    ] = pd.to_numeric(
        required[
            "leakage_probability"
        ],
        errors="coerce",
    )

    required[
        "future_customer_value_90d"
    ] = pd.to_numeric(
        required[
            "future_customer_value_90d"
        ],
        errors="coerce",
    )

    required[
        "lifetime_revenue"
    ] = pd.to_numeric(
        required[
            "lifetime_revenue"
        ],
        errors="coerce",
    )

    required = required.dropna(
        subset=[
            "leakage_probability",
            "future_customer_value_90d",
        ]
    )

    return required.reset_index(
        drop=True
    )