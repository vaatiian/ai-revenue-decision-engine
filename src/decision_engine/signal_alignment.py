"""
Production signal alignment for the Customer Decision Engine.

This module converts governed production scorer outputs into the common
customer-level signal contract required by the deterministic Decision Engine.

Responsibilities
----------------
- normalize scorer-specific field names;
- restore point-in-time provenance intentionally omitted by fitted scorers;
- reproduce research-parity Leakage economic enrichment;
- reproduce research-parity FCV percentile tiers;
- select temporally valid Reactivation signals;
- select the latest eligible Recommendation signal;
- compute signal availability and state-aware operational-validity flags;
- preserve one row per customer.

This module does not:
- load model artifacts;
- generate model features;
- score models;
- assign primary customer actions;
- calculate action priority scores;
- assign contact policy.

Those responsibilities belong to other production layers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision_engine.spec import (
    CUSTOMER_STATE_ACTIVE,
    CUSTOMER_STATE_AT_RISK,
    CUSTOMER_STATE_DORMANT,
    CUSTOMER_STATE_LONG_DORMANT,
    REACTIVATION_MAX_SIGNAL_AGE_DAYS,
)

from src.features.customer_snapshot import (
    normalise_decision_date,
)


# ================================================================
# RESEARCH-PARITY CONSTANTS
# ================================================================


LEAKAGE_MINIMUM_EVENT_LOSS_RATE = 0.50


LEAKAGE_ECONOMIC_PRIORITY_LABELS = (
    "P5 — Lowest Exposure",
    "P4",
    "P3",
    "P2",
    "P1 — Highest Exposure",
)


LEAKAGE_RISK_BAND_LABELS = (
    "Low Risk",
    "Medium Risk",
    "High Risk",
)


LEAKAGE_VALUE_BAND_LABELS = (
    "Low Value",
    "Medium Value",
    "High Value",
)


FCV_TIER_BINS = (
    0.00,
    0.50,
    0.80,
    0.90,
    0.95,
    1.00,
)


FCV_TIER_LABELS = (
    "Core",
    "Growth",
    "High Value",
    "Priority",
    "Premier",
)


RECOMMENDATION_CURRENTNESS_SECONDS = 60


# ================================================================
# ALIGNMENT OUTPUT CONTRACT
# ================================================================


DECISION_SIGNAL_COLUMNS = (
    # Leakage
    "leakage_probability",
    "leakage_probability_percentile",
    "baseline_revenue",
    "leakage_exposure_score",
    "leakage_exposure_percentile",
    "economic_priority_band",
    "risk_band",
    "value_band",

    # FCV
    "fcv_history_180d_revenue",
    "future_active_probability",
    "future_customer_value_90d",
    "fcv_percentile",
    "fcv_tier",

    # Reactivation
    "reactivation_probability",
    "reactivation_prediction_date",
    "reactivation_score_age_days",

    # Recommendation
    "recommendation_prediction_date",
    "repeat_top1_product",
    "discovery_top1_product",

    # Signal-state flags
    "leakage_signal_available",
    "fcv_signal_available",
    "reactivation_signal_available",
    "recommendation_signal_available",
    "recommendation_is_current",

    # Operational validity
    "leakage_operationally_valid",
    "reactivation_operationally_valid",
    "recommendation_operationally_valid",
)


# ================================================================
# GENERIC VALIDATION
# ================================================================


def _validate_dataframe(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """
    Ensure an alignment input is a pandas DataFrame.
    """

    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )


def _validate_required_columns(
    frame: pd.DataFrame,
    required_columns,
    name: str,
) -> None:
    """
    Fail closed when an input is missing contract columns.
    """

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(
                missing
            )
        )


def _validate_unique_customer(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """
    Require one row per customer.
    """

    if frame[
        "customer"
    ].isna().any():
        raise ValueError(
            f"{name} contains missing customer IDs."
        )

    if not frame[
        "customer"
    ].is_unique:
        raise ValueError(
            f"{name} must contain one row per customer."
        )


def _validate_probability(
    series: pd.Series,
    name: str,
) -> None:
    """
    Validate non-missing probabilities when present.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid_original = (
        series.notna()
        &
        values.isna()
    )

    if invalid_original.any():
        raise ValueError(
            f"{name} contains non-numeric values."
        )

    present = (
        values.dropna()
    )

    if (
        (
            present
            <
            0
        ).any()
        or
        (
            present
            >
            1
        ).any()
    ):
        raise ValueError(
            f"{name} must lie within [0, 1]."
        )


def _validate_non_negative(
    series: pd.Series,
    name: str,
) -> None:
    """
    Validate non-negative numeric values when present.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid_original = (
        series.notna()
        &
        values.isna()
    )

    if invalid_original.any():
        raise ValueError(
            f"{name} contains non-numeric values."
        )

    if (
        values.dropna()
        <
        0
    ).any():
        raise ValueError(
            f"{name} cannot contain negative values."
        )


def _research_qcut(
    values: pd.Series,
    q: int,
    labels,
    name: str,
) -> pd.Series:
    """
    Apply the exact research qcut policy.

    Research notebooks assume enough score variation to form every requested
    quantile. Production fails closed if that assumption is violated rather
    than silently inventing different bands.
    """

    try:

        result = pd.qcut(
            values,
            q=q,
            labels=list(
                labels
            ),
            duplicates="drop",
        )

    except ValueError as exc:

        raise ValueError(
            f"Unable to create research-parity {name} quantile bands."
        ) from exc

    return result


# ================================================================
# LEAKAGE ALIGNMENT
# ================================================================


def align_leakage_signals(
    leakage_scores: pd.DataFrame,
    leakage_context: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize and enrich production Revenue Leakage scores.

    Native scorer:
        revenue_leakage_probability

    Decision Engine:
        leakage_probability

    Economic enrichment reproduces Notebook 04:

        leakage_exposure_score
        =
        leakage_probability
        × 0.50
        × baseline_revenue
    """

    _validate_dataframe(
        leakage_scores,
        "Leakage scores",
    )

    _validate_dataframe(
        leakage_context,
        "Leakage context",
    )

    _validate_required_columns(
        leakage_scores,
        (
            "customer",
            "revenue_leakage_probability",
        ),
        "Leakage scores",
    )

    _validate_required_columns(
        leakage_context,
        (
            "customer",
            "baseline_revenue",
        ),
        "Leakage context",
    )

    scores = (
        leakage_scores[
            [
                "customer",
                "revenue_leakage_probability",
            ]
        ]
        .copy()
    )

    context = (
        leakage_context[
            [
                "customer",
                "baseline_revenue",
            ]
        ]
        .copy()
    )

    _validate_unique_customer(
        scores,
        "Leakage scores",
    )

    _validate_unique_customer(
        context,
        "Leakage context",
    )

    _validate_probability(
        scores[
            "revenue_leakage_probability"
        ],
        "revenue_leakage_probability",
    )

    _validate_non_negative(
        context[
            "baseline_revenue"
        ],
        "baseline_revenue",
    )

    result = (
        scores
        .merge(
            context,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "revenue_leakage_probability":
                    "leakage_probability",
            }
        )
    )

    if result[
        "baseline_revenue"
    ].isna().any():
        raise ValueError(
            "Leakage context is missing baseline revenue for scored customers."
        )

    result[
        "leakage_probability_percentile"
    ] = (
        result[
            "leakage_probability"
        ]
        .rank(
            pct=True
        )
    )

    result[
        "leakage_exposure_score"
    ] = (
        result[
            "leakage_probability"
        ]
        *
        LEAKAGE_MINIMUM_EVENT_LOSS_RATE
        *
        result[
            "baseline_revenue"
        ]
    )

    result[
        "leakage_exposure_percentile"
    ] = (
        result[
            "leakage_exposure_score"
        ]
        .rank(
            pct=True
        )
    )

    result[
        "economic_priority_band"
    ] = (
        _research_qcut(
            values=(
                result[
                    "leakage_exposure_score"
                ]
            ),
            q=5,
            labels=(
                LEAKAGE_ECONOMIC_PRIORITY_LABELS
            ),
            name="Leakage economic-priority",
        )
    )

    result[
        "risk_band"
    ] = (
        _research_qcut(
            values=(
                result[
                    "leakage_probability"
                ]
            ),
            q=3,
            labels=(
                LEAKAGE_RISK_BAND_LABELS
            ),
            name="Leakage risk",
        )
    )

    result[
        "value_band"
    ] = (
        _research_qcut(
            values=(
                result[
                    "baseline_revenue"
                ]
            ),
            q=3,
            labels=(
                LEAKAGE_VALUE_BAND_LABELS
            ),
            name="Leakage value",
        )
    )

    return (
        result[
            [
                "customer",
                "leakage_probability",
                "leakage_probability_percentile",
                "baseline_revenue",
                "leakage_exposure_score",
                "leakage_exposure_percentile",
                "economic_priority_band",
                "risk_band",
                "value_band",
            ]
        ]
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# FCV ALIGNMENT
# ================================================================


def align_fcv_signals(
    fcv_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 05 FCV percentile ranking and operating tiers.
    """

    _validate_dataframe(
        fcv_scores,
        "FCV scores",
    )

    required_columns = (
        "customer",
        "fcv_history_180d_revenue",
        "future_active_probability",
        "future_customer_value_90d",
    )

    _validate_required_columns(
        fcv_scores,
        required_columns,
        "FCV scores",
    )

    result = (
        fcv_scores[
            list(
                required_columns
            )
        ]
        .copy()
    )

    _validate_unique_customer(
        result,
        "FCV scores",
    )

    _validate_probability(
        result[
            "future_active_probability"
        ],
        "future_active_probability",
    )

    _validate_non_negative(
        result[
            "fcv_history_180d_revenue"
        ],
        "fcv_history_180d_revenue",
    )

    _validate_non_negative(
        result[
            "future_customer_value_90d"
        ],
        "future_customer_value_90d",
    )

    result[
        "fcv_percentile"
    ] = (
        result[
            "future_customer_value_90d"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    result[
        "fcv_tier"
    ] = pd.cut(
        result[
            "fcv_percentile"
        ],
        bins=list(
            FCV_TIER_BINS
        ),
        labels=list(
            FCV_TIER_LABELS
        ),
        include_lowest=True,
    )

    return (
        result[
            [
                "customer",
                "fcv_history_180d_revenue",
                "future_active_probability",
                "future_customer_value_90d",
                "fcv_percentile",
                "fcv_tier",
            ]
        ]
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# REACTIVATION ALIGNMENT
# ================================================================


def align_reactivation_signals(
    reactivation_scores: pd.DataFrame,
    reactivation_features: pd.DataFrame,
    decision_date,
) -> pd.DataFrame:
    """
    Restore Reactivation prediction-date provenance after model scoring.

    ReactivationScorer intentionally returns only the customer key,
    probability and model metadata. The prediction timestamp comes from the
    point-in-time feature batch that was scored.
    """

    _validate_dataframe(
        reactivation_scores,
        "Reactivation scores",
    )

    _validate_dataframe(
        reactivation_features,
        "Reactivation features",
    )

    _validate_required_columns(
        reactivation_scores,
        (
            "customer",
            "reactivation_probability",
        ),
        "Reactivation scores",
    )

    _validate_required_columns(
        reactivation_features,
        (
            "customer",
            "prediction_date",
        ),
        "Reactivation features",
    )

    scores = (
        reactivation_scores[
            [
                "customer",
                "reactivation_probability",
            ]
        ]
        .copy()
    )

    features = (
        reactivation_features[
            [
                "customer",
                "prediction_date",
            ]
        ]
        .copy()
    )

    _validate_unique_customer(
        scores,
        "Reactivation scores",
    )

    _validate_unique_customer(
        features,
        "Reactivation features",
    )

    _validate_probability(
        scores[
            "reactivation_probability"
        ],
        "reactivation_probability",
    )

    features[
        "prediction_date"
    ] = pd.to_datetime(
        features[
            "prediction_date"
        ],
        errors="coerce",
    )

    if features[
        "prediction_date"
    ].isna().any():
        raise ValueError(
            "Reactivation features contain invalid prediction dates."
        )

    result = (
        scores
        .merge(
            features,
            on="customer",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "prediction_date":
                    "reactivation_prediction_date",
            }
        )
    )

    if result[
        "reactivation_prediction_date"
    ].isna().any():
        raise ValueError(
            "Reactivation feature provenance is missing for scored customers."
        )

    snapshot_date = (
        normalise_decision_date(
            decision_date
        )
    )

    result[
        "reactivation_score_age_days"
    ] = (
        (
            snapshot_date
            -
            result[
                "reactivation_prediction_date"
            ]
        )
        .dt.total_seconds()
        /
        86400
    )

    return (
        result
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# RECOMMENDATION ALIGNMENT
# ================================================================


def align_recommendation_signals(
    recommendation_scores: pd.DataFrame,
    decision_date,
) -> pd.DataFrame:
    """
    Select the latest Recommendation episode available by decision time.

    Production recommendation prediction_date equals the originating episode's
    order_date. Multiple historical episodes may exist for one customer.
    """

    _validate_dataframe(
        recommendation_scores,
        "Recommendation scores",
    )

    required_columns = (
        "customer",
        "invoice",
        "prediction_date",
        "repeat_top1_product",
        "discovery_top1_product",
    )

    _validate_required_columns(
        recommendation_scores,
        required_columns,
        "Recommendation scores",
    )

    result = (
        recommendation_scores[
            list(
                required_columns
            )
        ]
        .copy()
    )

    if result[
        "customer"
    ].isna().any():
        raise ValueError(
            "Recommendation scores contain missing customer IDs."
        )

    result[
        "prediction_date"
    ] = pd.to_datetime(
        result[
            "prediction_date"
        ],
        errors="coerce",
    )

    if result[
        "prediction_date"
    ].isna().any():
        raise ValueError(
            "Recommendation scores contain invalid prediction dates."
        )

    snapshot_date = (
        normalise_decision_date(
            decision_date
        )
    )

    result = (
        result.loc[
            result[
                "prediction_date"
            ]
            <=
            snapshot_date
        ]
        .copy()
    )

    if result.empty:

        return pd.DataFrame(
            columns=[
                "customer",
                "recommendation_prediction_date",
                "repeat_top1_product",
                "discovery_top1_product",
            ]
        )

    result = (
        result
        .sort_values(
            [
                "customer",
                "prediction_date",
                "invoice",
            ]
        )
        .drop_duplicates(
            subset="customer",
            keep="last",
        )
        .rename(
            columns={
                "prediction_date":
                    "recommendation_prediction_date",
            }
        )
    )

    return (
        result[
            [
                "customer",
                "recommendation_prediction_date",
                "repeat_top1_product",
                "discovery_top1_product",
            ]
        ]
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# COMMON SIGNAL ALIGNMENT
# ================================================================


def align_decision_signals(
    decision_snapshot: pd.DataFrame,
    *,
    leakage_scores: pd.DataFrame | None = None,
    leakage_context: pd.DataFrame | None = None,
    fcv_scores: pd.DataFrame | None = None,
    reactivation_scores: pd.DataFrame | None = None,
    reactivation_features: pd.DataFrame | None = None,
    recommendation_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach all available model signals to a Decision Engine snapshot.

    Missing scorer families remain missing rather than being fabricated.
    Availability and operational-validity flags explicitly record coverage.
    """

    _validate_dataframe(
        decision_snapshot,
        "Decision snapshot",
    )

    _validate_required_columns(
        decision_snapshot,
        (
            "customer",
            "decision_date",
            "last_order_date",
            "customer_state",
        ),
        "Decision snapshot",
    )

    _validate_unique_customer(
        decision_snapshot,
        "Decision snapshot",
    )

    result = (
        decision_snapshot.copy()
    )

    if result.empty:

        for column in DECISION_SIGNAL_COLUMNS:
            result[
                column
            ] = pd.Series(
                dtype="object"
            )

        return result

    decision_dates = pd.to_datetime(
        result[
            "decision_date"
        ],
        errors="coerce",
    )

    if decision_dates.isna().any():
        raise ValueError(
            "Decision snapshot contains invalid decision dates."
        )

    if decision_dates.nunique() != 1:
        raise ValueError(
            "Decision snapshot must contain exactly one decision date."
        )

    decision_date = (
        decision_dates.iloc[0]
    )

    # ------------------------------------------------------------
    # Leakage
    # ------------------------------------------------------------

    if leakage_scores is not None:

        if leakage_context is None:
            raise ValueError(
                "Leakage context is required when Leakage scores are supplied."
            )

        leakage = (
            align_leakage_signals(
                leakage_scores=leakage_scores,
                leakage_context=leakage_context,
            )
        )

        result = result.merge(
            leakage,
            on="customer",
            how="left",
            validate="one_to_one",
        )

    # ------------------------------------------------------------
    # FCV
    # ------------------------------------------------------------

    if fcv_scores is not None:

        fcv = (
            align_fcv_signals(
                fcv_scores
            )
        )

        result = result.merge(
            fcv,
            on="customer",
            how="left",
            validate="one_to_one",
        )

    # ------------------------------------------------------------
    # Reactivation
    # ------------------------------------------------------------

    if reactivation_scores is not None:

        if reactivation_features is None:
            raise ValueError(
                "Reactivation features are required when Reactivation "
                "scores are supplied."
            )

        reactivation = (
            align_reactivation_signals(
                reactivation_scores=(
                    reactivation_scores
                ),
                reactivation_features=(
                    reactivation_features
                ),
                decision_date=decision_date,
            )
        )

        result = result.merge(
            reactivation,
            on="customer",
            how="left",
            validate="one_to_one",
        )

    # ------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------

    if recommendation_scores is not None:

        recommendation = (
            align_recommendation_signals(
                recommendation_scores=(
                    recommendation_scores
                ),
                decision_date=decision_date,
            )
        )

        result = result.merge(
            recommendation,
            on="customer",
            how="left",
            validate="one_to_one",
        )

    # ------------------------------------------------------------
    # Guarantee signal columns
    # ------------------------------------------------------------

    for column in DECISION_SIGNAL_COLUMNS:

        if column not in result.columns:
            result[
                column
            ] = pd.NA

    # ------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------

    result[
        "leakage_signal_available"
    ] = (
        result[
            "leakage_probability"
        ]
        .notna()
    )

    result[
        "fcv_signal_available"
    ] = (
        result[
            "future_customer_value_90d"
        ]
        .notna()
    )

    result[
        "reactivation_signal_available"
    ] = (
        result[
            "reactivation_probability"
        ]
        .notna()
    )

    result[
        "recommendation_signal_available"
    ] = (
        result[
            [
                "repeat_top1_product",
                "discovery_top1_product",
            ]
        ]
        .notna()
        .any(
            axis=1
        )
    )

    # ------------------------------------------------------------
    # Recommendation currentness
    # ------------------------------------------------------------

    recommendation_dates = pd.to_datetime(
        result[
            "recommendation_prediction_date"
        ],
        errors="coerce",
    )

    last_order_dates = pd.to_datetime(
        result[
            "last_order_date"
        ],
        errors="coerce",
    )

    result[
        "recommendation_is_current"
    ] = (
        recommendation_dates.notna()
        &
        last_order_dates.notna()
        &
        (
            (
                recommendation_dates
                -
                last_order_dates
            )
            .abs()
            .dt.total_seconds()
            <=
            RECOMMENDATION_CURRENTNESS_SECONDS
        )
    )

    # ------------------------------------------------------------
    # Operational validity
    # ------------------------------------------------------------

    result[
        "leakage_operationally_valid"
    ] = (
        result[
            "leakage_signal_available"
        ]
        &
        result[
            "customer_state"
        ]
        .isin(
            [
                CUSTOMER_STATE_ACTIVE,
                CUSTOMER_STATE_AT_RISK,
            ]
        )
    )

    reactivation_age = pd.to_numeric(
        result[
            "reactivation_score_age_days"
        ],
        errors="coerce",
    )

    result[
        "reactivation_operationally_valid"
    ] = (
        result[
            "reactivation_signal_available"
        ]
        &
        reactivation_age.between(
            0,
            REACTIVATION_MAX_SIGNAL_AGE_DAYS,
            inclusive="both",
        )
        &
        result[
            "customer_state"
        ]
        .isin(
            [
                CUSTOMER_STATE_DORMANT,
                CUSTOMER_STATE_LONG_DORMANT,
            ]
        )
    )

    result[
        "recommendation_operationally_valid"
    ] = (
        result[
            "recommendation_signal_available"
        ]
        &
        result[
            "recommendation_is_current"
        ]
        &
        result[
            "customer_state"
        ]
        .isin(
            [
                CUSTOMER_STATE_ACTIVE,
                CUSTOMER_STATE_AT_RISK,
            ]
        )
    )

    # ------------------------------------------------------------
    # Final guarantees
    # ------------------------------------------------------------

    if not result[
        "customer"
    ].is_unique:
        raise RuntimeError(
            "Signal alignment produced duplicate customer rows."
        )

    if len(
        result
    ) != len(
        decision_snapshot
    ):
        raise RuntimeError(
            "Signal alignment changed the Decision Engine customer population."
        )

    return (
        result
        .sort_values(
            "customer"
        )
        .reset_index(
            drop=True
        )
    )