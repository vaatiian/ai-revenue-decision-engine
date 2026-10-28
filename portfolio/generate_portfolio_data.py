"""
Deterministic portfolio-data generator.

This module converts governed production outputs into compact,
public-safe datasets used by the Streamlit portfolio application.

Principles
----------
- production source files are read-only;
- customer, order, and product identifiers are anonymised;
- source schemas are locked and fail closed on drift;
- raw future-conditioned recommendation evaluation rows are never exported;
- governed model manifests are reduced to public-safe model cards;
- generated JSON is deterministic;
- original production outputs are never modified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


# ================================================================
# PROJECT PATHS
# ================================================================

CUSTOMER_DECISIONS_SOURCE = Path(
    "outputs/phase12_smoke/customer/"
    "customer-batch/2011-08-31/customer_decisions.csv"
)

REPEAT_PURCHASE_SOURCE = Path(
    "outputs/phase12_smoke/repeat/"
    "repeat-purchase/2011-08-31/repeat_purchase_scores.csv"
)

DEMAND_FORECAST_SOURCE = Path(
    "outputs/phase12_smoke/demand/"
    "demand/2011-11-21/demand_forecasts.csv"
)

DEFAULT_PORTFOLIO_DATA_DIR = Path(
    "portfolio/data"
)


# ================================================================
# VERIFIED PORTFOLIO EVIDENCE
# ================================================================

VERIFIED_REGRESSION_TEST_COUNT = 1633

VERIFIED_PRODUCTION_SCORERS_AVAILABLE = 6

VERIFIED_PRODUCTION_SCORERS_TOTAL = 6


# ================================================================
# SOURCE SCHEMA CONTRACTS
# ================================================================

CUSTOMER_DECISIONS_SOURCE_COLUMNS = (
    "customer",
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
    "recommended_product",
    "repeat_top1_product",
    "discovery_top1_product",
)

REPEAT_PURCHASE_SOURCE_COLUMNS = (
    "customer",
    "first_invoice",
    "first_order_date",
    "repeat_purchase_score",
    "model_name",
    "model_version",
    "model_type",
    "target_name",
    "governance_decision",
    "scored_at_utc",
)

DEMAND_FORECAST_SOURCE_COLUMNS = (
    "forecast_origin_week",
    "target_week",
    "horizon_weeks",
    "seasonal_analogue_week",
    "seasonal_analogue_revenue",
    "recent_four_week_average",
    "forecast_revenue",
    "seasonal_weight",
    "recent_weight",
    "model_name",
    "model_version",
    "model_type",
    "target_name",
    "governance_decision",
    "scored_at_utc",
)


# ================================================================
# PUBLIC OUTPUT SCHEMAS
# ================================================================

CUSTOMER_DECISIONS_PORTFOLIO_COLUMNS = (
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
)

REPEAT_PURCHASE_PORTFOLIO_COLUMNS = (
    "portfolio_customer_id",
    "portfolio_first_order_id",
    "first_order_date",
    "repeat_purchase_score",
    "model_name",
    "model_version",
    "model_type",
    "target_name",
    "governance_decision",
    "scored_at_utc",
)

DEMAND_FORECAST_PORTFOLIO_COLUMNS = (
    *DEMAND_FORECAST_SOURCE_COLUMNS,
)


# ================================================================
# GOVERNED MODEL MANIFEST REGISTER
# ================================================================

MODEL_MANIFESTS = {
    "reactivation": Path(
        "artifacts/models/reactivation/"
        "reactivation_full_random_forest_v1.0.0.json"
    ),
    "revenue_leakage": Path(
        "artifacts/models/leakage/"
        "revenue_leakage_full_gradient_boosting_v1.0.0.json"
    ),
    "future_customer_value": Path(
        "artifacts/models/fcv/"
        "fcv_stage_a_full_random_forest_v1.0.0.json"
    ),
    "repeat_purchase": Path(
        "artifacts/models/repeat_purchase/"
        "repeat_purchase_logistic_v1.0.0.json"
    ),
    "recommendation": Path(
        "artifacts/models/recommendation/"
        "recommendation_deterministic_ranker_v1.0.0.json"
    ),
    "demand_forecast": Path(
        "artifacts/models/demand/"
        "demand_weekly_seasonal_blend_v1.0.0.json"
    ),
}

MODEL_DISPLAY_NAMES = {
    "reactivation": "Reactivation Propensity",
    "revenue_leakage": "Revenue Leakage Risk",
    "future_customer_value": "Future Customer Value",
    "repeat_purchase": "Repeat Purchase Propensity",
    "recommendation": "Next Best Product",
    "demand_forecast": "Weekly Demand Forecast",
}


# ================================================================
# SAFE JSON
# ================================================================

def _json_safe(
    value: Any,
) -> Any:
    """
    Convert common pandas/numpy values into JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                _json_safe(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    if pd.isna(
        value
    ):
        return None

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except (
            ValueError,
            TypeError,
            AttributeError,
        ):
            pass

    return value


# ================================================================
# SOURCE VALIDATION
# ================================================================

def _require_file(
    path: Path,
) -> None:
    """
    Fail closed when a required governed source is missing.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Required portfolio source does not exist: {path}"
        )


def _validate_exact_columns(
    frame: pd.DataFrame,
    expected_columns: tuple[str, ...],
    source_name: str,
) -> None:
    """
    Require the exact governed source schema and order.
    """

    observed = tuple(
        frame.columns
    )

    if observed != expected_columns:
        raise RuntimeError(
            f"{source_name} schema drift detected. "
            f"Expected {expected_columns}, observed {observed}."
        )


# ================================================================
# IDENTIFIER NORMALISATION
# ================================================================

def _canonical_identifier(
    value: Any,
) -> str | None:
    """
    Convert an identifier into a stable canonical string.

    Integral numeric identifiers are normalised so 123.0 and 123 map
    to the same portfolio identifier.
    """

    if pd.isna(
        value
    ):
        return None

    if isinstance(
        value,
        bool,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        int,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        float,
    ):
        if math.isfinite(
            value
        ) and value.is_integer():
            return str(
                int(
                    value
                )
            )

        return format(
            value,
            ".15g",
        )

    text = str(
        value
    ).strip()

    if not text:
        return None

    return text


def _build_identifier_mapping(
    values,
    prefix: str,
    width: int,
) -> dict[str, str]:
    """
    Build a deterministic anonymisation map from sorted unique IDs.
    """

    canonical = {
        identifier
        for identifier in (
            _canonical_identifier(
                value
            )
            for value in values
        )
        if identifier is not None
    }

    ordered = sorted(
        canonical
    )

    mapping = {
        raw:
            f"{prefix}-{index:0{width}d}"
        for index, raw
        in enumerate(
            ordered,
            start=1,
        )
    }

    if len(
        set(
            mapping.values()
        )
    ) != len(
        mapping
    ):
        raise RuntimeError(
            f"Duplicate anonymised {prefix} identifiers generated."
        )

    return mapping


def _map_identifier_series(
    series: pd.Series,
    mapping: dict[str, str],
) -> pd.Series:
    """
    Apply a deterministic mapping while preserving missing values.
    """

    canonical = series.map(
        _canonical_identifier
    )

    mapped = canonical.map(
        mapping
    )

    unexpected = (
        canonical.notna()
        &
        mapped.isna()
    )

    if unexpected.any():
        raise RuntimeError(
            "An identifier could not be anonymised deterministically."
        )

    return mapped


# ================================================================
# DISPLAY NORMALISATION
# ================================================================

def _normalise_display_text(
    value: Any,
) -> Any:
    """
    Clean presentation-only text without changing production sources.
    """

    if pd.isna(
        value
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        return value

    cleaned = (
        value
        .replace(
            "▒",
            "—",
        )
        .replace(
            "\ufffd",
            "—",
        )
        .strip()
    )

    return cleaned


def _normalise_display_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply display normalisation to string/object columns.
    """

    result = frame.copy()

    for column in result.columns:

        if (
            pd.api.types.is_object_dtype(
                result[
                    column
                ]
            )
            or
            pd.api.types.is_string_dtype(
                result[
                    column
                ]
            )
        ):
            result[
                column
            ] = (
                result[
                    column
                ]
                .map(
                    _normalise_display_text
                )
            )

    return result


# ================================================================
# CUSTOMER DECISION PORTFOLIO DATA
# ================================================================

def build_customer_decisions_portfolio(
    source: pd.DataFrame,
    customer_mapping: dict[str, str],
    product_mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Build the public-safe customer decision dataset.
    """

    _validate_exact_columns(
        frame=source,
        expected_columns=(
            CUSTOMER_DECISIONS_SOURCE_COLUMNS
        ),
        source_name=(
            "Customer Decision source"
        ),
    )

    result = source.copy()

    result[
        "portfolio_customer_id"
    ] = _map_identifier_series(
        result[
            "customer"
        ],
        customer_mapping,
    )

    product_columns = {
        "recommended_product":
            "portfolio_recommended_product",
        "repeat_top1_product":
            "portfolio_repeat_top1_product",
        "discovery_top1_product":
            "portfolio_discovery_top1_product",
    }

    for (
        source_column,
        destination_column,
    ) in product_columns.items():

        result[
            destination_column
        ] = _map_identifier_series(
            result[
                source_column
            ],
            product_mapping,
        )

    result = result.drop(
        columns=[
            "customer",
            "recommended_product",
            "repeat_top1_product",
            "discovery_top1_product",
        ]
    )

    result = result[
        list(
            CUSTOMER_DECISIONS_PORTFOLIO_COLUMNS
        )
    ]

    result = _normalise_display_columns(
        result
    )

    if (
        result[
            "portfolio_customer_id"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Portfolio customer decision data contains "
            "missing anonymised customer identifiers."
        )

    if not result[
        "portfolio_customer_id"
    ].is_unique:
        raise RuntimeError(
            "Portfolio customer decision data contains "
            "duplicate customer identifiers."
        )

    forbidden = {
        "customer",
        "recommended_product",
        "repeat_top1_product",
        "discovery_top1_product",
    }

    if forbidden.intersection(
        result.columns
    ):
        raise RuntimeError(
            "Raw identifiers remain in customer portfolio output."
        )

    return (
        result
        .sort_values(
            "portfolio_customer_id"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# REPEAT PURCHASE PORTFOLIO DATA
# ================================================================

def build_repeat_purchase_portfolio(
    source: pd.DataFrame,
    customer_mapping: dict[str, str],
    order_mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Build the public-safe Repeat Purchase scoring dataset.
    """

    _validate_exact_columns(
        frame=source,
        expected_columns=(
            REPEAT_PURCHASE_SOURCE_COLUMNS
        ),
        source_name=(
            "Repeat Purchase source"
        ),
    )

    result = source.copy()

    result[
        "portfolio_customer_id"
    ] = _map_identifier_series(
        result[
            "customer"
        ],
        customer_mapping,
    )

    result[
        "portfolio_first_order_id"
    ] = _map_identifier_series(
        result[
            "first_invoice"
        ],
        order_mapping,
    )

    result = result.drop(
        columns=[
            "customer",
            "first_invoice",
        ]
    )

    result = result[
        list(
            REPEAT_PURCHASE_PORTFOLIO_COLUMNS
        )
    ]

    result = _normalise_display_columns(
        result
    )

    if (
        result[
            "portfolio_customer_id"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Repeat Purchase portfolio data contains "
            "missing anonymised customer identifiers."
        )

    if (
        result[
            "portfolio_first_order_id"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Repeat Purchase portfolio data contains "
            "missing anonymised order identifiers."
        )

    forbidden = {
        "customer",
        "first_invoice",
    }

    if forbidden.intersection(
        result.columns
    ):
        raise RuntimeError(
            "Raw identifiers remain in Repeat Purchase portfolio output."
        )

    return (
        result
        .sort_values(
            [
                "portfolio_customer_id",
                "portfolio_first_order_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# DEMAND PORTFOLIO DATA
# ================================================================

def build_demand_forecasts_portfolio(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the public aggregate demand-forecast dataset.
    """

    _validate_exact_columns(
        frame=source,
        expected_columns=(
            DEMAND_FORECAST_SOURCE_COLUMNS
        ),
        source_name=(
            "Demand Forecast source"
        ),
    )

    result = source.copy()

    result = result[
        list(
            DEMAND_FORECAST_PORTFOLIO_COLUMNS
        )
    ]

    result = _normalise_display_columns(
        result
    )

    if result[
        "horizon_weeks"
    ].duplicated().any():
        raise RuntimeError(
            "Demand portfolio data contains duplicate horizons."
        )

    return (
        result
        .sort_values(
            "horizon_weeks"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# MODEL MANIFEST EXTRACTION
# ================================================================

def _load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load one governed JSON manifest.
    """

    _require_file(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(
            handle
        )

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"Model manifest must contain a JSON object: {path}"
        )

    return value


def _extract_governance(
    system_id: str,
    manifest: dict[str, Any],
) -> Any:
    """
    Normalize heterogeneous manifest governance structures.
    """

    if system_id == "recommendation":

        repeat = (
            manifest
            .get(
                "repeat_channel",
                {},
            )
            .get(
                "research_decision"
            )
        )

        discovery = (
            manifest
            .get(
                "discovery_channel",
                {},
            )
            .get(
                "research_decision"
            )
        )

        return {
            "repeat_channel":
                repeat,
            "discovery_channel":
                discovery,
        }

    if "research_final_decision" in manifest:
        return manifest[
            "research_final_decision"
        ]

    if "governance_decision" in manifest:
        return manifest[
            "governance_decision"
        ]

    governance = manifest.get(
        "governance"
    )

    if isinstance(
        governance,
        dict,
    ):
        return governance.get(
            "decision"
        )

    return None


def _extract_metrics(
    system_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract only governed, public-safe evaluation metrics.
    """

    if system_id == "future_customer_value":
        return dict(
            manifest.get(
                "research_test_metrics",
                {},
            )
        )

    if system_id == "repeat_purchase":
        return dict(
            manifest.get(
                "reported_final_test_metrics",
                {},
            )
        )

    if system_id == "recommendation":
        return dict(
            manifest.get(
                "final_holdout_performance",
                {},
            )
        )

    if system_id == "demand_forecast":
        return dict(
            manifest.get(
                "reported_historical_performance",
                {},
            )
        )

    return {}


def _extract_prediction_horizon(
    system_id: str,
    manifest: dict[str, Any],
) -> Any:
    """
    Normalize model horizon metadata for portfolio presentation.
    """

    if system_id == "reactivation":
        return manifest.get(
            "target_horizon_days"
        )

    if system_id == "revenue_leakage":
        return manifest.get(
            "outcome_window_days"
        )

    if system_id == "future_customer_value":
        return manifest.get(
            "outcome_window_days"
        )

    if system_id == "repeat_purchase":
        return manifest.get(
            "prediction_horizon_days"
        )

    if system_id == "recommendation":
        return manifest.get(
            "outcome_window_days"
        )

    if system_id == "demand_forecast":
        return manifest.get(
            "forecast_horizons_weeks"
        )

    return None


def _extract_validation_method(
    system_id: str,
    manifest: dict[str, Any],
) -> Any:
    """
    Normalize validation metadata.
    """

    if "research_validation_design" in manifest:
        return manifest[
            "research_validation_design"
        ]

    if system_id == "demand_forecast":
        validation = manifest.get(
            "validation",
            {},
        )

        if isinstance(
            validation,
            dict,
        ):
            return validation.get(
                "method"
            )

    if system_id == "recommendation":
        return (
            "Temporal holdout with frozen post-holdout governance"
        )

    return None


def build_model_registry(
    project_root: Path,
) -> dict[str, Any]:
    """
    Build public-safe model cards from governed manifests.

    Hashes and binary artifact paths are deliberately omitted.
    """

    systems = []

    for (
        system_id,
        relative_path,
    ) in MODEL_MANIFESTS.items():

        manifest = _load_json(
            project_root
            /
            relative_path
        )

        system = {
            "system_id":
                system_id,

            "display_name":
                MODEL_DISPLAY_NAMES[
                    system_id
                ],

            "model_name":
                manifest.get(
                    "model_name"
                ),

            "model_version":
                manifest.get(
                    "model_version"
                ),

            "model_type":
                manifest.get(
                    "model_type"
                ),

            "target_name":
                manifest.get(
                    "target_name"
                ),

            "feature_count":
                manifest.get(
                    "feature_count"
                ),

            "algorithm":
                manifest.get(
                    "algorithm"
                ),

            "prediction_horizon":
                _extract_prediction_horizon(
                    system_id,
                    manifest,
                ),

            "validation_method":
                _extract_validation_method(
                    system_id,
                    manifest,
                ),

            "governance":
                _extract_governance(
                    system_id,
                    manifest,
                ),

            "metrics":
                _extract_metrics(
                    system_id,
                    manifest,
                ),
        }

        if system_id == "future_customer_value":

            system[
                "architecture"
            ] = {
                "stage_a":
                    manifest.get(
                        "model_name"
                    ),

                "stage_b":
                    manifest.get(
                        "stage_b_model_name"
                    ),

                "formula":
                    manifest.get(
                        "complete_model_formula"
                    ),
            }

            system[
                "approved_use"
            ] = manifest.get(
                "approved_use"
            )

            system[
                "limitations"
            ] = [
                manifest.get(
                    "primary_limitation"
                )
            ]

        elif system_id == "recommendation":

            system[
                "architecture"
            ] = manifest.get(
                "architecture"
            )

            system[
                "approved_use"
            ] = {
                "repeat_channel":
                    (
                        manifest
                        .get(
                            "repeat_channel",
                            {},
                        )
                        .get(
                            "approved_use"
                        )
                    ),

                "discovery_channel":
                    (
                        manifest
                        .get(
                            "discovery_channel",
                            {},
                        )
                        .get(
                            "approved_use"
                        )
                    ),
            }

            system[
                "limitations"
            ] = [
                (
                    manifest
                    .get(
                        "discovery_channel",
                        {},
                    )
                    .get(
                        "prohibited_interpretation"
                    )
                )
            ]

        elif system_id == "repeat_purchase":

            system[
                "approved_use"
            ] = manifest.get(
                "recommended_use"
            )

            system[
                "limitations"
            ] = manifest.get(
                "prohibited_interpretations"
            )

        elif system_id == "demand_forecast":

            system[
                "architecture"
            ] = manifest.get(
                "architecture"
            )

            system[
                "approved_use"
            ] = (
                manifest
                .get(
                    "governance",
                    {},
                )
                .get(
                    "decision_description"
                )
            )

            system[
                "limitations"
            ] = [
                (
                    manifest
                    .get(
                        "planning_ranges",
                        {},
                    )
                    .get(
                        "interpretation"
                    )
                )
            ]

        else:

            system[
                "approved_use"
            ] = None

            system[
                "limitations"
            ] = []

        systems.append(
            _json_safe(
                system
            )
        )

    if len(
        systems
    ) != 6:
        raise RuntimeError(
            "Portfolio model registry must contain exactly "
            "six production intelligence systems."
        )

    return {
        "schema_version":
            "1.0.0",

        "system_count":
            len(
                systems
            ),

        "systems":
            systems,
    }


# ================================================================
# SUMMARY
# ================================================================

def build_portfolio_summary(
    customer_decisions: pd.DataFrame,
    repeat_purchase: pd.DataFrame,
    demand_forecasts: pd.DataFrame,
    model_registry: dict[str, Any],
) -> dict[str, Any]:
    """
    Build compact verified proof points for application startup.
    """

    if customer_decisions.empty:
        raise RuntimeError(
            "Customer portfolio dataset must not be empty."
        )

    if repeat_purchase.empty:
        raise RuntimeError(
            "Repeat Purchase portfolio dataset must not be empty."
        )

    if demand_forecasts.empty:
        raise RuntimeError(
            "Demand portfolio dataset must not be empty."
        )

    action_counts = (
        customer_decisions[
            "primary_action"
        ]
        .fillna(
            "Not Available"
        )
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_dict()
    )

    state_counts = (
        customer_decisions[
            "customer_state"
        ]
        .fillna(
            "Not Available"
        )
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_dict()
    )

    priority_counts = (
        customer_decisions[
            "priority_tier"
        ]
        .fillna(
            "Not Available"
        )
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_dict()
    )

    horizon_values = (
        demand_forecasts[
            "horizon_weeks"
        ]
        .astype(int)
        .sort_values()
        .tolist()
    )

    summary = {
        "schema_version":
            "1.0.0",

        "verified_evidence": {
            "production_intelligence_systems":
                int(
                    model_registry[
                        "system_count"
                    ]
                ),

            "automated_tests_passing":
                VERIFIED_REGRESSION_TEST_COUNT,

            "customer_decisions_generated":
                int(
                    len(
                        customer_decisions
                    )
                ),

            "repeat_purchase_scores_generated":
                int(
                    len(
                        repeat_purchase
                    )
                ),

            "governed_demand_horizons":
                horizon_values,

            "production_scoring_systems_available":
                (
                    VERIFIED_PRODUCTION_SCORERS_AVAILABLE
                ),

            "production_scoring_systems_total":
                (
                    VERIFIED_PRODUCTION_SCORERS_TOTAL
                ),
        },

        "decision_snapshot": {
            "decision_date":
                str(
                    customer_decisions[
                        "decision_date"
                    ].iloc[
                        0
                    ]
                ),

            "customer_state_counts":
                state_counts,

            "primary_action_counts":
                action_counts,

            "priority_tier_counts":
                priority_counts,
        },

        "demand_snapshot": {
            "forecast_origin_week":
                str(
                    demand_forecasts[
                        "forecast_origin_week"
                    ].iloc[
                        0
                    ]
                ),

            "horizons_weeks":
                horizon_values,

            "forecast_rows":
                int(
                    len(
                        demand_forecasts
                    )
                ),
        },

        "claims_note": (
            "Portfolio counts represent verified system and "
            "Docker smoke-test evidence, not realised business impact."
        ),
    }

    return _json_safe(
        summary
    )


# ================================================================
# DETERMINISTIC WRITERS
# ================================================================

def _write_json(
    value: dict[str, Any],
    path: Path,
) -> None:
    """
    Write deterministic formatted JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            _json_safe(
                value
            ),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

        handle.write(
            "\n"
        )


# ================================================================
# END-TO-END GENERATOR
# ================================================================

def generate_portfolio_data(
    project_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Generate all public-safe portfolio datasets.

    Parameters
    ----------
    project_root:
        Repository root containing governed source outputs.

    output_dir:
        Optional portfolio-data destination. Defaults to
        <project_root>/portfolio/data.

    Returns
    -------
    dict[str, Path]
        Paths to generated portfolio artifacts.
    """

    root = Path(
        project_root
    ).resolve()

    destination = (
        Path(
            output_dir
        ).resolve()
        if output_dir is not None
        else
        (
            root
            /
            DEFAULT_PORTFOLIO_DATA_DIR
        )
    )

    customer_source_path = (
        root
        /
        CUSTOMER_DECISIONS_SOURCE
    )

    repeat_source_path = (
        root
        /
        REPEAT_PURCHASE_SOURCE
    )

    demand_source_path = (
        root
        /
        DEMAND_FORECAST_SOURCE
    )

    for source_path in (
        customer_source_path,
        repeat_source_path,
        demand_source_path,
    ):
        _require_file(
            source_path
        )

    customer_source = pd.read_csv(
        customer_source_path
    )

    repeat_source = pd.read_csv(
        repeat_source_path
    )

    demand_source = pd.read_csv(
        demand_source_path
    )

    _validate_exact_columns(
        customer_source,
        CUSTOMER_DECISIONS_SOURCE_COLUMNS,
        "Customer Decision source",
    )

    _validate_exact_columns(
        repeat_source,
        REPEAT_PURCHASE_SOURCE_COLUMNS,
        "Repeat Purchase source",
    )

    _validate_exact_columns(
        demand_source,
        DEMAND_FORECAST_SOURCE_COLUMNS,
        "Demand Forecast source",
    )

    # ------------------------------------------------------------
    # Shared customer mapping
    #
    # One mapping is created across all public customer-level files
    # so the same source customer always receives the same portfolio ID.
    # ------------------------------------------------------------

    all_customer_values = pd.concat(
        [
            customer_source[
                "customer"
            ],
            repeat_source[
                "customer"
            ],
        ],
        ignore_index=True,
    )

    customer_mapping = (
        _build_identifier_mapping(
            values=all_customer_values,
            prefix="CUST",
            width=5,
        )
    )

    order_mapping = (
        _build_identifier_mapping(
            values=repeat_source[
                "first_invoice"
            ],
            prefix="ORD",
            width=6,
        )
    )

    product_values = pd.concat(
        [
            customer_source[
                "recommended_product"
            ],
            customer_source[
                "repeat_top1_product"
            ],
            customer_source[
                "discovery_top1_product"
            ],
        ],
        ignore_index=True,
    )

    product_mapping = (
        _build_identifier_mapping(
            values=product_values,
            prefix="PROD",
            width=5,
        )
    )

    customer_portfolio = (
        build_customer_decisions_portfolio(
            source=customer_source,
            customer_mapping=customer_mapping,
            product_mapping=product_mapping,
        )
    )

    repeat_portfolio = (
        build_repeat_purchase_portfolio(
            source=repeat_source,
            customer_mapping=customer_mapping,
            order_mapping=order_mapping,
        )
    )

    demand_portfolio = (
        build_demand_forecasts_portfolio(
            source=demand_source,
        )
    )

    model_registry = build_model_registry(
        root
    )

    portfolio_summary = (
        build_portfolio_summary(
            customer_decisions=(
                customer_portfolio
            ),
            repeat_purchase=(
                repeat_portfolio
            ),
            demand_forecasts=(
                demand_portfolio
            ),
            model_registry=(
                model_registry
            ),
        )
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    customer_output = (
        destination
        /
        "customer_decisions_portfolio.parquet"
    )

    repeat_output = (
        destination
        /
        "repeat_purchase_portfolio.parquet"
    )

    demand_output = (
        destination
        /
        "demand_forecasts_portfolio.parquet"
    )

    registry_output = (
        destination
        /
        "model_registry_portfolio.json"
    )

    summary_output = (
        destination
        /
        "portfolio_summary.json"
    )

    customer_portfolio.to_parquet(
        customer_output,
        index=False,
    )

    repeat_portfolio.to_parquet(
        repeat_output,
        index=False,
    )

    demand_portfolio.to_parquet(
        demand_output,
        index=False,
    )

    _write_json(
        model_registry,
        registry_output,
    )

    _write_json(
        portfolio_summary,
        summary_output,
    )

    return {
        "customer_decisions":
            customer_output,

        "repeat_purchase":
            repeat_output,

        "demand_forecasts":
            demand_output,

        "model_registry":
            registry_output,

        "portfolio_summary":
            summary_output,
    }


# ================================================================
# LOCAL CLI
# ================================================================

def main() -> int:
    """
    Generate portfolio data from the repository root.
    """

    project_root = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            1
        ]
    )

    outputs = generate_portfolio_data(
        project_root=project_root
    )

    print(
        "SUCCEEDED: generated portfolio data"
    )

    for (
        name,
        path,
    ) in outputs.items():
        print(
            f"{name}: {path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )