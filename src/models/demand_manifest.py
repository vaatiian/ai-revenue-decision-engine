"""
Governance manifest utilities for the frozen Notebook 07
business-level weekly demand forecasting architecture.

The final production forecaster is deterministic.

There is no fitted sklearn model artifact.

Governance therefore binds the production specification to:

- the frozen model metadata;
- the exact deterministic forecast parameters;
- the supported horizons;
- the validation and governance decision;
- the research notebook SHA-256;
- the production specification source SHA-256;
- the weekly model-feature source SHA-256;
- the production scorer source SHA-256.

Any governed-source change requires deliberate manifest regeneration and
review.
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any

from src.features.demand_model_features import (
    DEMAND_WEEK_FREQUENCY,
    DEMAND_ZERO_TRADING_WEEK_PERIODS,
)

from src.models.demand_spec import (
    DEMAND_FINAL_DECISION,
    DEMAND_FINAL_DECISION_DESCRIPTION,
    DEMAND_FINAL_HOLDOUT_STATUS,
    DEMAND_FINAL_MODEL_NAME,
    DEMAND_FORECAST_FREQUENCY,
    DEMAND_FORECAST_HORIZONS,
    DEMAND_MODEL_METADATA,
    DEMAND_PLANNING_INTERVAL_METHOD,
    DEMAND_PRIMARY_EVALUATION_METRIC,
    DEMAND_RECENT_WEIGHT,
    DEMAND_RECENT_WINDOW_WEEKS,
    DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_FORECAST_ACTUAL,
    DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_WAPE,
    DEMAND_REPORTED_APPROX_HORIZON_FORECAST_ACTUAL,
    DEMAND_REPORTED_APPROX_HORIZON_WAPE,
    DEMAND_SEASONAL_LAG_WEEKS,
    DEMAND_SEASONAL_WEIGHT,
    DEMAND_SERIALIZED_MODEL_REQUIRED,
    DEMAND_TARGET_DESCRIPTION,
    DEMAND_VALIDATION_METHOD,
    DEMAND_WEIGHT_TUNING_CLOSED,
    validate_demand_specification,
)


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        2
    ]
)

DEMAND_RESEARCH_NOTEBOOK_PATH = (
    PROJECT_ROOT
    /
    "notebooks"
    /
    "07_ML_Demand_Forecasting.ipynb"
)

DEMAND_SPEC_SOURCE_PATH = (
    PROJECT_ROOT
    /
    "src"
    /
    "models"
    /
    "demand_spec.py"
)

DEMAND_FEATURE_SOURCE_PATH = (
    PROJECT_ROOT
    /
    "src"
    /
    "features"
    /
    "demand_model_features.py"
)

DEMAND_SCORER_SOURCE_PATH = (
    PROJECT_ROOT
    /
    "src"
    /
    "models"
    /
    "demand_scorer.py"
)

DEMAND_MANIFEST_DIRECTORY = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "models"
    /
    "demand"
)

DEMAND_MANIFEST_FILENAME = (
    "demand_weekly_seasonal_blend_v1.0.0.json"
)

DEMAND_MANIFEST_PATH = (
    DEMAND_MANIFEST_DIRECTORY
    /
    DEMAND_MANIFEST_FILENAME
)


# ================================================================
# HASHING
# ================================================================

def sha256_file(
    path: str | Path,
) -> str:
    """
    Calculate SHA-256 for one file.
    """

    source = Path(
        path
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"Governed demand source not found: {source}"
        )

    digest = hashlib.sha256()

    with source.open(
        "rb"
    ) as handle:

        for block in iter(
            lambda:
                handle.read(
                    1024
                    *
                    1024
                ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


# ================================================================
# MANIFEST CONSTRUCTION
# ================================================================

def build_demand_manifest() -> dict[str, Any]:
    """
    Build the complete frozen demand-model governance manifest.
    """

    validate_demand_specification()

    governed_paths = {
        "research_notebook":
            DEMAND_RESEARCH_NOTEBOOK_PATH,

        "production_spec":
            DEMAND_SPEC_SOURCE_PATH,

        "weekly_feature_engine":
            DEMAND_FEATURE_SOURCE_PATH,

        "production_scorer":
            DEMAND_SCORER_SOURCE_PATH,
    }

    hashes = {
        name:
            sha256_file(
                path
            )
        for (
            name,
            path,
        )
        in governed_paths.items()
    }

    manifest = {
        "model_name":
            DEMAND_MODEL_METADATA.model_name,

        "model_version":
            DEMAND_MODEL_METADATA.model_version,

        "model_type":
            DEMAND_MODEL_METADATA.model_type,

        "target_name":
            DEMAND_MODEL_METADATA.target_name,

        "target_description":
            DEMAND_TARGET_DESCRIPTION,

        "forecast_frequency":
            DEMAND_FORECAST_FREQUENCY,

        "week_frequency":
            DEMAND_WEEK_FREQUENCY,

        "final_research_model_name":
            DEMAND_FINAL_MODEL_NAME,

        "serialized_model_required":
            DEMAND_SERIALIZED_MODEL_REQUIRED,

        "forecast_horizons_weeks":
            list(
                DEMAND_FORECAST_HORIZONS
            ),

        "architecture": {
            "seasonal_weight":
                DEMAND_SEASONAL_WEIGHT,

            "recent_weight":
                DEMAND_RECENT_WEIGHT,

            "seasonal_lag_weeks":
                DEMAND_SEASONAL_LAG_WEEKS,

            "recent_window_weeks":
                DEMAND_RECENT_WINDOW_WEEKS,

            "formula":
                (
                    "0.90 * seasonal_52_week_analogue "
                    "+ 0.10 * trailing_4_week_average_at_origin"
                ),
        },

        "calendar_governance": {
            "week_frequency":
                DEMAND_WEEK_FREQUENCY,

            "zero_trading_week_periods":
                list(
                    DEMAND_ZERO_TRADING_WEEK_PERIODS
                ),

            "unknown_interior_missing_weeks":
                "fail_closed",

            "partial_start_week":
                "excluded",

            "partial_end_week":
                "excluded",
        },

        "validation": {
            "method":
                DEMAND_VALIDATION_METHOD,

            "primary_metric":
                DEMAND_PRIMARY_EVALUATION_METRIC,

            "final_holdout_status":
                DEMAND_FINAL_HOLDOUT_STATUS,

            "weight_tuning_closed":
                DEMAND_WEIGHT_TUNING_CLOSED,
        },

        "reported_historical_performance": {
            "approx_horizon_wape":
                {
                    str(
                        horizon
                    ):
                        value
                    for (
                        horizon,
                        value,
                    )
                    in (
                        DEMAND_REPORTED_APPROX_HORIZON_WAPE.items()
                    )
                },

            "approx_horizon_forecast_actual":
                {
                    str(
                        horizon
                    ):
                        value
                    for (
                        horizon,
                        value,
                    )
                    in (
                        DEMAND_REPORTED_APPROX_HORIZON_FORECAST_ACTUAL.items()
                    )
                },

            "approx_four_week_aggregate_wape":
                (
                    DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_WAPE
                ),

            "approx_four_week_aggregate_forecast_actual":
                (
                    DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_FORECAST_ACTUAL
                ),
        },

        "planning_ranges": {
            "method":
                DEMAND_PLANNING_INTERVAL_METHOD,

            "interpretation":
                (
                    "Historical planning ranges; not independently "
                    "calibrated guaranteed future prediction intervals"
                ),
        },

        "governance": {
            "decision":
                DEMAND_FINAL_DECISION,

            "decision_description":
                DEMAND_FINAL_DECISION_DESCRIPTION,

            "architecture_frozen":
                True,

            "weight_tuning_closed":
                DEMAND_WEIGHT_TUNING_CLOSED,

            "post_freeze_tuning_permitted":
                False,
        },

        "source_files": {
            "research_notebook":
                str(
                    DEMAND_RESEARCH_NOTEBOOK_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "production_spec":
                str(
                    DEMAND_SPEC_SOURCE_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "weekly_feature_engine":
                str(
                    DEMAND_FEATURE_SOURCE_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "production_scorer":
                str(
                    DEMAND_SCORER_SOURCE_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
        },

        "source_sha256": hashes,
    }

    return manifest


# ================================================================
# MANIFEST IO
# ================================================================

def write_demand_manifest(
    path: str | Path = DEMAND_MANIFEST_PATH,
) -> Path:
    """
    Build and write the frozen demand governance manifest.
    """

    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = build_demand_manifest()

    with destination.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    return destination


def load_demand_manifest(
    path: str | Path = DEMAND_MANIFEST_PATH,
) -> dict[str, Any]:
    """
    Load a demand governance manifest.
    """

    source = Path(
        path
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"Demand manifest not found: {source}"
        )

    with source.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if not isinstance(
        manifest,
        dict,
    ):
        raise ValueError(
            "Demand manifest must contain a JSON object."
        )

    return manifest


# ================================================================
# MANIFEST VALIDATION
# ================================================================

def validate_demand_manifest(
    manifest: dict[str, Any],
    verify_source_hashes: bool = True,
) -> None:
    """
    Validate a demand governance manifest against the frozen
    production specification.

    Source hashes are verified by default.
    """

    validate_demand_specification()

    if (
        manifest.get(
            "model_name"
        )
        !=
        DEMAND_MODEL_METADATA.model_name
    ):
        raise ValueError(
            "Demand manifest model_name does not match the frozen "
            "production specification."
        )

    if (
        manifest.get(
            "model_version"
        )
        !=
        DEMAND_MODEL_METADATA.model_version
    ):
        raise ValueError(
            "Demand manifest model_version does not match the frozen "
            "production specification."
        )

    if (
        manifest.get(
            "model_type"
        )
        !=
        DEMAND_MODEL_METADATA.model_type
    ):
        raise ValueError(
            "Demand manifest model_type does not match the frozen "
            "production specification."
        )

    if (
        manifest.get(
            "target_name"
        )
        !=
        DEMAND_MODEL_METADATA.target_name
    ):
        raise ValueError(
            "Demand manifest target_name does not match the frozen "
            "production specification."
        )

    if (
        manifest.get(
            "serialized_model_required"
        )
        is not False
    ):
        raise ValueError(
            "Frozen demand forecaster must not require a serialized "
            "model artifact."
        )

    if (
        tuple(
            manifest.get(
                "forecast_horizons_weeks",
                [],
            )
        )
        !=
        DEMAND_FORECAST_HORIZONS
    ):
        raise ValueError(
            "Demand manifest forecast horizons do not match the "
            "frozen production specification."
        )

    architecture = manifest.get(
        "architecture"
    )

    if not isinstance(
        architecture,
        dict,
    ):
        raise ValueError(
            "Demand manifest architecture section is missing."
        )

    expected_architecture = {
        "seasonal_weight":
            DEMAND_SEASONAL_WEIGHT,

        "recent_weight":
            DEMAND_RECENT_WEIGHT,

        "seasonal_lag_weeks":
            DEMAND_SEASONAL_LAG_WEEKS,

        "recent_window_weeks":
            DEMAND_RECENT_WINDOW_WEEKS,
    }

    for (
        field,
        expected,
    ) in expected_architecture.items():

        if (
            architecture.get(
                field
            )
            !=
            expected
        ):
            raise ValueError(
                "Demand manifest architecture mismatch: "
                f"{field} expected {expected!r}, "
                f"found {architecture.get(field)!r}."
            )

    calendar = manifest.get(
        "calendar_governance"
    )

    if not isinstance(
        calendar,
        dict,
    ):
        raise ValueError(
            "Demand manifest calendar_governance section is missing."
        )

    if (
        calendar.get(
            "week_frequency"
        )
        !=
        DEMAND_WEEK_FREQUENCY
    ):
        raise ValueError(
            "Demand manifest weekly calendar frequency mismatch."
        )

    if (
        tuple(
            calendar.get(
                "zero_trading_week_periods",
                [],
            )
        )
        !=
        DEMAND_ZERO_TRADING_WEEK_PERIODS
    ):
        raise ValueError(
            "Demand manifest zero-trading week register mismatch."
        )

    validation = manifest.get(
        "validation"
    )

    if not isinstance(
        validation,
        dict,
    ):
        raise ValueError(
            "Demand manifest validation section is missing."
        )

    if (
        validation.get(
            "primary_metric"
        )
        !=
        DEMAND_PRIMARY_EVALUATION_METRIC
    ):
        raise ValueError(
            "Demand manifest primary evaluation metric mismatch."
        )

    if (
        validation.get(
            "weight_tuning_closed"
        )
        is not True
    ):
        raise ValueError(
            "Demand manifest must record closed weight tuning."
        )

    governance = manifest.get(
        "governance"
    )

    if not isinstance(
        governance,
        dict,
    ):
        raise ValueError(
            "Demand manifest governance section is missing."
        )

    if (
        governance.get(
            "decision"
        )
        !=
        DEMAND_FINAL_DECISION
    ):
        raise ValueError(
            "Demand manifest governance decision mismatch."
        )

    if (
        governance.get(
            "architecture_frozen"
        )
        is not True
    ):
        raise ValueError(
            "Demand manifest must record a frozen architecture."
        )

    if (
        governance.get(
            "post_freeze_tuning_permitted"
        )
        is not False
    ):
        raise ValueError(
            "Demand manifest must prohibit post-freeze tuning on the "
            "existing research history."
        )

    if not verify_source_hashes:
        return

    source_hashes = manifest.get(
        "source_sha256"
    )

    if not isinstance(
        source_hashes,
        dict,
    ):
        raise ValueError(
            "Demand manifest source_sha256 section is missing."
        )

    governed_paths = {
        "research_notebook":
            DEMAND_RESEARCH_NOTEBOOK_PATH,

        "production_spec":
            DEMAND_SPEC_SOURCE_PATH,

        "weekly_feature_engine":
            DEMAND_FEATURE_SOURCE_PATH,

        "production_scorer":
            DEMAND_SCORER_SOURCE_PATH,
    }

    for (
        source_name,
        source_path,
    ) in governed_paths.items():

        expected_hash = source_hashes.get(
            source_name
        )

        if not expected_hash:
            raise ValueError(
                "Demand manifest is missing SHA-256 for "
                f"{source_name}."
            )

        actual_hash = sha256_file(
            source_path
        )

        if actual_hash != expected_hash:
            raise ValueError(
                "Demand governed source SHA-256 mismatch for "
                f"{source_name}. "
                f"Expected {expected_hash}, found {actual_hash}."
            )


def validate_demand_manifest_file(
    path: str | Path = DEMAND_MANIFEST_PATH,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    """
    Load and validate a manifest file, returning the validated object.
    """

    manifest = load_demand_manifest(
        path
    )

    validate_demand_manifest(
        manifest,
        verify_source_hashes=verify_source_hashes,
    )

    return manifest