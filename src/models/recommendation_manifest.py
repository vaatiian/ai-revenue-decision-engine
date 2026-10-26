"""
Governance manifest support for the frozen Next Best Product
recommendation architecture.

Unlike the fitted sklearn production models, the governed recommendation
model is deterministic and has no serialized .joblib artifact.

Governance therefore freezes:
- model metadata;
- channel architecture;
- ranking formulas;
- support thresholds;
- temporal rules;
- recommendation depths;
- optional hybrid allocation;
- final research decisions;
- final holdout performance;
- SHA-256 hashes of the governed production source files.

The source hashes provide the deterministic equivalent of model-artifact
hash verification. Any governed source-code change can therefore be
detected before production scoring is approved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.models.recommendation_spec import (
    RECOMMENDATION_AFFINITY_FORMULA,
    RECOMMENDATION_AFFINITY_SIMILARITY_NAME,
    RECOMMENDATION_DEFAULT_TOP_N,
    RECOMMENDATION_DISCOVERY_APPROVED_USE,
    RECOMMENDATION_DISCOVERY_DECISION,
    RECOMMENDATION_DISCOVERY_EXCLUDE_HISTORY,
    RECOMMENDATION_DISCOVERY_FALLBACK,
    RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10,
    RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_DISCOVERY_MODEL_NAME,
    RECOMMENDATION_DISCOVERY_PROHIBITED_INTERPRETATION,
    RECOMMENDATION_EVALUATION_DEPTHS,
    RECOMMENDATION_FUTURE_INFORMATION_ALLOWED,
    RECOMMENDATION_GLOBAL_HISTORY_RULE,
    RECOMMENDATION_HOLDOUT_CONSUMED,
    RECOMMENDATION_HYBRID_DISCOVERY_FRACTION,
    RECOMMENDATION_HYBRID_FILL_ORDER,
    RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_HYBRID_REPEAT_FRACTION,
    RECOMMENDATION_HYBRID_ROLE,
    RECOMMENDATION_MINIMUM_ITEM_ORDERS,
    RECOMMENDATION_MINIMUM_PAIR_ORDERS,
    RECOMMENDATION_MODEL_METADATA,
    RECOMMENDATION_OPTIONAL_HYBRID_NAME,
    RECOMMENDATION_OUTCOME_WINDOW_DAYS,
    RECOMMENDATION_POST_HOLDOUT_TUNING_PERMITTED,
    RECOMMENDATION_PRIMARY_DISCOVERY_METRIC,
    RECOMMENDATION_PRIMARY_GENERAL_METRICS,
    RECOMMENDATION_REPEAT_APPROVED_USE,
    RECOMMENDATION_REPEAT_DECISION,
    RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20,
    RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10,
    RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_REPEAT_MODEL_NAME,
    RECOMMENDATION_REPEAT_PRIMARY_SORT,
    RECOMMENDATION_REPEAT_PRIMARY_SORT_ASCENDING,
    RECOMMENDATION_REPEAT_TIE_BREAK,
    RECOMMENDATION_REPEAT_TIE_BREAK_ASCENDING,
    RECOMMENDATION_SOURCE_WEIGHT_FORMULA,
    RECOMMENDATION_TARGET,
    RECOMMENDATION_TEST_END,
    RECOMMENDATION_TEST_START,
    RECOMMENDATION_VALIDATION_END,
    RECOMMENDATION_VALIDATION_START,
    RECOMMENDATION_CUSTOMER_HISTORY_RULE,
)


# ================================================================
# PROJECT / MANIFEST PATH
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_RECOMMENDATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "recommendation"
    / "recommendation_deterministic_ranker_v1.0.0.json"
)


# ================================================================
# GOVERNED SOURCE FILES
# ================================================================

RECOMMENDATION_GOVERNED_SOURCE_FILES = (
    "src/models/recommendation_spec.py",
    "src/models/recommendation_engine.py",
    "src/models/recommendation_scorer.py",
)


# ================================================================
# HASHING
# ================================================================

def sha256_file(
    path: str | Path,
) -> str:
    """
    Return the SHA-256 hash of a file.
    """

    file_path = Path(
        path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


# ================================================================
# SOURCE HASH REGISTER
# ================================================================

def build_source_hash_register(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, str]:
    """
    Hash every governed recommendation production source file.
    """

    root = Path(
        project_root
    )

    hashes = {}

    for relative_path in (
        RECOMMENDATION_GOVERNED_SOURCE_FILES
    ):
        hashes[
            relative_path
        ] = sha256_file(
            root
            /
            relative_path
        )

    return hashes


# ================================================================
# MANIFEST BUILDER
# ================================================================

def build_recommendation_manifest(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """
    Build the exact governance manifest for the frozen deterministic
    recommendation architecture.
    """

    return {
        "manifest_schema_version":
            "1.0.0",

        "model_name":
            RECOMMENDATION_MODEL_METADATA.model_name,

        "model_version":
            RECOMMENDATION_MODEL_METADATA.model_version,

        "model_type":
            RECOMMENDATION_MODEL_METADATA.model_type,

        "target_name":
            RECOMMENDATION_MODEL_METADATA.target_name,

        "target_description":
            RECOMMENDATION_TARGET,

        "artifact_type":
            "deterministic_algorithm",

        "serialized_model_artifact_required":
            False,

        "architecture": {
            "repeat_channel":
                RECOMMENDATION_REPEAT_MODEL_NAME,

            "discovery_channel":
                RECOMMENDATION_DISCOVERY_MODEL_NAME,

            "optional_hybrid":
                RECOMMENDATION_OPTIONAL_HYBRID_NAME,

            "hybrid_role":
                RECOMMENDATION_HYBRID_ROLE,
        },

        "outcome_window_days":
            RECOMMENDATION_OUTCOME_WINDOW_DAYS,

        "repeat_channel": {
            "model_name":
                RECOMMENDATION_REPEAT_MODEL_NAME,

            "primary_sort":
                RECOMMENDATION_REPEAT_PRIMARY_SORT,

            "primary_sort_ascending":
                RECOMMENDATION_REPEAT_PRIMARY_SORT_ASCENDING,

            "tie_break":
                RECOMMENDATION_REPEAT_TIE_BREAK,

            "tie_break_ascending":
                RECOMMENDATION_REPEAT_TIE_BREAK_ASCENDING,

            "research_decision":
                RECOMMENDATION_REPEAT_DECISION,

            "approved_use":
                RECOMMENDATION_REPEAT_APPROVED_USE,
        },

        "discovery_channel": {
            "model_name":
                RECOMMENDATION_DISCOVERY_MODEL_NAME,

            "similarity_name":
                RECOMMENDATION_AFFINITY_SIMILARITY_NAME,

            "similarity_formula":
                RECOMMENDATION_AFFINITY_FORMULA,

            "minimum_item_orders":
                RECOMMENDATION_MINIMUM_ITEM_ORDERS,

            "minimum_pair_orders":
                RECOMMENDATION_MINIMUM_PAIR_ORDERS,

            "source_weight_formula":
                RECOMMENDATION_SOURCE_WEIGHT_FORMULA,

            "exclude_customer_history":
                RECOMMENDATION_DISCOVERY_EXCLUDE_HISTORY,

            "fallback":
                RECOMMENDATION_DISCOVERY_FALLBACK,

            "research_decision":
                RECOMMENDATION_DISCOVERY_DECISION,

            "approved_use":
                RECOMMENDATION_DISCOVERY_APPROVED_USE,

            "prohibited_interpretation":
                RECOMMENDATION_DISCOVERY_PROHIBITED_INTERPRETATION,
        },

        "hybrid": {
            "name":
                RECOMMENDATION_OPTIONAL_HYBRID_NAME,

            "repeat_fraction":
                RECOMMENDATION_HYBRID_REPEAT_FRACTION,

            "discovery_fraction":
                RECOMMENDATION_HYBRID_DISCOVERY_FRACTION,

            "fill_order":
                list(
                    RECOMMENDATION_HYBRID_FILL_ORDER
                ),

            "role":
                RECOMMENDATION_HYBRID_ROLE,
        },

        "recommendation_depths": {
            "evaluation_depths":
                list(
                    RECOMMENDATION_EVALUATION_DEPTHS
                ),

            "production_default_top_n":
                RECOMMENDATION_DEFAULT_TOP_N,
        },

        "temporal_governance": {
            "global_history_rule":
                RECOMMENDATION_GLOBAL_HISTORY_RULE,

            "customer_history_rule":
                RECOMMENDATION_CUSTOMER_HISTORY_RULE,

            "future_information_allowed":
                RECOMMENDATION_FUTURE_INFORMATION_ALLOWED,
        },

        "metrics": {
            "primary_general_metrics":
                list(
                    RECOMMENDATION_PRIMARY_GENERAL_METRICS
                ),

            "primary_discovery_metric":
                RECOMMENDATION_PRIMARY_DISCOVERY_METRIC,
        },

        "development_periods": {
            "validation_start":
                RECOMMENDATION_VALIDATION_START,

            "validation_end":
                RECOMMENDATION_VALIDATION_END,

            "test_start":
                RECOMMENDATION_TEST_START,

            "test_end":
                RECOMMENDATION_TEST_END,
        },

        "holdout_governance": {
            "holdout_consumed":
                RECOMMENDATION_HOLDOUT_CONSUMED,

            "post_holdout_tuning_permitted":
                RECOMMENDATION_POST_HOLDOUT_TUNING_PERMITTED,
        },

        "final_holdout_performance": {
            "repeat_hit_rate_at_10":
                RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10,

            "repeat_hit_rate_at_20":
                RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20,

            "repeat_eligible_hit_rate_at_20":
                RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20,

            "discovery_hit_rate_at_10":
                RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10,

            "discovery_hit_rate_at_20":
                RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20,

            "hybrid_hit_rate_at_20":
                RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20,
        },

        "governed_source_sha256":
            build_source_hash_register(
                project_root=project_root
            ),
    }


# ================================================================
# MANIFEST LOADING
# ================================================================

def load_recommendation_manifest(
    path: str | Path = DEFAULT_RECOMMENDATION_MANIFEST_PATH,
) -> dict[str, Any]:
    """
    Load and minimally validate a recommendation governance manifest.
    """

    manifest_path = Path(
        path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Recommendation governance manifest not found: "
            f"{manifest_path}"
        )

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            manifest = json.load(
                handle
            )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Recommendation governance manifest is not valid JSON."
        ) from exc

    required_fields = {
        "manifest_schema_version",
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "target_description",
        "artifact_type",
        "serialized_model_artifact_required",
        "architecture",
        "outcome_window_days",
        "repeat_channel",
        "discovery_channel",
        "hybrid",
        "recommendation_depths",
        "temporal_governance",
        "metrics",
        "development_periods",
        "holdout_governance",
        "final_holdout_performance",
        "governed_source_sha256",
    }

    missing = (
        required_fields
        -
        set(
            manifest
        )
    )

    if missing:
        raise ValueError(
            "Recommendation manifest is missing required fields: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    return manifest


# ================================================================
# MANIFEST CONTRACT VALIDATION
# ================================================================

def validate_recommendation_manifest(
    manifest: dict[str, Any],
    project_root: str | Path = PROJECT_ROOT,
    verify_source_hashes: bool = True,
) -> None:
    """
    Validate a manifest against the exact frozen recommendation
    production specification.

    When verify_source_hashes=True, the current governed production
    source files must exactly match the hashes stored in the manifest.
    """

    expected = build_recommendation_manifest(
        project_root=project_root
    )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    for field in (
        "manifest_schema_version",
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "target_description",
        "artifact_type",
        "serialized_model_artifact_required",
        "outcome_window_days",
    ):
        if (
            manifest.get(
                field
            )
            !=
            expected[
                field
            ]
        ):
            raise ValueError(
                "Recommendation manifest mismatch: "
                f"{field}."
            )

    # ------------------------------------------------------------
    # Frozen nested architecture
    # ------------------------------------------------------------

    for field in (
        "architecture",
        "repeat_channel",
        "discovery_channel",
        "hybrid",
        "recommendation_depths",
        "temporal_governance",
        "metrics",
        "development_periods",
        "holdout_governance",
        "final_holdout_performance",
    ):
        if (
            manifest.get(
                field
            )
            !=
            expected[
                field
            ]
        ):
            raise ValueError(
                "Recommendation manifest mismatch: "
                f"{field}."
            )

    # ------------------------------------------------------------
    # Governed source hashes
    # ------------------------------------------------------------

    source_hashes = manifest.get(
        "governed_source_sha256"
    )

    if not isinstance(
        source_hashes,
        dict,
    ):
        raise ValueError(
            "Recommendation manifest governed_source_sha256 "
            "must be a dictionary."
        )

    expected_source_files = set(
        RECOMMENDATION_GOVERNED_SOURCE_FILES
    )

    manifest_source_files = set(
        source_hashes
    )

    if (
        manifest_source_files
        !=
        expected_source_files
    ):
        raise ValueError(
            "Recommendation manifest governed source-file register "
            "does not match the frozen production contract."
        )

    if verify_source_hashes:

        current_hashes = (
            build_source_hash_register(
                project_root=project_root
            )
        )

        if (
            source_hashes
            !=
            current_hashes
        ):
            raise ValueError(
                "Recommendation governed source SHA-256 validation "
                "failed; production recommendation code has changed."
            )