"""
Central production score registry.

This module is the single static discovery contract for governed production
scoring capabilities in the AI Revenue Decision Engine.

Responsibilities
----------------
1. Provide stable public score IDs.
2. Describe scorer implementations without instantiating them.
3. Expose model identity, keys, features and output contracts.
4. Preserve differences between fitted and deterministic scorers.
5. Describe artifact and manifest dependencies.
6. Preserve governance decisions exactly where formally defined.
7. Fail closed if the static registry becomes internally inconsistent.

This module does NOT:
- load fitted model artifacts;
- execute scorers;
- test runtime artifact availability;
- train or tune models;
- change frozen research specifications.

Runtime availability is introduced separately in Phase 5E.
"""

from dataclasses import dataclass
from pathlib import Path


# ================================================================
# REACTIVATION
# ================================================================

from src.models.reactivation_scorer import (
    DEFAULT_REACTIVATION_ARTIFACT_PATH,
    DEFAULT_REACTIVATION_MANIFEST_PATH,
    ReactivationScorer,
)
from src.models.reactivation_spec import (
    REACTIVATION_MODEL_FEATURES,
    REACTIVATION_MODEL_METADATA,
)


# ================================================================
# REVENUE LEAKAGE
# ================================================================

from src.models.leakage_scorer import (
    DEFAULT_LEAKAGE_ARTIFACT_PATH,
    DEFAULT_LEAKAGE_MANIFEST_PATH,
    LeakageScorer,
)
from src.models.leakage_spec import (
    LEAKAGE_MODEL_FEATURES,
    LEAKAGE_MODEL_METADATA,
    LEAKAGE_OUTCOME_WINDOW_DAYS,
    LEAKAGE_PREDICTION_UNIT,
)


# ================================================================
# FUTURE CUSTOMER VALUE
# ================================================================

from src.models.fcv_scorer import (
    DEFAULT_FCV_ARTIFACT_PATH,
    DEFAULT_FCV_MANIFEST_PATH,
    FCVScorer,
)
from src.models.fcv_spec import (
    FCV_FINAL_DECISION,
    FCV_MODEL_FEATURES,
    FCV_MODEL_METADATA,
    FCV_OUTCOME_WINDOW_DAYS,
    FCV_PREDICTION_UNIT,
)


# ================================================================
# RECOMMENDATION
# ================================================================

from src.models.recommendation_manifest import (
    DEFAULT_RECOMMENDATION_MANIFEST_PATH,
)
from src.models.recommendation_scorer import (
    RECOMMENDATION_SCORE_COLUMNS,
    RecommendationScorer,
)
from src.models.recommendation_spec import (
    RECOMMENDATION_DISCOVERY_DECISION,
    RECOMMENDATION_MODEL_METADATA,
    RECOMMENDATION_PREDICTION_UNIT,
    RECOMMENDATION_REPEAT_DECISION,
)


# ================================================================
# DEMAND FORECASTING
# ================================================================

from src.models.demand_manifest import (
    DEMAND_MANIFEST_PATH,
)
from src.models.demand_scorer import (
    DEMAND_SCORE_OUTPUT_COLUMNS,
    DemandScorer,
)
from src.models.demand_spec import (
    DEMAND_FINAL_DECISION,
    DEMAND_FORECAST_HORIZONS,
    DEMAND_MODEL_METADATA,
)


# ================================================================
# REPEAT PURCHASE
# ================================================================

from src.models.repeat_purchase_manifest import (
    DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH,
    DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH,
)
from src.models.repeat_purchase_scorer import (
    REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS,
    RepeatPurchaseScorer,
)
from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_FINAL_DECISION,
    REPEAT_PURCHASE_HORIZON_DAYS,
    REPEAT_PURCHASE_MODEL_FEATURES,
    REPEAT_PURCHASE_MODEL_METADATA,
    REPEAT_PURCHASE_PREDICTION_UNIT,
)


# ================================================================
# PUBLIC SCORE IDS
# ================================================================

REACTIVATION_SCORE_ID = (
    "reactivation"
)

REVENUE_LEAKAGE_SCORE_ID = (
    "revenue_leakage"
)

FUTURE_CUSTOMER_VALUE_SCORE_ID = (
    "future_customer_value"
)

REPEAT_PURCHASE_SCORE_ID = (
    "repeat_purchase"
)

RECOMMENDATION_SCORE_ID = (
    "recommendation"
)

DEMAND_FORECAST_SCORE_ID = (
    "demand_forecast"
)


CANONICAL_SCORE_IDS = (
    REACTIVATION_SCORE_ID,
    REVENUE_LEAKAGE_SCORE_ID,
    FUTURE_CUSTOMER_VALUE_SCORE_ID,
    REPEAT_PURCHASE_SCORE_ID,
    RECOMMENDATION_SCORE_ID,
    DEMAND_FORECAST_SCORE_ID,
)


# ================================================================
# SHARED OUTPUT METADATA
# ================================================================

STANDARD_MODEL_METADATA_COLUMNS = (
    "model_name",
    "model_version",
    "model_type",
    "target_name",
    "scored_at_utc",
)


# ================================================================
# FITTED MODEL OUTPUT CONTRACTS
# ================================================================

REACTIVATION_OUTPUT_COLUMNS = (
    "customer",
    "reactivation_probability",
    *STANDARD_MODEL_METADATA_COLUMNS,
)


REVENUE_LEAKAGE_OUTPUT_COLUMNS = (
    "customer",
    "revenue_leakage_probability",
    *STANDARD_MODEL_METADATA_COLUMNS,
)


FCV_OUTPUT_COLUMNS = (
    "customer",
    "future_active_probability",
    "fcv_history_180d_revenue",
    "future_customer_value_90d",
    *STANDARD_MODEL_METADATA_COLUMNS,
)


# ================================================================
# GOVERNANCE
# ================================================================

@dataclass(frozen=True)
class GovernanceDecision:
    """
    One formally defined governance decision.

    Some legacy scorers do not currently expose an explicit final decision.
    Their registry entry therefore carries an empty decisions tuple rather
    than inventing governance metadata.

    Recommendation has separate channel-level decisions.
    """

    scope: str
    decision: str


# ================================================================
# REGISTRY ENTRY
# ================================================================

@dataclass(frozen=True)
class ScoreRegistryEntry:
    """
    Immutable static description of one production scoring capability.

    scorer_class is stored as a class reference only. No scorer is
    instantiated when this registry module is imported.
    """

    score_id: str
    display_name: str
    domain: str

    scorer_type: str
    scorer_class: type

    model_name: str
    model_version: str
    model_type: str
    target_name: str

    prediction_unit: str | None
    horizon: str | None

    output_columns: tuple[str, ...]

    feature_names: tuple[str, ...] | None
    key_columns: tuple[str, ...]

    artifact_required: bool
    artifact_path: Path | None

    manifest_required: bool
    manifest_path: Path | None

    governance_decisions: tuple[
        GovernanceDecision,
        ...,
    ]

    deterministic: bool


# ================================================================
# REACTIVATION ENTRY
# ================================================================

REACTIVATION_ENTRY = (
    ScoreRegistryEntry(
        score_id=(
            REACTIVATION_SCORE_ID
        ),
        display_name=(
            "Customer Reactivation"
        ),
        domain="customer",
        scorer_type=(
            "fitted_classifier"
        ),
        scorer_class=(
            ReactivationScorer
        ),
        model_name=(
            REACTIVATION_MODEL_METADATA
            .model_name
        ),
        model_version=(
            REACTIVATION_MODEL_METADATA
            .model_version
        ),
        model_type=(
            REACTIVATION_MODEL_METADATA
            .model_type
        ),
        target_name=(
            REACTIVATION_MODEL_METADATA
            .target_name
        ),
        prediction_unit=None,
        horizon="90 days",
        output_columns=(
            REACTIVATION_OUTPUT_COLUMNS
        ),
        feature_names=tuple(
            REACTIVATION_MODEL_FEATURES
        ),
        key_columns=(
            "customer",
        ),
        artifact_required=True,
        artifact_path=Path(
            DEFAULT_REACTIVATION_ARTIFACT_PATH
        ),
        manifest_required=True,
        manifest_path=Path(
            DEFAULT_REACTIVATION_MANIFEST_PATH
        ),
        governance_decisions=(),
        deterministic=False,
    )
)


# ================================================================
# REVENUE LEAKAGE ENTRY
# ================================================================

REVENUE_LEAKAGE_ENTRY = (
    ScoreRegistryEntry(
        score_id=(
            REVENUE_LEAKAGE_SCORE_ID
        ),
        display_name=(
            "Revenue Leakage Risk"
        ),
        domain="customer",
        scorer_type=(
            "fitted_classifier"
        ),
        scorer_class=(
            LeakageScorer
        ),
        model_name=(
            LEAKAGE_MODEL_METADATA
            .model_name
        ),
        model_version=(
            LEAKAGE_MODEL_METADATA
            .model_version
        ),
        model_type=(
            LEAKAGE_MODEL_METADATA
            .model_type
        ),
        target_name=(
            LEAKAGE_MODEL_METADATA
            .target_name
        ),
        prediction_unit=(
            LEAKAGE_PREDICTION_UNIT
        ),
        horizon=(
            f"{LEAKAGE_OUTCOME_WINDOW_DAYS} days"
        ),
        output_columns=(
            REVENUE_LEAKAGE_OUTPUT_COLUMNS
        ),
        feature_names=tuple(
            LEAKAGE_MODEL_FEATURES
        ),
        key_columns=(
            "customer",
        ),
        artifact_required=True,
        artifact_path=Path(
            DEFAULT_LEAKAGE_ARTIFACT_PATH
        ),
        manifest_required=True,
        manifest_path=Path(
            DEFAULT_LEAKAGE_MANIFEST_PATH
        ),
        governance_decisions=(),
        deterministic=False,
    )
)


# ================================================================
# FUTURE CUSTOMER VALUE ENTRY
# ================================================================

FUTURE_CUSTOMER_VALUE_ENTRY = (
    ScoreRegistryEntry(
        score_id=(
            FUTURE_CUSTOMER_VALUE_SCORE_ID
        ),
        display_name=(
            "Future Customer Value"
        ),
        domain="customer",
        scorer_type=(
            "fitted_hurdle"
        ),
        scorer_class=(
            FCVScorer
        ),
        model_name=(
            FCV_MODEL_METADATA
            .model_name
        ),
        model_version=(
            FCV_MODEL_METADATA
            .model_version
        ),
        model_type=(
            FCV_MODEL_METADATA
            .model_type
        ),
        target_name=(
            FCV_MODEL_METADATA
            .target_name
        ),
        prediction_unit=(
            FCV_PREDICTION_UNIT
        ),
        horizon=(
            f"{FCV_OUTCOME_WINDOW_DAYS} days"
        ),
        output_columns=(
            FCV_OUTPUT_COLUMNS
        ),
        feature_names=tuple(
            FCV_MODEL_FEATURES
        ),
        key_columns=(
            "customer",
        ),
        artifact_required=True,
        artifact_path=Path(
            DEFAULT_FCV_ARTIFACT_PATH
        ),
        manifest_required=True,
        manifest_path=Path(
            DEFAULT_FCV_MANIFEST_PATH
        ),
        governance_decisions=(
            GovernanceDecision(
                scope="overall",
                decision=(
                    FCV_FINAL_DECISION
                ),
            ),
        ),
        deterministic=False,
    )
)


# ================================================================
# REPEAT PURCHASE ENTRY
# ================================================================

REPEAT_PURCHASE_ENTRY = (
    ScoreRegistryEntry(
        score_id=(
            REPEAT_PURCHASE_SCORE_ID
        ),
        display_name=(
            "Repeat Purchase Propensity"
        ),
        domain="customer",
        scorer_type=(
            "fitted_classifier"
        ),
        scorer_class=(
            RepeatPurchaseScorer
        ),
        model_name=(
            REPEAT_PURCHASE_MODEL_METADATA
            .model_name
        ),
        model_version=(
            REPEAT_PURCHASE_MODEL_METADATA
            .model_version
        ),
        model_type=(
            REPEAT_PURCHASE_MODEL_METADATA
            .model_type
        ),
        target_name=(
            REPEAT_PURCHASE_MODEL_METADATA
            .target_name
        ),
        prediction_unit=(
            REPEAT_PURCHASE_PREDICTION_UNIT
        ),
        horizon=(
            f"{REPEAT_PURCHASE_HORIZON_DAYS} days"
        ),
        output_columns=tuple(
            REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS
        ),
        feature_names=tuple(
            REPEAT_PURCHASE_MODEL_FEATURES
        ),
        key_columns=(
            "customer",
            "first_invoice",
        ),
        artifact_required=True,
        artifact_path=Path(
            DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH
        ),
        manifest_required=True,
        manifest_path=Path(
            DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH
        ),
        governance_decisions=(
            GovernanceDecision(
                scope="overall",
                decision=(
                    REPEAT_PURCHASE_FINAL_DECISION
                ),
            ),
        ),
        deterministic=False,
    )
)


# ================================================================
# RECOMMENDATION ENTRY
# ================================================================

RECOMMENDATION_ENTRY = (
    ScoreRegistryEntry(
        score_id=(
            RECOMMENDATION_SCORE_ID
        ),
        display_name=(
            "Next Best Product"
        ),
        domain="recommendation",
        scorer_type=(
            "deterministic_recommender"
        ),
        scorer_class=(
            RecommendationScorer
        ),
        model_name=(
            RECOMMENDATION_MODEL_METADATA
            .model_name
        ),
        model_version=(
            RECOMMENDATION_MODEL_METADATA
            .model_version
        ),
        model_type=(
            RECOMMENDATION_MODEL_METADATA
            .model_type
        ),
        target_name=(
            RECOMMENDATION_MODEL_METADATA
            .target_name
        ),
        prediction_unit=(
            RECOMMENDATION_PREDICTION_UNIT
        ),
        horizon=(
            "next qualifying order within 90 days"
        ),
        output_columns=tuple(
            RECOMMENDATION_SCORE_COLUMNS
        ),
        feature_names=None,
        key_columns=tuple(
            RecommendationScorer
            .key_columns
        ),
        artifact_required=False,
        artifact_path=None,
        manifest_required=True,
        manifest_path=Path(
            DEFAULT_RECOMMENDATION_MANIFEST_PATH
        ),
        governance_decisions=(
            GovernanceDecision(
                scope="repeat",
                decision=(
                    RECOMMENDATION_REPEAT_DECISION
                ),
            ),
            GovernanceDecision(
                scope="discovery",
                decision=(
                    RECOMMENDATION_DISCOVERY_DECISION
                ),
            ),
        ),
        deterministic=True,
    )
)


# ================================================================
# DEMAND FORECAST ENTRY
# ================================================================

DEMAND_FORECAST_ENTRY = (
    ScoreRegistryEntry(
        score_id=(
            DEMAND_FORECAST_SCORE_ID
        ),
        display_name=(
            "Weekly Demand Forecast"
        ),
        domain="demand",
        scorer_type=(
            "deterministic_forecaster"
        ),
        scorer_class=(
            DemandScorer
        ),
        model_name=(
            DEMAND_MODEL_METADATA
            .model_name
        ),
        model_version=(
            DEMAND_MODEL_METADATA
            .model_version
        ),
        model_type=(
            DEMAND_MODEL_METADATA
            .model_type
        ),
        target_name=(
            DEMAND_MODEL_METADATA
            .target_name
        ),
        prediction_unit=None,
        horizon=(
            ", ".join(
                str(
                    horizon
                )
                for horizon
                in DEMAND_FORECAST_HORIZONS
            )
            +
            " weeks"
        ),
        output_columns=tuple(
            DEMAND_SCORE_OUTPUT_COLUMNS
        ),
        feature_names=None,
        key_columns=(
            "forecast_origin_week",
            "horizon_weeks",
        ),
        artifact_required=False,
        artifact_path=None,
        manifest_required=True,
        manifest_path=Path(
            DEMAND_MANIFEST_PATH
        ),
        governance_decisions=(
            GovernanceDecision(
                scope="overall",
                decision=(
                    DEMAND_FINAL_DECISION
                ),
            ),
        ),
        deterministic=True,
    )
)


# ================================================================
# CENTRAL REGISTRY
# ================================================================

SCORE_REGISTRY = {
    REACTIVATION_ENTRY.score_id:
        REACTIVATION_ENTRY,

    REVENUE_LEAKAGE_ENTRY.score_id:
        REVENUE_LEAKAGE_ENTRY,

    FUTURE_CUSTOMER_VALUE_ENTRY.score_id:
        FUTURE_CUSTOMER_VALUE_ENTRY,

    REPEAT_PURCHASE_ENTRY.score_id:
        REPEAT_PURCHASE_ENTRY,

    RECOMMENDATION_ENTRY.score_id:
        RECOMMENDATION_ENTRY,

    DEMAND_FORECAST_ENTRY.score_id:
        DEMAND_FORECAST_ENTRY,
}


# ================================================================
# PUBLIC LOOKUP API
# ================================================================

def list_score_ids() -> tuple[
    str,
    ...,
]:
    """
    Return canonical public score IDs in stable production order.
    """

    return CANONICAL_SCORE_IDS


def list_score_entries() -> tuple[
    ScoreRegistryEntry,
    ...,
]:
    """
    Return all registry entries in stable canonical order.
    """

    return tuple(
        SCORE_REGISTRY[
            score_id
        ]
        for score_id
        in CANONICAL_SCORE_IDS
    )


def get_score_entry(
    score_id: str,
) -> ScoreRegistryEntry:
    """
    Return one static registry entry.

    Unknown score IDs fail closed.
    """

    if not isinstance(
        score_id,
        str,
    ):
        raise TypeError(
            "score_id must be a string."
        )

    if not score_id.strip():
        raise ValueError(
            "score_id must not be empty."
        )

    try:
        return SCORE_REGISTRY[
            score_id
        ]

    except KeyError as exc:

        supported = ", ".join(
            CANONICAL_SCORE_IDS
        )

        raise KeyError(
            f"Unknown score_id: "
            f"{score_id}. "
            f"Supported score IDs: "
            f"{supported}"
        ) from exc


# ================================================================
# STATIC REGISTRY VALIDATION
# ================================================================

def validate_score_registry() -> None:
    """
    Fail closed if the static production score registry drifts.
    """

    registry_ids = tuple(
        SCORE_REGISTRY.keys()
    )

    if (
        registry_ids
        !=
        CANONICAL_SCORE_IDS
    ):
        raise ValueError(
            "Score registry keys do not match "
            "the canonical score ID order."
        )

    if (
        len(
            set(
                CANONICAL_SCORE_IDS
            )
        )
        !=
        len(
            CANONICAL_SCORE_IDS
        )
    ):
        raise ValueError(
            "Canonical score IDs must be unique."
        )

    for (
        score_id,
        entry,
    ) in SCORE_REGISTRY.items():

        if not isinstance(
            entry,
            ScoreRegistryEntry,
        ):
            raise TypeError(
                f"Registry entry "
                f"{score_id} "
                "has an invalid type."
            )

        if (
            entry.score_id
            !=
            score_id
        ):
            raise ValueError(
                f"Registry key "
                f"{score_id} "
                "does not match entry "
                f"score_id "
                f"{entry.score_id}."
            )

        if not entry.display_name:
            raise ValueError(
                f"{score_id} is missing "
                "display_name."
            )

        if not entry.domain:
            raise ValueError(
                f"{score_id} is missing "
                "domain."
            )

        if not entry.scorer_type:
            raise ValueError(
                f"{score_id} is missing "
                "scorer_type."
            )

        if not isinstance(
            entry.scorer_class,
            type,
        ):
            raise TypeError(
                f"{score_id} scorer_class "
                "must be a class."
            )

        if not entry.model_name:
            raise ValueError(
                f"{score_id} is missing "
                "model_name."
            )

        if not entry.model_version:
            raise ValueError(
                f"{score_id} is missing "
                "model_version."
            )

        if not entry.model_type:
            raise ValueError(
                f"{score_id} is missing "
                "model_type."
            )

        if not entry.target_name:
            raise ValueError(
                f"{score_id} is missing "
                "target_name."
            )

        if not entry.output_columns:
            raise ValueError(
                f"{score_id} has no "
                "output columns."
            )

        if (
            len(
                entry.output_columns
            )
            !=
            len(
                set(
                    entry.output_columns
                )
            )
        ):
            raise ValueError(
                f"{score_id} contains "
                "duplicate output columns."
            )

        if not entry.key_columns:
            raise ValueError(
                f"{score_id} has no "
                "key columns."
            )

        for key_column in (
            entry.key_columns
        ):

            if (
                key_column
                not in
                entry.output_columns
            ):
                raise ValueError(
                    f"{score_id} key column "
                    f"{key_column} "
                    "is not present in its "
                    "output contract."
                )

        if (
            entry.feature_names
            is not None
        ):

            if not entry.feature_names:
                raise ValueError(
                    f"{score_id} has an "
                    "empty feature register."
                )

            if (
                len(
                    entry.feature_names
                )
                !=
                len(
                    set(
                        entry.feature_names
                    )
                )
            ):
                raise ValueError(
                    f"{score_id} contains "
                    "duplicate feature names."
                )

        if (
            entry.artifact_required
            and
            entry.artifact_path
            is None
        ):
            raise ValueError(
                f"{score_id} requires "
                "an artifact but has "
                "no artifact path."
            )

        if (
            not entry.artifact_required
            and
            entry.artifact_path
            is not None
        ):
            raise ValueError(
                f"{score_id} declares "
                "an artifact path although "
                "artifacts are not required."
            )

        if (
            entry.manifest_required
            and
            entry.manifest_path
            is None
        ):
            raise ValueError(
                f"{score_id} requires "
                "a manifest but has "
                "no manifest path."
            )

        if (
            not entry.manifest_required
            and
            entry.manifest_path
            is not None
        ):
            raise ValueError(
                f"{score_id} declares "
                "a manifest path although "
                "manifests are not required."
            )

        if (
            entry.deterministic
            and
            entry.artifact_required
        ):
            raise ValueError(
                f"{score_id} cannot be "
                "both deterministic and "
                "artifact-dependent."
            )

        scopes = tuple(
            decision.scope
            for decision
            in entry.governance_decisions
        )

        if (
            len(
                scopes
            )
            !=
            len(
                set(
                    scopes
                )
            )
        ):
            raise ValueError(
                f"{score_id} contains "
                "duplicate governance scopes."
            )

        for decision in (
            entry.governance_decisions
        ):

            if not isinstance(
                decision,
                GovernanceDecision,
            ):
                raise TypeError(
                    f"{score_id} contains "
                    "an invalid governance "
                    "decision object."
                )

            if not decision.scope:
                raise ValueError(
                    f"{score_id} contains "
                    "an empty governance scope."
                )

            if not decision.decision:
                raise ValueError(
                    f"{score_id} contains "
                    "an empty governance decision."
                )


# Validate the frozen registry contract at import time.
validate_score_registry()
