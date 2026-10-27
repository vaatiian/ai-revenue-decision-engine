from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.models.demand_scorer import (
    DEMAND_SCORE_OUTPUT_COLUMNS,
    DemandScorer,
)
from src.models.fcv_scorer import (
    FCVScorer,
)
from src.models.leakage_scorer import (
    LeakageScorer,
)
from src.models.reactivation_scorer import (
    ReactivationScorer,
)
from src.models.recommendation_scorer import (
    RECOMMENDATION_SCORE_COLUMNS,
    RecommendationScorer,
)
from src.models.repeat_purchase_scorer import (
    REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS,
    RepeatPurchaseScorer,
)
from src.models.score_registry import (
    CANONICAL_SCORE_IDS,
    DEMAND_FORECAST_ENTRY,
    FCV_OUTPUT_COLUMNS,
    FUTURE_CUSTOMER_VALUE_ENTRY,
    GovernanceDecision,
    REACTIVATION_ENTRY,
    REACTIVATION_OUTPUT_COLUMNS,
    RECOMMENDATION_ENTRY,
    REPEAT_PURCHASE_ENTRY,
    REVENUE_LEAKAGE_ENTRY,
    REVENUE_LEAKAGE_OUTPUT_COLUMNS,
    SCORE_REGISTRY,
    ScoreRegistryEntry,
    get_score_entry,
    list_score_entries,
    list_score_ids,
    validate_score_registry,
)


EXPECTED_SCORE_IDS = (
    "reactivation",
    "revenue_leakage",
    "future_customer_value",
    "repeat_purchase",
    "recommendation",
    "demand_forecast",
)


# ================================================================
# REGISTRY IDENTITY
# ================================================================

def test_registry_contains_exact_six_scores():
    assert (
        tuple(
            SCORE_REGISTRY.keys()
        )
        ==
        EXPECTED_SCORE_IDS
    )


def test_canonical_score_ids_are_exact():
    assert (
        CANONICAL_SCORE_IDS
        ==
        EXPECTED_SCORE_IDS
    )


def test_list_score_ids_is_stable():
    assert (
        list_score_ids()
        ==
        EXPECTED_SCORE_IDS
    )


def test_list_score_entries_is_stable():
    entries = (
        list_score_entries()
    )

    assert tuple(
        entry.score_id
        for entry
        in entries
    ) == EXPECTED_SCORE_IDS


@pytest.mark.parametrize(
    "score_id",
    EXPECTED_SCORE_IDS,
)
def test_get_score_entry_returns_registered_object(
    score_id,
):
    assert (
        get_score_entry(
            score_id
        )
        is
        SCORE_REGISTRY[
            score_id
        ]
    )


def test_get_score_entry_rejects_unknown_score():
    with pytest.raises(
        KeyError,
        match="Unknown score_id",
    ):
        get_score_entry(
            "does_not_exist"
        )


def test_get_score_entry_rejects_non_string():
    with pytest.raises(
        TypeError,
        match="score_id must be a string",
    ):
        get_score_entry(
            123
        )


def test_get_score_entry_rejects_empty_string():
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        get_score_entry(
            ""
        )


def test_get_score_entry_rejects_whitespace_only():
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        get_score_entry(
            "   "
        )


# ================================================================
# IMMUTABILITY
# ================================================================

def test_registry_entries_are_immutable():
    entry = get_score_entry(
        "reactivation"
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        entry.score_id = "changed"


def test_governance_decisions_are_immutable():
    decision = GovernanceDecision(
        scope="overall",
        decision="GO",
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        decision.scope = "changed"


def test_registry_entry_type_is_correct():
    assert isinstance(
        REACTIVATION_ENTRY,
        ScoreRegistryEntry,
    )


# ================================================================
# REACTIVATION CONTRACT
# ================================================================

def test_reactivation_contract():
    entry = REACTIVATION_ENTRY

    assert (
        entry.scorer_class
        is
        ReactivationScorer
    )

    assert (
        entry.scorer_type
        ==
        "fitted_classifier"
    )

    assert (
        entry.output_columns
        ==
        REACTIVATION_OUTPUT_COLUMNS
    )

    assert (
        entry.key_columns
        ==
        ("customer",)
    )

    assert (
        "reactivation_probability"
        in
        entry.output_columns
    )

    assert (
        entry.feature_names
        is not None
    )

    assert (
        entry.artifact_required
        is True
    )

    assert (
        entry.manifest_required
        is True
    )

    assert (
        entry.deterministic
        is False
    )


def test_reactivation_legacy_governance_gap_is_explicit():
    assert (
        REACTIVATION_ENTRY
        .governance_decisions
        ==
        ()
    )


# ================================================================
# REVENUE LEAKAGE CONTRACT
# ================================================================

def test_revenue_leakage_contract():
    entry = (
        REVENUE_LEAKAGE_ENTRY
    )

    assert (
        entry.scorer_class
        is
        LeakageScorer
    )

    assert (
        entry.scorer_type
        ==
        "fitted_classifier"
    )

    assert (
        entry.output_columns
        ==
        REVENUE_LEAKAGE_OUTPUT_COLUMNS
    )

    assert (
        entry.key_columns
        ==
        ("customer",)
    )

    assert (
        "revenue_leakage_probability"
        in
        entry.output_columns
    )

    assert (
        entry.feature_names
        is not None
    )

    assert (
        entry.artifact_required
        is True
    )

    assert (
        entry.manifest_required
        is True
    )

    assert (
        entry.deterministic
        is False
    )


def test_leakage_legacy_governance_gap_is_explicit():
    assert (
        REVENUE_LEAKAGE_ENTRY
        .governance_decisions
        ==
        ()
    )


# ================================================================
# FCV CONTRACT
# ================================================================

def test_fcv_contract():
    entry = (
        FUTURE_CUSTOMER_VALUE_ENTRY
    )

    assert (
        entry.scorer_class
        is
        FCVScorer
    )

    assert (
        entry.scorer_type
        ==
        "fitted_hurdle"
    )

    assert (
        entry.output_columns
        ==
        FCV_OUTPUT_COLUMNS
    )

    assert (
        entry.key_columns
        ==
        ("customer",)
    )

    assert (
        "future_active_probability"
        in
        entry.output_columns
    )

    assert (
        "fcv_history_180d_revenue"
        in
        entry.output_columns
    )

    assert (
        "future_customer_value_90d"
        in
        entry.output_columns
    )

    assert (
        entry.artifact_required
        is True
    )

    assert (
        entry.manifest_required
        is True
    )

    assert (
        entry.deterministic
        is False
    )

    assert (
        len(
            entry.governance_decisions
        )
        ==
        1
    )

    assert (
        entry.governance_decisions[
            0
        ].scope
        ==
        "overall"
    )

    assert (
        entry.governance_decisions[
            0
        ].decision
        ==
        "CONDITIONAL GO"
    )


# ================================================================
# REPEAT PURCHASE CONTRACT
# ================================================================

def test_repeat_purchase_contract():
    entry = (
        REPEAT_PURCHASE_ENTRY
    )

    assert (
        entry.scorer_class
        is
        RepeatPurchaseScorer
    )

    assert (
        entry.scorer_type
        ==
        "fitted_classifier"
    )

    assert (
        entry.output_columns
        ==
        tuple(
            REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS
        )
    )

    assert (
        entry.key_columns
        ==
        (
            "customer",
            "first_invoice",
        )
    )

    assert (
        entry.artifact_required
        is True
    )

    assert (
        entry.manifest_required
        is True
    )

    assert (
        entry.deterministic
        is False
    )

    assert (
        entry.governance_decisions[
            0
        ].scope
        ==
        "overall"
    )

    assert (
        entry.governance_decisions[
            0
        ].decision
        ==
        "CONDITIONAL GO"
    )


def test_repeat_purchase_feature_contract_is_registered():
    assert (
        REPEAT_PURCHASE_ENTRY
        .feature_names
        is not None
    )

    assert (
        len(
            REPEAT_PURCHASE_ENTRY
            .feature_names
        )
        ==
        18
    )


# ================================================================
# RECOMMENDATION CONTRACT
# ================================================================

def test_recommendation_contract():
    entry = (
        RECOMMENDATION_ENTRY
    )

    assert (
        entry.scorer_class
        is
        RecommendationScorer
    )

    assert (
        entry.scorer_type
        ==
        "deterministic_recommender"
    )

    assert (
        entry.output_columns
        ==
        tuple(
            RECOMMENDATION_SCORE_COLUMNS
        )
    )

    assert (
        entry.artifact_required
        is False
    )

    assert (
        entry.artifact_path
        is None
    )

    assert (
        entry.manifest_required
        is True
    )

    assert (
        entry.deterministic
        is True
    )


def test_recommendation_governance_contract():
    decisions = tuple(
        (
            item.scope,
            item.decision,
        )
        for item
        in RECOMMENDATION_ENTRY
        .governance_decisions
    )

    assert decisions == (
        (
            "repeat",
            "GO",
        ),
        (
            "discovery",
            "CONDITIONAL GO",
        ),
    )


def test_recommendation_has_no_tabular_feature_register():
    assert (
        RECOMMENDATION_ENTRY
        .feature_names
        is None
    )


# ================================================================
# DEMAND CONTRACT
# ================================================================

def test_demand_contract():
    entry = (
        DEMAND_FORECAST_ENTRY
    )

    assert (
        entry.scorer_class
        is
        DemandScorer
    )

    assert (
        entry.scorer_type
        ==
        "deterministic_forecaster"
    )

    assert (
        entry.output_columns
        ==
        tuple(
            DEMAND_SCORE_OUTPUT_COLUMNS
        )
    )

    assert (
        entry.key_columns
        ==
        (
            "forecast_origin_week",
            "horizon_weeks",
        )
    )

    assert (
        entry.artifact_required
        is False
    )

    assert (
        entry.artifact_path
        is None
    )

    assert (
        entry.manifest_required
        is True
    )

    assert (
        entry.deterministic
        is True
    )


def test_demand_governance_contract():
    assert (
        DEMAND_FORECAST_ENTRY
        .governance_decisions[
            0
        ].scope
        ==
        "overall"
    )

    assert (
        DEMAND_FORECAST_ENTRY
        .governance_decisions[
            0
        ].decision
        ==
        "GO"
    )


def test_demand_has_no_tabular_feature_register():
    assert (
        DEMAND_FORECAST_ENTRY
        .feature_names
        is None
    )


# ================================================================
# ARTIFACT / MANIFEST STATIC CONTRACT
# ================================================================

@pytest.mark.parametrize(
    "score_id",
    (
        "reactivation",
        "revenue_leakage",
        "future_customer_value",
        "repeat_purchase",
    ),
)
def test_fitted_scores_have_artifact_paths(
    score_id,
):
    entry = get_score_entry(
        score_id
    )

    assert (
        entry.artifact_required
        is True
    )

    assert isinstance(
        entry.artifact_path,
        Path,
    )


@pytest.mark.parametrize(
    "score_id",
    EXPECTED_SCORE_IDS,
)
def test_every_score_has_manifest_path(
    score_id,
):
    entry = get_score_entry(
        score_id
    )

    assert (
        entry.manifest_required
        is True
    )

    assert isinstance(
        entry.manifest_path,
        Path,
    )


def test_deterministic_scores_have_no_artifact():
    for score_id in (
        "recommendation",
        "demand_forecast",
    ):
        entry = get_score_entry(
            score_id
        )

        assert (
            entry.deterministic
            is True
        )

        assert (
            entry.artifact_required
            is False
        )

        assert (
            entry.artifact_path
            is None
        )


def test_fitted_scores_are_not_deterministic():
    for score_id in (
        "reactivation",
        "revenue_leakage",
        "future_customer_value",
        "repeat_purchase",
    ):
        entry = get_score_entry(
            score_id
        )

        assert (
            entry.deterministic
            is False
        )

        assert (
            entry.artifact_required
            is True
        )


# ================================================================
# OUTPUT / KEY CONTRACT
# ================================================================

@pytest.mark.parametrize(
    "score_id",
    EXPECTED_SCORE_IDS,
)
def test_output_columns_are_unique(
    score_id,
):
    output_columns = (
        get_score_entry(
            score_id
        )
        .output_columns
    )

    assert (
        len(
            output_columns
        )
        ==
        len(
            set(
                output_columns
            )
        )
    )


@pytest.mark.parametrize(
    "score_id",
    EXPECTED_SCORE_IDS,
)
def test_key_columns_exist_in_output_contract(
    score_id,
):
    entry = get_score_entry(
        score_id
    )

    assert all(
        key
        in
        entry.output_columns
        for key
        in entry.key_columns
    )


@pytest.mark.parametrize(
    "score_id",
    (
        "reactivation",
        "revenue_leakage",
        "future_customer_value",
        "repeat_purchase",
    ),
)
def test_fitted_feature_registers_are_nonempty(
    score_id,
):
    entry = get_score_entry(
        score_id
    )

    assert (
        entry.feature_names
        is not None
    )

    assert (
        len(
            entry.feature_names
        )
        >
        0
    )

    assert (
        len(
            entry.feature_names
        )
        ==
        len(
            set(
                entry.feature_names
            )
        )
    )


# ================================================================
# IDENTITY / STATIC VALIDATION
# ================================================================

def test_registry_entries_have_nonempty_identity():
    for entry in (
        list_score_entries()
    ):
        assert entry.score_id
        assert entry.display_name
        assert entry.domain
        assert entry.scorer_type
        assert entry.model_name
        assert entry.model_version
        assert entry.model_type
        assert entry.target_name


def test_registry_validation_passes():
    assert (
        validate_score_registry()
        is None
    )
