"""
System-level integration tests for the production score registry and
runtime availability layer.

Unlike the isolated unit tests, these checks exercise the real static
registry together with the repository's actual governed runtime
dependencies.

The objective is to prove that:
- all six production capabilities are registered;
- runtime health preserves the canonical registry order;
- public lookup and aggregate APIs agree;
- fitted scorers have governed artifacts and manifests;
- deterministic scorers have governed manifests without serialized
  model artifacts;
- the current repository is production-health complete.
"""

from src.models.score_availability import (
    ScoreAvailabilityStatus,
    all_scores_available,
    get_score_status,
    list_score_statuses,
)
from src.models.score_registry import (
    CANONICAL_SCORE_IDS,
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

FITTED_SCORE_IDS = (
    "reactivation",
    "revenue_leakage",
    "future_customer_value",
    "repeat_purchase",
)

DETERMINISTIC_SCORE_IDS = (
    "recommendation",
    "demand_forecast",
)


# ================================================================
# REGISTRY → RUNTIME INTEGRATION
# ================================================================


def test_runtime_registry_contains_exact_six_scores():

    validate_score_registry()

    assert CANONICAL_SCORE_IDS == EXPECTED_SCORE_IDS

    assert list_score_ids() == EXPECTED_SCORE_IDS

    assert tuple(
        entry.score_id
        for entry
        in list_score_entries()
    ) == EXPECTED_SCORE_IDS


def test_runtime_statuses_preserve_registry_order():

    statuses = (
        list_score_statuses()
    )

    assert tuple(
        status.score_id
        for status
        in statuses
    ) == EXPECTED_SCORE_IDS


def test_individual_runtime_lookup_matches_bulk_status():

    statuses = {
        status.score_id:
            status
        for status
        in list_score_statuses()
    }

    for score_id in EXPECTED_SCORE_IDS:

        individual = (
            get_score_status(
                score_id
            )
        )

        bulk = statuses[
            score_id
        ]

        assert individual.score_id == bulk.score_id
        assert individual.status == bulk.status
        assert individual.available == bulk.available
        assert individual.artifact_exists == bulk.artifact_exists
        assert individual.manifest_exists == bulk.manifest_exists


# ================================================================
# FITTED SCORER RUNTIME HEALTH
# ================================================================


def test_fitted_scores_have_available_artifacts_and_manifests():

    for score_id in FITTED_SCORE_IDS:

        entry = (
            get_score_entry(
                score_id
            )
        )

        status = (
            get_score_status(
                score_id
            )
        )

        assert entry.artifact_required is True
        assert entry.manifest_required is True
        assert entry.deterministic is False

        assert (
            status.status
            ==
            ScoreAvailabilityStatus.AVAILABLE
        )

        assert status.available is True
        assert status.artifact_exists is True
        assert status.manifest_exists is True


# ================================================================
# DETERMINISTIC SCORER RUNTIME HEALTH
# ================================================================


def test_deterministic_scores_require_manifest_but_no_artifact():

    for score_id in DETERMINISTIC_SCORE_IDS:

        entry = (
            get_score_entry(
                score_id
            )
        )

        status = (
            get_score_status(
                score_id
            )
        )

        assert entry.artifact_required is False
        assert entry.manifest_required is True
        assert entry.deterministic is True

        assert (
            status.status
            ==
            ScoreAvailabilityStatus.AVAILABLE
        )

        assert status.available is True
        assert status.artifact_exists is None
        assert status.manifest_exists is True


# ================================================================
# AGGREGATE PRODUCTION HEALTH
# ================================================================


def test_all_registered_scores_are_runtime_available():

    statuses = (
        list_score_statuses()
    )

    assert len(
        statuses
    ) == len(
        EXPECTED_SCORE_IDS
    )

    assert all(
        status.status
        ==
        ScoreAvailabilityStatus.AVAILABLE
        for status
        in statuses
    )

    assert all(
        status.available
        for status
        in statuses
    )

    assert all_scores_available() is True