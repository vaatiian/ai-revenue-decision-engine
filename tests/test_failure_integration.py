"""
System-level fail-closed integration tests.

These tests verify that failures crossing production component boundaries
remain explicit and governed.

Coverage:
- missing fitted-model manifest;
- missing fitted-model artifact;
- corrupt fitted-model artifact;
- invalid deterministic governance manifest;
- Leakage scores without required economic context;
- Reactivation scores without required prediction-date provenance.
"""

from dataclasses import replace

import pandas as pd
import pytest

from src.decision_engine.engine import (
    build_customer_decisions,
)

from src.models.score_availability import (
    ScoreAvailabilityStatus,
    evaluate_score_availability,
)

from src.models.score_registry import (
    REACTIVATION_ENTRY,
    RECOMMENDATION_ENTRY,
)


DECISION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# DECISION ENGINE FIXTURE
# ================================================================


def make_transactions():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-08-31 10:00:00",
                    "2011-05-15 10:00:00",
                ]
            ),

            "customer": [
                "A",
                "D",
            ],

            "invoice": [
                "A1",
                "D1",
            ],

            "quantity": [
                1,
                1,
            ],

            "unit_price": [
                100.0,
                50.0,
            ],

            "net_merchandise_revenue": [
                100.0,
                50.0,
            ],

            "is_cancellation": [
                False,
                False,
            ],

            "is_return": [
                False,
                False,
            ],
        }
    )


# ================================================================
# AVAILABILITY FAIL-CLOSED INTEGRATION
# ================================================================


def test_missing_fitted_manifest_fails_closed(
    tmp_path,
):

    entry = replace(
        REACTIVATION_ENTRY,
        manifest_path=(
            tmp_path
            /
            "missing_manifest.json"
        ),
        artifact_path=(
            tmp_path
            /
            "missing_model.joblib"
        ),
    )

    result = (
        evaluate_score_availability(
            entry
        )
    )

    assert (
        result.status
        ==
        ScoreAvailabilityStatus.MISSING_MANIFEST
    )

    assert result.available is False
    assert result.manifest_exists is False
    assert result.artifact_exists is False


def test_missing_fitted_artifact_fails_closed(
    tmp_path,
):

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )

    entry = replace(
        REACTIVATION_ENTRY,
        manifest_path=manifest_path,
        artifact_path=(
            tmp_path
            /
            "missing_model.joblib"
        ),
    )

    result = (
        evaluate_score_availability(
            entry
        )
    )

    assert (
        result.status
        ==
        ScoreAvailabilityStatus.MISSING_ARTIFACT
    )

    assert result.available is False
    assert result.manifest_exists is True
    assert result.artifact_exists is False


def test_corrupt_fitted_artifact_is_invalid(
    tmp_path,
):

    artifact_path = (
        tmp_path
        /
        "corrupt_model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    artifact_path.write_bytes(
        b"this is not a valid joblib model"
    )

    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )

    entry = replace(
        REACTIVATION_ENTRY,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
    )

    result = (
        evaluate_score_availability(
            entry
        )
    )

    assert (
        result.status
        ==
        ScoreAvailabilityStatus.INVALID_ARTIFACT
    )

    assert result.available is False
    assert result.artifact_exists is True
    assert result.manifest_exists is True

    assert result.detail


def test_invalid_deterministic_manifest_fails_closed(
    tmp_path,
):

    manifest_path = (
        tmp_path
        /
        "recommendation.json"
    )

    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )

    entry = replace(
        RECOMMENDATION_ENTRY,
        manifest_path=manifest_path,
    )

    result = (
        evaluate_score_availability(
            entry
        )
    )

    assert (
        result.status
        ==
        ScoreAvailabilityStatus.INVALID_MANIFEST
    )

    assert result.available is False
    assert result.artifact_exists is None
    assert result.manifest_exists is True

    assert result.detail


# ================================================================
# DECISION ENGINE FAIL-CLOSED INTEGRATION
# ================================================================


def test_engine_rejects_leakage_scores_without_context():

    leakage_scores = pd.DataFrame(
        {
            "customer": [
                "A",
            ],

            "revenue_leakage_probability": [
                0.80,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Leakage context is required "
            "when Leakage scores are supplied"
        ),
    ):

        build_customer_decisions(
            transactions=(
                make_transactions()
            ),
            decision_date=(
                DECISION_DATE
            ),
            leakage_scores=(
                leakage_scores
            ),
        )


def test_engine_rejects_reactivation_scores_without_provenance():

    reactivation_scores = pd.DataFrame(
        {
            "customer": [
                "D",
            ],

            "reactivation_probability": [
                0.80,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Reactivation features are required "
            "when Reactivation scores are supplied"
        ),
    ):

        build_customer_decisions(
            transactions=(
                make_transactions()
            ),
            decision_date=(
                DECISION_DATE
            ),
            reactivation_scores=(
                reactivation_scores
            ),
        )