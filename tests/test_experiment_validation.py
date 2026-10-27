"""
Tests for fail-closed experiment governance validation.
"""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.experiments.provenance import (
    build_score_provenance,
)
from src.experiments.record import (
    ExperimentRunRecord,
)
from src.experiments.spec import (
    EXPERIMENT_SCHEMA_VERSION,
    RUN_KIND_PRODUCTION_VALIDATION,
    RUN_STATUS_SUCCEEDED,
)
from src.experiments.validation import (
    validate_git_commit_sha,
    validate_record_against_current_score,
)


GIT_COMMIT = (
    "61424ff89ecf21325f7f7d10fee191fff5c1a804"
)


def make_current_record(
    score_id="reactivation",
):

    provenance = (
        build_score_provenance(
            score_id
        )
    )

    return ExperimentRunRecord(
        schema_version=(
            EXPERIMENT_SCHEMA_VERSION
        ),

        run_id=(
            f"{score_id}-validation-001"
        ),

        experiment_name=(
            f"{score_id}-production-validation"
        ),

        run_kind=(
            RUN_KIND_PRODUCTION_VALIDATION
        ),

        status=(
            RUN_STATUS_SUCCEEDED
        ),

        score_id=(
            provenance.score_id
        ),

        model_name=(
            provenance.model_name
        ),

        model_version=(
            provenance.model_version
        ),

        model_type=(
            provenance.model_type
        ),

        target_name=(
            provenance.target_name
        ),

        started_at_utc=(
            datetime(
                2026,
                10,
                27,
                12,
                0,
                tzinfo=timezone.utc,
            )
        ),

        completed_at_utc=(
            datetime(
                2026,
                10,
                27,
                12,
                1,
                tzinfo=timezone.utc,
            )
        ),

        git_commit=(
            GIT_COMMIT
        ),

        git_branch=(
            "main"
        ),

        manifest_path=(
            provenance.manifest_path
        ),

        manifest_sha256=(
            provenance.manifest_sha256
        ),

        artifact_path=(
            provenance.artifact_path
        ),

        artifact_sha256=(
            provenance.artifact_sha256
        ),

        data_reference=None,

        data_sha256=None,

        parameters={},

        metrics={},

        tags=(
            "production-validation",
        ),

        notes=None,

        error_message=None,
    )


# ================================================================
# GIT COMMIT CONTRACT
# ================================================================


def test_full_git_commit_sha_is_accepted():

    assert (
        validate_git_commit_sha(
            GIT_COMMIT
        )
        ==
        GIT_COMMIT
    )


def test_uppercase_git_commit_is_normalised():

    assert (
        validate_git_commit_sha(
            GIT_COMMIT.upper()
        )
        ==
        GIT_COMMIT
    )


def test_short_git_commit_fails_closed():

    with pytest.raises(
        ValueError,
        match="40 hexadecimal",
    ):

        validate_git_commit_sha(
            "61424ff"
        )


def test_non_hex_git_commit_fails_closed():

    with pytest.raises(
        ValueError,
        match="40 hexadecimal",
    ):

        validate_git_commit_sha(
            "z"
            *
            40
        )


def test_non_string_git_commit_fails_closed():

    with pytest.raises(
        TypeError,
        match="git_commit must be a string",
    ):

        validate_git_commit_sha(
            123
        )


# ================================================================
# VALID CURRENT PROVENANCE
# ================================================================


@pytest.mark.parametrize(
    "score_id",
    (
        "reactivation",
        "revenue_leakage",
        "future_customer_value",
        "repeat_purchase",
        "recommendation",
        "demand_forecast",
    ),
)
def test_all_six_current_scores_validate(
    score_id,
):

    record = (
        make_current_record(
            score_id
        )
    )

    assert (
        validate_record_against_current_score(
            record
        )
        is None
    )


# ================================================================
# MODEL IDENTITY DRIFT
# ================================================================


@pytest.mark.parametrize(
    "field,value",
    (
        (
            "model_name",
            "Wrong Model",
        ),
        (
            "model_version",
            "9.9.9",
        ),
        (
            "model_type",
            "wrong_type",
        ),
        (
            "target_name",
            "wrong_target",
        ),
    ),
)
def test_model_identity_drift_fails_closed(
    field,
    value,
):

    record = (
        make_current_record()
    )

    broken = replace(
        record,
        **{
            field:
                value
        },
    )

    with pytest.raises(
        ValueError,
        match="Mismatched fields",
    ):

        validate_record_against_current_score(
            broken
        )


# ================================================================
# MANIFEST DRIFT
# ================================================================


def test_manifest_path_drift_fails_closed():

    record = (
        make_current_record()
    )

    broken = replace(
        record,
        manifest_path=(
            "artifacts/models/reactivation/"
            "wrong_manifest.json"
        ),
    )

    with pytest.raises(
        ValueError,
        match="manifest_path",
    ):

        validate_record_against_current_score(
            broken
        )


def test_manifest_hash_drift_fails_closed():

    record = (
        make_current_record()
    )

    broken = replace(
        record,
        manifest_sha256=(
            "f"
            *
            64
        ),
    )

    with pytest.raises(
        ValueError,
        match="manifest_sha256",
    ):

        validate_record_against_current_score(
            broken
        )


# ================================================================
# ARTIFACT DRIFT
# ================================================================


def test_fitted_artifact_path_drift_fails_closed():

    record = (
        make_current_record(
            "reactivation"
        )
    )

    broken = replace(
        record,
        artifact_path=(
            "artifacts/models/reactivation/"
            "wrong_model.joblib"
        ),
    )

    with pytest.raises(
        ValueError,
        match="artifact_path",
    ):

        validate_record_against_current_score(
            broken
        )


def test_fitted_artifact_hash_drift_fails_closed():

    record = (
        make_current_record(
            "reactivation"
        )
    )

    broken = replace(
        record,
        artifact_sha256=(
            "f"
            *
            64
        ),
    )

    with pytest.raises(
        ValueError,
        match="artifact_sha256",
    ):

        validate_record_against_current_score(
            broken
        )


def test_deterministic_record_validates_without_artifact():

    record = (
        make_current_record(
            "recommendation"
        )
    )

    assert record.artifact_path is None
    assert record.artifact_sha256 is None

    assert (
        validate_record_against_current_score(
            record
        )
        is None
    )


# ================================================================
# API VALIDATION
# ================================================================


def test_wrong_record_type_fails_closed():

    with pytest.raises(
        TypeError,
        match="ExperimentRunRecord",
    ):

        validate_record_against_current_score(
            {}
        )


def test_verify_source_hashes_must_be_bool():

    record = (
        make_current_record()
    )

    with pytest.raises(
        TypeError,
        match="verify_source_hashes must be a bool",
    ):

        validate_record_against_current_score(
            record,
            verify_source_hashes="yes",
        )


def test_unknown_score_id_fails_closed():

    record = (
        make_current_record()
    )

    broken = replace(
        record,
        score_id=(
            "unknown_score"
        ),
    )

    with pytest.raises(
        KeyError,
        match="Unknown score_id",
    ):

        validate_record_against_current_score(
            broken
        )