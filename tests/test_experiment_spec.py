"""
Contract tests for the frozen experiment logging specification.
"""

import pytest

from src.experiments.spec import (
    EXPERIMENT_OPTIONAL_FIELDS,
    EXPERIMENT_REQUIRED_FIELDS,
    EXPERIMENT_RUN_FIELDS,
    EXPERIMENT_RUN_KINDS,
    EXPERIMENT_RUN_STATUSES,
    EXPERIMENT_SCHEMA_VERSION,
    RUN_KIND_BACKTEST,
    RUN_KIND_EVALUATION,
    RUN_KIND_PRODUCTION_VALIDATION,
    RUN_KIND_TRAINING,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    validate_experiment_spec,
    validate_run_kind,
    validate_run_status,
)


# ================================================================
# VERSION
# ================================================================


def test_experiment_schema_version_is_frozen():

    assert (
        EXPERIMENT_SCHEMA_VERSION
        ==
        "1.0.0"
    )


# ================================================================
# RUN KINDS
# ================================================================


def test_run_kinds_are_frozen():

    assert (
        EXPERIMENT_RUN_KINDS
        ==
        (
            RUN_KIND_TRAINING,
            RUN_KIND_EVALUATION,
            RUN_KIND_BACKTEST,
            RUN_KIND_PRODUCTION_VALIDATION,
        )
    )

    assert (
        EXPERIMENT_RUN_KINDS
        ==
        (
            "training",
            "evaluation",
            "backtest",
            "production_validation",
        )
    )


@pytest.mark.parametrize(
    "run_kind",
    EXPERIMENT_RUN_KINDS,
)
def test_supported_run_kind_is_accepted(
    run_kind,
):

    assert (
        validate_run_kind(
            run_kind
        )
        ==
        run_kind
    )


def test_unknown_run_kind_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported run_kind",
    ):

        validate_run_kind(
            "unknown"
        )


def test_non_string_run_kind_fails_closed():

    with pytest.raises(
        TypeError,
        match="run_kind must be a string",
    ):

        validate_run_kind(
            123
        )


# ================================================================
# RUN STATUS
# ================================================================


def test_run_statuses_are_frozen():

    assert (
        EXPERIMENT_RUN_STATUSES
        ==
        (
            RUN_STATUS_RUNNING,
            RUN_STATUS_SUCCEEDED,
            RUN_STATUS_FAILED,
        )
    )

    assert (
        EXPERIMENT_RUN_STATUSES
        ==
        (
            "RUNNING",
            "SUCCEEDED",
            "FAILED",
        )
    )


@pytest.mark.parametrize(
    "status",
    EXPERIMENT_RUN_STATUSES,
)
def test_supported_status_is_accepted(
    status,
):

    assert (
        validate_run_status(
            status
        )
        ==
        status
    )


def test_unknown_status_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported experiment status",
    ):

        validate_run_status(
            "UNKNOWN"
        )


def test_non_string_status_fails_closed():

    with pytest.raises(
        TypeError,
        match="status must be a string",
    ):

        validate_run_status(
            None
        )


# ================================================================
# RECORD SCHEMA
# ================================================================


def test_run_fields_are_exact():

    assert (
        EXPERIMENT_RUN_FIELDS
        ==
        (
            "schema_version",
            "run_id",
            "experiment_name",
            "run_kind",
            "status",
            "score_id",
            "model_name",
            "model_version",
            "model_type",
            "target_name",
            "started_at_utc",
            "completed_at_utc",
            "git_commit",
            "git_branch",
            "manifest_path",
            "manifest_sha256",
            "artifact_path",
            "artifact_sha256",
            "data_reference",
            "data_sha256",
            "parameters",
            "metrics",
            "tags",
            "notes",
            "error_message",
        )
    )


def test_required_and_optional_fields_partition_schema():

    required = set(
        EXPERIMENT_REQUIRED_FIELDS
    )

    optional = set(
        EXPERIMENT_OPTIONAL_FIELDS
    )

    assert not (
        required
        &
        optional
    )

    assert (
        required
        |
        optional
    ) == set(
        EXPERIMENT_RUN_FIELDS
    )


def test_fitted_and_deterministic_artifact_contract_is_supported():

    assert (
        "manifest_path"
        in
        EXPERIMENT_REQUIRED_FIELDS
    )

    assert (
        "manifest_sha256"
        in
        EXPERIMENT_REQUIRED_FIELDS
    )

    assert (
        "artifact_path"
        in
        EXPERIMENT_OPTIONAL_FIELDS
    )

    assert (
        "artifact_sha256"
        in
        EXPERIMENT_OPTIONAL_FIELDS
    )


def test_data_provenance_is_optional_but_supported():

    assert (
        "data_reference"
        in
        EXPERIMENT_OPTIONAL_FIELDS
    )

    assert (
        "data_sha256"
        in
        EXPERIMENT_OPTIONAL_FIELDS
    )


def test_run_specific_parameters_and_metrics_are_required():

    assert (
        "parameters"
        in
        EXPERIMENT_REQUIRED_FIELDS
    )

    assert (
        "metrics"
        in
        EXPERIMENT_REQUIRED_FIELDS
    )


def test_spec_self_validation_passes():

    assert (
        validate_experiment_spec()
        is None
    )