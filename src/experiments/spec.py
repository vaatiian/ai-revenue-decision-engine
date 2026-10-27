"""
Frozen experiment-run contract.

This module defines the stable schema used by the production experiment
logging subsystem.

Design principles
-----------------
1. Experiment records describe execution events.
2. Governed model manifests remain the authoritative source of model
   governance, frozen research configuration, and approved-use metadata.
3. Experiment records reference governed artifacts and manifests rather
   than duplicating them as a second source of truth.
4. Both fitted and deterministic production capabilities are supported.
5. The contract is deliberately storage-backend agnostic.
"""

from __future__ import annotations


# ================================================================
# SCHEMA VERSION
# ================================================================


EXPERIMENT_SCHEMA_VERSION = "1.0.0"


# ================================================================
# RUN KINDS
# ================================================================


RUN_KIND_TRAINING = "training"

RUN_KIND_EVALUATION = "evaluation"

RUN_KIND_BACKTEST = "backtest"

RUN_KIND_PRODUCTION_VALIDATION = (
    "production_validation"
)


EXPERIMENT_RUN_KINDS = (
    RUN_KIND_TRAINING,
    RUN_KIND_EVALUATION,
    RUN_KIND_BACKTEST,
    RUN_KIND_PRODUCTION_VALIDATION,
)


# ================================================================
# RUN STATUS
# ================================================================


RUN_STATUS_RUNNING = "RUNNING"

RUN_STATUS_SUCCEEDED = "SUCCEEDED"

RUN_STATUS_FAILED = "FAILED"


EXPERIMENT_RUN_STATUSES = (
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    RUN_STATUS_FAILED,
)


# ================================================================
# RECORD SCHEMA
# ================================================================


EXPERIMENT_RUN_FIELDS = (
    # Schema
    "schema_version",

    # Run identity
    "run_id",
    "experiment_name",
    "run_kind",
    "status",

    # Production score/model identity
    "score_id",
    "model_name",
    "model_version",
    "model_type",
    "target_name",

    # Execution timing
    "started_at_utc",
    "completed_at_utc",

    # Source-control provenance
    "git_commit",
    "git_branch",

    # Governed model provenance
    "manifest_path",
    "manifest_sha256",
    "artifact_path",
    "artifact_sha256",

    # Data provenance
    "data_reference",
    "data_sha256",

    # Run-specific experiment information
    "parameters",
    "metrics",

    # Human / operational metadata
    "tags",
    "notes",
    "error_message",
)


# ================================================================
# REQUIRED FIELDS
# ================================================================


EXPERIMENT_REQUIRED_FIELDS = (
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
    "git_commit",
    "git_branch",
    "manifest_path",
    "manifest_sha256",
    "parameters",
    "metrics",
    "tags",
)


# ================================================================
# OPTIONAL FIELDS
# ================================================================


EXPERIMENT_OPTIONAL_FIELDS = (
    "completed_at_utc",
    "artifact_path",
    "artifact_sha256",
    "data_reference",
    "data_sha256",
    "notes",
    "error_message",
)


# ================================================================
# VALIDATION
# ================================================================


def validate_run_kind(
    run_kind: str,
) -> str:
    """
    Validate and return one frozen experiment run kind.
    """

    if not isinstance(
        run_kind,
        str,
    ):
        raise TypeError(
            "run_kind must be a string."
        )

    if not run_kind.strip():
        raise ValueError(
            "run_kind must not be empty."
        )

    if (
        run_kind
        not in
        EXPERIMENT_RUN_KINDS
    ):
        supported = ", ".join(
            EXPERIMENT_RUN_KINDS
        )

        raise ValueError(
            f"Unsupported run_kind: "
            f"{run_kind}. "
            f"Supported run kinds: "
            f"{supported}"
        )

    return run_kind


def validate_run_status(
    status: str,
) -> str:
    """
    Validate and return one frozen experiment run status.
    """

    if not isinstance(
        status,
        str,
    ):
        raise TypeError(
            "status must be a string."
        )

    if not status.strip():
        raise ValueError(
            "status must not be empty."
        )

    if (
        status
        not in
        EXPERIMENT_RUN_STATUSES
    ):
        supported = ", ".join(
            EXPERIMENT_RUN_STATUSES
        )

        raise ValueError(
            f"Unsupported experiment status: "
            f"{status}. "
            f"Supported statuses: "
            f"{supported}"
        )

    return status


def validate_experiment_spec() -> None:
    """
    Fail closed if the frozen experiment-run schema drifts internally.
    """

    if not EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            "Experiment schema version must not be empty."
        )

    if len(
        EXPERIMENT_RUN_FIELDS
    ) != len(
        set(
            EXPERIMENT_RUN_FIELDS
        )
    ):
        raise ValueError(
            "Experiment run fields must be unique."
        )

    if len(
        EXPERIMENT_REQUIRED_FIELDS
    ) != len(
        set(
            EXPERIMENT_REQUIRED_FIELDS
        )
    ):
        raise ValueError(
            "Experiment required fields must be unique."
        )

    if len(
        EXPERIMENT_OPTIONAL_FIELDS
    ) != len(
        set(
            EXPERIMENT_OPTIONAL_FIELDS
        )
    ):
        raise ValueError(
            "Experiment optional fields must be unique."
        )

    required = set(
        EXPERIMENT_REQUIRED_FIELDS
    )

    optional = set(
        EXPERIMENT_OPTIONAL_FIELDS
    )

    all_fields = set(
        EXPERIMENT_RUN_FIELDS
    )

    overlap = (
        required
        &
        optional
    )

    if overlap:
        raise ValueError(
            "Experiment fields cannot be both required and optional: "
            f"{sorted(overlap)}"
        )

    if (
        required
        |
        optional
    ) != all_fields:
        missing_classification = (
            all_fields
            -
            required
            -
            optional
        )

        unknown_classification = (
            (
                required
                |
                optional
            )
            -
            all_fields
        )

        raise ValueError(
            "Experiment required/optional field classification "
            "does not exactly match the frozen run schema. "
            f"Unclassified fields: "
            f"{sorted(missing_classification)}. "
            f"Unknown classified fields: "
            f"{sorted(unknown_classification)}."
        )

    if len(
        EXPERIMENT_RUN_KINDS
    ) != len(
        set(
            EXPERIMENT_RUN_KINDS
        )
    ):
        raise ValueError(
            "Experiment run kinds must be unique."
        )

    if len(
        EXPERIMENT_RUN_STATUSES
    ) != len(
        set(
            EXPERIMENT_RUN_STATUSES
        )
    ):
        raise ValueError(
            "Experiment run statuses must be unique."
        )

    for run_kind in (
        EXPERIMENT_RUN_KINDS
    ):
        validate_run_kind(
            run_kind
        )

    for status in (
        EXPERIMENT_RUN_STATUSES
    ):
        validate_run_status(
            status
        )


validate_experiment_spec()