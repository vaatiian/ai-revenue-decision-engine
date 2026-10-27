"""
Cross-component validation for experiment records.

This module validates a run record against the currently governed production
score definition.

Important
---------
This validation is intended for run creation / current-production validation.

Historical JSONL records remain readable even if a future model version,
artifact, or manifest replaces the currently governed production capability.
"""

from __future__ import annotations

from src.experiments.provenance import (
    build_score_provenance,
)
from src.experiments.record import (
    ExperimentRunRecord,
)


# ================================================================
# GIT COMMIT VALIDATION
# ================================================================


def validate_git_commit_sha(
    git_commit: str,
) -> str:
    """
    Validate one full 40-character Git commit SHA.
    """

    if not isinstance(
        git_commit,
        str,
    ):
        raise TypeError(
            "git_commit must be a string."
        )

    if len(
        git_commit
    ) != 40:
        raise ValueError(
            "git_commit must contain exactly 40 hexadecimal characters."
        )

    try:

        int(
            git_commit,
            16,
        )

    except ValueError as exc:

        raise ValueError(
            "git_commit must contain exactly 40 hexadecimal characters."
        ) from exc

    return git_commit.lower()


# ================================================================
# CURRENT GOVERNED SCORE VALIDATION
# ================================================================


def validate_record_against_current_score(
    record: ExperimentRunRecord,
    *,
    verify_source_hashes: bool = True,
) -> None:
    """
    Validate one experiment record against current production governance.

    Checks:
    - registered score identity;
    - model name/version/type/target;
    - manifest path and actual SHA-256;
    - fitted artifact path and actual SHA-256 where required;
    - deterministic no-artifact contract;
    - full Git commit SHA syntax.

    The function returns None when the record is valid.
    """

    if not isinstance(
        record,
        ExperimentRunRecord,
    ):
        raise TypeError(
            "record must be an ExperimentRunRecord."
        )

    if not isinstance(
        verify_source_hashes,
        bool,
    ):
        raise TypeError(
            "verify_source_hashes must be a bool."
        )

    validate_git_commit_sha(
        record.git_commit
    )

    provenance = (
        build_score_provenance(
            record.score_id,
            verify_source_hashes=(
                verify_source_hashes
            ),
        )
    )

    expected = {
        "model_name":
            provenance.model_name,

        "model_version":
            provenance.model_version,

        "model_type":
            provenance.model_type,

        "target_name":
            provenance.target_name,

        "manifest_path":
            provenance.manifest_path,

        "manifest_sha256":
            provenance.manifest_sha256,

        "artifact_path":
            provenance.artifact_path,

        "artifact_sha256":
            provenance.artifact_sha256,
    }

    observed = {
        "model_name":
            record.model_name,

        "model_version":
            record.model_version,

        "model_type":
            record.model_type,

        "target_name":
            record.target_name,

        "manifest_path":
            record.manifest_path,

        "manifest_sha256":
            record.manifest_sha256,

        "artifact_path":
            record.artifact_path,

        "artifact_sha256":
            record.artifact_sha256,
    }

    mismatches = []

    for field in expected:

        if (
            observed[
                field
            ]
            !=
            expected[
                field
            ]
        ):

            mismatches.append(
                field
            )

    if mismatches:

        raise ValueError(
            "Experiment record does not match current governed "
            "score provenance. Mismatched fields: "
            +
            ", ".join(
                mismatches
            )
        )

    return None