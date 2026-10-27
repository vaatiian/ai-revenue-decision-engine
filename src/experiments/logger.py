"""
Integrated production experiment logger.

This module composes:
- production score provenance;
- immutable experiment records;
- current-governance validation;
- append-only JSONL persistence.

It is the primary public entry point for logging governed experiment runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.experiments.provenance import (
    build_score_provenance,
)
from src.experiments.record import (
    ExperimentRunRecord,
)
from src.experiments.spec import (
    EXPERIMENT_SCHEMA_VERSION,
)
from src.experiments.store import (
    DEFAULT_EXPERIMENT_STORE_PATH,
    append_experiment_run,
)
from src.experiments.validation import (
    validate_record_against_current_score,
)


# ================================================================
# RUN-ID GENERATION
# ================================================================


def generate_run_id() -> str:
    """
    Generate one globally unique experiment run identifier.
    """

    return str(
        uuid4()
    )


# ================================================================
# RECORD BUILDING
# ================================================================


def build_experiment_run(
    *,
    score_id: str,
    experiment_name: str,
    run_kind: str,
    status: str,
    started_at_utc,
    completed_at_utc,
    git_commit: str,
    git_branch: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any],
    tags,
    run_id: str | None = None,
    data_reference: str | None = None,
    data_sha256: str | None = None,
    notes: str | None = None,
    error_message: str | None = None,
    verify_source_hashes: bool = True,
) -> ExperimentRunRecord:
    """
    Build and validate one governed production experiment record.

    Model identity and artifact/manifest provenance come exclusively from
    the production score registry and availability layer.
    """

    if not isinstance(
        verify_source_hashes,
        bool,
    ):
        raise TypeError(
            "verify_source_hashes must be a bool."
        )

    provenance = (
        build_score_provenance(
            score_id,
            verify_source_hashes=(
                verify_source_hashes
            ),
        )
    )

    resolved_run_id = (
        generate_run_id()
        if run_id is None
        else run_id
    )

    record = ExperimentRunRecord(
        schema_version=(
            EXPERIMENT_SCHEMA_VERSION
        ),

        run_id=(
            resolved_run_id
        ),

        experiment_name=(
            experiment_name
        ),

        run_kind=(
            run_kind
        ),

        status=(
            status
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
            started_at_utc
        ),

        completed_at_utc=(
            completed_at_utc
        ),

        git_commit=(
            git_commit
        ),

        git_branch=(
            git_branch
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

        data_reference=(
            data_reference
        ),

        data_sha256=(
            data_sha256
        ),

        parameters=(
            parameters
        ),

        metrics=(
            metrics
        ),

        tags=(
            tags
        ),

        notes=(
            notes
        ),

        error_message=(
            error_message
        ),
    )

    validate_record_against_current_score(
        record,
        verify_source_hashes=(
            verify_source_hashes
        ),
    )

    return record


# ================================================================
# BUILD + PERSIST
# ================================================================


def log_experiment_run(
    *,
    score_id: str,
    experiment_name: str,
    run_kind: str,
    status: str,
    started_at_utc,
    completed_at_utc,
    git_commit: str,
    git_branch: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any],
    tags,
    run_id: str | None = None,
    data_reference: str | None = None,
    data_sha256: str | None = None,
    notes: str | None = None,
    error_message: str | None = None,
    verify_source_hashes: bool = True,
    store_path: str | Path = (
        DEFAULT_EXPERIMENT_STORE_PATH
    ),
) -> ExperimentRunRecord:
    """
    Build, validate, persist, and return one experiment run.
    """

    record = (
        build_experiment_run(
            score_id=(
                score_id
            ),
            experiment_name=(
                experiment_name
            ),
            run_kind=(
                run_kind
            ),
            status=(
                status
            ),
            started_at_utc=(
                started_at_utc
            ),
            completed_at_utc=(
                completed_at_utc
            ),
            git_commit=(
                git_commit
            ),
            git_branch=(
                git_branch
            ),
            parameters=(
                parameters
            ),
            metrics=(
                metrics
            ),
            tags=(
                tags
            ),
            run_id=(
                run_id
            ),
            data_reference=(
                data_reference
            ),
            data_sha256=(
                data_sha256
            ),
            notes=(
                notes
            ),
            error_message=(
                error_message
            ),
            verify_source_hashes=(
                verify_source_hashes
            ),
        )
    )

    append_experiment_run(
        record,
        path=(
            store_path
        ),
    )

    return record