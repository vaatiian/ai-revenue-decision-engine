"""
Immutable experiment-run record.

This module turns the frozen experiment specification into a validated,
deeply immutable runtime object.

The record stores execution metadata only. Governed model manifests remain
the authoritative source for frozen model configuration and governance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from src.experiments.spec import (
    EXPERIMENT_RUN_FIELDS,
    EXPERIMENT_SCHEMA_VERSION,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    validate_run_kind,
    validate_run_status,
)


# ================================================================
# VALIDATION HELPERS
# ================================================================


def _validate_non_empty_string(
    value,
    name: str,
) -> str:
    """
    Validate and return one non-empty string.
    """

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{name} must not be empty."
        )

    return value


def _validate_optional_string(
    value,
    name: str,
):
    """
    Validate an optional string.
    """

    if value is None:
        return None

    return _validate_non_empty_string(
        value,
        name,
    )


def _normalise_utc_timestamp(
    value,
    name: str,
):
    """
    Convert one timestamp-like value to a timezone-aware UTC datetime.
    """

    if value is None:
        return None

    try:

        timestamp = pd.Timestamp(
            value
        )

    except Exception as exc:

        raise ValueError(
            f"{name} is not a valid timestamp."
        ) from exc

    if pd.isna(
        timestamp
    ):
        raise ValueError(
            f"{name} is not a valid timestamp."
        )

    if timestamp.tzinfo is None:

        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return timestamp.to_pydatetime()


def _validate_sha256(
    value,
    name: str,
):
    """
    Validate one optional lowercase/uppercase SHA-256 hexadecimal digest.
    """

    if value is None:
        return None

    value = _validate_non_empty_string(
        value,
        name,
    )

    if len(
        value
    ) != 64:
        raise ValueError(
            f"{name} must contain exactly 64 hexadecimal characters."
        )

    try:

        int(
            value,
            16,
        )

    except ValueError as exc:

        raise ValueError(
            f"{name} must contain exactly 64 hexadecimal characters."
        ) from exc

    return value.lower()


def _freeze_mapping(
    value,
    name: str,
) -> Mapping[str, Any]:
    """
    Copy and freeze one string-keyed mapping.
    """

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    copied = dict(
        value
    )

    invalid_keys = [
        key
        for key
        in copied
        if not isinstance(
            key,
            str,
        )
    ]

    if invalid_keys:
        raise TypeError(
            f"{name} keys must be strings."
        )

    return MappingProxyType(
        copied
    )


def _freeze_tags(
    value,
) -> tuple[str, ...]:
    """
    Validate and freeze experiment tags.
    """

    if isinstance(
        value,
        str,
    ):
        raise TypeError(
            "tags must be an iterable of strings, not a single string."
        )

    try:

        tags = tuple(
            value
        )

    except TypeError as exc:

        raise TypeError(
            "tags must be an iterable of strings."
        ) from exc

    for tag in tags:

        _validate_non_empty_string(
            tag,
            "tag",
        )

    if len(
        tags
    ) != len(
        set(
            tags
        )
    ):
        raise ValueError(
            "tags must not contain duplicates."
        )

    return tags


# ================================================================
# IMMUTABLE RECORD
# ================================================================


@dataclass(
    frozen=True
)
class ExperimentRunRecord:
    """
    One immutable experiment execution record.
    """

    schema_version: str

    run_id: str
    experiment_name: str
    run_kind: str
    status: str

    score_id: str
    model_name: str
    model_version: str
    model_type: str
    target_name: str

    started_at_utc: datetime
    completed_at_utc: datetime | None

    git_commit: str
    git_branch: str

    manifest_path: str
    manifest_sha256: str

    artifact_path: str | None
    artifact_sha256: str | None

    data_reference: str | None
    data_sha256: str | None

    parameters: Mapping[str, Any]
    metrics: Mapping[str, Any]

    tags: tuple[str, ...]

    notes: str | None
    error_message: str | None

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and deeply freeze mutable inputs.
        """

        if (
            self.schema_version
            !=
            EXPERIMENT_SCHEMA_VERSION
        ):
            raise ValueError(
                "Experiment record schema_version does not match "
                "EXPERIMENT_SCHEMA_VERSION."
            )

        for (
            name,
            value,
        ) in (
            (
                "run_id",
                self.run_id,
            ),
            (
                "experiment_name",
                self.experiment_name,
            ),
            (
                "score_id",
                self.score_id,
            ),
            (
                "model_name",
                self.model_name,
            ),
            (
                "model_version",
                self.model_version,
            ),
            (
                "model_type",
                self.model_type,
            ),
            (
                "target_name",
                self.target_name,
            ),
            (
                "git_commit",
                self.git_commit,
            ),
            (
                "git_branch",
                self.git_branch,
            ),
            (
                "manifest_path",
                self.manifest_path,
            ),
        ):

            _validate_non_empty_string(
                value,
                name,
            )

        validate_run_kind(
            self.run_kind
        )

        validate_run_status(
            self.status
        )

        started = (
            _normalise_utc_timestamp(
                self.started_at_utc,
                "started_at_utc",
            )
        )

        completed = (
            _normalise_utc_timestamp(
                self.completed_at_utc,
                "completed_at_utc",
            )
        )

        object.__setattr__(
            self,
            "started_at_utc",
            started,
        )

        object.__setattr__(
            self,
            "completed_at_utc",
            completed,
        )

        if (
            completed is not None
            and
            completed
            <
            started
        ):
            raise ValueError(
                "completed_at_utc cannot be earlier than started_at_utc."
            )

        if (
            self.status
            ==
            RUN_STATUS_RUNNING
            and
            completed is not None
        ):
            raise ValueError(
                "RUNNING experiment records cannot have completed_at_utc."
            )

        if (
            self.status
            in (
                RUN_STATUS_SUCCEEDED,
                RUN_STATUS_FAILED,
            )
            and
            completed is None
        ):
            raise ValueError(
                "Completed experiment records must have completed_at_utc."
            )

        manifest_hash = (
            _validate_sha256(
                self.manifest_sha256,
                "manifest_sha256",
            )
        )

        object.__setattr__(
            self,
            "manifest_sha256",
            manifest_hash,
        )

        artifact_path = (
            _validate_optional_string(
                self.artifact_path,
                "artifact_path",
            )
        )

        artifact_hash = (
            _validate_sha256(
                self.artifact_sha256,
                "artifact_sha256",
            )
        )

        if (
            artifact_path is None
        ) != (
            artifact_hash is None
        ):
            raise ValueError(
                "artifact_path and artifact_sha256 must either both be "
                "provided or both be None."
            )

        object.__setattr__(
            self,
            "artifact_path",
            artifact_path,
        )

        object.__setattr__(
            self,
            "artifact_sha256",
            artifact_hash,
        )

        data_reference = (
            _validate_optional_string(
                self.data_reference,
                "data_reference",
            )
        )

        data_hash = (
            _validate_sha256(
                self.data_sha256,
                "data_sha256",
            )
        )

        if (
            data_hash is not None
            and
            data_reference is None
        ):
            raise ValueError(
                "data_sha256 requires data_reference."
            )

        object.__setattr__(
            self,
            "data_reference",
            data_reference,
        )

        object.__setattr__(
            self,
            "data_sha256",
            data_hash,
        )

        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(
                self.parameters,
                "parameters",
            ),
        )

        object.__setattr__(
            self,
            "metrics",
            _freeze_mapping(
                self.metrics,
                "metrics",
            ),
        )

        object.__setattr__(
            self,
            "tags",
            _freeze_tags(
                self.tags
            ),
        )

        object.__setattr__(
            self,
            "notes",
            _validate_optional_string(
                self.notes,
                "notes",
            ),
        )

        object.__setattr__(
            self,
            "error_message",
            _validate_optional_string(
                self.error_message,
                "error_message",
            ),
        )

        if (
            self.status
            ==
            RUN_STATUS_FAILED
            and
            self.error_message is None
        ):
            raise ValueError(
                "FAILED experiment records require error_message."
            )

        if (
            self.status
            !=
            RUN_STATUS_FAILED
            and
            self.error_message is not None
        ):
            raise ValueError(
                "Only FAILED experiment records may contain error_message."
            )


# ================================================================
# SERIALISABLE REPRESENTATION
# ================================================================


def experiment_run_to_dict(
    record: ExperimentRunRecord,
) -> dict[str, Any]:
    """
    Convert one validated immutable run record to a JSON-ready dictionary.

    JSON encoding itself belongs to the storage layer.
    """

    if not isinstance(
        record,
        ExperimentRunRecord,
    ):
        raise TypeError(
            "record must be an ExperimentRunRecord."
        )

    values = {
        "schema_version":
            record.schema_version,

        "run_id":
            record.run_id,

        "experiment_name":
            record.experiment_name,

        "run_kind":
            record.run_kind,

        "status":
            record.status,

        "score_id":
            record.score_id,

        "model_name":
            record.model_name,

        "model_version":
            record.model_version,

        "model_type":
            record.model_type,

        "target_name":
            record.target_name,

        "started_at_utc":
            record.started_at_utc.isoformat(),

        "completed_at_utc":
            (
                record.completed_at_utc.isoformat()
                if record.completed_at_utc
                is not None
                else None
            ),

        "git_commit":
            record.git_commit,

        "git_branch":
            record.git_branch,

        "manifest_path":
            record.manifest_path,

        "manifest_sha256":
            record.manifest_sha256,

        "artifact_path":
            record.artifact_path,

        "artifact_sha256":
            record.artifact_sha256,

        "data_reference":
            record.data_reference,

        "data_sha256":
            record.data_sha256,

        "parameters":
            dict(
                record.parameters
            ),

        "metrics":
            dict(
                record.metrics
            ),

        "tags":
            list(
                record.tags
            ),

        "notes":
            record.notes,

        "error_message":
            record.error_message,
    }

    if tuple(
        values.keys()
    ) != EXPERIMENT_RUN_FIELDS:
        raise RuntimeError(
            "Experiment run dictionary does not match "
            "EXPERIMENT_RUN_FIELDS."
        )

    return values