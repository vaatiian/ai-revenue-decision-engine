"""
Production score provenance adapter for experiment logging.

This module bridges the production score registry and runtime availability
layer into the experiment logging subsystem.

Responsibilities
----------------
- resolve one canonical score_id;
- verify the registered score is currently AVAILABLE;
- capture stable model identity from the production registry;
- compute execution-time SHA-256 hashes for governed manifests/artifacts;
- support both fitted and deterministic production capabilities.

The adapter does not duplicate model-manifest contents. Governed manifests
remain the authoritative source of frozen model configuration and governance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from src.models.score_availability import (
    ScoreAvailabilityStatus,
    get_score_status,
)

from src.models.score_registry import (
    ScoreRegistryEntry,
    get_score_entry,
)


# ================================================================
# IMMUTABLE PROVENANCE CONTRACT
# ================================================================


@dataclass(
    frozen=True
)
class ScoreRunProvenance:
    """
    Immutable governed-model provenance for one experiment run.
    """

    score_id: str

    model_name: str
    model_version: str
    model_type: str
    target_name: str

    scorer_type: str
    deterministic: bool

    manifest_path: str
    manifest_sha256: str

    artifact_path: str | None
    artifact_sha256: str | None


# ================================================================
# HASHING
# ================================================================


def sha256_file(
    path: str | Path,
) -> str:
    """
    Return the SHA-256 digest of one existing regular file.
    """

    file_path = Path(
        path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"SHA-256 path must be a file: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024
                *
                1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ================================================================
# PATH HELPERS
# ================================================================


def _stable_path_string(
    path: Path,
) -> str:
    """
    Convert one repository path to stable forward-slash representation.
    """

    return path.as_posix()


# ================================================================
# ENTRY ADAPTER
# ================================================================


def build_entry_provenance(
    entry: ScoreRegistryEntry,
    *,
    verify_source_hashes: bool = True,
) -> ScoreRunProvenance:
    """
    Build governed experiment provenance from one registry entry.

    The entry must correspond to the canonical registry entry currently
    registered under its score_id.
    """

    if not isinstance(
        entry,
        ScoreRegistryEntry,
    ):
        raise TypeError(
            "entry must be a ScoreRegistryEntry."
        )

    if not isinstance(
        verify_source_hashes,
        bool,
    ):
        raise TypeError(
            "verify_source_hashes must be a bool."
        )

    canonical_entry = (
        get_score_entry(
            entry.score_id
        )
    )

    if (
        entry
        !=
        canonical_entry
    ):
        raise ValueError(
            "Experiment provenance requires the canonical registered "
            f"entry for score_id: {entry.score_id}"
        )

    status = (
        get_score_status(
            entry.score_id,
            verify_source_hashes=(
                verify_source_hashes
            ),
        )
    )

    if (
        status.status
        !=
        ScoreAvailabilityStatus.AVAILABLE
    ):
        raise RuntimeError(
            "Cannot build experiment provenance for unavailable score "
            f"{entry.score_id}: "
            f"{status.status.value}. "
            f"{status.detail}"
        )

    if not status.available:
        raise RuntimeError(
            "Score availability contract is inconsistent: "
            f"{entry.score_id} is not marked available."
        )

    if not entry.manifest_required:
        raise ValueError(
            "Experiment provenance requires a governed manifest."
        )

    if entry.manifest_path is None:
        raise ValueError(
            "Registered score is missing manifest_path."
        )

    manifest_path = Path(
        entry.manifest_path
    )

    manifest_hash = (
        sha256_file(
            manifest_path
        )
    )

    artifact_path_string = None
    artifact_hash = None

    if entry.artifact_required:

        if entry.artifact_path is None:
            raise ValueError(
                "Registered fitted score is missing artifact_path."
            )

        artifact_path = Path(
            entry.artifact_path
        )

        artifact_path_string = (
            _stable_path_string(
                artifact_path
            )
        )

        artifact_hash = (
            sha256_file(
                artifact_path
            )
        )

    else:

        if entry.artifact_path is not None:
            raise ValueError(
                "Deterministic score must not register an artifact_path."
            )

    return ScoreRunProvenance(
        score_id=(
            entry.score_id
        ),

        model_name=(
            entry.model_name
        ),

        model_version=(
            entry.model_version
        ),

        model_type=(
            entry.model_type
        ),

        target_name=(
            entry.target_name
        ),

        scorer_type=(
            entry.scorer_type
        ),

        deterministic=(
            entry.deterministic
        ),

        manifest_path=(
            _stable_path_string(
                manifest_path
            )
        ),

        manifest_sha256=(
            manifest_hash
        ),

        artifact_path=(
            artifact_path_string
        ),

        artifact_sha256=(
            artifact_hash
        ),
    )


# ================================================================
# PUBLIC SCORE-ID ADAPTER
# ================================================================


def build_score_provenance(
    score_id: str,
    *,
    verify_source_hashes: bool = True,
) -> ScoreRunProvenance:
    """
    Build experiment provenance for one canonical production score_id.
    """

    if not isinstance(
        verify_source_hashes,
        bool,
    ):
        raise TypeError(
            "verify_source_hashes must be a bool."
        )

    entry = (
        get_score_entry(
            score_id
        )
    )

    return build_entry_provenance(
        entry,
        verify_source_hashes=(
            verify_source_hashes
        ),
    )