"""
Runtime availability and health checks for production scores.

The static score registry describes what scoring capabilities exist.

This module answers a separate runtime question:

    Can this governed score run correctly now?

Availability checks are read-only. They do not train models, modify
manifests, regenerate artifacts, or execute scoring predictions.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.models.score_registry import (
    ScoreRegistryEntry,
    get_score_entry,
    list_score_entries,
)

from src.models.reactivation_scorer import (
    load_reactivation_artifact,
    load_reactivation_manifest,
    validate_reactivation_artifact,
    validate_reactivation_manifest,
)

from src.models.leakage_scorer import (
    load_leakage_artifact,
    load_leakage_manifest,
    validate_leakage_artifact,
    validate_leakage_manifest,
)

from src.models.fcv_scorer import (
    load_fcv_artifact,
    load_fcv_manifest,
    validate_fcv_artifact,
    validate_fcv_manifest,
)

from src.models.recommendation_manifest import (
    load_recommendation_manifest,
    validate_recommendation_manifest,
)

from src.models.demand_manifest import (
    load_demand_manifest,
    validate_demand_manifest,
)

from src.models.repeat_purchase_manifest import (
    load_repeat_purchase_manifest,
    validate_repeat_purchase_manifest,
)

from src.models.repeat_purchase_scorer import (
    RepeatPurchaseScorer,
    load_repeat_purchase_model,
)


# ================================================================
# STATUS ENUM
# ================================================================


class ScoreAvailabilityStatus(
    str,
    Enum,
):
    """
    Governed runtime state for one score.
    """

    AVAILABLE = "AVAILABLE"

    MISSING_ARTIFACT = (
        "MISSING_ARTIFACT"
    )

    MISSING_MANIFEST = (
        "MISSING_MANIFEST"
    )

    INVALID_ARTIFACT = (
        "INVALID_ARTIFACT"
    )

    INVALID_MANIFEST = (
        "INVALID_MANIFEST"
    )

    UNAVAILABLE = (
        "UNAVAILABLE"
    )


# ================================================================
# STATUS RESULT
# ================================================================


@dataclass(frozen=True)
class ScoreAvailability:
    """
    Immutable runtime health result for one registered score.
    """

    score_id: str

    status: ScoreAvailabilityStatus

    available: bool

    artifact_exists: bool | None

    manifest_exists: bool | None

    detail: str


# ================================================================
# RESULT HELPERS
# ================================================================


def _result(
    entry: ScoreRegistryEntry,
    status: ScoreAvailabilityStatus,
    detail: str,
    artifact_exists: bool | None,
    manifest_exists: bool | None,
) -> ScoreAvailability:
    """
    Build one immutable availability result.
    """

    return ScoreAvailability(
        score_id=entry.score_id,
        status=status,
        available=(
            status
            ==
            ScoreAvailabilityStatus.AVAILABLE
        ),
        artifact_exists=artifact_exists,
        manifest_exists=manifest_exists,
        detail=detail,
    )


def _path_exists(
    path: Path | None,
) -> bool:
    """
    Return True only for an existing filesystem file.
    """

    if path is None:
        return False

    return Path(
        path
    ).is_file()


# ================================================================
# ARTIFACT VALIDATION DISPATCH
# ================================================================


def _validate_artifact(
    entry: ScoreRegistryEntry,
) -> None:
    """
    Validate one fitted score's serialized artifact.

    Manifest validation is deliberately separate so runtime health can
    distinguish INVALID_ARTIFACT from INVALID_MANIFEST.
    """

    if (
        entry.score_id
        ==
        "reactivation"
    ):

        model = (
            load_reactivation_artifact(
                entry.artifact_path
            )
        )

        validate_reactivation_artifact(
            model
        )

        return

    if (
        entry.score_id
        ==
        "revenue_leakage"
    ):

        model = (
            load_leakage_artifact(
                entry.artifact_path
            )
        )

        validate_leakage_artifact(
            model
        )

        return

    if (
        entry.score_id
        ==
        "future_customer_value"
    ):

        model = (
            load_fcv_artifact(
                entry.artifact_path
            )
        )

        validate_fcv_artifact(
            model
        )

        return

    if (
        entry.score_id
        ==
        "repeat_purchase"
    ):

        model = (
            load_repeat_purchase_model(
                entry.artifact_path
            )
        )

        RepeatPurchaseScorer(
            model=model
        )

        return

    raise ValueError(
        "No fitted artifact validator is registered for "
        f"{entry.score_id}."
    )


# ================================================================
# MANIFEST VALIDATION DISPATCH
# ================================================================


def _validate_manifest(
    entry: ScoreRegistryEntry,
    verify_source_hashes: bool,
) -> None:
    """
    Validate one score's governance manifest.
    """

    if (
        entry.score_id
        ==
        "reactivation"
    ):

        manifest = (
            load_reactivation_manifest(
                entry.manifest_path
            )
        )

        validate_reactivation_manifest(
            manifest,
            artifact_path=(
                entry.artifact_path
            ),
        )

        return

    if (
        entry.score_id
        ==
        "revenue_leakage"
    ):

        manifest = (
            load_leakage_manifest(
                entry.manifest_path
            )
        )

        validate_leakage_manifest(
            manifest,
            artifact_path=(
                entry.artifact_path
            ),
        )

        return

    if (
        entry.score_id
        ==
        "future_customer_value"
    ):

        manifest = (
            load_fcv_manifest(
                entry.manifest_path
            )
        )

        validate_fcv_manifest(
            manifest,
            artifact_path=(
                entry.artifact_path
            ),
        )

        return

    if (
        entry.score_id
        ==
        "repeat_purchase"
    ):

        manifest = (
            load_repeat_purchase_manifest(
                entry.manifest_path
            )
        )

        validate_repeat_purchase_manifest(
            manifest=manifest,
            artifact_path=(
                entry.artifact_path
            ),
        )

        return

    if (
        entry.score_id
        ==
        "recommendation"
    ):

        manifest = (
            load_recommendation_manifest(
                entry.manifest_path
            )
        )

        validate_recommendation_manifest(
            manifest,
            verify_source_hashes=(
                verify_source_hashes
            ),
        )

        return

    if (
        entry.score_id
        ==
        "demand_forecast"
    ):

        manifest = (
            load_demand_manifest(
                entry.manifest_path
            )
        )

        validate_demand_manifest(
            manifest,
            verify_source_hashes=(
                verify_source_hashes
            ),
        )

        return

    raise ValueError(
        "No manifest validator is registered for "
        f"{entry.score_id}."
    )


# ================================================================
# AVAILABILITY EVALUATION
# ================================================================


def evaluate_score_availability(
    entry: ScoreRegistryEntry,
    verify_source_hashes: bool = True,
) -> ScoreAvailability:
    """
    Evaluate runtime availability for one registry entry.

    Validation order:
    1. Required manifest exists.
    2. Required artifact exists.
    3. Artifact loads and satisfies its frozen production contract.
    4. Manifest loads and satisfies its governance contract.

    Deterministic scores skip the artifact stage.
    """

    if not isinstance(
        entry,
        ScoreRegistryEntry,
    ):
        raise TypeError(
            "entry must be a ScoreRegistryEntry."
        )

    manifest_exists = (
        _path_exists(
            entry.manifest_path
        )
        if entry.manifest_required
        else None
    )

    artifact_exists = (
        _path_exists(
            entry.artifact_path
        )
        if entry.artifact_required
        else None
    )

    # ------------------------------------------------------------
    # MANIFEST EXISTS
    # ------------------------------------------------------------

    if (
        entry.manifest_required
        and
        not manifest_exists
    ):

        return _result(
            entry=entry,
            status=(
                ScoreAvailabilityStatus
                .MISSING_MANIFEST
            ),
            detail=(
                "Required governance manifest is missing."
            ),
            artifact_exists=(
                artifact_exists
            ),
            manifest_exists=False,
        )

    # ------------------------------------------------------------
    # ARTIFACT EXISTS
    # ------------------------------------------------------------

    if (
        entry.artifact_required
        and
        not artifact_exists
    ):

        return _result(
            entry=entry,
            status=(
                ScoreAvailabilityStatus
                .MISSING_ARTIFACT
            ),
            detail=(
                "Required serialized model artifact is missing."
            ),
            artifact_exists=False,
            manifest_exists=(
                manifest_exists
            ),
        )

    # ------------------------------------------------------------
    # ARTIFACT CONTRACT
    # ------------------------------------------------------------

    if entry.artifact_required:

        try:

            _validate_artifact(
                entry
            )

        except Exception as exc:

            return _result(
                entry=entry,
                status=(
                    ScoreAvailabilityStatus
                    .INVALID_ARTIFACT
                ),
                detail=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                artifact_exists=True,
                manifest_exists=(
                    manifest_exists
                ),
            )

    # ------------------------------------------------------------
    # MANIFEST CONTRACT
    # ------------------------------------------------------------

    if entry.manifest_required:

        try:

            _validate_manifest(
                entry=entry,
                verify_source_hashes=(
                    verify_source_hashes
                ),
            )

        except Exception as exc:

            return _result(
                entry=entry,
                status=(
                    ScoreAvailabilityStatus
                    .INVALID_MANIFEST
                ),
                detail=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                artifact_exists=(
                    artifact_exists
                ),
                manifest_exists=True,
            )

    # ------------------------------------------------------------
    # AVAILABLE
    # ------------------------------------------------------------

    return _result(
        entry=entry,
        status=(
            ScoreAvailabilityStatus
            .AVAILABLE
        ),
        detail=(
            "Runtime dependencies and governance checks passed."
        ),
        artifact_exists=(
            artifact_exists
        ),
        manifest_exists=(
            manifest_exists
        ),
    )


# ================================================================
# PUBLIC STATUS API
# ================================================================


def get_score_status(
    score_id: str,
    verify_source_hashes: bool = True,
) -> ScoreAvailability:
    """
    Return runtime availability for one canonical score.
    """

    entry = get_score_entry(
        score_id
    )

    return (
        evaluate_score_availability(
            entry=entry,
            verify_source_hashes=(
                verify_source_hashes
            ),
        )
    )


def list_score_statuses(
    verify_source_hashes: bool = True,
) -> tuple[
    ScoreAvailability,
    ...,
]:
    """
    Return runtime availability for every registered score.
    """

    return tuple(
        evaluate_score_availability(
            entry=entry,
            verify_source_hashes=(
                verify_source_hashes
            ),
        )
        for entry
        in list_score_entries()
    )


def all_scores_available(
    verify_source_hashes: bool = True,
) -> bool:
    """
    Return True only when every registered score is AVAILABLE.
    """

    return all(
        status.available
        for status
        in list_score_statuses(
            verify_source_hashes=(
                verify_source_hashes
            )
        )
    )