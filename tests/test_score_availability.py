from dataclasses import FrozenInstanceError, replace

import pytest

import src.models.score_availability as availability_module

from src.models.score_availability import (
    ScoreAvailability,
    ScoreAvailabilityStatus,
    all_scores_available,
    evaluate_score_availability,
    get_score_status,
    list_score_statuses,
)

from src.models.score_registry import (
    DEMAND_FORECAST_ENTRY,
    REACTIVATION_ENTRY,
    RECOMMENDATION_ENTRY,
    REPEAT_PURCHASE_ENTRY,
    get_score_entry,
)


# ================================================================
# STATUS CONTRACT
# ================================================================


def test_status_values_are_frozen():

    assert (
        ScoreAvailabilityStatus.AVAILABLE.value
        ==
        "AVAILABLE"
    )

    assert (
        ScoreAvailabilityStatus.MISSING_ARTIFACT.value
        ==
        "MISSING_ARTIFACT"
    )

    assert (
        ScoreAvailabilityStatus.MISSING_MANIFEST.value
        ==
        "MISSING_MANIFEST"
    )

    assert (
        ScoreAvailabilityStatus.INVALID_ARTIFACT.value
        ==
        "INVALID_ARTIFACT"
    )

    assert (
        ScoreAvailabilityStatus.INVALID_MANIFEST.value
        ==
        "INVALID_MANIFEST"
    )

    assert (
        ScoreAvailabilityStatus.UNAVAILABLE.value
        ==
        "UNAVAILABLE"
    )


def test_availability_result_is_immutable():

    result = ScoreAvailability(
        score_id="example",
        status=(
            ScoreAvailabilityStatus.AVAILABLE
        ),
        available=True,
        artifact_exists=True,
        manifest_exists=True,
        detail="ok",
    )

    with pytest.raises(
        FrozenInstanceError
    ):

        result.available = False


# ================================================================
# INPUT VALIDATION
# ================================================================


def test_evaluate_rejects_invalid_entry_type():

    with pytest.raises(
        TypeError,
        match="ScoreRegistryEntry",
    ):

        evaluate_score_availability(
            "reactivation"
        )


# ================================================================
# MISSING FILE STATES
# ================================================================


def test_missing_manifest_is_reported_first(
    tmp_path,
):

    entry = replace(
        REACTIVATION_ENTRY,
        manifest_path=(
            tmp_path
            /
            "missing.json"
        ),
        artifact_path=(
            tmp_path
            /
            "missing.joblib"
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

    assert (
        result.available
        is False
    )

    assert (
        result.manifest_exists
        is False
    )

    assert (
        result.artifact_exists
        is False
    )


def test_missing_artifact_is_reported(
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
            "missing.joblib"
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

    assert (
        result.manifest_exists
        is True
    )

    assert (
        result.artifact_exists
        is False
    )


# ================================================================
# INVALID ARTIFACT
# ================================================================


def test_invalid_reactivation_artifact_is_classified(
    tmp_path,
    monkeypatch,
):

    artifact_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    artifact_path.write_bytes(
        b"artifact"
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

    def broken_loader(path):

        raise RuntimeError(
            "broken artifact"
        )

    monkeypatch.setattr(
        availability_module,
        "load_reactivation_artifact",
        broken_loader,
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

    assert (
        result.available
        is False
    )

    assert (
        "broken artifact"
        in
        result.detail
    )


def test_invalid_repeat_purchase_artifact_is_classified(
    tmp_path,
    monkeypatch,
):

    artifact_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    artifact_path.write_bytes(
        b"artifact"
    )

    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )

    entry = replace(
        REPEAT_PURCHASE_ENTRY,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
    )

    def broken_loader(path):

        raise RuntimeError(
            "repeat artifact broken"
        )

    monkeypatch.setattr(
        availability_module,
        "load_repeat_purchase_model",
        broken_loader,
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

    assert (
        result.available
        is False
    )

    assert (
        "repeat artifact broken"
        in
        result.detail
    )


# ================================================================
# INVALID MANIFEST
# ================================================================


def test_invalid_fitted_manifest_is_classified(
    tmp_path,
    monkeypatch,
):

    artifact_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    artifact_path.write_bytes(
        b"artifact"
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

    monkeypatch.setattr(
        availability_module,
        "_validate_artifact",
        lambda entry: None,
    )

    def broken_manifest(
        entry,
        verify_source_hashes,
    ):

        raise ValueError(
            "manifest mismatch"
        )

    monkeypatch.setattr(
        availability_module,
        "_validate_manifest",
        broken_manifest,
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

    assert (
        result.available
        is False
    )

    assert (
        "manifest mismatch"
        in
        result.detail
    )


def test_invalid_deterministic_manifest_is_classified(
    tmp_path,
    monkeypatch,
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

    def broken_manifest(
        entry,
        verify_source_hashes,
    ):

        raise ValueError(
            "source hash mismatch"
        )

    monkeypatch.setattr(
        availability_module,
        "_validate_manifest",
        broken_manifest,
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

    assert (
        result.artifact_exists
        is None
    )

    assert (
        result.manifest_exists
        is True
    )

    assert (
        "source hash mismatch"
        in
        result.detail
    )


# ================================================================
# AVAILABLE STATE
# ================================================================


def test_fitted_score_can_be_available(
    tmp_path,
    monkeypatch,
):

    artifact_path = (
        tmp_path
        /
        "model.joblib"
    )

    manifest_path = (
        tmp_path
        /
        "manifest.json"
    )

    artifact_path.write_bytes(
        b"artifact"
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

    monkeypatch.setattr(
        availability_module,
        "_validate_artifact",
        lambda entry: None,
    )

    monkeypatch.setattr(
        availability_module,
        "_validate_manifest",
        lambda entry, verify_source_hashes: None,
    )

    result = (
        evaluate_score_availability(
            entry
        )
    )

    assert (
        result.status
        ==
        ScoreAvailabilityStatus.AVAILABLE
    )

    assert (
        result.available
        is True
    )

    assert (
        result.artifact_exists
        is True
    )

    assert (
        result.manifest_exists
        is True
    )


def test_deterministic_score_can_be_available(
    tmp_path,
    monkeypatch,
):

    manifest_path = (
        tmp_path
        /
        "demand.json"
    )

    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )

    entry = replace(
        DEMAND_FORECAST_ENTRY,
        manifest_path=manifest_path,
    )

    monkeypatch.setattr(
        availability_module,
        "_validate_manifest",
        lambda entry, verify_source_hashes: None,
    )

    result = (
        evaluate_score_availability(
            entry
        )
    )

    assert (
        result.status
        ==
        ScoreAvailabilityStatus.AVAILABLE
    )

    assert (
        result.available
        is True
    )

    assert (
        result.artifact_exists
        is None
    )

    assert (
        result.manifest_exists
        is True
    )


# ================================================================
# SOURCE HASH FLAG
# ================================================================


def test_verify_source_hashes_flag_is_forwarded(
    tmp_path,
    monkeypatch,
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

    observed = []

    def fake_validate(
        entry,
        verify_source_hashes,
    ):

        observed.append(
            verify_source_hashes
        )

    monkeypatch.setattr(
        availability_module,
        "_validate_manifest",
        fake_validate,
    )

    result = (
        evaluate_score_availability(
            entry,
            verify_source_hashes=False,
        )
    )

    assert (
        result.status
        ==
        ScoreAvailabilityStatus.AVAILABLE
    )

    assert observed == [
        False
    ]


# ================================================================
# PUBLIC API
# ================================================================


def test_get_score_status_uses_registry(
    monkeypatch,
):

    expected_entry = (
        get_score_entry(
            "reactivation"
        )
    )

    observed = []

    def fake_evaluate(
        entry,
        verify_source_hashes=True,
    ):

        observed.append(
            entry
        )

        return ScoreAvailability(
            score_id=entry.score_id,
            status=(
                ScoreAvailabilityStatus.AVAILABLE
            ),
            available=True,
            artifact_exists=True,
            manifest_exists=True,
            detail="ok",
        )

    monkeypatch.setattr(
        availability_module,
        "evaluate_score_availability",
        fake_evaluate,
    )

    result = get_score_status(
        "reactivation"
    )

    assert (
        result.score_id
        ==
        "reactivation"
    )

    assert observed == [
        expected_entry
    ]


def test_get_score_status_rejects_unknown_id():

    with pytest.raises(
        KeyError,
        match="Unknown score_id",
    ):

        get_score_status(
            "unknown_score"
        )


def test_list_score_statuses_preserves_registry_order(
    monkeypatch,
):

    def fake_evaluate(
        entry,
        verify_source_hashes=True,
    ):

        return ScoreAvailability(
            score_id=entry.score_id,
            status=(
                ScoreAvailabilityStatus.AVAILABLE
            ),
            available=True,
            artifact_exists=None,
            manifest_exists=True,
            detail="ok",
        )

    monkeypatch.setattr(
        availability_module,
        "evaluate_score_availability",
        fake_evaluate,
    )

    statuses = (
        list_score_statuses()
    )

    assert tuple(
        item.score_id
        for item
        in statuses
    ) == (
        "reactivation",
        "revenue_leakage",
        "future_customer_value",
        "repeat_purchase",
        "recommendation",
        "demand_forecast",
    )


def test_all_scores_available_true(
    monkeypatch,
):

    monkeypatch.setattr(
        availability_module,
        "list_score_statuses",
        lambda verify_source_hashes=True: (
            ScoreAvailability(
                score_id="a",
                status=(
                    ScoreAvailabilityStatus.AVAILABLE
                ),
                available=True,
                artifact_exists=None,
                manifest_exists=True,
                detail="ok",
            ),
            ScoreAvailability(
                score_id="b",
                status=(
                    ScoreAvailabilityStatus.AVAILABLE
                ),
                available=True,
                artifact_exists=True,
                manifest_exists=True,
                detail="ok",
            ),
        ),
    )

    assert (
        all_scores_available()
        is True
    )


def test_all_scores_available_false(
    monkeypatch,
):

    monkeypatch.setattr(
        availability_module,
        "list_score_statuses",
        lambda verify_source_hashes=True: (
            ScoreAvailability(
                score_id="a",
                status=(
                    ScoreAvailabilityStatus.AVAILABLE
                ),
                available=True,
                artifact_exists=True,
                manifest_exists=True,
                detail="ok",
            ),
            ScoreAvailability(
                score_id="b",
                status=(
                    ScoreAvailabilityStatus.MISSING_ARTIFACT
                ),
                available=False,
                artifact_exists=False,
                manifest_exists=True,
                detail="missing",
            ),
        ),
    )

    assert (
        all_scores_available()
        is False
    )