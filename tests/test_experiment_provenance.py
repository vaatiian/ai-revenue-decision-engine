"""
Tests for production registry -> experiment provenance integration.
"""

import hashlib
from pathlib import Path

import pytest

from src.experiments.provenance import (
    ScoreRunProvenance,
    build_entry_provenance,
    build_score_provenance,
    sha256_file,
)

from src.models.score_registry import (
    get_score_entry,
    list_score_ids,
)


FITTED_SCORE_IDS = (
    "reactivation",
    "revenue_leakage",
    "future_customer_value",
    "repeat_purchase",
)


DETERMINISTIC_SCORE_IDS = (
    "recommendation",
    "demand_forecast",
)


# ================================================================
# HASHING
# ================================================================


def test_sha256_file_matches_hashlib(
    tmp_path,
):

    path = (
        tmp_path
        /
        "sample.bin"
    )

    payload = (
        b"experiment provenance"
    )

    path.write_bytes(
        payload
    )

    expected = (
        hashlib.sha256(
            payload
        )
        .hexdigest()
    )

    assert (
        sha256_file(
            path
        )
        ==
        expected
    )


def test_sha256_missing_file_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "missing.bin"
    )

    with pytest.raises(
        FileNotFoundError,
        match="missing file",
    ):

        sha256_file(
            path
        )


def test_sha256_directory_fails_closed(
    tmp_path,
):

    with pytest.raises(
        ValueError,
        match="must be a file",
    ):

        sha256_file(
            tmp_path
        )


# ================================================================
# CANONICAL SIX
# ================================================================


@pytest.mark.parametrize(
    "score_id",
    list_score_ids(),
)
def test_all_registered_scores_build_provenance(
    score_id,
):

    result = (
        build_score_provenance(
            score_id
        )
    )

    assert isinstance(
        result,
        ScoreRunProvenance,
    )

    assert (
        result.score_id
        ==
        score_id
    )

    assert len(
        result.manifest_sha256
    ) == 64


@pytest.mark.parametrize(
    "score_id",
    list_score_ids(),
)
def test_provenance_identity_matches_registry(
    score_id,
):

    entry = (
        get_score_entry(
            score_id
        )
    )

    result = (
        build_score_provenance(
            score_id
        )
    )

    assert (
        result.model_name
        ==
        entry.model_name
    )

    assert (
        result.model_version
        ==
        entry.model_version
    )

    assert (
        result.model_type
        ==
        entry.model_type
    )

    assert (
        result.target_name
        ==
        entry.target_name
    )

    assert (
        result.scorer_type
        ==
        entry.scorer_type
    )

    assert (
        result.deterministic
        ==
        entry.deterministic
    )


# ================================================================
# FITTED / DETERMINISTIC CONTRACT
# ================================================================


@pytest.mark.parametrize(
    "score_id",
    FITTED_SCORE_IDS,
)
def test_fitted_scores_capture_artifact_provenance(
    score_id,
):

    result = (
        build_score_provenance(
            score_id
        )
    )

    assert (
        result.artifact_path
        is not None
    )

    assert (
        result.artifact_sha256
        is not None
    )

    assert len(
        result.artifact_sha256
    ) == 64

    assert Path(
        result.artifact_path
    ).is_file()


@pytest.mark.parametrize(
    "score_id",
    DETERMINISTIC_SCORE_IDS,
)
def test_deterministic_scores_have_no_artifact_provenance(
    score_id,
):

    result = (
        build_score_provenance(
            score_id
        )
    )

    assert result.deterministic is True
    assert result.artifact_path is None
    assert result.artifact_sha256 is None


# ================================================================
# MANIFEST PROVENANCE
# ================================================================


@pytest.mark.parametrize(
    "score_id",
    list_score_ids(),
)
def test_manifest_hash_is_actual_file_hash(
    score_id,
):

    result = (
        build_score_provenance(
            score_id
        )
    )

    assert Path(
        result.manifest_path
    ).is_file()

    assert (
        result.manifest_sha256
        ==
        sha256_file(
            result.manifest_path
        )
    )


def test_paths_use_stable_forward_slashes():

    result = (
        build_score_provenance(
            "reactivation"
        )
    )

    assert "\\" not in (
        result.manifest_path
    )

    assert "\\" not in (
        result.artifact_path
    )


# ================================================================
# PUBLIC CONTRACT
# ================================================================


def test_entry_adapter_matches_score_id_adapter():

    entry = (
        get_score_entry(
            "future_customer_value"
        )
    )

    from_entry = (
        build_entry_provenance(
            entry
        )
    )

    from_id = (
        build_score_provenance(
            "future_customer_value"
        )
    )

    assert (
        from_entry
        ==
        from_id
    )


def test_unknown_score_id_fails_closed():

    with pytest.raises(
        KeyError,
        match="Unknown score_id",
    ):

        build_score_provenance(
            "unknown_score"
        )


def test_non_string_score_id_fails_closed():

    with pytest.raises(
        TypeError,
        match="score_id must be a string",
    ):

        build_score_provenance(
            123
        )


def test_verify_source_hashes_must_be_bool():

    with pytest.raises(
        TypeError,
        match="verify_source_hashes must be a bool",
    ):

        build_score_provenance(
            "reactivation",
            verify_source_hashes="yes",
        )