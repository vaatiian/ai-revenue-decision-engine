import json

import pytest

from src.models.recommendation_manifest import (
    RECOMMENDATION_GOVERNED_SOURCE_FILES,
    build_recommendation_manifest,
    build_source_hash_register,
    load_recommendation_manifest,
    sha256_file,
    validate_recommendation_manifest,
)
from src.models.recommendation_spec import (
    RECOMMENDATION_DISCOVERY_DECISION,
    RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10,
    RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_MODEL_METADATA,
    RECOMMENDATION_REPEAT_DECISION,
    RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20,
    RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10,
    RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20,
)


# ================================================================
# HASHING
# ================================================================

def test_sha256_file(
    tmp_path,
):
    path = (
        tmp_path
        /
        "sample.txt"
    )

    path.write_text(
        "hello",
        encoding="utf-8",
    )

    import hashlib

    expected = hashlib.sha256(
        b"hello"
    ).hexdigest()

    assert (
        sha256_file(
            path
        )
        ==
        expected
    )


def test_sha256_missing_file_fails(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError
    ):
        sha256_file(
            tmp_path
            /
            "missing.txt"
        )


# ================================================================
# SOURCE HASH REGISTER
# ================================================================

def test_source_hash_register_contains_exact_governed_files():
    hashes = build_source_hash_register()

    assert (
        set(
            hashes
        )
        ==
        set(
            RECOMMENDATION_GOVERNED_SOURCE_FILES
        )
    )


def test_source_hash_values_are_sha256_strings():
    hashes = build_source_hash_register()

    for value in hashes.values():

        assert isinstance(
            value,
            str,
        )

        assert len(
            value
        ) == 64

        int(
            value,
            16,
        )


# ================================================================
# MANIFEST BUILD
# ================================================================

def test_manifest_uses_frozen_model_metadata():
    manifest = (
        build_recommendation_manifest()
    )

    assert (
        manifest[
            "model_name"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.model_name
    )

    assert (
        manifest[
            "model_version"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.model_version
    )

    assert (
        manifest[
            "model_type"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.model_type
    )

    assert (
        manifest[
            "target_name"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.target_name
    )


def test_manifest_marks_model_as_deterministic():
    manifest = (
        build_recommendation_manifest()
    )

    assert (
        manifest[
            "artifact_type"
        ]
        ==
        "deterministic_algorithm"
    )

    assert (
        manifest[
            "serialized_model_artifact_required"
        ]
        is False
    )


def test_manifest_freezes_two_channel_architecture():
    manifest = (
        build_recommendation_manifest()
    )

    architecture = manifest[
        "architecture"
    ]

    assert (
        architecture[
            "repeat_channel"
        ]
        ==
        "Customer Repeat Frequency"
    )

    assert (
        architecture[
            "discovery_channel"
        ]
        ==
        "Item-to-Item Affinity"
    )

    assert (
        architecture[
            "optional_hybrid"
        ]
        ==
        "70% Repeat / 30% Discovery"
    )


def test_manifest_freezes_discovery_thresholds():
    manifest = (
        build_recommendation_manifest()
    )

    discovery = manifest[
        "discovery_channel"
    ]

    assert (
        discovery[
            "minimum_item_orders"
        ]
        ==
        5
    )

    assert (
        discovery[
            "minimum_pair_orders"
        ]
        ==
        2
    )


def test_manifest_freezes_hybrid_allocation():
    manifest = (
        build_recommendation_manifest()
    )

    hybrid = manifest[
        "hybrid"
    ]

    assert (
        hybrid[
            "repeat_fraction"
        ]
        ==
        pytest.approx(
            0.70
        )
    )

    assert (
        hybrid[
            "discovery_fraction"
        ]
        ==
        pytest.approx(
            0.30
        )
    )

    assert (
        hybrid[
            "fill_order"
        ]
        ==
        [
            "repeat",
            "discovery",
        ]
    )


def test_manifest_freezes_recommendation_depths():
    manifest = (
        build_recommendation_manifest()
    )

    depths = manifest[
        "recommendation_depths"
    ]

    assert (
        depths[
            "evaluation_depths"
        ]
        ==
        [
            5,
            10,
            20,
        ]
    )

    assert (
        depths[
            "production_default_top_n"
        ]
        ==
        20
    )


def test_manifest_freezes_temporal_governance():
    manifest = (
        build_recommendation_manifest()
    )

    temporal = manifest[
        "temporal_governance"
    ]

    assert (
        temporal[
            "future_information_allowed"
        ]
        is False
    )

    assert (
        "strictly before"
        in
        temporal[
            "global_history_rule"
        ]
    )

    assert (
        "prediction order"
        in
        temporal[
            "customer_history_rule"
        ]
    )


def test_manifest_freezes_holdout_governance():
    manifest = (
        build_recommendation_manifest()
    )

    governance = manifest[
        "holdout_governance"
    ]

    assert (
        governance[
            "holdout_consumed"
        ]
        is True
    )

    assert (
        governance[
            "post_holdout_tuning_permitted"
        ]
        is False
    )


def test_manifest_freezes_research_decisions():
    manifest = (
        build_recommendation_manifest()
    )

    assert (
        manifest[
            "repeat_channel"
        ][
            "research_decision"
        ]
        ==
        RECOMMENDATION_REPEAT_DECISION
    )

    assert (
        manifest[
            "discovery_channel"
        ][
            "research_decision"
        ]
        ==
        RECOMMENDATION_DISCOVERY_DECISION
    )


def test_manifest_freezes_holdout_performance():
    manifest = (
        build_recommendation_manifest()
    )

    performance = manifest[
        "final_holdout_performance"
    ]

    assert (
        performance[
            "repeat_hit_rate_at_10"
        ]
        ==
        pytest.approx(
            RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10
        )
    )

    assert (
        performance[
            "repeat_hit_rate_at_20"
        ]
        ==
        pytest.approx(
            RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20
        )
    )

    assert (
        performance[
            "repeat_eligible_hit_rate_at_20"
        ]
        ==
        pytest.approx(
            RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20
        )
    )

    assert (
        performance[
            "discovery_hit_rate_at_10"
        ]
        ==
        pytest.approx(
            RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10
        )
    )

    assert (
        performance[
            "discovery_hit_rate_at_20"
        ]
        ==
        pytest.approx(
            RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20
        )
    )

    assert (
        performance[
            "hybrid_hit_rate_at_20"
        ]
        ==
        pytest.approx(
            RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20
        )
    )


# ================================================================
# MANIFEST VALIDATION
# ================================================================

def test_valid_manifest_passes():
    manifest = (
        build_recommendation_manifest()
    )

    validate_recommendation_manifest(
        manifest
    )


@pytest.mark.parametrize(
    "field",
    [
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "artifact_type",
        "outcome_window_days",
    ],
)
def test_modified_top_level_contract_fails(
    field,
):
    manifest = (
        build_recommendation_manifest()
    )

    manifest[
        field
    ] = "tampered"

    with pytest.raises(
        ValueError,
        match="manifest mismatch",
    ):
        validate_recommendation_manifest(
            manifest
        )


@pytest.mark.parametrize(
    "field",
    [
        "architecture",
        "repeat_channel",
        "discovery_channel",
        "hybrid",
        "recommendation_depths",
        "temporal_governance",
        "metrics",
        "development_periods",
        "holdout_governance",
        "final_holdout_performance",
    ],
)
def test_modified_nested_contract_fails(
    field,
):
    manifest = (
        build_recommendation_manifest()
    )

    manifest[
        field
    ] = {
        "tampered":
            True
    }

    with pytest.raises(
        ValueError,
        match="manifest mismatch",
    ):
        validate_recommendation_manifest(
            manifest
        )


def test_missing_governed_source_fails():
    manifest = (
        build_recommendation_manifest()
    )

    first_source = (
        RECOMMENDATION_GOVERNED_SOURCE_FILES[
            0
        ]
    )

    del manifest[
        "governed_source_sha256"
    ][
        first_source
    ]

    with pytest.raises(
        ValueError,
        match="source-file register",
    ):
        validate_recommendation_manifest(
            manifest,
            verify_source_hashes=False,
        )


def test_modified_source_hash_fails():
    manifest = (
        build_recommendation_manifest()
    )

    first_source = (
        RECOMMENDATION_GOVERNED_SOURCE_FILES[
            0
        ]
    )

    manifest[
        "governed_source_sha256"
    ][
        first_source
    ] = (
        "0"
        *
        64
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 validation failed",
    ):
        validate_recommendation_manifest(
            manifest,
            verify_source_hashes=True,
        )


# ================================================================
# MANIFEST LOADING
# ================================================================

def test_load_valid_manifest(
    tmp_path,
):
    path = (
        tmp_path
        /
        "manifest.json"
    )

    manifest = (
        build_recommendation_manifest()
    )

    path.write_text(
        json.dumps(
            manifest
        ),
        encoding="utf-8",
    )

    loaded = (
        load_recommendation_manifest(
            path
        )
    )

    assert (
        loaded[
            "model_name"
        ]
        ==
        RECOMMENDATION_MODEL_METADATA.model_name
    )


def test_missing_manifest_fails(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError
    ):
        load_recommendation_manifest(
            tmp_path
            /
            "missing.json"
        )


def test_invalid_json_manifest_fails(
    tmp_path,
):
    path = (
        tmp_path
        /
        "invalid.json"
    )

    path.write_text(
        "{not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="not valid JSON",
    ):
        load_recommendation_manifest(
            path
        )


def test_manifest_missing_required_field_fails(
    tmp_path,
):
    path = (
        tmp_path
        /
        "incomplete.json"
    )

    manifest = (
        build_recommendation_manifest()
    )

    del manifest[
        "architecture"
    ]

    path.write_text(
        json.dumps(
            manifest
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        load_recommendation_manifest(
            path
        )