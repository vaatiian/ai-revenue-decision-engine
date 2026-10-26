import copy
import json

from pathlib import Path

import pytest

from src.features.demand_model_features import (
    DEMAND_WEEK_FREQUENCY,
    DEMAND_ZERO_TRADING_WEEK_PERIODS,
)

from src.models.demand_manifest import (
    DEMAND_FEATURE_SOURCE_PATH,
    DEMAND_MANIFEST_FILENAME,
    DEMAND_RESEARCH_NOTEBOOK_PATH,
    DEMAND_SCORER_SOURCE_PATH,
    DEMAND_SPEC_SOURCE_PATH,
    build_demand_manifest,
    load_demand_manifest,
    sha256_file,
    validate_demand_manifest,
    validate_demand_manifest_file,
    write_demand_manifest,
)

from src.models.demand_spec import (
    DEMAND_FINAL_DECISION,
    DEMAND_FORECAST_HORIZONS,
    DEMAND_MODEL_METADATA,
    DEMAND_RECENT_WEIGHT,
    DEMAND_RECENT_WINDOW_WEEKS,
    DEMAND_SEASONAL_LAG_WEEKS,
    DEMAND_SEASONAL_WEIGHT,
)


# ================================================================
# FILE / HASH HELPERS
# ================================================================

def test_sha256_file_returns_64_character_hex_digest(
    tmp_path,
):
    source = (
        tmp_path
        /
        "source.txt"
    )

    source.write_text(
        "demand governance",
        encoding="utf-8",
    )

    digest = sha256_file(
        source
    )

    assert len(
        digest
    ) == 64

    int(
        digest,
        16,
    )


def test_sha256_file_is_deterministic(
    tmp_path,
):
    source = (
        tmp_path
        /
        "source.txt"
    )

    source.write_text(
        "same bytes",
        encoding="utf-8",
    )

    assert (
        sha256_file(
            source
        )
        ==
        sha256_file(
            source
        )
    )


def test_sha256_file_changes_when_file_changes(
    tmp_path,
):
    source = (
        tmp_path
        /
        "source.txt"
    )

    source.write_text(
        "version one",
        encoding="utf-8",
    )

    first = sha256_file(
        source
    )

    source.write_text(
        "version two",
        encoding="utf-8",
    )

    second = sha256_file(
        source
    )

    assert first != second


def test_sha256_missing_file_fails(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Governed demand source not found",
    ):
        sha256_file(
            tmp_path
            /
            "missing.py"
        )


# ================================================================
# GOVERNED SOURCE PATHS
# ================================================================

def test_research_notebook_exists():
    assert (
        DEMAND_RESEARCH_NOTEBOOK_PATH
        .is_file()
    )


def test_demand_spec_source_exists():
    assert (
        DEMAND_SPEC_SOURCE_PATH
        .is_file()
    )


def test_demand_feature_source_exists():
    assert (
        DEMAND_FEATURE_SOURCE_PATH
        .is_file()
    )


def test_demand_scorer_source_exists():
    assert (
        DEMAND_SCORER_SOURCE_PATH
        .is_file()
    )


def test_manifest_filename_is_versioned():
    assert (
        DEMAND_MANIFEST_FILENAME
        ==
        "demand_weekly_seasonal_blend_v1.0.0.json"
    )


# ================================================================
# MANIFEST CONSTRUCTION
# ================================================================

def test_build_manifest_model_identity():
    manifest = build_demand_manifest()

    assert (
        manifest[
            "model_name"
        ]
        ==
        DEMAND_MODEL_METADATA.model_name
    )

    assert (
        manifest[
            "model_version"
        ]
        ==
        DEMAND_MODEL_METADATA.model_version
    )

    assert (
        manifest[
            "model_type"
        ]
        ==
        DEMAND_MODEL_METADATA.model_type
    )

    assert (
        manifest[
            "target_name"
        ]
        ==
        DEMAND_MODEL_METADATA.target_name
    )


def test_manifest_records_no_serialized_model():
    manifest = build_demand_manifest()

    assert (
        manifest[
            "serialized_model_required"
        ]
        is False
    )


def test_manifest_records_frozen_horizons():
    manifest = build_demand_manifest()

    assert (
        tuple(
            manifest[
                "forecast_horizons_weeks"
            ]
        )
        ==
        DEMAND_FORECAST_HORIZONS
    )


def test_manifest_records_90_10_architecture():
    manifest = build_demand_manifest()

    architecture = manifest[
        "architecture"
    ]

    assert (
        architecture[
            "seasonal_weight"
        ]
        ==
        DEMAND_SEASONAL_WEIGHT
    )

    assert (
        architecture[
            "recent_weight"
        ]
        ==
        DEMAND_RECENT_WEIGHT
    )

    assert (
        architecture[
            "seasonal_lag_weeks"
        ]
        ==
        DEMAND_SEASONAL_LAG_WEEKS
    )

    assert (
        architecture[
            "recent_window_weeks"
        ]
        ==
        DEMAND_RECENT_WINDOW_WEEKS
    )


def test_manifest_records_weekly_calendar():
    manifest = build_demand_manifest()

    calendar = manifest[
        "calendar_governance"
    ]

    assert (
        calendar[
            "week_frequency"
        ]
        ==
        DEMAND_WEEK_FREQUENCY
    )


def test_manifest_records_exact_zero_trading_weeks():
    manifest = build_demand_manifest()

    calendar = manifest[
        "calendar_governance"
    ]

    assert (
        tuple(
            calendar[
                "zero_trading_week_periods"
            ]
        )
        ==
        DEMAND_ZERO_TRADING_WEEK_PERIODS
    )


def test_manifest_fails_closed_on_unknown_missing_weeks():
    manifest = build_demand_manifest()

    assert (
        manifest[
            "calendar_governance"
        ][
            "unknown_interior_missing_weeks"
        ]
        ==
        "fail_closed"
    )


def test_manifest_records_go_decision():
    manifest = build_demand_manifest()

    assert (
        manifest[
            "governance"
        ][
            "decision"
        ]
        ==
        DEMAND_FINAL_DECISION
    )


def test_manifest_records_frozen_architecture():
    manifest = build_demand_manifest()

    assert (
        manifest[
            "governance"
        ][
            "architecture_frozen"
        ]
        is True
    )

    assert (
        manifest[
            "governance"
        ][
            "post_freeze_tuning_permitted"
        ]
        is False
    )


def test_manifest_contains_all_governed_hashes():
    manifest = build_demand_manifest()

    hashes = manifest[
        "source_sha256"
    ]

    assert set(
        hashes
    ) == {
        "research_notebook",
        "production_spec",
        "weekly_feature_engine",
        "production_scorer",
    }


@pytest.mark.parametrize(
    "source_name",
    [
        "research_notebook",
        "production_spec",
        "weekly_feature_engine",
        "production_scorer",
    ],
)
def test_manifest_hashes_are_sha256(
    source_name,
):
    manifest = build_demand_manifest()

    digest = manifest[
        "source_sha256"
    ][
        source_name
    ]

    assert len(
        digest
    ) == 64

    int(
        digest,
        16,
    )


def test_manifest_hashes_match_current_sources():
    manifest = build_demand_manifest()

    assert (
        manifest[
            "source_sha256"
        ][
            "research_notebook"
        ]
        ==
        sha256_file(
            DEMAND_RESEARCH_NOTEBOOK_PATH
        )
    )

    assert (
        manifest[
            "source_sha256"
        ][
            "production_spec"
        ]
        ==
        sha256_file(
            DEMAND_SPEC_SOURCE_PATH
        )
    )

    assert (
        manifest[
            "source_sha256"
        ][
            "weekly_feature_engine"
        ]
        ==
        sha256_file(
            DEMAND_FEATURE_SOURCE_PATH
        )
    )

    assert (
        manifest[
            "source_sha256"
        ][
            "production_scorer"
        ]
        ==
        sha256_file(
            DEMAND_SCORER_SOURCE_PATH
        )
    )


# ================================================================
# WRITE / LOAD
# ================================================================

def test_write_and_load_manifest(
    tmp_path,
):
    path = (
        tmp_path
        /
        "demand.json"
    )

    written = write_demand_manifest(
        path
    )

    assert written == path

    assert path.is_file()

    manifest = load_demand_manifest(
        path
    )

    assert (
        manifest[
            "model_name"
        ]
        ==
        DEMAND_MODEL_METADATA.model_name
    )


def test_written_manifest_is_valid_json(
    tmp_path,
):
    path = (
        tmp_path
        /
        "demand.json"
    )

    write_demand_manifest(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        parsed = json.load(
            handle
        )

    assert isinstance(
        parsed,
        dict,
    )


def test_load_missing_manifest_fails(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Demand manifest not found",
    ):
        load_demand_manifest(
            tmp_path
            /
            "missing.json"
        )


# ================================================================
# VALIDATION
# ================================================================

def test_valid_manifest_passes():
    manifest = build_demand_manifest()

    validate_demand_manifest(
        manifest
    )


def test_validate_manifest_file_passes(
    tmp_path,
):
    path = (
        tmp_path
        /
        "demand.json"
    )

    write_demand_manifest(
        path
    )

    loaded = validate_demand_manifest_file(
        path
    )

    assert (
        loaded[
            "model_name"
        ]
        ==
        DEMAND_MODEL_METADATA.model_name
    )


def test_wrong_model_name_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "model_name"
    ] = "Wrong Model"

    with pytest.raises(
        ValueError,
        match="model_name",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_model_version_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "model_version"
    ] = "999.0.0"

    with pytest.raises(
        ValueError,
        match="model_version",
    ):
        validate_demand_manifest(
            bad
        )


def test_serialized_model_requirement_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "serialized_model_required"
    ] = True

    with pytest.raises(
        ValueError,
        match="serialized",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_horizons_fail():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "forecast_horizons_weeks"
    ] = [
        1,
        2,
        3,
    ]

    with pytest.raises(
        ValueError,
        match="forecast horizons",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_seasonal_weight_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "architecture"
    ][
        "seasonal_weight"
    ] = 0.80

    with pytest.raises(
        ValueError,
        match="seasonal_weight",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_recent_weight_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "architecture"
    ][
        "recent_weight"
    ] = 0.20

    with pytest.raises(
        ValueError,
        match="recent_weight",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_seasonal_lag_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "architecture"
    ][
        "seasonal_lag_weeks"
    ] = 51

    with pytest.raises(
        ValueError,
        match="seasonal_lag_weeks",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_recent_window_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "architecture"
    ][
        "recent_window_weeks"
    ] = 8

    with pytest.raises(
        ValueError,
        match="recent_window_weeks",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_calendar_frequency_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "calendar_governance"
    ][
        "week_frequency"
    ] = "W-SAT"

    with pytest.raises(
        ValueError,
        match="calendar frequency",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_zero_trading_register_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "calendar_governance"
    ][
        "zero_trading_week_periods"
    ] = []

    with pytest.raises(
        ValueError,
        match="zero-trading",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_governance_decision_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "governance"
    ][
        "decision"
    ] = "NO GO"

    with pytest.raises(
        ValueError,
        match="governance decision",
    ):
        validate_demand_manifest(
            bad
        )


def test_post_freeze_tuning_permission_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "governance"
    ][
        "post_freeze_tuning_permitted"
    ] = True

    with pytest.raises(
        ValueError,
        match="post-freeze tuning",
    ):
        validate_demand_manifest(
            bad
        )


def test_wrong_source_hash_fails():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "source_sha256"
    ][
        "production_scorer"
    ] = (
        "0"
        *
        64
    )

    with pytest.raises(
        ValueError,
        match="production_scorer",
    ):
        validate_demand_manifest(
            bad,
            verify_source_hashes=True,
        )


def test_source_hash_verification_can_be_disabled():
    manifest = build_demand_manifest()

    bad = copy.deepcopy(
        manifest
    )

    bad[
        "source_sha256"
    ][
        "production_scorer"
    ] = (
        "0"
        *
        64
    )

    validate_demand_manifest(
        bad,
        verify_source_hashes=False,
    )