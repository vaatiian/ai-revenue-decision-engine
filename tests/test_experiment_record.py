"""
Tests for immutable experiment-run records.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from src.experiments.record import (
    ExperimentRunRecord,
    experiment_run_to_dict,
)
from src.experiments.spec import (
    EXPERIMENT_RUN_FIELDS,
    EXPERIMENT_SCHEMA_VERSION,
    RUN_KIND_EVALUATION,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
)


SHA_A = (
    "a"
    *
    64
)

SHA_B = (
    "b"
    *
    64
)

SHA_C = (
    "c"
    *
    64
)


def make_record(
    **overrides,
):

    values = {
        "schema_version":
            EXPERIMENT_SCHEMA_VERSION,

        "run_id":
            "run-001",

        "experiment_name":
            "reactivation-final-evaluation",

        "run_kind":
            RUN_KIND_EVALUATION,

        "status":
            RUN_STATUS_SUCCEEDED,

        "score_id":
            "reactivation",

        "model_name":
            "Full Random Forest",

        "model_version":
            "1.0.0",

        "model_type":
            "classification",

        "target_name":
            "reactivation_within_next_90_days",

        "started_at_utc":
            datetime(
                2026,
                10,
                27,
                10,
                0,
                tzinfo=timezone.utc,
            ),

        "completed_at_utc":
            datetime(
                2026,
                10,
                27,
                10,
                5,
                tzinfo=timezone.utc,
            ),

        "git_commit":
            "61424ff89ecf21325f7f7d10fee191fff5c1a804",

        "git_branch":
            "main",

        "manifest_path":
            (
                "artifacts/models/reactivation/"
                "reactivation_full_random_forest_v1.0.0.json"
            ),

        "manifest_sha256":
            SHA_A,

        "artifact_path":
            (
                "artifacts/models/reactivation/"
                "reactivation_full_random_forest_v1.0.0.joblib"
            ),

        "artifact_sha256":
            SHA_B,

        "data_reference":
            "cleaned_online_retail.parquet",

        "data_sha256":
            SHA_C,

        "parameters":
            {
                "evaluation_split":
                    "test",
            },

        "metrics":
            {
                "pr_auc":
                    0.615,
            },

        "tags":
            (
                "production",
                "temporal",
            ),

        "notes":
            "Final governed evaluation.",

        "error_message":
            None,
    }

    values.update(
        overrides
    )

    return ExperimentRunRecord(
        **values
    )


# ================================================================
# CONSTRUCTION / IMMUTABILITY
# ================================================================


def test_valid_record_can_be_created():

    record = (
        make_record()
    )

    assert (
        record.run_id
        ==
        "run-001"
    )

    assert (
        record.status
        ==
        RUN_STATUS_SUCCEEDED
    )


def test_record_is_frozen():

    record = (
        make_record()
    )

    with pytest.raises(
        FrozenInstanceError
    ):

        record.status = (
            RUN_STATUS_FAILED
        )


def test_parameter_mapping_is_read_only():

    record = (
        make_record()
    )

    assert isinstance(
        record.parameters,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):

        record.parameters[
            "new"
        ] = 1


def test_metric_mapping_is_read_only():

    record = (
        make_record()
    )

    assert isinstance(
        record.metrics,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):

        record.metrics[
            "new"
        ] = 1


def test_input_mapping_is_copied_before_freezing():

    parameters = {
        "split":
            "test",
    }

    record = (
        make_record(
            parameters=parameters
        )
    )

    parameters[
        "split"
    ] = "changed"

    assert (
        record.parameters[
            "split"
        ]
        ==
        "test"
    )


def test_tags_are_frozen_as_tuple():

    record = (
        make_record(
            tags=[
                "production",
                "evaluation",
            ]
        )
    )

    assert (
        record.tags
        ==
        (
            "production",
            "evaluation",
        )
    )


# ================================================================
# TIMING
# ================================================================


def test_naive_timestamps_are_normalised_to_utc():

    record = (
        make_record(
            started_at_utc=(
                "2026-10-27 10:00:00"
            ),
            completed_at_utc=(
                "2026-10-27 10:05:00"
            ),
        )
    )

    assert (
        record.started_at_utc.tzinfo
        is not None
    )

    assert (
        record.completed_at_utc.tzinfo
        is not None
    )


def test_completed_time_cannot_precede_start():

    with pytest.raises(
        ValueError,
        match="cannot be earlier",
    ):

        make_record(
            started_at_utc=(
                "2026-10-27T10:05:00Z"
            ),
            completed_at_utc=(
                "2026-10-27T10:00:00Z"
            ),
        )


def test_running_record_has_no_completion_time():

    record = (
        make_record(
            status=(
                RUN_STATUS_RUNNING
            ),
            completed_at_utc=None,
        )
    )

    assert (
        record.completed_at_utc
        is None
    )


def test_running_record_rejects_completion_time():

    with pytest.raises(
        ValueError,
        match="RUNNING",
    ):

        make_record(
            status=(
                RUN_STATUS_RUNNING
            ),
        )


def test_completed_record_requires_completion_time():

    with pytest.raises(
        ValueError,
        match="must have completed_at_utc",
    ):

        make_record(
            completed_at_utc=None,
        )


# ================================================================
# HASH / ARTIFACT CONTRACT
# ================================================================


def test_sha256_is_normalised_to_lowercase():

    record = (
        make_record(
            manifest_sha256=(
                "A"
                *
                64
            )
        )
    )

    assert (
        record.manifest_sha256
        ==
        SHA_A
    )


def test_invalid_manifest_sha256_fails():

    with pytest.raises(
        ValueError,
        match="64 hexadecimal",
    ):

        make_record(
            manifest_sha256="abc"
        )


def test_artifact_path_and_hash_must_be_paired():

    with pytest.raises(
        ValueError,
        match="artifact_path and artifact_sha256",
    ):

        make_record(
            artifact_sha256=None
        )


def test_deterministic_record_can_have_no_artifact():

    record = (
        make_record(
            score_id="recommendation",
            artifact_path=None,
            artifact_sha256=None,
        )
    )

    assert record.artifact_path is None
    assert record.artifact_sha256 is None


def test_data_hash_requires_data_reference():

    with pytest.raises(
        ValueError,
        match="data_sha256 requires data_reference",
    ):

        make_record(
            data_reference=None,
            data_sha256=SHA_C,
        )


# ================================================================
# STATUS / ERROR CONTRACT
# ================================================================


def test_failed_record_requires_error_message():

    with pytest.raises(
        ValueError,
        match="FAILED experiment records require error_message",
    ):

        make_record(
            status=(
                RUN_STATUS_FAILED
            ),
            error_message=None,
        )


def test_failed_record_accepts_error_message():

    record = (
        make_record(
            status=(
                RUN_STATUS_FAILED
            ),
            error_message=(
                "training failed"
            ),
        )
    )

    assert (
        record.error_message
        ==
        "training failed"
    )


def test_successful_record_rejects_error_message():

    with pytest.raises(
        ValueError,
        match="Only FAILED",
    ):

        make_record(
            error_message=(
                "should not be here"
            )
        )


# ================================================================
# TAG / MAPPING VALIDATION
# ================================================================


def test_duplicate_tags_fail_closed():

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):

        make_record(
            tags=(
                "production",
                "production",
            )
        )


def test_single_string_is_not_valid_tags_collection():

    with pytest.raises(
        TypeError,
        match="not a single string",
    ):

        make_record(
            tags="production"
        )


def test_parameter_keys_must_be_strings():

    with pytest.raises(
        TypeError,
        match="parameters keys must be strings",
    ):

        make_record(
            parameters={
                1:
                    "invalid",
            }
        )


# ================================================================
# SERIALISABLE DICTIONARY
# ================================================================


def test_record_to_dict_has_exact_schema():

    output = (
        experiment_run_to_dict(
            make_record()
        )
    )

    assert tuple(
        output.keys()
    ) == EXPERIMENT_RUN_FIELDS


def test_record_to_dict_converts_collections():

    output = (
        experiment_run_to_dict(
            make_record()
        )
    )

    assert isinstance(
        output[
            "parameters"
        ],
        dict,
    )

    assert isinstance(
        output[
            "metrics"
        ],
        dict,
    )

    assert isinstance(
        output[
            "tags"
        ],
        list,
    )


def test_record_to_dict_serialises_timestamps():

    output = (
        experiment_run_to_dict(
            make_record()
        )
    )

    assert (
        output[
            "started_at_utc"
        ]
        ==
        "2026-10-27T10:00:00+00:00"
    )

    assert (
        output[
            "completed_at_utc"
        ]
        ==
        "2026-10-27T10:05:00+00:00"
    )


def test_record_to_dict_rejects_wrong_type():

    with pytest.raises(
        TypeError,
        match="ExperimentRunRecord",
    ):

        experiment_run_to_dict(
            {}
        )