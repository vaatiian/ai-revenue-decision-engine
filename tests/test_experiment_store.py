"""
Tests for the append-only JSONL experiment store.
"""

from datetime import datetime, timezone
import json

import pytest

from src.experiments.record import (
    ExperimentRunRecord,
)
from src.experiments.spec import (
    EXPERIMENT_SCHEMA_VERSION,
    RUN_KIND_EVALUATION,
    RUN_STATUS_SUCCEEDED,
)
from src.experiments.store import (
    append_experiment_run,
    get_experiment_run,
    list_experiment_runs,
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
    run_id="run-001",
):

    return ExperimentRunRecord(
        schema_version=(
            EXPERIMENT_SCHEMA_VERSION
        ),

        run_id=run_id,

        experiment_name=(
            "reactivation-final-evaluation"
        ),

        run_kind=(
            RUN_KIND_EVALUATION
        ),

        status=(
            RUN_STATUS_SUCCEEDED
        ),

        score_id=(
            "reactivation"
        ),

        model_name=(
            "Full Random Forest"
        ),

        model_version=(
            "1.0.0"
        ),

        model_type=(
            "classification"
        ),

        target_name=(
            "reactivation_within_next_90_days"
        ),

        started_at_utc=(
            datetime(
                2026,
                10,
                27,
                10,
                0,
                tzinfo=timezone.utc,
            )
        ),

        completed_at_utc=(
            datetime(
                2026,
                10,
                27,
                10,
                5,
                tzinfo=timezone.utc,
            )
        ),

        git_commit=(
            "61424ff89ecf21325f7f7d10fee191fff5c1a804"
        ),

        git_branch=(
            "main"
        ),

        manifest_path=(
            "artifacts/models/reactivation/"
            "reactivation_full_random_forest_v1.0.0.json"
        ),

        manifest_sha256=(
            SHA_A
        ),

        artifact_path=(
            "artifacts/models/reactivation/"
            "reactivation_full_random_forest_v1.0.0.joblib"
        ),

        artifact_sha256=(
            SHA_B
        ),

        data_reference=(
            "cleaned_online_retail.parquet"
        ),

        data_sha256=(
            SHA_C
        ),

        parameters={
            "split":
                "test",
        },

        metrics={
            "pr_auc":
                0.615,
        },

        tags=(
            "production",
            "temporal",
        ),

        notes=None,

        error_message=None,
    )


# ================================================================
# EMPTY / MISSING STORE
# ================================================================


def test_missing_store_returns_empty_tuple(
    tmp_path,
):

    path = (
        tmp_path
        /
        "missing.jsonl"
    )

    assert (
        list_experiment_runs(
            path
        )
        ==
        ()
    )


# ================================================================
# APPEND
# ================================================================


def test_append_creates_store(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    result = (
        append_experiment_run(
            make_record(),
            path,
        )
    )

    assert result == path
    assert path.exists()
    assert path.is_file()


def test_append_creates_parent_directory(
    tmp_path,
):

    path = (
        tmp_path
        /
        "nested"
        /
        "runs.jsonl"
    )

    append_experiment_run(
        make_record(),
        path,
    )

    assert path.exists()


def test_append_writes_one_json_object_per_line(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    append_experiment_run(
        make_record(),
        path,
    )

    lines = (
        path.read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert len(
        lines
    ) == 1

    payload = json.loads(
        lines[0]
    )

    assert (
        payload[
            "run_id"
        ]
        ==
        "run-001"
    )


def test_multiple_appends_preserve_order(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    append_experiment_run(
        make_record(
            "run-001"
        ),
        path,
    )

    append_experiment_run(
        make_record(
            "run-002"
        ),
        path,
    )

    records = (
        list_experiment_runs(
            path
        )
    )

    assert [
        record.run_id
        for record
        in records
    ] == [
        "run-001",
        "run-002",
    ]


def test_duplicate_run_id_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    append_experiment_run(
        make_record(),
        path,
    )

    with pytest.raises(
        ValueError,
        match="already exists",
    ):

        append_experiment_run(
            make_record(),
            path,
        )


def test_append_rejects_wrong_record_type(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    with pytest.raises(
        TypeError,
        match="ExperimentRunRecord",
    ):

        append_experiment_run(
            {},
            path,
        )


# ================================================================
# READ / ROUND TRIP
# ================================================================


def test_round_trip_preserves_record(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    expected = (
        make_record()
    )

    append_experiment_run(
        expected,
        path,
    )

    records = (
        list_experiment_runs(
            path
        )
    )

    assert len(
        records
    ) == 1

    actual = records[0]

    assert (
        actual.run_id
        ==
        expected.run_id
    )

    assert (
        actual.parameters
        ==
        expected.parameters
    )

    assert (
        actual.metrics
        ==
        expected.metrics
    )

    assert (
        actual.tags
        ==
        expected.tags
    )


def test_get_experiment_run_returns_exact_record(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    append_experiment_run(
        make_record(
            "run-001"
        ),
        path,
    )

    append_experiment_run(
        make_record(
            "run-002"
        ),
        path,
    )

    record = (
        get_experiment_run(
            "run-002",
            path,
        )
    )

    assert (
        record.run_id
        ==
        "run-002"
    )


def test_get_unknown_run_id_fails(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    append_experiment_run(
        make_record(),
        path,
    )

    with pytest.raises(
        KeyError,
        match="Unknown experiment run_id",
    ):

        get_experiment_run(
            "missing",
            path,
        )


def test_get_rejects_empty_run_id(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):

        get_experiment_run(
            "",
            path,
        )


# ================================================================
# FAIL-CLOSED STORE INTEGRITY
# ================================================================


def test_invalid_json_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    path.write_text(
        "{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):

        list_experiment_runs(
            path
        )


def test_empty_line_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    path.write_text(
        "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="empty line",
    ):

        list_experiment_runs(
            path
        )


def test_non_object_json_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    path.write_text(
        "[]\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid record",
    ):

        list_experiment_runs(
            path
        )


def test_schema_drift_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    payload = {
        "run_id":
            "run-001",
    }

    path.write_text(
        json.dumps(
            payload
        )
        +
        "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid record",
    ):

        list_experiment_runs(
            path
        )


def test_duplicate_run_ids_in_existing_store_fail_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    append_experiment_run(
        make_record(),
        path,
    )

    line = (
        path.read_text(
            encoding="utf-8"
        )
        .splitlines()[
            0
        ]
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            line
        )

        handle.write(
            "\n"
        )

    with pytest.raises(
        ValueError,
        match="duplicate run_id",
    ):

        list_experiment_runs(
            path
        )


def test_directory_path_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "directory"
    )

    path.mkdir()

    with pytest.raises(
        ValueError,
        match="must be a file",
    ):

        list_experiment_runs(
            path
        )