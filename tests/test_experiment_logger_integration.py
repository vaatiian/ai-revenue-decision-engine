"""
End-to-end integration tests for governed experiment logging.
"""

from datetime import datetime, timezone
from uuid import UUID

import pytest

from src.experiments.logger import (
    build_experiment_run,
    generate_run_id,
    log_experiment_run,
)
from src.experiments.record import (
    ExperimentRunRecord,
)
from src.experiments.spec import (
    RUN_KIND_EVALUATION,
    RUN_KIND_PRODUCTION_VALIDATION,
    RUN_STATUS_FAILED,
    RUN_STATUS_SUCCEEDED,
)
from src.experiments.store import (
    get_experiment_run,
    list_experiment_runs,
)
from src.experiments.validation import (
    validate_record_against_current_score,
)


GIT_COMMIT = (
    "61424ff89ecf21325f7f7d10fee191fff5c1a804"
)

START = datetime(
    2026,
    10,
    27,
    13,
    0,
    tzinfo=timezone.utc,
)

END = datetime(
    2026,
    10,
    27,
    13,
    5,
    tzinfo=timezone.utc,
)


# ================================================================
# RUN-ID GENERATION
# ================================================================


def test_generated_run_id_is_valid_uuid():

    run_id = (
        generate_run_id()
    )

    parsed = UUID(
        run_id
    )

    assert (
        str(
            parsed
        )
        ==
        run_id
    )


def test_generated_run_ids_are_unique():

    first = (
        generate_run_id()
    )

    second = (
        generate_run_id()
    )

    assert (
        first
        !=
        second
    )


# ================================================================
# RECORD BUILDING
# ================================================================


@pytest.mark.parametrize(
    "score_id",
    (
        "reactivation",
        "revenue_leakage",
        "future_customer_value",
        "repeat_purchase",
        "recommendation",
        "demand_forecast",
    ),
)
def test_all_six_scores_build_governed_run_record(
    score_id,
):

    record = (
        build_experiment_run(
            score_id=(
                score_id
            ),

            experiment_name=(
                f"{score_id}-validation"
            ),

            run_kind=(
                RUN_KIND_PRODUCTION_VALIDATION
            ),

            status=(
                RUN_STATUS_SUCCEEDED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
            ),

            parameters={},

            metrics={},

            tags=(
                "integration",
            ),

            run_id=(
                f"{score_id}-run"
            ),
        )
    )

    assert isinstance(
        record,
        ExperimentRunRecord,
    )

    assert (
        record.score_id
        ==
        score_id
    )

    assert (
        validate_record_against_current_score(
            record
        )
        is None
    )


def test_fitted_run_contains_artifact_provenance():

    record = (
        build_experiment_run(
            score_id=(
                "reactivation"
            ),

            experiment_name=(
                "reactivation-evaluation"
            ),

            run_kind=(
                RUN_KIND_EVALUATION
            ),

            status=(
                RUN_STATUS_SUCCEEDED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
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
                "fitted",
            ),

            run_id=(
                "fitted-run"
            ),
        )
    )

    assert (
        record.artifact_path
        is not None
    )

    assert (
        record.artifact_sha256
        is not None
    )


def test_deterministic_run_has_no_artifact():

    record = (
        build_experiment_run(
            score_id=(
                "recommendation"
            ),

            experiment_name=(
                "recommendation-validation"
            ),

            run_kind=(
                RUN_KIND_PRODUCTION_VALIDATION
            ),

            status=(
                RUN_STATUS_SUCCEEDED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
            ),

            parameters={
                "top_n":
                    20,
            },

            metrics={},

            tags=(
                "deterministic",
            ),

            run_id=(
                "deterministic-run"
            ),
        )
    )

    assert (
        record.artifact_path
        is None
    )

    assert (
        record.artifact_sha256
        is None
    )


# ================================================================
# END-TO-END PERSISTENCE
# ================================================================


def test_fitted_run_logs_and_round_trips(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    logged = (
        log_experiment_run(
            score_id=(
                "reactivation"
            ),

            experiment_name=(
                "reactivation-final-evaluation"
            ),

            run_kind=(
                RUN_KIND_EVALUATION
            ),

            status=(
                RUN_STATUS_SUCCEEDED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
            ),

            parameters={
                "evaluation_split":
                    "test",
            },

            metrics={
                "pr_auc":
                    0.615,
            },

            tags=(
                "integration",
                "fitted",
            ),

            run_id=(
                "reactivation-e2e"
            ),

            store_path=(
                path
            ),
        )
    )

    loaded = (
        get_experiment_run(
            "reactivation-e2e",
            path,
        )
    )

    assert (
        loaded.run_id
        ==
        logged.run_id
    )

    assert (
        loaded.score_id
        ==
        "reactivation"
    )

    assert (
        loaded.metrics[
            "pr_auc"
        ]
        ==
        0.615
    )

    assert (
        validate_record_against_current_score(
            loaded
        )
        is None
    )


def test_deterministic_run_logs_and_round_trips(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    logged = (
        log_experiment_run(
            score_id=(
                "demand_forecast"
            ),

            experiment_name=(
                "demand-production-validation"
            ),

            run_kind=(
                RUN_KIND_PRODUCTION_VALIDATION
            ),

            status=(
                RUN_STATUS_SUCCEEDED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
            ),

            parameters={
                "horizons":
                    [
                        1,
                        2,
                        4,
                    ],
            },

            metrics={
                "wape":
                    0.0916,
            },

            tags=(
                "integration",
                "deterministic",
            ),

            run_id=(
                "demand-e2e"
            ),

            store_path=(
                path
            ),
        )
    )

    loaded = (
        get_experiment_run(
            "demand-e2e",
            path,
        )
    )

    assert (
        loaded.run_id
        ==
        logged.run_id
    )

    assert (
        loaded.score_id
        ==
        "demand_forecast"
    )

    assert (
        loaded.artifact_path
        is None
    )

    assert (
        loaded.artifact_sha256
        is None
    )


def test_multiple_score_families_share_one_store(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    for score_id in (
        "reactivation",
        "recommendation",
        "demand_forecast",
    ):

        log_experiment_run(
            score_id=(
                score_id
            ),

            experiment_name=(
                f"{score_id}-validation"
            ),

            run_kind=(
                RUN_KIND_PRODUCTION_VALIDATION
            ),

            status=(
                RUN_STATUS_SUCCEEDED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
            ),

            parameters={},

            metrics={},

            tags=(
                "integration",
            ),

            run_id=(
                f"{score_id}-run"
            ),

            store_path=(
                path
            ),
        )

    records = (
        list_experiment_runs(
            path
        )
    )

    assert [
        record.score_id
        for record
        in records
    ] == [
        "reactivation",
        "recommendation",
        "demand_forecast",
    ]


# ================================================================
# FAILED RUNS
# ================================================================


def test_failed_run_can_be_logged(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    record = (
        log_experiment_run(
            score_id=(
                "reactivation"
            ),

            experiment_name=(
                "reactivation-failed-run"
            ),

            run_kind=(
                RUN_KIND_EVALUATION
            ),

            status=(
                RUN_STATUS_FAILED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
            ),

            parameters={},

            metrics={},

            tags=(
                "failure-test",
            ),

            error_message=(
                "Synthetic evaluation failure."
            ),

            run_id=(
                "failed-run"
            ),

            store_path=(
                path
            ),
        )
    )

    assert (
        record.status
        ==
        RUN_STATUS_FAILED
    )

    assert (
        record.error_message
        ==
        "Synthetic evaluation failure."
    )


# ================================================================
# FAIL-CLOSED INTEGRATION
# ================================================================


def test_unknown_score_is_not_persisted(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    with pytest.raises(
        KeyError,
        match="Unknown score_id",
    ):

        log_experiment_run(
            score_id=(
                "unknown_score"
            ),

            experiment_name=(
                "invalid"
            ),

            run_kind=(
                RUN_KIND_EVALUATION
            ),

            status=(
                RUN_STATUS_SUCCEEDED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
            ),

            parameters={},

            metrics={},

            tags=(),

            run_id=(
                "invalid-run"
            ),

            store_path=(
                path
            ),
        )

    assert not path.exists()


def test_invalid_git_commit_is_not_persisted(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    with pytest.raises(
        ValueError,
        match="40 hexadecimal",
    ):

        log_experiment_run(
            score_id=(
                "reactivation"
            ),

            experiment_name=(
                "invalid-git"
            ),

            run_kind=(
                RUN_KIND_EVALUATION
            ),

            status=(
                RUN_STATUS_SUCCEEDED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                "61424ff"
            ),

            git_branch=(
                "main"
            ),

            parameters={},

            metrics={},

            tags=(),

            run_id=(
                "invalid-git-run"
            ),

            store_path=(
                path
            ),
        )

    assert not path.exists()


def test_duplicate_run_id_fails_closed(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    common = {
        "score_id":
            "recommendation",

        "experiment_name":
            "recommendation-validation",

        "run_kind":
            RUN_KIND_PRODUCTION_VALIDATION,

        "status":
            RUN_STATUS_SUCCEEDED,

        "started_at_utc":
            START,

        "completed_at_utc":
            END,

        "git_commit":
            GIT_COMMIT,

        "git_branch":
            "main",

        "parameters":
            {},

        "metrics":
            {},

        "tags":
            (
                "integration",
            ),

        "run_id":
            "duplicate-run",

        "store_path":
            path,
    }

    log_experiment_run(
        **common
    )

    with pytest.raises(
        ValueError,
        match="already exists",
    ):

        log_experiment_run(
            **common
        )

    records = (
        list_experiment_runs(
            path
        )
    )

    assert len(
        records
    ) == 1


def test_invalid_failed_record_is_not_persisted(
    tmp_path,
):

    path = (
        tmp_path
        /
        "runs.jsonl"
    )

    with pytest.raises(
        ValueError,
        match="require error_message",
    ):

        log_experiment_run(
            score_id=(
                "reactivation"
            ),

            experiment_name=(
                "broken-failed-run"
            ),

            run_kind=(
                RUN_KIND_EVALUATION
            ),

            status=(
                RUN_STATUS_FAILED
            ),

            started_at_utc=(
                START
            ),

            completed_at_utc=(
                END
            ),

            git_commit=(
                GIT_COMMIT
            ),

            git_branch=(
                "main"
            ),

            parameters={},

            metrics={},

            tags=(),

            error_message=None,

            run_id=(
                "broken-run"
            ),

            store_path=(
                path
            ),
        )

    assert not path.exists()