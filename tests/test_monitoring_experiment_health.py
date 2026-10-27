"""
Tests for experiment-store health monitoring.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import src.monitoring.experiment_health as experiment_health_module

from src.experiments.spec import (
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.experiment_health import (
    EXPERIMENT_STORE_CHECK_NAME,
    monitor_experiment_store,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_EXPERIMENT_STORE,
)


OBSERVED_AT = datetime(
    2026,
    10,
    27,
    19,
    0,
    tzinfo=timezone.utc,
)


def fake_record(
    run_id,
    score_id,
    status,
):

    return SimpleNamespace(
        run_id=(
            run_id
        ),

        score_id=(
            score_id
        ),

        status=(
            status
        ),
    )


# ================================================================
# VALID NON-EMPTY STORE
# ================================================================


def test_valid_nonempty_store_is_healthy(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "experiments.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (
                fake_record(
                    "run-001",
                    "reactivation",
                    RUN_STATUS_SUCCEEDED,
                ),
            ),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert isinstance(
        result,
        MonitoringCheck,
    )

    assert (
        result.health
        ==
        MONITORING_HEALTHY
    )

    assert (
        result.component
        ==
        MONITOR_COMPONENT_EXPERIMENT_STORE
    )

    assert (
        result.check_name
        ==
        EXPERIMENT_STORE_CHECK_NAME
    )


def test_valid_store_metrics_are_correct(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "experiments.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (
                fake_record(
                    "run-001",
                    "reactivation",
                    RUN_STATUS_SUCCEEDED,
                ),
                fake_record(
                    "run-002",
                    "revenue_leakage",
                    RUN_STATUS_FAILED,
                ),
                fake_record(
                    "run-003",
                    "future_customer_value",
                    RUN_STATUS_RUNNING,
                ),
            ),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert (
        result.metrics[
            "store_exists"
        ]
        is True
    )

    assert (
        result.metrics[
            "total_runs"
        ]
        ==
        3
    )

    assert (
        result.metrics[
            "status_counts"
        ][
            RUN_STATUS_SUCCEEDED
        ]
        ==
        1
    )

    assert (
        result.metrics[
            "status_counts"
        ][
            RUN_STATUS_FAILED
        ]
        ==
        1
    )

    assert (
        result.metrics[
            "status_counts"
        ][
            RUN_STATUS_RUNNING
        ]
        ==
        1
    )


# ================================================================
# HISTORICAL STATUS POLICY
# ================================================================


@pytest.mark.parametrize(
    "status",
    (
        RUN_STATUS_RUNNING,
        RUN_STATUS_SUCCEEDED,
        RUN_STATUS_FAILED,
    ),
)
def test_historical_status_does_not_independently_make_store_unhealthy(
    monkeypatch,
    tmp_path,
    status,
):

    path = (
        tmp_path
        /
        "experiments.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (
                fake_record(
                    "run-001",
                    "reactivation",
                    status,
                ),
            ),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert (
        result.health
        ==
        MONITORING_HEALTHY
    )


def test_failed_runs_are_counted_as_history(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "experiments.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (
                fake_record(
                    "run-001",
                    "reactivation",
                    RUN_STATUS_FAILED,
                ),
                fake_record(
                    "run-002",
                    "reactivation",
                    RUN_STATUS_FAILED,
                ),
            ),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert (
        result.health
        ==
        MONITORING_HEALTHY
    )

    assert (
        result.metrics[
            "status_counts"
        ][
            RUN_STATUS_FAILED
        ]
        ==
        2
    )


# ================================================================
# MISSING / EMPTY
# ================================================================


def test_missing_store_is_degraded(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "missing.jsonl"
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert (
        result.health
        ==
        MONITORING_DEGRADED
    )

    assert (
        result.metrics[
            "store_exists"
        ]
        is False
    )

    assert (
        result.metrics[
            "total_runs"
        ]
        ==
        0
    )


def test_existing_empty_store_is_degraded(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "empty.jsonl"
    )

    path.touch()

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert (
        result.health
        ==
        MONITORING_DEGRADED
    )

    assert (
        result.metrics[
            "store_exists"
        ]
        is True
    )


# ================================================================
# MALFORMED STORE
# ================================================================


@pytest.mark.parametrize(
    "exception",
    (
        ValueError(
            "schema drift"
        ),
        RuntimeError(
            "duplicate run id"
        ),
        OSError(
            "read failure"
        ),
    ),
)
def test_store_validation_failure_is_unhealthy(
    monkeypatch,
    tmp_path,
    exception,
):

    path = (
        tmp_path
        /
        "broken.jsonl"
    )

    path.write_text(
        "broken\n",
        encoding="utf-8",
    )

    def fail(
        path,
    ):

        raise exception

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        fail,
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert (
        result.health
        ==
        MONITORING_UNHEALTHY
    )

    assert (
        result.metrics[
            "store_exists"
        ]
        is True
    )

    assert (
        result.metrics[
            "total_runs"
        ]
        ==
        0
    )


def test_unhealthy_check_preserves_failure_detail(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "broken.jsonl"
    )

    path.write_text(
        "broken\n",
        encoding="utf-8",
    )

    def fail(
        path,
    ):

        raise ValueError(
            "invalid experiment record"
        )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        fail,
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert len(
        result.details
    ) == 1

    assert (
        "ValueError"
        in
        result.details[
            0
        ]
    )

    assert (
        "invalid experiment record"
        in
        result.details[
            0
        ]
    )


# ================================================================
# STATUS COUNTS
# ================================================================


def test_status_counts_include_all_frozen_statuses(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "experiments.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (
                fake_record(
                    "run-001",
                    "reactivation",
                    RUN_STATUS_SUCCEEDED,
                ),
            ),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert set(
        result.metrics[
            "status_counts"
        ].keys()
    ) == {
        RUN_STATUS_RUNNING,
        RUN_STATUS_SUCCEEDED,
        RUN_STATUS_FAILED,
    }


def test_status_counts_are_read_only(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "experiments.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (
                fake_record(
                    "run-001",
                    "reactivation",
                    RUN_STATUS_SUCCEEDED,
                ),
            ),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    with pytest.raises(
        TypeError
    ):

        result.metrics[
            "status_counts"
        ][
            RUN_STATUS_SUCCEEDED
        ] = 99


# ================================================================
# DETAILS / SUMMARY
# ================================================================


def test_valid_history_details_include_run_identity(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "experiments.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (
                fake_record(
                    "run-001",
                    "reactivation",
                    RUN_STATUS_SUCCEEDED,
                ),
            ),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert (
        result.details
        ==
        (
            "run-001: reactivation — SUCCEEDED",
        )
    )


def test_valid_history_summary_reports_run_count(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "experiments.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_health_module,
        "list_experiment_runs",
        lambda path:
            (
                fake_record(
                    "run-001",
                    "reactivation",
                    RUN_STATUS_SUCCEEDED,
                ),
                fake_record(
                    "run-002",
                    "revenue_leakage",
                    RUN_STATUS_SUCCEEDED,
                ),
            ),
    )

    result = (
        monitor_experiment_store(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert (
        result.summary
        ==
        "Experiment store is valid with 2 historical runs."
    )