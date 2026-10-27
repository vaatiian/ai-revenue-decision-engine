"""
Tests for Decision Engine output health monitoring.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

import src.monitoring.decision_health as decision_health_module

from src.decision_engine.spec import (
    DECISION_OUTPUT_COLUMNS,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.decision_health import (
    DECISION_OUTPUT_CHECK_NAME,
    monitor_decision_output,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_DECISION_OUTPUT,
)


OBSERVED_AT = datetime(
    2026,
    10,
    27,
    19,
    30,
    tzinfo=timezone.utc,
)


def make_nonempty_frame():

    values = {
        column:
            None
        for column
        in DECISION_OUTPUT_COLUMNS
    }

    values[
        "customer"
    ] = "CUST-001"

    values[
        "decision_date"
    ] = "2011-08-31"

    return pd.DataFrame(
        [
            values,
        ],
        columns=(
            DECISION_OUTPUT_COLUMNS
        ),
    )


# ================================================================
# HEALTHY
# ================================================================


def test_valid_nonempty_output_is_healthy(
    monkeypatch,
):

    frame = (
        make_nonempty_frame()
    )

    monkeypatch.setattr(
        decision_health_module,
        "validate_decision_engine_output",
        lambda output, decision_date=None:
            None,
    )

    result = (
        monitor_decision_output(
            frame,
            observed_at_utc=(
                OBSERVED_AT
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
        MONITOR_COMPONENT_DECISION_OUTPUT
    )

    assert (
        result.check_name
        ==
        DECISION_OUTPUT_CHECK_NAME
    )


def test_healthy_metrics_are_correct(
    monkeypatch,
):

    frame = (
        make_nonempty_frame()
    )

    monkeypatch.setattr(
        decision_health_module,
        "validate_decision_engine_output",
        lambda output, decision_date=None:
            None,
    )

    result = (
        monitor_decision_output(
            frame,
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert (
        result.metrics[
            "row_count"
        ]
        ==
        1
    )

    assert (
        result.metrics[
            "unique_customers"
        ]
        ==
        1
    )

    assert (
        result.metrics[
            "decision_date"
        ]
        ==
        "2011-08-31"
    )


def test_decision_date_is_forwarded(
    monkeypatch,
):

    observed = {}

    frame = (
        make_nonempty_frame()
    )

    def fake_validate(
        output,
        decision_date=None,
    ):

        observed[
            "decision_date"
        ] = (
            decision_date
        )

    monkeypatch.setattr(
        decision_health_module,
        "validate_decision_engine_output",
        fake_validate,
    )

    monitor_decision_output(
        frame,
        observed_at_utc=(
            OBSERVED_AT
        ),
        decision_date=(
            "2011-08-31"
        ),
    )

    assert (
        observed[
            "decision_date"
        ]
        ==
        "2011-08-31"
    )


# ================================================================
# DEGRADED EMPTY OUTPUT
# ================================================================


def test_real_empty_canonical_output_is_degraded():

    frame = pd.DataFrame(
        columns=(
            DECISION_OUTPUT_COLUMNS
        )
    )

    result = (
        monitor_decision_output(
            frame,
            observed_at_utc=(
                OBSERVED_AT
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
            "row_count"
        ]
        ==
        0
    )

    assert (
        result.metrics[
            "unique_customers"
        ]
        ==
        0
    )


# ================================================================
# UNHEALTHY
# ================================================================


@pytest.mark.parametrize(
    "exception",
    (
        TypeError(
            "invalid type"
        ),
        ValueError(
            "policy violation"
        ),
        RuntimeError(
            "validation failure"
        ),
    ),
)
def test_validation_failure_is_unhealthy(
    monkeypatch,
    exception,
):

    frame = (
        make_nonempty_frame()
    )

    def fail(
        output,
        decision_date=None,
    ):

        raise exception

    monkeypatch.setattr(
        decision_health_module,
        "validate_decision_engine_output",
        fail,
    )

    result = (
        monitor_decision_output(
            frame,
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert (
        result.health
        ==
        MONITORING_UNHEALTHY
    )


def test_non_dataframe_is_unhealthy():

    result = (
        monitor_decision_output(
            {},
            observed_at_utc=(
                OBSERVED_AT
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
            "row_count"
        ]
        ==
        0
    )


def test_schema_drift_is_unhealthy():

    frame = pd.DataFrame(
        {
            "customer":
                [
                    "CUST-001",
                ],
        }
    )

    result = (
        monitor_decision_output(
            frame,
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert (
        result.health
        ==
        MONITORING_UNHEALTHY
    )


def test_validation_failure_detail_is_preserved(
    monkeypatch,
):

    frame = (
        make_nonempty_frame()
    )

    def fail(
        output,
        decision_date=None,
    ):

        raise ValueError(
            "synthetic policy violation"
        )

    monkeypatch.setattr(
        decision_health_module,
        "validate_decision_engine_output",
        fail,
    )

    result = (
        monitor_decision_output(
            frame,
            observed_at_utc=(
                OBSERVED_AT
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
        "synthetic policy violation"
        in
        result.details[
            0
        ]
    )


# ================================================================
# SUMMARY
# ================================================================


def test_healthy_summary_reports_customer_count(
    monkeypatch,
):

    frame = (
        make_nonempty_frame()
    )

    monkeypatch.setattr(
        decision_health_module,
        "validate_decision_engine_output",
        lambda output, decision_date=None:
            None,
    )

    result = (
        monitor_decision_output(
            frame,
            observed_at_utc=(
                OBSERVED_AT
            ),
        )
    )

    assert (
        result.summary
        ==
        "Decision Engine output is valid for 1 customer."
    )