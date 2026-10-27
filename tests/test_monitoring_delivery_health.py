"""
Tests for delivery-audit health monitoring.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import src.monitoring.delivery_health as delivery_health_module

from src.delivery.spec import (
    DELIVERY_DISPOSITION_HOLD_CONTROL,
    DELIVERY_DISPOSITION_SUPPRESSED,
    DELIVERY_DISPOSITION_TREATMENT,
    DELIVERY_DISPOSITIONS,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.delivery_health import (
    DELIVERY_AUDIT_CHECK_NAME,
    monitor_delivery_audit,
)

from src.monitoring.spec import (
    MONITORING_DEGRADED,
    MONITORING_HEALTHY,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_DELIVERY_AUDIT,
)


OBSERVED_AT = datetime(
    2026,
    10,
    27,
    20,
    0,
    tzinfo=timezone.utc,
)


def fake_payload(
    delivery_id,
    customer,
    disposition,
):

    return SimpleNamespace(
        delivery_id=(
            delivery_id
        ),

        customer=(
            customer
        ),

        delivery_disposition=(
            disposition
        ),
    )


# ================================================================
# HEALTHY
# ================================================================


def test_valid_nonempty_audit_is_healthy(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        delivery_health_module,
        "list_delivery_audit",
        lambda path:
            (
                fake_payload(
                    "delivery-001",
                    "CUST-001",
                    DELIVERY_DISPOSITION_TREATMENT,
                ),
            ),
    )

    result = (
        monitor_delivery_audit(
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
        MONITOR_COMPONENT_DELIVERY_AUDIT
    )

    assert (
        result.check_name
        ==
        DELIVERY_AUDIT_CHECK_NAME
    )


def test_delivery_metrics_are_correct(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        delivery_health_module,
        "list_delivery_audit",
        lambda path:
            (
                fake_payload(
                    "delivery-001",
                    "CUST-001",
                    DELIVERY_DISPOSITION_TREATMENT,
                ),
                fake_payload(
                    "delivery-002",
                    "CUST-002",
                    DELIVERY_DISPOSITION_HOLD_CONTROL,
                ),
                fake_payload(
                    "delivery-003",
                    "CUST-003",
                    DELIVERY_DISPOSITION_SUPPRESSED,
                ),
            ),
    )

    result = (
        monitor_delivery_audit(
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
            "audit_exists"
        ]
        is True
    )

    assert (
        result.metrics[
            "total_deliveries"
        ]
        ==
        3
    )

    counts = (
        result.metrics[
            "disposition_counts"
        ]
    )

    assert (
        counts[
            DELIVERY_DISPOSITION_TREATMENT
        ]
        ==
        1
    )

    assert (
        counts[
            DELIVERY_DISPOSITION_HOLD_CONTROL
        ]
        ==
        1
    )

    assert (
        counts[
            DELIVERY_DISPOSITION_SUPPRESSED
        ]
        ==
        1
    )


# ================================================================
# DEGRADED
# ================================================================


def test_missing_audit_is_degraded(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "missing.jsonl"
    )

    monkeypatch.setattr(
        delivery_health_module,
        "list_delivery_audit",
        lambda path:
            (),
    )

    result = (
        monitor_delivery_audit(
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
            "audit_exists"
        ]
        is False
    )


def test_existing_empty_audit_is_degraded(
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
        delivery_health_module,
        "list_delivery_audit",
        lambda path:
            (),
    )

    result = (
        monitor_delivery_audit(
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
            "audit_exists"
        ]
        is True
    )


# ================================================================
# UNHEALTHY
# ================================================================


@pytest.mark.parametrize(
    "exception",
    (
        ValueError(
            "invalid delivery audit"
        ),
        RuntimeError(
            "duplicate delivery id"
        ),
        OSError(
            "read failure"
        ),
    ),
)
def test_audit_failure_is_unhealthy(
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
        delivery_health_module,
        "list_delivery_audit",
        fail,
    )

    result = (
        monitor_delivery_audit(
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


def test_unhealthy_detail_is_preserved(
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
            "synthetic audit corruption"
        )

    monkeypatch.setattr(
        delivery_health_module,
        "list_delivery_audit",
        fail,
    )

    result = (
        monitor_delivery_audit(
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
        "synthetic audit corruption"
        in
        result.details[
            0
        ]
    )


# ================================================================
# DISPOSITION COUNTS
# ================================================================


def test_disposition_counts_cover_full_contract(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        delivery_health_module,
        "list_delivery_audit",
        lambda path:
            (
                fake_payload(
                    "delivery-001",
                    "CUST-001",
                    DELIVERY_DISPOSITION_TREATMENT,
                ),
            ),
    )

    result = (
        monitor_delivery_audit(
            observed_at_utc=(
                OBSERVED_AT
            ),
            path=(
                path
            ),
        )
    )

    assert tuple(
        result.metrics[
            "disposition_counts"
        ].keys()
    ) == DELIVERY_DISPOSITIONS


def test_disposition_counts_are_read_only(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        delivery_health_module,
        "list_delivery_audit",
        lambda path:
            (
                fake_payload(
                    "delivery-001",
                    "CUST-001",
                    DELIVERY_DISPOSITION_TREATMENT,
                ),
            ),
    )

    result = (
        monitor_delivery_audit(
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
            "disposition_counts"
        ][
            DELIVERY_DISPOSITION_TREATMENT
        ] = 99


# ================================================================
# DETAILS / SUMMARY
# ================================================================


def test_healthy_details_include_delivery_identity(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        delivery_health_module,
        "list_delivery_audit",
        lambda path:
            (
                fake_payload(
                    "delivery-001",
                    "CUST-001",
                    DELIVERY_DISPOSITION_TREATMENT,
                ),
            ),
    )

    result = (
        monitor_delivery_audit(
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
            "delivery-001: CUST-001 — TREATMENT",
        )
    )


def test_healthy_summary_reports_delivery_count(
    monkeypatch,
    tmp_path,
):

    path = (
        tmp_path
        /
        "audit.jsonl"
    )

    path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        delivery_health_module,
        "list_delivery_audit",
        lambda path:
            (
                fake_payload(
                    "delivery-001",
                    "CUST-001",
                    DELIVERY_DISPOSITION_TREATMENT,
                ),
                fake_payload(
                    "delivery-002",
                    "CUST-002",
                    DELIVERY_DISPOSITION_SUPPRESSED,
                ),
            ),
    )

    result = (
        monitor_delivery_audit(
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
        "Delivery audit is valid with 2 delivery records."
    )