"""
Tests for production score-runtime monitoring.
"""

from datetime import datetime, timezone

import pytest

import src.monitoring.score_health as score_health_module

from src.models.score_availability import (
    ScoreAvailability,
    ScoreAvailabilityStatus,
)

from src.monitoring.check import (
    MonitoringCheck,
)

from src.monitoring.score_health import (
    SCORE_RUNTIME_CHECK_NAME,
    monitor_score_runtime,
)

from src.monitoring.spec import (
    MONITORING_HEALTHY,
    MONITORING_UNHEALTHY,
    MONITOR_COMPONENT_SCORE_RUNTIME,
)


OBSERVED_AT = datetime(
    2026,
    10,
    27,
    18,
    30,
    tzinfo=timezone.utc,
)


def available(
    score_id,
):

    return ScoreAvailability(
        score_id=(
            score_id
        ),

        status=(
            ScoreAvailabilityStatus.AVAILABLE
        ),

        available=True,

        artifact_exists=True,

        manifest_exists=True,

        detail=(
            "Score is available."
        ),
    )


def unavailable(
    score_id,
    status=(
        ScoreAvailabilityStatus.MISSING_ARTIFACT
    ),
):

    return ScoreAvailability(
        score_id=(
            score_id
        ),

        status=(
            status
        ),

        available=False,

        artifact_exists=False,

        manifest_exists=True,

        detail=(
            "Required runtime dependency is unavailable."
        ),
    )


# ================================================================
# HEALTHY
# ================================================================


def test_all_available_scores_are_healthy(
    monkeypatch,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                available(
                    "score_a"
                ),
                available(
                    "score_b"
                ),
            ),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
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
        MONITOR_COMPONENT_SCORE_RUNTIME
    )

    assert (
        result.check_name
        ==
        SCORE_RUNTIME_CHECK_NAME
    )


def test_healthy_metrics_are_correct(
    monkeypatch,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                available(
                    "score_a"
                ),
                available(
                    "score_b"
                ),
                available(
                    "score_c"
                ),
            ),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    assert (
        result.metrics[
            "total_scores"
        ]
        ==
        3
    )

    assert (
        result.metrics[
            "available_scores"
        ]
        ==
        3
    )

    assert (
        result.metrics[
            "unavailable_scores"
        ]
        ==
        0
    )

    assert (
        result.metrics[
            "all_scores_available"
        ]
        is True
    )


# ================================================================
# UNHEALTHY
# ================================================================


def test_one_unavailable_score_is_unhealthy(
    monkeypatch,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                available(
                    "score_a"
                ),
                unavailable(
                    "score_b"
                ),
            ),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    assert (
        result.health
        ==
        MONITORING_UNHEALTHY
    )

    assert (
        result.metrics[
            "available_scores"
        ]
        ==
        1
    )

    assert (
        result.metrics[
            "unavailable_scores"
        ]
        ==
        1
    )

    assert (
        result.metrics[
            "all_scores_available"
        ]
        is False
    )


@pytest.mark.parametrize(
    "status",
    (
        ScoreAvailabilityStatus.MISSING_ARTIFACT,
        ScoreAvailabilityStatus.MISSING_MANIFEST,
        ScoreAvailabilityStatus.INVALID_ARTIFACT,
        ScoreAvailabilityStatus.INVALID_MANIFEST,
        ScoreAvailabilityStatus.UNAVAILABLE,
    ),
)
def test_every_non_available_status_is_unhealthy(
    monkeypatch,
    status,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                unavailable(
                    "score_a",
                    status,
                ),
            ),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    assert (
        result.health
        ==
        MONITORING_UNHEALTHY
    )


def test_empty_score_registry_is_unhealthy(
    monkeypatch,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    assert (
        result.health
        ==
        MONITORING_UNHEALTHY
    )

    assert (
        result.metrics[
            "total_scores"
        ]
        ==
        0
    )


# ================================================================
# STATUS COUNTS
# ================================================================


def test_status_counts_cover_all_availability_states(
    monkeypatch,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                available(
                    "score_a"
                ),
                unavailable(
                    "score_b"
                ),
            ),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    counts = (
        result.metrics[
            "status_counts"
        ]
    )

    assert set(
        counts.keys()
    ) == {
        status.value
        for status
        in ScoreAvailabilityStatus
    }

    assert (
        counts[
            "AVAILABLE"
        ]
        ==
        1
    )

    assert (
        counts[
            "MISSING_ARTIFACT"
        ]
        ==
        1
    )


def test_status_counts_are_read_only(
    monkeypatch,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                available(
                    "score_a"
                ),
            ),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    with pytest.raises(
        TypeError
    ):

        result.metrics[
            "status_counts"
        ][
            "AVAILABLE"
        ] = 99


# ================================================================
# DETAILS / SUMMARY
# ================================================================


def test_details_include_each_score_status(
    monkeypatch,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                available(
                    "score_a"
                ),
                unavailable(
                    "score_b"
                ),
            ),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    assert len(
        result.details
    ) == 2

    assert (
        "score_a: AVAILABLE"
        in
        result.details[
            0
        ]
    )

    assert (
        "score_b: MISSING_ARTIFACT"
        in
        result.details[
            1
        ]
    )


def test_healthy_summary_reports_population(
    monkeypatch,
):

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        lambda verify_source_hashes=True:
            (
                available(
                    "score_a"
                ),
                available(
                    "score_b"
                ),
            ),
    )

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    assert (
        result.summary
        ==
        "2/2 production scores are available."
    )


# ================================================================
# OPTIONS
# ================================================================


def test_verify_source_hashes_is_forwarded(
    monkeypatch,
):

    observed = {}

    def fake_list_score_statuses(
        verify_source_hashes=True,
    ):

        observed[
            "verify_source_hashes"
        ] = (
            verify_source_hashes
        )

        return (
            available(
                "score_a"
            ),
        )

    monkeypatch.setattr(
        score_health_module,
        "list_score_statuses",
        fake_list_score_statuses,
    )

    monitor_score_runtime(
        observed_at_utc=(
            OBSERVED_AT
        ),
        verify_source_hashes=False,
    )

    assert (
        observed[
            "verify_source_hashes"
        ]
        is False
    )


def test_verify_source_hashes_must_be_bool():

    with pytest.raises(
        TypeError,
        match="verify_source_hashes must be a bool",
    ):

        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            ),
            verify_source_hashes=(
                "yes"
            ),
        )


# ================================================================
# REAL PRODUCTION REGISTRY
# ================================================================


def test_current_six_production_scores_are_healthy():

    result = (
        monitor_score_runtime(
            observed_at_utc=(
                OBSERVED_AT
            )
        )
    )

    assert (
        result.health
        ==
        MONITORING_HEALTHY
    )

    assert (
        result.metrics[
            "total_scores"
        ]
        ==
        6
    )

    assert (
        result.metrics[
            "available_scores"
        ]
        ==
        6
    )

    assert (
        result.metrics[
            "unavailable_scores"
        ]
        ==
        0
    )