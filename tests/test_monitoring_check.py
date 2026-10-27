"""
Tests for immutable production MonitoringCheck records.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.monitoring.check import (
    MonitoringCheck,
    monitoring_check_to_dict,
)

from src.monitoring.spec import (
    MONITORING_CHECK_FIELDS,
    MONITORING_HEALTHY,
    MONITORING_SCHEMA_VERSION,
    MONITOR_COMPONENT_SCORE_RUNTIME,
)


OBSERVED_AT = datetime(
    2026,
    10,
    27,
    18,
    0,
    tzinfo=timezone.utc,
)


def make_check(
    **overrides,
):

    values = {
        "schema_version":
            MONITORING_SCHEMA_VERSION,

        "component":
            MONITOR_COMPONENT_SCORE_RUNTIME,

        "check_name":
            "example_check",

        "health":
            MONITORING_HEALTHY,

        "observed_at_utc":
            OBSERVED_AT,

        "summary":
            "Example monitoring check is healthy.",

        "metrics":
            {
                "count":
                    1,
            },

        "details":
            (
                "example: healthy",
            ),
    }

    values.update(
        overrides
    )

    return MonitoringCheck(
        **values
    )


# ================================================================
# VALID CONSTRUCTION
# ================================================================


def test_valid_monitoring_check_can_be_created():

    check = (
        make_check()
    )

    assert (
        check.check_name
        ==
        "example_check"
    )


def test_monitoring_check_is_frozen():

    check = (
        make_check()
    )

    with pytest.raises(
        FrozenInstanceError
    ):

        check.health = (
            "UNHEALTHY"
        )


def test_metrics_mapping_is_read_only():

    check = (
        make_check()
    )

    with pytest.raises(
        TypeError
    ):

        check.metrics[
            "count"
        ] = 2


def test_details_are_frozen_as_tuple():

    check = (
        make_check(
            details=[
                "first",
                "second",
            ]
        )
    )

    assert (
        check.details
        ==
        (
            "first",
            "second",
        )
    )


# ================================================================
# VALIDATION
# ================================================================


def test_wrong_schema_version_fails_closed():

    with pytest.raises(
        ValueError,
        match="schema_version",
    ):

        make_check(
            schema_version=(
                "9.9.9"
            )
        )


def test_unknown_component_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported monitoring component",
    ):

        make_check(
            component=(
                "unknown"
            )
        )


def test_unknown_health_fails_closed():

    with pytest.raises(
        ValueError,
        match="Unsupported monitoring health",
    ):

        make_check(
            health=(
                "UNKNOWN"
            )
        )


def test_empty_check_name_fails_closed():

    with pytest.raises(
        ValueError,
        match="check_name must not be empty",
    ):

        make_check(
            check_name=(
                "   "
            )
        )


def test_empty_summary_fails_closed():

    with pytest.raises(
        ValueError,
        match="summary must not be empty",
    ):

        make_check(
            summary=(
                ""
            )
        )


def test_metrics_must_be_mapping():

    with pytest.raises(
        TypeError,
        match="metrics must be a mapping",
    ):

        make_check(
            metrics=[
                1,
                2,
            ]
        )


def test_metric_keys_must_be_strings():

    with pytest.raises(
        TypeError,
        match="metrics keys must be strings",
    ):

        make_check(
            metrics={
                1:
                    "value",
            }
        )


def test_details_must_be_iterable_of_strings():

    with pytest.raises(
        TypeError,
        match="iterable of strings",
    ):

        make_check(
            details=(
                "not-a-collection"
            )
        )


def test_invalid_detail_member_fails_closed():

    with pytest.raises(
        TypeError,
        match="detail must be a string",
    ):

        make_check(
            details=(
                123,
            )
        )


# ================================================================
# TIMESTAMP
# ================================================================


def test_naive_timestamp_is_normalised_to_utc():

    check = (
        make_check(
            observed_at_utc=(
                "2026-10-27 18:00:00"
            )
        )
    )

    assert (
        check.observed_at_utc.tzinfo
        is not None
    )


def test_invalid_timestamp_fails_closed():

    with pytest.raises(
        ValueError,
        match="not a valid timestamp",
    ):

        make_check(
            observed_at_utc=(
                "not-a-date"
            )
        )


# ================================================================
# SERIALISATION
# ================================================================


def test_monitoring_check_to_dict_has_exact_schema():

    output = (
        monitoring_check_to_dict(
            make_check()
        )
    )

    assert tuple(
        output.keys()
    ) == MONITORING_CHECK_FIELDS


def test_monitoring_check_to_dict_serialises_timestamp():

    output = (
        monitoring_check_to_dict(
            make_check()
        )
    )

    assert (
        output[
            "observed_at_utc"
        ]
        ==
        "2026-10-27T18:00:00+00:00"
    )


def test_monitoring_check_to_dict_converts_details_to_list():

    output = (
        monitoring_check_to_dict(
            make_check()
        )
    )

    assert (
        output[
            "details"
        ]
        ==
        [
            "example: healthy",
        ]
    )


def test_monitoring_check_to_dict_rejects_wrong_type():

    with pytest.raises(
        TypeError,
        match="MonitoringCheck",
    ):

        monitoring_check_to_dict(
            {}
        )