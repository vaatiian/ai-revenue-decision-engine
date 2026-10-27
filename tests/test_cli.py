"""
Tests for the production command-line interface.
"""

from types import SimpleNamespace

import pytest

import src.cli as cli_module

from src.monitoring.spec import (
    MONITORING_HEALTHY,
    MONITORING_UNHEALTHY,
)


# ================================================================
# TEST HELPERS
# ================================================================


def fake_health_check(
    *,
    health,
    available,
    total,
    unavailable,
):

    return SimpleNamespace(
        health=health,

        summary=(
            f"{available}/{total} production scores are available."
        ),

        metrics={
            "available_scores":
                available,

            "total_scores":
                total,

            "unavailable_scores":
                unavailable,
        },
    )


# ================================================================
# HEALTH COMMAND
# ================================================================


def test_health_command_returns_zero_when_runtime_is_healthy(
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        cli_module,
        "monitor_score_runtime",
        lambda *,
        observed_at_utc,
        verify_source_hashes=True:
            fake_health_check(
                health=(
                    MONITORING_HEALTHY
                ),
                available=6,
                total=6,
                unavailable=0,
            ),
    )

    exit_code = (
        cli_module.main(
            [
                "health",
            ]
        )
    )

    output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        exit_code
        ==
        0
    )

    assert (
        "HEALTHY"
        in
        output
    )

    assert (
        "6/6"
        in
        output
    )


def test_health_command_returns_one_when_runtime_is_unhealthy(
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        cli_module,
        "monitor_score_runtime",
        lambda *,
        observed_at_utc,
        verify_source_hashes=True:
            fake_health_check(
                health=(
                    MONITORING_UNHEALTHY
                ),
                available=5,
                total=6,
                unavailable=1,
            ),
    )

    exit_code = (
        cli_module.main(
            [
                "health",
            ]
        )
    )

    output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        exit_code
        ==
        1
    )

    assert (
        "UNHEALTHY"
        in
        output
    )

    assert (
        "5/6"
        in
        output
    )


def test_health_command_calls_runtime_monitor(
    monkeypatch,
):

    observed = {}

    def fake_monitor(
        *,
        observed_at_utc,
        verify_source_hashes=True,
    ):

        observed[
            "observed_at_utc"
        ] = (
            observed_at_utc
        )

        return fake_health_check(
            health=(
                MONITORING_HEALTHY
            ),
            available=6,
            total=6,
            unavailable=0,
        )

    monkeypatch.setattr(
        cli_module,
        "monitor_score_runtime",
        fake_monitor,
    )

    exit_code = (
        cli_module.main(
            [
                "health",
            ]
        )
    )

    assert (
        exit_code
        ==
        0
    )

    assert (
        observed[
            "observed_at_utc"
        ]
        is not None
    )


def test_health_command_forwards_hash_verification(
    monkeypatch,
):

    observed = {}

    def fake_monitor(
        *,
        observed_at_utc,
        verify_source_hashes=True,
    ):

        observed[
            "verify_source_hashes"
        ] = (
            verify_source_hashes
        )

        return fake_health_check(
            health=(
                MONITORING_HEALTHY
            ),
            available=6,
            total=6,
            unavailable=0,
        )

    monkeypatch.setattr(
        cli_module,
        "monitor_score_runtime",
        fake_monitor,
    )

    cli_module.main(
        [
            "health",
        ]
    )

    assert (
        observed[
            "verify_source_hashes"
        ]
        is True
    )


def test_health_command_can_disable_hash_verification(
    monkeypatch,
):

    observed = {}

    def fake_monitor(
        *,
        observed_at_utc,
        verify_source_hashes=True,
    ):

        observed[
            "verify_source_hashes"
        ] = (
            verify_source_hashes
        )

        return fake_health_check(
            health=(
                MONITORING_HEALTHY
            ),
            available=6,
            total=6,
            unavailable=0,
        )

    monkeypatch.setattr(
        cli_module,
        "monitor_score_runtime",
        fake_monitor,
    )

    cli_module.main(
        [
            "health",
            "--no-verify-source-hashes",
        ]
    )

    assert (
        observed[
            "verify_source_hashes"
        ]
        is False
    )


# ================================================================
# CLI CONTRACT
# ================================================================


def test_no_command_fails_with_usage_error():

    with pytest.raises(
        SystemExit
    ) as exc_info:

        cli_module.main(
            []
        )

    assert (
        exc_info.value.code
        ==
        2
    )


def test_unknown_command_fails_with_usage_error():

    with pytest.raises(
        SystemExit
    ) as exc_info:

        cli_module.main(
            [
                "unknown",
            ]
        )

    assert (
        exc_info.value.code
        ==
        2
    )