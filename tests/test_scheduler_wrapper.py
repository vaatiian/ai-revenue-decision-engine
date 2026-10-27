"""
Tests for the thin scheduled-execution shell wrapper.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)

WRAPPER_PATH = (
    PROJECT_ROOT
    /
    "scripts"
    /
    "run_scheduled_job.sh"
)


# ================================================================
# TEST HELPERS
# ================================================================


def make_fake_python(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
]:

    capture_path = (
        tmp_path
        /
        "captured_args.txt"
    )

    fake_python = (
        tmp_path
        /
        "fake-python"
    )

    fake_python.write_text(
        (
            "#!/bin/sh\n"
            "set -eu\n"
            ': > "$CAPTURE_PATH"\n'
            "for argument in \"$@\"; do\n"
            "    printf '%s\\n' \"$argument\" >> \"$CAPTURE_PATH\"\n"
            "done\n"
            'exit "${FAKE_EXIT_CODE:-0}"\n'
        ),
        encoding="utf-8",
    )

    fake_python.chmod(
        0o755
    )

    return (
        fake_python,
        capture_path,
    )


def run_wrapper(
    tmp_path: Path,
    arguments: list[str],
    *,
    extra_env: dict[str, str] | None = None,
):

    fake_python, capture_path = (
        make_fake_python(
            tmp_path
        )
    )

    env = os.environ.copy()

    env.update(
        {
            "PYTHON_BIN":
                str(
                    fake_python
                ),

            "CAPTURE_PATH":
                str(
                    capture_path
                ),
        }
    )

    if extra_env:

        env.update(
            extra_env
        )

    result = subprocess.run(
        [
            "sh",
            str(
                WRAPPER_PATH
            ),
            *arguments,
        ],
        cwd=(
            PROJECT_ROOT
        ),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    captured_arguments = []

    if capture_path.exists():

        captured_arguments = (
            capture_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

    return (
        result,
        captured_arguments,
    )


# ================================================================
# WRAPPER CONTRACT
# ================================================================


def test_scheduler_wrapper_exists():

    assert WRAPPER_PATH.exists()

    assert WRAPPER_PATH.is_file()


def test_scheduler_wrapper_requires_exactly_two_arguments(
    tmp_path,
):

    result, captured = run_wrapper(
        tmp_path,
        [],
    )

    assert result.returncode == 2

    assert captured == []

    assert (
        "Usage:"
        in
        result.stderr
    )


def test_scheduler_wrapper_rejects_extra_arguments(
    tmp_path,
):

    result, captured = run_wrapper(
        tmp_path,
        [
            "demand",
            "2011-11-28",
            "unexpected",
        ],
    )

    assert result.returncode == 2

    assert captured == []


def test_scheduler_wrapper_rejects_unknown_job(
    tmp_path,
):

    result, captured = run_wrapper(
        tmp_path,
        [
            "unknown",
            "2011-08-31",
        ],
    )

    assert result.returncode == 2

    assert captured == []

    assert (
        "Unsupported scheduled job"
        in
        result.stderr
    )


# ================================================================
# CUSTOMER JOB
# ================================================================


def test_customer_job_routes_exact_governed_cli_contract(
    tmp_path,
):

    result, captured = run_wrapper(
        tmp_path,
        [
            "customer",
            "2011-08-31",
        ],
    )

    assert result.returncode == 0

    assert captured == [
        "-m",
        "src.cli",
        "customer-batch",
        "--data",
        "/app/data/cleaned_online_retail.parquet",
        "--decision-date",
        "2011-08-31",
        "--output-dir",
        "/app/outputs",
        "--delivery-audit",
        (
            "/app/artifacts/delivery/"
            "delivery_audit.jsonl"
        ),
    ]


def test_customer_job_supports_explicit_runtime_path_overrides(
    tmp_path,
):

    result, captured = run_wrapper(
        tmp_path,
        [
            "customer",
            "2011-09-30",
        ],
        extra_env={
            "DATA_PATH":
                "/runtime/data.parquet",

            "OUTPUT_DIR":
                "/runtime/outputs",

            "DELIVERY_AUDIT_PATH":
                "/runtime/state/delivery.jsonl",

            "RECOMMENDATION_EPISODES_PATH":
                "/runtime/episodes.parquet",
        },
    )

    assert result.returncode == 0

    assert captured == [
        "-m",
        "src.cli",
        "customer-batch",
        "--data",
        "/runtime/data.parquet",
        "--decision-date",
        "2011-09-30",
        "--output-dir",
        "/runtime/outputs",
        "--delivery-audit",
        "/runtime/state/delivery.jsonl",
        "--recommendation-episodes",
        "/runtime/episodes.parquet",
    ]


# ================================================================
# REPEAT PURCHASE JOB
# ================================================================


def test_repeat_purchase_job_routes_exact_governed_cli_contract(
    tmp_path,
):

    result, captured = run_wrapper(
        tmp_path,
        [
            "repeat-purchase",
            "2011-11-30",
        ],
    )

    assert result.returncode == 0

    assert captured == [
        "-m",
        "src.cli",
        "repeat-purchase",
        "--data",
        "/app/data/cleaned_online_retail.parquet",
        "--modeling-end-date",
        "2011-11-30",
        "--output-dir",
        "/app/outputs",
    ]


# ================================================================
# DEMAND JOB
# ================================================================


def test_demand_job_routes_exact_governed_cli_contract(
    tmp_path,
):

    result, captured = run_wrapper(
        tmp_path,
        [
            "demand",
            "2011-11-28",
        ],
    )

    assert result.returncode == 0

    assert captured == [
        "-m",
        "src.cli",
        "demand",
        "--data",
        "/app/data/cleaned_online_retail.parquet",
        "--forecast-origin-week",
        "2011-11-28",
        "--output-dir",
        "/app/outputs",
    ]


# ================================================================
# EXIT CODE PRESERVATION
# ================================================================


def test_scheduler_wrapper_preserves_cli_exit_code(
    tmp_path,
):

    result, captured = run_wrapper(
        tmp_path,
        [
            "demand",
            "2011-11-28",
        ],
        extra_env={
            "FAKE_EXIT_CODE":
                "7",
        },
    )

    assert captured

    assert result.returncode == 7