"""
Production command-line interface for the AI Revenue Decision Engine.

The CLI provides small operational entry points suitable for local,
container, and scheduled execution.

Current commands
----------------
health
    Verify that all governed production scores are runtime-available.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Sequence

from src.monitoring.score_health import (
    monitor_score_runtime,
)

from src.monitoring.spec import (
    MONITORING_HEALTHY,
)


# ================================================================
# PARSER
# ================================================================


def build_parser() -> argparse.ArgumentParser:
    """
    Build the production CLI parser.
    """

    parser = argparse.ArgumentParser(
        prog=(
            "ai-revenue-decision-engine"
        ),

        description=(
            "Operational CLI for the AI Revenue Decision Engine."
        ),
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    health_parser = (
        subparsers.add_parser(
            "health",
            help=(
                "Check governed production score runtime availability."
            ),
        )
    )

    health_parser.add_argument(
        "--no-verify-source-hashes",
        action="store_true",
        help=(
            "Skip governed manifest source-hash verification."
        ),
    )

    return parser


# ================================================================
# HEALTH
# ================================================================


def run_health_command(
    *,
    verify_source_hashes: bool = True,
) -> int:
    """
    Execute the production score-runtime health check.

    Exit codes
    ----------
    0
        All governed production scores are available.

    1
        One or more governed production scores are unavailable.
    """

    check = (
        monitor_score_runtime(
            observed_at_utc=(
                datetime.now(
                    timezone.utc
                )
            ),

            verify_source_hashes=(
                verify_source_hashes
            ),
        )
    )

    available = (
        check.metrics[
            "available_scores"
        ]
    )

    total = (
        check.metrics[
            "total_scores"
        ]
    )

    unavailable = (
        check.metrics[
            "unavailable_scores"
        ]
    )

    print(
        (
            f"{check.health}: "
            f"{available}/{total} production scores available; "
            f"{unavailable} unavailable."
        )
    )

    if (
        check.health
        ==
        MONITORING_HEALTHY
    ):
        return 0

    return 1


# ================================================================
# ENTRY POINT
# ================================================================


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """
    Execute the production CLI.
    """

    parser = (
        build_parser()
    )

    args = (
        parser.parse_args(
            argv
        )
    )

    if (
        args.command
        ==
        "health"
    ):

        return run_health_command(
            verify_source_hashes=(
                not
                args.no_verify_source_hashes
            )
        )

    parser.error(
        f"Unsupported command: {args.command}"
    )

    return 2


if __name__ == "__main__":

    sys.exit(
        main()
    )