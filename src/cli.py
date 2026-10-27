"""
Production command-line interface for the AI Revenue Decision Engine.

The CLI provides operational entry points suitable for local,
container, and scheduled execution.

Commands
--------
health
    Verify governed production score runtime availability.

customer-batch
    Execute one governed month-end customer-decision batch, create
    deterministic delivery payloads, export run artifacts, and append
    the delivery audit ledger.

repeat-purchase
    Execute one standalone governed Repeat Purchase propensity batch.

demand
    Execute one governed weekly Demand Forecast batch.
"""

from __future__ import annotations

import argparse
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.data.loader import (
    load_transactions,
)
from src.delivery.audit import (
    DEFAULT_DELIVERY_AUDIT_PATH,
    append_delivery_batch_audit,
)
from src.delivery.builder import (
    build_delivery_batch,
)
from src.delivery.export import (
    export_delivery_csv,
    export_delivery_jsonl,
)
from src.deployment.customer_batch import (
    run_customer_decision_batch,
)
from src.deployment.demand_batch import (
    run_demand_batch,
)
from src.deployment.repeat_purchase_batch import (
    run_repeat_purchase_batch,
)
from src.deployment.runtime import (
    assert_delivery_ids_not_audited,
    build_deterministic_delivery_ids,
    customer_batch_output_paths,
    demand_output_paths,
    load_recommendation_episodes,
    repeat_purchase_output_paths,
    write_dataframe_csv,
)
from src.models.demand_spec import (
    DEMAND_FORECAST_HORIZONS,
)
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

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------

    health_parser = subparsers.add_parser(
        "health",
        help=(
            "Check governed production score runtime availability."
        ),
    )

    health_parser.add_argument(
        "--no-verify-source-hashes",
        action="store_true",
        help=(
            "Skip governed manifest source-hash verification."
        ),
    )

    # ------------------------------------------------------------
    # CUSTOMER BATCH
    # ------------------------------------------------------------

    customer_parser = subparsers.add_parser(
        "customer-batch",
        help=(
            "Run the governed month-end customer decision batch."
        ),
    )

    customer_parser.add_argument(
        "--data",
        required=True,
        help=(
            "Canonical transaction CSV or parquet path."
        ),
    )

    customer_parser.add_argument(
        "--decision-date",
        required=True,
        help=(
            "Month-end decision date."
        ),
    )

    customer_parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Writable root directory for run outputs."
        ),
    )

    customer_parser.add_argument(
        "--recommendation-episodes",
        default=None,
        help=(
            "Optional governed recommendation episode CSV/parquet."
        ),
    )

    customer_parser.add_argument(
        "--delivery-audit",
        default=str(
            DEFAULT_DELIVERY_AUDIT_PATH
        ),
        help=(
            "Persistent append-only delivery audit JSONL path."
        ),
    )

    # ------------------------------------------------------------
    # REPEAT PURCHASE
    # ------------------------------------------------------------

    repeat_parser = subparsers.add_parser(
        "repeat-purchase",
        help=(
            "Run the standalone governed Repeat Purchase batch."
        ),
    )

    repeat_parser.add_argument(
        "--data",
        required=True,
        help=(
            "Canonical transaction CSV or parquet path."
        ),
    )

    repeat_parser.add_argument(
        "--modeling-end-date",
        required=True,
        help=(
            "Explicit Repeat Purchase observation boundary."
        ),
    )

    repeat_parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Writable root directory for run outputs."
        ),
    )

    repeat_parser.add_argument(
        "--include-opening-month",
        action="store_true",
        help=(
            "Disable the historical-parity opening-month exclusion."
        ),
    )

    # ------------------------------------------------------------
    # DEMAND
    # ------------------------------------------------------------

    demand_parser = subparsers.add_parser(
        "demand",
        help=(
            "Run the governed weekly Demand Forecast batch."
        ),
    )

    demand_parser.add_argument(
        "--data",
        required=True,
        help=(
            "Canonical transaction CSV or parquet path."
        ),
    )

    demand_parser.add_argument(
        "--forecast-origin-week",
        required=True,
        help=(
            "Monday week-start for the final complete observed week."
        ),
    )

    demand_parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Writable root directory for run outputs."
        ),
    )

    demand_parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(
            DEMAND_FORECAST_HORIZONS
        ),
        help=(
            "Forecast horizons in weeks. "
            f"Supported: {DEMAND_FORECAST_HORIZONS}."
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

    check = monitor_score_runtime(
        observed_at_utc=(
            datetime.now(
                timezone.utc
            )
        ),
        verify_source_hashes=(
            verify_source_hashes
        ),
    )

    available = check.metrics[
        "available_scores"
    ]

    total = check.metrics[
        "total_scores"
    ]

    unavailable = check.metrics[
        "unavailable_scores"
    ]

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
# CUSTOMER BATCH
# ================================================================


def run_customer_batch_command(
    *,
    data_path: str | Path,
    decision_date,
    output_dir: str | Path,
    recommendation_episodes_path: str | Path | None = None,
    delivery_audit_path: str | Path = (
        DEFAULT_DELIVERY_AUDIT_PATH
    ),
) -> int:
    """
    Execute and persist one governed customer-decision batch.
    """

    transactions, _ = load_transactions(
        data_path
    )

    recommendation_episodes = None

    if recommendation_episodes_path is not None:

        recommendation_episodes = (
            load_recommendation_episodes(
                recommendation_episodes_path
            )
        )

    observed_at_utc = datetime.now(
        timezone.utc
    )

    result = run_customer_decision_batch(
        transactions=transactions,
        decision_date=decision_date,
        recommendation_episodes=(
            recommendation_episodes
        ),
        recommendation_scored_at=(
            observed_at_utc
        ),
    )

    output_paths = customer_batch_output_paths(
        output_dir=output_dir,
        decision_date=(
            result.decision_date
        ),
    )

    delivery_ids = (
        build_deterministic_delivery_ids(
            decisions=(
                result.decisions
            ),
            decision_date=(
                result.decision_date
            ),
        )
    )

    assert_delivery_ids_not_audited(
        delivery_ids=delivery_ids,
        audit_path=(
            delivery_audit_path
        ),
    )

    payloads = build_delivery_batch(
        decisions=(
            result.decisions
        ),
        delivery_created_at_utc=(
            observed_at_utc
        ),
        delivery_ids=(
            delivery_ids
        ),
    )

    write_dataframe_csv(
        dataframe=(
            result.decisions
        ),
        path=(
            output_paths.decisions_csv
        ),
    )

    export_delivery_csv(
        payloads=payloads,
        path=(
            output_paths.delivery_csv
        ),
    )

    export_delivery_jsonl(
        payloads=payloads,
        path=(
            output_paths.delivery_jsonl
        ),
    )

    append_delivery_batch_audit(
        payloads=payloads,
        path=(
            delivery_audit_path
        ),
    )

    print(
        (
            "SUCCEEDED: customer-batch "
            f"{result.decision_date.date().isoformat()} "
            f"customers={len(result.decisions)} "
            f"output={output_paths.run_dir}"
        )
    )

    return 0


# ================================================================
# REPEAT PURCHASE
# ================================================================


def run_repeat_purchase_command(
    *,
    data_path: str | Path,
    modeling_end_date,
    output_dir: str | Path,
    exclude_opening_month: bool = True,
) -> int:
    """
    Execute and persist one governed Repeat Purchase batch.
    """

    transactions, _ = load_transactions(
        data_path
    )

    result = run_repeat_purchase_batch(
        transactions=transactions,
        modeling_end_date=(
            modeling_end_date
        ),
        exclude_opening_month=(
            exclude_opening_month
        ),
    )

    output_paths = repeat_purchase_output_paths(
        output_dir=output_dir,
        modeling_end_date=(
            result.modeling_end_date
        ),
    )

    write_dataframe_csv(
        dataframe=(
            result.scores
        ),
        path=(
            output_paths.scores_csv
        ),
    )

    print(
        (
            "SUCCEEDED: repeat-purchase "
            f"{result.modeling_end_date.date().isoformat()} "
            f"customers={len(result.scores)} "
            f"output={output_paths.run_dir}"
        )
    )

    return 0


# ================================================================
# DEMAND
# ================================================================


def run_demand_command(
    *,
    data_path: str | Path,
    forecast_origin_week,
    output_dir: str | Path,
    horizons,
) -> int:
    """
    Execute and persist one governed weekly Demand Forecast batch.
    """

    transactions, _ = load_transactions(
        data_path
    )

    observed_at_utc = datetime.now(
        timezone.utc
    )

    result = run_demand_batch(
        transactions=transactions,
        forecast_origin_week=(
            forecast_origin_week
        ),
        horizons=(
            horizons
        ),
        scored_at_utc=(
            observed_at_utc
        ),
    )

    output_paths = demand_output_paths(
        output_dir=output_dir,
        forecast_origin_week=(
            result.forecast_origin_week
        ),
    )

    write_dataframe_csv(
        dataframe=(
            result.forecasts
        ),
        path=(
            output_paths.forecasts_csv
        ),
    )

    print(
        (
            "SUCCEEDED: demand "
            f"{result.forecast_origin_week.date().isoformat()} "
            f"horizons={result.horizons} "
            f"output={output_paths.run_dir}"
        )
    )

    return 0


# ================================================================
# ENTRY POINT
# ================================================================


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """
    Execute the production CLI.

    Exit codes
    ----------
    0
        Successful governed execution.

    1
        Governed runtime/execution failure.

    2
        Invalid command-line invocation, owned by argparse.
    """

    parser = build_parser()

    args = parser.parse_args(
        argv
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

    try:

        if (
            args.command
            ==
            "customer-batch"
        ):

            return run_customer_batch_command(
                data_path=(
                    args.data
                ),
                decision_date=(
                    args.decision_date
                ),
                output_dir=(
                    args.output_dir
                ),
                recommendation_episodes_path=(
                    args.recommendation_episodes
                ),
                delivery_audit_path=(
                    args.delivery_audit
                ),
            )

        if (
            args.command
            ==
            "repeat-purchase"
        ):

            return run_repeat_purchase_command(
                data_path=(
                    args.data
                ),
                modeling_end_date=(
                    args.modeling_end_date
                ),
                output_dir=(
                    args.output_dir
                ),
                exclude_opening_month=(
                    not
                    args.include_opening_month
                ),
            )

        if (
            args.command
            ==
            "demand"
        ):

            return run_demand_command(
                data_path=(
                    args.data
                ),
                forecast_origin_week=(
                    args.forecast_origin_week
                ),
                output_dir=(
                    args.output_dir
                ),
                horizons=(
                    tuple(
                        args.horizons
                    )
                ),
            )

    except Exception as exc:

        print(
            f"FAILED: {exc}",
            file=sys.stderr,
        )

        return 1

    parser.error(
        f"Unsupported command: {args.command}"
    )

    return 2


if __name__ == "__main__":

    sys.exit(
        main()
    )