"""
Scheduled deployment runtime utilities.

This module owns file-system and run-identity concerns shared by the
scheduler-facing CLI commands.

Responsibilities
----------------
- create deterministic run-specific output locations;
- write DataFrame outputs without silent overwrite;
- load optional recommendation episode files;
- derive deterministic customer delivery identifiers;
- detect delivery-audit collisions before export.

It does not perform model scoring or make business decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from src.delivery.audit import (
    list_delivery_audit,
)


# ================================================================
# OUTPUT PATH CONTRACTS
# ================================================================


@dataclass(
    frozen=True,
)
class CustomerBatchOutputPaths:
    run_dir: Path
    decisions_csv: Path
    delivery_csv: Path
    delivery_jsonl: Path


@dataclass(
    frozen=True,
)
class RepeatPurchaseOutputPaths:
    run_dir: Path
    scores_csv: Path


@dataclass(
    frozen=True,
)
class DemandOutputPaths:
    run_dir: Path
    forecasts_csv: Path


# ================================================================
# DATE TOKEN
# ================================================================


def _date_token(
    value,
    *,
    name: str,
) -> str:
    """
    Convert one timestamp-like run boundary to YYYY-MM-DD.
    """

    try:

        timestamp = pd.Timestamp(
            value
        )

    except Exception as exc:

        raise ValueError(
            f"{name} is not a valid timestamp."
        ) from exc

    if pd.isna(
        timestamp
    ):
        raise ValueError(
            f"{name} is not a valid timestamp."
        )

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(
            None
        )

    return (
        timestamp
        .normalize()
        .date()
        .isoformat()
    )


# ================================================================
# RUN-SPECIFIC OUTPUT PATHS
# ================================================================


def customer_batch_output_paths(
    output_dir: str | Path,
    decision_date,
) -> CustomerBatchOutputPaths:
    """
    Build deterministic output paths for one customer-decision run.
    """

    token = _date_token(
        decision_date,
        name="decision_date",
    )

    run_dir = (
        Path(
            output_dir
        )
        /
        "customer-batch"
        /
        token
    )

    return CustomerBatchOutputPaths(
        run_dir=run_dir,
        decisions_csv=(
            run_dir
            /
            "customer_decisions.csv"
        ),
        delivery_csv=(
            run_dir
            /
            "delivery_payloads.csv"
        ),
        delivery_jsonl=(
            run_dir
            /
            "delivery_payloads.jsonl"
        ),
    )


def repeat_purchase_output_paths(
    output_dir: str | Path,
    modeling_end_date,
) -> RepeatPurchaseOutputPaths:
    """
    Build deterministic output paths for one Repeat Purchase run.
    """

    token = _date_token(
        modeling_end_date,
        name="modeling_end_date",
    )

    run_dir = (
        Path(
            output_dir
        )
        /
        "repeat-purchase"
        /
        token
    )

    return RepeatPurchaseOutputPaths(
        run_dir=run_dir,
        scores_csv=(
            run_dir
            /
            "repeat_purchase_scores.csv"
        ),
    )


def demand_output_paths(
    output_dir: str | Path,
    forecast_origin_week,
) -> DemandOutputPaths:
    """
    Build deterministic output paths for one weekly Demand run.
    """

    token = _date_token(
        forecast_origin_week,
        name="forecast_origin_week",
    )

    run_dir = (
        Path(
            output_dir
        )
        /
        "demand"
        /
        token
    )

    return DemandOutputPaths(
        run_dir=run_dir,
        forecasts_csv=(
            run_dir
            /
            "demand_forecasts.csv"
        ),
    )


# ================================================================
# GENERIC DATAFRAME EXPORT
# ================================================================


def write_dataframe_csv(
    dataframe: pd.DataFrame,
    path: str | Path,
) -> Path:
    """
    Persist one DataFrame as UTF-8 CSV without silent overwrite.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    export_path = Path(
        path
    )

    if (
        export_path.exists()
        and
        export_path.is_dir()
    ):
        raise ValueError(
            "DataFrame export path must be a file."
        )

    export_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with export_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:

        dataframe.to_csv(
            handle,
            index=False,
        )

    return export_path


# ================================================================
# RECOMMENDATION EPISODE INPUT
# ================================================================


def load_recommendation_episodes(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load an optional governed recommendation episode table.

    Supported formats:
    - parquet
    - csv

    Frozen episode-schema validation remains owned by RecommendationScorer.
    """

    source_path = Path(
        path
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Recommendation episode source does not exist: "
            f"{source_path.resolve()}"
        )

    if not source_path.is_file():
        raise ValueError(
            "Recommendation episode source is not a file: "
            f"{source_path.resolve()}"
        )

    extension = (
        source_path
        .suffix
        .lower()
    )

    if extension == ".parquet":
        return pd.read_parquet(
            source_path
        )

    if extension == ".csv":
        return pd.read_csv(
            source_path,
            low_memory=False,
        )

    raise ValueError(
        "Unsupported recommendation episode file type: "
        f"{extension}. Supported formats: .csv, .parquet"
    )


# ================================================================
# DETERMINISTIC DELIVERY IDENTITY
# ================================================================


def build_deterministic_delivery_ids(
    decisions: pd.DataFrame,
    decision_date,
) -> dict[str, str]:
    """
    Derive one stable delivery ID per Decision Engine customer.

    Identity basis:
        deployment namespace
        + normalized decision date
        + customer

    Therefore the same governed run boundary and customer always map to
    the same delivery ID. A rerun collides with the append-only audit and
    fails closed instead of creating a second customer treatment.
    """

    if not isinstance(
        decisions,
        pd.DataFrame,
    ):
        raise TypeError(
            "decisions must be a pandas DataFrame."
        )

    if "customer" not in decisions.columns:
        raise ValueError(
            "Decision output is missing customer."
        )

    if decisions[
        "customer"
    ].isna().any():
        raise ValueError(
            "Decision output contains missing customer values."
        )

    customer_values = (
        decisions[
            "customer"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if len(
        customer_values
    ) != len(
        set(
            customer_values
        )
    ):
        raise ValueError(
            "Decision output contains duplicate customers."
        )

    token = _date_token(
        decision_date,
        name="decision_date",
    )

    identifiers = {}

    for customer in customer_values:

        identity = (
            "ai-revenue-decision-engine"
            "|customer-delivery"
            f"|{token}"
            f"|{customer}"
        )

        identifiers[
            customer
        ] = str(
            uuid5(
                NAMESPACE_URL,
                identity,
            )
        )

    return identifiers


def assert_delivery_ids_not_audited(
    delivery_ids: dict[str, str],
    audit_path: str | Path,
) -> None:
    """
    Fail before export when a deterministic delivery identity already exists.
    """

    if not isinstance(
        delivery_ids,
        dict,
    ):
        raise TypeError(
            "delivery_ids must be a dictionary."
        )

    existing = list_delivery_audit(
        audit_path
    )

    existing_ids = {
        payload.delivery_id
        for payload
        in existing
    }

    requested_ids = set(
        delivery_ids.values()
    )

    collisions = (
        requested_ids
        &
        existing_ids
    )

    if collisions:
        raise ValueError(
            "Scheduled customer batch has already been audited for "
            "one or more deterministic delivery IDs: "
            f"{sorted(collisions)}"
        )